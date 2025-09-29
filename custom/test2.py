import os
# ---------------- CUDA allocator (set before importing torch) ----------------
# Stronger than just expandable_segments. Helps keep big pools reusable.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
import time
from collections import OrderedDict

# -----------------------------
# Configuration
# -----------------------------
model_id   = "Qwen/Qwen1.5-MoE-A2.7B"
gpu_device = "cuda:0"
cpu_device = "cpu"
dtype      = torch.float16

# VRAM budget for trunk/router; experts will be cached separately via our LRU
max_memory = {
    0:   "8GiB",     # If KV ends up on CPU, raise this to e.g. "10GiB"
    "cpu":"300GiB"
}

# Expert cache caps (choose one primary; the other is a guardrail)
GPU_CACHE_MAX_BYTES   = 2_000_000_000   # ~2.0 GB for expert weights total
GPU_CACHE_MAX_EXPERTS = 8               # or at most 8 experts resident on GPU

# -----------------------------
# 1) Load config & empty skeleton
# -----------------------------
print(f"1. Loading config and creating model skeleton for {model_id}...")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# -----------------------------
# 2) Build device map (experts to CPU)
# -----------------------------
print("2. Inferring device map (8GB VRAM limit enforced)...")
no_split = getattr(empty_model, "_no_split_modules", ["QwenBlock", "Block"])

device_map = infer_auto_device_map(
    empty_model,
    max_memory=max_memory,
    no_split_module_classes=no_split,
)

# Force all expert modules to CPU (keep trunk/router on GPU within budget)
for name in list(device_map.keys()):
    if "experts" in name:
        device_map[name] = cpu_device

# -----------------------------
# 3) Download checkpoints & dispatch
# -----------------------------
print("3. Downloading checkpoint and dispatching weights...")
local_ckpt = snapshot_download(
    model_id,
    allow_patterns=["*.safetensors", "*.bin", "config.json", "*.json"]
)
model = load_checkpoint_and_dispatch(
    empty_model,
    checkpoint=local_ckpt,
    device_map=device_map,
    no_split_module_classes=no_split,
    dtype=dtype,
    offload_folder=None,
)
print("   Model successfully loaded and dispatched across GPU and CPU RAM.")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# ======================================================================
#                   EXPERT GPU LRU CACHE (YOUR CODE)
# ======================================================================

def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def module_param_bytes(mod: nn.Module) -> int:
    s = 0
    for p in mod.parameters(recurse=True):
        s += tensor_nbytes(p.data)
    for b in mod.buffers(recurse=True):
        s += tensor_nbytes(b.data)
    return s

def pin_state_dict(cpu_module: nn.Module):
    pinned = {}
    with torch.no_grad():
        for k, v in cpu_module.state_dict().items():
            t = v.detach().contiguous().to("cpu", copy=True)
            try:
                t = t.pin_memory()
            except RuntimeError:
                pass
            pinned[k] = t
    return pinned

class ExpertGPUCache:
    """
    LRU cache for CUDA-resident expert modules.
    - Capacity tracked by both bytes and count.
    - Uses pinned CPU state_dict for fast, async, non_blocking loads.
    - Optional prefetch using a dedicated CUDA stream.
    """
    def __init__(self, device:str, dtype:torch.dtype,
                 max_bytes:int = GPU_CACHE_MAX_BYTES,
                 max_count:int = GPU_CACHE_MAX_EXPERTS):
        self.device = torch.device(device)
        self.dtype  = dtype
        self.max_bytes = max_bytes
        self.max_count = max_count

        self._cache: OrderedDict[int, nn.Module] = OrderedDict()  # expert_id -> cuda module
        self._sizes: dict[int, int] = {}                          # bytes by expert
        self._pinned_sd: dict[int, dict] = {}                     # expert_id -> pinned state_dict
        self._bytes_total = 0

        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.main_stream = torch.cuda.current_stream(device=self.device)

    def _ensure_capacity(self, need_bytes:int):
        while (self._bytes_total + need_bytes > self.max_bytes) or \
              (len(self._cache) + 1 > self.max_count):
            evict_id, evict_mod = self._cache.popitem(last=False)
            ev_sz = self._sizes.pop(evict_id, 0)
            self._bytes_total -= ev_sz
            del evict_mod

    def _build_cuda_module(self, cpu_expert: nn.Module) -> nn.Module:
        cls = cpu_expert.__class__
        if hasattr(cpu_expert, "config"):
            m = cls(config=cpu_expert.config).to(self.device, dtype=self.dtype)
        else:
            import copy as _copy
            m = _copy.deepcopy(cpu_expert).to(self.device, dtype=self.dtype)
        return m

    def _load_weights_async(self, cuda_mod: nn.Module, expert_id: int):
        if expert_id not in self._pinned_sd:
            self._pinned_sd[expert_id] = pin_state_dict(cpu_expert_registry[expert_id])

        sd = self._pinned_sd[expert_id]
        with torch.cuda.stream(self.prefetch_stream):
            for k, v_cpu in sd.items():
                ref = cuda_mod
                comps = k.split(".")
                for c in comps[:-1]:
                    ref = getattr(ref, c)
                leaf_name = comps[-1]
                if hasattr(ref, leaf_name):
                    t = getattr(ref, leaf_name)
                    t.data.copy_(v_cpu, non_blocking=True)
        self.main_stream.wait_stream(self.prefetch_stream)
        torch.cuda.current_stream(self.device).synchronize()

    def get(self, expert_id: int, cpu_expert: nn.Module) -> nn.Module:
        if expert_id in self._cache:
            mod = self._cache.pop(expert_id)
            self._cache[expert_id] = mod  # MRU
            return mod

        cuda_mod = self._build_cuda_module(cpu_expert)
        need_bytes = module_param_bytes(cuda_mod)

        self._ensure_capacity(need_bytes)
        self._load_weights_async(cuda_mod, expert_id)

        self._cache[expert_id] = cuda_mod  # MRU
        self._sizes[expert_id] = need_bytes
        self._bytes_total += need_bytes
        return cuda_mod

    def prefetch_experts(self, ids: list[int]):
        with torch.no_grad():
            for eid in ids:
                if eid in self._cache:
                    mod = self._cache.pop(eid)
                    self._cache[eid] = mod
                else:
                    if eid not in cpu_expert_registry:
                        continue
                    cpu_mod = cpu_expert_registry[eid]
                    cuda_mod = self._build_cuda_module(cpu_mod)
                    need_bytes = module_param_bytes(cuda_mod)
                    self._ensure_capacity(need_bytes)
                    self._load_weights_async(cuda_mod, eid)
                    self._cache[eid] = cuda_mod
                    self._sizes[eid] = need_bytes
                    self._bytes_total += need_bytes

# ------------------------------------------------------------------------------
# Build a registry mapping expert_id -> CPU expert module for quick access
# ------------------------------------------------------------------------------
cpu_expert_registry: dict[int, nn.Module] = {}
expert_name_map: dict[int, str] = {}
eid = 0
for name, module in model.named_modules():
    if isinstance(module, nn.ModuleList) and "experts" in name:
        for idx, expert in enumerate(module):
            try:
                dev = next(expert.parameters()).device.type
            except StopIteration:
                dev = cpu_device
            if dev == cpu_device:
                cpu_expert_registry[eid] = expert
                expert_name_map[eid] = f"{name}[{idx}]"
                eid += 1
print(f"Discovered {len(cpu_expert_registry)} CPU-resident experts for dynamic caching.")

# Create one global cache manager
expert_cache = ExpertGPUCache(device=gpu_device, dtype=dtype,
                              max_bytes=GPU_CACHE_MAX_BYTES, max_count=GPU_CACHE_MAX_EXPERTS)

# ======================================================================
#             PATCH EXPERT FORWARD TO USE THE LRU CACHE (YOUR CODE)
# ======================================================================
def make_cached_forward(expert_id: int, cpu_expert: nn.Module):
    def wrapped_forward(*args, **kwargs):
        cuda_dev = torch.device(gpu_device)

        def to_cuda(x):
            if torch.is_tensor(x):
                return x.to(cuda_dev, non_blocking=True)
            return x

        args_cuda = tuple(to_cuda(a) for a in args)
        kwargs_cuda = {k: to_cuda(v) for k, v in kwargs.items()}

        cuda_expert = expert_cache.get(expert_id, cpu_expert)

        with torch.no_grad():
            out = cuda_expert(*args_cuda, **kwargs_cuda)
        return out
    return wrapped_forward

patched = 0
for eid, cpu_expert in cpu_expert_registry.items():
    cpu_expert.forward = make_cached_forward(eid, cpu_expert)
    patched += 1
print(f"Patched {patched} experts with persistent CUDA-cache wrapper.")

# ======================================================================
#                  STICKY-KV PREFILL + MANUAL DECODE
# ======================================================================
from transformers.cache_utils import StaticCache

def gb(x): return x / 1e9

model.eval()
model.config.use_cache = True  # must be True to retain KV

# Demo prompts; add more to increase decode concurrency and VRAM usage
PROMPTS = [
    "Explain the concept of dynamic offloading in large language models in one paragraph."
    # Add more prompts to expand batch and absorb more freed VRAM
]

print("\n" + "="*72)
print("Tokenizing inputs and moving to GPU...")
inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=False)
inputs = {k: v.to(gpu_device) for k, v in inputs.items()}

B = inputs["input_ids"].size(0)
prompt_len = inputs["input_ids"].size(1)
max_new = 128  # choose based on your experiment

# 1) Pre-allocate a SINGLE GPU-resident cache large enough for decode
kv = StaticCache(
    max_batch_size=B,
    max_cache_len=prompt_len + max_new,
    config=model.config,
    device=torch.device(gpu_device),  # Keep KV on GPU to keep VRAM “used”
    dtype=model.dtype,
)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

# 2) PREFILL: writes into our pre-allocated KV (no mid-allocs later)
with torch.inference_mode():
    _ = model(**inputs, use_cache=True, past_key_values=kv)

torch.cuda.synchronize()
print(f"Peak during prefill:    {gb(torch.cuda.max_memory_allocated()):.2f} GB")
print(f"Current after prefill:  {gb(torch.cuda.memory_allocated()):.2f} GB")

# Optional: sanity-print where KV lives
first_k = kv.key_cache[0] if isinstance(kv.key_cache, list) else kv.key_cache
print(f"KV[0] device/dtype/shape: {first_k.device}, {first_k.dtype}, {tuple(first_k.shape)}")

# 3) MANUAL DECODE LOOP that REUSES THE SAME CACHE AND BUFFERS
cur_ids = inputs["input_ids"]
# Pre-allocate attention mask with final length; fill leading ones
attn_mask = torch.zeros((B, prompt_len + max_new), dtype=torch.long, device=gpu_device)
attn_mask[:, :prompt_len] = 1
cur_len = prompt_len

start = time.time()
for step in range(max_new):
    with torch.inference_mode():
        step_input = cur_ids[:, cur_len-1:cur_len].contiguous()          # last token only
        logits = model(
            input_ids=step_input,
            attention_mask=attn_mask[:, :cur_len],                       # slice only
            use_cache=True,
            past_key_values=kv                                           # SAME cache object
        ).logits

        next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_ids], dim=1)                  # grow sequence view
        attn_mask[:, cur_len] = 1                                        # in-place update
        cur_len += 1

        if (step + 1) % 16 == 0:
            torch.cuda.synchronize()
            print(f"After {step+1:4d} decode tokens: {gb(torch.cuda.memory_allocated()):.2f} GB")

end = time.time()
torch.cuda.synchronize()
print(f"Final steady-state alloc: {gb(torch.cuda.memory_allocated()):.2f} GB")
print(f"Decode time for {max_new} tokens (batch {B}): {end - start:.2f}s")

# 4) Print outputs
for i in range(B):
    text = tokenizer.decode(cur_ids[i], skip_special_tokens=True)
    print("\n--- OUTPUT", i, "---")
    print(text)

# ---------------- Notes ----------------
# • If KV shows up on CPU instead of GPU, it means some attention blocks ran on CPU
#   (due to strict 8GiB cap). Increase GPU budget in `max_memory` slightly or pin
#   attention submodules to GPU in `device_map` to keep KV on GPU.
# • To “use all VRAM”, raise batch size (PROMPTS length) and/or max_new, and keep
#   the preallocated cache large so PyTorch reuses the big pool throughout decode.
