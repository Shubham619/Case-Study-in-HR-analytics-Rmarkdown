import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True")

import time
from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

# ============================ Config ============================
model_id   = "Qwen/Qwen1.5-MoE-A2.7B"
gpu_device = "cuda:0"
cpu_device = "cpu"
dtype      = torch.float16

GPU_CACHE_MAX_BYTES   = 2_000_000_000
GPU_CACHE_MAX_EXPERTS = 8
MAX_NEW_TOKENS        = 128

# ==================== Load full model on CPU ====================
print(f"Loading {model_id} fully on CPU …")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    trust_remote_code=True,
    device_map={"": cpu_device},   # no meta tensors
)
model.eval(); model.config.use_cache = True

# ==================== Move trunk + gates to GPU =================
def move_trunk_and_gates(model, gpu_device, dtype):
    dev = torch.device(gpu_device)
    for name, mod in model.named_modules():
        if "experts" in name:
            continue  # skip experts
        has_param = any(True for _ in mod.parameters(recurse=False))
        if has_param:
            mod.to(dev, dtype=dtype)
    for li, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.to(dev, dtype=dtype)
            if hasattr(layer.mlp.gate, "proj"):
                layer.mlp.gate.proj.to(dev, dtype=dtype)

print("Moving trunk + gates to GPU …")
move_trunk_and_gates(model, gpu_device, dtype)

# ================= Expert LRU cache =============================
def tensor_nbytes(t: torch.Tensor) -> int: return t.numel() * t.element_size()
def module_param_bytes(mod: nn.Module) -> int:
    return sum(tensor_nbytes(p.data) for p in mod.parameters(recurse=True))

def pin_state_dict(cpu_module: nn.Module):
    pinned = {}
    with torch.no_grad():
        for k, v in cpu_module.state_dict().items():
            t = v.detach().contiguous().to("cpu", copy=True)
            try: t = t.pin_memory()
            except RuntimeError: pass
            pinned[k] = t
    return pinned

class ExpertGPUCache:
    def __init__(self, device, dtype, max_bytes, max_count):
        self.device = torch.device(device); self.dtype = dtype
        self.max_bytes = max_bytes; self.max_count = max_count
        self._cache = OrderedDict(); self._sizes = {}; self._pinned_sd = {}
        self._bytes_total = 0

    def _ensure_capacity(self, need_bytes):
        while (self._bytes_total + need_bytes > self.max_bytes) or (len(self._cache) >= self.max_count):
            eid, mod = self._cache.popitem(last=False)
            self._bytes_total -= self._sizes.pop(eid, 0)
            del mod

    def _build_cuda_module(self, cpu_expert):
        import copy
        return copy.deepcopy(cpu_expert).to(self.device, dtype=self.dtype)

    def _load_weights(self, cuda_mod, expert_id):
        if expert_id not in self._pinned_sd:
            self._pinned_sd[expert_id] = pin_state_dict(cpu_expert_registry[expert_id])
        sd = self._pinned_sd[expert_id]
        for k, v_cpu in sd.items():
            ref = cuda_mod
            parts = k.split(".")
            for c in parts[:-1]:
                if hasattr(ref, c): ref = getattr(ref, c)
            leaf = parts[-1]
            if hasattr(ref, leaf):
                getattr(ref, leaf).data.copy_(v_cpu, non_blocking=True)

    def get(self, eid, cpu_expert):
        if eid in self._cache:
            mod = self._cache.pop(eid); self._cache[eid] = mod; return mod
        cuda_mod = self._build_cuda_module(cpu_expert)
        need_bytes = module_param_bytes(cuda_mod)
        self._ensure_capacity(need_bytes)
        self._load_weights(cuda_mod, eid)
        self._cache[eid] = cuda_mod; self._sizes[eid] = need_bytes
        self._bytes_total += need_bytes
        return cuda_mod

# ================== Discover experts (CPU) ======================
cpu_expert_registry, expert_name_map = {}, {}
eid = 0
for name, module in model.named_modules():
    if isinstance(module, nn.ModuleList) and "experts" in name:
        for idx, expert in enumerate(module):
            cpu_expert_registry[eid] = expert
            expert_name_map[eid] = f"{name}[{idx}]"
            eid += 1
print(f"CPU-resident experts: {len(cpu_expert_registry)}")

expert_cache = ExpertGPUCache(gpu_device, dtype, GPU_CACHE_MAX_BYTES, GPU_CACHE_MAX_EXPERTS)

# ================= Wrap experts ================================
expert_call_counts = defaultdict(int)
def make_cached_forward(eid, cpu_expert):
    def fwd(*args, **kwargs):
        expert_call_counts[eid] += 1
        args = [a.to(gpu_device, non_blocking=True) if torch.is_tensor(a) else a for a in args]
        kwargs = {k: v.to(gpu_device, non_blocking=True) if torch.is_tensor(v) else v for k,v in kwargs.items()}
        cuda_expert = expert_cache.get(eid, cpu_expert)
        return cuda_expert(*args, **kwargs)
    return fwd

for eid, exp in cpu_expert_registry.items():
    exp.forward = make_cached_forward(eid, exp)

# ================= Inputs + KV ================================
PROMPTS = ["Explain dynamic offloading in large language models."]
inputs = tokenizer(PROMPTS, return_tensors="pt").to(gpu_device)

B = inputs["input_ids"].size(0)
prompt_len = inputs["input_ids"].size(1)

kv = StaticCache(
    max_batch_size=B,
    max_cache_len=prompt_len + MAX_NEW_TOKENS,
    config=model.config,
    device=torch.device(gpu_device),
    dtype=dtype,
)

# ================= Prefill ================================
print("Running prefill …")
with torch.inference_mode():
    _ = model(**inputs, use_cache=True, past_key_values=kv)

print(f"Prefill done. Experts called: {dict(list(expert_call_counts.items())[:10])}")

# ================= Decode with promotions =================
def get_free_mem():
    return torch.cuda.memory_reserved() - torch.cuda.memory_allocated()

def move_expert_to_gpu(layer_idx, expert_idx):
    exp = model.model.layers[layer_idx].mlp.experts[expert_idx]
    exp.to(gpu_device)
    print(f"--> Expert {expert_idx} from layer {layer_idx} moved to GPU")

sorted_eids = sorted(expert_call_counts, key=lambda e: expert_call_counts[e], reverse=True)
moved_experts = set()

cur_ids = inputs["input_ids"]
attn_mask = torch.zeros((B, prompt_len + MAX_NEW_TOKENS), dtype=torch.long, device=gpu_device)
attn_mask[:, :prompt_len] = 1
cur_len = prompt_len

print("Decoding …")
for step in range(MAX_NEW_TOKENS):
    with torch.inference_mode():
        step_input = cur_ids[:, cur_len-1:cur_len]
        logits = model(input_ids=step_input, attention_mask=attn_mask[:, :cur_len],
                       use_cache=True, past_key_values=kv).logits
        next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_ids], dim=1)
        attn_mask[:, cur_len] = 1
        cur_len += 1

    free_now = get_free_mem()
    if free_now > 2 * 1024**3:  # ≥2 GB free
        count = 0
        for li, layer in enumerate(model.model.layers):
            for eid in sorted_eids:
                key = (li, eid)
                if key not in moved_experts:
                    move_expert_to_gpu(li, eid)
                    moved_experts.add(key)
                    count += 1
                    if count >= 5: break
            if count >= 5: break

print("\n--- OUTPUT ---")
print(tokenizer.decode(cur_ids[0], skip_special_tokens=True))
