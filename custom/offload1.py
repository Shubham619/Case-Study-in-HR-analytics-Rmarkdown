import os
# Keep Flash; reduce fragmentation and reuse freed VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True")

import time
from collections import OrderedDict, defaultdict
import types
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import StaticCache

# ============================ Config ============================
MODEL_ID   = "Qwen/Qwen1.5-MoE-A2.7B"
DTYPE      = torch.float16        # or torch.bfloat16
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_NEW    = 128
SAFETY_GB  = 2.0                  # keep this many GB free
VRAM_TRUNK = "10GiB"              # budget for trunk/router/attention
PRINT_DEBUG = True

# Expert-cache caps (both enforced)
GPU_CACHE_MAX_BYTES   = 2_000_000_000   # ~2.0 GB bytes cap for resident experts
GPU_CACHE_MAX_EXPERTS = 8               # count cap

PROMPTS = ["Explain the concept of dynamic offloading in large language models in one paragraph."]

# ============= Globals (modules must not hold CUDA Streams) =============
EXPERT_CACHE = None                     # set after class def
EXPERT_CALL_COUNTS = defaultdict(int)   # eid -> routed batch-count
PINNED_EIDS = set()                     # eids we never evict (e.g. shared experts)
PREFILL_MODE = True                     # evict experts immediately after prefill compute
CPU_EXPERT_REGISTRY = {}                # eid -> CPU expert module
EXPERT_NAME_MAP = {}                    # eid -> dotted path
EXPERT_PARAM_BYTES = {}                 # eid -> bytes estimate

# ==================== Helper: sizes & memory ====================
def tensor_nbytes(t: torch.Tensor) -> int: return t.numel() * t.element_size()
def module_param_bytes(mod: nn.Module) -> int:
    s = 0
    for p in mod.parameters(recurse=True): s += tensor_nbytes(p.data)
    for b in mod.buffers(recurse=True):   s += tensor_nbytes(b.data)
    return s
def gb(x): return x / 1e9

# ================= Expert LRU cache (resident-on-GPU modules) =================
def pin_state_dict(cpu_module: nn.Module):
    """
    Create a pinned-memory snapshot for fast H→D copies.
    Guard against 'meta' tensors (no storage to copy).
    """
    pinned = {}
    with torch.no_grad():
        sd = cpu_module.state_dict()
        for k, v in sd.items():
            if getattr(v, "is_meta", False) or v.device.type == "meta":
                raise RuntimeError(
                    f"Expert weight '{k}' is on meta device. "
                    "Ensure experts were placed on CPU before loading weights."
                )
            t = v.detach().contiguous().to("cpu", copy=True)
            try:  t = t.pin_memory()
            except RuntimeError:  pass
            pinned[k] = t
    return pinned

class ExpertGPUCache:
    """
    LRU cache for CUDA-resident experts.
    - Uses pinned CPU state_dict for faster H→D copies.
    - Enforces both bytes and count caps.
    - Skips eviction of PINNED_EIDS unless unavoidable.
    """
    def __init__(self, device: torch.device, dtype: torch.dtype,
                 max_bytes: int, max_count: int):
        self.device = device
        self.dtype  = dtype
        self.max_bytes = int(max_bytes)
        self.max_count = int(max_count)
        self._cache: OrderedDict[int, nn.Module] = OrderedDict()  # eid -> cuda module
        self._sizes: dict[int, int] = {}
        self._pinned_sd: dict[int, dict] = {}
        self._bytes_total = 0
        if device.type == "cuda":
            self.prefetch_stream = torch.cuda.Stream(device=device)
            self.main_stream    = torch.cuda.current_stream(device=device)
        else:
            self.prefetch_stream = None
            self.main_stream = None

    def _evict_one_lru(self) -> bool:
        """Evict one non-pinned expert from LRU side. Returns True if evicted."""
        for eid, mod in list(self._cache.items()):
            if eid in PINNED_EIDS:
                # move pinned to MRU to avoid checking repeatedly
                self._cache.pop(eid)
                self._cache[eid] = mod
                continue
            # evict this one
            self._cache.pop(eid)
            self._bytes_total -= self._sizes.pop(eid, 0)
            del mod
            return True
        return False

    def _ensure_capacity(self, need_bytes: int):
        while (self._bytes_total + need_bytes > self.max_bytes) or (len(self._cache) + 1 > self.max_count):
            if not self._evict_one_lru():
                # Nothing to evict without violating pin — break to let upstream decide
                break

    def _build_cuda_module(self, cpu_expert: nn.Module) -> nn.Module:
        cls = cpu_expert.__class__
        if hasattr(cpu_expert, "config"):
            m = cls(config=getattr(cpu_expert, "config", None)).to(self.device, dtype=self.dtype)
        else:
            import copy as _copy
            m = _copy.deepcopy(cpu_expert).to(self.device, dtype=self.dtype)
        return m

    def _load_weights_async(self, cuda_mod: nn.Module, eid: int):
        if self.device.type != "cuda":
            # CPU-only fallback (no streams)
            cuda_mod.load_state_dict(CPU_EXPERT_REGISTRY[eid].state_dict())
            return
        if eid not in self._pinned_sd:
            self._pinned_sd[eid] = pin_state_dict(CPU_EXPERT_REGISTRY[eid])
        sd = self._pinned_sd[eid]
        with torch.cuda.stream(self.prefetch_stream):
            for k, v_cpu in sd.items():
                ref = cuda_mod
                parts = k.split(".")
                for p in parts[:-1]:
                    ref = getattr(ref, p)
                leaf = parts[-1]
                if hasattr(ref, leaf):
                    t = getattr(ref, leaf)
                    t.data.copy_(v_cpu, non_blocking=True)
        self.main_stream.wait_stream(self.prefetch_stream)
        torch.cuda.current_stream(self.device).synchronize()

    def get(self, eid: int, cpu_expert: nn.Module) -> nn.Module:
        # Hit
        if eid in self._cache:
            mod = self._cache.pop(eid)
            self._cache[eid] = mod  # MRU
            return mod
        # Miss
        cuda_mod  = self._build_cuda_module(cpu_expert)
        need_bytes = EXPERT_PARAM_BYTES.get(eid, module_param_bytes(cuda_mod))
        EXPERT_PARAM_BYTES[eid] = need_bytes

        self._ensure_capacity(need_bytes)
        # If still no room, we may violate pin as last resort
        while (self._bytes_total + need_bytes > self.max_bytes) or (len(self._cache) + 1 > self.max_count):
            if not self._evict_one_lru():
                raise RuntimeError("ExpertGPUCache: cannot satisfy capacity for new expert (pinned set too large).")

        self._load_weights_async(cuda_mod, eid)
        self._cache[eid] = cuda_mod
        self._sizes[eid] = need_bytes
        self._bytes_total += need_bytes
        return cuda_mod

    def prefetch_experts(self, eids: list[int]):
        with torch.no_grad():
            for eid in eids:
                if eid in self._cache:
                    mod = self._cache.pop(eid)
                    self._cache[eid] = mod
                else:
                    if eid not in CPU_EXPERT_REGISTRY:
                        continue
                    cpu_mod = CPU_EXPERT_REGISTRY[eid]
                    cuda_mod = self._build_cuda_module(cpu_mod)
                    need_bytes = EXPERT_PARAM_BYTES.get(eid, module_param_bytes(cuda_mod))
                    EXPERT_PARAM_BYTES[eid] = need_bytes
                    self._ensure_capacity(need_bytes)
                    # Last resort if still over
                    while (self._bytes_total + need_bytes > self.max_bytes) or (len(self._cache) + 1 > self.max_count):
                        if not self._evict_one_lru():
                            if PRINT_DEBUG:
                                print("[Prefetch] Cannot grow cache further without evicting pinned experts.")
                            return
                    self._load_weights_async(cuda_mod, eid)
                    self._cache[eid] = cuda_mod
                    self._sizes[eid] = need_bytes
                    self._bytes_total += need_bytes

    def evict(self, eid: int):
        if eid in self._cache and eid not in PINNED_EIDS:
            mod = self._cache.pop(eid)
            self._bytes_total -= self._sizes.pop(eid, 0)
            del mod

# instantiate global cache (after class def)
EXPERT_CACHE = ExpertGPUCache(device=DEVICE, dtype=DTYPE,
                              max_bytes=GPU_CACHE_MAX_BYTES, max_count=GPU_CACHE_MAX_EXPERTS)

# ---------- Expert discovery (name+shape heuristics; no device moves) ----------
def is_mlp_like(module: nn.Module) -> bool:
    # Experts are typically MLPs: at least two Linear layers
    lin = sum(1 for _ in module.modules() if isinstance(_, nn.Linear))
    return lin >= 2

def discover_expert_lists(model: nn.Module):
    """
    Return a list of (layer_path, mlp_module) that contain experts.
    Works for Qwen MoE (mlp.experts + mlp.shared_expert).
    """
    found = []
    for path, mod in model.named_modules():
        # Heuristic: a module with attribute 'experts' that is a ModuleList of MLP-like blocks
        if hasattr(mod, "experts") and isinstance(getattr(mod, "experts"), nn.ModuleList):
            exps = getattr(mod, "experts")
            if len(exps) >= 2 and is_mlp_like(exps[0]):
                found.append((path, mod))
    return found

class ExpertProxy(nn.Module):
    """
    Dropped into the model in place of each expert (routing and shared).
    Holds only (expert_id, cpu_expert). Uses global EXPERT_CACHE & EXPERT_CALL_COUNTS.
    """
    def __init__(self, eid: int, cpu_expert: nn.Module):
        super().__init__()
        self.eid = eid
        self.cpu_expert = cpu_expert  # keep as submodule so state_dict still sees weights

    def forward(self, *args, **kwargs):
        # Count usage
        EXPERT_CALL_COUNTS[self.eid] += 1

        # Determine caller device from first tensor arg/kwarg
        def first_tensor_device():
            for x in args:
                if torch.is_tensor(x): return x.device
            for v in kwargs.values():
                if torch.is_tensor(v): return v.device
            try: return next(self.cpu_expert.parameters()).device
            except StopIteration: return torch.device("cpu")

        caller_dev = first_tensor_device()
        cuda_dev   = DEVICE

        # Move inputs to CUDA for compute (if not already)
        def to_cuda(x): return x.to(cuda_dev, non_blocking=True) if torch.is_tensor(x) and x.device != cuda_dev else x
        args_cuda   = tuple(to_cuda(a) for a in args)
        kwargs_cuda = {k: to_cuda(v) for k, v in kwargs.items()}

        # Get/instantiate CUDA expert & run
        cuda_expert = EXPERT_CACHE.get(self.eid, self.cpu_expert)
        with torch.no_grad():
            out = cuda_expert(*args_cuda, **kwargs_cuda)

        # Prefill: immediately drop this CUDA expert to keep VRAM for KV
        if PREFILL_MODE:
            EXPERT_CACHE.evict(self.eid)

        # Ensure outputs match caller device
        def to_caller(y):
            if torch.is_tensor(y) and y.device != caller_dev: return y.to(caller_dev, non_blocking=True)
            if isinstance(y, (list, tuple)): return type(y)(to_caller(t) for t in y)
            if isinstance(y, dict): return {k: to_caller(v) for k, v in y.items()}
            return y

        return to_caller(out)

# ==================== Load model (CPU) then move trunk to GPU ====================
print(f"Loading {MODEL_ID} …")
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
model  = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map={"": "cpu"}, low_cpu_mem_usage=True, trust_remote_code=True
)
tok    = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()

# Move trunk (attn/router/embeddings/ln) to GPU; keep experts on CPU
if DEVICE.type == "cuda":
    # Embedding & head
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.to(DEVICE)
    if hasattr(model, "lm_head"):
        model.lm_head = model.lm_head.to(DEVICE)
    # Layers
    for layer in model.model.layers:
        if hasattr(layer, "input_layernorm"):
            layer.input_layernorm = layer.input_layernorm.to(DEVICE)
        if hasattr(layer, "post_attention_layernorm"):
            layer.post_attention_layernorm = layer.post_attention_layernorm.to(DEVICE)
        if hasattr(layer, "self_attn"):
            layer.self_attn = layer.self_attn.to(DEVICE)
        # Router / gates to GPU; experts stay CPU (we'll proxy them)
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            if hasattr(mlp, "gate"):               mlp.gate = mlp.gate.to(DEVICE)
            if hasattr(mlp, "shared_expert_gate"): mlp.shared_expert_gate = mlp.shared_expert_gate.to(DEVICE)

# Discover expert banks and replace each expert with a Proxy
eid = 0
for path, mlp in discover_expert_lists(model):
    # Routing experts
    exps: nn.ModuleList = mlp.experts
    for idx, exp in enumerate(list(exps)):
        CPU_EXPERT_REGISTRY[eid] = exp
        EXPERT_NAME_MAP[eid] = f"{path}.experts[{idx}]"
        # estimate bytes once (CPU params)
        EXPERT_PARAM_BYTES[eid] = module_param_bytes(exp)
        exps[idx] = ExpertProxy(eid, exp)  # replace in place
        eid += 1
    # Shared expert (if present)
    if hasattr(mlp, "shared_expert") and isinstance(mlp.shared_expert, nn.Module):
        exp = mlp.shared_expert
        CPU_EXPERT_REGISTRY[eid] = exp
        EXPERT_NAME_MAP[eid] = f"{path}.shared_expert"
        EXPERT_PARAM_BYTES[eid] = module_param_bytes(exp)
        mlp.shared_expert = ExpertProxy(eid, exp)
        eid += 1

print(f"Inserted proxies for {len(CPU_EXPERT_REGISTRY)} experts.")
if PRINT_DEBUG:
    for k in list(CPU_EXPERT_REGISTRY.keys())[:6]:
        print("  ->", k, EXPERT_NAME_MAP[k])

# Optional: prefer FA2 if available (keeps Flash)
try:
    model.config.attn_implementation = "flash_attention_2"
except Exception:
    pass
model.config.use_cache = True

# =================== Tokenize & build single GPU KV (one prefill) ===================
inputs = tok(PROMPTS, return_tensors="pt", padding=True, truncation=False)
if DEVICE.type == "cuda":
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

B = inputs["input_ids"].size(0)
prompt_len = inputs["input_ids"].size(1)
kv = StaticCache(
    max_batch_size=B,
    max_cache_len=prompt_len + MAX_NEW,
    config=model.config,
    device=DEVICE,
    dtype=model.dtype if hasattr(model, "dtype") else DTYPE,
)

torch.cuda.synchronize() if DEVICE.type == "cuda" else None
torch.cuda.reset_peak_memory_stats() if DEVICE.type == "cuda" else None
base_alloc = torch.cuda.memory_allocated(DEVICE) if DEVICE.type == "cuda" else 0

# ------------------------- ONE PREFILL (fills KV & counts experts) -------------------
if PRINT_DEBUG: print("\n[Prefill] running single forward to warm KV & rank experts…")
with torch.inference_mode():
    _ = model(**inputs, use_cache=True, past_key_values=kv)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()
    prefill_peak = torch.cuda.max_memory_allocated(DEVICE)
    post_prefill = torch.cuda.memory_allocated(DEVICE)
    free_bytes, total_bytes = torch.cuda.mem_get_info(DEVICE)
    print(f"[MEM] base={gb(base_alloc):.2f} GB  peak_prefill={gb(prefill_peak):.2f} GB  "
          f"post_prefill={gb(post_prefill):.2f} GB  free={gb(free_bytes):.2f}/{gb(total_bytes):.2f} GB")

# Rank experts by usage
sorted_eids = sorted(EXPERT_CALL_COUNTS.keys(), key=lambda k: EXPERT_CALL_COUNTS[k], reverse=True)
if PRINT_DEBUG:
    print("\n[Prefill usage] top experts:")
    for i, e in enumerate(sorted_eids[:12]):
        print(f"  {i+1:2d}. EID {e:<4d} calls={EXPERT_CALL_COUNTS[e]:<6d}  {EXPERT_NAME_MAP.get(e,'')}")

# ------------------ Prefetch top-K into GPU (no 2nd prefill; pin shared) --------------------
PREFILL_MODE = False  # stop evict-on-use
if DEVICE.type == "cuda":
    free_now, total = torch.cuda.mem_get_info(DEVICE)
    safety = int(SAFETY_GB * 1e9)
    effective_free = max(0, free_now - safety)

    # How many by bytes?
    per_bytes = (EXPERT_PARAM_BYTES[sorted_eids[0]] if sorted_eids else 50_000_000)
    max_by_bytes = max(0, effective_free // max(per_bytes, 1))
    target_keep  = int(min(GPU_CACHE_MAX_EXPERTS, max_by_bytes))
    target_keep  = max(0, target_keep)

    # tighten cache caps to the plan
    EXPERT_CACHE.max_count = target_keep
    EXPERT_CACHE.max_bytes = int(min(GPU_CACHE_MAX_BYTES, per_bytes * target_keep))

    # Pin shared experts always (if any in list)
    for eid in sorted_eids:
        name = EXPERT_NAME_MAP.get(eid, "")
        if "shared_expert" in name:
            PINNED_EIDS.add(eid)

    top_keep_ids = sorted_eids[:target_keep]
    EXPERT_CACHE.prefetch_experts(top_keep_ids)
    if PRINT_DEBUG:
        print(f"\n[Plan] Keep top-{target_keep} experts on GPU (~{gb(per_bytes):.2f} GB each); "
              f"caps → bytes={gb(EXPERT_CACHE.max_bytes):.2f} GB, count={EXPERT_CACHE.max_count}")
        print(f"[Plan] Prefetched EIDs: {top_keep_ids}")
        if PINNED_EIDS:
            print(f"[Pin] Shared experts pinned: {sorted(list(PINNED_EIDS))}")

# ===================== Dynamic promoter (grow during decode) =========================
prefill_rank = sorted_eids[:]     # hottest → coldest
_promote_cursor = len(top_keep_ids) if DEVICE.type == "cuda" else 0

def try_promote_one_more_expert():
    global _promote_cursor
    if DEVICE.type != "cuda": return False
    if _promote_cursor >= len(prefill_rank): return False
    fb, _ = torch.cuda.mem_get_info(DEVICE)
    per_eid = prefill_rank[_promote_cursor]
    need = EXPERT_PARAM_BYTES.get(per_eid, 50_000_000)
    cushion = max(2 << 20, need // 8)  # +2MB or +12.5%
    if fb <= int(SAFETY_GB * 1e9) + need + cushion:
        return False
    # Respect caps (no relaxation here; safe)
    if (len(EXPERT_CACHE._cache) + 1 > EXPERT_CACHE.max_count) or \
       (EXPERT_CACHE._bytes_total + need > EXPERT_CACHE.max_bytes):
        return False
    EXPERT_CACHE.prefetch_experts([per_eid])
    if PRINT_DEBUG:
        print(f"[Promote] Pinned EID {per_eid} to GPU (free~{gb(torch.cuda.mem_get_info(DEVICE)[0]):.2f} GB)")
    _promote_cursor += 1
    return True

# ================================ Decode (reuse same KV) =============================
if PRINT_DEBUG: print("\n[Decode] starting greedy decode…")
cur_ids = inputs["input_ids"]  # (B, L)
attn_mask = torch.zeros((B, prompt_len + MAX_NEW), dtype=torch.long, device=DEVICE)
attn_mask[:, :prompt_len] = 1
cur_len = prompt_len

start = time.time()
for step in range(MAX_NEW):
    with torch.inference_mode():
        step_input = cur_ids[:, cur_len-1:cur_len].contiguous()
        out = model(
            input_ids=step_input,
            attention_mask=attn_mask[:, :cur_len],
            use_cache=True,
            past_key_values=kv
        )
        logits = out.logits
        next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_ids], dim=1)
        attn_mask[:, cur_len] = 1
        cur_len += 1

    # Opportunistic promotion every 8 steps
    if (step + 1) % 8 == 0:
        try_promote_one_more_expert()

    if PRINT_DEBUG and (step + 1) % 16 == 0 and DEVICE.type == "cuda":
        torch.cuda.synchronize()
        print(f"after {step+1:4d} tokens: mem_alloc={gb(torch.cuda.memory_allocated(DEVICE)):.2f} GB")

if DEVICE.type == "cuda":
    torch.cuda.synchronize()
print(f"\n[Decode] {MAX_NEW} tokens in {time.time()-start:.2f}s  |  "
      f"final_alloc={gb(torch.cuda.memory_allocated(DEVICE)):.2f} GB" if DEVICE.type == "cuda" else "")

# Output
for i in range(B):
    print(f"\n--- OUTPUT {i} ---\n{tok.decode(cur_ids[i], skip_special_tokens=True)}")

# Sanity checks
if DEVICE.type == "cuda":
    # KV tensors on CUDA
    for layer_idx in range(len(kv.key_cache)):
        k = kv.key_cache[layer_idx]; v = kv.value_cache[layer_idx]
        assert k.device.type == "cuda" and v.device.type == "cuda", "KV cache tensor left CUDA!"
    # Utilization summary
    used_eids = [k for k,v in EXPERT_CALL_COUNTS.items() if v > 0]
    print(f"\n[Summary] Experts discovered: {len(CPU_EXPERT_REGISTRY)} | used in prompt: {len(used_eids)}")
    print(f"[Summary] Resident experts now: {len(EXPERT_CACHE._cache)} | "
          f"bytes={gb(EXPERT_CACHE._bytes_total):.2f} GB | pinned={len(PINNED_EIDS)}")




import os
# Allocator tuned to reduce fragmentation (keeps freed VRAM reusable)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True")

import time
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import StaticCache
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

# ============================ Config ============================
model_id   = "Qwen/Qwen1.5-MoE-A2.7B"
gpu_device = "cuda:0"
cpu_device = "cpu"
dtype      = torch.float16

# Keep trunk/router/attention on GPU within this budget (experts → CPU)
max_memory = {
    0:   "10GiB",   # slight bump helps keep attention on GPU with KV
    "cpu":"300GiB"
}

# Expert cache knobs (kept & tuned at runtime)
GPU_CACHE_MAX_BYTES   = 2_000_000_000   # ~2.0 GB cap for resident experts
GPU_CACHE_MAX_EXPERTS = 8               # at most 8 experts resident
MAX_NEW_TOKENS        = 128
SAFETY_GB             = 2.0             # VRAM cushion we always leave free

# ======= Globals (so modules don’t hold non-picklable CUDA Streams) =======
EXPERT_CACHE = None               # set after class def
EXPERT_CALL_COUNTS = defaultdict(int)

# ==================== Load & dispatch (experts → CPU, attn → GPU) ====================
print(f"Loading config & skeleton for {model_id} …")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

print("Inferring device_map …")
no_split = getattr(empty_model, "_no_split_modules", ["QwenBlock", "Block"])
device_map = infer_auto_device_map(empty_model, max_memory=max_memory, no_split_module_classes=no_split)

# Heuristics: experts → CPU; attention/trunk → GPU (by name)
for name in list(device_map.keys()):
    low = name.lower()
    if ("experts" in low) or ("expert" in low) or ("moe" in low) or ("mixture" in low) or ("sparse" in low) or ("mixtral" in low):
        device_map[name] = cpu_device
    elif ("attn" in low) or ("attention" in low) or ("rotary" in low):
        device_map[name] = gpu_device  # keep attention & KV writers on GPU

print("Downloading checkpoints & dispatching …")
local_ckpt = snapshot_download(model_id, allow_patterns=["*.safetensors","*.bin","config.json","*.json"])
model = load_checkpoint_and_dispatch(
    empty_model,
    checkpoint=local_ckpt,
    device_map=device_map,
    no_split_module_classes=no_split,
    dtype=dtype,
    offload_folder=None,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.eval(); model.config.use_cache = True

# ================= Expert LRU cache (resident-on-GPU modules) =================
def tensor_nbytes(t: torch.Tensor) -> int: return t.numel() * t.element_size()
def module_param_bytes(mod: nn.Module) -> int:
    s = 0
    for p in mod.parameters(recurse=True): s += tensor_nbytes(p.data)
    for b in mod.buffers(recurse=True):   s += tensor_nbytes(b.data)
    return s

def pin_state_dict(cpu_module: nn.Module):
    """
    Create a pinned-memory snapshot for fast H→D copies.
    Guard against 'meta' tensors (no storage to copy).
    """
    pinned = {}
    with torch.no_grad():
        sd = cpu_module.state_dict()
        for k, v in sd.items():
            if getattr(v, "is_meta", False) or v.device.type == "meta":
                raise RuntimeError(
                    f"Expert weight '{k}' is on meta device. "
                    "Ensure device_map placed this expert on CPU before dispatch."
                )
            t = v.detach().contiguous().to("cpu", copy=True)
            try:
                t = t.pin_memory()
            except RuntimeError:
                pass
            pinned[k] = t
    return pinned

class ExpertGPUCache:
    def __init__(self, device:str, dtype:torch.dtype,
                 max_bytes:int=GPU_CACHE_MAX_BYTES, max_count:int=GPU_CACHE_MAX_EXPERTS):
        self.device = torch.device(device); self.dtype = dtype
        self.max_bytes = max_bytes; self.max_count = max_count
        self._cache: OrderedDict[int, nn.Module] = OrderedDict()
        self._sizes: dict[int,int] = {}
        self._pinned_sd: dict[int,dict] = {}
        self._bytes_total = 0
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.main_stream = torch.cuda.current_stream(device=self.device)

    def _ensure_capacity(self, need_bytes:int):
        while (self._bytes_total + need_bytes > self.max_bytes) or (len(self._cache)+1 > self.max_count):
            evict_id, evict_mod = self._cache.popitem(last=False)
            self._bytes_total -= self._sizes.pop(evict_id, 0)
            del evict_mod

    def _build_cuda_module(self, cpu_expert: nn.Module) -> nn.Module:
        cls = cpu_expert.__class__
        if hasattr(cpu_expert, "config"):
            m = cls(config=cpu_expert.config).to(self.device, dtype=self.dtype)
        else:
            import copy as _copy
            m = _copy.deepcopy(cpu_expert).to(self.device, dtype=self.dtype)
        return m

    def _load_weights_async(self, cuda_mod: nn.Module, expert_id:int):
        if expert_id not in self._pinned_sd:
            self._pinned_sd[expert_id] = pin_state_dict(CPU_EXPERT_REGISTRY[expert_id])
        sd = self._pinned_sd[expert_id]
        with torch.cuda.stream(self.prefetch_stream):
            for k, v_cpu in sd.items():
                ref = cuda_mod
                comps = k.split(".")
                for c in comps[:-1]: ref = getattr(ref, c)
                leaf = comps[-1]
                if hasattr(ref, leaf):
                    t = getattr(ref, leaf)
                    t.data.copy_(v_cpu, non_blocking=True)
        self.main_stream.wait_stream(self.prefetch_stream)
        torch.cuda.current_stream(self.device).synchronize()

    def get(self, expert_id:int, cpu_expert: nn.Module) -> nn.Module:
        if expert_id in self._cache:
            mod = self._cache.pop(expert_id); self._cache[expert_id] = mod; return mod
        cuda_mod  = self._build_cuda_module(cpu_expert)
        need_bytes = module_param_bytes(cuda_mod)
        self._ensure_capacity(need_bytes)
        self._load_weights_async(cuda_mod, expert_id)
        self._cache[expert_id] = cuda_mod
        self._sizes[expert_id] = need_bytes
        self._bytes_total += need_bytes
        return cuda_mod

    def prefetch_experts(self, ids:list[int]):
        with torch.no_grad():
            for eid in ids:
                if eid in self._cache:
                    mod = self._cache.pop(eid); self._cache[eid] = mod
                else:
                    if eid not in CPU_EXPERT_REGISTRY: continue
                    cpu_mod = CPU_EXPERT_REGISTRY[eid]
                    cuda_mod = self._build_cuda_module(cpu_mod)
                    need_bytes = module_param_bytes(cuda_mod)
                    self._ensure_capacity(need_bytes)
                    self._load_weights_async(cuda_mod, eid)
                    self._cache[eid] = cuda_mod
                    self._sizes[eid] = need_bytes
                    self._bytes_total += need_bytes

# instantiate global cache after class definition
EXPERT_CACHE = ExpertGPUCache(device=gpu_device, dtype=dtype,
                              max_bytes=GPU_CACHE_MAX_BYTES, max_count=GPU_CACHE_MAX_EXPERTS)

# ---------- Robust expert discovery (name+shape heuristics; no device moves) ----------
def is_mlp_like(module: nn.Module) -> bool:
    # Experts are typically MLPs: at least two Linear layers
    lin = sum(1 for _ in module.modules() if isinstance(_, nn.Linear))
    return lin >= 2

def discover_expert_lists(model: nn.Module):
    """
    Return a list of (module_path, ModuleList) that look like expert banks.
    """
    patterns = ("experts", "expert", "moe", "mixture", "sparse", "mixtral")
    found = []
    for path, mod in model.named_modules():
        if not isinstance(mod, nn.ModuleList):
            continue
        low = path.lower()
        looks_like = any(p in low for p in patterns) or (len(mod) >= 4 and is_mlp_like(mod[0]))
        if not looks_like:
            continue
        # homogeneous & MLP-like
        base_cls = mod[0].__class__
        if not all(isinstance(m, base_cls) for m in mod):
            continue
        if not is_mlp_like(mod[0]):
            continue
        found.append((path, mod))
    return found

# Build registry and replace experts with Proxy modules (avoid recursion & pickling)
CPU_EXPERT_REGISTRY: dict[int, nn.Module] = {}
EXPERT_NAME_MAP: dict[int, str] = {}

class ExpertProxy(nn.Module):
    """
    Thin wrapper placed in the model in place of each expert.
    Holds only (expert_id, cpu_expert). Uses global EXPERT_CACHE & EXPERT_CALL_COUNTS.
    """
    def __init__(self, expert_id: int, cpu_expert: nn.Module):
        super().__init__()
        self.expert_id = expert_id
        self.cpu_expert = cpu_expert  # keep as submodule so state_dict still sees weights

    def forward(self, *args, **kwargs):
        # Count usage
        EXPERT_CALL_COUNTS[self.expert_id] += 1

        # Determine caller device (first tensor in args/kwargs)
        def first_tensor_device():
            for x in args:
                if torch.is_tensor(x): return x.device
            for v in kwargs.values():
                if torch.is_tensor(v): return v.device
            try: return next(self.cpu_expert.parameters()).device
            except StopIteration: return torch.device("cpu")

        caller_dev = first_tensor_device()
        cuda_dev   = torch.device(gpu_device)

        # Move inputs to CUDA
        def to_cuda(x): return x.to(cuda_dev, non_blocking=True) if torch.is_tensor(x) else x
        args_cuda   = tuple(to_cuda(a) for a in args)
        kwargs_cuda = {k: to_cuda(v) for k, v in kwargs.items()}

        # Get/instantiate CUDA expert & run
        cuda_expert = EXPERT_CACHE.get(self.expert_id, self.cpu_expert)
        with torch.no_grad():
            out = cuda_expert(*args_cuda, **kwargs_cuda)

        # Move outputs back to caller device (prevents device mismatch)
        def to_caller(y):
            if torch.is_tensor(y): return y.to(caller_dev, non_blocking=True)
            if isinstance(y, (list, tuple)): return type(y)(to_caller(t) for t in y)
            if isinstance(y, dict): return {k: to_caller(v) for k, v in y.items()}
            return y

        return to_caller(out)

# Discover expert lists and replace each item with a proxy
eid = 0
expert_lists = discover_expert_lists(model)
for path, modlist in expert_lists:
    for idx, expert in enumerate(modlist):
        proxy = ExpertProxy(eid, expert)
        modlist[idx] = proxy  # replace in-place
        CPU_EXPERT_REGISTRY[eid] = proxy.cpu_expert  # keep original CPU module in registry
        EXPERT_NAME_MAP[eid] = f"{path}[{idx}]"
        eid += 1

print(f"Inserted proxies for {len(CPU_EXPERT_REGISTRY)} experts.")
for k in list(CPU_EXPERT_REGISTRY.keys())[:6]:
    print("  ->", k, EXPERT_NAME_MAP[k])

# =================== Prepare inputs, single GPU KV, ONE prefill ======================
PROMPTS = ["Explain the concept of dynamic offloading in large language models in one paragraph."]
inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=False)
inputs = {k: v.to(gpu_device) for k, v in inputs.items()}

B = inputs["input_ids"].size(0)
prompt_len = inputs["input_ids"].size(1)
max_new = MAX_NEW_TOKENS

# Single, GPU-resident KV for the whole run
kv = StaticCache(
    max_batch_size=B,
    max_cache_len=prompt_len + max_new,  # reserve up to final length
    config=model.config,
    device=torch.device(gpu_device),
    dtype=model.dtype,
)

def gb(x): return x/1e9
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
base_alloc = torch.cuda.memory_allocated()

# ------------------------- ONE PREFILL (fills KV & counts experts) -------------------
with torch.inference_mode():
    _ = model(**inputs, use_cache=True, past_key_values=kv)

torch.cuda.synchronize()
prefill_peak = torch.cuda.max_memory_allocated()
post_prefill = torch.cuda.memory_allocated()
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"\n[MEM] base={gb(base_alloc):.2f} GB  peak_prefill={gb(prefill_peak):.2f} GB  "
      f"post_prefill={gb(post_prefill):.2f} GB  free={gb(free_bytes):.2f}/{gb(total_bytes):.2f} GB")

# ------------------ Decide which experts to pin (NO 2nd prefill) --------------------
# Rank by prefill usage
sorted_eids = sorted(EXPERT_CALL_COUNTS.keys(), key=lambda k: EXPERT_CALL_COUNTS[k], reverse=True)

print("\n[Prefill usage] top experts:")
for i, e in enumerate(sorted_eids[:12]):
    print(f"  {i+1:2d}. EID {e:<4d} calls={EXPERT_CALL_COUNTS[e]:<6d}  {EXPERT_NAME_MAP.get(e,'')}")

# Probe per-expert size once (temporary CUDA instantiation; NOT another prefill)
def probe_expert_bytes(eid_probe:int) -> int:
    cpu_mod = CPU_EXPERT_REGISTRY[eid_probe]
    cls = cpu_mod.__class__
    if hasattr(cpu_mod, "config"):
        tmp = cls(config=cpu_mod.config).to(gpu_device, dtype=dtype)
    else:
        import copy as _copy
        tmp = _copy.deepcopy(cpu_mod).to(gpu_device, dtype=dtype)
    nbytes = module_param_bytes(tmp)
    del tmp; torch.cuda.synchronize()
    return nbytes

if sorted_eids:
    per_expert_bytes = probe_expert_bytes(sorted_eids[0])
else:
    per_expert_bytes = 50_000_000  # fallback estimate if none used (unlikely)

safety = int(SAFETY_GB * 1e9)
max_by_bytes = max(0, (free_bytes - safety) // max(per_expert_bytes, 1))
target_keep = int(min(GPU_CACHE_MAX_EXPERTS, max_by_bytes))
target_keep = max(0, target_keep)

# Tighten caps to the plan (keep both knobs)
EXPERT_CACHE.max_count = target_keep
EXPERT_CACHE.max_bytes = int(min(GPU_CACHE_MAX_BYTES, per_expert_bytes * target_keep))

# Prefetch the chosen top-K NOW (still before any decode tokens)
top_keep_ids = sorted_eids[:target_keep]
EXPERT_CACHE.prefetch_experts(top_keep_ids)
print(f"\n[Plan] Keep top-{target_keep} experts on GPU (~{gb(per_expert_bytes):.2f} GB each); "
      f"cache caps → bytes={gb(EXPERT_CACHE.max_bytes):.2f} GB, count={EXPERT_CACHE.max_count}")
print(f"[Plan] Prefetched EIDs: {top_keep_ids}")

# ===================== Dynamic promoter (grow during decode) =========================
prefill_rank = sorted_eids[:]                  # hottest → coldest
_promote_cursor = len(top_keep_ids)            # next index to promote

def cuda_free_bytes():
    fb, _ = torch.cuda.mem_get_info()
    return int(fb)

def try_promote_one_more_expert(safety_bytes:int, per_expert_bytes:int):
    global _promote_cursor
    if _promote_cursor >= len(prefill_rank):
        return False  # nothing left to promote

    free_now = cuda_free_bytes()
    need = per_expert_bytes + max(2 << 20, per_expert_bytes // 8)  # cushion
    if free_now <= safety_bytes + need:
        return False

    # Respect knobs; allow gentle growth if obvious headroom exists
    if len(EXPERT_CACHE._cache) >= EXPERT_CACHE.max_count:
        if EXPERT_CACHE._bytes_total + per_expert_bytes <= EXPERT_CACHE.max_bytes:
            EXPERT_CACHE.max_count += 1
        elif EXPERT_CACHE._bytes_total + per_expert_bytes <= EXPERT_CACHE.max_bytes + (per_expert_bytes // 2):
            EXPERT_CACHE.max_bytes += per_expert_bytes
        else:
            return False

    if (len(EXPERT_CACHE._cache) + 1 > EXPERT_CACHE.max_count) or \
       (EXPERT_CACHE._bytes_total + per_expert_bytes > EXPERT_CACHE.max_bytes):
        return False

    eid_promote = prefill_rank[_promote_cursor]
    _promote_cursor += 1
    EXPERT_CACHE.prefetch_experts([eid_promote])
    print(f"[Promote] Pinned EID {eid_promote} to GPU (free~{cuda_free_bytes()/1e9:.2f} GB)")
    return True

# ================================ Decode (reuse same KV) =============================
B = inputs["input_ids"].size(0)
cur_ids = inputs["input_ids"]
attn_mask = torch.zeros((B, prompt_len + MAX_NEW_TOKENS), dtype=torch.long, device=gpu_device)
attn_mask[:, :prompt_len] = 1
cur_len = prompt_len

start = time.time()
for step in range(MAX_NEW_TOKENS):
    with torch.inference_mode():
        step_input = cur_ids[:, cur_len-1:cur_len].contiguous()
        logits = model(
            input_ids=step_input,
            attention_mask=attn_mask[:, :cur_len],
            use_cache=True,
            past_key_values=kv
        ).logits
        next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_ids], dim=1)
        attn_mask[:, cur_len] = 1
        cur_len += 1

    # Opportunistically promote one more expert every 8 steps if there's headroom
    if (step + 1) % 8 == 0 and per_expert_bytes:
        try_promote_one_more_expert(safety_bytes=safety, per_expert_bytes=per_expert_bytes)

    if (step + 1) % 16 == 0:
        torch.cuda.synchronize()
        print(f"after {step+1:4d} decode tokens: mem_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

torch.cuda.synchronize()
print(f"\n[Decode] {MAX_NEW_TOKENS} tokens in {time.time()-start:.2f}s  |  final_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

# Output
for i in range(B):
    print(f"\n--- OUTPUT {i} ---\n{tokenizer.decode(cur_ids[i], skip_special_tokens=True)}")


import os
# allocator tuned to reduce fragmentation (keeps freed VRAM reusable)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True")

import time
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import StaticCache
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

# ============================ Config ============================
model_id   = "Qwen/Qwen1.5-MoE-A2.7B"
gpu_device = "cuda:0"
cpu_device = "cpu"
dtype      = torch.float16

# Keep trunk/router/attention on GPU within this budget (experts → CPU)
max_memory = {
    0:   "10GiB",   # small bump to keep attention on GPU with KV
    "cpu":"300GiB"
}

# Expert cache knobs (kept & tuned at runtime)
GPU_CACHE_MAX_BYTES   = 2_000_000_000   # ~2.0 GB cap for resident experts
GPU_CACHE_MAX_EXPERTS = 8               # at most 8 experts resident
MAX_NEW_TOKENS        = 128
SAFETY_GB             = 2.0             # VRAM cushion we always leave free

# ==================== Load & dispatch (experts → CPU, attn → GPU) ====================
print(f"Loading config & skeleton for {model_id} …")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

print("Inferring device_map …")
no_split = getattr(empty_model, "_no_split_modules", ["QwenBlock", "Block"])
device_map = infer_auto_device_map(empty_model, max_memory=max_memory, no_split_module_classes=no_split)

# Heuristics: experts → CPU; attention/trunk → GPU (by name)
for name in list(device_map.keys()):
    low = name.lower()
    if ("experts" in low) or ("expert" in low) or ("moe" in low) or ("mixture" in low) or ("sparse" in low) or ("mixtral" in low):
        device_map[name] = cpu_device
    elif ("attn" in low) or ("attention" in low) or ("rotary" in low):
        device_map[name] = gpu_device  # keep attention & KV writers on GPU

print("Downloading checkpoints & dispatching …")
local_ckpt = snapshot_download(model_id, allow_patterns=["*.safetensors","*.bin","config.json","*.json"])
model = load_checkpoint_and_dispatch(
    empty_model,
    checkpoint=local_ckpt,
    device_map=device_map,
    no_split_module_classes=no_split,
    dtype=dtype,
    offload_folder=None,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.eval(); model.config.use_cache = True

# ================= Expert LRU cache (resident-on-GPU modules) =================
def tensor_nbytes(t: torch.Tensor) -> int: return t.numel() * t.element_size()
def module_param_bytes(mod: nn.Module) -> int:
    s = 0
    for p in mod.parameters(recurse=True): s += tensor_nbytes(p.data)
    for b in mod.buffers(recurse=True):   s += tensor_nbytes(b.data)
    return s

def pin_state_dict(cpu_module: nn.Module):
    """
    Create a pinned-memory snapshot for fast H→D copies.
    Guard against 'meta' tensors (no storage to copy).
    """
    pinned = {}
    with torch.no_grad():
        sd = cpu_module.state_dict()
        for k, v in sd.items():
            if getattr(v, "is_meta", False) or v.device.type == "meta":
                raise RuntimeError(
                    f"Expert weight '{k}' is on meta device. "
                    "Ensure device_map placed this expert on CPU before dispatch."
                )
            t = v.detach().contiguous().to("cpu", copy=True)
            try:
                t = t.pin_memory()
            except RuntimeError:
                pass
            pinned[k] = t
    return pinned

class ExpertGPUCache:
    def __init__(self, device:str, dtype:torch.dtype,
                 max_bytes:int=GPU_CACHE_MAX_BYTES, max_count:int=GPU_CACHE_MAX_EXPERTS):
        self.device = torch.device(device); self.dtype = dtype
        self.max_bytes = max_bytes; self.max_count = max_count
        self._cache: OrderedDict[int, nn.Module] = OrderedDict()
        self._sizes: dict[int,int] = {}
        self._pinned_sd: dict[int,dict] = {}
        self._bytes_total = 0
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.main_stream = torch.cuda.current_stream(device=self.device)

    def _ensure_capacity(self, need_bytes:int):
        while (self._bytes_total + need_bytes > self.max_bytes) or (len(self._cache)+1 > self.max_count):
            evict_id, evict_mod = self._cache.popitem(last=False)
            self._bytes_total -= self._sizes.pop(evict_id, 0)
            del evict_mod

    def _build_cuda_module(self, cpu_expert: nn.Module) -> nn.Module:
        cls = cpu_expert.__class__
        if hasattr(cpu_expert, "config"):
            m = cls(config=cpu_expert.config).to(self.device, dtype=self.dtype)
        else:
            import copy as _copy
            m = _copy.deepcopy(cpu_expert).to(self.device, dtype=self.dtype)
        return m

    def _load_weights_async(self, cuda_mod: nn.Module, expert_id:int):
        if expert_id not in self._pinned_sd:
            self._pinned_sd[expert_id] = pin_state_dict(cpu_expert_registry[expert_id])
        sd = self._pinned_sd[expert_id]
        with torch.cuda.stream(self.prefetch_stream):
            for k, v_cpu in sd.items():
                ref = cuda_mod
                comps = k.split(".")
                for c in comps[:-1]: ref = getattr(ref, c)
                leaf = comps[-1]
                if hasattr(ref, leaf):
                    t = getattr(ref, leaf)
                    t.data.copy_(v_cpu, non_blocking=True)
        self.main_stream.wait_stream(self.prefetch_stream)
        torch.cuda.current_stream(self.device).synchronize()

    def get(self, expert_id:int, cpu_expert: nn.Module) -> nn.Module:
        if expert_id in self._cache:
            mod = self._cache.pop(expert_id); self._cache[expert_id] = mod; return mod
        cuda_mod  = self._build_cuda_module(cpu_expert)
        need_bytes = module_param_bytes(cuda_mod)
        self._ensure_capacity(need_bytes)
        self._load_weights_async(cuda_mod, expert_id)
        self._cache[expert_id] = cuda_mod
        self._sizes[expert_id] = need_bytes
        self._bytes_total += need_bytes
        return cuda_mod

    def prefetch_experts(self, ids:list[int]):
        with torch.no_grad():
            for eid in ids:
                if eid in self._cache:
                    mod = self._cache.pop(eid); self._cache[eid] = mod
                else:
                    if eid not in cpu_expert_registry: continue
                    cpu_mod = cpu_expert_registry[eid]
                    cuda_mod = self._build_cuda_module(cpu_mod)
                    need_bytes = module_param_bytes(cuda_mod)
                    self._ensure_capacity(need_bytes)
                    self._load_weights_async(cuda_mod, eid)
                    self._cache[eid] = cuda_mod
                    self._sizes[eid] = need_bytes
                    self._bytes_total += need_bytes

# ---------- Robust expert discovery (name+shape heuristics; no device moves) ----------
def is_mlp_like(module: nn.Module) -> bool:
    # Experts are typically MLPs: at least two Linear layers
    lin = sum(1 for _ in module.modules() if isinstance(_, nn.Linear))
    return lin >= 2

def discover_expert_lists(model: nn.Module):
    """
    Return a list of (module_path, ModuleList) that look like expert banks.
    """
    patterns = ("experts", "expert", "moe", "mixture", "sparse", "mixtral")
    found = []
    for path, mod in model.named_modules():
        if not isinstance(mod, nn.ModuleList):
            continue
        low = path.lower()
        looks_like = any(p in low for p in patterns) or (len(mod) >= 4 and is_mlp_like(mod[0]))
        if not looks_like:
            continue
        # homogeneous & MLP-like
        base_cls = mod[0].__class__
        if not all(isinstance(m, base_cls) for m in mod):
            continue
        if not is_mlp_like(mod[0]):
            continue
        found.append((path, mod))
    return found

# Build registry and replace experts with Proxy modules (avoid recursion)
cpu_expert_registry: dict[int, nn.Module] = {}
expert_name_map: dict[int, str] = {}
expert_id_of_proxy: dict[nn.Module, int] = {}  # Proxy -> id (for debugging)

class ExpertProxy(nn.Module):
    """
    Thin wrapper placed in the model in place of each expert.
    It holds the original CPU expert (as a submodule so state_dict still sees it),
    routes compute to a cached CUDA clone, and returns outputs on the caller's device.
    """
    def __init__(self, expert_id: int, cpu_expert: nn.Module, cache_ref, counts_ref):
        super().__init__()
        self.expert_id = expert_id
        self.cpu_expert = cpu_expert  # keep as submodule
        self._cache_ref = cache_ref
        self._counts_ref = counts_ref

    def forward(self, *args, **kwargs):
        # Count usage
        self._counts_ref[self.expert_id] += 1

        # Determine caller device (first tensor in args/kwargs)
        def first_tensor_device():
            for x in args:
                if torch.is_tensor(x): return x.device
            for v in kwargs.values():
                if torch.is_tensor(v): return v.device
            try: return next(self.cpu_expert.parameters()).device
            except StopIteration: return torch.device("cpu")

        caller_dev = first_tensor_device()
        cuda_dev   = torch.device(gpu_device)

        # Move inputs to CUDA
        def to_cuda(x): return x.to(cuda_dev, non_blocking=True) if torch.is_tensor(x) else x
        args_cuda   = tuple(to_cuda(a) for a in args)
        kwargs_cuda = {k: to_cuda(v) for k, v in kwargs.items()}

        # Get/instantiate CUDA expert & run
        cuda_expert = self._cache_ref.get(self.expert_id, self.cpu_expert)
        with torch.no_grad():
            out = cuda_expert(*args_cuda, **kwargs_cuda)

        # Move outputs back to caller device (prevents device mismatch)
        def to_caller(y):
            if torch.is_tensor(y): return y.to(caller_dev, non_blocking=True)
            if isinstance(y, (list, tuple)): return type(y)(to_caller(t) for t in y)
            if isinstance(y, dict): return {k: to_caller(v) for k, v in y.items()}
            return y

        return to_caller(out)

# Create cache and usage counter BEFORE inserting proxies
expert_cache = ExpertGPUCache(device=gpu_device, dtype=dtype,
                              max_bytes=GPU_CACHE_MAX_BYTES, max_count=GPU_CACHE_MAX_EXPERTS)
expert_call_counts = defaultdict(int)

# Discover expert lists and replace each item with a proxy
eid = 0
expert_lists = discover_expert_lists(model)
for path, modlist in expert_lists:
    for idx, expert in enumerate(modlist):
        proxy = ExpertProxy(eid, expert, expert_cache, expert_call_counts)
        # replace in-place
        modlist[idx] = proxy
        cpu_expert_registry[eid] = proxy.cpu_expert  # keep original CPU module in registry
        expert_name_map[eid] = f"{path}[{idx}]"
        expert_id_of_proxy[proxy] = eid
        eid += 1

print(f"Inserted proxies for {len(cpu_expert_registry)} experts.")
for k in list(cpu_expert_registry.keys())[:6]:
    print("  ->", k, expert_name_map[k])

# =================== Prepare inputs, single GPU KV, ONE prefill ======================
PROMPTS = ["Explain the concept of dynamic offloading in large language models in one paragraph."]
inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=False)
inputs = {k: v.to(gpu_device) for k, v in inputs.items()}

B = inputs["input_ids"].size(0)
prompt_len = inputs["input_ids"].size(1)
max_new = MAX_NEW_TOKENS

# Single, GPU-resident KV for the whole run
kv = StaticCache(
    max_batch_size=B,
    max_cache_len=prompt_len + max_new,  # reserve up to final length
    config=model.config,
    device=torch.device(gpu_device),
    dtype=model.dtype,
)

def gb(x): return x/1e9
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
base_alloc = torch.cuda.memory_allocated()

# ------------------------- ONE PREFILL (fills KV & counts experts) -------------------
with torch.inference_mode():
    _ = model(**inputs, use_cache=True, past_key_values=kv)

torch.cuda.synchronize()
prefill_peak = torch.cuda.max_memory_allocated()
post_prefill = torch.cuda.memory_allocated()
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"\n[MEM] base={gb(base_alloc):.2f} GB  peak_prefill={gb(prefill_peak):.2f} GB  "
      f"post_prefill={gb(post_prefill):.2f} GB  free={gb(free_bytes):.2f}/{gb(total_bytes):.2f} GB")

# ------------------ Decide which experts to pin (NO 2nd prefill) --------------------
# Rank by prefill usage
sorted_eids = sorted(expert_call_counts.keys(), key=lambda k: expert_call_counts[k], reverse=True)

print("\n[Prefill usage] top experts:")
for i, e in enumerate(sorted_eids[:12]):
    print(f"  {i+1:2d}. EID {e:<4d} calls={expert_call_counts[e]:<6d}  {expert_name_map.get(e,'')}")

# Probe per-expert size once (temporary CUDA instantiation; NOT another prefill)
def probe_expert_bytes(eid_probe:int) -> int:
    cpu_mod = cpu_expert_registry[eid_probe]
    cls = cpu_mod.__class__
    if hasattr(cpu_mod, "config"):
        tmp = cls(config=cpu_mod.config).to(gpu_device, dtype=dtype)
    else:
        import copy as _copy
        tmp = _copy.deepcopy(cpu_mod).to(gpu_device, dtype=dtype)
    nbytes = module_param_bytes(tmp)
    del tmp; torch.cuda.synchronize()
    return nbytes

if sorted_eids:
    per_expert_bytes = probe_expert_bytes(sorted_eids[0])
else:
    per_expert_bytes = 50_000_000  # fallback estimate if none used (unlikely)

safety = int(SAFETY_GB * 1e9)
max_by_bytes = max(0, (free_bytes - safety) // max(per_expert_bytes, 1))
target_keep = int(min(GPU_CACHE_MAX_EXPERTS, max_by_bytes))
target_keep = max(0, target_keep)

# Tighten caps to the plan (keep both knobs)
expert_cache.max_count = target_keep
expert_cache.max_bytes = int(min(GPU_CACHE_MAX_BYTES, per_expert_bytes * target_keep))

# Prefetch the chosen top-K NOW (still before any decode tokens)
top_keep_ids = sorted_eids[:target_keep]
expert_cache.prefetch_experts(top_keep_ids)
print(f"\n[Plan] Keep top-{target_keep} experts on GPU (~{gb(per_expert_bytes):.2f} GB each); "
      f"cache caps → bytes={gb(expert_cache.max_bytes):.2f} GB, count={expert_cache.max_count}")
print(f"[Plan] Prefetched EIDs: {top_keep_ids}")

# ===================== Dynamic promoter (grow during decode) =========================
prefill_rank = sorted_eids[:]                  # hottest → coldest
_promote_cursor = len(top_keep_ids)            # next index to promote

def cuda_free_bytes():
    fb, _ = torch.cuda.mem_get_info()
    return int(fb)

def try_promote_one_more_expert(safety_bytes:int, per_expert_bytes:int):
    global _promote_cursor
    if _promote_cursor >= len(prefill_rank):
        return False  # nothing left to promote

    free_now = cuda_free_bytes()
    need = per_expert_bytes + max(2 << 20, per_expert_bytes // 8)  # cushion
    if free_now <= safety_bytes + need:
        return False

    # Respect knobs; allow gentle growth if obvious headroom exists
    if len(expert_cache._cache) >= expert_cache.max_count:
        if expert_cache._bytes_total + per_expert_bytes <= expert_cache.max_bytes:
            expert_cache.max_count += 1
        elif expert_cache._bytes_total + per_expert_bytes <= expert_cache.max_bytes + (per_expert_bytes // 2):
            expert_cache.max_bytes += per_expert_bytes
        else:
            return False

    if (len(expert_cache._cache) + 1 > expert_cache.max_count) or \
       (expert_cache._bytes_total + per_expert_bytes > expert_cache.max_bytes):
        return False

    eid_promote = prefill_rank[_promote_cursor]
    _promote_cursor += 1
    expert_cache.prefetch_experts([eid_promote])
    print(f"[Promote] Pinned EID {eid_promote} to GPU (free~{cuda_free_bytes()/1e9:.2f} GB)")
    return True

# ================================ Decode (reuse same KV) =============================
B = inputs["input_ids"].size(0)
cur_ids = inputs["input_ids"]
attn_mask = torch.zeros((B, prompt_len + MAX_NEW_TOKENS), dtype=torch.long, device=gpu_device)
attn_mask[:, :prompt_len] = 1
cur_len = prompt_len

start = time.time()
for step in range(MAX_NEW_TOKENS):
    with torch.inference_mode():
        step_input = cur_ids[:, cur_len-1:cur_len].contiguous()
        logits = model(
            input_ids=step_input,
            attention_mask=attn_mask[:, :cur_len],
            use_cache=True,
            past_key_values=kv
        ).logits
        next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_ids], dim=1)
        attn_mask[:, cur_len] = 1
        cur_len += 1

    # Opportunistically promote one more expert every 8 steps if there's headroom
    if (step + 1) % 8 == 0 and per_expert_bytes:
        try_promote_one_more_expert(safety_bytes=safety, per_expert_bytes=per_expert_bytes)

    if (step + 1) % 16 == 0:
        torch.cuda.synchronize()
        print(f"after {step+1:4d} decode tokens: mem_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

torch.cuda.synchronize()
print(f"\n[Decode] {MAX_NEW_TOKENS} tokens in {time.time()-start:.2f}s  |  final_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

# Output
for i in range(B):
    print(f"\n--- OUTPUT {i} ---\n{tokenizer.decode(cur_ids[i], skip_special_tokens=True)}")



import os
# Allocator tuned to reduce fragmentation (keeps freed VRAM reusable)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True")

import time
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import StaticCache
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

# ============================ Config ============================
model_id   = "Qwen/Qwen1.5-MoE-A2.7B"
gpu_device = "cuda:0"
cpu_device = "cpu"
dtype      = torch.float16

# Keep trunk/router/attention on GPU within this budget (experts → CPU)
max_memory = {
    0:   "10GiB",   # slight bump helps keep attention on GPU with KV
    "cpu":"300GiB"
}

# Expert cache knobs (kept & tuned at runtime)
GPU_CACHE_MAX_BYTES   = 2_000_000_000   # ~2.0 GB cap for resident experts
GPU_CACHE_MAX_EXPERTS = 8               # at most 8 experts resident
MAX_NEW_TOKENS        = 128
SAFETY_GB             = 2.0             # VRAM cushion we always leave free

# ==================== Load & dispatch (experts → CPU, attn → GPU) ====================
print(f"Loading config & skeleton for {model_id} …")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

print("Inferring device_map …")
no_split = getattr(empty_model, "_no_split_modules", ["QwenBlock", "Block"])
device_map = infer_auto_device_map(empty_model, max_memory=max_memory, no_split_module_classes=no_split)

# Heuristics: experts → CPU; attention/trunk → GPU (by name)
for name in list(device_map.keys()):
    low = name.lower()
    if ("experts" in low) or ("expert" in low) or ("moe" in low) or ("mixture" in low) or ("sparse" in low) or ("mixtral" in low):
        device_map[name] = cpu_device
    elif ("attn" in low) or ("attention" in low) or ("rotary" in low):
        device_map[name] = gpu_device  # keep attention & KV writers on GPU

print("Downloading checkpoints & dispatching …")
local_ckpt = snapshot_download(model_id, allow_patterns=["*.safetensors","*.bin","config.json","*.json"])
model = load_checkpoint_and_dispatch(
    empty_model,
    checkpoint=local_ckpt,
    device_map=device_map,
    no_split_module_classes=no_split,
    dtype=dtype,
    offload_folder=None,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.eval(); model.config.use_cache = True

# ================= Expert LRU cache (resident-on-GPU modules) =================
def tensor_nbytes(t: torch.Tensor) -> int: return t.numel() * t.element_size()
def module_param_bytes(mod: nn.Module) -> int:
    s = 0
    for p in mod.parameters(recurse=True): s += tensor_nbytes(p.data)
    for b in mod.buffers(recurse=True):   s += tensor_nbytes(b.data)
    return s

def pin_state_dict(cpu_module: nn.Module):
    """
    Create a pinned-memory snapshot for fast H→D copies.
    Guard against 'meta' tensors (no storage to copy).
    """
    pinned = {}
    with torch.no_grad():
        sd = cpu_module.state_dict()
        for k, v in sd.items():
            if getattr(v, "is_meta", False) or v.device.type == "meta":
                raise RuntimeError(
                    f"Expert weight '{k}' is on meta device. "
                    "Ensure device_map placed this expert on CPU before dispatch."
                )
            t = v.detach().contiguous().to("cpu", copy=True)
            try:
                t = t.pin_memory()
            except RuntimeError:
                pass
            pinned[k] = t
    return pinned

class ExpertGPUCache:
    def __init__(self, device:str, dtype:torch.dtype,
                 max_bytes:int=GPU_CACHE_MAX_BYTES, max_count:int=GPU_CACHE_MAX_EXPERTS):
        self.device = torch.device(device); self.dtype = dtype
        self.max_bytes = max_bytes; self.max_count = max_count
        self._cache: OrderedDict[int, nn.Module] = OrderedDict()
        self._sizes: dict[int,int] = {}
        self._pinned_sd: dict[int,dict] = {}
        self._bytes_total = 0
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.main_stream = torch.cuda.current_stream(device=self.device)

    def _ensure_capacity(self, need_bytes:int):
        while (self._bytes_total + need_bytes > self.max_bytes) or (len(self._cache)+1 > self.max_count):
            evict_id, evict_mod = self._cache.popitem(last=False)
            self._bytes_total -= self._sizes.pop(evict_id, 0)
            del evict_mod

    def _build_cuda_module(self, cpu_expert: nn.Module) -> nn.Module:
        cls = cpu_expert.__class__
        if hasattr(cpu_expert, "config"):
            m = cls(config=cpu_expert.config).to(self.device, dtype=self.dtype)
        else:
            import copy as _copy
            m = _copy.deepcopy(cpu_expert).to(self.device, dtype=self.dtype)
        return m

    def _load_weights_async(self, cuda_mod: nn.Module, expert_id:int):
        if expert_id not in self._pinned_sd:
            self._pinned_sd[expert_id] = pin_state_dict(cpu_expert_registry[expert_id])
        sd = self._pinned_sd[expert_id]
        with torch.cuda.stream(self.prefetch_stream):
            for k, v_cpu in sd.items():
                ref = cuda_mod
                comps = k.split(".")
                for c in comps[:-1]: ref = getattr(ref, c)
                leaf = comps[-1]
                if hasattr(ref, leaf):
                    t = getattr(ref, leaf)
                    t.data.copy_(v_cpu, non_blocking=True)
        self.main_stream.wait_stream(self.prefetch_stream)
        torch.cuda.current_stream(self.device).synchronize()

    def get(self, expert_id:int, cpu_expert: nn.Module) -> nn.Module:
        if expert_id in self._cache:
            mod = self._cache.pop(expert_id); self._cache[expert_id] = mod; return mod
        cuda_mod  = self._build_cuda_module(cpu_expert)
        need_bytes = module_param_bytes(cuda_mod)
        self._ensure_capacity(need_bytes)
        self._load_weights_async(cuda_mod, expert_id)
        self._cache[expert_id] = cuda_mod
        self._sizes[expert_id] = need_bytes
        self._bytes_total += need_bytes
        return cuda_mod

    def prefetch_experts(self, ids:list[int]):
        with torch.no_grad():
            for eid in ids:
                if eid in self._cache:
                    mod = self._cache.pop(eid); self._cache[eid] = mod
                else:
                    if eid not in cpu_expert_registry: continue
                    cpu_mod = cpu_expert_registry[eid]
                    cuda_mod = self._build_cuda_module(cpu_mod)
                    need_bytes = module_param_bytes(cuda_mod)
                    self._ensure_capacity(need_bytes)
                    self._load_weights_async(cuda_mod, eid)
                    self._cache[eid] = cuda_mod
                    self._sizes[eid] = need_bytes
                    self._bytes_total += need_bytes

# ---------- Robust expert discovery (name+shape heuristics; no device moves) ----------
def is_mlp_like(module: nn.Module) -> bool:
    # Experts are typically MLPs: at least two Linear layers
    lin = sum(1 for _ in module.modules() if isinstance(_, nn.Linear))
    return lin >= 2

def discover_experts(model: nn.Module):
    """
    Find expert ModuleLists even if not literally named 'experts'.
    NO device moves here to avoid meta→cpu errors. Device placement was
    already handled by accelerate's device_map during dispatch.
    """
    patterns = ("experts", "expert", "moe", "mixture", "sparse", "mixtral")
    registry: dict[int, nn.Module] = {}
    name_map: dict[int, str] = {}
    eid = 0

    for path, mod in model.named_modules():
        if not isinstance(mod, nn.ModuleList):
            continue

        low = path.lower()
        looks_like_expert_list = (
            any(p in low for p in patterns) or
            (len(mod) >= 4 and is_mlp_like(mod[0]))
        )
        if not looks_like_expert_list:
            continue

        base_cls = mod[0].__class__
        if not all(isinstance(m, base_cls) for m in mod):
            continue
        if not is_mlp_like(mod[0]):
            continue

        for idx, expert in enumerate(mod):
            registry[eid] = expert
            name_map[eid] = f"{path}[{idx}]"
            eid += 1

    return registry, name_map

cpu_expert_registry, expert_name_map = discover_experts(model)
print(f"Discovered {len(cpu_expert_registry)} experts (no device moves during discovery).")
for eid_dbg in list(cpu_expert_registry.keys())[:6]:
    print("  ->", eid_dbg, expert_name_map[eid_dbg])

if not cpu_expert_registry:
    print("[WARN] No experts discovered. Candidate ModuleLists:")
    for path, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
            print(f"  candid: {path}  len={len(mod)}  first_cls={mod[0].__class__.__name__}  "
                  f"lin_in_first={sum(1 for _ in mod[0].modules() if isinstance(_, nn.Linear))}")

expert_cache = ExpertGPUCache(device=gpu_device, dtype=dtype,
                              max_bytes=GPU_CACHE_MAX_BYTES, max_count=GPU_CACHE_MAX_EXPERTS)

# ============== Wrap experts: LRU + count usage; RETURN to caller device ==============
expert_call_counts = defaultdict(int)

def make_cached_forward(expert_id:int, cpu_expert: nn.Module):
    def wrapped_forward(*args, **kwargs):
        expert_call_counts[expert_id] += 1  # prefill+decode usage signal

        # Determine caller's device from first tensor (keeps MoE block happy)
        def first_tensor_device():
            for x in args:
                if torch.is_tensor(x): return x.device
            for v in kwargs.values():
                if torch.is_tensor(v): return v.device
            try: return next(cpu_expert.parameters()).device
            except StopIteration: return torch.device("cpu")

        caller_dev = first_tensor_device()
        cuda_dev   = torch.device(gpu_device)

        # Move inputs to CUDA for compute
        def to_cuda(x): return x.to(cuda_dev, non_blocking=True) if torch.is_tensor(x) else x
        args_cuda   = tuple(to_cuda(a) for a in args)
        kwargs_cuda = {k: to_cuda(v) for k, v in kwargs.items()}

        # Run CUDA expert
        cuda_expert = expert_cache.get(expert_id, cpu_expert)
        with torch.no_grad():
            out = cuda_expert(*args_cuda, **kwargs_cuda)

        # Return outputs on the caller's device (prevents device-mismatch)
        def to_caller(y):
            if torch.is_tensor(y): return y.to(caller_dev, non_blocking=True)
            if isinstance(y, (list, tuple)): return type(y)(to_caller(t) for t in y)
            if isinstance(y, dict): return {k: to_caller(v) for k, v in y.items()}
            return y

        return to_caller(out)
    return wrapped_forward

for eid_wrap, cpu_expert in cpu_expert_registry.items():
    cpu_expert.forward = make_cached_forward(eid_wrap, cpu_expert)

# =================== Prepare inputs, single GPU KV, ONE prefill ======================
PROMPTS = ["Explain the concept of dynamic offloading in large language models in one paragraph."]
inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=False)
inputs = {k: v.to(gpu_device) for k, v in inputs.items()}

B = inputs["input_ids"].size(0)
prompt_len = inputs["input_ids"].size(1)
max_new = MAX_NEW_TOKENS

# Single, GPU-resident KV for the whole run
kv = StaticCache(
    max_batch_size=B,
    max_cache_len=prompt_len + max_new,  # reserve up to final length
    config=model.config,
    device=torch.device(gpu_device),
    dtype=model.dtype,
)

def gb(x): return x/1e9
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
base_alloc = torch.cuda.memory_allocated()

# ------------------------- ONE PREFILL (fills KV & counts experts) -------------------
with torch.inference_mode():
    _ = model(**inputs, use_cache=True, past_key_values=kv)

torch.cuda.synchronize()
prefill_peak = torch.cuda.max_memory_allocated()
post_prefill = torch.cuda.memory_allocated()
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"\n[MEM] base={gb(base_alloc):.2f} GB  peak_prefill={gb(prefill_peak):.2f} GB  "
      f"post_prefill={gb(post_prefill):.2f} GB  free={gb(free_bytes):.2f}/{gb(total_bytes):.2f} GB")

# ------------------ Decide which experts to pin (NO 2nd prefill) --------------------
# Rank by prefill usage
sorted_eids = sorted(expert_call_counts.keys(), key=lambda k: expert_call_counts[k], reverse=True)

print("\n[Prefill usage] top experts:")
for i, e in enumerate(sorted_eids[:12]):
    print(f"  {i+1:2d}. EID {e:<4d} calls={expert_call_counts[e]:<6d}  {expert_name_map.get(e,'')}")

# Probe per-expert size once (temporary CUDA instantiation; NOT another prefill)
def probe_expert_bytes(eid_probe:int) -> int:
    cpu_mod = cpu_expert_registry[eid_probe]
    cls = cpu_mod.__class__
    if hasattr(cpu_mod, "config"):
        tmp = cls(config=cpu_mod.config).to(gpu_device, dtype=dtype)
    else:
        import copy as _copy
        tmp = _copy.deepcopy(cpu_mod).to(gpu_device, dtype=dtype)
    nbytes = module_param_bytes(tmp)
    del tmp; torch.cuda.synchronize()
    return nbytes

if sorted_eids:
    per_expert_bytes = probe_expert_bytes(sorted_eids[0])
else:
    per_expert_bytes = 50_000_000  # 50MB fallback if no experts found

safety = int(SAFETY_GB * 1e9)
max_by_bytes = max(0, (free_bytes - safety) // max(per_expert_bytes, 1))
target_keep = int(min(GPU_CACHE_MAX_EXPERTS, max_by_bytes))
target_keep = max(0, target_keep)

# Tighten caps to the plan (keep both knobs)
expert_cache.max_count = target_keep
expert_cache.max_bytes = int(min(GPU_CACHE_MAX_BYTES, per_expert_bytes * target_keep))

# Prefetch the chosen top-K NOW (still before any decode tokens)
top_keep_ids = sorted_eids[:target_keep]
expert_cache.prefetch_experts(top_keep_ids)
print(f"\n[Plan] Keep top-{target_keep} experts on GPU (~{gb(per_expert_bytes):.2f} GB each); "
      f"cache caps → bytes={gb(expert_cache.max_bytes):.2f} GB, count={expert_cache.max_count}")
print(f"[Plan] Prefetched EIDs: {top_keep_ids}")

# ===================== Dynamic promoter (grow during decode) =========================
prefill_rank = sorted_eids[:]                  # hottest → coldest
_promote_cursor = len(top_keep_ids)            # next index to promote

def cuda_free_bytes():
    fb, _ = torch.cuda.mem_get_info()
    return int(fb)

def try_promote_one_more_expert(safety_bytes:int, per_expert_bytes:int):
    global _promote_cursor
    if _promote_cursor >= len(prefill_rank):
        return False  # nothing left to promote

    free_now = cuda_free_bytes()
    need = per_expert_bytes + max(2 << 20, per_expert_bytes // 8)  # cushion
    if free_now <= safety_bytes + need:
        return False

    # Respect knobs; allow gentle growth if obvious headroom exists
    if len(expert_cache._cache) >= expert_cache.max_count:
        if expert_cache._bytes_total + per_expert_bytes <= expert_cache.max_bytes:
            expert_cache.max_count += 1
        elif expert_cache._bytes_total + per_expert_bytes <= expert_cache.max_bytes + (per_expert_bytes // 2):
            expert_cache.max_bytes += per_expert_bytes
        else:
            return False

    if (len(expert_cache._cache) + 1 > expert_cache.max_count) or \
       (expert_cache._bytes_total + per_expert_bytes > expert_cache.max_bytes):
        return False

    eid_promote = prefill_rank[_promote_cursor]
    _promote_cursor += 1
    expert_cache.prefetch_experts([eid_promote])
    print(f"[Promote] Pinned EID {eid_promote} to GPU (free~{cuda_free_bytes()/1e9:.2f} GB)")
    return True

# ================================ Decode (reuse same KV) =============================
PROMPTS = PROMPTS  # keep previous prompts
B = inputs["input_ids"].size(0)
cur_ids = inputs["input_ids"]
attn_mask = torch.zeros((B, prompt_len + max_new), dtype=torch.long, device=gpu_device)
attn_mask[:, :prompt_len] = 1
cur_len = prompt_len

start = time.time()
for step in range(MAX_NEW_TOKENS):
    with torch.inference_mode():
        step_input = cur_ids[:, cur_len-1:cur_len].contiguous()
        logits = model(
            input_ids=step_input,
            attention_mask=attn_mask[:, :cur_len],
            use_cache=True,
            past_key_values=kv
        ).logits
        next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_ids], dim=1)
        attn_mask[:, cur_len] = 1
        cur_len += 1

    # Opportunistically promote one more expert every 8 steps if there's headroom
    if (step + 1) % 8 == 0 and per_expert_bytes:
        try_promote_one_more_expert(safety_bytes=safety, per_expert_bytes=per_expert_bytes)

    if (step + 1) % 16 == 0:
        torch.cuda.synchronize()
        print(f"after {step+1:4d} decode tokens: mem_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

torch.cuda.synchronize()
print(f"\n[Decode] {MAX_NEW_TOKENS} tokens in {time.time()-start:.2f}s  |  final_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

# Output
for i in range(B):
    print(f"\n--- OUTPUT {i} ---\n{tokenizer.decode(cur_ids[i], skip_special_tokens=True)}")








import os
# Keep Flash attention; allocator tuned to reduce fragmentation & reuse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True")

import time
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import StaticCache
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

# ============================ Config ============================
model_id   = "Qwen/Qwen1.5-MoE-A2.7B"
gpu_device = "cuda:0"
cpu_device = "cpu"
dtype      = torch.float16

# Keep trunk/router/attention on GPU within this budget (experts -> CPU)
max_memory = {
    0:   "10GiB",   # small bump helps ensure attention stays on GPU with KV
    "cpu":"300GiB"
}

# Expert cache knobs (kept & tuned at runtime)
GPU_CACHE_MAX_BYTES   = 2_000_000_000   # ~2.0 GB cap for resident experts
GPU_CACHE_MAX_EXPERTS = 8               # at most 8 experts resident
MAX_NEW_TOKENS        = 128
SAFETY_GB             = 2.0             # VRAM buffer we keep free

# ==================== Load & dispatch (experts → CPU, attn → GPU) ====================
print(f"Loading config & skeleton for {model_id} …")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

print("Inferring device_map …")
no_split = getattr(empty_model, "_no_split_modules", ["QwenBlock", "Block"])
device_map = infer_auto_device_map(empty_model, max_memory=max_memory, no_split_module_classes=no_split)

# Heuristics: experts → CPU; attention/trunk → GPU (by name)
for name in list(device_map.keys()):
    low = name.lower()
    if ("experts" in low) or ("expert" in low) or ("moe" in low) or ("mixture" in low) or ("sparse" in low) or ("mixtral" in low):
        device_map[name] = cpu_device
    elif ("attn" in low) or ("attention" in low) or ("rotary" in low):
        device_map[name] = gpu_device  # keep attention & KV writers on GPU

print("Downloading checkpoints & dispatching …")
local_ckpt = snapshot_download(model_id, allow_patterns=["*.safetensors","*.bin","config.json","*.json"])
model = load_checkpoint_and_dispatch(
    empty_model,
    checkpoint=local_ckpt,
    device_map=device_map,
    no_split_module_classes=no_split,
    dtype=dtype,
    offload_folder=None,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.eval(); model.config.use_cache = True

# ================= Expert LRU cache (resident-on-GPU modules) =================
def tensor_nbytes(t: torch.Tensor) -> int: return t.numel() * t.element_size()
def module_param_bytes(mod: nn.Module) -> int:
    s = 0
    for p in mod.parameters(recurse=True): s += tensor_nbytes(p.data)
    for b in mod.buffers(recurse=True):   s += tensor_nbytes(b.data)
    return s

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
    def __init__(self, device:str, dtype:torch.dtype,
                 max_bytes:int=GPU_CACHE_MAX_BYTES, max_count:int=GPU_CACHE_MAX_EXPERTS):
        self.device = torch.device(device); self.dtype = dtype
        self.max_bytes = max_bytes; self.max_count = max_count
        self._cache: OrderedDict[int, nn.Module] = OrderedDict()
        self._sizes: dict[int,int] = {}
        self._pinned_sd: dict[int,dict] = {}
        self._bytes_total = 0
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.main_stream = torch.cuda.current_stream(device=self.device)

    def _ensure_capacity(self, need_bytes:int):
        while (self._bytes_total + need_bytes > self.max_bytes) or (len(self._cache)+1 > self.max_count):
            evict_id, evict_mod = self._cache.popitem(last=False)
            self._bytes_total -= self._sizes.pop(evict_id, 0)
            del evict_mod

    def _build_cuda_module(self, cpu_expert: nn.Module) -> nn.Module:
        cls = cpu_expert.__class__
        if hasattr(cpu_expert, "config"):
            m = cls(config=cpu_expert.config).to(self.device, dtype=self.dtype)
        else:
            import copy as _copy
            m = _copy.deepcopy(cpu_expert).to(self.device, dtype=self.dtype)
        return m

    def _load_weights_async(self, cuda_mod: nn.Module, expert_id:int):
        if expert_id not in self._pinned_sd:
            self._pinned_sd[expert_id] = pin_state_dict(cpu_expert_registry[expert_id])
        sd = self._pinned_sd[expert_id]
        with torch.cuda.stream(self.prefetch_stream):
            for k, v_cpu in sd.items():
                ref = cuda_mod
                comps = k.split(".")
                for c in comps[:-1]: ref = getattr(ref, c)
                leaf = comps[-1]
                if hasattr(ref, leaf):
                    t = getattr(ref, leaf)
                    t.data.copy_(v_cpu, non_blocking=True)
        self.main_stream.wait_stream(self.prefetch_stream)
        torch.cuda.current_stream(self.device).synchronize()

    def get(self, expert_id:int, cpu_expert: nn.Module) -> nn.Module:
        if expert_id in self._cache:
            mod = self._cache.pop(expert_id); self._cache[expert_id] = mod; return mod
        cuda_mod  = self._build_cuda_module(cpu_expert)
        need_bytes = module_param_bytes(cuda_mod)
        self._ensure_capacity(need_bytes)
        self._load_weights_async(cuda_mod, expert_id)
        self._cache[expert_id] = cuda_mod
        self._sizes[expert_id] = need_bytes
        self._bytes_total += need_bytes
        return cuda_mod

    def prefetch_experts(self, ids:list[int]):
        with torch.no_grad():
            for eid in ids:
                if eid in self._cache:
                    mod = self._cache.pop(eid); self._cache[eid] = mod
                else:
                    if eid not in cpu_expert_registry: continue
                    cpu_mod = cpu_expert_registry[eid]
                    cuda_mod = self._build_cuda_module(cpu_mod)
                    need_bytes = module_param_bytes(cuda_mod)
                    self._ensure_capacity(need_bytes)
                    self._load_weights_async(cuda_mod, eid)
                    self._cache[eid] = cuda_mod
                    self._sizes[eid] = need_bytes
                    self._bytes_total += need_bytes

# ---------- Robust expert discovery (structure-based) ----------
def is_mlp_like(module: nn.Module) -> bool:
    lin = sum(1 for _ in module.modules() if isinstance(_, nn.Linear))
    return lin >= 2

def discover_and_force_cpu_experts(model: nn.Module, force_cpu: bool = True):
    patterns = ("experts", "expert", "moe", "mixture", "sparse", "mixtral")
    registry: dict[int, nn.Module] = {}
    name_map: dict[int, str] = {}
    eid = 0

    for path, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList):
            low = path.lower()
            looks_like_expert_list = (
                any(p in low for p in patterns) or
                (len(mod) >= 4 and is_mlp_like(mod[0]))
            )
            if not looks_like_expert_list:
                continue

            base_cls = mod[0].__class__
            homogeneous = all(isinstance(m, base_cls) for m in mod)
            if not homogeneous:
                continue

            if not is_mlp_like(mod[0]):
                continue

            for idx, expert in enumerate(mod):
                if force_cpu:
                    try:
                        dev = next(expert.parameters()).device
                        if dev.type != "cpu":
                            expert.to(cpu_device, dtype=dtype)
                    except StopIteration:
                        pass
                registry[eid] = expert
                name_map[eid] = f"{path}[{idx}]"
                eid += 1

    return registry, name_map

cpu_expert_registry, expert_name_map = discover_and_force_cpu_experts(model, force_cpu=True)
print(f"Discovered {len(cpu_expert_registry)} experts (forced to CPU).")
for eid_dbg in list(cpu_expert_registry.keys())[:6]:
    print("  ->", eid_dbg, expert_name_map[eid_dbg])

# If STILL empty, dump hints to refine the pattern
if not cpu_expert_registry:
    print("[WARN] No experts discovered. Candidate ModuleLists:")
    for path, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
            print(f"  candid: {path}  len={len(mod)}  first_cls={mod[0].__class__.__name__}  lin_in_first={sum(1 for _ in mod[0].modules() if isinstance(_, nn.Linear))}")

expert_cache = ExpertGPUCache(device=gpu_device, dtype=dtype,
                              max_bytes=GPU_CACHE_MAX_BYTES, max_count=GPU_CACHE_MAX_EXPERTS)

# ============== Wrap experts: LRU + count usage; RETURN to caller device ==============
expert_call_counts = defaultdict(int)

def make_cached_forward(expert_id:int, cpu_expert: nn.Module):
    def wrapped_forward(*args, **kwargs):
        expert_call_counts[expert_id] += 1  # prefill+decode usage signal

        # Determine caller's device from first tensor (keeps MoE block happy)
        def first_tensor_device():
            for x in args:
                if torch.is_tensor(x): return x.device
            for v in kwargs.values():
                if torch.is_tensor(v): return v.device
            try: return next(cpu_expert.parameters()).device
            except StopIteration: return torch.device("cpu")

        caller_dev = first_tensor_device()
        cuda_dev   = torch.device(gpu_device)

        # Move inputs to CUDA for compute
        def to_cuda(x): return x.to(cuda_dev, non_blocking=True) if torch.is_tensor(x) else x
        args_cuda   = tuple(to_cuda(a) for a in args)
        kwargs_cuda = {k: to_cuda(v) for k, v in kwargs.items()}

        # Run CUDA expert
        cuda_expert = expert_cache.get(expert_id, cpu_expert)
        with torch.no_grad():
            out = cuda_expert(*args_cuda, **kwargs_cuda)

        # Return outputs on the caller's device (prevents device-mismatch)
        def to_caller(y):
            if torch.is_tensor(y): return y.to(caller_dev, non_blocking=True)
            if isinstance(y, (list, tuple)): return type(y)(to_caller(t) for t in y)
            if isinstance(y, dict): return {k: to_caller(v) for k, v in y.items()}
            return y

        return to_caller(out)
    return wrapped_forward

for eid_wrap, cpu_expert in cpu_expert_registry.items():
    cpu_expert.forward = make_cached_forward(eid_wrap, cpu_expert)

# =================== Prepare inputs, single GPU KV, ONE prefill ======================
PROMPTS = ["Explain the concept of dynamic offloading in large language models in one paragraph."]
inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=False)
inputs = {k: v.to(gpu_device) for k, v in inputs.items()}

B = inputs["input_ids"].size(0)
prompt_len = inputs["input_ids"].size(1)
max_new = MAX_NEW_TOKENS

# Single, GPU-resident KV for the whole run
kv = StaticCache(
    max_batch_size=B,
    max_cache_len=prompt_len + max_new,  # reserve up to final length
    config=model.config,
    device=torch.device(gpu_device),
    dtype=model.dtype,
)

def gb(x): return x/1e9
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
base_alloc = torch.cuda.memory_allocated()

# ------------------------- ONE PREFILL (fills KV & counts experts) -------------------
with torch.inference_mode():
    _ = model(**inputs, use_cache=True, past_key_values=kv)

torch.cuda.synchronize()
prefill_peak = torch.cuda.max_memory_allocated()
post_prefill = torch.cuda.memory_allocated()
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"\n[MEM] base={gb(base_alloc):.2f} GB  peak_prefill={gb(prefill_peak):.2f} GB  "
      f"post_prefill={gb(post_prefill):.2f} GB  free={gb(free_bytes):.2f}/{gb(total_bytes):.2f} GB")

# ------------------ Decide which experts to pin (NO 2nd prefill) --------------------
# Rank by prefill usage
sorted_eids = sorted(expert_call_counts.keys(), key=lambda k: expert_call_counts[k], reverse=True)

print("\n[Prefill usage] top experts:")
for i, e in enumerate(sorted_eids[:12]):
    print(f"  {i+1:2d}. EID {e:<4d} calls={expert_call_counts[e]:<6d}  {expert_name_map.get(e,'')}")

# Probe per-expert size once (temporary CUDA instantiation; NOT another prefill)
def probe_expert_bytes(eid_probe:int) -> int:
    cpu_mod = cpu_expert_registry[eid_probe]
    cls = cpu_mod.__class__
    if hasattr(cpu_mod, "config"):
        tmp = cls(config=cpu_mod.config).to(gpu_device, dtype=dtype)
    else:
        import copy as _copy
        tmp = _copy.deepcopy(cpu_mod).to(gpu_device, dtype=dtype)
    nbytes = module_param_bytes(tmp)
    del tmp; torch.cuda.synchronize()
    return nbytes

if sorted_eids:
    per_expert_bytes = probe_expert_bytes(sorted_eids[0])
else:
    per_expert_bytes = 50_000_000  # 50MB fallback if no experts found

safety = int(SAFETY_GB * 1e9)
max_by_bytes = max(0, (free_bytes - safety) // max(per_expert_bytes, 1))
target_keep = int(min(GPU_CACHE_MAX_EXPERTS, max_by_bytes))
target_keep = max(0, target_keep)

# Tighten caps to the plan (keep both knobs)
expert_cache.max_count = target_keep
expert_cache.max_bytes = int(min(GPU_CACHE_MAX_BYTES, per_expert_bytes * target_keep))

# Prefetch the chosen top-K NOW (still before any decode tokens)
top_keep_ids = sorted_eids[:target_keep]
expert_cache.prefetch_experts(top_keep_ids)
print(f"\n[Plan] Keep top-{target_keep} experts on GPU (~{gb(per_expert_bytes):.2f} GB each); "
      f"cache caps → bytes={gb(expert_cache.max_bytes):.2f} GB, count={expert_cache.max_count}")
print(f"[Plan] Prefetched EIDs: {top_keep_ids}")

# ===================== Dynamic promoter (grow during decode) =========================
prefill_rank = sorted_eids[:]                  # hottest → coldest
_promote_cursor = len(top_keep_ids)            # next index to promote

def cuda_free_bytes():
    fb, _ = torch.cuda.mem_get_info()
    return int(fb)

def try_promote_one_more_expert(safety_bytes:int, per_expert_bytes:int):
    global _promote_cursor
    if _promote_cursor >= len(prefill_rank):
        return False  # nothing left to promote

    free_now = cuda_free_bytes()
    need = per_expert_bytes + max(2 << 20, per_expert_bytes // 8)  # cushion
    if free_now <= safety_bytes + need:
        return False

    # Respect knobs; allow gentle growth if obvious headroom exists
    if len(expert_cache._cache) >= expert_cache.max_count:
        if expert_cache._bytes_total + per_expert_bytes <= expert_cache.max_bytes:
            expert_cache.max_count += 1
        elif expert_cache._bytes_total + per_expert_bytes <= expert_cache.max_bytes + (per_expert_bytes // 2):
            expert_cache.max_bytes += per_expert_bytes
        else:
            return False

    if (len(expert_cache._cache) + 1 > expert_cache.max_count) or \
       (expert_cache._bytes_total + per_expert_bytes > expert_cache.max_bytes):
        return False

    eid_promote = prefill_rank[_promote_cursor]
    _promote_cursor += 1
    expert_cache.prefetch_experts([eid_promote])
    print(f"[Promote] Pinned EID {eid_promote} to GPU (free~{cuda_free_bytes()/1e9:.2f} GB)")
    return True

# ================================ Decode (reuse same KV) =============================
PROMPTS = PROMPTS  # keep previous prompts
B = inputs["input_ids"].size(0)
cur_ids = inputs["input_ids"]
attn_mask = torch.zeros((B, prompt_len + max_new), dtype=torch.long, device=gpu_device)
attn_mask[:, :prompt_len] = 1
cur_len = prompt_len

start = time.time()
for step in range(MAX_NEW_TOKENS):
    with torch.inference_mode():
        step_input = cur_ids[:, cur_len-1:cur_len].contiguous()
        logits = model(
            input_ids=step_input,
            attention_mask=attn_mask[:, :cur_len],
            use_cache=True,
            past_key_values=kv
        ).logits
        next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_ids], dim=1)
        attn_mask[:, cur_len] = 1
        cur_len += 1

    # Opportunistically promote one more expert every 8 steps if there's headroom
    if (step + 1) % 8 == 0 and per_expert_bytes:
        try_promote_one_more_expert(safety_bytes=safety, per_expert_bytes=per_expert_bytes)

    if (step + 1) % 16 == 0:
        torch.cuda.synchronize()
        print(f"after {step+1:4d} decode tokens: mem_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

torch.cuda.synchronize()
print(f"\n[Decode] {MAX_NEW_TOKENS} tokens in {time.time()-start:.2f}s  |  final_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

# Output
for i in range(B):
    print(f"\n--- OUTPUT {i} ---\n{tokenizer.decode(cur_ids[i], skip_special_tokens=True)}")





import os
# allocator tuned to reduce fragmentation (keeps freed VRAM usable)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True")

import time
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import StaticCache
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

# ============================ Config ============================
model_id   = "Qwen/Qwen1.5-MoE-A2.7B"
gpu_device = "cuda:0"
cpu_device = "cpu"
dtype      = torch.float16

# Keep trunk/router/attention on GPU within this budget (experts -> CPU)
max_memory = {
    0:   "10GiB",   # slight bump helps keep attention on GPU with KV
    "cpu":"300GiB"
}

# Expert cache knobs (kept & tuned at runtime)
GPU_CACHE_MAX_BYTES   = 2_000_000_000   # ~2.0 GB cap for resident experts
GPU_CACHE_MAX_EXPERTS = 8               # at most 8 experts resident
MAX_NEW_TOKENS        = 128
SAFETY_GB             = 2.0             # headroom before promoting more experts

# ==================== Load & dispatch (experts → CPU, attn → GPU) ====================
print(f"Loading config & skeleton for {model_id} …")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

print("Inferring device_map …")
no_split = getattr(empty_model, "_no_split_modules", ["QwenBlock", "Block"])
device_map = infer_auto_device_map(empty_model, max_memory=max_memory, no_split_module_classes=no_split)

# Force attention/trunk on GPU, experts on CPU (by name heuristics)
for name in list(device_map.keys()):
    low = name.lower()
    if ("experts" in low):
        device_map[name] = cpu_device
    elif ("attn" in low) or ("attention" in low) or ("rotary" in low):
        device_map[name] = gpu_device  # keep attention & KV writers on GPU

print("Downloading checkpoints & dispatching …")
local_ckpt = snapshot_download(model_id, allow_patterns=["*.safetensors","*.bin","config.json","*.json"])
model = load_checkpoint_and_dispatch(
    empty_model,
    checkpoint=local_ckpt,
    device_map=device_map,
    no_split_module_classes=no_split,
    dtype=dtype,
    offload_folder=None,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.eval(); model.config.use_cache = True

# ================= Expert LRU cache (resident-on-GPU modules) =================
def tensor_nbytes(t: torch.Tensor) -> int: return t.numel() * t.element_size()
def module_param_bytes(mod: nn.Module) -> int:
    s = 0
    for p in mod.parameters(recurse=True): s += tensor_nbytes(p.data)
    for b in mod.buffers(recurse=True):   s += tensor_nbytes(b.data)
    return s

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
    def __init__(self, device:str, dtype:torch.dtype,
                 max_bytes:int=GPU_CACHE_MAX_BYTES, max_count:int=GPU_CACHE_MAX_EXPERTS):
        self.device = torch.device(device); self.dtype = dtype
        self.max_bytes = max_bytes; self.max_count = max_count
        self._cache: OrderedDict[int, nn.Module] = OrderedDict()
        self._sizes: dict[int,int] = {}
        self._pinned_sd: dict[int,dict] = {}
        self._bytes_total = 0
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.main_stream = torch.cuda.current_stream(device=self.device)

    def _ensure_capacity(self, need_bytes:int):
        while (self._bytes_total + need_bytes > self.max_bytes) or (len(self._cache)+1 > self.max_count):
            evict_id, evict_mod = self._cache.popitem(last=False)
            self._bytes_total -= self._sizes.pop(evict_id, 0)
            del evict_mod

    def _build_cuda_module(self, cpu_expert: nn.Module) -> nn.Module:
        cls = cpu_expert.__class__
        if hasattr(cpu_expert, "config"):
            m = cls(config=cpu_expert.config).to(self.device, dtype=self.dtype)
        else:
            import copy as _copy
            m = _copy.deepcopy(cpu_expert).to(self.device, dtype=self.dtype)
        return m

    def _load_weights_async(self, cuda_mod: nn.Module, expert_id:int):
        if expert_id not in self._pinned_sd:
            self._pinned_sd[expert_id] = pin_state_dict(cpu_expert_registry[expert_id])
        sd = self._pinned_sd[expert_id]
        with torch.cuda.stream(self.prefetch_stream):
            for k, v_cpu in sd.items():
                ref = cuda_mod
                comps = k.split(".")
                for c in comps[:-1]: ref = getattr(ref, c)
                leaf = comps[-1]
                if hasattr(ref, leaf):
                    t = getattr(ref, leaf)
                    t.data.copy_(v_cpu, non_blocking=True)
        self.main_stream.wait_stream(self.prefetch_stream)
        torch.cuda.current_stream(self.device).synchronize()

    def get(self, expert_id:int, cpu_expert: nn.Module) -> nn.Module:
        if expert_id in self._cache:
            mod = self._cache.pop(expert_id); self._cache[expert_id] = mod; return mod
        cuda_mod  = self._build_cuda_module(cpu_expert)
        need_bytes = module_param_bytes(cuda_mod)
        self._ensure_capacity(need_bytes)
        self._load_weights_async(cuda_mod, expert_id)
        self._cache[expert_id] = cuda_mod
        self._sizes[expert_id] = need_bytes
        self._bytes_total += need_bytes
        return cuda_mod

    def prefetch_experts(self, ids:list[int]):
        with torch.no_grad():
            for eid in ids:
                if eid in self._cache:
                    mod = self._cache.pop(eid); self._cache[eid] = mod
                else:
                    if eid not in cpu_expert_registry: continue
                    cpu_mod = cpu_expert_registry[eid]
                    cuda_mod = self._build_cuda_module(cpu_mod)
                    need_bytes = module_param_bytes(cuda_mod)
                    self._ensure_capacity(need_bytes)
                    self._load_weights_async(cuda_mod, eid)
                    self._cache[eid] = cuda_mod
                    self._sizes[eid] = need_bytes
                    self._bytes_total += need_bytes

# Build registry of CPU experts
cpu_expert_registry: dict[int, nn.Module] = {}
expert_name_map: dict[int,str] = {}
eid = 0
for name, module in model.named_modules():
    if isinstance(module, nn.ModuleList) and "experts" in name:
        for idx, expert in enumerate(module):
            try: dev = next(expert.parameters()).device.type
            except StopIteration: dev = cpu_device
            if dev == cpu_device:
                cpu_expert_registry[eid] = expert
                expert_name_map[eid] = f"{name}[{idx}]"
                eid += 1
print(f"CPU-resident experts discovered: {len(cpu_expert_registry)}")

expert_cache = ExpertGPUCache(device=gpu_device, dtype=dtype,
                              max_bytes=GPU_CACHE_MAX_BYTES, max_count=GPU_CACHE_MAX_EXPERTS)

# ============== Wrap experts: route via LRU & count usage; RETURN to caller device ==============
expert_call_counts = defaultdict(int)

def make_cached_forward(expert_id:int, cpu_expert: nn.Module):
    def wrapped_forward(*args, **kwargs):
        expert_call_counts[expert_id] += 1  # prefill+decode usage signal

        # Determine caller's device from first tensor (keeps block happy)
        def first_tensor_device():
            for x in args:
                if torch.is_tensor(x): return x.device
            for v in kwargs.values():
                if torch.is_tensor(v): return v.device
            try: return next(cpu_expert.parameters()).device
            except StopIteration: return torch.device("cpu")

        caller_dev = first_tensor_device()
        cuda_dev   = torch.device(gpu_device)

        # Move inputs to CUDA for compute
        def to_cuda(x): return x.to(cuda_dev, non_blocking=True) if torch.is_tensor(x) else x
        args_cuda   = tuple(to_cuda(a) for a in args)
        kwargs_cuda = {k: to_cuda(v) for k, v in kwargs.items()}

        # Run CUDA expert
        cuda_expert = expert_cache.get(expert_id, cpu_expert)
        with torch.no_grad():
            out = cuda_expert(*args_cuda, **kwargs_cuda)

        # Return outputs on the caller's device (prevents device-mismatch)
        def to_caller(y):
            if torch.is_tensor(y): return y.to(caller_dev, non_blocking=True)
            if isinstance(y, (list, tuple)): return type(y)(to_caller(t) for t in y)
            if isinstance(y, dict): return {k: to_caller(v) for k, v in y.items()}
            return y

        return to_caller(out)
    return wrapped_forward

for eid_, cpu_expert in cpu_expert_registry.items():
    cpu_expert.forward = make_cached_forward(eid_, cpu_expert)

# =================== Prepare inputs, single GPU KV, ONE prefill ======================
PROMPTS = ["Explain the concept of dynamic offloading in large language models in one paragraph."]
inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=False)
inputs = {k: v.to(gpu_device) for k, v in inputs.items()}

B = inputs["input_ids"].size(0)
prompt_len = inputs["input_ids"].size(1)
max_new = MAX_NEW_TOKENS

# Single, GPU-resident KV for the whole run
kv = StaticCache(
    max_batch_size=B,
    max_cache_len=prompt_len + max_new,  # reserve up to final length
    config=model.config,
    device=torch.device(gpu_device),
    dtype=model.dtype,
)

def gb(x): return x/1e9
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
base_alloc = torch.cuda.memory_allocated()

# ------------------------- ONE PREFILL (fills KV & counts experts) -------------------
with torch.inference_mode():
    _ = model(**inputs, use_cache=True, past_key_values=kv)

torch.cuda.synchronize()
prefill_peak = torch.cuda.max_memory_allocated()
post_prefill = torch.cuda.memory_allocated()
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"\n[MEM] base={gb(base_alloc):.2f} GB  peak_prefill={gb(prefill_peak):.2f} GB  "
      f"post_prefill={gb(post_prefill):.2f} GB  free={gb(free_bytes):.2f}/{gb(total_bytes):.2f} GB")

# ------------------ Decide which experts to pin (NO 2nd prefill) --------------------
# Rank by prefill usage
sorted_eids = sorted(expert_call_counts.keys(), key=lambda k: expert_call_counts[k], reverse=True)
print("\n[Prefill usage] top experts:")
for i, e in enumerate(sorted_eids[:12]):
    print(f"  {i+1:2d}. EID {e:<4d} calls={expert_call_counts[e]:<6d}  {expert_name_map.get(e,'')}")

# Probe per-expert size once (temporary CUDA instantiation; NOT another prefill)
def probe_expert_bytes(eid_probe:int) -> int:
    cpu_mod = cpu_expert_registry[eid_probe]
    cls = cpu_mod.__class__
    if hasattr(cpu_mod, "config"):
        tmp = cls(config=cpu_mod.config).to(gpu_device, dtype=dtype)
    else:
        import copy as _copy
        tmp = _copy.deepcopy(cpu_mod).to(gpu_device, dtype=dtype)
    nbytes = module_param_bytes(tmp)
    del tmp; torch.cuda.synchronize()
    return nbytes

per_expert_bytes = probe_expert_bytes(sorted_eids[0]) if sorted_eids else 50_000_000
safety = int(SAFETY_GB * 1e9)
max_by_bytes = max(0, (free_bytes - safety) // max(per_expert_bytes, 1))
target_keep = int(min(GPU_CACHE_MAX_EXPERTS, max_by_bytes))
target_keep = max(0, target_keep)

# Tighten caps to the plan (keep both knobs)
expert_cache.max_count = target_keep
expert_cache.max_bytes = int(min(GPU_CACHE_MAX_BYTES, per_expert_bytes * target_keep))

# Prefetch the chosen top-K NOW (still before any decode tokens)
top_keep_ids = sorted_eids[:target_keep]
expert_cache.prefetch_experts(top_keep_ids)
print(f"\n[Plan] Keep top-{target_keep} experts on GPU (~{gb(per_expert_bytes):.2f} GB each); "
      f"cache caps → bytes={gb(expert_cache.max_bytes):.2f} GB, count={expert_cache.max_count}")
print(f"[Plan] Prefetched EIDs: {top_keep_ids}")

# ===================== Dynamic promoter (grow during decode) =========================
prefill_rank = sorted_eids[:]                  # hottest → coldest
_promote_cursor = len(top_keep_ids)            # next index to promote

def cuda_free_bytes():
    fb, _ = torch.cuda.mem_get_info()
    return int(fb)

def try_promote_one_more_expert(safety_bytes:int, per_expert_bytes:int):
    global _promote_cursor
    if _promote_cursor >= len(prefill_rank):
        return False  # nothing left to promote

    free_now = cuda_free_bytes()
    # need room for one more expert plus a small cushion
    need = per_expert_bytes + max(2 << 20, per_expert_bytes // 8)  # +2MB or +12.5%
    if free_now <= safety_bytes + need:
        return False

    # Respect knobs; allow gentle growth if obvious headroom exists
    if len(expert_cache._cache) >= expert_cache.max_count:
        if expert_cache._bytes_total + per_expert_bytes <= expert_cache.max_bytes:
            expert_cache.max_count += 1
        elif expert_cache._bytes_total + per_expert_bytes <= expert_cache.max_bytes + (per_expert_bytes // 2):
            expert_cache.max_bytes += per_expert_bytes  # relax bytes cap minimally
        else:
            return False  # cannot grow within policy

    # Final guard
    if (len(expert_cache._cache) + 1 > expert_cache.max_count) or \
       (expert_cache._bytes_total + per_expert_bytes > expert_cache.max_bytes):
        return False

    eid = prefill_rank[_promote_cursor]
    _promote_cursor += 1
    expert_cache.prefetch_experts([eid])
    print(f"[Promote] Pinned EID {eid} to GPU (free~{cuda_free_bytes()/1e9:.2f} GB)")
    return True

# ================================ Decode (reuse same KV) =============================
cur_ids = inputs["input_ids"]
attn_mask = torch.zeros((B, prompt_len + max_new), dtype=torch.long, device=gpu_device)
attn_mask[:, :prompt_len] = 1
cur_len = prompt_len

start = time.time()
for step in range(MAX_NEW_TOKENS):
    with torch.inference_mode():
        step_input = cur_ids[:, cur_len-1:cur_len].contiguous()
        logits = model(
            input_ids=step_input,
            attention_mask=attn_mask[:, :cur_len],
            use_cache=True,
            past_key_values=kv
        ).logits
        next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_ids], dim=1)
        attn_mask[:, cur_len] = 1
        cur_len += 1

    # Opportunistically promote one more expert every 8 steps if there's headroom
    if (step + 1) % 8 == 0 and per_expert_bytes:
        try_promote_one_more_expert(safety_bytes=safety, per_expert_bytes=per_expert_bytes)

    if (step + 1) % 16 == 0:
        torch.cuda.synchronize()
        print(f"after {step+1:4d} decode tokens: mem_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

torch.cuda.synchronize()
print(f"\n[Decode] {MAX_NEW_TOKENS} tokens in {time.time()-start:.2f}s  |  final_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

# Output
for i in range(B):
    print(f"\n--- OUTPUT {i} ---\n{tokenizer.decode(cur_ids[i], skip_special_tokens=True)}")
