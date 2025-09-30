import os
# Keep Flash; allocator tuned to reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True")

import time
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import StaticCache
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

# ============================ Config ============================
model_id   = "Qwen/Qwen1.5-MoE-A2.7B"
gpu_device = "cuda:0"
cpu_device = "cpu"
dtype      = torch.float16

max_memory = {0: "8GiB", "cpu": "300GiB"}  # keep attention+trunk under ~8GiB
GPU_CACHE_MAX_BYTES   = 2_000_000_000      # soft cap (can grow during decode)
GPU_CACHE_MAX_EXPERTS = 8                  # soft cap (can grow during decode)
MAX_NEW_TOKENS        = 128
SAFETY_GB             = 2.0

# ---- Prefill selection knobs ----
EXPERT_SELECT_MODE  = "usage_share"  # "usage_share" or "count"
EXPERT_TOP_RATIO    = 0.50           # 50% of routed usage (or 50% by count)
EXPERT_VRAM_RATIO   = 0.50           # use up to 50% of headroom for experts

# ---- Decode-time promotion knobs ----
FREEPOOL_THRESHOLD_GB = 1.5          # (reserved-allocated) ≥ threshold → promote
PROMOTE_EVERY_STEPS   = 4            # check every N steps
PROMOTE_BATCH_SIZE    = 3            # move up to K experts per check

# ==================== Load & dispatch (experts→CPU) ====================
print(f"Loading config for {model_id} …")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# Build a normal (non-meta) model skeleton to avoid "config[...] assert failed"
empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

print("Inferring device_map …")
no_split = getattr(empty_model, "_no_split_modules", ["QwenBlock", "Block"])
device_map = infer_auto_device_map(
    empty_model,
    max_memory=max_memory,
    no_split_module_classes=no_split
)

# Force experts to CPU (materialized there)
for name in list(device_map.keys()):
    if "experts" in name:
        device_map[name] = cpu_device

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

# ================= Expert LRU cache (yours, kept) ===================
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

# Registry (experts on CPU)
cpu_expert_registry: dict[int, nn.Module] = {}
expert_name_map: dict[int,str] = {}
eid = 0
for name, module in model.named_modules():
    if isinstance(module, nn.ModuleList) and "experts" in name:
        for idx, expert in enumerate(module):
            cpu_expert_registry[eid] = expert
            expert_name_map[eid] = f"{name}[{idx}]"
            eid += 1
print(f"CPU-resident experts discovered: {len(cpu_expert_registry)}")

expert_cache = ExpertGPUCache(device=gpu_device, dtype=dtype,
                              max_bytes=GPU_CACHE_MAX_BYTES, max_count=GPU_CACHE_MAX_EXPERTS)

# ============== Wrap experts: route via LRU & count usage (prefill only) ==============
expert_call_counts = defaultdict(int)
def make_cached_forward(expert_id:int, cpu_expert: nn.Module):
    def wrapped_forward(*args, **kwargs):
        expert_call_counts[expert_id] += 1  # count (prefill + decode)
        cuda_dev = torch.device(gpu_device)
        def to_cuda(x): return x.to(cuda_dev, non_blocking=True) if torch.is_tensor(x) else x
        args_cuda = tuple(to_cuda(a) for a in args)
        kwargs_cuda = {k: to_cuda(v) for k,v in kwargs.items()}
        cuda_expert = expert_cache.get(expert_id, cpu_expert)
        with torch.no_grad():
            return cuda_expert(*args_cuda, **kwargs_cuda)
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
print(f"\n[MEM] base={gb(base_alloc):.2f} GB  peak_prefill={gb(prefill_peak):.2f} GB  post_prefill={gb(post_prefill):.2f} GB  free={gb(free_bytes):.2f}/{gb(total_bytes):.2f} GB")

# ------------------ Ratio-based plan (from prefill usage) --------------------
sorted_eids = sorted(expert_call_counts.keys(), key=lambda k: expert_call_counts[k], reverse=True)
print("\n[Prefill usage] top experts:")
for i, e in enumerate(sorted_eids[:12]):
    print(f"  {i+1:2d}. EID {e:<4d} calls={expert_call_counts[e]:<6d}  {expert_name_map.get(e,'')}")

# Probe one expert size (temporary CUDA instance; NOT another prefill)
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

def plan_expert_selection_by_ratio(call_counts: dict[int,int],
                                   free_bytes_after_prefill: int,
                                   safety_bytes: int,
                                   count_cap: int,
                                   bytes_cap: int,
                                   mode: str,
                                   top_ratio: float,
                                   vram_ratio: float):
    if not call_counts:
        return [], 0, 0, {"reason":"no_calls"}

    sorted_eids_local = sorted(call_counts.keys(), key=lambda k: call_counts[k], reverse=True)
    total_calls = sum(call_counts.values())

    if mode == "usage_share":
        target_count = 0
        cum = 0
        for eid_ in sorted_eids_local:
            cum += call_counts[eid_]; target_count += 1
            if cum / max(1,total_calls) >= top_ratio: break
    elif mode == "count":
        target_count = max(1, int(len(sorted_eids_local) * top_ratio + 0.999))  # ceil
    else:
        raise ValueError("EXPERT_SELECT_MODE must be 'usage_share' or 'count'")

    free_budget = max(0, free_bytes_after_prefill - safety_bytes)
    bytes_budget = min(bytes_cap, int(free_budget * vram_ratio))

    per_expert_bytes = probe_expert_bytes(sorted_eids_local[0]) if sorted_eids_local else 50_000_000
    fit_by_bytes = 0 if per_expert_bytes == 0 else (bytes_budget // per_expert_bytes)

    final_keep = max(0, min(target_count, fit_by_bytes, count_cap))
    top_keep_ids = sorted_eids_local[:final_keep]
    max_bytes_final = min(bytes_cap, per_expert_bytes * final_keep)
    max_count_final = final_keep

    stats = {
        "mode": mode, "ratio": top_ratio, "vram_ratio": vram_ratio,
        "total_experts": len(sorted_eids_local),
        "total_calls": total_calls,
        "target_count": target_count,
        "per_expert_bytes_gb": per_expert_bytes/1e9,
        "bytes_budget_gb": bytes_budget/1e9,
        "fit_by_bytes": int(fit_by_bytes),
        "final_keep": final_keep,
    }
    return top_keep_ids, int(max_bytes_final), int(max_count_final), stats

safety = int(SAFETY_GB * 1e9)
top_keep_ids, bytes_cap_final, count_cap_final, plan_stats = plan_expert_selection_by_ratio(
    call_counts=expert_call_counts,
    free_bytes_after_prefill=free_bytes,
    safety_bytes=safety,
    count_cap=GPU_CACHE_MAX_EXPERTS,
    bytes_cap=GPU_CACHE_MAX_BYTES,
    mode=EXPERT_SELECT_MODE,
    top_ratio=EXPERT_TOP_RATIO,
    vram_ratio=EXPERT_VRAM_RATIO,
)

# Apply caps and prefetch the initial top-N
expert_cache.max_bytes = bytes_cap_final
expert_cache.max_count = count_cap_final
expert_cache.prefetch_experts(top_keep_ids)

print("\n[Ratio Plan]")
for k, v in plan_stats.items(): print(f"  {k}: {v}")
print(f"  selected_eids: {top_keep_ids}")
print(f"  caps → bytes={bytes_cap_final/1e9:.2f} GB  count={count_cap_final}")

# ================= Decode-time promotion machinery =================
promotion_queue = [e for e in sorted_eids if e not in top_keep_ids]

def free_pool_bytes():
    alloc = torch.cuda.memory_allocated()
    reserv = torch.cuda.memory_reserved()
    return max(0, reserv - alloc)

def maybe_bump_caps():
    """Increase expert_cache.max_bytes opportunistically based on current free pool."""
    headroom = free_pool_bytes()
    target = int(max(0, headroom - int(SAFETY_GB * 1e9)) * EXPERT_VRAM_RATIO)
    if target > expert_cache.max_bytes:
        expert_cache.max_bytes = min(target, GPU_CACHE_MAX_BYTES)

def promote_some_experts():
    """Prefetch up to PROMOTE_BATCH_SIZE experts from the queue if we have allocator headroom."""
    if not promotion_queue:
        return
    headroom_gb = free_pool_bytes() / 1e9
    if headroom_gb < FREEPOOL_THRESHOLD_GB:
        return
    maybe_bump_caps()
    batch = []
    while promotion_queue and len(batch) < PROMOTE_BATCH_SIZE:
        eid_ = promotion_queue.pop(0)
        if eid_ in cpu_expert_registry:
            batch.append(eid_)
    if batch:
        expert_cache.prefetch_experts(batch)
        print(f"[Promote] Loaded experts {batch} to GPU | headroom≈{headroom_gb:.2f} GB | cap={expert_cache.max_bytes/1e9:.2f} GB")

# ================================ Decode (reuse same KV) =============================
cur_ids   = inputs["input_ids"]
attn_mask = torch.zeros((B, prompt_len + max_new), dtype=torch.long, device=gpu_device)
attn_mask[:, :prompt_len] = 1
cur_len = prompt_len

start = time.time()
for step in range(max_new):
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

    # Opportunistic expert promotion between decode steps
    if (step + 1) % PROMOTE_EVERY_STEPS == 0:
        promote_some_experts()

    if (step + 1) % 16 == 0:
        torch.cuda.synchronize()
        pool_free_gb = (free_pool_bytes())/1e9
        print(f"after {step+1:4d} decode tokens: mem_alloc={gb(torch.cuda.memory_allocated()):.2f} GB | pool_free={pool_free_gb:.2f} GB")

torch.cuda.synchronize()
print(f"\n[Decode] {MAX_NEW_TOKENS} tokens in {time.time()-start:.2f}s  |  final_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

# Output
for i in range(B):
    print(f"\n--- OUTPUT {i} ---\n{tokenizer.decode(cur_ids[i], skip_special_tokens=True)}")
    
    
    
    
    
import os
# Keep Flash; allocator tuned to reduce fragmentation
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

max_memory = {0: "8GiB", "cpu": "300GiB"}  # keep attention+trunk under ~8GiB
GPU_CACHE_MAX_BYTES   = 2_000_000_000      # soft cap, will be tightened after prefill
GPU_CACHE_MAX_EXPERTS = 8                  # soft cap, will be tightened after prefill
MAX_NEW_TOKENS        = 128
SAFETY_GB             = 2.0

# ---- New knobs: ratio-based expert selection (prefill) ----
EXPERT_SELECT_MODE  = "usage_share"  # "usage_share" or "count"
EXPERT_TOP_RATIO    = 0.50           # 50% of routed usage (or top 50% by count)
EXPERT_VRAM_RATIO   = 0.50           # 50% of free VRAM (after prefill) given to experts

# ---- NEW: decode-time promotion scheduler ----
FREEPOOL_THRESHOLD_GB = 1.5          # promote only if (reserved-allocated) ≥ this
PROMOTE_EVERY_STEPS   = 4            # check every N decode steps
PROMOTE_BATCH_SIZE    = 3            # move up to K experts per check

# ==================== Load & dispatch (experts→CPU) ====================
print(f"Loading config & skeleton for {model_id} …")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

print("Inferring device_map …")
no_split = getattr(empty_model, "_no_split_modules", ["QwenBlock", "Block"])
device_map = infer_auto_device_map(empty_model, max_memory=max_memory, no_split_module_classes=no_split)
for name in list(device_map.keys()):
    if "experts" in name:
        device_map[name] = cpu_device  # experts on CPU by design

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

# ================= Expert LRU cache (unchanged core) ===================
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

# Registry (experts on CPU)
cpu_expert_registry: dict[int, nn.Module] = {}
expert_name_map: dict[int,str] = {}
eid = 0
for name, module in model.named_modules():
    if isinstance(module, nn.ModuleList) and "experts" in name:
        for idx, expert in enumerate(module):
            try: _ = next(expert.parameters())
            except StopIteration: pass
            cpu_expert_registry[eid] = expert
            expert_name_map[eid] = f"{name}[{idx}]"
            eid += 1
print(f"CPU-resident experts discovered: {len(cpu_expert_registry)}")

expert_cache = ExpertGPUCache(device=gpu_device, dtype=dtype,
                              max_bytes=GPU_CACHE_MAX_BYTES, max_count=GPU_CACHE_MAX_EXPERTS)

# ============== Wrap experts: route via LRU & count usage (prefill only) ==============
expert_call_counts = defaultdict(int)
def make_cached_forward(expert_id:int, cpu_expert: nn.Module):
    def wrapped_forward(*args, **kwargs):
        expert_call_counts[expert_id] += 1  # count during prefill & decode; we’ll use prefill results
        cuda_dev = torch.device(gpu_device)
        def to_cuda(x): return x.to(cuda_dev, non_blocking=True) if torch.is_tensor(x) else x
        args_cuda = tuple(to_cuda(a) for a in args)
        kwargs_cuda = {k: to_cuda(v) for k,v in kwargs.items()}
        cuda_expert = expert_cache.get(expert_id, cpu_expert)
        with torch.no_grad():
            return cuda_expert(*args_cuda, **kwargs_cuda)
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
print(f"\n[MEM] base={gb(base_alloc):.2f} GB  peak_prefill={gb(prefill_peak):.2f} GB  post_prefill={gb(post_prefill):.2f} GB  free={gb(free_bytes):.2f}/{gb(total_bytes):.2f} GB")

# ------------------ Ratio-based plan (NO 2nd prefill) --------------------
sorted_eids = sorted(expert_call_counts.keys(), key=lambda k: expert_call_counts[k], reverse=True)
print("\n[Prefill usage] top experts:")
for i, e in enumerate(sorted_eids[:12]):
    print(f"  {i+1:2d}. EID {e:<4d} calls={expert_call_counts[e]:<6d}  {expert_name_map.get(e,'')}")

# Probe one expert size (temporary CUDA instance; NOT another prefill)
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

def plan_expert_selection_by_ratio(call_counts: dict[int,int],
                                   free_bytes_after_prefill: int,
                                   safety_bytes: int,
                                   count_cap: int,
                                   bytes_cap: int,
                                   mode: str,
                                   top_ratio: float,
                                   vram_ratio: float):
    if not call_counts:
        return [], 0, 0, {"reason":"no_calls"}

    sorted_eids_local = sorted(call_counts.keys(), key=lambda k: call_counts[k], reverse=True)
    total_calls = sum(call_counts.values())

    if mode == "usage_share":
        target_count = 0
        cum = 0
        for eid_ in sorted_eids_local:
            cum += call_counts[eid_]; target_count += 1
            if cum / max(1,total_calls) >= top_ratio: break
    elif mode == "count":
        target_count = max(1, int(len(sorted_eids_local) * top_ratio + 0.999))  # ceil
    else:
        raise ValueError("EXPERT_SELECT_MODE must be 'usage_share' or 'count'")

    free_budget = max(0, free_bytes_after_prefill - safety_bytes)
    bytes_budget = min(bytes_cap, int(free_budget * vram_ratio))

    per_expert_bytes = probe_expert_bytes(sorted_eids_local[0]) if sorted_eids_local else 50_000_000
    fit_by_bytes = 0 if per_expert_bytes == 0 else (bytes_budget // per_expert_bytes)

    final_keep = max(0, min(target_count, fit_by_bytes, count_cap))
    top_keep_ids = sorted_eids_local[:final_keep]
    max_bytes_final = min(bytes_cap, per_expert_bytes * final_keep)
    max_count_final = final_keep

    stats = {
        "mode": mode, "ratio": top_ratio, "vram_ratio": vram_ratio,
        "total_experts": len(sorted_eids_local),
        "total_calls": total_calls,
        "target_count": target_count,
        "per_expert_bytes_gb": per_expert_bytes/1e9,
        "bytes_budget_gb": bytes_budget/1e9,
        "fit_by_bytes": int(fit_by_bytes),
        "final_keep": final_keep,
    }
    return top_keep_ids, int(max_bytes_final), int(max_count_final), stats

safety = int(SAFETY_GB * 1e9)
top_keep_ids, bytes_cap_final, count_cap_final, plan_stats = plan_expert_selection_by_ratio(
    call_counts=expert_call_counts,
    free_bytes_after_prefill=free_bytes,
    safety_bytes=safety,
    count_cap=GPU_CACHE_MAX_EXPERTS,
    bytes_cap=GPU_CACHE_MAX_BYTES,
    mode=EXPERT_SELECT_MODE,
    top_ratio=EXPERT_TOP_RATIO,
    vram_ratio=EXPERT_VRAM_RATIO,
)

# Apply caps to your LRU and prefetch
expert_cache.max_bytes = bytes_cap_final
expert_cache.max_count = count_cap_final
expert_cache.prefetch_experts(top_keep_ids)

print("\n[Ratio Plan]")
for k, v in plan_stats.items(): print(f"  {k}: {v}")
print(f"  selected_eids: {top_keep_ids}")
print(f"  caps → bytes={bytes_cap_final/1e9:.2f} GB  count={count_cap_final}")

# ================= NEW: decode-time promotion machinery =================
promotion_queue = [e for e in sorted_eids if e not in top_keep_ids]

def free_pool_bytes():
    alloc = torch.cuda.memory_allocated()
    reserv = torch.cuda.memory_reserved()
    return max(0, reserv - alloc)

def maybe_bump_caps():
    """Increase expert_cache.max_bytes opportunistically based on current free pool."""
    headroom = free_pool_bytes()
    target = int(max(0, headroom - int(SAFETY_GB * 1e9)) * EXPERT_VRAM_RATIO)
    if target > expert_cache.max_bytes:
        expert_cache.max_bytes = min(target, GPU_CACHE_MAX_BYTES)

def promote_some_experts():
    """Prefetch up to PROMOTE_BATCH_SIZE experts from the queue if we have allocator headroom."""
    if not promotion_queue:
        return
    headroom_gb = free_pool_bytes() / 1e9
    if headroom_gb < FREEPOOL_THRESHOLD_GB:
        return
    maybe_bump_caps()
    batch = []
    while promotion_queue and len(batch) < PROMOTE_BATCH_SIZE:
        eid_ = promotion_queue.pop(0)
        if eid_ in cpu_expert_registry:
            batch.append(eid_)
    if batch:
        expert_cache.prefetch_experts(batch)
        print(f"[Promote] Loaded experts {batch} to GPU | headroom≈{headroom_gb:.2f} GB | cap={expert_cache.max_bytes/1e9:.2f} GB")

# ================================ Decode (reuse same KV) =============================
B = inputs["input_ids"].size(0)
cur_ids   = inputs["input_ids"]
attn_mask = torch.zeros((B, prompt_len + max_new), dtype=torch.long, device=gpu_device)
attn_mask[:, :prompt_len] = 1
cur_len = prompt_len

start = time.time()
for step in range(max_new):
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

    # NEW: opportunistic expert promotion between decode steps
    if (step + 1) % PROMOTE_EVERY_STEPS == 0:
        promote_some_experts()

    if (step + 1) % 16 == 0:
        torch.cuda.synchronize()
        pool_free_gb = (free_pool_bytes())/1e9
        print(f"after {step+1:4d} decode tokens: mem_alloc={gb(torch.cuda.memory_allocated()):.2f} GB | pool_free={pool_free_gb:.2f} GB")

torch.cuda.synchronize()
print(f"\n[Decode] {MAX_NEW_TOKENS} tokens in {time.time()-start:.2f}s  |  final_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

# Output
for i in range(B):
    print(f"\n--- OUTPUT {i} ---\n{tokenizer.decode(cur_ids[i], skip_special_tokens=True)}")
