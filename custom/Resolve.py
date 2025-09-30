import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","backend:cudaMallocAsync,expandable_segments:True")

import time
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

# ============================ Config ============================
MODEL_ID   = "Qwen/Qwen1.5-MoE-A2.7B"
GPU        = "cuda:0"
CPU        = "cpu"
DTYPE      = torch.float16

MAX_NEW_TOKENS        = 128
SAFETY_GB             = 2.0  # keep this much headroom always

# Expert cache caps (can grow opportunistically during decode)
GPU_CACHE_MAX_BYTES   = 2_000_000_000
GPU_CACHE_MAX_EXPERTS = 8

# Prefill → ranking knobs
EXPERT_SELECT_MODE  = "usage_share"   # "usage_share" or "count"
EXPERT_TOP_RATIO    = 0.50            # take experts covering 50% of routed usage (or top 50% by count)
EXPERT_VRAM_RATIO   = 0.50            # use up to 50% of available headroom for experts

# Decode-time promotion knobs
FREEPOOL_THRESHOLD_GB = 1.5           # promote only if (reserved-allocated) ≥ this
PROMOTE_EVERY_STEPS   = 4             # check every N decode steps
PROMOTE_BATCH_SIZE    = 3             # promote up to K experts each time

# ============================ Load model (CPU) ============================
print(f"Loading {MODEL_ID} on CPU …")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    trust_remote_code=True,
    device_map={"": CPU},  # fully materialized, no meta tensors, no accelerate asserts
)
model.config.use_cache = True
model.eval()

# =================== Utilities: move trunk+gates → GPU ===================
def _to_cuda_if_has_params(mod: nn.Module, device: torch.device, dtype: torch.dtype):
    has_param = any(True for _ in mod.parameters(recurse=False))
    has_buf   = any(True for _ in mod.buffers(recurse=False))
    if has_param or has_buf:
        mod.to(device, dtype=dtype)

def move_trunk_and_gates_to_gpu(model: nn.Module, device: str, dtype: torch.dtype):
    dev = torch.device(device)
    # 1) move everything except experts
    for name, mod in model.named_modules():
        if "experts" in name:
            continue  # keep experts on CPU
        _to_cuda_if_has_params(mod, dev, dtype)
    # 2) force every gate (and optional gate.proj) onto GPU explicitly
    for idx, layer in enumerate(getattr(model, "model").layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            layer.mlp.gate.to(dev, dtype=dtype)
            if hasattr(layer.mlp.gate, "proj"):
                layer.mlp.gate.proj.to(dev, dtype=dtype)

def assert_gate_ready(model: nn.Module, device: str):
    dev_type = torch.device(device).type
    for i, layer in enumerate(getattr(model, "model").layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            g = layer.mlp.gate
            assert any(True for _ in g.parameters()), f"Layer {i} gate has no parameters"
            for p in g.parameters():
                assert p.device.type == dev_type, f"Layer {i} gate on {p.device}, expected {dev_type}"
                assert p.numel() > 0, f"Layer {i} gate param is empty"
            if hasattr(g, "proj"):
                for p in g.proj.parameters():
                    assert p.device.type == dev_type, f"Layer {i} gate.proj on {p.device}, expected {dev_type}"
                    assert p.numel() > 0, f"Layer {i} gate.proj param is empty"

# ================= Expert cache (GPU LRU; weights from CPU) ==============
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
            t = v.detach().contiguous().to(CPU, copy=True)
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

# ================== Discover experts (stay on CPU) ======================
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
expert_cache = ExpertGPUCache(device=GPU, dtype=DTYPE,
                              max_bytes=GPU_CACHE_MAX_BYTES, max_count=GPU_CACHE_MAX_EXPERTS)

# ============== Wrap experts to route via cache & count usage ============
expert_call_counts = defaultdict(int)
def make_cached_forward(expert_id:int, cpu_expert: nn.Module):
    def wrapped_forward(*args, **kwargs):
        expert_call_counts[expert_id] += 1
        cuda_dev = torch.device(GPU)
        def to_cuda(x): return x.to(cuda_dev, non_blocking=True) if torch.is_tensor(x) else x
        args_cuda   = tuple(to_cuda(a) for a in args)
        kwargs_cuda = {k: to_cuda(v) for k,v in kwargs.items()}
        cuda_expert = expert_cache.get(expert_id, cpu_expert)
        with torch.no_grad():
            return cuda_expert(*args_cuda, **kwargs_cuda)
    return wrapped_forward

for eid_, cpu_expert in cpu_expert_registry.items():
    cpu_expert.forward = make_cached_forward(eid_, cpu_expert)

# =================== Inputs, move trunk+gates, KV =======================
PROMPTS = ["Explain the concept of dynamic offloading in large language models in one paragraph."]
inputs = tok(PROMPTS, return_tensors="pt", padding=True, truncation=False)

# 1) move trunk + all gates/proj to GPU (experts remain CPU)
print("Moving trunk + all gates to GPU …")
move_trunk_and_gates_to_gpu(model, GPU, DTYPE)
assert_gate_ready(model, GPU)

# 2) put inputs on GPU
inputs = {k: v.to(GPU) for k, v in inputs.items()}

B = inputs["input_ids"].size(0)
prompt_len = inputs["input_ids"].size(1)
max_new = MAX_NEW_TOKENS

# 3) allocate static KV on GPU
kv = StaticCache(
    max_batch_size=B,
    max_cache_len=prompt_len + max_new,
    config=model.config,
    device=torch.device(GPU),
    dtype=model.dtype,
)

def gb(x): return x/1e9
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
base_alloc = torch.cuda.memory_allocated()

# ============================== PREFILL ================================
with torch.inference_mode():
    _ = model(**inputs, use_cache=True, past_key_values=kv)

torch.cuda.synchronize()
prefill_peak   = torch.cuda.max_memory_allocated()
post_prefill   = torch.cuda.memory_allocated()
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"\n[MEM] base={gb(base_alloc):.2f} GB | peak_prefill={gb(prefill_peak):.2f} GB | post_prefill={gb(post_prefill):.2f} GB | free={gb(free_bytes):.2f}/{gb(total_bytes):.2f} GB")

# ================= Rank experts by prefill usage & prefetch some =======
sorted_eids = sorted(expert_call_counts.keys(), key=lambda k: expert_call_counts[k], reverse=True)
print("\n[Prefill usage] top experts:")
for i, e in enumerate(sorted_eids[:12]):
    print(f"  {i+1:2d}. EID {e:<4d} calls={expert_call_counts[e]:<6d}  {expert_name_map.get(e,'')}")

def probe_expert_bytes(eid_probe:int) -> int:
    cpu_mod = cpu_expert_registry[eid_probe]
    cls = cpu_mod.__class__
    if hasattr(cpu_mod, "config"):
        tmp = cls(config=cpu_mod.config).to(GPU, dtype=DTYPE)
    else:
        import copy as _copy
        tmp = _copy.deepcopy(cpu_mod).to(GPU, dtype=DTYPE)
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
    sorted_local = sorted(call_counts.keys(), key=lambda k: call_counts[k], reverse=True)
    total_calls = sum(call_counts.values())
    if mode == "usage_share":
        target_count, cum = 0, 0
        for eid_ in sorted_local:
            cum += call_counts[eid_]; target_count += 1
            if cum / max(1,total_calls) >= top_ratio: break
    elif mode == "count":
        target_count = max(1, int(len(sorted_local) * top_ratio + 0.999))
    else:
        raise ValueError("EXPERT_SELECT_MODE must be 'usage_share' or 'count'")
    free_budget   = max(0, free_bytes_after_prefill - safety_bytes)
    bytes_budget  = min(bytes_cap, int(free_budget * vram_ratio))
    per_bytes     = probe_expert_bytes(sorted_local[0]) if sorted_local else 50_000_000
    fit_by_bytes  = 0 if per_bytes == 0 else (bytes_budget // per_bytes)
    final_keep    = max(0, min(target_count, fit_by_bytes, count_cap))
    top_keep_ids  = sorted_local[:final_keep]
    max_bytes_fin = min(bytes_cap, per_bytes * final_keep)
    max_count_fin = final_keep
    stats = {
        "mode": mode, "ratio": top_ratio, "vram_ratio": vram_ratio,
        "total_experts": len(sorted_local),
        "total_calls": total_calls,
        "target_count": target_count,
        "per_expert_bytes_gb": per_bytes/1e9,
        "bytes_budget_gb": bytes_budget/1e9,
        "fit_by_bytes": int(fit_by_bytes),
        "final_keep": final_keep,
    }
    return top_keep_ids, int(max_bytes_fin), int(max_count_fin), stats

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
expert_cache.max_bytes = bytes_cap_final
expert_cache.max_count = count_cap_final
expert_cache.prefetch_experts(top_keep_ids)

print("\n[Ratio Plan]")
for k, v in plan_stats.items(): print(f"  {k}: {v}")
print(f"  selected_eids: {top_keep_ids}")
print(f"  caps → bytes={bytes_cap_final/1e9:.2f} GB | count={count_cap_final}")

# ================ Decode-time promotion (VRAM-aware) ====================
promotion_queue = [e for e in sorted_eids if e not in top_keep_ids]

def free_pool_bytes():
    # allocator headroom inside CUDA pool
    alloc = torch.cuda.memory_allocated()
    reserv = torch.cuda.memory_reserved()
    return max(0, reserv - alloc)

def maybe_bump_caps():
    headroom = free_pool_bytes()
    target = int(max(0, headroom - int(SAFETY_GB * 1e9)) * EXPERT_VRAM_RATIO)
    if target > expert_cache.max_bytes:
        expert_cache.max_bytes = target  # bounded by actual headroom + eviction

def promote_some_experts():
    if not promotion_queue: return
    headroom_gb = free_pool_bytes() / 1e9
    if headroom_gb < FREEPOOL_THRESHOLD_GB: return
    maybe_bump_caps()
    batch = []
    while promotion_queue and len(batch) < PROMOTE_BATCH_SIZE:
        eid_ = promotion_queue.pop(0)
        if eid_ in cpu_expert_registry:
            batch.append(eid_)
    if batch:
        expert_cache.prefetch_experts(batch)
        print(f"[Promote] Experts {batch} → GPU | pool_free≈{headroom_gb:.2f} GB | cap={expert_cache.max_bytes/1e9:.2f} GB")

# ================================ Decode ================================
cur_ids   = inputs["input_ids"]
attn_mask = torch.zeros((B, prompt_len + max_new), dtype=torch.long, device=GPU)
attn_mask[:, :prompt_len] = 1
cur_len = prompt_len

start = time.time()
for step in range(max_new):
    with torch.inference_mode():
        step_input = cur_ids[:, cur_len-1:cur_len].contiguous()
        logits = model(
            input_ids=step_input.to(GPU),
            attention_mask=attn_mask[:, :cur_len],
            use_cache=True,
            past_key_values=kv
        ).logits
        next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_ids.to(cur_ids.device)], dim=1)
        attn_mask[:, cur_len] = 1
        cur_len += 1

    if (step + 1) % PROMOTE_EVERY_STEPS == 0:
        promote_some_experts()

    if (step + 1) % 16 == 0:
        torch.cuda.synchronize()
        print(f"after {step+1:4d} tokens: alloc={gb(torch.cuda.memory_allocated()):.2f} GB | pool_free={gb(free_pool_bytes()):.2f} GB")

torch.cuda.synchronize()
print(f"\n[Decode] {MAX_NEW_TOKENS} tokens in {time.time()-start:.2f}s | final_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

for i in range(B):
    print(f"\n--- OUTPUT {i} ---\n{tok.decode(cur_ids[i], skip_special_tokens=True)}")
