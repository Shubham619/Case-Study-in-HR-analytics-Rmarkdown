# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate.utils import infer_auto_device_map
from accelerate import dispatch_model, init_empty_weights

# =========================
# Config
# =========================
model_id = "deepseek-ai/deepseek-moe-16b-base"
gpu      = "cuda:0"
dtype    = torch.bfloat16                     # or torch.float16 if needed
vram_gib = "20GiB"                            # GPU budget for trunk/router/KV
dram_gib = "200GiB"                           # DRAM budget for experts

# =========================
# 1) Build meta skeleton + infer auto device_map
# =========================
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

device_map = infer_auto_device_map(
    empty_model,
    max_memory={0: vram_gib, "cpu": dram_gib},
    no_split_module_classes=["Block"],  # keep transformer blocks whole
)

# Force all experts to CPU (storage). We'll run compute on CUDA via per-call GPU clones.
for name in list(device_map.keys()):
    if "experts" in name:
        device_map[name] = "cpu"

print("🔥 Customised auto device map (first ~25):")
for i, (k, v) in enumerate(device_map.items()):
    if i >= 25: break
    print(f"{k} -> {v}")
print("...")

# =========================
# 2) Load and dispatch model per device_map (NO disk offload)
# =========================
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    trust_remote_code=True,
)
model = dispatch_model(
    model,
    device_map=device_map,
    offload_dir=None  # important: don't use disk
)

# =========================
# 3) Patch experts: DRAM storage, CUDA compute (NO GPU caching)
#    >>> Corrected to use .to_empty(...) for module cloning <<<
# =========================
def patch_expert_forward_no_cache(expert: nn.Module, gpu_device: str = "cuda:0"):
    """
    Keep expert instance/type intact (router-safe).
    Storage remains on CPU (as dispatched). On each forward:
      - make temporary GPU clone via to_empty(device=...)
      - load CPU state_dict into it
      - run original forward on CUDA
      - delete the clone (free VRAM)
    """
    original_forward = expert.forward
    cuda_dev = torch.device(gpu_device)

    # Ensure these are REAL tensors (dispatch_model has loaded them)
    for p in expert.parameters(recurse=True):
        if getattr(p, "is_meta", False):
            raise RuntimeError("Expert still has meta tensors; dispatch/load failed.")

    def wrapped_forward(*args, **kwargs):
        # Inputs should already be on CUDA; ensure anyway (no-op if already CUDA)
        args   = [a.to(cuda_dev, non_blocking=True) if torch.is_tensor(a) else a for a in args]
        kwargs = {k: (v.to(cuda_dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in kwargs.items()}

        # >>> CORRECT: create an EMPTY GPU clone of the module (no data copied implicitly)
        expert_gpu = expert.to_empty(device=cuda_dev)

        # Load weights/buffers from CPU expert (safe; name-aligned)
        expert_gpu.load_state_dict(expert.state_dict(), strict=True)

        # Ensure dtype on GPU clone (if needed)
        for p in expert_gpu.parameters(recurse=True):
            p.data = p.data.to(dtype)
        for b in expert_gpu.buffers(recurse=True):
            if b is not None:
                b.data = b.data.to(dtype)

        # Bind original forward to the GPU clone and compute on CUDA
        with torch.no_grad():
            out = original_forward.__get__(expert_gpu, type(expert_gpu))(*args, **kwargs)

        # Free temporary GPU clone
        del expert_gpu
        torch.cuda.empty_cache()
        return out

    expert.forward = wrapped_forward

# Patch every ModuleList named "...experts"
patched = 0
for name, module in model.named_modules():
    if "experts" in name and isinstance(module, nn.ModuleList):
        for idx, expert in enumerate(module):
            patch_expert_forward_no_cache(expert, gpu_device=gpu)
            patched += 1
print(f"Patched experts (no GPU cache): {patched}")
assert patched > 0, "No experts found to patch; check module names."

# =========================
# 4) Tokenizer + quick sanity
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Sanity: some params on CPU (experts), some on CUDA (trunk/router)
cpu_params  = sum(p.numel() for p in model.parameters() if p.device.type == "cpu")
cuda_params = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
print(f"Param split — CPU: {cpu_params/1e9:.2f}B | CUDA: {cuda_params/1e9:.2f}B")

with torch.no_grad():
    t = tokenizer("sanity.", return_tensors="pt").to(gpu)
    logits = model(**t).logits
    print("logits mean/std:",
          float(logits.float().mean()), float(logits.float().std()),
          "NaN?", bool(torch.isnan(logits).any()))

# =========================
# 5) Generate (force new tokens so it won't echo prompt)
# =========================
prompt = "DeepSeek is transforming inference efficiency."
inputs = tokenizer(prompt, return_tensors="pt").to(gpu)

gen_kwargs = dict(
    max_new_tokens=64,
    do_sample=True, top_p=0.95, top_k=50, temperature=0.9,
    repetition_penalty=1.05, min_new_tokens=16,
    eos_token_id=tokenizer.eos_token_id,
)

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
