# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"
GPU      = "cuda:0"
DTYPE    = torch.bfloat16          # or torch.float16 if needed
VRAM_BUDGET = "20GiB"
DRAM_BUDGET = "200GiB"

# 0) Get a LOCAL checkpoint folder (required by load_checkpoint_and_dispatch)
ckpt_dir = snapshot_download(MODEL_ID)

# 1) Build meta skeleton
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# IMPORTANT: use the model’s own no-split classes if provided
no_split = getattr(empty_model, "_no_split_modules", None)
if not no_split:
    # Fallback — try common names used by this repo (adjust if needed)
    no_split = ["DeepseekDecoderLayer", "DeepseekBlock", "Block"]

# 2) Infer device map (experts -> CPU storage)
device_map = infer_auto_device_map(
    empty_model,
    max_memory={0: VRAM_BUDGET, "cpu": DRAM_BUDGET},
    no_split_module_classes=no_split,
)

for name in list(device_map.keys()):
    if "experts" in name:
        device_map[name] = "cpu"

print("== device_map (first ~25) ==")
for i, (k, v) in enumerate(device_map.items()):
    if i >= 25: break
    print(f"{k} -> {v}")
print("...")

# 3) Stream weights into the meta skeleton (materializes tensors!)
model = load_checkpoint_and_dispatch(
    empty_model,
    checkpoint=ckpt_dir,                        # MUST be a local path
    device_map=device_map,
    no_split_module_classes=no_split,
    dtype=DTYPE,
    offload_folder=None,                        # no disk
)

# ---- Helper: check for any lingering meta tensors
def first_meta_param(mod: nn.Module):
    for n, p in mod.named_parameters(recurse=True):
        if getattr(p, "is_meta", False) or (hasattr(p, "device") and p.device.type == "meta"):
            return n
    for n, b in mod.named_buffers(recurse=True):
        if b is not None and (getattr(b, "is_meta", False) or (hasattr(b, "device") and b.device.type == "meta")):
            return n
    return None

meta_name = first_meta_param(model)
if meta_name:
    # Some trust_remote_code models lazily build submodules; do a tiny warm-up to instantiate them
    print(f"[warn] Found meta tensor at: {meta_name}. Doing a 1-token warmup to materialize lazies...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    with torch.no_grad():
        dummy = tok(".", return_tensors="pt").to(GPU)
        _ = model(**dummy)  # forces construction/loading paths
    meta_name = first_meta_param(model)
    if meta_name:
        raise RuntimeError(f"Still meta after warmup: {meta_name}. "
                           f"Double-check no_split={no_split} and that checkpoint folder is correct.")

# 4) Patch experts: store on CPU, compute on CUDA (NO GPU cache)
def patch_expert_forward_no_cache(expert: nn.Module, gpu_device: str = "cuda:0"):
    """
    Keep expert object/type intact (router-safe).
    On each forward:
      - build a temporary EMPTY GPU clone (to_empty)
      - load CPU state_dict into it
      - run original forward bound to the GPU clone
      - free the clone
    """
    original_forward = expert.forward
    cuda_dev = torch.device(gpu_device)

    # Re-validate: no meta tensors inside expert
    bad = first_meta_param(expert)
    if bad:
        raise RuntimeError(f"Expert still meta at {bad}. Load/dispatch did not materialize this module.")

    def wrapped_forward(*args, **kwargs):
        # Ensure inputs are on CUDA (usually already true)
        args   = [a.to(cuda_dev, non_blocking=True) if torch.is_tensor(a) else a for a in args]
        kwargs = {k: (v.to(cuda_dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in kwargs.items()}

        # Structure-only GPU clone
        expert_gpu = expert.to_empty(device=cuda_dev)
        # Load weights/buffers by name (safe; no meta movement)
        expert_gpu.load_state_dict(expert.state_dict(), strict=True)

        # Make sure dtype is correct on the clone
        for p in expert_gpu.parameters(recurse=True):
            p.data = p.data.to(DTYPE)
        for b in expert_gpu.buffers(recurse=True):
            if b is not None:
                b.data = b.data.to(DTYPE)

        with torch.no_grad():
            out = original_forward.__get__(expert_gpu, type(expert_gpu))(*args, **kwargs)

        # Free the temporary clone
        del expert_gpu
        torch.cuda.empty_cache()
        return out

    expert.forward = wrapped_forward

# Find and patch ModuleLists named "...experts"
patched = 0
for name, module in model.named_modules():
    if "experts" in name and isinstance(module, nn.ModuleList):
        for idx, expert in enumerate(module):
            patch_expert_forward_no_cache(expert, gpu_device=GPU)
            patched += 1
print(f"Patched experts (no GPU cache): {patched}")
assert patched > 0, "No experts found; check your model module names."

# 5) Tokenizer + quick sanity + generate
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Sanity: some params on CPU (experts), some on CUDA (trunk/router)
cpu_params  = sum(p.numel() for p in model.parameters() if p.device.type == "cpu")
cuda_params = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
print(f"Param split — CPU: {cpu_params/1e9:.2f}B | CUDA: {cuda_params/1e9:.2f}B")

with torch.no_grad():
    t = tokenizer("sanity.", return_tensors="pt").to(GPU)
    logits = model(**t).logits
    print("logits mean/std:", float(logits.float().mean()), float(logits.float().std()),
          "NaN?", bool(torch.isnan(logits).any()))

prompt = "DeepSeek is transforming inference efficiency."
inputs = tokenizer(prompt, return_tensors="pt").to(GPU)
gen_kwargs = dict(
    max_new_tokens=64,
    do_sample=True, top_p=0.95, top_k=50, temperature=0.9,
    repetition_penalty=1.05, min_new_tokens=16,
    eos_token_id=tokenizer.eos_token_id,
)
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
