# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from contextlib import contextmanager

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
from huggingface_hub import snapshot_download

# =========================
# Config
# =========================
repo_id   = "Qwen/Qwen1.5-MoE-A2.7B"   # <-- change to your model if needed
dtype     = torch.bfloat16             # or torch.float16 if needed
gpu       = "cuda:0"
vram_gib  = "20GiB"                    # your GPU budget for trunk/router/attn/KV
dram_gib  = "200GiB"                   # DRAM budget

# =========================
# Helpers
# =========================
@contextmanager
def no_grad_infer():
    state = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        yield
    finally:
        torch.set_grad_enabled(state)

def _move_module_data_(module: nn.Module, device: torch.device):
    # Move only .data (in place) to avoid changing Parameter objects
    for p in module.parameters(recurse=True):
        p.data = p.data.to(device, non_blocking=True)
    for b in module.buffers(recurse=True):
        b.data = b.data.to(device, non_blocking=True)

def _patch_expert_forward_storage_cpu_compute_gpu(expert: nn.Module, gpu_device: str = "cuda:0"):
    """
    Keep the expert instance/type intact (so the router still recognizes it).
    Store weights/buffers on CPU between calls; on each forward:
      CPU -> CUDA (compute) -> CPU (store).
    """
    original_forward = expert.forward
    cpu_dev  = torch.device("cpu")
    cuda_dev = torch.device(gpu_device)

    # Ensure storage starts on CPU
    _move_module_data_(expert, cpu_dev)

    def wrapped_forward(*args, **kwargs):
        with no_grad_infer():
            # Inputs should already be CUDA; make robust:
            args   = [a.to(cuda_dev, non_blocking=True) if torch.is_tensor(a) else a for a in args]
            kwargs = {k: (v.to(cuda_dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in kwargs.items()}

            # Bring expert params to CUDA for compute
            _move_module_data_(expert, cuda_dev)

            # Run compute on GPU
            out = original_forward(*args, **kwargs)

            # Return expert params to CPU for DRAM storage
            _move_module_data_(expert, cpu_dev)

            return out

    expert.forward = wrapped_forward  # monkey-patch in place

# =========================
# 0) Download checkpoint locally (IMPORTANT for load_checkpoint_and_dispatch)
# =========================
local_ckpt_dir = snapshot_download(
    repo_id=repo_id,
    # revision="main",  # optionally pin a revision/tag/commit
    # you can ignore non-needed artifacts to speed up downloads
    # ignore_patterns=["*.h5", "*.safetensors.index.json"],
)

# =========================
# 1) Build skeleton on meta (no memory spike)
# =========================
config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
with init_empty_weights():
    empty = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# =========================
# 2) Infer device map; force MoE experts to CPU
# =========================
max_memory = {0: vram_gib, "cpu": dram_gib}
device_map = infer_auto_device_map(
    empty,
    max_memory=max_memory,
    no_split_module_classes=["Block"],  # keep transformer blocks intact
)

# Force *all* submodules whose path contains "experts" to CPU
for name in list(device_map.keys()):
    if "experts" in name:
        device_map[name] = "cpu"

print("== Device map (first ~25 entries) ==")
for i, (k, v) in enumerate(device_map.items()):
    if i >= 25: break
    print(f"{k} -> {v}")
print("...")

# =========================
# 3) Stream weights into mapped devices (NO Hub ID here — must be local path)
# =========================
model = load_checkpoint_and_dispatch(
    empty,
    checkpoint=local_ckpt_dir,                 # <--- the local folder we downloaded
    device_map=device_map,
    no_split_module_classes=["Block"],
    dtype=dtype,
)

# =========================
# 4) Patch every MoE expert: storage=DRAM (CPU), compute=CUDA
# =========================
patched = 0
for name, module in model.named_modules():
    # Qwen/DeepSeek typically have ModuleList named "experts"
    if "experts" in name and isinstance(module, nn.ModuleList):
        for idx, expert in enumerate(module):
            _patch_expert_forward_storage_cpu_compute_gpu(expert, gpu_device=gpu)
            patched += 1
print(f"Patched experts: {patched}")
assert patched > 0, "No experts were found to patch. Check your model's module names."

# =========================
# 5) Tokenizer + sanity checks
# =========================
tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

# Quick logits sanity (non-zero/non-NaN)
with torch.no_grad():
    test_in = tok("Sanity test.", return_tensors="pt").to(gpu)
    test_out = model(**test_in)
    last = test_out.logits[0, -1].float()
    print(f"logits mean={last.mean().item():.6f}, std={last.std().item():.6f}, NaN? {torch.isnan(last).any().item()}")

# Show parameter split across devices
cpu_params  = sum(p.numel() for p in model.parameters() if p.device.type == "cpu")
cuda_params = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
print(f"Param split — CPU: {cpu_params/1e9:.2f}B | CUDA: {cuda_params/1e9:.2f}B")

# =========================
# 6) Generate (force new tokens to avoid prompt echo)
# =========================
gen_kwargs = dict(
    max_new_tokens=96,
    do_sample=True,
    top_p=0.95,
    top_k=50,
    temperature=0.9,
    repetition_penalty=1.05,
    min_new_tokens=16,                 # force extension beyond prompt
    eos_token_id=tok.eos_token_id,
)

prompt = "Briefly explain Mixture-of-Experts routing and why it helps large language models."
inputs = tok(prompt, return_tensors="pt").to(gpu)

with torch.no_grad():
    out_ids = model.generate(**inputs, **gen_kwargs)

print("\n=== OUTPUT ===")
print(tok.decode(out_ids[0], skip_special_tokens=True))
