import torch
import torch.nn as nn
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

# -------------------------------
# Utilities
# -------------------------------
@contextmanager
def _no_grad_inference():
    was_training = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        yield
    finally:
        torch.set_grad_enabled(was_training)

def _move_module_(module: nn.Module, device: torch.device):
    # In-place move of params & buffers (data only) to avoid re-wrapping
    for p in module.parameters(recurse=True):
        p.data = p.data.to(device, non_blocking=True)
    for b in module.buffers(recurse=True):
        b.data = b.data.to(device, non_blocking=True)

def _patch_expert_for_gpu_compute_storage_cpu(expert: nn.Module, gpu_device: str = "cuda:0"):
    """
    Keep expert object/type the same.
    Store weights on CPU; on forward:
      CPU -> GPU (compute) -> CPU (store)
    """
    original_forward = expert.forward
    cpu_device = torch.device("cpu")
    cuda_device = torch.device(gpu_device)

    # Ensure expert weights/buffers are on CPU for storage
    _move_module_(expert, cpu_device)

    def wrapped_forward(*args, **kwargs):
        with _no_grad_inference():
            # Inputs should already be on CUDA from the trunk/router; ensure anyway
            args = [a.to(cuda_device, non_blocking=True) if torch.is_tensor(a) else a for a in args]
            kwargs = {k: (v.to(cuda_device, non_blocking=True) if torch.is_tensor(v) else v)
                      for k, v in kwargs.items()}

            # Move expert weights to CUDA for compute
            _move_module_(expert, cuda_device)

            # Compute on GPU
            out = original_forward(*args, **kwargs)

            # Move expert weights back to CPU for storage
            _move_module_(expert, cpu_device)

            return out

    expert.forward = wrapped_forward  # monkey-patch in place


# -------------------------------
# 0) Config
# -------------------------------
model_id = "deepseek-ai/deepseek-moe-16b-base"  # or your variant
gpu = "cuda:0"
dtype = torch.bfloat16

# -------------------------------
# 1) Init skeleton on meta (no memory)
# -------------------------------
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
with init_empty_weights():
    empty = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# -------------------------------
# 2) Infer device map (experts -> CPU; trunk/router -> GPU as budget allows)
# -------------------------------
max_memory = {
    0: "20GiB",     # your VRAM budget for trunk/router/attn/KV
    "cpu": "200GiB" # DRAM budget
}
device_map = infer_auto_device_map(
    empty,
    max_memory=max_memory,
    no_split_module_classes=["Block"],  # keep transformer blocks intact
)

# Force all experts to CPU so they materialize in DRAM
for name in list(device_map.keys()):
    if "experts" in name:
        device_map[name] = "cpu"

print("== Device map (truncated) ==")
for k, v in list(device_map.items())[:20]:
    print(k, "->", v)
print("...")

# -------------------------------
# 3) Load real weights (streamed to mapped devices)
# -------------------------------
model = load_checkpoint_and_dispatch(
    empty,
    checkpoint=model_id,
    device_map=device_map,
    no_split_module_classes=["Block"],
    dtype=dtype,
)

# Move the non-expert trunk/router to GPU if not already
# (infer_auto_device_map should have mapped them, but ensure model main device)
# Do NOT .to(gpu) the whole model, it will drag experts too; only rely on device_map.

# -------------------------------
# 4) Patch every MoE expert in place
# -------------------------------
# We keep expert instances & types intact; we only swap their forward to CPU-storage/GPU-compute.
expert_count = 0
for name, module in model.named_modules():
    if "experts" in name and isinstance(module, nn.ModuleList):
        for idx, expert in enumerate(module):
            _patch_expert_for_gpu_compute_storage_cpu(expert, gpu_device=gpu)
            expert_count += 1
print(f"Patched experts for DRAM storage + GPU compute: {expert_count}")

# -------------------------------
# 5) Tokenizer & generation sanity settings
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Ensure we actually generate NEW tokens; avoid early-stop echo:
gen_kwargs = dict(
    max_new_tokens=64,
    do_sample=True,        # sampling forces decoding beyond prompt
    top_p=0.95,
    top_k=50,
    temperature=0.8,
    repetition_penalty=1.05,   # reduce verbatim echo
    eos_token_id=tokenizer.eos_token_id,
)

# -------------------------------
# 6) Quick sanity checks
# -------------------------------
# 6a) Verify some params are on CPU (experts) and some on CUDA (trunk)
cpu_params = sum(p.numel() for p in model.parameters() if p.device.type == "cpu")
cuda_params = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
print(f"Param split — CPU: {cpu_params/1e9:.2f}B, CUDA: {cuda_params/1e9:.2f}B")

# 6b) Verify logits aren't all-zero / NaN
with torch.no_grad():
    test_in = tokenizer("Sanity test.", return_tensors="pt").to(gpu)
    test_out = model(**test_in)
    last_logits = test_out.logits[0, -1]
    print(f"logits mean: {float(last_logits.float().mean()):.6f}, "
          f"std: {float(last_logits.float().std()):.6f}, "
          f"nan? {torch.isnan(last_logits).any().item()}")

# -------------------------------
# 7) Generate
# -------------------------------
prompt = "DeepSeek MoE: explain the benefits of routing tokens to experts."
inputs = tokenizer(prompt, return_tensors="pt").to(gpu)

with torch.no_grad():
    out_ids = model.generate(**inputs, **gen_kwargs)

text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
print("\n=== OUTPUT ===")
print(text)
