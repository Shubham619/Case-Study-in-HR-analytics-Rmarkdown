python3 setup.py install -- \
  -DCMAKE_C_COMPILER=$(which gcc) \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
  -DUSE_CUDA=ON \
  -DUSE_SYSTEM_NVTX=ON \
  -DCMAKE_BUILD_TYPE=Release

import torch
import time
import psutil
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name     = "openlm-research/open_llama_7b_v2"
device         = "cuda"
context_lengths = [512, 1024, 2048]  # tokens
num_gen_steps  = 50   # steady‑state steps


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model     = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

process = psutil.Process()

def measure_mem():
    """Return (cpu_rss_GiB, gpu_reserved_GiB)."""
    cpu = process.memory_info().rss / (1024**3)
    gpu = torch.cuda.memory_reserved(device) / (1024**3)
    return cpu, gpu


ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
full_text = "\n\n".join(ds["text"])
enc = tokenizer(full_text, return_tensors="pt")
all_ids = enc.input_ids[0]  # (N_total,)
print(f"Loaded real text with {all_ids.size(0)} tokens.")

def run_scenario(context_ids, offload_to_cpu: bool):
    torch.cuda.reset_peak_memory_stats(device)

    input_ids = context_ids.unsqueeze(0).to(device)  # (1, L)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    if offload_to_cpu:
        past = tuple((k.cpu(), v.cpu()) for k, v in past)

    next_ids = input_ids[:, -1:].contiguous()
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        if offload_to_cpu:
            pg = tuple((k.cuda(), v.cuda()) for k, v in past)
            out = model(next_ids, past_key_values=pg, use_cache=True)
            past = tuple((k.cpu(), v.cpu()) for k, v in out.past_key_values)
        else:
            out = model(next_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
    torch.cuda.synchronize()
    ttft = (time.time() - t0) * 1000.0
    cpu1, gpu1 = measure_mem()

    next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(num_gen_steps):
        with torch.no_grad():
            if offload_to_cpu:
                pg = tuple((k.cuda(), v.cuda()) for k, v in past)
                out = model(next_ids, past_key_values=pg, use_cache=True)
                past = tuple((k.cpu(), v.cpu()) for k, v in out.past_key_values)
            else:
                out = model(next_ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
        next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize()
    duration = time.time() - t0
    throughput = num_gen_steps / duration
    cpu2, gpu2 = measure_mem()

    peak_cpu = max(cpu1, cpu2)
    peak_gpu = torch.cuda.max_memory_reserved(device) / (1024**3)

    return ttft, throughput, peak_gpu, peak_cpu

print(f"{'CtxLen':>6s} │ {'Mode':>12s} │ {'TTFT(ms)':>9s} │ {'TPS':>8s} │ {'GPU(GiB)':>9s} │ {'CPU(GiB)':>9s}")
print("-" * 70)
for L in context_lengths:
    context_ids = all_ids[:L]

    ttft, tps, gpu_mem, cpu_mem = run_scenario(context_ids, offload_to_cpu=False)
    print(f"{L:6d} │ {'GPU-cache':>12s} │ {ttft:9.1f} │ {tps:8.1f} │ {gpu_mem:9.2f} │ {cpu_mem:9.2f}")

    ttft, tps, gpu_mem, cpu_mem = run_scenario(context_ids, offload_to_cpu=True)
    print(f"{L:6d} │ {'DRAM-offload':>12s} │ {ttft:9.1f} │ {tps:8.1f} │ {gpu_mem:9.2f} │ {cpu_mem:9.2f}")





# 1) locate the first PtiConfig.cmake under ~/local (or anywhere in $HOME)
PTI_CONFIG=$(find $HOME -type f -iname PtiConfig.cmake | head -n1)
[ -z "$PTI_CONFIG" ] && { echo "❌ PtiConfig.cmake not found – build/install Pti first"; exit 1; }

# 2) derive the install prefix (two levels up from the file)
PTI_PREFIX=$(dirname $(dirname "$PTI_CONFIG"))

# 3) one-line install with GPU
CUDA_HOME=/usr/local/cuda \
CUDACXX=$CUDA_HOME/bin/nvcc \
CMAKE_PREFIX_PATH="$CUDA_HOME:$PTI_PREFIX:$CMAKE_PREFIX_PATH" \
python3 setup.py install -- \
  -DUSE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
# 1) find your PTI install prefix (if you have PtiConfig.cmake anywhere under $HOME)
PTI_PREFIX=$(dirname "$(dirname "$(find $HOME -type f -iname PtiConfig.cmake 2>/dev/null | head -n1)")")

# 2) set CUDA and NVCC
CUDA_HOME=/usr/local/cuda
CUDACXX=$CUDA_HOME/bin/nvcc

# 3) build & install in one shot with GPU
CMAKE_PREFIX_PATH="$CUDA_HOME${PTI_PREFIX:+:$PTI_PREFIX}:$CMAKE_PREFIX_PATH" \
python3 setup.py install -- \
  -DUSE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
sudo apt update
sudo apt install -y \
  build-essential cmake git curl \
  python3-dev python3-pip python3-venv \
  libopenblas-dev libblas-dev liblapack-dev \
  ninja-build libgflags-dev libgoogle-glog-dev \
  libomp-dev
cd ~/src/pytorch
# clean up any prior build
rm -rf build/

# set up CUDA and optional Pti prefix (if you have PTIConfig.cmake installed)
CUDA_HOME=/usr/local/cuda
CUDACXX=$CUDA_HOME/bin/nvcc

# (optional) auto-discover your Pti install under $HOME
PTI_CFG=$(find $HOME -type f -iname PtiConfig.cmake 2>/dev/null | head -n1)
if [ -n "$PTI_CFG" ]; then
  PTI_PREFIX=$(dirname "$(dirname "$PTI_CFG")")
  CMAKE_PREFIX_PATH="$CUDA_HOME:$PTI_PREFIX:$CMAKE_PREFIX_PATH"
else
  CMAKE_PREFIX_PATH="$CUDA_HOME:$CMAKE_PREFIX_PATH"
fi

# final one-liner to build & install
CUDACXX=$CUDACXX \
CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
python3 setup.py clean --all install -- \
  -DUSE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

