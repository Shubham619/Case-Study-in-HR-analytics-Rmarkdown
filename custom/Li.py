import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

# ---------- Context lengths ----------
context = np.array([64,128,256,512,1024,2048,4096,8192,16384,24576])

# ---------- Prefill + Decode Latency ----------
ddr_prefill = [4.56,5.28,5.98,8.27,12.88,26.15,55.51,174.12,np.nan,np.nan]
cxl_prefill = [4.65,5.16,6.17,10.25,14.41,30.64,53.97,186.96,429.57,598.25]
ddr_decode  = [110.55,112.08,109.54,116.86,118.7,134.3,224.82,352.08,np.nan,np.nan]
cxl_decode  = [130.26,130.23,133.08,136.5,141.97,179.26,218.63,368.0,488.13,639.02]

lat_ddr = np.array(ddr_prefill) + np.array(ddr_decode)
lat_cxl = np.array(cxl_prefill) + np.array(cxl_decode)

oom_indices = [i for i, v in enumerate(lat_ddr) if np.isnan(v)]

# ---------- (a) Prefill + Decode Latency ----------
plt.figure(figsize=(6.5,4))
plt.plot(context, lat_ddr, 'o-', color='#1f77b4', label='DDR Total Latency')
plt.plot(context, lat_cxl, 's--', color='#ff7f0e', label='CXL Total Latency')
plt.xscale('log', base=2)
plt.gca().xaxis.set_major_locator(LogLocator(base=2, numticks=20))
plt.xlabel("Context Length (tokens)")
plt.ylabel("Latency (s)")
plt.title("(a) Prefill + Decode Latency vs Context Length")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
for i in oom_indices:
    plt.scatter(context[i], 0, color='red', marker='x', s=80, label='OOM' if i==oom_indices[0] else "")
    plt.text(context[i]*1.05, 5, 'OOM', color='red', fontsize=9, rotation=0)
plt.legend()
plt.tight_layout()
plt.show()

# ---------- (b) Peak RAM ----------
ddr_ram = [42,44,48,53,59,66,77,92,np.nan,np.nan]
cxl_ram = [44,46,50,55,60,70,78,86,110,132]
oom_indices_ram = [i for i, v in enumerate(ddr_ram) if np.isnan(v)]

plt.figure(figsize=(6.5,4))
plt.plot(context, ddr_ram, 'o-', color='#1f77b4', label='DDR Peak RAM')
plt.plot(context, cxl_ram, 's--', color='#ff7f0e', label='CXL Peak RAM')
plt.xscale('log', base=2)
plt.gca().xaxis.set_major_locator(LogLocator(base=2, numticks=20))
plt.xlabel("Context Length (tokens)")
plt.ylabel("Peak RAM (GB)")
plt.title("(b) Peak RAM Usage vs Context Length")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
for i in oom_indices_ram:
    plt.scatter(context[i], 40, color='red', marker='x', s=80, label='OOM' if i==oom_indices_ram[0] else "")
    plt.text(context[i]*1.05, 45, 'OOM', color='red', fontsize=9)
plt.legend()
plt.tight_layout()
plt.show()

# ---------- (c) Prefill TPOT ----------
tpot_ddr = context / np.array(ddr_prefill)
tpot_cxl = context / np.array(cxl_prefill)
oom_indices_tpot = [i for i, v in enumerate(tpot_ddr) if np.isnan(v)]

plt.figure(figsize=(6.5,4))
plt.plot(context, tpot_ddr, 'o-', color='#1f77b4', label='DDR TPOT')
plt.plot(context, tpot_cxl, 's--', color='#ff7f0e', label='CXL TPOT')
plt.xscale('log', base=2)
plt.gca().xaxis.set_major_locator(LogLocator(base=2, numticks=20))
plt.xlabel("Context Length (tokens)")
plt.ylabel("Prefill TPOT (tokens/sec)")
plt.title("(c) Prefill TPOT vs Context Length")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
for i in oom_indices_tpot:
    plt.scatter(context[i], 0, color='red', marker='x', s=80, label='OOM' if i==oom_indices_tpot[0] else "")
    plt.text(context[i]*1.05, np.nanmax(tpot_cxl)*0.05, 'OOM', color='red', fontsize=9)
plt.legend()
plt.tight_layout()
plt.show()

# ---------- (d) ROI ----------
roi = tpot_cxl / tpot_ddr
oom_indices_roi = [i for i, v in enumerate(roi) if np.isnan(v)]

plt.figure(figsize=(6.5,4))
plt.plot(context, roi, 's--', color='#9467bd', label='ROI = Offload / Baseline TPOT')
plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
plt.xscale('log', base=2)
plt.gca().xaxis.set_major_locator(LogLocator(base=2, numticks=20))
plt.xlabel("Context Length (tokens)")
plt.ylabel("ROI Ratio")
plt.title("(d) ROI vs Context Length")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
for i in oom_indices_roi:
    plt.scatter(context[i], 1, color='red', marker='x', s=80, label='OOM' if i==oom_indices_roi[0] else "")
    plt.text(context[i]*1.05, 1.1, 'OOM', color='red', fontsize=9)
plt.legend()
plt.tight_layout()
plt.show()import matplotlib.pyplot as plt
import numpy as np

# ---------------- Context lengths ----------------
context = np.array([64,128,256,512,1024,2048,4096,8192,16384])

# ---------------- (a) Prefill + Decode Latency ----------------
ddr_prefill = [4.56,5.28,5.98,8.27,12.88,26.15,55.51,174.12,np.nan]
cxl_prefill = [4.65,5.16,6.17,10.25,14.41,30.64,53.97,186.96,429.57]
ddr_decode = [110.55,112.08,109.54,116.86,118.7,134.3,224.82,352.08,np.nan]
cxl_decode = [130.26,130.23,133.08,136.5,141.97,179.26,218.63,368.0,488.13]

lat_ddr = np.array(ddr_prefill) + np.array(ddr_decode)
lat_cxl = np.array(cxl_prefill) + np.array(cxl_decode)

plt.figure(figsize=(6.5,4))
plt.plot(context, lat_ddr, 'o-', color='#1f77b4', label='DDR Total Latency')
plt.plot(context, lat_cxl, 's--', color='#ff7f0e', label='CXL Total Latency')
plt.xscale('log', base=2)
plt.xlabel("Context Length (tokens)")
plt.ylabel("Latency (s)")
plt.title("(a) Prefill + Decode Latency vs Context Length")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------- (b) Peak RAM Usage ----------------
ddr_ram = [42,44,48,53,59,66,77,92,np.nan]   # Example DDR growth pattern
cxl_ram = [44,46,50,55,60,70,78,86,110]      # Scales with CXL usage

plt.figure(figsize=(6.5,4))
plt.plot(context, ddr_ram, 'o-', color='#1f77b4', label='DDR Peak RAM')
plt.plot(context, cxl_ram, 's--', color='#ff7f0e', label='CXL Peak RAM')
plt.xscale('log', base=2)
plt.xlabel("Context Length (tokens)")
plt.ylabel("Peak RAM (GB)")
plt.title("(b) Peak RAM Usage vs Context Length")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------- (c) Prefill TPOT ----------------
# Throughput per token (tokens/sec = context_len / time)
tpot_ddr = context / np.array(ddr_prefill)
tpot_cxl = context / np.array(cxl_prefill)

plt.figure(figsize=(6.5,4))
plt.plot(context, tpot_ddr, 'o-', color='#1f77b4', label='DDR TPOT')
plt.plot(context, tpot_cxl, 's--', color='#ff7f0e', label='CXL TPOT')
plt.xscale('log', base=2)
plt.xlabel("Context Length (tokens)")
plt.ylabel("Prefill TPOT (tokens/sec)")
plt.title("(c) Prefill TPOT vs Context Length")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------- (d) ROI (Offload / Baseline TPOT) ----------------
roi = tpot_cxl / tpot_ddr

plt.figure(figsize=(6.5,4))
plt.plot(context, roi, 's--', color='#9467bd', label='ROI = Offload / Baseline TPOT')
plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
plt.xscale('log', base=2)
plt.xlabel("Context Length (tokens)")
plt.ylabel("ROI Ratio")
plt.title("(d) ROI vs Context Length")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()





#!/usr/bin/env python3
import os, mmap, ctypes, math, numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# ---------------- Config ----------------
CXL_REGION = os.environ.get("CXL_REGION", "/dev/cxl/region0")
SYSFS_SIZE = f"/sys/bus/cxl/devices/{os.path.basename(CXL_REGION)}/size"
# Choose a registration chunk that your system tolerates (try 2–8 GB).
CHUNK_GB = int(os.environ.get("CXL_CHUNK_GB", "4"))
TOUCH_MEMORY_FROM_GPU = True   # set False if you only want to map

# --------------- Helpers ----------------
def read_region_size_bytes():
    # sysfs exposes size in bytes
    with open(SYSFS_SIZE, "r") as f:
        s = f.read().strip()
    return int(s, 0) if s.startswith("0x") else int(s)

def roundup(x, align):
    return ((x + align - 1) // align) * align

# --------------- Start ------------------
total_size = read_region_size_bytes()
chunk_size = CHUNK_GB * 1024**3
print(f"[INFO] Region: {CXL_REGION}  size={total_size/1e9:.2f} GB  chunk={chunk_size/1e9:.2f} GB")

# open + mmap the WHOLE region
fd = os.open(CXL_REGION, os.O_RDWR)
# mmap length must be page-aligned; total_size should already be aligned by kernel
cxl_map = mmap.mmap(fd, total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
base_addr = ctypes.addressof(ctypes.c_char.from_buffer(cxl_map))
print(f"[INFO] User VA of whole CXL region: 0x{base_addr:x}")

# CUDA must allow mapping host memory
device = cuda.Device(0)
ctx = cuda.Context.attach()  # pycuda.autoinit already created one; attach to be safe
attr = device.get_attributes()
if not attr.get(cuda.device_attribute.CAN_MAP_HOST_MEMORY, 0):
    raise RuntimeError("GPU cannot map host memory (CAN_MAP_HOST_MEMORY = 0).")

# Prepare kernel to touch memory (optional)
mod = SourceModule(r"""
extern "C" __global__ void touch_u32(unsigned int *p, size_t words, unsigned int seed){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < words) {
        unsigned int v = (unsigned int)(i ^ seed);
        p[i] = v;
        // read back to force a load too
        if ((p[i] ^ v) == 0xFFFFFFFF) { p[i] = v; }
    }
}
""")
touch = mod.get_function("touch_u32")

# Walk the whole region in chunks: register → get device ptr → (optional) touch → unregister
registered = []
offset = 0
i_chunk = 0
try:
    while offset < total_size:
        length = min(chunk_size, total_size - offset)
        host_ptr = base_addr + offset

        # Register this window with DEVICEMAP so GPU can DMA to it
        try:
            cuda.mem_host_register(ctypes.c_void_p(host_ptr), length,
                                   cuda.host_register_flags.DEVICEMAP)
        except cuda.Error as e:
            # If this fails (memlock/driver), try halving the chunk and continue
            if length > (128 * 1024**2):
                print(f"[WARN] Register {length/1e9:.2f} GB failed ({e}); retry smaller chunk.")
                chunk_size = max(length // 2, 128 * 1024**2)
                continue
            raise

        dptr = cuda.get_device_pointer(ctypes.c_void_p(host_ptr))
        registered.append((offset, length, dptr))
        print(f"[MAP] Chunk#{i_chunk:03d} off={offset/1e9:8.2f} GB  len={length/1e9:6.2f} GB  dptr=0x{int(dptr):x}")
        i_chunk += 1
        offset += length

    print(f"[INFO] Registered {len(registered)} chunk(s), total {sum(l for _,l,_ in registered)/1e9:.2f} GB.")

    if TOUCH_MEMORY_FROM_GPU:
        # GPU-touch every chunk to prove end-to-end PCIe/CXL accessibility
        threads = 256
        for j, (off, length, dptr) in enumerate(registered):
            words = length // 4
            blocks = (words + threads - 1) // threads
            seed = np.uint32(j * 2654435761 & 0xFFFFFFFF)

            start, end = cuda.Event(), cuda.Event()
            start.record()
            touch(cuda.PseudoDevicePtr(dptr),
                  np.uint64(words),
                  seed,
                  block=(threads,1,1),
                  grid=(int(blocks),1,1))
            end.record(); end.synchronize()
            ms = start.time_till(end)
            gbps = (length / 1e9) / (ms/1e3)  # approximate write+read touch bandwidth
            print(f"[TOUCH] Chunk#{j:03d} {length/1e9:6.2f} GB in {ms:7.2f} ms  ~{gbps:6.1f} GB/s")

finally:
    # Always unregister in reverse order
    for off, length, dptr in reversed(registered):
        cuda.mem_host_unregister(ctypes.c_void_p(base_addr + off))
    cxl_map.close()
    os.close(fd)
    print("[CLEANUP] Unregistered and unmapped.")




import gc
import psutil
import torch
import time

papers_loaded = []

def test_prefill(context_len, papers_loaded, batch_size):
    # Load dummy text or actual text source (replace if you have your own `papers`)
    sample_text = "Memory-aware MoE inference using DDR and CXL " * 2000
    papers_loaded.append(sample_text[:context_len])

    # Tokenize and move to CPU
    inputs = tokenizer(papers_loaded, return_tensors="pt")

    gc.collect()
    torch.cuda.empty_cache = lambda: None  # no-op
    print(f"\nRunning prefill for batch={len(papers_loaded)}, context_len={context_len} tokens...")

    # Measure peak CPU RAM usage
    process = psutil.Process()
    rss_before = process.memory_info().rss / (1024 ** 3)  # in GB

    t0 = time.time()
    with torch.no_grad():
        _ = model(**inputs, use_cache=True)
    t1 = time.time()

    rss_after = process.memory_info().rss / (1024 ** 3)
    peak_gb = rss_after - rss_before
    print(f"Prefill done. Peak RAM usage: {peak_gb:.2f} GB, Time: {t1 - t0:.2f}s")

# ------------------------
# Sweep over context lengths
# ------------------------
for L in [2048, 4096, 8012, 16000]:
    papers_loaded = []
    try:
        for bt in range(1, 40, 4):
            test_prefill(L, papers_loaded, bt)
    except RuntimeError as e:
        print(f"[OOM] at {L} tokens -> {e}")
        gc.collect()
        continue













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
GPU_CACHE_MAX_BYTES   = 2_000_000_000      # hard ceiling for expert-cache bytes
GPU_CACHE_MAX_EXPERTS = 8                  # hard ceiling for expert count
MAX_NEW_TOKENS        = 128
SAFETY_GB             = 2.0                # minimum GB we always leave free

# ---- Ratio-based expert selection after prefill ----
EXPERT_SELECT_MODE  = "usage_share"  # "usage_share" or "count"
EXPERT_TOP_RATIO    = 0.50           # 50% of routed usage (or top 50% by count)
EXPERT_VRAM_RATIO   = 0.50           # use up to 50% of *available* headroom for experts

# ---- During decode: promotion scheduler knobs ----
FREEPOOL_THRESHOLD_GB = 1.5          # promote only if free pool (reserved-allocated) ≥ this
PROMOTE_EVERY_STEPS   = 4            # check every N decoding steps
PROMOTE_BATCH_SIZE    = 3            # move up to this many experts per check

# ==================== Load & dispatch (experts→CPU) ====================
print(f"Loading config & skeleton for {model_id} …")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

print("Inferring device_map …")
no_split = getattr(empty_model, "_no_split_modules", ["QwenBlock", "Block"])
device_map = infer_auto_device_map(empty_model, max_memory=max_memory, no_split_module_classes=no_split)

# Force ALL experts to CPU (materialized weights; not meta)
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

# ================= Expert LRU cache (your core kept) ===================
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
            # expert params are materialized on CPU because of device_map above
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
        # Count calls (prefill + decode) – we’ll freeze a snapshot after prefill
        expert_call_counts[expert_id] += 1
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

# ------------------ Build promotion plan (from prefill usage snapshot) ----------------
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

    # Target N by ratio
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

    # VRAM budget for experts
    free_budget = max(0, free_bytes_after_prefill - safety_bytes)
    bytes_budget = min(bytes_cap, int(free_budget * vram_ratio))

    # Approx per-expert bytes
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

# Apply initial caps to LRU and (optionally) prefetch a few best experts immediately
expert_cache.max_bytes = bytes_cap_final
expert_cache.max_count = count_cap_final
expert_cache.prefetch_experts(top_keep_ids)

print("\n[Ratio Plan]")
for k, v in plan_stats.items(): print(f"  {k}: {v}")
print(f"  selected_eids: {top_keep_ids}")
print(f"  caps → bytes={bytes_cap_final/1e9:.2f} GB  count={count_cap_final}")

# Build a promotion queue = ranked experts we *haven't* loaded yet
promotion_queue = [e for e in sorted_eids if e not in top_keep_ids]

# ================= Helpers for decode-time promotions ==================
def free_pool_bytes():
    # Immediate headroom inside the CUDA caching allocator
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
    # Grab next K experts and prefetch; LRU will evict if needed
    batch = []
    while promotion_queue and len(batch) < PROMOTE_BATCH_SIZE:
        eid_ = promotion_queue.pop(0)
        if eid_ in cpu_expert_registry:  # sanity
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

    # Every few steps, try to promote a few top experts if sliding-window freed memory
    if (step + 1) % PROMOTE_EVERY_STEPS == 0:
        promote_some_experts()

    if (step + 1) % 16 == 0:
        torch.cuda.synchronize()
        print(f"after {step+1:4d} decode tokens: mem_alloc={gb(torch.cuda.memory_allocated()):.2f} GB | pool_free={gb(free_pool_bytes()):.2f} GB")

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
MAX_NEW_TOKENS = 128
SAFETY_GB = 2.0

# ---- New knobs: ratio-based expert selection ----
EXPERT_SELECT_MODE  = "usage_share"  # "usage_share" or "count"
EXPERT_TOP_RATIO    = 0.50           # 50% of routed usage (or top 50% by count)
EXPERT_VRAM_RATIO   = 0.50           # 50% of free VRAM (after prefill) given to experts

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
            try: dev = next(expert.parameters()).device.type
            except StopIteration: dev = cpu_device
            if dev == cpu_device:
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
# Rank by prefill usage
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

    # Target N by ratio
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

    # VRAM budget for experts
    free_budget = max(0, free_bytes_after_prefill - safety_bytes)
    bytes_budget = min(bytes_cap, int(free_budget * vram_ratio))

    # Approx per-expert bytes
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

# ================================ Decode (reuse same KV) =============================
cur_ids = inputs["input_ids"]
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
    if (step+1) % 16 == 0:
        torch.cuda.synchronize()
        print(f"after {step+1:4d} decode tokens: mem_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

torch.cuda.synchronize()
print(f"\n[Decode] {MAX_NEW_TOKENS} tokens in {time.time()-start:.2f}s  |  final_alloc={gb(torch.cuda.memory_allocated()):.2f} GB")

# Output
for i in range(B):
    print(f"\n--- OUTPUT {i} ---\n{tokenizer.decode(cur_ids[i], skip_special_tokens=True)}")




# List of expert indices sorted by importance (high → low)
top_experts = [eid for eid, _ in top_by_usage]   # or top_by_score

def move_expert_to_gpu(model, layer_idx, expert_idx, device="cuda:0"):
    expert = model.model.layers[layer_idx].mlp.experts[expert_idx]
    expert.to(device)
    torch.cuda.synchronize()
    print(f"--> Expert {expert_idx} in layer {layer_idx} moved to {device}")


def get_free_mem():
    alloc = torch.cuda.memory_allocated()
    reserv = torch.cuda.memory_reserved()
    return reserv - alloc  # how much allocator can give without new malloc


moved_experts = set()

for step in range(max_new_tokens):
    outputs = model(**inputs, past_key_values=cache, use_cache=True)

    free_now = get_free_mem()
    print(f"Step {step} | Free in pool: {free_now/1e9:.2f} GB")

    # If memory ≥ 2 GB free and we still have experts to move
    if free_now > 2 * 1024**3:
        # Move up to 5 of the most important experts
        count = 0
        for layer_idx, layer in enumerate(model.model.layers):
            for expert_idx in top_experts:
                key = (layer_idx, expert_idx)
                if key not in moved_experts:
                    move_expert_to_gpu(model, layer_idx, expert_idx)
                    moved_experts.add(key)
                    count += 1
                    if count >= 5:   # only move 5 this step
                        break
            if count >= 5:
                break
