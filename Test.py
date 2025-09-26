class CPUStorageExpert(nn.Module):
    def __init__(self, expert, gpu_device="cuda:0"):
        super().__init__()
        self.expert_cpu = expert.to("cpu")   # weights in DRAM
        self.gpu_device = gpu_device
        self.expert_gpu = None               # cached GPU copy

    def _sync_weights(self):
        if self.expert_gpu is None:
            # First time: create GPU module and load weights
            self.expert_gpu = self.expert_cpu.to_empty(device=self.gpu_device)
            self.expert_gpu.load_state_dict(self.expert_cpu.state_dict(), strict=True)
        else:
            # Refresh weights if CPU changed
            self.expert_gpu.load_state_dict(self.expert_cpu.state_dict(), strict=True)

    def forward(self, *args, **kwargs):
        # Ensure inputs are on GPU
        args = [a.to(self.gpu_device, non_blocking=True) if torch.is_tensor(a) else a
                for a in args]
        kwargs = {k: v.to(self.gpu_device, non_blocking=True) if torch.is_tensor(v) else v
                  for k, v in kwargs.items()}

        # Sync once, reuse GPU copy
        self._sync_weights()

        return self.expert_gpu(*args, **kwargs)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate.utils import infer_auto_device_map
from accelerate import init_empty_weights, dispatch_model

# -------------------------------
# Expert wrapper: store weights in DRAM, compute on GPU
# -------------------------------
class CPUStorageExpert(nn.Module):
    def __init__(self, expert, gpu_device="cuda:0"):
        super().__init__()
        self.expert_cpu = expert.to("cpu")  # keep weights in DRAM
        self.gpu_device = gpu_device

    def forward(self, *args, **kwargs):
        # inputs -> GPU
        args = [a.to(self.gpu_device, non_blocking=True) if torch.is_tensor(a) else a for a in args]
        kwargs = {k: v.to(self.gpu_device, non_blocking=True) if torch.is_tensor(v) else v
                  for k, v in kwargs.items()}
        # temp GPU copy of weights
        expert_gpu = self.expert_cpu.to(self.gpu_device, non_blocking=True)
        out = expert_gpu(*args, **kwargs)
        # free GPU copy
        del expert_gpu
        torch.cuda.empty_cache()
        return out

# -------------------------------
# 1. Load config only (no weights yet)
# -------------------------------
model_id = "deepseek-ai/deepseek-moe-16b-base"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

with init_empty_weights():   # build skeleton on "meta" device
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# -------------------------------
# 2. Infer device map (trunk/router stay GPU, experts we'll patch)
# -------------------------------
max_memory = {
    0: "20GiB",    # GPU VRAM budget
    "cpu": "200GiB"  # fallback DRAM budget
}

device_map = infer_auto_device_map(
    empty_model,
    max_memory=max_memory,
    no_split_module_classes=["Block"],  # keep transformer blocks whole
)

# Force experts to CPU for now (storage only)
for name in list(device_map.keys()):
    if "experts" in name:
        device_map[name] = "cpu"

print("🔥 Customised auto device map:")
for k, v in device_map.items():
    print(f"{k} -> {v}")

# -------------------------------
# 3. Load model with dispatch_model
# -------------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = dispatch_model(
    model,
    device_map=device_map,   # this streams weights safely
    offload_dir=None         # no disk offload
)

# -------------------------------
# 4. Patch MoE experts with CPUStorageExpert
# -------------------------------
for name, module in model.named_modules():
    if "experts" in name and isinstance(module, nn.ModuleList):
        for idx, expert in enumerate(module):
            module[idx] = CPUStorageExpert(expert, gpu_device="cuda:0")

print("✅ MoE experts now stored in DRAM but compute on GPU")

# -------------------------------
# 5. Run generation
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

prompt = "DeepSeek MoE experts stored in DRAM, compute on GPU."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------
# Expert wrapper: store in DRAM, compute on GPU
# -------------------------------
class CPUStorageExpert(nn.Module):
    def __init__(self, expert, gpu_device="cuda:0"):
        super().__init__()
        self.expert_cpu = expert.to("cpu")      # store weights in DRAM
        self.gpu_device = gpu_device

    def forward(self, *args, **kwargs):
        # move inputs to GPU
        args = [a.to(self.gpu_device, non_blocking=True) if torch.is_tensor(a) else a for a in args]
        kwargs = {k: v.to(self.gpu_device, non_blocking=True) if torch.is_tensor(v) else v
                  for k, v in kwargs.items()}

        # bring expert weights from DRAM -> GPU temporarily
        expert_gpu = self.expert_cpu.to(self.gpu_device, non_blocking=True)

        out = expert_gpu(*args, **kwargs)   # run compute on CUDA

        # free GPU copy (storage remains in CPU)
        del expert_gpu
        torch.cuda.empty_cache()

        return out

# -------------------------------
# Load tokenizer + model
# -------------------------------
model_id = "deepseek-ai/deepseek-moe-16b-base"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda:0")

# -------------------------------
# Replace all experts with wrapper
# -------------------------------
for name, module in model.named_modules():
    if "experts" in name and isinstance(module, nn.ModuleList):
        for idx, expert in enumerate(module):
            module[idx] = CPUStorageExpert(expert, gpu_device="cuda:0")

print("✅ MoE experts now stored in DRAM but computed on GPU")

# -------------------------------
# Run generation
# -------------------------------
prompt = "DeepSeek MoE experts stored in DRAM, compute on GPU."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))





import os, gc, ctypes
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- NUMA helpers ----------------
try:
    libnuma = ctypes.CDLL("libnuma.so.1")
    libnuma.numa_available.restype = ctypes.c_int
    libnuma.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
    libnuma.numa_alloc_onnode.restype  = ctypes.c_void_p
    libnuma.numa_free.argtypes         = [ctypes.c_void_p, ctypes.c_size_t]

    if libnuma.numa_available() < 0:
        raise RuntimeError("NUMA not available on this system")
except OSError:
    print("libnuma.so.1 not found. Make sure libnuma is installed on your system.")
    exit()

def torch_dtype_to_numpy(dtype):
    if dtype == torch.float32: return np.float32
    if dtype == torch.float16: return np.float16
    if dtype == torch.bfloat16: return np.float16
    if dtype == torch.int64:   return np.int64
    if dtype == torch.int32:   return np.int32
    if dtype == torch.int16:   return np.int16
    if dtype == torch.int8:    return np.int8
    if dtype == torch.uint8:   return np.uint8
    raise ValueError(f"Unsupported dtype: {dtype}")

def alloc_numpy_on_node(shape, np_dtype, numa_node):
    nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
    ptr = libnuma.numa_alloc_onnode(nbytes, int(numa_node))
    if not ptr:
        raise MemoryError(f"numa_alloc_onnode failed for {nbytes} bytes on node {numa_node}")
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
    arr = np.frombuffer(buf, dtype=np_dtype, count=int(np.prod(shape))).reshape(shape)
    
    def _free():
        libnuma.numa_free(ptr, nbytes)
        
    return arr, _free

# ---------------- Custom Cache Class ----------------
class NumaKVCache:
    def __init__(self, max_batch_size, max_cache_len):
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.key_cache = []
        self.value_cache = []
        self.free_functions = []
        self.seq_len = 0
    
    def __len__(self):
        return len(self.key_cache)
    
    def __getitem__(self, layer_idx):
        return (self.key_cache[layer_idx], self.value_cache[layer_idx])
    
    def free_numa(self):
        for fn in self.free_functions:
            fn()

def build_numa_cache_from_warmup(warmup_cache, batch_size, target_max_len, prompt_len, node_id=1):
    numa_cache = NumaKVCache(batch_size, target_max_len)
    
    if len(warmup_cache) == 0:
        raise ValueError("Warmup cache is empty")
    
    k_sample, v_sample = warmup_cache[0]
    batch_size_actual, num_heads, seq_len, head_dim = k_sample.shape
    
    print(f"Detected cache shape: batch={batch_size_actual}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")

    for i in range(len(warmup_cache)):
        k_src, v_src = warmup_cache[i]
        
        k_dtype = k_src.dtype
        v_dtype = v_src.dtype

        full_shape = (batch_size, num_heads, target_max_len, head_dim)
        
        np_k_dtype = torch_dtype_to_numpy(k_dtype)
        np_v_dtype = torch_dtype_to_numpy(v_dtype)

        try:
            np_k, free_k = alloc_numpy_on_node(full_shape, np_k_dtype, node_id)
            np_v, free_v = alloc_numpy_on_node(full_shape, np_v_dtype, node_id)
        except MemoryError as e:
            print(f"Failed to allocate memory for layer {i} on NUMA node {numa_node}: {e}")
            numa_cache.free_numa()
            raise e

        np_k.fill(0)
        np_v.fill(0)

        k_src_cpu = k_src.detach().to("cpu").contiguous().numpy()
        v_src_cpu = v_src.detach().to("cpu").contiguous().numpy()
        
        np_k[:, :, :prompt_len, :] = k_src_cpu
        np_v[:, :, :prompt_len, :] = v_src_cpu
        
        tk = torch.from_numpy(np_k)
        tv = torch.from_numpy(np_v)

        if tk.dtype != k_dtype: tk = tk.to(k_dtype)
        if tv.dtype != v_dtype: tv = tv.to(v_dtype)
        
        tk.requires_grad = False
        tv.requires_grad = False
        
        numa_cache.key_cache.append(tk)
        numa_cache.value_cache.append(tv)
        numa_cache.free_functions.extend([free_k, free_v])

    numa_cache.seq_len = prompt_len
    return numa_cache

# ---------------- Main Script ----------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float32)
    
    model_id = "deepseek-ai/deepseek-moe-16b-base"
    device = "cpu"

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()

    model.to(torch.float32)

    prompt = "DeepSeek is transforming inference efficiency."
    inputs = tok(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Step 1: Get the initial key/value pairs for the prompt
        warmup_output = model(
            input_ids=inputs.input_ids,
            use_cache=True,
        )
        
        # Manually create the initial cache tuple from the output
        warmup_cache = warmup_output.past_key_values
        
        # Step 2: Build the NUMA cache
        batch_size = inputs.input_ids.shape[0]
        prompt_len = inputs.input_ids.shape[1]
        target_max_len = 2048
        numa_node = 1
        
        numa_cache = build_numa_cache_from_warmup(
            warmup_cache=warmup_cache,
            batch_size=batch_size,
            target_max_len=target_max_len,
            prompt_len=prompt_len,
            node_id=numa_node
        )

        generated_ids = inputs.input_ids.tolist()
        
        # Get the first token
        next_token_logits = warmup_output.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        generated_ids[0].append(next_token_id)
        
        current_length = prompt_len + 1

        # Step 3: The manual generation loop
        for _ in range(63):
            # The input for the next step is only the last generated token
            new_input_ids = torch.tensor([[generated_ids[0][-1]]], dtype=torch.long).to(device)
            
            output = model(
                input_ids=new_input_ids,
                past_key_values=numa_cache,
                use_cache=True,
            )
            
            # Manually update the NumaKVCache with the new key/value pairs
            new_cache = output.past_key_values
            for layer_idx in range(len(new_cache)):
                new_k, new_v = new_cache[layer_idx]
                
                # Check for bounds to prevent errors and slice the full tensor
                if current_length < numa_cache.max_cache_len:
                    # The fix: Get the last token from the new_cache
                    numa_cache.key_cache[layer_idx][:, :, current_length, :].copy_(new_k[:, :, -1, :].unsqueeze(2))
                    numa_cache.value_cache[layer_idx][:, :, current_length, :].copy_(new_v[:, :, -1, :].unsqueeze(2))

            next_token_logits = output.logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            generated_ids[0].append(next_token_id)
            
            current_length += 1

    final_output = tok.decode(generated_ids[0])
    print(final_output)

    if hasattr(numa_cache, "free_numa"):
        numa_cache.free_numa()
        del numa_cache
        gc.collect()
