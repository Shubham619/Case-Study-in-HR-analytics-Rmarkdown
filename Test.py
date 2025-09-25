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
    """Converts a torch dtype to a numpy dtype."""
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
    """Allocates a NumPy array on a specific NUMA node using libnuma."""
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
    """Custom KV cache that mimics StaticCache but uses NUMA allocation"""
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

# ---------------- Cache building ----------------
def build_numa_cache_from_warmup(warmup_cache, batch_size, target_max_len, prompt_len, node_id=1):
    """
    Creates a NUMA-backed cache directly from warmup cache without using StaticCache
    """
    numa_cache = NumaKVCache(batch_size, target_max_len)
    
    if len(warmup_cache) == 0:
        raise ValueError("Warmup cache is empty")
    
    # Get shape information from the first layer's tensors
    k_sample, v_sample = warmup_cache[0]
    batch_size_actual, num_heads, seq_len, head_dim = k_sample.shape
    
    print(f"Detected cache shape: batch={batch_size_actual}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")

    # Iterate through all layers, creating a NUMA-backed tensor for each one
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
            print(f"Failed to allocate memory for layer {i} on NUMA node {node_id}: {e}")
            numa_cache.free_numa()
            raise e

        np_k.fill(0)
        np_v.fill(0)

        k_src_cpu = k_src.detach().to("cpu").contiguous().numpy()
        v_src_cpu = v_src.detach().to("cpu").contiguous().numpy()
        
        # Copy only the prompt portion
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

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    model.to(torch.float32)

    prompt = "DeepSeek is transforming inference efficiency."
    inputs = tok(prompt, return_tensors="pt").to(device)
    
    # --- Custom Generation Loop ---
    # 1. Get the first token's output to set up the cache
    with torch.no_grad():
        first_output = model(
            input_ids=inputs.input_ids,
            use_cache=True,
        )
        
        warmup_cache = first_output.past_key_values
        
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
        
        # The first output already generated the first token, so we update the generated IDs.
        next_token_logits = first_output.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        generated_ids[0].append(next_token_id)
        
        current_length = prompt_len + 1

        # 2. Enter the generation loop to produce the remaining tokens
        for _ in range(63):
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
                numa_cache.key_cache[layer_idx][:, :, current_length, :] = new_k
                numa_cache.value_cache[layer_idx][:, :, current_length, :] = new_v
            
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
