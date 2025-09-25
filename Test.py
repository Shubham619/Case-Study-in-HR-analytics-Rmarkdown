import os, gc, ctypes
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import StaticCache

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

# ---------------- Cache building ----------------
def build_target_cache_on_numa(model, batch_size, target_max_len):
    """
    Initializes a StaticCache object with the correct configuration and
    explicitly sets the config attribute to avoid errors.
    """
    cache = StaticCache(
        max_batch_size=batch_size,
        max_cache_len=target_max_len,
        config=model.config
    )
    # The fix: Manually and explicitly assign the model config to the cache object
    cache.config = model.config
    
    return cache

def offload_from_warmup_to_numa(target_cache, warmup_cache, prompt_len, node_id=1):
    """
    Creates NUMA-backed tensors for the target cache and copies data from the
    warm-up cache into them.
    """
    new_keys, new_vals, free_fns = [], [], []

    # Get dimensions directly from the warmup cache tensors instead of config
    if len(warmup_cache) == 0:
        raise ValueError("Warmup cache is empty")
    
    # Get shape information from the first layer's tensors
    k_sample, v_sample = warmup_cache[0]
    batch_size, num_heads, seq_len, head_dim = k_sample.shape
    
    print(f"Detected cache shape: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")

    # Iterate through all layers, creating a NUMA-backed tensor for each one
    for i in range(len(warmup_cache)):
        k_src, v_src = warmup_cache[i]
        
        k_dtype = k_src.dtype
        v_dtype = v_src.dtype

        full_shape = (target_cache.max_batch_size, 
                      num_heads, 
                      target_cache.max_cache_len, 
                      head_dim)
        
        np_k_dtype = torch_dtype_to_numpy(k_dtype)
        np_v_dtype = torch_dtype_to_numpy(v_dtype)

        try:
            np_k, free_k = alloc_numpy_on_node(full_shape, np_k_dtype, node_id)
            np_v, free_v = alloc_numpy_on_node(full_shape, np_v_dtype, node_id)
        except MemoryError as e:
            print(f"Failed to allocate memory for layer {i} on NUMA node {node_id}: {e}")
            raise e

        np_k.fill(0)
        np_v.fill(0)

        # Copy data from warmup cache
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
        
        new_keys.append(tk)
        new_vals.append(tv)

        free_fns.extend([free_k, free_v])

    target_cache.key_cache = new_keys
    target_cache.value_cache = new_vals
    target_cache.free_numa = lambda: [fn() for fn in free_fns]

    return target_cache

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

    # The fix: Explicitly cast the entire model to float32 for CPU operations
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
        
        # This gives us the correctly populated cache tensors from the model's forward pass
        warmup_cache = first_output.past_key_values
        
        # 2. Set up the NUMA-backed cache
        batch_size = inputs.input_ids.shape[0]
        prompt_len = inputs.input_ids.shape[1]
        target_max_len = 2048
        numa_node = 1
        
        target_cache = build_target_cache_on_numa(model, batch_size, target_max_len)
        
        # 3. Offload the cache to NUMA
        target_cache = offload_from_warmup_to_numa(
            target_cache=target_cache,
            warmup_cache=warmup_cache,
            prompt_len=prompt_len,
            node_id=numa_node
        )

        # 4. Enter the generation loop
        generated_ids = inputs.input_ids.tolist()
        for _ in range(64):
            # The input for the next step is only the last generated token
            new_input_ids = torch.tensor([[generated_ids[0][-1]]], dtype=torch.long).to(device)
            current_length = len(generated_ids[0])

            # Manually create attention_mask and position_ids for the new token
            attention_mask = torch.ones(
                1, current_length + 1, dtype=torch.long, device=device
            )
            position_ids = torch.tensor(
                [[current_length]], dtype=torch.long, device=device
            )

            # Pass the manually created tensors to the forward pass
            output = model(
                input_ids=new_input_ids,
                past_key_values=target_cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

            next_token_logits = output.logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            generated_ids[0].append(next_token_id)

            # The model's forward pass automatically updates the `target_cache` in-place
            # because we passed it by reference. No need to re-assign.

    # 5. Decode the final result
    final_output = tok.decode(generated_ids[0])
    print(final_output)

    # 6. Free NUMA pages
    if hasattr(target_cache, "free_numa"):
        target_cache.free_numa()
        del target_cache
        gc.collect()
