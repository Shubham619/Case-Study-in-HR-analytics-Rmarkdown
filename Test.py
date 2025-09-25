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
    if dtype == torch.bfloat16: return np.float16  # Note: This loses precision
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
    Initializes a StaticCache object with the correct configuration.
    """
    cache = StaticCache(
        max_batch_size=batch_size,
        max_cache_len=target_max_len,
        config=model.config
    )
    return cache

def offload_from_warmup_to_numa(target_cache, warmup_cache, prompt_len, node_id=1):
    """
    Creates NUMA-backed tensors for the target cache and copies data from the
    warm-up cache into them.
    """
    new_keys, new_vals, free_fns = [], [], []

    # Get head_dim from the model config
    try:
        head_dim = target_cache.config.hidden_size // target_cache.config.num_attention_heads
    except AttributeError:
        raise AttributeError("Could not determine head dimension from model config. Check for `hidden_size` and `num_attention_heads`.")

    # Iterate through all layers, creating a NUMA-backed tensor for each one
    for i in range(len(warmup_cache)):
        k_src, v_src = warmup_cache[i]
        
        k_dtype = k_src.dtype
        v_dtype = v_src.dtype

        full_shape = (target_cache.max_batch_size, 
                      target_cache.config.num_attention_heads, 
                      target_cache.max_cache_len, 
                      head_dim)
        
        np_k_dtype = torch_dtype_to_numpy(k_dtype)
        np_v_dtype = torch_dtype_to_numpy(v_dtype)

        try:
            np_k, free_k = alloc_numpy_on_node(full_shape, np_k_dtype, node_id)
            np_v, free_v = alloc_numpy_on_node(full_shape, np_v_dtype, node_id)
        except MemoryError as e:
            print(f"Failed to allocate memory for layer {i} on NUMA node {node_id}: {e}")
            # Clean up previously allocated memory
            for fn in free_fns:
                fn()
            raise e

        np_k.fill(0)
        np_v.fill(0)

        # Copy data from warmup cache - ensure proper shape handling
        k_src_cpu = k_src.detach().to("cpu").contiguous().numpy()
        v_src_cpu = v_src.detach().to("cpu").contiguous().numpy()
        
        # Handle potential shape mismatches
        src_shape = k_src_cpu.shape
        if len(src_shape) == 4:  # [batch, heads, seq_len, head_dim]
            np_k[:, :, :prompt_len, :] = k_src_cpu
            np_v[:, :, :prompt_len, :] = v_src_cpu
        else:
            raise ValueError(f"Unexpected cache tensor shape: {src_shape}")
        
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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,  # Use float32 directly for CPU
        device_map=None  # Ensure we control device placement
    )
    model.to(device)
    model.eval()

    prompt = "DeepSeek is transforming inference efficiency."
    inputs = tok(prompt, return_tensors="pt").to(device)
    
    # --- Custom Generation Loop ---
    try:
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
            generated_ids = inputs.input_ids.clone()
            current_length = prompt_len
            
            for step in range(64):
                # The input for the next step is only the last generated token
                new_input_ids = generated_ids[:, -1:].clone()

                # Create proper attention_mask for the full sequence
                attention_mask = torch.ones(
                    batch_size, current_length, dtype=torch.long, device=device
                )
                
                # Position IDs should be the current position
                position_ids = torch.tensor(
                    [[current_length - 1]], dtype=torch.long, device=device
                )

                # Forward pass with the cache
                output = model(
                    input_ids=new_input_ids,
                    past_key_values=target_cache,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                )

                # Get next token
                next_token_logits = output.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                current_length += 1
                
                # Update the cache - it's updated in-place by the model
                target_cache = output.past_key_values
                
                # Break if we hit EOS token
                if next_token_id.item() == tok.eos_token_id:
                    break

        # 5. Decode the final result
        final_output = tok.decode(generated_ids[0], skip_special_tokens=True)
        print(final_output)

    finally:
        # 6. Free NUMA pages
        if 'target_cache' in locals() and hasattr(target_cache, "free_numa"):
            target_cache.free_numa()
            del target_cache
        gc.collect()
