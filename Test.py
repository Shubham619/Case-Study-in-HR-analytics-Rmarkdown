def rehome_static_kv_cache_to_node(kv_cache, node=1):
    new_keys, new_vals = [], []
    for i, (k, v) in enumerate(zip(kv_cache.key_cache, kv_cache.value_cache)):
        # shape matches full max_cache_len
        shape = k.shape
        dtype = k.dtype

        # allocate buffer on NUMA node
        np_dtype = np.float16 if dtype in (torch.float16, torch.bfloat16) else np.float32
        nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
        ptr = libnuma.numa_alloc_onnode(nbytes, node)
        buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
        np_arr = np.frombuffer(buf, dtype=np_dtype, count=int(np.prod(shape))).reshape(shape)

        # copy full tensor (including unused zero slots)
        np.copyto(np_arr, k.cpu().numpy())

        # wrap back as torch tensor
        new_k = torch.from_numpy(np_arr).to(k.device)
        new_v = torch.from_numpy(np_arr.copy()).to(v.device)  # do same for value

        new_keys.append(new_k)
        new_vals.append(new_v)

        # update buffers
        kv_cache._buffers[f"key_cache_{i}"] = new_k
        kv_cache._buffers[f"value_cache_{i}"] = new_v

    kv_cache.key_cache = new_keys
    kv_cache.value_cache = new_vals

    return kv_cache

kv_cache = StaticCache(
    max_batch_size=inputs.input_ids.shape[0],
    max_cache_len=128,
    config=model.config
)

with torch.no_grad():
    _ = model(**inputs, past_key_values=kv_cache)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# --- DebugCache class ---
class DebugCache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k, v = super().update(key_states, value_states, layer_idx, cache_kwargs=cache_kwargs)
        print(f"[DebugCache Update] Layer {layer_idx}: added {key_states.shape[-2]} tokens, new total={k.shape[-2]}")
        return k, v

# --- Load model ---
model_id = "deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()

# --- Create your custom cache ---
kv_cache = DebugCache()

# --- Prefill ---
prompt = "DeepSeek is transforming inference efficiency."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model(**inputs, use_cache=True, past_key_values=kv_cache)

# --- Decode loop (manual) ---
next_ids = inputs["input_ids"]
for step in range(5):  # generate 5 tokens
    outputs = model(next_ids[:, -1:], use_cache=True, past_key_values=kv_cache)
    next_token = outputs.logits[:, -1:].argmax(dim=-1)
    next_ids = torch.cat([next_ids, next_token[:, None]], dim=-1)
    print(f"[Step {step+1}] Generated token ID {next_token.item()}")

# --- Decode final text ---
print("Generated text:", tokenizer.decode(next_ids[0], skip_special_tokens=import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# --- Your DebugCache class ---
class DebugCache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k, v = super().update(key_states, value_states, layer_idx, cache_kwargs=cache_kwargs)
        print(f"[DebugCache Update] Layer {layer_idx}: added {key_states.shape[-2]} tokens")
        return k, v

# --- Load model and tokenizer (as before) ---
model_id = "deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()

# --- Create your custom cache instance ---
# This is the key change. We will pass this to the `generate` method.
my_debug_cache = DebugCache()

# --- Use the `generate` method ---
prompt = "DeepSeek is transforming inference efficiency."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Now, we use the `generate` method, which is the high-level API for text generation.
# It handles the pre-fill and decode loop internally.
outputs = model.generate(
    **inputs,
    max_new_tokens=20, # How many tokens to generate
    pad_token_id=tokenizer.eos_token_id, # Or other appropriate token
    past_key_values=my_debug_cache, # Pass your custom cache here
    use_cache=True,
)

# Decode final text
print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))

import numpy as np
import torch
from transformers.cache_utils import StaticCache

class NUMAStaticCache(StaticCache):
    def __init__(self, *args, node=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.node = node

    def allocate(self, shape, dtype, device):
        # Map torch dtype → numpy dtype
        if dtype == torch.float32:
            np_dtype = np.float32
        elif dtype == torch.float16:
            np_dtype = np.float16
        else:
            np_dtype = np.float32  # fallback

        # --- inline use of your alloc_on_node ---
        np_dtype = np.dtype(np_dtype)
        nbytes = int(np.prod(shape)) * np_dtype.itemsize
        libnuma.numa_set_strict(1)
        ptr = libnuma.numa_alloc_onnode(nbytes, int(self.node))
        if not ptr:
            raise MemoryError(f"numa_alloc_onnode failed for {nbytes} bytes on node {self.node}")

        base = (ctypes.c_uint8 * nbytes).from_address(ptr)
        arr = np.frombuffer(base, dtype=np_dtype, count=int(np.prod(shape))).reshape(shape)

        # Wrap numpy array into torch tensor
        return torch.from_numpy(arr).to(dtype)




import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Load model on CPU ---
model_id = "Qwen/Qwen-7B"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=False   # ensure weights materialize (no meta tensors)
).to("cpu")

use_cache = False  # manual cache

# --- Manual generation loop ---
def manual_generate(model, tok, prompt, max_new_tokens):
    model.eval()
    input_ids = tok(prompt, return_tensors="pt").input_ids.to("cpu")

    # Model parts
    embeddings = model.transformer.wte
    h_layers = model.transformer.h
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    generated_ids = input_ids

    for _ in range(max_new_tokens):
        # --- 1. Embedding on GPU ---
        input_gpu = input_ids.to("cuda:0", non_blocking=True)
        embeddings_gpu = embeddings.to("cuda:0", non_blocking=True)
        hidden_states = embeddings_gpu(input_gpu)
        del embeddings_gpu
        torch.cuda.empty_cache()

        # --- 2. Transformer blocks (keep activations on GPU) ---
        for i, layer in enumerate(h_layers):
            layer_gpu = layer.to("cuda:0", non_blocking=True)
            with torch.no_grad():
                hidden_states = layer_gpu(hidden_states)[0]  # forward pass
            del layer_gpu
            torch.cuda.empty_cache()

        # --- 3. Final layers ---
        ln_f_gpu = ln_f.to("cuda:0", non_blocking=True)
        lm_head_gpu = lm_head.to("cuda:0", non_blocking=True)
        with torch.no_grad():
            final_hidden = ln_f_gpu(hidden_states)
            logits = lm_head_gpu(final_hidden)

        # --- 4. Sample next token ---
        next_token = torch.argmax(logits[:, -1, :], dim=-1)

        # --- 5. Drop GPU copies, keep only CPU state ---
        del ln_f_gpu, lm_head_gpu, hidden_states, logits, final_hidden
        torch.cuda.empty_cache()

        # Update inputs
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).to("cpu")], dim=-1)
        input_ids = next_token.unsqueeze(0).to("cpu")

    return generated_ids

# Example:
# out_ids = manual_generate(model, tok, "The future of AI is", 20)
# print(tok.decode(out_ids[0]))


# ---------- NUMA-based rehome of a Static KV cache ----------

import ctypes, numpy as np, torch
from typing import Tuple, List

# Load libnuma
libnuma = ctypes.CDLL("libnuma.so.1")
libnuma.numa_available.restype = ctypes.c_int
libnuma.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
libnuma.numa_alloc_onnode.restype  = ctypes.c_void_p
libnuma.numa_free.argtypes         = [ctypes.c_void_p, ctypes.c_size_t]

if libnuma.numa_available() < 0:
    raise RuntimeError("NUMA not available on this system")

_TORCH2NP = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.bfloat16: np.float16,   # safest portable storage; bfloat16 numpy dtype isn't universal
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.uint8: np.uint8,
}

def _alloc_numpy_on_node(shape: Tuple[int, ...], np_dtype: np.dtype, node: int):
    """Allocate a NumPy array backed by numa_alloc_onnode (no copy)."""
    nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
    ptr = libnuma.numa_alloc_onnode(nbytes, int(node))
    if not ptr:
        raise MemoryError(f"numa_alloc_onnode failed for {nbytes} bytes on node {node}")

    # Build a ctypes byte-buffer from the raw pointer
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
    # Create a NumPy view on that memory and reshape
    np_arr = np.frombuffer(buf, dtype=np_dtype, count=int(np.prod(shape))).reshape(shape)

    # Free function closes over size
    def _free():
        libnuma.numa_free(ptr, nbytes)

    return np_arr, ptr, _free

def _to_numpy_dtype(t_dtype: torch.dtype):
    if t_dtype not in _TORCH2NP:
        # default to float32 if unsupported
        return np.float32
    return _TORCH2NP[t_dtype]

def _ensure_list(x):
    return x if isinstance(x, list) else [x]

def _same_container_type(old, new_list: List[torch.Tensor]):
    # Return list or squeeze back to tensor to match original container type
    if isinstance(old, list):
        return new_list
    assert len(new_list) == 1
    return new_list[0]

def rehome_static_kv_cache_to_node(kv_cache, node: int = 1):
    """
    Re-allocate kv_cache.key_cache and kv_cache.value_cache on a given NUMA node.
    Replaces tensors in-place with new CPU tensors backed by NUMA memory.
    """
    numa_ptrs = []
    free_fns  = []

    def rehome_one_container(container):
        new_tensors = []
        for t in _ensure_list(container):
            if not isinstance(t, torch.Tensor):
                raise TypeError("KV cache entries must be torch.Tensor")
            # We'll keep CPU tensors; ensure a contiguous CPU source for copy
            src = t.detach().to("cpu").contiguous()
            shape = tuple(src.shape)
            np_dtype = _to_numpy_dtype(src.dtype)

            np_arr, ptr, free_fn = _alloc_numpy_on_node(shape, np_dtype, node)
            # Copy bytes from old tensor -> NUMA array (no casting)
            np.copyto(np_arr, src.numpy(), casting="no")

            # Wrap as a torch tensor without copying
            new_t = torch.from_numpy(np_arr)
            new_t.requires_grad = False

            # If original was bfloat16, upcasted to fp16 in NumPy: keep Torch dtype same as source
            if new_t.dtype != src.dtype:
                new_t = new_t.view(dtype=src.dtype)

            new_tensors.append(new_t)

            numa_ptrs.append((ptr, np_arr.nbytes))
            free_fns.append(free_fn)

        return _same_container_type(container, new_tensors)

    # --- rehome key and value caches ---
    kv_cache.key_cache   = rehome_one_container(kv_cache.key_cache)
    kv_cache.value_cache = rehome_one_container(kv_cache.value_cache)

    # (Optional) remember where we put them & how to free
    setattr(kv_cache, "_numa_node", int(node))
    setattr(kv_cache, "_numa_raw_ptrs", numa_ptrs)
    setattr(kv_cache, "free_numa_cache", lambda: [fn() for fn in free_fns])

    # If your class tracks a 'node' field, update it
    if hasattr(kv_cache, "node"):
        kv_cache.node = int(node)

    return kv_cache

# ----------------- usage -----------------
# Assuming you already built your StaticCache (kv_cache) for the model:
# kv_cache = build_static_cache_somehow(...)
# Move everything to NUMA node 1:
# rehome_static_kv_cache_to_node(kv_cache, node=1)

# Later, if you want to explicitly free those buffers (usually not necessary until process exit):
# kv_cache.free_numa_cache()

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import StaticCache

# 1. Load model & tokenizer
model_id = "gpt2"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# 2. Build a StaticCache for your model (CPU by default)
kv_cache = StaticCache(
    max_batch_size=1,
    max_cache_len=512,
    config=model.config
)

# 3. Re-home it to NUMA node 1
from numa_helpers import rehome_static_kv_cache_to_node   # your helper
rehome_static_kv_cache_to_node(kv_cache, node=1)

# 4. Use the cache in generation
inputs = tok("DeepSeek is transforming inference efficiency.", return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=50,
    past_key_values=kv_cache   # pass our NUMA-backed cache
)

print(tok.decode(out[0]))


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import StaticCache

import ctypes, numpy as np

# ---------------- NUMA helper ---------------- #
libnuma = ctypes.CDLL("libnuma.so.1")
libnuma.numa_available.restype = ctypes.c_int
libnuma.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
libnuma.numa_alloc_onnode.restype  = ctypes.c_void_p
libnuma.numa_free.argtypes         = [ctypes.c_void_p, ctypes.c_size_t]

if libnuma.numa_available() < 0:
    raise RuntimeError("NUMA not available on this system")

_TORCH2NP = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.bfloat16: np.float16,   # fallback: store as fp16, view back
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.uint8: np.uint8,
}

def _to_numpy_dtype(t_dtype: torch.dtype):
    return _TORCH2NP.get(t_dtype, np.float32)

def _alloc_numpy_on_node(shape, np_dtype, node: int):
    nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
    ptr = libnuma.numa_alloc_onnode(nbytes, int(node))
    if not ptr:
        raise MemoryError(f"numa_alloc_onnode failed for {nbytes} bytes on node {node}")
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
    np_arr = np.frombuffer(buf, dtype=np_dtype, count=int(np.prod(shape))).reshape(shape)
    def _free(): libnuma.numa_free(ptr, nbytes)
    return np_arr, ptr, _free

def _ensure_list(x):
    return x if isinstance(x, list) else [x]

def _same_container_type(old, new_list):
    if isinstance(old, list):
        return new_list
    assert len(new_list) == 1
    return new_list[0]

def rehome_static_kv_cache_to_node(kv_cache, node: int = 1):
    numa_ptrs, free_fns = [], []

    def rehome_one_container(container):
        new_tensors = []
        for t in _ensure_list(container):
            if t is None:   # leave None entries intact
                new_tensors.append(None)
                continue

            src = t.detach().to("cpu").contiguous()
            shape = tuple(src.shape)
            np_dtype = _to_numpy_dtype(src.dtype)

            np_arr, ptr, free_fn = _alloc_numpy_on_node(shape, np_dtype, node)
            np.copyto(np_arr, src.numpy(), casting="no")

            new_t = torch.from_numpy(np_arr)
            new_t.requires_grad = False
            if new_t.dtype != src.dtype:
                new_t = new_t.view(dtype=src.dtype)

            new_tensors.append(new_t)
            numa_ptrs.append((ptr, np_arr.nbytes))
            free_fns.append(free_fn)

        return _same_container_type(container, new_tensors)

    kv_cache.key_cache   = rehome_one_container(kv_cache.key_cache)
    kv_cache.value_cache = rehome_one_container(kv_cache.value_cache)

    setattr(kv_cache, "_numa_node", int(node))
    setattr(kv_cache, "_numa_raw_ptrs", numa_ptrs)
    setattr(kv_cache, "free_numa_cache", lambda: [fn() for fn in free_fns])
    return kv_cache
# ------------------------------------------------ #

# 1. Load model & tokenizer
model_id = "gpt2"   # or your larger model
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval()

# 2. Build a StaticCache
prompt = "DeepSeek is transforming inference efficiency."
inputs = tok(prompt, return_tensors="pt")

kv_cache = StaticCache(
    max_batch_size=inputs.input_ids.shape[0],
    max_cache_len=128,
    config=model.config
)

# 3. Warm-up forward pass so cache fills
with torch.no_grad():
    _ = model(**inputs, past_key_values=kv_cache)

print("Before rehome:", kv_cache.key_cache[0] is None)  # should now be False

# 4. Re-home to NUMA node 1
rehome_static_kv_cache_to_node(kv_cache, node=1)

# 5. Continue generation with NUMA-backed cache
out = model.generate(
    **inputs,
    past_key_values=kv_cache,
    max_new_tokens=50
)

print(tok.decode(out[0]))


