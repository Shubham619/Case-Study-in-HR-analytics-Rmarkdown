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


