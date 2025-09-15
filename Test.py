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




