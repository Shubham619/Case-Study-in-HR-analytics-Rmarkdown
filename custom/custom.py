import torch
import time
import psutil
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, OffloadedCache

# ─── CONFIG ────────────────────────────────────────────────────────────────
model_name      = "openlm-research/open_llama_7b_v2"
device_str      = "cuda:0"
context_lengths = [512, 1024, 2048]
num_gen_steps   = 50

# ─── SETUP ─────────────────────────────────────────────────────────────────
device  = torch.device(device_str)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model     = (AutoModelForCausalLM
             .from_pretrained(model_name, torch_dtype=torch.float16)
             .to(device)
             .eval())
process = psutil.Process()

def measure_cpu_gib():
    return process.memory_info().rss / (1024**3)

# Load a long real‑text stream once
ds       = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
full_txt = "\n\n".join(ds["text"])
enc      = tokenizer(full_txt, return_tensors="pt")
all_ids  = enc.input_ids[0]

# Baseline CPU RSS (model weights, code, tokenizer, dataset)
baseline_cpu = measure_cpu_gib()

# We'll compare two modes:
modes = [
    ("GPU_Cache", DynamicCache()),
    ("Offload_Cache", OffloadedCache(device=device))
]

# Print header
print(
    f"{'Mode':>12s} │ {'CtxLen':>6s} │ {'TTFT(ms)':>9s} │ {'TPS':>8s} │ "
    f"{'GPU(GiB)':>9s} │ {'CPU_offload(GiB)':>16s}"
)
print("-" * 80)

for mode_name, cache in modes:
    for L in context_lengths:
        # 1) Build prompt of length L
        ctx_ids = all_ids[:L].unsqueeze(0).to(device)

        # 2) Prefill cache (writes KV into `cache`)
        with torch.no_grad():
            _ = model(ctx_ids, past_key_values=cache, use_cache=True)

        # 3) Reset GPU allocator stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        # 4) Measure CPU RSS bump ≈ KV offload size
        cpu_off = measure_cpu_gib() - baseline_cpu

        # 5) Prepare first next‑token input
        next_ids = ctx_ids[:, -1:].contiguous()

        # 6) Micro‑warmup of generation path
        with torch.no_grad():
            _ = model(next_ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()

        # 7) TTFT measurement
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            out = model(next_ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        ttft_ms = (time.time() - t0) * 1000.0

        # 8) Record peak GPU memory
        peak_gpu = torch.cuda.max_memory_reserved(device) / (1024**3)

        # 9) Steady‑state throughput
        next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(num_gen_steps):
            with torch.no_grad():
                out = model(next_ids, past_key_values=cache, use_cache=True)
            next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()
        tps = num_gen_steps / (time.time() - t0)

        # 10) Print row
        print(
            f"{mode_name:>12s} │ {L:6d} │ {ttft_ms:9.1f} │ "
            f"{tps:8.1f} │ {peak_gpu:9.2f} │ {cpu_off:16.2f}"
        )
