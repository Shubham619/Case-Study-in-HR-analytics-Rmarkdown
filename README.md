
import torch
import time
import psutil
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── CONFIG ────────────────────────────────────────────────────────────────
model_name     = "openlm-research/open_llama_7b_v2"
device         = "cuda"
context_lengths = [512, 1024, 2048]  # tokens
num_gen_steps  = 50   # steady‑state steps

# ─── LOAD MODEL + TOKENIZER ───────────────────────────────────────────────
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

# ─── PREPARE REAL CONTEXT ─────────────────────────────────────────────────
# Load the first split of WikiText‑2 and concatenate until we exceed the max length
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
full_text = "\n\n".join(ds["text"])
enc = tokenizer(full_text, return_tensors="pt")
all_ids = enc.input_ids[0]  # (N_total,)
print(f"Loaded real text with {all_ids.size(0)} tokens.")

def run_scenario(context_ids, offload_to_cpu: bool):
    torch.cuda.reset_peak_memory_stats(device)
    # Warm up
    input_ids = context_ids.unsqueeze(0).to(device)  # (1, L)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    if offload_to_cpu:
        past = tuple((k.cpu(), v.cpu()) for k, v in past)

    # Measure TTFT
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

    # Steady‑state throughput
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

# ─── MAIN BENCHMARK LOOP ──────────────────────────────────────────────────
print(f"{'CtxLen':>6s} │ {'Mode':>12s} │ {'TTFT(ms)':>9s} │ {'TPS':>8s} │ {'GPU(GiB)':>9s} │ {'CPU(GiB)':>9s}")
print("-" * 70)
for L in context_lengths:
    context_ids = all_ids[:L]

    # On‑GPU cache
    ttft, tps, gpu_mem, cpu_mem = run_scenario(context_ids, offload_to_cpu=False)
    print(f"{L:6d} │ {'GPU-cache':>12s} │ {ttft:9.1f} │ {tps:8.1f} │ {gpu_mem:9.2f} │ {cpu_mem:9.2f}")

    # Offload cache
    ttft, tps, gpu_mem, cpu_mem = run_scenario(context_ids, offload_to_cpu=True)
    print(f"{L:6d} │ {'DRAM-offload':>12s} │ {ttft:9.1f} │ {tps:8.1f} │ {gpu_mem:9.2f} │ {cpu_mem:9.2f}")
