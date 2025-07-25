import torch
import time
import psutil
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── CONFIG ────────────────────────────────────────────────────────────────
model_name      = "openlm-research/open_llama_7b_v2"
device          = "cuda:0"
context_lengths = [512, 1024, 2048]
num_gen_steps   = 50   # steady‑state tokens to generate

# ─── SETUP ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model     = (AutoModelForCausalLM
             .from_pretrained(model_name, torch_dtype=torch.float16)
             .to(device)
             .eval())
process = psutil.Process()

def measure_mem():
    """Return (cpu_rss_GiB, gpu_reserved_GiB)."""
    cpu = process.memory_info().rss / (1024**3)
    gpu = torch.cuda.memory_reserved(device) / (1024**3)
    return cpu, gpu

# ─── PREPARE REAL CONTEXT ───────────────────────────────────────────────────
ds       = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
full_txt = "\n\n".join(ds["text"])
enc      = tokenizer(full_txt, return_tensors="pt")
all_ids  = enc.input_ids[0]  # long stream of tokens

# ─── OFFLOAD BENCHMARK ──────────────────────────────────────────────────────
print(f"{'CtxLen':>6s} │ {'TTFT(ms)':>9s} │ {'TPS':>8s} │ {'GPU(GiB)':>9s} │ {'CPU(GiB)':>9s}")
print("-" * 60)
for L in context_lengths:
    # 1) build prompt of length L
    ctx_ids   = all_ids[:L].unsqueeze(0).to(device)

    # --- warm‑up (no cache passed) ---
    with torch.no_grad():
        out  = model(ctx_ids, use_cache=True)
    past = out.past_key_values                # cache lives on GPU
    # immediately offload it to DRAM
    past = tuple((k.cpu(), v.cpu()) for k, v in past)

    # --- TTFT measurement ---
    next_ids = ctx_ids[:, -1:].contiguous()
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        # bring cache to GPU
        past_gpu = tuple((k.to(device), v.to(device)) for k, v in past)
        out      = model(next_ids, past_key_values=past_gpu, use_cache=True)
        # offload again
        past     = tuple((k.cpu(), v.cpu()) for k, v in out.past_key_values)
    torch.cuda.synchronize()
    ttft      = (time.time() - t0) * 1000.0
    cpu1, gpu1 = measure_mem()

    # --- steady‑state throughput ---
    next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(num_gen_steps):
        with torch.no_grad():
            past_gpu = tuple((k.to(device), v.to(device)) for k, v in past)
            out      = model(next_ids, past_key_values=past_gpu, use_cache=True)
            past     = tuple((k.cpu(), v.cpu()) for k, v in out.past_key_values)
        next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize()
    duration  = time.time() - t0
    tps       = num_gen_steps / duration
    cpu2, gpu2 = measure_mem()

    peak_cpu = max(cpu1, cpu2)
    peak_gpu = torch.cuda.max_memory_reserved(device) / (1024**3)

    print(f"{L:6d} │ {ttft:9.1f} │ {tps:8.1f} │ {peak_gpu:9.2f} │ {peak_cpu:9.2f}")


