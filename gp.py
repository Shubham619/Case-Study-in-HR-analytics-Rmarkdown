import time
import requests
import uuid
import random
from collections import defaultdict

API_URL = "http://localhost:8000/v1/completions"

# -------------------------
#  Dataset: arXiv-like text
# -------------------------
ARXIV_TEXTS = [
    "In this paper, we propose a novel transformer architecture that improves efficiency "
    "by introducing block-sparse attention mechanisms, validated on large-scale datasets "
    "with empirical results across NLP benchmarks. " * 2,
    "We study the effects of quantization and low-rank adaptation on large language models, "
    "with results demonstrating improvements in memory footprint and inference latency. " * 2,
    "This article explores distributed training strategies across GPU and CPU heterogeneous "
    "clusters, emphasizing memory-aware scheduling for scaling to trillion-parameter models. " * 2,
    "We present a systematic evaluation of memory offloading techniques for attention layers, "
    "comparing GPU VRAM only, hybrid VRAM+CPU DRAM, and quantized setups. " * 2,
]

def sample_arxiv_prompts(n: int, min_tokens=1500, max_tokens=2500):
    """Generate n long prompts that look like arXiv paper content."""
    prompts = []
    for _ in range(n):
        base = random.choice(ARXIV_TEXTS)
        repeat_factor = random.randint(min_tokens // 50, max_tokens // 50)
        prompts.append(base * repeat_factor)
    return prompts


# -------------------------
#  Load generator
# -------------------------
def run_batch(n_users, prompts, max_new_tokens=256, repeats=1):
    """Run multiple users sequentially and collect timing stats."""
    stats = {"ttft": [], "latency_per_token": [], "throughput": []}

    for r in range(repeats):
        for i, prompt in enumerate(prompts[:n_users]):
            user_id = f"user{i}"
            session_id = f"{user_id}-{uuid.uuid4().hex[:6]}"

            payload = {
                "model": "tiiuae/falcon-7b",
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "session_id": session_id,
                "temperature": 0.0,
            }

            start_time = time.time()
            resp = requests.post(API_URL, json=payload)
            resp.raise_for_status()
            result = resp.json()
            end_time = time.time()
            elapsed = end_time - start_time

            usage = result.get("usage", {})
            total_tokens = usage.get("total_tokens", max_new_tokens)

            # First token latency approx
            ttft = elapsed / total_tokens if total_tokens > 0 else elapsed
            per_token = elapsed / max_new_tokens
            throughput = total_tokens / elapsed if elapsed > 0 else 0

            stats["ttft"].append(ttft)
            stats["latency_per_token"].append(per_token)
            stats["throughput"].append(throughput)

    return {k: sum(v)/len(v) if v else 0 for k, v in stats.items()}


# -------------------------
#  Metrics from vLLM server
# -------------------------
def get_vllm_metrics(url="http://localhost:8000/metrics"):
    try:
        r = requests.get(url, timeout=5)
        metrics = {}
        for line in r.text.splitlines():
            if "kv_cache_usage_bytes" in line:
                parts = line.split(" ")
                if len(parts) == 2:
                    metrics["kv_cache_usage_bytes"] = float(parts[1]) / (1024**2)
            if "vllm_scheduler_running_requests" in line:
                parts = line.split(" ")
                if len(parts) == 2:
                    metrics["running_requests"] = int(float(parts[1]))
        return metrics
    except Exception as e:
        return {"error": str(e)}


# -------------------------
#  Experiment Scenarios
# -------------------------
def baseline_experiment():
    print("\n=== Baseline: VRAM only (24GB) ===")
    prompts = sample_arxiv_prompts(8, min_tokens=50, max_tokens=512)
    stats = run_batch(n_users=4, prompts=prompts, max_new_tokens=256, repeats=2)
    print("[BASELINE]", stats)
    print("Metrics:", get_vllm_metrics())

def cpu_offload_experiment():
    print("\n=== Optimized: VRAM + DDR offload (23GB DDR) ===")
    prompts = sample_arxiv_prompts(8, min_tokens=150, max_tokens=412)
    stats = run_batch(n_users=8, prompts=prompts, max_new_tokens=256, repeats=2)
    print("[CPU OFFLOAD]", stats)
    print("Metrics:", get_vllm_metrics())

def quantization_experiment():
    print("\n=== Quantized model: 12GB VRAM + 12GB VRAM for KV ===")
    prompts = sample_arxiv_prompts(8, min_tokens=150, max_tokens=412)
    stats = run_batch(n_users=6, prompts=prompts, max_new_tokens=256, repeats=2)
    print("[QUANTIZED]", stats)
    print("Metrics:", get_vllm_metrics())


# -------------------------
#  Run All
# -------------------------
if __name__ == "__main__":
    baseline_experiment()
    cpu_offload_experiment()
    quantization_experiment()
