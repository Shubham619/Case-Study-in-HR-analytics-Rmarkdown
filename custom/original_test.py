#!/usr/bin/env python3
# CPU-only, no pcm-memory. Includes 32k ctx. Mimics paper logic:
# - Two setups: baseline (DDR: KV+MoE) vs offload (DDR: KV, CXL: MoE)
# - Experiments per setup: Prefill, TTFT (new_token=1), Throughput (steady TPS)
# - Tracks: time, peak RAM, CPU util
# - Evaluates SLO (TTFT <= 0.400s) and finds max stable batch under SLO
# - Writes a tidy CSV for plotting

import os, gc, time, random, threading, subprocess
import psutil
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from attach_numa_backing_nofetch import offload_experts_to_numa_inplace

# =========================
# CONFIG
# =========================
MODEL_NAME = "./deepseek_moe_16b_base"     # <- your model path
DEVICE = torch.device("cpu")
CONTEXT_LIST = [2048, 4096, 8192, 16384, 32768]  # includes 32k
BATCH_LIST   = [1, 2, 4]                    # used for SLO sweep (can extend)
MAX_NEW_TOKENS_TTFT = 1
MAX_NEW_TOKENS_TPS  = 256
TTFT_SLO_SEC = 0.400                        # 400 ms SLO like paper
NUMA_NODE = 1                                # CXL node for your offload helper
RESULTS_CSV = "ddr_cxl_results_no_pcm.csv"

# =========================
# MONITORS (your style)
# =========================
class MemMonitor:
    def __init__(self, interval=0.1):
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = False
        self._thread = None
    def _track(self):
        while not self._stop:
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)
    def start(self):
        self._stop = False
        self._thread = threading.Thread(target=self._track, daemon=True)
        self._thread.start()
    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join()
        return self.peak / 1e9  # GB

class CPUUtilMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self._stop = False
        self._thread = None
        self.util = []
    def _track(self):
        while not self._stop:
            self.util.append(psutil.cpu_percent(interval=None))
            time.sleep(self.interval)
    def start(self):
        self._stop = False
        self._thread = threading.Thread(target=self._track, daemon=True)
        self._thread.start()
    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join()
        return sum(self.util) / len(self.util) if self.util else 0.0

# =========================
# SYSTEM HELPERS
# =========================
def clear_linux_caches():
    # requires sudo; best-effort if not available
    try:
        subprocess.run(["sudo", "bash", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=False)
    except Exception:
        pass

def gc_full():
    gc.collect()
    gc.collect()

# =========================
# DATA / MODEL
# =========================
def load_data_and_model():
    papers = load_dataset("scientific_papers", "arxiv", split="test")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, truncation=True, padding=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map={"": "cpu"}, trust_remote_code=True
    ).eval()
    return papers, tok, model

def make_inputs(tokenizer, dataset, context_len, batch_size=1):
    # pull articles, slice to approx token length proxy by chars (simple)
    samples = random.sample(sorted(dataset["article"]), batch_size)
    samples = [s[:context_len] for s in samples]
    return tokenizer(samples, return_tensors="pt", padding=True)

# =========================
# MEASUREMENT PRIMITIVES
# =========================
@torch.no_grad()
def run_prefill(model, inputs):
    mm, cm = MemMonitor(0.1), CPUUtilMonitor(0.2)
    mm.start(); cm.start()
    t0 = time.time()
    ok = True
    try:
        _ = model(**inputs, use_cache=True)  # build KV to mimic serving prefill
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): ok = False
        else: raise
    t1 = time.time()
    peak_gb = mm.stop()
    cpu_avg = cm.stop()
    return ok, (t1 - t0), peak_gb, cpu_avg

@torch.no_grad()
def run_ttft(model, inputs):
    mm, cm = MemMonitor(0.1), CPUUtilMonitor(0.2)
    mm.start(); cm.start()
    t0 = time.time()
    ok = True
    try:
        _ = model.generate(**inputs, use_cache=True, max_new_tokens=MAX_NEW_TOKENS_TTFT, do_sample=False)
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): ok = False
        else: raise
    t1 = time.time()
    peak_gb = mm.stop()
    cpu_avg = cm.stop()
    return ok, (t1 - t0), peak_gb, cpu_avg

@torch.no_grad()
def run_throughput(model, inputs):
    # 1) TTFT for steady calc
    t0 = time.time()
    _ = model.generate(**inputs, use_cache=True, max_new_tokens=1, do_sample=False)
    ttft_s = time.time() - t0

    # 2) Full decode for throughput
    mm, cm = MemMonitor(0.1), CPUUtilMonitor(0.2)
    mm.start(); cm.start()
    t0 = time.time()
    _ = model.generate(**inputs, use_cache=True, max_new_tokens=MAX_NEW_TOKENS_TPS, do_sample=False)
    t1 = time.time()
    peak_gb = mm.stop()
    cpu_avg = cm.stop()

    total = t1 - t0
    tps_total  = MAX_NEW_TOKENS_TPS / max(total, 1e-6)
    tps_steady = (MAX_NEW_TOKENS_TPS - 1) / max(total - ttft_s, 1e-6)
    return True, ttft_s, total, tps_total, tps_steady, peak_gb, cpu_avg

# =========================
# EXPERIMENT LOGIC (paper-like)
# =========================
def max_stable_batch_under_slo(model, tok, ds, context_len, batches=BATCH_LIST, slo_s=TTFT_SLO_SEC):
    """Find largest batch whose TTFT <= SLO (paper’s serving logic)."""
    best = None
    for b in batches:
        inputs = make_inputs(tok, ds, context_len, batch_size=b)
        ok, ttft_s, _, _ = run_ttft(model, inputs)
        if not ok: break
        if ttft_s <= slo_s:
            best = (b, ttft_s)
        else:
            # beyond SLO; stop increasing
            break
    return best  # (batch, ttft_s) or None

def run_setup(setup, papers, tok, model):
    """Run full suite for a given memory setup."""
    results = []
    print(f"\n=== {setup.upper()} ===")
    # Apply placement policy
    if setup == "offload":
        offload_experts_to_numa_inplace(model, numa_node=NUMA_NODE)
    # Sweep contexts
    for ctx in CONTEXT_LIST:
        clear_linux_caches(); gc_full()
        inputs = make_inputs(tok, papers, ctx, batch_size=1)

        # Exp-1: Prefill
        ok, t_prefill, pk_prefill, cpu_prefill = run_prefill(model, inputs)
        results.append(dict(setup=setup, exp="prefill", context=ctx, batch=1,
                            ttft_s=t_prefill, tps_total=None, tps_steady=None,
                            peak_ram_gb=pk_prefill, cpu_util=cpu_prefill, ok=ok))
        print(f"{setup}|ctx={ctx}|prefill: time={t_prefill:.3f}s peak={pk_prefill:.2f}GB cpu={cpu_prefill:.1f}% ok={ok}")
        if not ok:
            # capacity wall reached; stop further ctx
            break

        # Exp-2: TTFT (batch=1)
        ok, t_ttft, pk_ttft, cpu_ttft = run_ttft(model, inputs)
        results.append(dict(setup=setup, exp="ttft", context=ctx, batch=1,
                            ttft_s=t_ttft, tps_total=None, tps_steady=None,
                            peak_ram_gb=pk_ttft, cpu_util=cpu_ttft, ok=ok))
        print(f"{setup}|ctx={ctx}|ttft  : time={t_ttft:.3f}s peak={pk_ttft:.2f}GB cpu={cpu_ttft:.1f}% ok={ok}")

        # Exp-3: Throughput (batch=1)
        ok, ttft_hint, total_time, tps_total, tps_steady, pk_tps, cpu_tps = run_throughput(model, inputs)
        results.append(dict(setup=setup, exp="throughput", context=ctx, batch=1,
                            ttft_s=ttft_hint, tps_total=tps_total, tps_steady=tps_steady,
                            peak_ram_gb=pk_tps, cpu_util=cpu_tps, ok=ok))
        print(f"{setup}|ctx={ctx}|tput  : tps_total={tps_total:.2f} tps_steady={tps_steady:.2f} peak={pk_tps:.2f}GB cpu={cpu_tps:.1f}%")

        # Paper-like SLO sweep: max batch stably meeting TTFT SLO
        best = max_stable_batch_under_slo(model, tok, papers, ctx, batches=BATCH_LIST, slo_s=TTFT_SLO_SEC)
        if best:
            b_est, ttft_b = best
            results.append(dict(setup=setup, exp="max_batch_slo", context=ctx, batch=b_est,
                                ttft_s=ttft_b, tps_total=None, tps_steady=None,
                                peak_ram_gb=None, cpu_util=None, ok=True))
            print(f"{setup}|ctx={ctx}|SLO  : max_batch={b_est} (TTFT={ttft_b:.3f}s <= {TTFT_SLO_SEC:.3f}s)")
        else:
            results.append(dict(setup=setup, exp="max_batch_slo", context=ctx, batch=None,
                                ttft_s=None, tps_total=None, tps_steady=None,
                                peak_ram_gb=None, cpu_util=None, ok=False))
            print(f"{setup}|ctx={ctx}|SLO  : none within {TTFT_SLO_SEC*1000:.0f}ms")
    return results

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Load once
    papers, tok, model = load_data_and_model()

    all_rows = []
    for setup in ["baseline", "offload"]:
        rows = run_setup(setup, papers, tok, model)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved: {RESULTS_CSV}\n")
    print(df)
