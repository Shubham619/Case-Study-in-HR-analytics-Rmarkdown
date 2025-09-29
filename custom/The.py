# ------------------ Ratio-based plan for which experts to pin ------------------
# Add these knobs near your config:
EXPERT_SELECT_MODE  = "usage_share"  # "usage_share" or "count"
EXPERT_TOP_RATIO    = 0.50           # 0.50 → keep experts covering 50% usage (or top 50% by count)
EXPERT_VRAM_RATIO   = 0.50           # fraction of free VRAM (after prefill) to devote to experts

def probe_expert_bytes(eid_probe:int) -> int:
    cpu_mod = cpu_expert_registry[eid_probe]
    cls = cpu_mod.__class__
    if hasattr(cpu_mod, "config"):
        tmp = cls(config=cpu_mod.config).to(gpu_device, dtype=dtype)
    else:
        import copy as _copy
        tmp = _copy.deepcopy(cpu_mod).to(gpu_device, dtype=dtype)
    nbytes = module_param_bytes(tmp)
    del tmp
    torch.cuda.synchronize()
    return nbytes

def plan_expert_selection_by_ratio(
    call_counts: dict[int,int],
    free_bytes_after_prefill: int,
    safety_bytes: int,
    count_cap: int,
    bytes_cap: int,
    mode: str,
    top_ratio: float,
    vram_ratio: float,
) -> tuple[list[int], int, int, dict]:
    """
    Decide which experts to keep on GPU by ratio, bounded by bytes & count caps.
    Returns: (top_keep_ids, max_bytes_final, max_count_final, stats)
    """
    if not call_counts:
        return [], 0, 0, {"reason": "no_calls"}

    # Sort by frequency (desc)
    sorted_eids = sorted(call_counts.keys(), key=lambda k: call_counts[k], reverse=True)
    total_calls = sum(call_counts.values())

    # Target by ratio:
    if mode == "usage_share":
        target_count = 0
        cum = 0
        for eid in sorted_eids:
            cum += call_counts[eid]
            target_count += 1
            if cum / max(1, total_calls) >= top_ratio:
                break
    elif mode == "count":
        target_count = max(1, int((len(sorted_eids) * top_ratio + 0.999)))  # ceil
    else:
        raise ValueError("EXPERT_SELECT_MODE must be 'usage_share' or 'count'")

    # Bytes budget = min(user bytes cap, chosen fraction of free bytes)
    free_budget = max(0, free_bytes_after_prefill - safety_bytes)
    bytes_budget = min(bytes_cap, int(free_budget * vram_ratio))

    # Probe approximate per-expert size (assume fairly uniform experts)
    per_expert_bytes = probe_expert_bytes(sorted_eids[0])

    # How many fit by bytes?
    fit_by_bytes = 0 if per_expert_bytes == 0 else (bytes_budget // per_expert_bytes)

    # Final keep count = bounded by target ratio, bytes, and count cap
    final_keep = max(0, min(target_count, fit_by_bytes, count_cap))

    top_keep_ids = sorted_eids[:final_keep]
    max_bytes_final = min(bytes_cap, per_expert_bytes * final_keep)
    max_count_final = final_keep

    stats = {
        "mode": mode,
        "ratio": top_ratio,
        "vram_ratio": vram_ratio,
        "total_experts": len(sorted_eids),
        "total_calls": total_calls,
        "target_count": target_count,
        "per_expert_bytes_gb": per_expert_bytes / 1e9,
        "bytes_budget_gb": bytes_budget / 1e9,
        "fit_by_bytes": int(fit_by_bytes),
        "final_keep": final_keep,
    }
    return top_keep_ids, max_bytes_final, max_count_final, stats

# -------- Use it right after your ONE prefill (you already computed free_bytes) --------
safety = int(SAFETY_GB * 1e9)

top_keep_ids, bytes_cap_final, count_cap_final, plan_stats = plan_expert_selection_by_ratio(
    call_counts=expert_call_counts,
    free_bytes_after_prefill=free_bytes,
    safety_bytes=safety,
    count_cap=GPU_CACHE_MAX_EXPERTS,
    bytes_cap=GPU_CACHE_MAX_BYTES,
    mode=EXPERT_SELECT_MODE,
    top_ratio=EXPERT_TOP_RATIO,
    vram_ratio=EXPERT_VRAM_RATIO,
)

# Apply the final caps and prefetch the chosen experts
expert_cache.max_bytes = int(bytes_cap_final)
expert_cache.max_count = int(count_cap_final)
expert_cache.prefetch_experts(top_keep_ids)

print("\n[Ratio Plan]")
for k, v in plan_stats.items():
    print(f"  {k}: {v}")
print(f"  selected_eids: {top_keep_ids}")
print(f"  caps → bytes={bytes_cap_final/1e9:.2f} GB  count={count_cap_final}")
