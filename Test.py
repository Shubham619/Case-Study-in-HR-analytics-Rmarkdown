#!/usr/bin/env python3
# gsat_rl_allinone.py
#
# One-file end-to-end: training and evaluation of stress patterns on a
# memory system.  The module includes a complete, hard-coded catalog of
# the patterns defined in the original Google "pattern.cc" (64- and
# 32-bit walking patterns, checkerboards, etc.), fully expanded into
# 32/64/128/256-bit variants and their inverted forms.  It supports an
# interactive driver (via subprocess) to send read/write requests to
# Ramulator2 and provides baseline exhaustive testing and a simple
# ε-greedy bandit learner.  Metrics such as time-to-first fault (TTFF),
# events per minute and latency distribution are reported at the end
# of a run.  Only standard library modules are used and no external
# dependencies are required.

import argparse
import json
import random
import subprocess
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

MASK32 = 0xFFFFFFFF


###############################################################################
# A) Complete pattern.cc hardcoded catalog
###############################################################################

@dataclass(frozen=True)
class PatternFamily:
    """Represents a family of patterns.

    Each pattern family carries a list of 32-bit words (`data_u32`) and a set
    of weights for different bus widths.  The original C++ code defines the
    pattern length to be the number of entries minus one (the last element is
    considered a sentinel value that is not part of the repeating sequence).
    """

    name: str
    data_u32: List[int]                # includes last sentinel (C++ uses len-1)
    weights: Tuple[int, int, int, int] # w32, w64, w128, w256

    @property
    def count(self) -> int:
        """Return the effective length of the pattern (excluding sentinel)."""
        return max(0, len(self.data_u32) - 1)


@dataclass(frozen=True)
class PatternVariant:
    """Represents a specific variant (family × width × inversion)."""

    family: PatternFamily
    buswidth: int
    invert: bool
    weight: int
    busshift: int

    @property
    def label(self) -> str:
        """Return a human-readable label for the variant."""
        return f"{self.family.name}{'~' if self.invert else ''}{self.buswidth}"

    def word(self, i: int) -> int:
        """Compute the i-th 32-bit word in the sequence for this variant.

        The pattern repeats over `family.count` entries but advances more
        slowly as the bus width increases (controlled by `busshift`).  If
        `invert` is true, the bits are flipped.
        """
        cnt = self.family.count
        if cnt <= 0:
            base = 0
        else:
            idx = ((i >> self.busshift) % cnt)
            base = self.family.data_u32[idx] & MASK32
        if self.invert:
            base ^= MASK32
        return base & MASK32


def _busshift(buswidth: int) -> int:
    """Return shift amount (0..3) corresponding to buswidth."""
    if buswidth == 32:
        return 0
    if buswidth == 64:
        return 1
    if buswidth == 128:
        return 2
    if buswidth == 256:
        return 3
    raise ValueError(f"Unsupported buswidth: {buswidth}")


# The following arrays are the exact 32-bit patterns from pattern.cc.  The
# sentinel value at the end of each array is included to preserve parity
# with the original C++ code; it is not used in the repeating sequence.

walkingOnes_data = [
    0x00000001, 0x00000002, 0x00000004, 0x00000008,
    0x00000010, 0x00000020, 0x00000040, 0x00000080,
    0x00000100, 0x00000200, 0x00000400, 0x00000800,
    0x00001000, 0x00002000, 0x00004000, 0x00008000,
    0x00010000, 0x00020000, 0x00040000, 0x00080000,
    0x00100000, 0x00200000, 0x00400000, 0x00800000,
    0x01000000, 0x02000000, 0x04000000, 0x08000000,
    0x10000000, 0x20000000, 0x40000000, 0x80000000,
    0x40000000, 0x20000000, 0x10000000, 0x08000000,
    0x04000000, 0x02000000, 0x01000000, 0x00800000,
    0x00400000, 0x00200000, 0x00100000, 0x00080000,
    0x00040000, 0x00020000, 0x00010000, 0x00008000,
    0x00004000, 0x00002000, 0x00001000, 0x00000800,
    0x00000400, 0x00000200, 0x00000100, 0x00000080,
    0x00000040, 0x00000020, 0x00000010, 0x00000008,
    0x00000004, 0x00000002, 0x00000001, 0x00000000,
]

walkingInvOnes_data = [
    0x00000001, 0xfffffffe, 0x00000002, 0xfffffffd,
    0x00000004, 0xfffffffb, 0x00000008, 0xfffffff7,
    0x00000010, 0xffffffef, 0x00000020, 0xffffffdf,
    0x00000040, 0xffffffbf, 0x00000080, 0xffffff7f,
    0x00000100, 0xfffffeff, 0x00000200, 0xfffffdff,
    0x00000400, 0xfffffbff, 0x00000800, 0xfffff7ff,
    0x00001000, 0xffffefff, 0x00002000, 0xffffdfff,
    0x00004000, 0xffffbfff, 0x00008000, 0xffff7fff,
    0x00010000, 0xfffeffff, 0x00020000, 0xfffdffff,
    0x00040000, 0xfffbffff, 0x00080000, 0xfff7ffff,
    0x00100000, 0xffefffff, 0x00200000, 0xffdfffff,
    0x00400000, 0xffbfffff, 0x00800000, 0xff7fffff,
    0x01000000, 0xfeffffff, 0x02000000, 0xfdffffff,
    0x04000000, 0xfbffffff, 0x08000000, 0xf7ffffff,
    0x10000000, 0xefffffff, 0x20000000, 0xdfffffff,
    0x40000000, 0xbfffffff, 0x80000000, 0x7fffffff,
    0x40000000, 0xbfffffff, 0x20000000, 0xdfffffff,
    0x10000000, 0xefffffff, 0x08000000, 0xf7ffffff,
    0x04000000, 0xfbffffff, 0x02000000, 0xfdffffff,
    0x01000000, 0xfeffffff, 0x00800000, 0xff7fffff,
    0x00400000, 0xffbfffff, 0x00200000, 0xffdfffff,
    0x00100000, 0xffefffff, 0x00080000, 0xfff7ffff,
    0x00040000, 0xfffbffff, 0x00020000, 0xfffdffff,
    0x00010000, 0xfffeffff, 0x00008000, 0xffff7fff,
    0x00004000, 0xffffbfff, 0x00002000, 0xffffdfff,
    0x00001000, 0xffffefff, 0x00000800, 0xfffff7ff,
    0x00000400, 0xfffffbff, 0x00000200, 0xfffffdff,
    0x00000100, 0xfffffeff, 0x00000080, 0xffffff7f,
    0x00000040, 0xffffffbf, 0x00000020, 0xffffffdf,
    0x00000010, 0xffffffef, 0x00000008, 0xfffffff7,
    0x00000004, 0xfffffffb, 0x00000002, 0xfffffffd,
    0x00000001, 0xfffffffe, 0x00000000, 0xffffffff,
]

walkingZeros_data = [
    0xfffffffe, 0xfffffffd, 0xfffffffb, 0xfffffff7,
    0xffffffef, 0xffffffdf, 0xffffffbf, 0xffffff7f,
    0xfffffeff, 0xfffffdff, 0xfffffbff, 0xfffff7ff,
    0xffffefff, 0xffffdfff, 0xffffbfff, 0xffff7fff,
    0xfffeffff, 0xfffdffff, 0xfffbffff, 0xfff7ffff,
    0xffefffff, 0xffdfffff, 0xffbfffff, 0xff7fffff,
    0xfeffffff, 0xfdffffff, 0xfbffffff, 0xf7ffffff,
    0xefffffff, 0xdfffffff, 0xbfffffff, 0x7fffffff,
    0xbfffffff, 0xdfffffff, 0xefffffff, 0xf7ffffff,
    0xfbffffff, 0xfdffffff, 0xfeffffff, 0xff7fffff,
    0xffbfffff, 0xffdfffff, 0xffefffff, 0xfff7ffff,
    0xfffbffff, 0xfffdffff, 0xfffeffff, 0xffff7fff,
    0xffffbfff, 0xffffdfff, 0xffffefff, 0xfffff7ff,
    0xfffffbff, 0xfffffdff, 0xfffffeff, 0xffffff7f,
    0xffffffbf, 0xffffffdf, 0xffffffef, 0xfffffff7,
    0xfffffffb, 0xfffffffd, 0xfffffffe, 0xffffffff,
]

OneZero_data = [0x00000000, 0xffffffff]
JustZero_data = [0x00000000, 0x00000000]
JustOne_data = [0xffffffff, 0xffffffff]
JustFive_data = [0x55555555, 0x55555555]
JustA_data = [0xaaaaaaaa, 0xaaaaaaaa]
FiveA_data = [0x55555555, 0xaaaaaaaa]
FiveA8_data = [0x5aa5a55a, 0xa55a5aa5, 0xa55a5aa5, 0x5aa5a55a]
Long8b10b_data = [0x16161616, 0x16161616]
Short8b10b_data = [0xb5b5b5b5, 0xb5b5b5b5]
Checker8b10b_data = [0xb5b5b5b5, 0x4a4a4a4a]
Five7_data = [0x55555557, 0x55575555]
Zero2fd_data = [0x00020002, 0xfffdfffd]

FAMILIES = [
    PatternFamily("walkingOnes",    walkingOnes_data,    (1, 1, 2, 1)),
    PatternFamily("walkingInvOnes", walkingInvOnes_data, (2, 2, 5, 5)),
    PatternFamily("walkingZeros",   walkingZeros_data,   (1, 1, 2, 1)),
    PatternFamily("OneZero",        OneZero_data,        (5, 5, 15, 5)),
    PatternFamily("JustZero",       JustZero_data,       (2, 0, 0, 0)),
    PatternFamily("JustOne",        JustOne_data,        (2, 0, 0, 0)),
    PatternFamily("JustFive",       JustFive_data,       (2, 0, 0, 0)),
    PatternFamily("JustA",          JustA_data,          (2, 0, 0, 0)),
    PatternFamily("FiveA",          FiveA_data,          (1, 1, 1, 1)),
    PatternFamily("FiveA8",         FiveA8_data,         (1, 1, 1, 1)),
    PatternFamily("Long8b10b",      Long8b10b_data,      (2, 0, 0, 0)),
    PatternFamily("Short8b10b",     Short8b10b_data,     (2, 0, 0, 0)),
    PatternFamily("Checker8b10b",   Checker8b10b_data,   (1, 0, 0, 1)),
    PatternFamily("Five7",          Five7_data,          (0, 2, 0, 0)),
    PatternFamily("Zero2fd",        Zero2fd_data,        (0, 2, 0, 0)),
]


def all_variants(include_zero_weight: bool = False) -> List[PatternVariant]:
    """Return all pattern variants (optionally including zero-weight ones)."""
    variants: List[PatternVariant] = []
    for fam in FAMILIES:
        for inv in (False, True):
            for bw, w in zip((32, 64, 128, 256), fam.weights):
                if not include_zero_weight and w <= 0:
                    continue
                variants.append(
                    PatternVariant(
                        family=fam,
                        buswidth=bw,
                        invert=inv,
                        weight=w,
                        busshift=_busshift(bw),
                    )
                )
    return variants


def top_k_by_weight(vars_: List[PatternVariant], k: int) -> List[PatternVariant]:
    """Select the top-k variants by descending weight (deterministic)."""
    ranked = sorted(vars_, key=lambda v: (-v.weight, v.label))
    return ranked[:k]


###############################################################################
# B) Ramulator interactive client
###############################################################################

class RamulatorProc:
    """A wrapper around a Ramulator interactive driver process.

    The process must print `READY` when started, and then accept one of two
    protocols for submitting requests: either a simpler `REQWAIT` form (A) or
    a more detailed `REQ` form (B).  The class auto-detects which protocol
    applies by probing the process.  See the user documentation for details.
    """

    def __init__(self, exe: str, config: str, max_cycles: int = 200_000):
        self.exe = exe
        self.config = config
        self.max_cycles = max_cycles
        self.proc: Optional[subprocess.Popen[str]] = None
        self.proto: Optional[str] = None  # "A" or "B"

    def start(self) -> None:
        """Launch the driver and detect its protocol."""
        self.proc = subprocess.Popen(
            [self.exe, self.config],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        first = self.proc.stdout.readline().strip()
        if first != "READY":
            raise RuntimeError(f"Driver not READY. Got: {first}")

        # Attempt to detect protocol by sending a probe.
        probe = self._cmd("REQWAIT 0x100000 0 0")
        if probe.startswith("OK") or probe == "STALLED":
            self.proto = "A"
            return

        probe = self._cmd(f"REQ R 0x100000 0 {self.max_cycles}")
        if probe.startswith("DONE") or probe.startswith("TIMEOUT") or probe == "STALLED":
            self.proto = "B"
            return

        raise RuntimeError(f"Unknown driver protocol. Response: {probe}")

    def _cmd(self, line: str) -> str:
        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def req(self, addr: int, is_write: bool, ctx: int = 0) -> Tuple[bool, Optional[int], str]:
        """Issue a single memory request.

        Returns (accepted, latency_cycles, raw_response).  If the request is
        stalled or rejected, `accepted` will be False and `latency_cycles`
        will be None.  For protocol B, the latency returned is the sum of
        memory cycles for both the write and the read.
        """
        if self.proc is None:
            raise RuntimeError("Ramulator driver has not been started")
        addr_hex = hex(addr)

        # Protocol A: REQWAIT <addr_hex> <is_write> <ctx>
        if self.proto == "A":
            resp = self._cmd(f"REQWAIT {addr_hex} {1 if is_write else 0} {ctx}")
            if resp == "STALLED":
                return False, None, resp
            if resp.startswith("OK"):
                parts = resp.split()
                lat = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                return True, lat, resp
            return False, None, resp

        # Protocol B: REQ <R|W> <addr_hex> <ctx> <max_cycles>
        if self.proto == "B":
            rw = "W" if is_write else "R"
            resp = self._cmd(f"REQ {rw} {addr_hex} {ctx} {self.max_cycles}")
            if resp == "STALLED":
                return False, None, resp
            if resp.startswith("DONE") or resp.startswith("TIMEOUT"):
                parts = resp.split()
                lat = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                return True, lat, resp
            return False, None, resp

        raise RuntimeError("Protocol not determined")

    def close(self) -> None:
        """Terminate the driver process cleanly."""
        if not self.proc:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write("EXIT\n")
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None


###############################################################################
# C) Event definition and metrics
###############################################################################

def latency_tail(latencies: List[int]) -> Dict[str, Optional[float]]:
    """Compute basic latency distribution metrics.

    Returns a dictionary with p50, p95, p99 and mean.  If the list is empty
    the values are None.  This function avoids using statistics module to
    remain dependency-free.
    """
    if not latencies:
        return {"p50": None, "p95": None, "p99": None, "mean": None}
    s = sorted(latencies)
    n = len(s)

    def percentile(p: float) -> float:
        idx = int(round(p / 100.0 * (n - 1)))
        idx = max(0, min(idx, n - 1))
        return float(s[idx])

    mean_val = float(sum(s) / n)
    return {
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
        "mean": mean_val,
    }


def is_event(accepted: bool, lat: Optional[int], raw: str, spike_thresh: int) -> bool:
    """Determine whether a response qualifies as a 'fault event'."""
    if not accepted:
        return True  # STALLED or unknown errors
    if raw.startswith("TIMEOUT"):
        return True
    if lat is not None and lat >= spike_thresh:
        return True
    return False


###############################################################################
# D) Core step logic
###############################################################################

def step_variant(
    ram: RamulatorProc,
    v: PatternVariant,
    addr_base: int,
    stride: int,
    n_addrs: int,
    ctx: int,
    spike_thresh: int,
) -> Tuple[bool, Optional[int], bool, str]:
    """Perform a single write/read test on one address for the given variant.

    The address is selected deterministically based on the variant label to
    provide coverage across the address space.  Returns a tuple of
    (accepted, latency_cycles, event, raw_response).
    """
    h = sum(ord(c) for c in v.label) & MASK32
    idx = (h % n_addrs)
    addr = addr_base + idx * stride

    # Issue write then read
    acc_w, lat_w, raw_w = ram.req(addr, True, ctx)
    if not acc_w:
        return False, None, True, raw_w
    acc_r, lat_r, raw_r = ram.req(addr, False, ctx)
    if not acc_r:
        return False, None, True, raw_r

    lat: Optional[int] = None
    if lat_w is not None and lat_r is not None:
        lat = lat_w + lat_r
    event = is_event(True, lat, raw_r, spike_thresh)
    return True, lat, event, raw_r


###############################################################################
# E) Baseline exhaustive and RL bandit logic
###############################################################################

def run_baseline(
    ram_exe: str,
    config: str,
    action_set: List[PatternVariant],
    steps_per_action: int,
    addr_base: int,
    stride: int,
    n_addrs: int,
    ctx: int,
    spike_thresh: int,
) -> Dict[str, Any]:
    """Exhaustively apply each pattern variant a fixed number of times.

    Returns a dictionary summarizing the run, including TTFF and event rate.
    """
    ram = RamulatorProc(ram_exe, config)
    ram.start()

    start = time.time()
    first_event_t: Optional[float] = None
    first_event_step: Optional[int] = None
    first_event_action: Optional[str] = None

    total_steps = 0
    events = 0
    latencies: List[int] = []
    coverage = set()

    for ai, v in enumerate(action_set):
        for _ in range(steps_per_action):
            accepted, lat, ev, _ = step_variant(
                ram,
                v,
                addr_base=addr_base,
                stride=stride,
                n_addrs=n_addrs,
                ctx=ctx,
                spike_thresh=spike_thresh,
            )
            total_steps += 1
            coverage.add(v.label)
            if lat is not None:
                latencies.append(lat)
            if ev:
                events += 1
                if first_event_t is None:
                    first_event_t = time.time()
                    first_event_step = total_steps
                    first_event_action = v.label

    elapsed = time.time() - start
    ttff_s = None if first_event_t is None else (first_event_t - start)
    ram.close()
    return {
        "mode": "baseline",
        "elapsed_s": elapsed,
        "total_steps": total_steps,
        "actions": len(action_set),
        "steps_per_action": steps_per_action,
        "events": events,
        "events_per_min": (events / (elapsed / 60.0)) if elapsed > 0 else None,
        "ttff_s": ttff_s,
        "ttff_steps": first_event_step,
        "ttff_action": first_event_action,
        "coverage_actions": len(coverage),
        "latency_tail": latency_tail(latencies),
    }


def train_bandit(
    ram_exe: str,
    config: str,
    action_set: List[PatternVariant],
    episodes: int,
    steps_per_ep: int,
    epsilon: float,
    alpha: float,
    addr_base: int,
    stride: int,
    n_addrs: int,
    ctx: int,
    spike_thresh: int,
    seed: int,
) -> Dict[str, Any]:
    """Train a simple ε-greedy bandit over the pattern variants.

    Returns a report containing the learned Q values, coverage statistics and
    TTFF.  A reward of 100 is given for each fault event, plus a small
    bonus proportional to latency to encourage exploring higher latency
    operations.
    """
    rng = random.Random(seed)
    ram = RamulatorProc(ram_exe, config)
    ram.start()

    K = len(action_set)
    Q = [0.0] * K
    pulls = [0] * K

    start = time.time()
    first_event_t: Optional[float] = None
    first_event_step: Optional[int] = None
    first_event_action: Optional[str] = None

    total_steps = 0
    events = 0
    latencies: List[int] = []
    coverage = set()

    def reward_fn(ev: bool, lat: Optional[int]) -> float:
        r = 0.0
        if ev:
            r += 100.0
        if lat is not None:
            r += min(lat / 1000.0, 10.0)
        return r

    for ep in range(episodes):
        for _ in range(steps_per_ep):
            # Choose action using ε-greedy strategy.
            if rng.random() < epsilon:
                a = rng.randrange(K)
            else:
                a = max(range(K), key=lambda i: Q[i])
            v = action_set[a]
            coverage.add(v.label)
            accepted, lat, ev, _ = step_variant(
                ram,
                v,
                addr_base=addr_base,
                stride=stride,
                n_addrs=n_addrs,
                ctx=ctx,
                spike_thresh=spike_thresh,
            )
            total_steps += 1
            if lat is not None:
                latencies.append(lat)
            if ev:
                events += 1
                if first_event_t is None:
                    first_event_t = time.time()
                    first_event_step = total_steps
                    first_event_action = v.label
            # Bandit update
            pulls[a] += 1
            r = reward_fn(ev, lat)
            Q[a] = Q[a] + alpha * (r - Q[a])
        # Progress log at 25%, 50%, 75% and end.
        if (ep + 1) in {max(1, episodes // 4), max(1, episodes // 2), max(1, (3 * episodes) // 4), episodes}:
            best_idx = max(range(K), key=lambda i: Q[i])
            print(
                f"[train] ep={ep+1}/{episodes} best={action_set[best_idx].label} "
                f"Q={Q[best_idx]:.2f} events={events}",
            )

    elapsed = time.time() - start
    ttff_s = None if first_event_t is None else (first_event_t - start)
    best_idx = max(range(K), key=lambda i: Q[i])
    ram.close()
    return {
        "mode": "train",
        "seed": seed,
        "episodes": episodes,
        "steps_per_ep": steps_per_ep,
        "epsilon": epsilon,
        "alpha": alpha,
        "elapsed_s": elapsed,
        "total_steps": total_steps,
        "events": events,
        "events_per_min": (events / (elapsed / 60.0)) if elapsed > 0 else None,
        "ttff_s": ttff_s,
        "ttff_steps": first_event_step,
        "ttff_action": first_event_action,
        "coverage_actions": len(coverage),
        "latency_tail": latency_tail(latencies),
        "best_index": best_idx,
        "best_action": action_set[best_idx].label,
        "Q": Q,
        "pulls": pulls,
        "action_labels": [v.label for v in action_set],
    }


def eval_best(
    ram_exe: str,
    config: str,
    action_set: List[PatternVariant],
    best_action_label: str,
    eval_steps: int,
    addr_base: int,
    stride: int,
    n_addrs: int,
    ctx: int,
    spike_thresh: int,
) -> Dict[str, Any]:
    """Evaluate one pattern variant for a number of steps and gather stats."""
    label_to_idx = {v.label: i for i, v in enumerate(action_set)}
    if best_action_label not in label_to_idx:
        raise ValueError(
            f"best_action_label '{best_action_label}' not found in current action_set"
        )
    best_variant = action_set[label_to_idx[best_action_label]]
    ram = RamulatorProc(ram_exe, config)
    ram.start()
    start = time.time()
    first_event_t: Optional[float] = None
    first_event_step: Optional[int] = None
    events = 0
    latencies: List[int] = []
    coverage = {best_variant.label}
    for step in range(1, eval_steps + 1):
        accepted, lat, ev, _ = step_variant(
            ram,
            best_variant,
            addr_base=addr_base,
            stride=stride,
            n_addrs=n_addrs,
            ctx=ctx,
            spike_thresh=spike_thresh,
        )
        if lat is not None:
            latencies.append(lat)
        if ev:
            events += 1
            if first_event_t is None:
                first_event_t = time.time()
                first_event_step = step
    elapsed = time.time() - start
    ttff_s = None if first_event_t is None else (first_event_t - start)
    ram.close()
    return {
        "mode": "eval",
        "best_action": best_variant.label,
        "elapsed_s": elapsed,
        "eval_steps": eval_steps,
        "events": events,
        "events_per_min": (events / (elapsed / 60.0)) if elapsed > 0 else None,
        "ttff_s": ttff_s,
        "ttff_steps": first_event_step,
        "coverage_actions": len(coverage),
        "latency_tail": latency_tail(latencies),
    }


###############################################################################
# F) CLI
###############################################################################

def make_action_set(kind: str) -> List[PatternVariant]:
    """Return the appropriate action set by type."""
    variants = all_variants(include_zero_weight=False)
    if kind == "all":
        return variants
    if kind == "top64":
        return top_k_by_weight(variants, 64)
    raise ValueError("action_set must be 'top64' or 'all'")


def main() -> None:
    ap = argparse.ArgumentParser(description="GSAT RL and baseline runner")
    ap.add_argument(
        "--exe",
        required=True,
        help="Path to the interactive driver executable (must print READY)",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config used by the driver",
    )

    # Prefer underscore options (no shell/argparse surprises). Keep hyphen aliases too.
    ap.add_argument(
        "--action_set",
        "--action-set",
        dest="action_set",
        default="top64",
        choices=["top64", "all"],
        help="Use top64 weighted variants or all selectable variants",
    )
    ap.add_argument(
        "--addr_base",
        "--addr-base",
        dest="addr_base",
        type=lambda x: int(x, 0),
        default=0x100000,
        help="Base address (hex or decimal)",
    )
    ap.add_argument("--stride", type=int, default=64, help="Stride in bytes")
    ap.add_argument(
        "--n_addrs",
        "--n-addrs",
        dest="n_addrs",
        type=int,
        default=1024,
        help="Address window size (count)",
    )
    ap.add_argument("--ctx", type=int, default=0, help="Context ID")
    ap.add_argument(
        "--spike_thresh",
        "--spike-thresh",
        dest="spike_thresh",
        type=int,
        default=4000,
        help="Latency threshold in cycles to consider a spike",
    )

    subparsers = ap.add_subparsers(dest="cmd", required=True)

    b = subparsers.add_parser("baseline", help="Run baseline exhaustive test")
    b.add_argument(
        "--steps_per_action",
        "--steps-per-action",
        dest="steps_per_action",
        type=int,
        default=200,
        help="Steps per action variant",
    )

    t = subparsers.add_parser("train", help="Train RL agent")
    t.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    t.add_argument(
        "--steps_per_ep",
        "--steps-per-ep",
        dest="steps_per_ep",
        type=int,
        default=50,
        help="Steps per episode",
    )
    t.add_argument("--epsilon", type=float, default=0.2, help="Exploration rate")
    t.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    t.add_argument("--seed", type=int, default=1, help="Random seed")
    t.add_argument("--out", default="rl_model.json", help="Output model file")

    e = subparsers.add_parser("eval", help="Evaluate a saved RL model")
    e.add_argument("--model", required=True, help="Model JSON from training")
    e.add_argument(
        "--eval_steps",
        "--eval-steps",
        dest="eval_steps",
        type=int,
        default=2000,
        help="Steps for evaluation",
    )

    c = subparsers.add_parser("compare", help="Baseline vs RL comparison")
    c.add_argument(
        "--steps_per_action",
        "--steps-per-action",
        dest="steps_per_action",
        type=int,
        default=200,
        help="Steps per action in baseline",
    )
    c.add_argument("--episodes", type=int, default=200)
    c.add_argument(
        "--steps_per_ep",
        "--steps-per-ep",
        dest="steps_per_ep",
        type=int,
        default=50,
    )
    c.add_argument("--epsilon", type=float, default=0.2)
    c.add_argument("--alpha", type=float, default=0.1)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument(
        "--eval_steps",
        "--eval-steps",
        dest="eval_steps",
        type=int,
        default=2000,
    )
    c.add_argument("--out", default="compare_report.json")

    args = ap.parse_args()
    action_set = make_action_set(args.action_set)

    if args.cmd == "baseline":
        report = run_baseline(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            steps_per_action=args.steps_per_action,
            addr_base=args.addr_base,
            stride=args.stride,
            n_addrs=args.n_addrs,
            ctx=args.ctx,
            spike_thresh=args.spike_thresh,
        )
        print(json.dumps(report, indent=2))
        return

    if args.cmd == "train":
        report = train_bandit(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            episodes=args.episodes,
            steps_per_ep=args.steps_per_ep,
            epsilon=args.epsilon,
            alpha=args.alpha,
            addr_base=args.addr_base,
            stride=args.stride,
            n_addrs=args.n_addrs,
            ctx=args.ctx,
            spike_thresh=args.spike_thresh,
            seed=args.seed,
        )
        # Save model
        out = {
            "meta": {
                "exe": args.exe,
                "config": args.config,
                "action_set": args.action_set,
                "spike_thresh": args.spike_thresh,
                "addr_base": args.addr_base,
                "stride": args.stride,
                "n_addrs": args.n_addrs,
                "ctx": args.ctx,
            },
            "train_report": {k: report[k] for k in report if k not in ("Q", "pulls", "action_labels")},
            "Q": report["Q"],
            "pulls": report["pulls"],
            "action_labels": report["action_labels"],
            "best_action": report["best_action"],
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved model: {args.out}")
        print(f"Best action: {report['best_action']}")
        return

    if args.cmd == "eval":
        with open(args.model) as f:
            model = json.load(f)
        best_action = model["best_action"]
        report = eval_best(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            best_action_label=best_action,
            eval_steps=args.eval_steps,
            addr_base=args.addr_base,
            stride=args.stride,
            n_addrs=args.n_addrs,
            ctx=args.ctx,
            spike_thresh=args.spike_thresh,
        )
        print(json.dumps(report, indent=2))
        return

    if args.cmd == "compare":
        base = run_baseline(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            steps_per_action=args.steps_per_action,
            addr_base=args.addr_base,
            stride=args.stride,
            n_addrs=args.n_addrs,
            ctx=args.ctx,
            spike_thresh=args.spike_thresh,
        )
        print("\n[compare] baseline done")

        train_rep = train_bandit(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            episodes=args.episodes,
            steps_per_ep=args.steps_per_ep,
            epsilon=args.epsilon,
            alpha=args.alpha,
            addr_base=args.addr_base,
            stride=args.stride,
            n_addrs=args.n_addrs,
            ctx=args.ctx,
            spike_thresh=args.spike_thresh,
            seed=args.seed,
        )
        best_action = train_rep["best_action"]
        print("\n[compare] train done, best =", best_action)

        ev = eval_best(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            best_action_label=best_action,
            eval_steps=args.eval_steps,
            addr_base=args.addr_base,
            stride=args.stride,
            n_addrs=args.n_addrs,
            ctx=args.ctx,
            spike_thresh=args.spike_thresh,
        )
        print("\n[compare] eval done")

        summary = {
            "baseline_ttff_s": base.get("ttff_s"),
            "rl_train_ttff_s": train_rep.get("ttff_s"),
            "rl_eval_ttff_s": ev.get("ttff_s"),
            "baseline_events_per_min": base.get("events_per_min"),
            "rl_train_events_per_min": train_rep.get("events_per_min"),
            "rl_eval_events_per_min": ev.get("events_per_min"),
            "best_action": best_action,
        }

        report = {
            "baseline": base,
            "train": {k: train_rep[k] for k in train_rep if k not in ("Q", "pulls", "action_labels")},
            "eval_best": ev,
            "summary": summary,
        }

        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved compare report: {args.out}")
        print(json.dumps(summary, indent=2))
        return

    raise RuntimeError("No valid subcommand provided")


if __name__ == "__main__":
    main()

    main()




#!/usr/bin/env python3
# gsat_rl_allinone.py
#
# One‑file end‑to‑end: training and evaluation of stress patterns on a
# memory system.  The module includes a complete, hard‑coded catalog of
# the patterns defined in the original Google "pattern.cc" (64‑ and
# 32‑bit walking patterns, checkerboards, etc.), fully expanded into
# 32/64/128/256‑bit variants and their inverted forms.  It supports an
# interactive driver (via subprocess) to send read/write requests to
# Ramulator2 and provides baseline exhaustive testing and a simple
# ε‑greedy bandit learner.  Metrics such as time‑to‑first fault (TTFF),
# events per minute and latency distribution are reported at the end
# of a run.  Only standard library modules are used and no external
# dependencies are required.

import argparse
import json
import random
import subprocess
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

MASK32 = 0xFFFFFFFF


###############################################################################
# A) Complete pattern.cc hardcoded catalog
###############################################################################

@dataclass(frozen=True)
class PatternFamily:
    """Represents a family of patterns.

    Each pattern family carries a list of 32‑bit words (`data_u32`) and a set
    of weights for different bus widths.  The original C++ code defines the
    pattern length to be the number of entries minus one (the last element is
    considered a sentinel value that is not part of the repeating sequence).
    """

    name: str
    data_u32: List[int]                # includes last sentinel (C++ uses len‑1)
    weights: Tuple[int, int, int, int] # w32, w64, w128, w256

    @property
    def count(self) -> int:
        """Return the effective length of the pattern (excluding sentinel)."""
        return max(0, len(self.data_u32) - 1)


@dataclass(frozen=True)
class PatternVariant:
    """Represents a specific variant (family × width × inversion)."""

    family: PatternFamily
    buswidth: int
    invert: bool
    weight: int
    busshift: int

    @property
    def label(self) -> str:
        """Return a human‑readable label for the variant."""
        return f"{self.family.name}{'~' if self.invert else ''}{self.buswidth}"

    def word(self, i: int) -> int:
        """Compute the i‑th 32‑bit word in the sequence for this variant.

        The pattern repeats over `family.count` entries but advances more
        slowly as the bus width increases (controlled by `busshift`).  If
        `invert` is true, the bits are flipped.
        """
        cnt = self.family.count
        if cnt <= 0:
            base = 0
        else:
            idx = ((i >> self.busshift) % cnt)
            base = self.family.data_u32[idx] & MASK32
        if self.invert:
            base ^= MASK32
        return base & MASK32


def _busshift(buswidth: int) -> int:
    """Return shift amount (0..3) corresponding to buswidth."""
    if buswidth == 32:
        return 0
    if buswidth == 64:
        return 1
    if buswidth == 128:
        return 2
    if buswidth == 256:
        return 3
    raise ValueError(f"Unsupported buswidth: {buswidth}")


# The following arrays are the exact 32‑bit patterns from pattern.cc.  The
# sentinel value at the end of each array is included to preserve parity
# with the original C++ code; it is not used in the repeating sequence.

walkingOnes_data = [
    0x00000001, 0x00000002, 0x00000004, 0x00000008,
    0x00000010, 0x00000020, 0x00000040, 0x00000080,
    0x00000100, 0x00000200, 0x00000400, 0x00000800,
    0x00001000, 0x00002000, 0x00004000, 0x00008000,
    0x00010000, 0x00020000, 0x00040000, 0x00080000,
    0x00100000, 0x00200000, 0x00400000, 0x00800000,
    0x01000000, 0x02000000, 0x04000000, 0x08000000,
    0x10000000, 0x20000000, 0x40000000, 0x80000000,
    0x40000000, 0x20000000, 0x10000000, 0x08000000,
    0x04000000, 0x02000000, 0x01000000, 0x00800000,
    0x00400000, 0x00200000, 0x00100000, 0x00080000,
    0x00040000, 0x00020000, 0x00010000, 0x00008000,
    0x00004000, 0x00002000, 0x00001000, 0x00000800,
    0x00000400, 0x00000200, 0x00000100, 0x00000080,
    0x00000040, 0x00000020, 0x00000010, 0x00000008,
    0x00000004, 0x00000002, 0x00000001, 0x00000000,
]

walkingInvOnes_data = [
    0x00000001, 0xfffffffe, 0x00000002, 0xfffffffd,
    0x00000004, 0xfffffffb, 0x00000008, 0xfffffff7,
    0x00000010, 0xffffffef, 0x00000020, 0xffffffdf,
    0x00000040, 0xffffffbf, 0x00000080, 0xffffff7f,
    0x00000100, 0xfffffeff, 0x00000200, 0xfffffdff,
    0x00000400, 0xfffffbff, 0x00000800, 0xfffff7ff,
    0x00001000, 0xffffefff, 0x00002000, 0xffffdfff,
    0x00004000, 0xffffbfff, 0x00008000, 0xffff7fff,
    0x00010000, 0xfffeffff, 0x00020000, 0xfffdffff,
    0x00040000, 0xfffbffff, 0x00080000, 0xfff7ffff,
    0x00100000, 0xffefffff, 0x00200000, 0xffdfffff,
    0x00400000, 0xffbfffff, 0x00800000, 0xff7fffff,
    0x01000000, 0xfeffffff, 0x02000000, 0xfdffffff,
    0x04000000, 0xfbffffff, 0x08000000, 0xf7ffffff,
    0x10000000, 0xefffffff, 0x20000000, 0xdfffffff,
    0x40000000, 0xbfffffff, 0x80000000, 0x7fffffff,
    0x40000000, 0xbfffffff, 0x20000000, 0xdfffffff,
    0x10000000, 0xefffffff, 0x08000000, 0xf7ffffff,
    0x04000000, 0xfbffffff, 0x02000000, 0xfdffffff,
    0x01000000, 0xfeffffff, 0x00800000, 0xff7fffff,
    0x00400000, 0xffbfffff, 0x00200000, 0xffdfffff,
    0x00100000, 0xffefffff, 0x00080000, 0xfff7ffff,
    0x00040000, 0xfffbffff, 0x00020000, 0xfffdffff,
    0x00010000, 0xfffeffff, 0x00008000, 0xffff7fff,
    0x00004000, 0xffffbfff, 0x00002000, 0xffffdfff,
    0x00001000, 0xffffefff, 0x00000800, 0xfffff7ff,
    0x00000400, 0xfffffbff, 0x00000200, 0xfffffdff,
    0x00000100, 0xfffffeff, 0x00000080, 0xffffff7f,
    0x00000040, 0xffffffbf, 0x00000020, 0xffffffdf,
    0x00000010, 0xffffffef, 0x00000008, 0xfffffff7,
    0x00000004, 0xfffffffb, 0x00000002, 0xfffffffd,
    0x00000001, 0xfffffffe, 0x00000000, 0xffffffff,
]

walkingZeros_data = [
    0xfffffffe, 0xfffffffd, 0xfffffffb, 0xfffffff7,
    0xffffffef, 0xffffffdf, 0xffffffbf, 0xffffff7f,
    0xfffffeff, 0xfffffdff, 0xfffffbff, 0xfffff7ff,
    0xffffefff, 0xffffdfff, 0xffffbfff, 0xffff7fff,
    0xfffeffff, 0xfffdffff, 0xfffbffff, 0xfff7ffff,
    0xffefffff, 0xffdfffff, 0xffbfffff, 0xff7fffff,
    0xfeffffff, 0xfdffffff, 0xfbffffff, 0xf7ffffff,
    0xefffffff, 0xdfffffff, 0xbfffffff, 0x7fffffff,
    0xbfffffff, 0xdfffffff, 0xefffffff, 0xf7ffffff,
    0xfbffffff, 0xfdffffff, 0xfeffffff, 0xff7fffff,
    0xffbfffff, 0xffdfffff, 0xffefffff, 0xfff7ffff,
    0xfffbffff, 0xfffdffff, 0xfffeffff, 0xffff7fff,
    0xffffbfff, 0xffffdfff, 0xffffefff, 0xfffff7ff,
    0xfffffbff, 0xfffffdff, 0xfffffeff, 0xffffff7f,
    0xffffffbf, 0xffffffdf, 0xffffffef, 0xfffffff7,
    0xfffffffb, 0xfffffffd, 0xfffffffe, 0xffffffff,
]

OneZero_data = [0x00000000, 0xffffffff]
JustZero_data = [0x00000000, 0x00000000]
JustOne_data = [0xffffffff, 0xffffffff]
JustFive_data = [0x55555555, 0x55555555]
JustA_data = [0xaaaaaaaa, 0xaaaaaaaa]
FiveA_data = [0x55555555, 0xaaaaaaaa]
FiveA8_data = [0x5aa5a55a, 0xa55a5aa5, 0xa55a5aa5, 0x5aa5a55a]
Long8b10b_data = [0x16161616, 0x16161616]
Short8b10b_data = [0xb5b5b5b5, 0xb5b5b5b5]
Checker8b10b_data = [0xb5b5b5b5, 0x4a4a4a4a]
Five7_data = [0x55555557, 0x55575555]
Zero2fd_data = [0x00020002, 0xfffdfffd]

FAMILIES = [
    PatternFamily("walkingOnes",    walkingOnes_data,    (1, 1, 2, 1)),
    PatternFamily("walkingInvOnes", walkingInvOnes_data, (2, 2, 5, 5)),
    PatternFamily("walkingZeros",   walkingZeros_data,   (1, 1, 2, 1)),
    PatternFamily("OneZero",        OneZero_data,        (5, 5, 15, 5)),
    PatternFamily("JustZero",       JustZero_data,       (2, 0, 0, 0)),
    PatternFamily("JustOne",        JustOne_data,        (2, 0, 0, 0)),
    PatternFamily("JustFive",       JustFive_data,       (2, 0, 0, 0)),
    PatternFamily("JustA",          JustA_data,          (2, 0, 0, 0)),
    PatternFamily("FiveA",          FiveA_data,          (1, 1, 1, 1)),
    PatternFamily("FiveA8",         FiveA8_data,         (1, 1, 1, 1)),
    PatternFamily("Long8b10b",      Long8b10b_data,      (2, 0, 0, 0)),
    PatternFamily("Short8b10b",     Short8b10b_data,     (2, 0, 0, 0)),
    PatternFamily("Checker8b10b",   Checker8b10b_data,   (1, 0, 0, 1)),
    PatternFamily("Five7",          Five7_data,          (0, 2, 0, 0)),
    PatternFamily("Zero2fd",        Zero2fd_data,        (0, 2, 0, 0)),
]


def all_variants(include_zero_weight: bool = False) -> List[PatternVariant]:
    """Return all pattern variants (optionally including zero‑weight ones)."""
    variants: List[PatternVariant] = []
    for fam in FAMILIES:
        for inv in (False, True):
            for bw, w in zip((32, 64, 128, 256), fam.weights):
                if not include_zero_weight and w <= 0:
                    continue
                variants.append(
                    PatternVariant(
                        family=fam,
                        buswidth=bw,
                        invert=inv,
                        weight=w,
                        busshift=_busshift(bw),
                    )
                )
    return variants


def top_k_by_weight(vars_: List[PatternVariant], k: int) -> List[PatternVariant]:
    """Select the top‑k variants by descending weight (deterministic)."""
    ranked = sorted(vars_, key=lambda v: (-v.weight, v.label))
    return ranked[:k]


###############################################################################
# B) Ramulator interactive client
###############################################################################

class RamulatorProc:
    """A wrapper around a Ramulator interactive driver process.

    The process must print `READY` when started, and then accept one of two
    protocols for submitting requests: either a simpler `REQWAIT` form (A) or
    a more detailed `REQ` form (B).  The class auto‑detects which protocol
    applies by probing the process.  See the user documentation for details.
    """

    def __init__(self, exe: str, config: str, max_cycles: int = 200_000):
        self.exe = exe
        self.config = config
        self.max_cycles = max_cycles
        self.proc: Optional[subprocess.Popen[str]] = None
        self.proto: Optional[str] = None  # "A" or "B"

    def start(self) -> None:
        """Launch the driver and detect its protocol."""
        self.proc = subprocess.Popen(
            [self.exe, self.config],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        first = self.proc.stdout.readline().strip()
        if first != "READY":
            raise RuntimeError(f"Driver not READY. Got: {first}")

        # Attempt to detect protocol by sending a probe.
        probe = self._cmd("REQWAIT 0x100000 0 0")
        if probe.startswith("OK") or probe == "STALLED":
            self.proto = "A"
            return

        probe = self._cmd(f"REQ R 0x100000 0 {self.max_cycles}")
        if probe.startswith("DONE") or probe.startswith("TIMEOUT") or probe == "STALLED":
            self.proto = "B"
            return

        raise RuntimeError(f"Unknown driver protocol. Response: {probe}")

    def _cmd(self, line: str) -> str:
        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def req(self, addr: int, is_write: bool, ctx: int = 0) -> Tuple[bool, Optional[int], str]:
        """Issue a single memory request.

        Returns (accepted, latency_cycles, raw_response).  If the request is
        stalled or rejected, `accepted` will be False and `latency_cycles`
        will be None.  For protocol B, the latency returned is the sum of
        memory cycles for both the write and the read.
        """
        if self.proc is None:
            raise RuntimeError("Ramulator driver has not been started")
        addr_hex = hex(addr)

        # Protocol A: REQWAIT <addr_hex> <is_write> <ctx>
        if self.proto == "A":
            resp = self._cmd(f"REQWAIT {addr_hex} {1 if is_write else 0} {ctx}")
            if resp == "STALLED":
                return False, None, resp
            if resp.startswith("OK"):
                parts = resp.split()
                lat = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                return True, lat, resp
            return False, None, resp

        # Protocol B: REQ <R|W> <addr_hex> <ctx> <max_cycles>
        if self.proto == "B":
            rw = "W" if is_write else "R"
            resp = self._cmd(f"REQ {rw} {addr_hex} {ctx} {self.max_cycles}")
            if resp == "STALLED":
                return False, None, resp
            if resp.startswith("DONE") or resp.startswith("TIMEOUT"):
                parts = resp.split()
                lat = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                return True, lat, resp
            return False, None, resp

        raise RuntimeError("Protocol not determined")

    def close(self) -> None:
        """Terminate the driver process cleanly."""
        if not self.proc:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write("EXIT\n")
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None


###############################################################################
# C) Event definition and metrics
###############################################################################

def latency_tail(latencies: List[int]) -> Dict[str, Optional[float]]:
    """Compute basic latency distribution metrics.

    Returns a dictionary with p50, p95, p99 and mean.  If the list is empty
    the values are None.  This function avoids using statistics module to
    remain dependency‑free.
    """
    if not latencies:
        return {"p50": None, "p95": None, "p99": None, "mean": None}
    s = sorted(latencies)
    n = len(s)

    def percentile(p: float) -> float:
        idx = int(round(p / 100.0 * (n - 1)))
        idx = max(0, min(idx, n - 1))
        return float(s[idx])

    mean_val = float(sum(s) / n)
    return {
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
        "mean": mean_val,
    }


def is_event(accepted: bool, lat: Optional[int], raw: str, spike_thresh: int) -> bool:
    """Determine whether a response qualifies as a 'fault event'."""
    if not accepted:
        return True  # STALLED or unknown errors
    if raw.startswith("TIMEOUT"):
        return True
    if lat is not None and lat >= spike_thresh:
        return True
    return False


###############################################################################
# D) Core step logic
###############################################################################

def step_variant(
    ram: RamulatorProc,
    v: PatternVariant,
    addr_base: int,
    stride: int,
    n_addrs: int,
    ctx: int,
    spike_thresh: int,
) -> Tuple[bool, Optional[int], bool, str]:
    """Perform a single write/read test on one address for the given variant.

    The address is selected deterministically based on the variant label to
    provide coverage across the address space.  Returns a tuple of
    (accepted, latency_cycles, event, raw_response).
    """
    h = sum(ord(c) for c in v.label) & MASK32
    idx = (h % n_addrs)
    addr = addr_base + idx * stride

    # Issue write then read
    acc_w, lat_w, raw_w = ram.req(addr, True, ctx)
    if not acc_w:
        return False, None, True, raw_w
    acc_r, lat_r, raw_r = ram.req(addr, False, ctx)
    if not acc_r:
        return False, None, True, raw_r

    lat: Optional[int] = None
    if lat_w is not None and lat_r is not None:
        lat = lat_w + lat_r
    event = is_event(True, lat, raw_r, spike_thresh)
    return True, lat, event, raw_r


###############################################################################
# E) Baseline exhaustive and RL bandit logic
###############################################################################

def run_baseline(
    ram_exe: str,
    config: str,
    action_set: List[PatternVariant],
    steps_per_action: int,
    addr_base: int,
    stride: int,
    n_addrs: int,
    ctx: int,
    spike_thresh: int,
) -> Dict[str, Any]:
    """Exhaustively apply each pattern variant a fixed number of times.

    Returns a dictionary summarizing the run, including TTFF and event rate.
    """
    ram = RamulatorProc(ram_exe, config)
    ram.start()

    start = time.time()
    first_event_t: Optional[float] = None
    first_event_step: Optional[int] = None
    first_event_action: Optional[str] = None

    total_steps = 0
    events = 0
    latencies: List[int] = []
    coverage = set()

    for ai, v in enumerate(action_set):
        for _ in range(steps_per_action):
            accepted, lat, ev, _ = step_variant(
                ram,
                v,
                addr_base=addr_base,
                stride=stride,
                n_addrs=n_addrs,
                ctx=ctx,
                spike_thresh=spike_thresh,
            )
            total_steps += 1
            coverage.add(v.label)
            if lat is not None:
                latencies.append(lat)
            if ev:
                events += 1
                if first_event_t is None:
                    first_event_t = time.time()
                    first_event_step = total_steps
                    first_event_action = v.label

    elapsed = time.time() - start
    ttff_s = None if first_event_t is None else (first_event_t - start)
    ram.close()
    return {
        "mode": "baseline",
        "elapsed_s": elapsed,
        "total_steps": total_steps,
        "actions": len(action_set),
        "steps_per_action": steps_per_action,
        "events": events,
        "events_per_min": (events / (elapsed / 60.0)) if elapsed > 0 else None,
        "ttff_s": ttff_s,
        "ttff_steps": first_event_step,
        "ttff_action": first_event_action,
        "coverage_actions": len(coverage),
        "latency_tail": latency_tail(latencies),
    }


def train_bandit(
    ram_exe: str,
    config: str,
    action_set: List[PatternVariant],
    episodes: int,
    steps_per_ep: int,
    epsilon: float,
    alpha: float,
    addr_base: int,
    stride: int,
    n_addrs: int,
    ctx: int,
    spike_thresh: int,
    seed: int,
) -> Dict[str, Any]:
    """Train a simple ε‑greedy bandit over the pattern variants.

    Returns a report containing the learned Q values, coverage statistics and
    TTFF.  A reward of 100 is given for each fault event, plus a small
    bonus proportional to latency to encourage exploring higher latency
    operations.
    """
    rng = random.Random(seed)
    ram = RamulatorProc(ram_exe, config)
    ram.start()

    K = len(action_set)
    Q = [0.0] * K
    pulls = [0] * K

    start = time.time()
    first_event_t: Optional[float] = None
    first_event_step: Optional[int] = None
    first_event_action: Optional[str] = None

    total_steps = 0
    events = 0
    latencies: List[int] = []
    coverage = set()

    def reward_fn(ev: bool, lat: Optional[int]) -> float:
        r = 0.0
        if ev:
            r += 100.0
        if lat is not None:
            r += min(lat / 1000.0, 10.0)
        return r

    for ep in range(episodes):
        for _ in range(steps_per_ep):
            # Choose action using ε‑greedy strategy.
            if rng.random() < epsilon:
                a = rng.randrange(K)
            else:
                a = max(range(K), key=lambda i: Q[i])
            v = action_set[a]
            coverage.add(v.label)
            accepted, lat, ev, _ = step_variant(
                ram,
                v,
                addr_base=addr_base,
                stride=stride,
                n_addrs=n_addrs,
                ctx=ctx,
                spike_thresh=spike_thresh,
            )
            total_steps += 1
            if lat is not None:
                latencies.append(lat)
            if ev:
                events += 1
                if first_event_t is None:
                    first_event_t = time.time()
                    first_event_step = total_steps
                    first_event_action = v.label
            # Bandit update
            pulls[a] += 1
            r = reward_fn(ev, lat)
            Q[a] = Q[a] + alpha * (r - Q[a])
        # Progress log at 25%, 50%, 75% and end.
        if (ep + 1) in {max(1, episodes // 4), max(1, episodes // 2), max(1, (3 * episodes) // 4), episodes}:
            best_idx = max(range(K), key=lambda i: Q[i])
            print(
                f"[train] ep={ep+1}/{episodes} best={action_set[best_idx].label} "
                f"Q={Q[best_idx]:.2f} events={events}",
            )

    elapsed = time.time() - start
    ttff_s = None if first_event_t is None else (first_event_t - start)
    best_idx = max(range(K), key=lambda i: Q[i])
    ram.close()
    return {
        "mode": "train",
        "seed": seed,
        "episodes": episodes,
        "steps_per_ep": steps_per_ep,
        "epsilon": epsilon,
        "alpha": alpha,
        "elapsed_s": elapsed,
        "total_steps": total_steps,
        "events": events,
        "events_per_min": (events / (elapsed / 60.0)) if elapsed > 0 else None,
        "ttff_s": ttff_s,
        "ttff_steps": first_event_step,
        "ttff_action": first_event_action,
        "coverage_actions": len(coverage),
        "latency_tail": latency_tail(latencies),
        "best_index": best_idx,
        "best_action": action_set[best_idx].label,
        "Q": Q,
        "pulls": pulls,
        "action_labels": [v.label for v in action_set],
    }


def eval_best(
    ram_exe: str,
    config: str,
    action_set: List[PatternVariant],
    best_action_label: str,
    eval_steps: int,
    addr_base: int,
    stride: int,
    n_addrs: int,
    ctx: int,
    spike_thresh: int,
) -> Dict[str, Any]:
    """Evaluate one pattern variant for a number of steps and gather stats."""
    label_to_idx = {v.label: i for i, v in enumerate(action_set)}
    if best_action_label not in label_to_idx:
        raise ValueError(
            f"best_action_label '{best_action_label}' not found in current action_set"
        )
    best_variant = action_set[label_to_idx[best_action_label]]
    ram = RamulatorProc(ram_exe, config)
    ram.start()
    start = time.time()
    first_event_t: Optional[float] = None
    first_event_step: Optional[int] = None
    events = 0
    latencies: List[int] = []
    coverage = {best_variant.label}
    for step in range(1, eval_steps + 1):
        accepted, lat, ev, _ = step_variant(
            ram,
            best_variant,
            addr_base=addr_base,
            stride=stride,
            n_addrs=n_addrs,
            ctx=ctx,
            spike_thresh=spike_thresh,
        )
        if lat is not None:
            latencies.append(lat)
        if ev:
            events += 1
            if first_event_t is None:
                first_event_t = time.time()
                first_event_step = step
    elapsed = time.time() - start
    ttff_s = None if first_event_t is None else (first_event_t - start)
    ram.close()
    return {
        "mode": "eval",
        "best_action": best_variant.label,
        "elapsed_s": elapsed,
        "eval_steps": eval_steps,
        "events": events,
        "events_per_min": (events / (elapsed / 60.0)) if elapsed > 0 else None,
        "ttff_s": ttff_s,
        "ttff_steps": first_event_step,
        "coverage_actions": len(coverage),
        "latency_tail": latency_tail(latencies),
    }


###############################################################################
# F) CLI
###############################################################################

def make_action_set(kind: str) -> List[PatternVariant]:
    """Return the appropriate action set by type."""
    variants = all_variants(include_zero_weight=False)
    if kind == "all":
        return variants
    if kind == "top64":
        return top_k_by_weight(variants, 64)
    raise ValueError("action‑set must be 'top64' or 'all'")


def main() -> None:
    ap = argparse.ArgumentParser(description="GSAT RL and baseline runner")
    ap.add_argument(
        "--exe",
        required=True,
        help="Path to the interactive driver executable (must print READY)",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config used by the driver",
    )
    ap.add_argument(
        "--action‑set",
        default="top64",
        choices=["top64", "all"],
        help="Use top64 weighted variants or all selectable variants",
    )
    ap.add_argument(
        "--addr‑base",
        type=lambda x: int(x, 0),
        default=0x100000,
        help="Base address (hex or decimal)",
    )
    ap.add_argument("--stride", type=int, default=64, help="Stride in bytes")
    ap.add_argument("--n‑addrs", type=int, default=1024, help="Address window size")
    ap.add_argument("--ctx", type=int, default=0, help="Context ID")
    ap.add_argument(
        "--spike‑thresh",
        type=int,
        default=4000,
        help="Latency threshold in cycles to consider a spike",
    )

    subparsers = ap.add_subparsers(dest="cmd", required=True)

    b = subparsers.add_parser("baseline", help="Run baseline exhaustive test")
    b.add_argument(
        "--steps‑per‑action", type=int, default=200, help="Steps per action variant"
    )

    t = subparsers.add_parser("train", help="Train RL agent")
    t.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    t.add_argument("--steps‑per‑ep", type=int, default=50, help="Steps per episode")
    t.add_argument("--epsilon", type=float, default=0.2, help="Exploration rate")
    t.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    t.add_argument("--seed", type=int, default=1, help="Random seed")
    t.add_argument("--out", default="rl_model.json", help="Output model file")

    e = subparsers.add_parser("eval", help="Evaluate a saved RL model")
    e.add_argument("--model", required=True, help="Model JSON from training")
    e.add_argument("--eval‑steps", type=int, default=2000, help="Steps for evaluation")

    c = subparsers.add_parser("compare", help="Baseline vs RL comparison")
    c.add_argument(
        "--steps‑per‑action", type=int, default=200, help="Steps per action in baseline"
    )
    c.add_argument("--episodes", type=int, default=200)
    c.add_argument("--steps‑per‑ep", type=int, default=50)
    c.add_argument("--epsilon", type=float, default=0.2)
    c.add_argument("--alpha", type=float, default=0.1)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--eval‑steps", type=int, default=2000)
    c.add_argument("--out", default="compare_report.json")

    args = ap.parse_args()
    action_set = make_action_set(args.action‑set)

    # Execute selected subcommand.
    if args.cmd == "baseline":
        report = run_baseline(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            steps_per_action=args.steps‑per‑action,
            addr_base=args.addr‑base,
            stride=args.stride,
            n_addrs=args.n‑addrs,
            ctx=args.ctx,
            spike_thresh=args.spike‑thresh,
        )
        print(json.dumps(report, indent=2))
        return

    if args.cmd == "train":
        report = train_bandit(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            episodes=args.episodes,
            steps_per_ep=args.steps_per_ep,
            epsilon=args.epsilon,
            alpha=args.alpha,
            addr_base=args.addr‑base,
            stride=args.stride,
            n_addrs=args.n‑addrs,
            ctx=args.ctx,
            spike_thresh=args.spike‑thresh,
            seed=args.seed,
        )
        # Dump minimal state for evaluation.
        model = {
            "meta": {
                "exe": args.exe,
                "config": args.config,
                "action_set": args.action‑set,
                "spike_thresh": args.spike‑thresh,
                "addr_base": args.addr‑base,
                "stride": args.stride,
                "n_addrs": args.n‑addrs,
                "ctx": args.ctx,
            },
            "train_report": {k: report[k] for k in report if k not in ("Q", "pulls", "action_labels")},
            "Q": report["Q"],
            "pulls": report["pulls"],
            "action_labels": report["action_labels"],
            "best_action": report["best_action"],
        }
        with open(args.out, "w") as f:
            json.dump(model, f, indent=2)
        print(f"\nSaved model: {args.out}")
        print(f"Best action: {report['best_action']}")
        return

    if args.cmd == "eval":
        # Load model
        with open(args.model) as f:
            model = json.load(f)
        best_action = model["best_action"]
        report = eval_best(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            best_action_label=best_action,
            eval_steps=args.eval_steps,
            addr_base=args.addr‑base,
            stride=args.stride,
            n_addrs=args.n‑addrs,
            ctx=args.ctx,
            spike_thresh=args.spike‑thresh,
        )
        print(json.dumps(report, indent=2))
        return

    if args.cmd == "compare":
        # Baseline
        base = run_baseline(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            steps_per_action=args.steps‑per‑action,
            addr_base=args.addr‑base,
            stride=args.stride,
            n_addrs=args.n‑addrs,
            ctx=args.ctx,
            spike_thresh=args.spike‑thresh,
        )
        print("\n[compare] baseline done")
        # Train
        train = train_bandit(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            episodes=args.episodes,
            steps_per_ep=args.steps_per_ep,
            epsilon=args.epsilon,
            alpha=args.alpha,
            addr_base=args.addr‑base,
            stride=args.stride,
            n_addrs=args.n‑addrs,
            ctx=args.ctx,
            spike_thresh=args.spike‑thresh,
            seed=args.seed,
        )
        best_action = train["best_action"]
        print(f"\n[compare] train done, best = {best_action}")
        # Eval
        ev = eval_best(
            ram_exe=args.exe,
            config=args.config,
            action_set=action_set,
            best_action_label=best_action,
            eval_steps=args.eval_steps,
            addr_base=args.addr‑base,
            stride=args.stride,
            n_addrs=args.n‑addrs,
            ctx=args.ctx,
            spike_thresh=args.spike‑thresh,
        )
        print("\n[compare] eval done")
        summary = {
            "baseline_ttff_s": base["ttff_s"],
            "rl_train_ttff_s": train["ttff_s"],
            "rl_eval_ttff_s": ev["ttff_s"],
            "baseline_events_per_min": base["events_per_min"],
            "rl_train_events_per_min": train["events_per_min"],
            "rl_eval_events_per_min": ev["events_per_min"],
            "best_action": best_action,
        }
        report = {
            "baseline": base,
            "train": {k: train[k] for k in train if k not in ("Q", "pulls", "action_labels")},
            "eval_best": ev,
            "summary": summary,
        }
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved compare report: {args.out}")
        print(json.dumps(summary, indent=2))
        return

    # Should never reach here
    raise RuntimeError("No valid subcommand provided")


if __name__ == "__main__":
    main()






#!/usr/bin/env python3
"""
rlmem_satonly_full.py

SAT/GSAT-only DRAM fault discovery prototype.

Run:
  python rlmem_satonly_full.py baseline --episodes 50
  python rlmem_satonly_full.py train --episodes 400 --out policy.json
  python rlmem_satonly_full.py eval --episodes 100 --policy policy.json
  python rlmem_satonly_full.py curve --episodes 30 --policy policy.json

No memtest. No external tools. Only Python stdlib.
"""

from __future__ import annotations
import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from array import array

# ----------------------------
# Utilities
# ----------------------------

def u32(x: int) -> int:
    return x & 0xFFFFFFFF

def popcount32(x: int) -> int:
    return (x & 0xFFFFFFFF).bit_count()

# ----------------------------
# GSAT patterns (from your pattern.cc snippet)
# ----------------------------

walkingOnes_data = [
  0x00000001, 0x00000002, 0x00000004, 0x00000008,
  0x00000010, 0x00000020, 0x00000040, 0x00000080,
  0x00000100, 0x00000200, 0x00000400, 0x00000800,
  0x00001000, 0x00002000, 0x00004000, 0x00008000,
  0x00010000, 0x00020000, 0x00040000, 0x00080000,
  0x00100000, 0x00200000, 0x00400000, 0x00800000,
  0x01000000, 0x02000000, 0x04000000, 0x08000000,
  0x10000000, 0x20000000, 0x40000000, 0x80000000,
  0x40000000, 0x20000000, 0x10000000, 0x08000000,
  0x04000000, 0x02000000, 0x01000000, 0x00800000,
  0x00400000, 0x00200000, 0x00100000, 0x00080000,
  0x00040000, 0x00020000, 0x00010000, 0x00008000,
  0x00004000, 0x00002000, 0x00001000, 0x00000800,
  0x00000400, 0x00000200, 0x00000100, 0x00000080,
  0x00000040, 0x00000020, 0x00000010, 0x00000008,
  0x00000004, 0x00000002, 0x00000001, 0x00000000
]

walkingInvOnes_data = [
  0x00000001, 0xfffffffe, 0x00000002, 0xfffffffd,
  0x00000004, 0xfffffffb, 0x00000008, 0xfffffff7,
  0x00000010, 0xffffffef, 0x00000020, 0xffffffdf,
  0x00000040, 0xffffffbf, 0x00000080, 0xffffff7f,
  0x00000100, 0xfffffeff, 0x00000200, 0xfffffdff,
  0x00000400, 0xfffffbff, 0x00000800, 0xfffff7ff,
  0x00001000, 0xffffefff, 0x00002000, 0xffffdfff,
  0x00004000, 0xffffbfff, 0x00008000, 0xffff7fff,
  0x00010000, 0xfffeffff, 0x00020000, 0xfffdffff,
  0x00040000, 0xfffbffff, 0x00080000, 0xfff7ffff,
  0x00100000, 0xffefffff, 0x00200000, 0xffdfffff,
  0x00400000, 0xffbfffff, 0x00800000, 0xff7fffff,
  0x01000000, 0xfeffffff, 0x02000000, 0xfdffffff,
  0x04000000, 0xfbffffff, 0x08000000, 0xf7ffffff,
  0x10000000, 0xefffffff, 0x20000000, 0xdfffffff,
  0x40000000, 0xbfffffff, 0x80000000, 0x7fffffff,
  0x40000000, 0xbfffffff, 0x20000000, 0xdfffffff,
  0x10000000, 0xefffffff, 0x08000000, 0xf7ffffff,
  0x04000000, 0xfbffffff, 0x02000000, 0xfdffffff,
  0x01000000, 0xfeffffff, 0x00800000, 0xff7fffff,
  0x00400000, 0xffbfffff, 0x00200000, 0xffdfffff,
  0x00100000, 0xffefffff, 0x00080000, 0xfff7ffff,
  0x00040000, 0xfffbffff, 0x00020000, 0xfffdffff,
  0x00010000, 0xfffeffff, 0x00008000, 0xffff7fff,
  0x00004000, 0xffffbfff, 0x00002000, 0xffffdfff,
  0x00001000, 0xffffefff, 0x00000800, 0xfffff7ff,
  0x00000400, 0xfffffbff, 0x00000200, 0xfffffdff,
  0x00000100, 0xfffffeff, 0x00000080, 0xffffff7f,
  0x00000040, 0xffffffbf, 0x00000020, 0xffffffdf,
  0x00000010, 0xffffffef, 0x00000008, 0xfffffff7,
  0x00000004, 0xfffffffb, 0x00000002, 0xfffffffd,
  0x00000001, 0xfffffffe, 0x00000000, 0xffffffff
]

walkingZeros_data = [
  0xfffffffe, 0xfffffffd, 0xfffffffb, 0xfffffff7,
  0xffffffef, 0xffffffdf, 0xffffffbf, 0xffffff7f,
  0xfffffeff, 0xfffffdff, 0xfffffbff, 0xfffff7ff,
  0xffffefff, 0xffffdfff, 0xffffbfff, 0xffff7fff,
  0xfffeffff, 0xfffdffff, 0xfffbffff, 0xfff7ffff,
  0xffefffff, 0xffdfffff, 0xffbfffff, 0xff7fffff,
  0xfeffffff, 0xfdffffff, 0xfbffffff, 0xf7ffffff,
  0xefffffff, 0xdfffffff, 0xbfffffff, 0x7fffffff,
  0xbfffffff, 0xdfffffff, 0xefffffff, 0xf7ffffff,
  0xfbffffff, 0xfdffffff, 0xfeffffff, 0xff7fffff,
  0xffbfffff, 0xffdfffff, 0xffefffff, 0xfff7ffff,
  0xfffbffff, 0xfffdffff, 0xfffeffff, 0xffff7fff,
  0xffffbfff, 0xffffdfff, 0xffffefff, 0xfffff7ff,
  0xfffffbff, 0xfffffdff, 0xfffffeff, 0xffffff7f,
  0xffffffbf, 0xffffffdf, 0xffffffef, 0xfffffff7,
  0xfffffffb, 0xfffffffd, 0xfffffffe, 0xffffffff
]

PATTERNS: Dict[str, List[int]] = {
    "walkingOnes": walkingOnes_data,
    "walkingInvOnes": walkingInvOnes_data,
    "walkingZeros": walkingZeros_data,
    "OneZero": [0x00000000, 0xffffffff],
    "JustZero": [0x00000000, 0x00000000],
    "JustOne": [0xffffffff, 0xffffffff],
    "JustFive": [0x55555555, 0x55555555],
    "JustA": [0xaaaaaaaa, 0xaaaaaaaa],
    "FiveA": [0x55555555, 0xaaaaaaaa],
    "FiveA8": [0x5aa5a55a, 0xa55a5aa5, 0xa55a5aa5, 0x5aa5a55a],
    "Long8b10b": [0x16161616, 0x16161616],
    "Short8b10b": [0xb5b5b5b5, 0xb5b5b5b5],
    "Checker8b10b": [0xb5b5b5b5, 0x4a4a4a4a],
    "Five7": [0x55555557, 0x55575555],
    "Zero2fd": [0x00020002, 0xfffdfffd],
}

BUSSHIFT = {32: 0, 64: 1, 128: 2, 256: 3}

def build_sat_variants() -> List[Dict]:
    out = []
    for name, vals in PATTERNS.items():
        for bw in (32, 64, 128, 256):
            for inv in (False, True):
                out.append({
                    "pattern": name,
                    "vals": vals,
                    "buswidth": bw,
                    "busshift": BUSSHIFT[bw],
                    "invert": inv,
                    "variant_name": f"{name}{'~' if inv else ''}{bw}",
                })
    return out

# ----------------------------
# Fault Injection Model
# ----------------------------

@dataclass
class FaultConfig:
    seed: int = 0

    # Make defaults non-zero so you don't get "None:0"
    stuck_prob: float = 2e-6            # stuck bits density
    intermittent_prob: float = 2e-7     # random flips
    retention_prob: float = 1e-6        # cells that will flip after time
    retention_flip_after: int = 80_000  # ticks before retention can trigger

    # Hammer-like
    cacheline_bytes: int = 64
    hammer_threshold: int = 12000       # accesses before hammer risk
    hammer_flip_prob: float = 3e-4      # flip prob once above threshold

class FaultModel:
    """
    Simplified DRAM-like faults.

    We maintain:
      - stuck cells: a chosen bit is stuck at 0 or 1
      - intermittent: random bit flip sometimes
      - retention: after time threshold, flip a specific bit in selected cells
      - hammer-like: repeated access to same cacheline triggers bit flip occasionally
    """
    def __init__(self, cfg: FaultConfig, mem_words: int):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.mem_words = mem_words
        self.time = 0
        self.last_fault_type = "none"

        # stuck: idx -> (bit, stuck_to_1 bool)
        self.stuck: Dict[int, Tuple[int, bool]] = {}
        # retention: idx -> (bit, flip_time)
        self.retention: Dict[int, Tuple[int, int]] = {}

        for i in range(mem_words):
            if self.rng.random() < cfg.stuck_prob:
                bit = self.rng.randrange(32)
                stuck_to_1 = bool(self.rng.getrandbits(1))
                self.stuck[i] = (bit, stuck_to_1)
            if self.rng.random() < cfg.retention_prob:
                bit = self.rng.randrange(32)
                flip_t = cfg.retention_flip_after + self.rng.randrange(0, cfg.retention_flip_after)
                self.retention[i] = (bit, flip_t)

        # hammer counters (by cacheline index)
        self.cl_access: Dict[int, int] = {}

    def on_time(self, t: int):
        self.time = t

    def on_access(self, addr: int):
        # cacheline index
        cl = addr // self.cfg.cacheline_bytes
        self.cl_access[cl] = self.cl_access.get(cl, 0) + 1

    def apply_on_read(self, addr: int, idx: int, val: int) -> int:
        self.last_fault_type = "none"
        v = val

        # stuck
        if idx in self.stuck:
            bit, stuck_to_1 = self.stuck[idx]
            if stuck_to_1:
                v = u32(v | (1 << bit))
            else:
                v = u32(v & ~(1 << bit))
            self.last_fault_type = "stuck"
            return v

        # retention
        if idx in self.retention:
            bit, flip_t = self.retention[idx]
            if self.time >= flip_t:
                v = u32(v ^ (1 << bit))
                self.last_fault_type = "retention"
                return v

        # hammer-like (based on access pressure to the cacheline)
        cl = addr // self.cfg.cacheline_bytes
        if self.cl_access.get(cl, 0) >= self.cfg.hammer_threshold:
            if self.rng.random() < self.cfg.hammer_flip_prob:
                bit = self.rng.randrange(32)
                v = u32(v ^ (1 << bit))
                self.last_fault_type = "hammer"
                return v

        # intermittent
        if self.rng.random() < self.cfg.intermittent_prob:
            bit = self.rng.randrange(32)
            v = u32(v ^ (1 << bit))
            self.last_fault_type = "intermittent"
            return v

        return v

# ----------------------------
# DRAM Backend: read32 / write32
# ----------------------------

@dataclass
class ErrorEvent:
    addr: int
    expected: int
    observed: int
    bitmask: int
    fault_type: str

class DRAMBackend:
    """
    Storage = array('I') of 32-bit words.
    Time model: each read/write costs 1 tick.
    """
    def __init__(self, mem_bytes: int, fault: FaultModel):
        assert mem_bytes % 4 == 0
        self.mem_words = mem_bytes // 4
        self.mem = array('I', [0] * self.mem_words)
        self.fault = fault
        self.time = 0

    def tick(self, cycles: int):
        self.time += int(cycles)
        self.fault.on_time(self.time)

    def write32(self, addr: int, val: int):
        idx = (addr >> 2) % self.mem_words
        self.mem[idx] = u32(val)
        self.time += 1
        self.fault.on_time(self.time)
        self.fault.on_access(addr)

    def read32(self, addr: int) -> int:
        idx = (addr >> 2) % self.mem_words
        self.time += 1
        self.fault.on_time(self.time)
        self.fault.on_access(addr)
        raw = int(self.mem[idx])
        return int(self.fault.apply_on_read(addr, idx, raw))

# ----------------------------
# SAT runner: fill + verify
# ----------------------------

def sat_fill_verify(backend: DRAMBackend, base_addr: int, size_bytes: int,
                    vals: List[int], busshift: int, invert: bool) -> List[ErrorEvent]:
    words = size_bytes // 4
    errors: List[ErrorEvent] = []

    # write phase
    for w in range(words):
        pi = (w >> busshift) % len(vals)
        v = vals[pi]
        if invert:
            v = u32(~v)
        backend.write32(base_addr + w * 4, v)

    # verify phase
    for w in range(words):
        pi = (w >> busshift) % len(vals)
        exp = vals[pi]
        if invert:
            exp = u32(~exp)
        addr = base_addr + w * 4
        got = backend.read32(addr)
        if got != exp:
            errors.append(ErrorEvent(
                addr=addr,
                expected=exp,
                observed=got,
                bitmask=u32(exp ^ got),
                fault_type=backend.fault.last_fault_type,
            ))

    return errors

# ----------------------------
# Signature / Coverage
# ----------------------------

def make_signature(fault_type: str, addr: int, bitmask: int) -> Tuple[str, int, int]:
    # bucket address to 4KB pages; bucket bitmask by popcount + low bits
    addr_bucket = addr // 4096
    bm_bucket = (popcount32(bitmask) << 8) | (bitmask & 0xFF)
    return (fault_type, addr_bucket, bm_bucket)

# ----------------------------
# Environment
# ----------------------------

@dataclass
class StepResult:
    dt: int
    errors: int
    new_sigs: int
    first_error_time: Optional[int]

class SatEnv:
    """
    SAT-only RL environment.
    Action = choose SAT variant (0..119).
    """
    def __init__(self, mem_mb: int, region_kb: int, seed: int):
        self.mem_bytes = mem_mb * 1024 * 1024
        self.region_bytes = min(region_kb * 1024, self.mem_bytes)
        self.seed = seed
        self.sat_vars = build_sat_variants()

        self.backend: Optional[DRAMBackend] = None
        self.seen_sigs: set = set()
        self.first_error_time: Optional[int] = None
        self.action_newsig: Dict[int, int] = {}

    def reset(self, fault_seed: int, cfg_override: Optional[dict] = None):
        cfg = FaultConfig(seed=fault_seed)
        if cfg_override:
            for k, v in cfg_override.items():
                setattr(cfg, k, v)
        fault = FaultModel(cfg, mem_words=self.mem_bytes // 4)
        self.backend = DRAMBackend(self.mem_bytes, fault)
        self.seen_sigs = set()
        self.first_error_time = None
        self.action_newsig = {i: 0 for i in range(len(self.sat_vars))}

    def num_actions(self) -> int:
        return len(self.sat_vars)

    def step(self, action_id: int) -> StepResult:
        assert self.backend is not None
        t0 = self.backend.time

        v = self.sat_vars[action_id]
        errs = sat_fill_verify(
            self.backend,
            base_addr=0,
            size_bytes=self.region_bytes,
            vals=v["vals"],
            busshift=v["busshift"],
            invert=v["invert"],
        )

        new = 0
        for e in errs:
            sig = make_signature(e.fault_type, e.addr, e.bitmask)
            if sig not in self.seen_sigs:
                self.seen_sigs.add(sig)
                new += 1

        if new:
            self.action_newsig[action_id] += new

        if errs and self.first_error_time is None:
            self.first_error_time = self.backend.time

        return StepResult(
            dt=self.backend.time - t0,
            errors=len(errs),
            new_sigs=new,
            first_error_time=self.first_error_time
        )

# ----------------------------
# RL: Thompson-sampling bandit for discovery
# ----------------------------

def bandit_train(episodes: int, steps: int, mem_mb: int, region_kb: int,
                 seed: int, out_path: str, cfg_override: Optional[dict]) -> dict:
    rng = random.Random(seed)
    env = SatEnv(mem_mb=mem_mb, region_kb=region_kb, seed=seed)

    # Beta priors per action
    a = [1.0] * env.num_actions()
    b = [1.0] * env.num_actions()

    def reward(res: StepResult, first_before: bool) -> float:
        # discovery reward + TTFF bonus - time penalty
        r = 10.0 * res.new_sigs
        if res.errors > 0 and first_before:
            r += 50.0
        r -= 1e-5 * res.dt
        return r

    for ep in range(episodes):
        env.reset(fault_seed=ep, cfg_override=cfg_override)
        for _ in range(steps):
            # Thompson sample
            best_aid = 0
            best_theta = -1.0
            for aid in range(env.num_actions()):
                theta = rng.betavariate(a[aid], b[aid])
                if theta > best_theta:
                    best_theta = theta
                    best_aid = aid

            first_before = (env.first_error_time is None)
            res = env.step(best_aid)
            r = reward(res, first_before)

            # Convert to Bernoulli success: any positive reward => success
            if r > 0:
                a[best_aid] += 1.0
            else:
                b[best_aid] += 1.0

    means = [a[i] / (a[i] + b[i]) for i in range(env.num_actions())]
    ranked = sorted(range(env.num_actions()), key=lambda i: means[i], reverse=True)

    policy = {
        "type": "topk_cycle",
        "topk": 20,
        "actions": ranked[:20],
        "posterior_mean_top50": {str(i): means[i] for i in ranked[:50]},
        "mem_mb": mem_mb,
        "region_kb": region_kb,
        "train_episodes": episodes,
        "train_steps": steps,
        "cfg_override": cfg_override or {},
    }

    with open(out_path, "w") as f:
        json.dump(policy, f, indent=2)

    # Print top variants
    print(f"[train] wrote policy: {out_path}")
    print("[train] top SAT variants:")
    for i in ranked[:10]:
        print(f"  aid={i:3d} mean={means[i]:.3f}  {env.sat_vars[i]['variant_name']}")

    return policy

# ----------------------------
# Evaluation
# ----------------------------

def summarize(ttff: List[Optional[int]], cov: List[int]) -> dict:
    tt = [x for x in ttff if x is not None]
    out = {}
    out["episodes"] = len(ttff)
    out["ttff_none"] = sum(1 for x in ttff if x is None)
    out["ttff_median"] = (sorted(tt)[len(tt)//2] if tt else None)
    out["cov_median"] = sorted(cov)[len(cov)//2] if cov else 0
    out["cov_p95"] = sorted(cov)[max(0, math.ceil(0.95*len(cov))-1)] if cov else 0
    return out

def run_random_baseline(episodes: int, steps: int, mem_mb: int, region_kb: int,
                        seed: int, cfg_override: Optional[dict]) -> dict:
    rng = random.Random(seed)
    env = SatEnv(mem_mb=mem_mb, region_kb=region_kb, seed=seed)
    ttff, cov = [], []

    for ep in range(episodes):
        env.reset(fault_seed=ep, cfg_override=cfg_override)
        for _ in range(steps):
            env.step(rng.randrange(env.num_actions()))
        ttff.append(env.first_error_time)
        cov.append(len(env.seen_sigs))

    out = summarize(ttff, cov)
    print("[baseline] random_sat:", out)
    return out

def eval_policy(episodes: int, steps: int, mem_mb: int, region_kb: int,
                policy_path: str, cfg_override: Optional[dict]) -> dict:
    with open(policy_path, "r") as f:
        pol = json.load(f)
    actions = pol["actions"]

    env = SatEnv(mem_mb=mem_mb, region_kb=region_kb, seed=0)
    ttff, cov = [], []
    agg_newsig = None

    for ep in range(episodes):
        env.reset(fault_seed=ep, cfg_override=cfg_override)
        for t in range(steps):
            env.step(actions[t % len(actions)])
        ttff.append(env.first_error_time)
        cov.append(len(env.seen_sigs))
        if agg_newsig is None:
            agg_newsig = env.action_newsig.copy()
        else:
            for k, v in env.action_newsig.items():
                agg_newsig[k] += v

    out = summarize(ttff, cov)
    print("[eval] policy:", out)

    # leaderboard
    ranked = sorted(agg_newsig.items(), key=lambda kv: kv[1], reverse=True)
    print("[eval] top patterns by NEW signature discoveries:")
    for aid, cnt in ranked[:10]:
        print(f"  aid={aid:3d} new_sigs={cnt:6d}  {env.sat_vars[aid]['variant_name']}")

    return out

def discovery_curve(episodes: int, steps: int, mem_mb: int, region_kb: int,
                    policy_path: Optional[str], seed: int,
                    cfg_override: Optional[dict]):
    rng = random.Random(seed)

    def run_one(actions: Optional[List[int]], ep_seed: int) -> Tuple[List[int], List[int]]:
        env = SatEnv(mem_mb=mem_mb, region_kb=region_kb, seed=0)
        env.reset(fault_seed=ep_seed, cfg_override=cfg_override)
        times = []
        covs = []
        for t in range(steps):
            if actions is None:
                aid = rng.randrange(env.num_actions())
            else:
                aid = actions[t % len(actions)]
            env.step(aid)
            times.append(env.backend.time)  # type: ignore
            covs.append(len(env.seen_sigs))
        return times, covs

    pol_actions = None
    if policy_path:
        with open(policy_path, "r") as f:
            pol_actions = json.load(f)["actions"]

    # aggregate curves by step index
    rand_T, rand_C = [], []
    pol_T, pol_C = [], []

    for ep in range(episodes):
        t, c = run_one(None, ep)
        rand_T.append(t); rand_C.append(c)
        if pol_actions:
            t2, c2 = run_one(pol_actions, ep)
            pol_T.append(t2); pol_C.append(c2)

    def avg(mat: List[List[int]]) -> List[float]:
        n = len(mat)
        m = len(mat[0])
        return [sum(mat[i][j] for i in range(n))/n for j in range(m)]

    Tm = avg(rand_T)
    Cm = avg(rand_C)
    auc_rand = 0.0
    for i in range(1, len(Tm)):
        auc_rand += 0.5 * (Cm[i] + Cm[i-1]) * (Tm[i] - Tm[i-1])

    print(f"[curve] random_sat: AUC={auc_rand:.2f} final_cov={Cm[-1]:.2f} final_time={Tm[-1]:.0f}")

    if pol_actions:
        Tp = avg(pol_T)
        Cp = avg(pol_C)
        auc_pol = 0.0
        for i in range(1, len(Tp)):
            auc_pol += 0.5 * (Cp[i] + Cp[i-1]) * (Tp[i] - Tp[i-1])
        print(f"[curve] policy:     AUC={auc_pol:.2f} final_cov={Cp[-1]:.2f} final_time={Tp[-1]:.0f}")

# ----------------------------
# CLI
# ----------------------------

def parse_cfg_override(args) -> dict:
    """
    Optional tuning knobs to avoid None:0 or to make TTFF more/less sensitive.
    """
    d = {}
    if args.stuck_prob is not None: d["stuck_prob"] = args.stuck_prob
    if args.intermittent_prob is not None: d["intermittent_prob"] = args.intermittent_prob
    if args.retention_prob is not None: d["retention_prob"] = args.retention_prob
    if args.retention_flip_after is not None: d["retention_flip_after"] = args.retention_flip_after
    if args.hammer_threshold is not None: d["hammer_threshold"] = args.hammer_threshold
    if args.hammer_flip_prob is not None: d["hammer_flip_prob"] = args.hammer_flip_prob
    return d

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--mem-mb", type=int, default=64)
        p.add_argument("--region-kb", type=int, default=256, help="bytes tested per action = region-kb")
        p.add_argument("--episodes", type=int, default=50)
        p.add_argument("--steps", type=int, default=15)
        p.add_argument("--seed", type=int, default=0)

        # fault tuning
        p.add_argument("--stuck-prob", type=float, default=None)
        p.add_argument("--intermittent-prob", type=float, default=None)
        p.add_argument("--retention-prob", type=float, default=None)
        p.add_argument("--retention-flip-after", type=int, default=None)
        p.add_argument("--hammer-threshold", type=int, default=None)
        p.add_argument("--hammer-flip-prob", type=float, default=None)

    p0 = sub.add_parser("baseline")
    add_common(p0)

    p1 = sub.add_parser("train")
    add_common(p1)
    p1.add_argument("--out", type=str, default="policy.json")

    p2 = sub.add_parser("eval")
    add_common(p2)
    p2.add_argument("--policy", type=str, required=True)

    p3 = sub.add_parser("curve")
    add_common(p3)
    p3.add_argument("--policy", type=str, default=None)

    args = ap.parse_args()
    cfg_override = parse_cfg_override(args)

    if args.cmd == "baseline":
        run_random_baseline(args.episodes, args.steps, args.mem_mb, args.region_kb, args.seed, cfg_override)

    elif args.cmd == "train":
        bandit_train(args.episodes, args.steps, args.mem_mb, args.region_kb, args.seed, args.out, cfg_override)

    elif args.cmd == "eval":
        eval_policy(args.episodes, args.steps, args.mem_mb, args.region_kb, args.policy, cfg_override)

    elif args.cmd == "curve":
        discovery_curve(args.episodes, args.steps, args.mem_mb, args.region_kb, args.policy, args.seed, cfg_override)

if __name__ == "__main__":
    main()



import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Model ---
MODEL = "Qwen/Qwen1.5-MoE-A2.7B"
dtype = torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL} on {device}…")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=dtype,
    device_map="auto",          # or {"": "cpu"} if you want DDR-only
    trust_remote_code=True,
)

# --- Prefill test ---
def test_prefill(context_len):
    prompt = "Data " * context_len
    inputs = tok(prompt, return_tensors="pt").to(device)

    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats()

    print(f"\nRunning prefill for {context_len} tokens…")
    with torch.no_grad():
        _ = model(**inputs, use_cache=True)

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Prefill done. Peak VRAM usage: {peak_gb:.2f} GB")

# --- Sweep different context lengths ---
for L in [2048, 4096, 8192, 16384]:
    try:
        test_prefill(L)
    except RuntimeError as e:
        print(f"OOM at {L} tokens -> {e}")
        break





import torch, gc
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen1.5-MoE-A2.7B"

# Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL)

# Long dummy context
context_len = 8192   # adjust to test 4k, 8k, 16k
prompt = "Data " * context_len

# Sampling params: no generation, just prefill
sp = SamplingParams(max_tokens=0)

# vLLM engine
kv_cfg = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
llm = LLM(
    model=MODEL,
    dtype="half",
    max_model_len=context_len,
    gpu_memory_utilization=0.40,
    cpu_offload_gb=64,         # offload experts if needed
    kv_transfer_config=kv_cfg,
    enforce_eager=True,
)

# Run prefill
torch.cuda.reset_peak_memory_stats()
gc.collect()

print(f"Running prefill for context length {context_len}...")
outs = llm.generate([prompt], sp)   # will only run prefill, no new tokens

peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Prefill done. Peak VRAM = {peak_gb:.2f} GB")




-*- coding: utf-8 -*-
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"
GPU      = "cuda:0"
DTYPE    = torch.bfloat16          # or torch.float16 if needed
VRAM_BUDGET = "20GiB"
DRAM_BUDGET = "200GiB"

# 0) Get a LOCAL checkpoint folder (required by load_checkpoint_and_dispatch)
ckpt_dir = snapshot_download(MODEL_ID)

# 1) Build meta skeleton
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# IMPORTANT: use the model’s own no-split classes if provided
no_split = getattr(empty_model, "_no_split_modules", None)
if not no_split:
    # Fallback — try common names used by this repo (adjust if needed)
    no_split = ["DeepseekDecoderLayer", "DeepseekBlock", "Block"]

# 2) Infer device map (experts -> CPU storage)
device_map = infer_auto_device_map(
    empty_model,
    max_memory={0: VRAM_BUDGET, "cpu": DRAM_BUDGET},
    no_split_module_classes=no_split,
)

for name in list(device_map.keys()):
    if "experts" in name:
        device_map[name] = "cpu"

print("== device_map (first ~25) ==")
for i, (k, v) in enumerate(device_map.items()):
    if i >= 25: break
    print(f"{k} -> {v}")
print("...")

# 3) Stream weights into the meta skeleton (materializes tensors!)
model = load_checkpoint_and_dispatch(
    empty_model,
    checkpoint=ckpt_dir,                        # MUST be a local path
    device_map=device_map,
    no_split_module_classes=no_split,
    dtype=DTYPE,
    offload_folder=None,                        # no disk
)

# ---- Helper: check for any lingering meta tensors
def first_meta_param(mod: nn.Module):
    for n, p in mod.named_parameters(recurse=True):
        if getattr(p, "is_meta", False) or (hasattr(p, "device") and p.device.type == "meta"):
            return n
    for n, b in mod.named_buffers(recurse=True):
        if b is not None and (getattr(b, "is_meta", False) or (hasattr(b, "device") and b.device.type == "meta")):
            return n
    return None

meta_name = first_meta_param(model)
if meta_name:
    # Some trust_remote_code models lazily build submodules; do a tiny warm-up to instantiate them
    print(f"[warn] Found meta tensor at: {meta_name}. Doing a 1-token warmup to materialize lazies...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    with torch.no_grad():
        dummy = tok(".", return_tensors="pt").to(GPU)
        _ = model(**dummy)  # forces construction/loading paths
    meta_name = first_meta_param(model)
    if meta_name:
        raise RuntimeError(f"Still meta after warmup: {meta_name}. "
                           f"Double-check no_split={no_split} and that checkpoint folder is correct.")

# 4) Patch experts: store on CPU, compute on CUDA (NO GPU cache)
def patch_expert_forward_no_cache(expert: nn.Module, gpu_device: str = "cuda:0"):
    """
    Keep expert object/type intact (router-safe).
    On each forward:
      - build a temporary EMPTY GPU clone (to_empty)
      - load CPU state_dict into it
      - run original forward bound to the GPU clone
      - free the clone
    """
    original_forward = expert.forward
    cuda_dev = torch.device(gpu_device)

    # Re-validate: no meta tensors inside expert
    bad = first_meta_param(expert)
    if bad:
        raise RuntimeError(f"Expert still meta at {bad}. Load/dispatch did not materialize this module.")

    def wrapped_forward(*args, **kwargs):
        # Ensure inputs are on CUDA (usually already true)
        args   = [a.to(cuda_dev, non_blocking=True) if torch.is_tensor(a) else a for a in args]
        kwargs = {k: (v.to(cuda_dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in kwargs.items()}

        # Structure-only GPU clone
        expert_gpu = expert.to_empty(device=cuda_dev)
        # Load weights/buffers by name (safe; no meta movement)
        expert_gpu.load_state_dict(expert.state_dict(), strict=True)

        # Make sure dtype is correct on the clone
        for p in expert_gpu.parameters(recurse=True):
            p.data = p.data.to(DTYPE)
        for b in expert_gpu.buffers(recurse=True):
            if b is not None:
                b.data = b.data.to(DTYPE)

        with torch.no_grad():
            out = original_forward.__get__(expert_gpu, type(expert_gpu))(*args, **kwargs)

        # Free the temporary clone
        del expert_gpu
        torch.cuda.empty_cache()
        return out

    expert.forward = wrapped_forward

# Find and patch ModuleLists named "...experts"
patched = 0
for name, module in model.named_modules():
    if "experts" in name and isinstance(module, nn.ModuleList):
        for idx, expert in enumerate(module):
            patch_expert_forward_no_cache(expert, gpu_device=GPU)
            patched += 1
print(f"Patched experts (no GPU cache): {patched}")
assert patched > 0, "No experts found; check your model module names."

# 5) Tokenizer + quick sanity + generate
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Sanity: some params on CPU (experts), some on CUDA (trunk/router)
cpu_params  = sum(p.numel() for p in model.parameters() if p.device.type == "cpu")
cuda_params = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
print(f"Param split — CPU: {cpu_params/1e9:.2f}B | CUDA: {cuda_params/1e9:.2f}B")

with torch.no_grad():
    t = tokenizer("sanity.", return_tensors="pt").to(GPU)
    logits = model(**t).logits
    print("logits mean/std:", float(logits.float().mean()), float(logits.float().std()),
          "NaN?", bool(torch.isnan(logits).any()))

prompt = "DeepSeek is transforming inference efficiency."
inputs = tokenizer(prompt, return_tensors="pt").to(GPU)
gen_kwargs = dict(
    max_new_tokens=64,
    do_sample=True, top_p=0.95, top_k=50, temperature=0.9,
    repetition_penalty=1.05, min_new_tokens=16,
    eos_token_id=tokenizer.eos_token_id,
)
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
