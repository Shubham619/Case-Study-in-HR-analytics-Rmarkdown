#!/usr/bin/env python3
"""
rlmem_final.py

Reinforcement‑learning DRAM pattern discovery.

This module implements a simple DRAM fault simulator along with a
Thompson‑sampling bandit agent.  The goal in **phase 1** is to
discover patterns that reliably induce correctable errors (CE) on
devices with known CE defects; in **phase 2** the goal shifts to
discovering entirely new fault signatures (both CE and UE).  In
addition to the classic GSAT patterns, the simulator also supports
procedurally generated patterns in phase 2.

This file contains no external dependencies beyond the Python
standard library.  To train or evaluate the agent run this file as
a script.  Example usage:

    # Phase‑1: train and evaluate (GSAT only)
    python rlmem_final.py train --phase 1 --episodes 300 --steps 8 --region-kb 64 --out pol_p1.json
    python rlmem_final.py eval  --phase 1 --episodes 80  --steps 8 --region-kb 64 --policy pol_p1.json

    # Phase‑2: train and evaluate (GSAT + generator actions)
    python rlmem_final.py train --phase 2 --episodes 500 --steps 10 --region-kb 64 --out pol_p2.json
    python rlmem_final.py eval  --phase 2 --episodes 120 --steps 10 --region-kb 64 --policy pol_p2.json

    # Compare strategies
    python rlmem_final.py compare --phase 1 --episodes 80 --steps 8  --region-kb 64 --policy pol_p1.json
    python rlmem_final.py compare --phase 2 --episodes 80 --steps 10 --region-kb 64 --policy pol_p2.json

The implementation is deliberately self contained; all helper
functions and classes are defined in this file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Iterable
from array import array

# Optional plotting deps. Training/eval works without them.
try:
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    np = None
    plt = None


###############################################################################
# 0) Helpers
###############################################################################

def u32(x: int) -> int:
    """Return x as a 32‑bit unsigned value."""
    return x & 0xFFFFFFFF


def popcount32(x: int) -> int:
    """Return the number of set bits in a 32‑bit integer."""
    return (x & 0xFFFFFFFF).bit_count()


def rotl32(x: int, r: int) -> int:
    """Rotate a 32‑bit integer left by r bits."""
    r &= 31
    return u32((x << r) | (x >> (32 - r)))


def bitswap32(x: int, mode: int) -> int:
    """
    Deterministic "bit permutation" family (cheap, stdlib‑only).

    mode 0: identity
    mode 1: reverse bits
    mode 2: swap nibbles
    mode 3: rotate 13
    mode 4: xor‑fold
    """
    x = u32(x)
    if mode == 0:
        return x
    if mode == 1:
        # reverse bits
        y = 0
        for i in range(32):
            y = (y << 1) | ((x >> i) & 1)
        return u32(y)
    if mode == 2:
        # swap nibbles
        y = 0
        for i in range(8):
            nib = (x >> (i * 4)) & 0xF
            y |= nib << ((7 - i) * 4)
        return u32(y)
    if mode == 3:
        return rotl32(x, 13)
    # mode 4
    y = x ^ (x >> 16)
    y = u32(y ^ (y >> 8))
    return y


###############################################################################
# 1) Legacy GSAT patterns (from pattern.cc subset)
###############################################################################

# A handful of classic pattern sequences used by GSAT testers.  These
# sequences are expressed as arrays of 32‑bit values.  When combined
# with different bus widths and optional inversion this yields a
# comprehensive set of legacy patterns.
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
    0x00000001, 0xFFFFFFFE, 0x00000002, 0xFFFFFFFD,
    0x00000004, 0xFFFFFFFB, 0x00000008, 0xFFFFFFF7,
    0x00000010, 0xFFFFFFEF, 0x00000020, 0xFFFFFFDF,
    0x00000040, 0xFFFFFFBF, 0x00000080, 0xFFFFFF7F,
    0x00000100, 0xFFFFFEFF, 0x00000200, 0xFFFFFDFF,
    0x00000400, 0xFFFFFBFF, 0x00000800, 0xFFFFF7FF,
    0x00001000, 0xFFFFEFFF, 0x00002000, 0xFFFFDFFF,
    0x00004000, 0xFFFFBFFF, 0x00008000, 0xFFFF7FFF,
    0x00010000, 0xFFFEFFFF, 0x00020000, 0xFFFDFFFF,
    0x00040000, 0xFFFBFFFF, 0x00080000, 0xFFF7FFFF,
    0x00100000, 0xFFEFFFFF, 0x00200000, 0xFFDFFFFF,
    0x00400000, 0xFFBFFFFF, 0x00800000, 0xFF7FFFFF,
    0x01000000, 0xFEFFFFFF, 0x02000000, 0xFDFFFFFF,
    0x04000000, 0xFBFFFFFF, 0x08000000, 0xF7FFFFFF,
    0x10000000, 0xEFFFFFFF, 0x20000000, 0xDFFFFFFF,
    0x40000000, 0xBFFFFFFF, 0x80000000, 0x7FFFFFFF,
    0x40000000, 0xBFFFFFFF, 0x20000000, 0xDFFFFFFF,
    0x10000000, 0xEFFFFFFF, 0x08000000, 0xF7FFFFFF,
    0x04000000, 0xFBFFFFFF, 0x02000000, 0xFDFFFFFF,
    0x01000000, 0xFEFFFFFF, 0x00800000, 0xFF7FFFFF,
    0x00400000, 0xFFBFFFFF, 0x00200000, 0xFFDFFFFF,
    0x00100000, 0xFFEFFFFF, 0x00080000, 0xFFF7FFFF,
    0x00040000, 0xFFFBFFFF, 0x00020000, 0xFFFDFFFF,
    0x00010000, 0xFFFEFFFF, 0x00008000, 0xFFFF7FFF,
    0x00004000, 0xFFFFBFFF, 0x00002000, 0xFFFFDFFF,
    0x00001000, 0xFFFFEFFF, 0x00000800, 0xFFFFF7FF,
    0x00000400, 0xFFFFFBFF, 0x00000200, 0xFFFFFDFF,
    0x00000100, 0xFFFFFEFF, 0x00000080, 0xFFFFFF7F,
    0x00000040, 0xFFFFFFBF, 0x00000020, 0xFFFFFFDF,
    0x00000010, 0xFFFFFFEF, 0x00000008, 0xFFFFFFF7,
    0x00000004, 0xFFFFFFFB, 0x00000002, 0xFFFFFFFD,
    0x00000001, 0xFFFFFFFE, 0x00000000, 0xFFFFFFFF,
]
walkingZeros_data = [
    0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFB, 0xFFFFFFF7,
    0xFFFFFFEF, 0xFFFFFFDF, 0xFFFFFFBF, 0xFFFFFF7F,
    0xFFFFFEFF, 0xFFFFFDFF, 0xFFFFFBFF, 0xFFFFF7FF,
    0xFFFFEFFF, 0xFFFFDFFF, 0xFFFFBFFF, 0xFFFF7FFF,
    0xFFFEFFFF, 0xFFFDFFFF, 0xFFFBFFFF, 0xFFF7FFFF,
    0xFFEFFFFF, 0xFFDFFFFF, 0xFFBFFFFF, 0xFF7FFFFF,
    0xFEFFFFFF, 0xFDFFFFFF, 0xFBFFFFFF, 0xF7FFFFFF,
    0xEFFFFFFF, 0xDFFFFFFF, 0xBFFFFFFF, 0x7FFFFFFF,
    0xBFFFFFFF, 0xDFFFFFFF, 0xEFFFFFFF, 0xF7FFFFFF,
    0xFBFFFFFF, 0xFDFFFFFF, 0xFEFFFFFF, 0xFF7FFFFF,
    0xFFBFFFFF, 0xFFDFFFFF, 0xFFEFFFFF, 0xFFF7FFFF,
    0xFFFBFFFF, 0xFFFDFFFF, 0xFFFEFFFF, 0xFFFF7FFF,
    0xFFFFBFFF, 0xFFFFDFFF, 0xFFFFEFFF, 0xFFFFF7FF,
    0xFFFFFBFF, 0xFFFFFDFF, 0xFFFFFEFF, 0xFFFFFF7F,
    0xFFFFFFBF, 0xFFFFFFDF, 0xFFFFFFEF, 0xFFFFFFF7,
    0xFFFFFFFB, 0xFFFFFFFD, 0xFFFFFFFE, 0xFFFFFFFF,
]

LEGACY_GSAT: Dict[str, List[int]] = {
    "walkingOnes": walkingOnes_data,
    "walkingInvOnes": walkingInvOnes_data,
    "walkingZeros": walkingZeros_data,
    "OneZero": [0x00000000, 0xFFFFFFFF],
    "JustZero": [0x00000000, 0x00000000],
    "JustOne": [0xFFFFFFFF, 0xFFFFFFFF],
    "JustFive": [0x55555555, 0x55555555],
    "JustA": [0xAAAAAAAA, 0xAAAAAAAA],
    "FiveA": [0x55555555, 0xAAAAAAAA],
    "FiveA8": [0x5AA5A55A, 0xA55A5AA5, 0xA55A5AA5, 0x5AA5A55A],
    "Long8b10b": [0x16161616, 0x16161616],
    "Short8b10b": [0xB5B5B5B5, 0xB5B5B5B5],
    "Checker8b10b": [0xB5B5B5B5, 0x4A4A4A4A],
    "Five7": [0x55555557, 0x55575555],
    "Zero2fd": [0x00020002, 0xFFFDFFFD],
}

# When combining patterns with bus widths we need to know the shift
# required to advance the pattern index at different word sizes.
BUSSHIFT = {32: 0, 64: 1, 128: 2, 256: 3}
WIDTHS = [32, 64, 128, 256]


def build_legacy_actions() -> List[Dict]:
    """Build a list of all legacy GSAT pattern actions."""
    actions: List[Dict] = []
    for name, vals in LEGACY_GSAT.items():
        for bw in WIDTHS:
            for inv in (False, True):
                actions.append({
                    "kind": "gsat",
                    "pattern": name,
                    "vals": vals,
                    "buswidth": bw,
                    "busshift": BUSSHIFT[bw],
                    "invert": inv,
                    "variant_name": f"GSAT:{name}{'~' if inv else ''}_{bw}",
                })
    return actions


def lfsr32_step(x: int) -> int:
    """Simple xorshift32 PRNG used for procedural pattern generation."""
    x = u32(x)
    x ^= u32(x << 13)
    x ^= u32(x >> 17)
    x ^= u32(x << 5)
    return u32(x)


def build_generator_actions() -> List[Dict]:
    """
    Phase‑2: actions that allow the agent to invent patterns.

    Each action defines a pattern generator:
      - seed: starting state of the LFSR
      - bitswap mode: one of several simple permutations
      - invert: whether to invert the generated stream
      - busshift (like GSAT width effect)
      - address stride / hotset to create locality/hammer pressure
    """
    actions: List[Dict] = []
    seeds = [0x12345678, 0x87654321, 0xA5A5A5A5, 0x1, 0xDEADBEEF]
    swap_modes = [0, 1, 2, 3, 4]
    strides = [1, 2, 4, 8, 16]      # in 32‑bit words
    hotsets = [0, 64, 256, 1024]    # if >0: repeatedly hit within first hotset words

    for seed in seeds:
        for sm in swap_modes:
            for inv in (False, True):
                for bw in (32, 64, 128, 256):
                    for stride in strides:
                        for hot in hotsets:
                            actions.append({
                                "kind": "gen",
                                "seed": seed,
                                "swap_mode": sm,
                                "invert": inv,
                                "busshift": BUSSHIFT[bw],
                                "buswidth": bw,
                                "stride": stride,
                                "hotset_words": hot,
                                "variant_name": f"GEN:seed={seed:08x},swap={sm},inv={int(inv)},bw={bw},str={stride},hot={hot}",
                            })
    return actions


###############################################################################
# 2) Fault model (two regimes: Phase‑1 CE devices, Phase‑2 discovery)
###############################################################################

@dataclass
class FaultConfig:
    """
    Configuration parameters for the fault model.  These values
    determine the density and behaviour of various defect types.
    """
    seed: int = 0
    # Phase‑1: "known CE devices": stronger single‑bit stuck/weak bits
    ce_site_prob: float = 5e-6         # probability of a hard CE site
    ce_flip_prob: float = 0.0          # optional transient CE on read

    # Phase‑2: additional defect types
    retention_prob: float = 2e-6
    retention_flip_after: int = 40_000

    hammer_threshold: int = 6_000
    hammer_flip_prob: float = 2e-3

    intermittent_prob: float = 1e-8

    # multi‑bit burst/UE‑like (mostly Phase‑2)
    burst_prob: float = 2e-7
    burst_width_min: int = 2
    burst_width_max: int = 4

    cacheline_bytes: int = 64

    # --- Phase‑1: pattern‑dependent CE mechanisms ---
    # 1) Transition CE: only triggers when a certain transition happens on that cell
    tf_prob: float = 6e-6               # density of transition‑fault sites
    tf_mode: int = 0                    # 0=random, 1=fail 0->1, 2=fail 1->0
    tf_flip_prob: float = 1.0           # probability fault triggers when the transition occurs

    # 2) Data‑dependent CE: triggers only if written word has certain property
    dd_prob: float = 6e-6               # density of data‑dependent weak sites
    dd_min_popcount: int = 24           # trigger if popcount(write_data) >= this
    dd_flip_prob: float = 1.0

    # 3) Simple coupling CE (optional): aggressor write flips victim bit
    cf_prob: float = 2e-6               # density of coupling pairs
    cf_flip_prob: float = 0.15          # chance to flip victim on aggressor write
    cf_distance_words: int = 64         # victim is idx + distance (wrap)


class FaultModel:
    """
    DRAM fault simulator.  Given a configuration and memory size this
    class injects hard faults (stuck bits and bursts), retention
    problems, hammer effects, intermittent flips, and pattern
    dependent CE faults.  Reads return the value stored in memory
    corrupted by any active fault mechanisms.
    """

    def __init__(self, cfg: FaultConfig, mem_words: int, phase: int):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.mem_words = mem_words
        self.phase = phase

        self.time = 0
        self.last_fault_type: str = "none"

        # Hard faults: idx -> (mask, forced_bits)
        self.hard_mask: Dict[int, int] = {}
        self.hard_forced: Dict[int, int] = {}

        # Retention: idx -> (bit, flip_time)
        self.retention: Dict[int, Tuple[int, int]] = {}

        # Hammer counters: cacheline -> count
        self.cl_access: Dict[int, int] = {}

        # Pattern‑dependent CE tables
        # Transition faults: idx -> (bit, fail_dir) where fail_dir: 1 means fail 0->1, 0 means fail 1->0
        self.tf: Dict[int, Tuple[int, int]] = {}
        # Data‑dependent weak: idx -> bit
        self.dd: Dict[int, int] = {}
        # Coupling faults: aggressor_idx -> (victim_idx, bit)
        self.cf: Dict[int, Tuple[int, int]] = {}

        # If a write‑time fault corrupts the stored value, we remember its type
        # so reads that observe the mismatch can attribute it.
        self.stored_fault_type: Dict[int, str] = {}

        # ------------------------------------------------------------------
        # Fast fault injection: sample a small number of sites instead of
        # iterating over the entire address space (critical for scalability).
        # ------------------------------------------------------------------

        def sample_count(prob: float) -> int:
            if prob <= 0.0:
                return 0
            exp = prob * mem_words
            base = int(exp)
            frac = exp - base
            return base + (1 if self.rng.random() < frac else 0)

        def sample_indices(count: int) -> List[int]:
            if count <= 0:
                return []
            if count >= mem_words:
                return list(range(mem_words))
            # For our typical tiny probabilities count is very small.
            # Use a set to avoid duplicates.
            s: set = set()
            # Safety cap to avoid pathological infinite loops.
            # With small counts this will never hit.
            while len(s) < count:
                s.add(self.rng.randrange(mem_words))
            return list(s)

        # Hard stuck CE sites (both phases)
        for idx in sample_indices(sample_count(cfg.ce_site_prob)):
            bit = self.rng.randrange(32)
            self._add_hard(idx, bit, is_burst=False)

        # Phase‑2 extras
        if phase >= 2:
            for idx in sample_indices(sample_count(cfg.burst_prob)):
                bit_start = self.rng.randrange(28)
                width = self.rng.randint(cfg.burst_width_min, cfg.burst_width_max)
                for b in range(bit_start, bit_start + width):
                    self._add_hard(idx, b, is_burst=True)

            for idx in sample_indices(sample_count(cfg.retention_prob)):
                bit = self.rng.randrange(32)
                flip_t = max(1, cfg.retention_flip_after + self.rng.randint(-5000, 5000))
                self.retention[idx] = (bit, flip_t)

        # Pattern‑dependent CE sites (both phases)
        for idx in sample_indices(sample_count(cfg.tf_prob)):
            bit = self.rng.randrange(32)
            if cfg.tf_mode == 1:
                fail_dir = 1  # fail 0->1
            elif cfg.tf_mode == 2:
                fail_dir = 0  # fail 1->0
            else:
                fail_dir = self.rng.getrandbits(1)
            self.tf[idx] = (bit, fail_dir)

        for idx in sample_indices(sample_count(cfg.dd_prob)):
            self.dd[idx] = self.rng.randrange(32)

        for idx in sample_indices(sample_count(cfg.cf_prob)):
            bit = self.rng.randrange(32)
            victim = (idx + cfg.cf_distance_words) % mem_words
            self.cf[idx] = (victim, bit)

    def _add_hard(self, idx: int, bit: int, is_burst: bool) -> None:
        """Helper to register a hard fault bit at the given word index."""
        mask = self.hard_mask.get(idx, 0) | (1 << bit)
        self.hard_mask[idx] = mask
        forced = self.hard_forced.get(idx, 0)
        # randomly force bit to 0 or 1
        if self.rng.getrandbits(1):
            forced |= (1 << bit)
        else:
            forced &= ~(1 << bit)
        self.hard_forced[idx] = forced

    def on_time(self, t: int) -> None:
        """Record the global time; used for retention faults."""
        self.time = t

    def on_access(self, addr: int) -> None:
        """Record an access to the cacheline for rowhammer tracking."""
        cl = addr // self.cfg.cacheline_bytes
        self.cl_access[cl] = self.cl_access.get(cl, 0) + 1

    def apply_on_write(self, addr: int, idx: int, old_val: int, new_val: int) -> Tuple[int, List[Tuple[int, int]]]:
        """Apply write‑time fault mechanisms.

        Returns:
          (stored_value, victim_updates)
        where victim_updates is a list of (victim_idx, xor_mask) that the
        backend should apply to memory to emulate coupling faults.

        IMPORTANT: Transition/data‑dependent/coupling faults must corrupt the
        stored value (or a victim's stored value) at write time; otherwise
        verification after the write pass will not observe a mismatch.
        """
        # Track cacheline pressure for rowhammer accounting
        self.on_access(addr)

        v = u32(new_val)
        old = u32(old_val)
        ftype: Optional[str] = None
        victim_updates: List[Tuple[int, int]] = []

        # Clear any previous stored fault annotation for this location,
        # because we are overwriting its contents.
        self.stored_fault_type.pop(idx, None)

        # Transition fault: corrupt the written value if a failing transition occurs.
        if idx in self.tf:
            bit, fail_dir = self.tf[idx]
            prev_b = (old >> bit) & 1
            now_b = (v >> bit) & 1
            if prev_b == 0 and now_b == 1 and fail_dir == 1:
                if self.rng.random() < self.cfg.tf_flip_prob:
                    v ^= (1 << bit)
                    ftype = "tf_ce"
            elif prev_b == 1 and now_b == 0 and fail_dir == 0:
                if self.rng.random() < self.cfg.tf_flip_prob:
                    v ^= (1 << bit)
                    ftype = "tf_ce"

        # Data‑dependent: corrupt the written value if the data matches a weak condition.
        if idx in self.dd:
            bit = self.dd[idx]
            if popcount32(v) >= self.cfg.dd_min_popcount:
                if self.rng.random() < self.cfg.dd_flip_prob:
                    v ^= (1 << bit)
                    if ftype is None:
                        ftype = "dd_ce"

        # Coupling fault: aggressor write flips a victim's stored bit.
        if idx in self.cf and self.rng.random() < self.cfg.cf_flip_prob:
            victim_idx, bit = self.cf[idx]
            victim_updates.append((victim_idx, 1 << bit))
            # Mark victim as write‑corrupted for later attribution.
            self.stored_fault_type[victim_idx] = "cf_ce"

        if ftype is not None:
            self.stored_fault_type[idx] = ftype

        return u32(v), victim_updates

    def apply_on_read(self, addr: int, idx: int, val: int) -> int:
        """
        Apply all fault mechanisms on a memory read.  The returned value
        may differ from the stored value due to hard faults, pattern
        dependent faults, row hammer, retention, and intermittent noise.
        The last_fault_type attribute is set to a string describing
        which fault mechanism caused the flip (if any).
        """
        # Default attribution from the most recent write‑time corruption.
        self.last_fault_type = self.stored_fault_type.get(idx, "none")
        v = u32(val)
        orig = v

        # Phase‑2: row hammer
        if self.phase >= 2:
            cl = addr // self.cfg.cacheline_bytes
            # If the hammer threshold is exceeded, randomly flip a bit
            if self.cl_access.get(cl, 0) > self.cfg.hammer_threshold:
                if self.rng.random() < self.cfg.hammer_flip_prob:
                    bit = self.rng.randrange(32)
                    v ^= (1 << bit)
                    self.last_fault_type = "hammer"
            # Retention: a bit flips after it has been stored for a long time
            if idx in self.retention:
                bit, flip_t = self.retention[idx]
                if self.time >= flip_t:
                    v ^= (1 << bit)
                    if self.last_fault_type == "none":
                        self.last_fault_type = "retention"
            # Intermittent: extremely rare sporadic flips
            if self.rng.random() < self.cfg.intermittent_prob:
                bit = self.rng.randrange(32)
                v ^= (1 << bit)
                if self.last_fault_type == "none":
                    self.last_fault_type = "intermittent"

        # Optional transient CE flip (available in both phases)
        if self.cfg.ce_flip_prob > 0 and self.rng.random() < self.cfg.ce_flip_prob:
            bit = self.rng.randrange(32)
            v ^= (1 << bit)
            if self.last_fault_type == "none":
                self.last_fault_type = "ce_transient"

        # NOTE: Transition/data‑dependent/coupling write faults are applied at write time
        # in apply_on_write(). We do not apply them again here.

        # Apply hard stuck/burst faults (these override any flips because the bits are physically stuck)
        if idx in self.hard_mask:
            mask = self.hard_mask[idx]
            forced = self.hard_forced[idx]
            pre = v
            v = (v & ~mask) | (forced & mask)
            if v != pre:
                # Hard faults dominate attribution.
                if popcount32(mask) > 1:
                    self.last_fault_type = "burst_ue"
                else:
                    self.last_fault_type = "stuck_ce"

        return u32(v)


###############################################################################
# 3) Backend + runner
###############################################################################

@dataclass
class ErrorEvent:
    """Represents a single data mismatch observed during verification."""
    addr: int
    expected: int
    observed: int
    bitmask: int
    fault_type: str
    is_ue: bool


class DRAMBackend:
    """
    Simple 32‑bit word addressable memory with a fault model.  Memory
    contents are stored in an array of unsigned ints.  Reads and
    writes interact with the fault model which applies timing and
    access effects.
    """

    def __init__(self, mem_bytes: int, fault: FaultModel):
        assert mem_bytes % 4 == 0
        self.mem_words = mem_bytes // 4
        self.mem = array('I', [0] * self.mem_words)
        self.fault = fault
        self.time = 0

    def write32(self, addr: int, val: int) -> None:
        idx = (addr >> 2) % self.mem_words
        old = int(self.mem[idx])
        new = u32(val)
        self.time += 1
        # Let the fault model know the time has advanced
        self.fault.on_time(self.time)
        # Apply write-time faults (transition/data-dependent/coupling)
        new2, victim_xors = self.fault.apply_on_write(addr, idx, old, new)
        self.mem[idx] = u32(new2)
        # Coupling can corrupt other locations.
        for v_idx, mask in victim_xors:
            self.mem[v_idx % self.mem_words] = u32(int(self.mem[v_idx % self.mem_words]) ^ int(mask))

    def read32(self, addr: int) -> int:
        idx = (addr >> 2) % self.mem_words
        self.time += 1
        # Advance time and record access for hammer
        self.fault.on_time(self.time)
        self.fault.on_access(addr)
        raw = int(self.mem[idx])
        # Apply faults on read
        return int(self.fault.apply_on_read(addr, idx, raw))


def sat_or_gen_fill_verify(
    backend: DRAMBackend,
    base_addr: int,
    size_bytes: int,
    action: Dict,
) -> List[ErrorEvent]:
    """
    Execute one pattern action.  Always performs a full write pass
    followed by a full verify pass over the specified region.  Returns
    a list of ErrorEvent objects describing any mismatches encountered.
    """
    words = size_bytes // 4
    errors: List[ErrorEvent] = []

    kind = action["kind"]
    busshift = action["busshift"]
    invert = action.get("invert", False)

    # Address schedule parameters
    stride = 1
    hotset = 0
    if kind == "gen":
        stride = int(action["stride"])
        hotset = int(action["hotset_words"])

    # Clamp hotset to region size to avoid out-of-range patterns when we
    # model only the test region.
    if hotset > 0:
        hotset = max(1, min(hotset, words))

    mod_words = hotset if hotset > 0 else words
    idxs: List[int] = [0] * words
    addrs: List[int] = [0] * words
    for w in range(words):
        idx = (w * stride) % mod_words
        idxs[w] = idx
        addrs[w] = base_addr + idx * 4

    # Precompute the value stream once; generator actions are otherwise
    # O(words^2) due to repeated stepping.
    exp_stream: array = array('I', [0] * words)
    final_per_idx: Optional[array] = array('I', [0] * mod_words) if hotset > 0 else None

    if kind == "gsat":
        vals = action["vals"]
        vlen = len(vals)
        for w in range(words):
            g = w >> busshift
            v = vals[g % vlen]
            v = u32(~v) if invert else u32(v)
            exp_stream[w] = v
            if final_per_idx is not None:
                final_per_idx[idxs[w]] = v
    else:
        seed = int(action["seed"])
        swap_mode = int(action["swap_mode"])
        x = u32(seed)
        g_prev = -1
        for w in range(words):
            g = w >> busshift
            # Advance the LFSR once per group (w>>busshift), matching the
            # original semantics: steps=(w>>busshift)+1.
            while g_prev < g:
                x = lfsr32_step(x)
                g_prev += 1
            v = bitswap32(x, swap_mode)
            v = u32(~v) if invert else u32(v)
            exp_stream[w] = v
            if final_per_idx is not None:
                final_per_idx[idxs[w]] = v

    # WRITE pass
    for w in range(words):
        backend.write32(addrs[w], int(exp_stream[w]))

    # VERIFY pass
    for w in range(words):
        addr = addrs[w]
        idx = idxs[w]
        exp = int(final_per_idx[idx]) if final_per_idx is not None else int(exp_stream[w])
        got = backend.read32(addr)
        if got != exp:
            diff = u32(exp ^ got)
            is_ue = popcount32(diff) > 1
            errors.append(ErrorEvent(
                addr=addr,
                expected=exp,
                observed=got,
                bitmask=diff,
                fault_type=backend.fault.last_fault_type,
                is_ue=is_ue,
            ))

    return errors


###############################################################################
# 4) Environment + signatures
###############################################################################

@dataclass
class StepResult:
    """
    Summary of one environment step.  Returned by SatEnv.step() and
    consumed by the reward functions.
    """
    dt: int
    errors: int
    new_sigs: int
    new_ce_sigs: int
    new_ue_sigs: int
    ce_count: int
    ue_count: int
    first_error_time: Optional[int]


def signature_phase1(e: ErrorEvent) -> Tuple:
    """
    Phase‑1 signatures focus on CE induction on known‑CE devices.  The
    signature buckets errors by their location and bit behaviour
    (popcount + lower eight bits of the bitmask) and whether the
    observed fault type was correctable.
    """
    page = e.addr // 4096
    bm = (popcount32(e.bitmask) << 8) | (e.bitmask & 0xFF)
    return ("CE" if not e.is_ue else "UE", e.fault_type, page, bm)


def signature_phase2(e: ErrorEvent) -> Tuple:
    """
    Phase‑2 signatures retain more detail to emphasise discovery of
    genuinely new defects.  The full bitmask is used to enforce
    novelty.
    """
    page = e.addr // 4096
    return ("UE" if e.is_ue else "CE", e.fault_type, page, e.bitmask)


class SatEnv:
    """
    Environment wrapper around the DRAMBackend and FaultModel.  It
    maintains a set of discovered signatures and tracks the time to
    first failure.  Each action corresponds to one pattern variant.
    """

    def __init__(self, mem_mb: int, region_kb: int, phase: int, seed: int):
        # We only model the tested region. This keeps the simulator fast and
        # makes results easier to interpret.
        self.total_mem_bytes = mem_mb * 1024 * 1024
        self.region_bytes = min(region_kb * 1024, self.total_mem_bytes)
        self.mem_bytes = self.region_bytes
        self.phase = phase
        self.seed = seed

        # Build the list of possible actions
        legacy = build_legacy_actions()
        self.actions: List[Dict] = legacy
        self.num_legacy_actions = len(legacy)
        if phase >= 2:
            self.actions += build_generator_actions()

        self.backend: Optional[DRAMBackend] = None
        self.seen_sigs: set = set()
        self.first_error_time: Optional[int] = None

    def num_actions(self) -> int:
        return len(self.actions)

    def reset(self, fault_seed: int, cfg_override: Optional[dict] = None) -> None:
        """
        Reset the environment with a new fault seed.  Optionally
        override some fault parameters via cfg_override.
        """
        cfg = FaultConfig(seed=fault_seed)
        if cfg_override:
            for k, v in cfg_override.items():
                setattr(cfg, k, v)
        fault = FaultModel(cfg, mem_words=self.mem_bytes // 4, phase=self.phase)
        self.backend = DRAMBackend(self.mem_bytes, fault)
        self.seen_sigs = set()
        self.first_error_time = None

    def step(self, action_id: int) -> StepResult:
        """
        Execute one action (pattern fill+verify) and return summary
        statistics.  Keeps track of how many new signatures were
        discovered and updates time to first failure.
        """
        assert self.backend is not None
        t0 = self.backend.time
        action = self.actions[action_id]

        errs = sat_or_gen_fill_verify(self.backend, 0, self.region_bytes, action)

        new_sigs = 0
        new_ce_sigs = 0
        new_ue_sigs = 0
        ce = 0
        ue = 0

        for e in errs:
            if e.is_ue:
                ue += 1
            else:
                ce += 1

            sig = signature_phase1(e) if self.phase == 1 else signature_phase2(e)
            if sig not in self.seen_sigs:
                self.seen_sigs.add(sig)
                new_sigs += 1
                if e.is_ue:
                    new_ue_sigs += 1
                else:
                    new_ce_sigs += 1

        if errs and self.first_error_time is None:
            self.first_error_time = self.backend.time

        return StepResult(
            dt=self.backend.time - t0,
            errors=len(errs),
            new_sigs=new_sigs,
            new_ce_sigs=new_ce_sigs,
            new_ue_sigs=new_ue_sigs,
            ce_count=ce,
            ue_count=ue,
            first_error_time=self.first_error_time,
        )


###############################################################################
# 5) Rewards (THIS is the key to your objective)
###############################################################################

def reward_phase1(res: StepResult, first_before: bool) -> float:
    """
    Phase‑1 reward encourages the agent to induce CE quickly and
    consistently on known‑CE devices.

      * Correctable errors (CE) are valuable; repeated CE counts for
        something but less than new signatures.
      * Discovering additional CE signatures is extra valuable (better
        coverage among CE sites).
      * Uncorrectable errors (UE) are not the target in phase 1 and
        receive a penalty.
      * A time‑to‑first‑failure (TTFF) bonus rewards finding the first
        error quickly.
      * A small time penalty discourages slow patterns.
    """
    r = 0.0
    # CE induction (repeatable)
    if res.ce_count > 0:
        r += 5.0 + 2.0 * res.ce_count
    # Extra credit: discovering additional CE signatures (do NOT reward UE novelty in phase 1)
    r += 12.0 * res.new_ce_sigs
    # TTFF bonus: only if an error occurred and this was the first in episode
    if res.errors > 0 and first_before:
        r += 25.0
    # Penalise UE (undesired in phase 1)
    if res.ue_count > 0:
        r -= 10.0 * res.ue_count
    # Time penalty
    r -= 1e-5 * res.dt
    return r


def reward_phase2(res: StepResult, first_before: bool) -> float:
    """
    Phase‑2 reward emphasises discovery of new defect signatures.

      * Novelty (new signatures) dominates the reward.
      * Uncorrectable errors often indicate stronger or more interesting
        issues and receive a sizeable bonus.
      * Repeating the same CE many times yields a small reward.
      * A TTFF bonus still helps to prioritise faster patterns.
      * A small time penalty discourages slow patterns.
    """
    r = 0.0
    # Novelty dominates
    r += 30.0 * res.new_sigs
    # Extra emphasis on novel UE signatures
    r += 10.0 * res.new_ue_sigs
    # UE often indicates stronger/interesting issues; weight it higher
    if res.ue_count > 0:
        r += 50.0 + 5.0 * res.ue_count
    # CE still counts, but less than novelty
    if res.ce_count > 0:
        r += 1.0 * res.ce_count
    # TTFF bonus
    if res.errors > 0 and first_before:
        r += 10.0
    # Time penalty
    r -= 1e-5 * res.dt
    return r


###############################################################################
# 6) Agent: Thompson‑sampling bandit
###############################################################################

def train_bandit(
    phase: int,
    episodes: int,
    steps: int,
    mem_mb: int,
    region_kb: int,
    seed: int,
    out_path: str,
    cfg_override: dict,
) -> None:
    """
    Train a Thompson‑sampling bandit to select the best pattern
    variants.  Maintains a Beta distribution over each action's
    success probability and samples to decide which action to play.
    """
    rng = random.Random(seed)
    env = SatEnv(mem_mb=mem_mb, region_kb=region_kb, phase=phase, seed=seed)

    # Initialise Beta priors for each action
    a: List[float] = [1.0] * env.num_actions()
    b: List[float] = [1.0] * env.num_actions()

    rew_fn = reward_phase1 if phase == 1 else reward_phase2

    print(f"[train] phase={phase} actions={env.num_actions()} episodes={episodes} steps={steps}")

    def success_weight(res: StepResult) -> Tuple[bool, float]:
        """Map a step result to (success, update_weight) for TS updates."""
        if phase == 1:
            # Replication: hit CE sites reliably and broaden CE signature coverage.
            success = (res.ce_count > 0)
            w = 1.0 + 0.35 * res.new_ce_sigs + 0.05 * min(res.ce_count, 40)
            # Penalize UE discovery during replication.
            if res.ue_count > 0:
                w *= 0.5
            return success, w
        # Discovery: prioritize new signatures (esp. new UE signatures).
        success = (res.new_sigs > 0) or (res.ue_count > 0)
        w = 1.0 + 0.40 * res.new_sigs + 0.60 * res.new_ue_sigs
        return success, w

    for ep in range(episodes):
        env.reset(fault_seed=ep, cfg_override=cfg_override)
        for _ in range(steps):
            # Sample theta for each action, pick the best
            best = 0
            best_theta = -1.0
            for i in range(env.num_actions()):
                theta = rng.betavariate(a[i], b[i])
                if theta > best_theta:
                    best_theta = theta
                    best = i
            first_before = (env.first_error_time is None)
            res = env.step(best)
            _ = rew_fn(res, first_before)  # keep for logging / future extensions

            ok, w = success_weight(res)
            if ok:
                a[best] += w
            else:
                b[best] += w

    means = [a[i] / (a[i] + b[i]) for i in range(env.num_actions())]
    ranked = sorted(range(env.num_actions()), key=lambda i: means[i], reverse=True)

    policy = {
        "phase": phase,
        "type": "topk_cycle",
        "topk": 20,
        "actions": ranked[:20],
        "posterior_mean_top50": {str(i): means[i] for i in ranked[:50]},
        "mem_mb": mem_mb,
        "region_kb": region_kb,
        "episodes": episodes,
        "steps": steps,
        "cfg_override": cfg_override,
    }

    with open(out_path, "w") as f:
        json.dump(policy, f, indent=2)

    print(f"[train] wrote policy: {out_path}")
    print("[train] top actions:")
    for i in ranked[:10]:
        print(f"  aid={i:4d} mean={means[i]:.3f}  {env.actions[i]['variant_name']}")


###############################################################################
# 7) Eval + Compare
###############################################################################

def run_eval(
    phase: int,
    episodes: int,
    steps: int,
    mem_mb: int,
    region_kb: int,
    strategy: Callable[[int, int], int],
    cfg_override: dict,
    restrict_legacy: bool = False,
) -> dict:
    """
    Evaluate a strategy (mapping from step to action id) over a
    number of episodes and return aggregate statistics.
    """
    env = SatEnv(mem_mb=mem_mb, region_kb=region_kb, phase=phase, seed=0)

    tot_ce = 0
    tot_ue = 0
    failures = 0
    ttff_list: List[int] = []
    global_cov: set = set()

    for ep in range(episodes):
        env.reset(fault_seed=ep + 10_000, cfg_override=cfg_override)  # eval seeds separate
        for s in range(steps):
            n = env.num_legacy_actions if (restrict_legacy and phase >= 2) else env.num_actions()
            aid = strategy(s, n) % n
            res = env.step(aid)
            tot_ce += res.ce_count
            tot_ue += res.ue_count
        if env.first_error_time is not None:
            failures += 1
            ttff_list.append(env.first_error_time)
        global_cov |= env.seen_sigs

    avg_ttff = sum(ttff_list) / len(ttff_list) if ttff_list else 0.0

    return {
        "phase": phase,
        "ce": tot_ce,
        "ue": tot_ue,
        "failures": failures,
        "avg_ttff": avg_ttff,
        "coverage": len(global_cov),
    }


def print_compare(rows: List[dict]) -> None:
    """Pretty print a comparison of multiple strategies."""
    print("\n" + "=" * 100)
    print(f"{'STRATEGY':<14} | {'FAIL(Ep)':<8} | {'COVERAGE':<10} | {'CE':<10} | {'UE':<10} | {'AVG_TTFF':<12}")
    print("-" * 100)
    for r in rows:
        print(f"{r['name']:<14} | {r['failures']:<8} | {r['coverage']:<10} | {r['ce']:<10} | {r['ue']:<10} | {r['avg_ttff']:<12.1f}")
    print("=" * 100 + "\n")


###############################################################################
# 7.5) Visualizations (Yield curve + Pattern-space PCA)
###############################################################################

def _require_plot_deps() -> None:
    if np is None or plt is None:
        raise RuntimeError(
            "Plotting requires 'numpy' and 'matplotlib'. Install them with: pip install numpy matplotlib"
        )


def _bandit_success_weight(phase: int, res: StepResult) -> Tuple[bool, float]:
    """Shared update rule used by training and online yield-curve TS."""
    if phase == 1:
        success = (res.ce_count > 0)
        w = 1.0 + 0.25 * min(res.ce_count, 20) + 0.4 * res.new_ce_sigs
        return success, w
    # phase 2
    success = (res.new_sigs > 0) or (res.ue_count > 0)
    w = 1.0 + 0.35 * res.new_sigs + 0.6 * res.new_ue_sigs
    return success, w


def _run_yield_trace(
    *,
    phase: int,
    steps: int,
    mem_mb: int,
    region_kb: int,
    cfg_override: dict,
    fault_seed: int,
    strategy: str,
    policy_actions: Optional[List[int]] = None,
    rng_seed: int = 123,
) -> Tuple[List[int], List[int], List[int]]:
    """Return (time, cumulative_unique_sigs, actions_taken) for a single run."""
    env = SatEnv(mem_mb=mem_mb, region_kb=region_kb, phase=phase, seed=rng_seed)
    env.reset(fault_seed=fault_seed, cfg_override=cfg_override)

    assert env.backend is not None
    n_all = env.num_actions()
    n_legacy = env.num_legacy_actions

    rng = random.Random(rng_seed)

    # For the online TS strategy
    a = [1.0] * n_all
    b = [1.0] * n_all

    xs: List[int] = []
    ys: List[int] = []
    acts: List[int] = []

    for s in range(steps):
        if strategy == "legacy_seq":
            aid = s % n_legacy
        elif strategy == "legacy_rand":
            aid = rng.randrange(n_legacy)
        elif strategy == "policy":
            if not policy_actions:
                raise ValueError("policy_actions is required for strategy='policy'")
            aid = int(policy_actions[s % len(policy_actions)])
        elif strategy == "online_ts":
            best = 0
            best_theta = -1.0
            # Thompson sample across all actions (still cheap: <1k actions)
            for i in range(n_all):
                theta = rng.betavariate(a[i], b[i])
                if theta > best_theta:
                    best_theta = theta
                    best = i
            aid = best
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        first_before = (env.first_error_time is None)
        res = env.step(aid)
        acts.append(aid)

        if strategy == "online_ts":
            ok, w = _bandit_success_weight(phase, res)
            if ok:
                a[aid] += w
            else:
                b[aid] += w

        xs.append(env.backend.time)
        ys.append(len(env.seen_sigs))

    return xs, ys, acts


def plot_yield_curve(
    *,
    phase: int,
    steps: int,
    mem_mb: int,
    region_kb: int,
    cfg_override: dict,
    fault_seed: int,
    out_png: str,
    policy_actions: Optional[List[int]] = None,
) -> None:
    """Generate the cumulative unique-error yield curve plot."""
    _require_plot_deps()

    curves: List[Tuple[str, List[int], List[int]]] = []

    x, y, _ = _run_yield_trace(
        phase=phase,
        steps=steps,
        mem_mb=mem_mb,
        region_kb=region_kb,
        cfg_override=cfg_override,
        fault_seed=fault_seed,
        strategy="legacy_seq",
    )
    curves.append(("Legacy-Sequential", x, y))

    x, y, _ = _run_yield_trace(
        phase=phase,
        steps=steps,
        mem_mb=mem_mb,
        region_kb=region_kb,
        cfg_override=cfg_override,
        fault_seed=fault_seed,
        strategy="legacy_rand",
    )
    curves.append(("Legacy-Random", x, y))

    x, y, _ = _run_yield_trace(
        phase=phase,
        steps=steps,
        mem_mb=mem_mb,
        region_kb=region_kb,
        cfg_override=cfg_override,
        fault_seed=fault_seed,
        strategy="online_ts",
    )
    curves.append(("RL-OnlineTS", x, y))

    if policy_actions:
        x, y, _ = _run_yield_trace(
            phase=phase,
            steps=steps,
            mem_mb=mem_mb,
            region_kb=region_kb,
            cfg_override=cfg_override,
            fault_seed=fault_seed,
            strategy="policy",
            policy_actions=policy_actions,
        )
        curves.append(("RL-Policy", x, y))

    plt.figure()
    for name, xs, ys in curves:
        plt.plot(xs, ys, label=name)
    plt.xlabel("DRAM cycles (simulated time)")
    plt.ylabel("Cumulative unique signatures (CE/UE)")
    plt.title(f"Yield Curve (phase {phase})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _pattern_stream(action: Dict, n_words: int) -> List[int]:
    """Generate the first n_words of the pattern's value stream (fast)."""
    kind = action["kind"]
    busshift = int(action.get("busshift", 0))
    invert = bool(action.get("invert", False))

    out: List[int] = [0] * n_words
    if kind == "gsat":
        vals = action["vals"]
        m = len(vals)
        for w in range(n_words):
            pi = ((w >> busshift) % m)
            v = u32(vals[pi])
            out[w] = u32(~v) if invert else v
        return out

    # generator
    seed = int(action["seed"])
    swap_mode = int(action["swap_mode"])
    x = seed
    g_prev = -1
    for w in range(n_words):
        g = (w >> busshift)
        while g_prev < g:
            x = lfsr32_step(x)
            g_prev += 1
        v = bitswap32(x, swap_mode)
        out[w] = u32(~v) if invert else u32(v)
    return out


def _pattern_features(action: Dict, n_words: int = 256) -> List[float]:
    """Cheap embedding of a pattern into a numeric feature vector for PCA."""
    _require_plot_deps()
    kind = action["kind"]
    busshift = int(action.get("busshift", 0))
    buswidth = int(action.get("buswidth", 32))
    invert = 1.0 if action.get("invert", False) else 0.0

    stream = _pattern_stream(action, n_words)
    pop = np.array([popcount32(v) for v in stream], dtype=np.float32) / 32.0
    mean_pop = float(pop.mean())
    std_pop = float(pop.std())

    if n_words >= 2:
        hd = np.array([popcount32(stream[i] ^ stream[i - 1]) for i in range(1, n_words)], dtype=np.float32) / 32.0
        mean_hd = float(hd.mean())
        repeat = float(np.mean(np.array(stream[1:]) == np.array(stream[:-1])))
    else:
        mean_hd = 0.0
        repeat = 0.0

    # Address locality knobs (only generator actions have them)
    stride = float(action.get("stride", 1)) if kind == "gen" else 1.0
    hot = float(action.get("hotset_words", 0)) if kind == "gen" else 0.0
    swap_mode = float(action.get("swap_mode", 0)) if kind == "gen" else 0.0
    seed_lo = float(action.get("seed", 0) & 0xFFFF) / 65535.0 if kind == "gen" else 0.0

    # Normalize / squash
    stride_feat = math.log2(max(1.0, stride)) / 5.0  # strides 1..32
    hot_feat = min(hot, float(n_words)) / float(n_words)

    return [
        1.0 if kind == "gen" else 0.0,
        invert,
        busshift / 3.0,
        buswidth / 256.0,
        mean_pop,
        std_pop,
        mean_hd,
        repeat,
        stride_feat,
        hot_feat,
        swap_mode / 4.0,
        seed_lo,
    ]


def _pca2(X: "np.ndarray") -> "np.ndarray":
    """2D PCA using eigh (no sklearn). X is (n,d)."""
    X0 = X - X.mean(axis=0, keepdims=True)
    cov = (X0.T @ X0) / max(1, (X0.shape[0] - 1))
    vals, vecs = np.linalg.eigh(cov)
    W = vecs[:, -2:]
    return X0 @ W


def plot_pattern_space(
    *,
    phase: int,
    mem_mb: int,
    region_kb: int,
    cfg_override: dict,
    out_png: str,
    policy_actions: Optional[List[int]] = None,
    n_words: int = 256,
    fault_seed: int = 123,
) -> None:
    """Plot legacy patterns vs successful RL patterns in a 2D PCA space."""
    _require_plot_deps()

    env = SatEnv(mem_mb=mem_mb, region_kb=region_kb, phase=phase, seed=0)

    # Choose 'successful' patterns
    red: List[int] = []
    if policy_actions:
        red = list(dict.fromkeys(int(a) for a in policy_actions))
    else:
        # No policy provided: run a short online TS and keep actions that produced errors.
        xs, ys, acts = _run_yield_trace(
            phase=phase,
            steps=min(120, env.num_actions()),
            mem_mb=mem_mb,
            region_kb=region_kb,
            cfg_override=cfg_override,
            fault_seed=fault_seed,
            strategy="online_ts",
        )
        # Re-run to measure which actions actually yielded errors (deterministic seed)
        env2 = SatEnv(mem_mb=mem_mb, region_kb=region_kb, phase=phase, seed=0)
        env2.reset(fault_seed=fault_seed, cfg_override=cfg_override)
        for aid in acts:
            res = env2.step(aid)
            if res.errors > 0:
                red.append(aid)
        red = list(dict.fromkeys(red))[:80]

    feats: List[List[float]] = []
    colors: List[int] = []  # 0=legacy blue, 1=rl red
    for i, act in enumerate(env.actions):
        feats.append(_pattern_features(act, n_words=n_words))
        if i in red:
            colors.append(1)
        elif act["kind"] == "gsat":
            colors.append(0)
        else:
            colors.append(2)  # other generator actions (grey)

    X = np.array(feats, dtype=np.float32)
    Y = _pca2(X)

    plt.figure()
    Yb = Y[np.array(colors) == 0]
    Yr = Y[np.array(colors) == 1]
    Yg = Y[np.array(colors) == 2]

    if len(Yb) > 0:
        plt.scatter(Yb[:, 0], Yb[:, 1], s=12, alpha=0.55, label="Legacy GSAT")
    if len(Yg) > 0:
        plt.scatter(Yg[:, 0], Yg[:, 1], s=10, alpha=0.18, label="Other Generated")
    if len(Yr) > 0:
        plt.scatter(Yr[:, 0], Yr[:, 1], s=18, alpha=0.85, label="RL Successful")

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title(f"Pattern Space (phase {phase})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


###############################################################################
# 8) CLI
###############################################################################

def parse_cfg(args) -> dict:
    """Convert CLI overrides into a dictionary for FaultConfig."""
    d: dict = {}
    # knobs you may want to override
    if args.ce_site_prob is not None:
        d["ce_site_prob"] = args.ce_site_prob
    if args.ce_flip_prob is not None:
        d["ce_flip_prob"] = args.ce_flip_prob
    if args.burst_prob is not None:
        d["burst_prob"] = args.burst_prob
    if args.retention_prob is not None:
        d["retention_prob"] = args.retention_prob
    if args.hammer_threshold is not None:
        d["hammer_threshold"] = args.hammer_threshold
    if args.hammer_flip_prob is not None:
        d["hammer_flip_prob"] = args.hammer_flip_prob
    return d


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(pp):
        pp.add_argument("--phase", type=int, choices=[1, 2], required=True)
        pp.add_argument("--mem-mb", type=int, default=64)
        pp.add_argument("--region-kb", type=int, default=64)
        pp.add_argument("--episodes", type=int, default=200)
        pp.add_argument("--steps", type=int, default=8)
        pp.add_argument("--seed", type=int, default=0)
        # fault overrides (optional)
        pp.add_argument("--ce-site-prob", type=float, default=None)
        pp.add_argument("--ce-flip-prob", type=float, default=None)
        pp.add_argument("--burst-prob", type=float, default=None)
        pp.add_argument("--retention-prob", type=float, default=None)
        pp.add_argument("--hammer-threshold", type=int, default=None)
        pp.add_argument("--hammer-flip-prob", type=float, default=None)

    p_train = sub.add_parser("train")
    add_common(p_train)
    p_train.add_argument("--out", default="policy.json")

    p_eval = sub.add_parser("eval")
    add_common(p_eval)
    p_eval.add_argument("--policy", required=True)

    p_comp = sub.add_parser("compare")
    add_common(p_comp)
    p_comp.add_argument("--policy", required=True)

    p_yield = sub.add_parser("yield", help="Generate a cumulative yield-curve plot (legacy vs RL)")
    add_common(p_yield)
    p_yield.add_argument("--out", default="yield_curve.png")
    p_yield.add_argument("--fault-seed", type=int, default=12345)
    p_yield.add_argument("--policy", default=None, help="Optional policy JSON to plot alongside online RL")

    p_ps = sub.add_parser("patternspace", help="Plot 2D PCA map of pattern space (legacy vs discovered)")
    add_common(p_ps)
    p_ps.add_argument("--out", default="pattern_space.png")
    p_ps.add_argument("--fault-seed", type=int, default=12345)
    p_ps.add_argument("--policy", default=None, help="Optional policy JSON to highlight actions")

    args = p.parse_args()
    cfg = parse_cfg(args)

    if args.cmd == "train":
        train_bandit(args.phase, args.episodes, args.steps, args.mem_mb, args.region_kb, args.seed, args.out, cfg)

    elif args.cmd == "eval":
        with open(args.policy, "r") as f:
            pol = json.load(f)
        actions = pol["actions"]
        def strat(step: int, n: int) -> int:
            return int(actions[step % len(actions)])
        r = run_eval(args.phase, args.episodes, args.steps, args.mem_mb, args.region_kb, strat, cfg)
        print(f"[eval] phase={args.phase} CE={r['ce']} UE={r['ue']} coverage={r['coverage']} failures={r['failures']} avg_TTFF={r['avg_ttff']:.1f}")

    elif args.cmd == "compare":
        with open(args.policy, "r") as f:
            pol = json.load(f)
        pol_actions = pol["actions"]
        rng = random.Random(123)
        def seq(step: int, n: int) -> int:
            return step % n
        def rnd(step: int, n: int) -> int:
            return rng.randrange(n)
        def rl(step: int, n: int) -> int:
            return int(pol_actions[step % len(pol_actions)])
        rows = []
        for name, st in [("Sequential", seq), ("Random", rnd), ("RL", rl)]:
            restrict = (args.phase >= 2 and name in ("Sequential", "Random"))
            rr = run_eval(args.phase, args.episodes, args.steps, args.mem_mb, args.region_kb, st, cfg, restrict_legacy=restrict)
            rr["name"] = name
            rows.append(rr)
        print_compare(rows)

    elif args.cmd == "yield":
        policy_actions = None
        if args.policy:
            with open(args.policy, "r") as f:
                pol = json.load(f)
            policy_actions = [int(x) for x in pol.get("actions", [])]
        plot_yield_curve(
            phase=args.phase,
            steps=args.steps,
            mem_mb=args.mem_mb,
            region_kb=args.region_kb,
            cfg_override=cfg,
            fault_seed=args.fault_seed,
            out_png=args.out,
            policy_actions=policy_actions,
        )
        print(f"[yield] wrote {args.out}")

    elif args.cmd == "patternspace":
        policy_actions = None
        if args.policy:
            with open(args.policy, "r") as f:
                pol = json.load(f)
            policy_actions = [int(x) for x in pol.get("actions", [])]
        plot_pattern_space(
            phase=args.phase,
            mem_mb=args.mem_mb,
            region_kb=args.region_kb,
            cfg_override=cfg,
            fault_seed=args.fault_seed,
            out_png=args.out,
            policy_actions=policy_actions,
        )
        print(f"[patternspace] wrote {args.out}")

if __name__ == "__main__":
    main()
