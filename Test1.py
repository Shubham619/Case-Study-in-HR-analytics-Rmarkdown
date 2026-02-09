# sat_pattern_catalog.py
# Hardcoded pattern.cc catalog (NO parsing). Includes all families and variants.

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

MASK32 = 0xFFFFFFFF

@dataclass(frozen=True)
class PatternFamily:
    name: str
    data_u32: List[int]          # full array as in pattern.cc (includes last sentinel)
    weights_32_64_128_256: Tuple[int, int, int, int]

    @property
    def count(self) -> int:
        # pattern.cc uses: (sizeof data / sizeof data[0]) - 1
        # So the last entry is effectively "sentinel"/not part of the repeating pattern length.
        return max(0, len(self.data_u32) - 1)

@dataclass(frozen=True)
class PatternVariant:
    family: PatternFamily
    buswidth: int                # 32/64/128/256
    invert: bool
    weight: int                  # selected weight for this buswidth (may be 0)
    busshift: int                # 32->0, 64->1, 128->2, 256->3

    @property
    def name(self) -> str:
        # matches C++ naming style: "<family><~?><buswidth>"
        return f"{self.family.name}{'~' if self.invert else ''}{self.buswidth}"

    def word(self, i: int) -> int:
        """
        Return the i-th 32-bit word of the pattern stream (repeats forever).
        busshift reduces how fast the pattern advances as bus width increases.
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

    def block_words(self, n_words: int, start_i: int = 0) -> List[int]:
        return [self.word(start_i + k) for k in range(n_words)]


# -------------------------
# 1) Hardcoded arrays (exactly from your paste)
# -------------------------

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

OneZero_data      = [0x00000000, 0xffffffff]
JustZero_data     = [0x00000000, 0x00000000]
JustOne_data      = [0xffffffff, 0xffffffff]
JustFive_data     = [0x55555555, 0x55555555]
JustA_data        = [0xaaaaaaaa, 0xaaaaaaaa]
FiveA_data        = [0x55555555, 0xaaaaaaaa]
FiveA8_data       = [0x5aa5a55a, 0xa55a5aa5, 0xa55a5aa5, 0x5aa5a55a]
Long8b10b_data    = [0x16161616, 0x16161616]
Short8b10b_data   = [0xb5b5b5b5, 0xb5b5b5b5]
Checker8b10b_data = [0xb5b5b5b5, 0x4a4a4a4a]
Five7_data        = [0x55555557, 0x55575555]
Zero2fd_data      = [0x00020002, 0xfffdfffd]


# -------------------------
# 2) Families + weights (exactly as in your paste)
# -------------------------

FAMILIES: List[PatternFamily] = [
    PatternFamily("walkingOnes",     walkingOnes_data,     (1, 1, 2, 1)),
    PatternFamily("walkingInvOnes",  walkingInvOnes_data,  (2, 2, 5, 5)),
    PatternFamily("walkingZeros",    walkingZeros_data,    (1, 1, 2, 1)),
    PatternFamily("OneZero",         OneZero_data,         (5, 5, 15, 5)),
    PatternFamily("JustZero",        JustZero_data,        (2, 0, 0, 0)),
    PatternFamily("JustOne",         JustOne_data,         (2, 0, 0, 0)),
    PatternFamily("JustFive",        JustFive_data,        (2, 0, 0, 0)),
    PatternFamily("JustA",           JustA_data,           (2, 0, 0, 0)),
    PatternFamily("FiveA",           FiveA_data,           (1, 1, 1, 1)),
    PatternFamily("FiveA8",          FiveA8_data,          (1, 1, 1, 1)),
    PatternFamily("Long8b10b",       Long8b10b_data,       (2, 0, 0, 0)),
    PatternFamily("Short8b10b",      Short8b10b_data,      (2, 0, 0, 0)),
    PatternFamily("Checker8b10b",    Checker8b10b_data,    (1, 0, 0, 1)),
    PatternFamily("Five7",           Five7_data,           (0, 2, 0, 0)),
    PatternFamily("Zero2fd",         Zero2fd_data,         (0, 2, 0, 0)),
]


def _busshift_for_width(buswidth: int) -> int:
    if buswidth == 32: return 0
    if buswidth == 64: return 1
    if buswidth == 128: return 2
    if buswidth == 256: return 3
    raise ValueError(f"Unsupported buswidth: {buswidth}")


def all_variants(include_zero_weight: bool = True) -> List[PatternVariant]:
    """
    Matches PatternList::Initialize() expansion:
    for each family: widths 32/64/128/256 and then inverted 32/64/128/256.
    In C++ it creates all variants, but selection uses weights.
    """
    variants: List[PatternVariant] = []
    for fam in FAMILIES:
        for invert in (False, True):
            for bw, w in zip((32, 64, 128, 256), fam.weights_32_64_128_256):
                if (not include_zero_weight) and (w <= 0):
                    continue
                variants.append(
                    PatternVariant(
                        family=fam,
                        buswidth=bw,
                        invert=invert,
                        weight=w,
                        busshift=_busshift_for_width(bw),
                    )
                )
    return variants


def selectable_variants() -> List[PatternVariant]:
    """Variants with weight > 0 (these are actually selectable like GetRandomPattern())."""
    return all_variants(include_zero_weight=False)


def weighted_random_variant(rng: Optional[random.Random] = None) -> PatternVariant:
    """
    Equivalent to PatternList::GetRandomPattern() over patterns with weight > 0.
    """
    rng = rng or random
    vars_ = selectable_variants()
    weights = [v.weight for v in vars_]
    # Python 3.6+: random.choices supports weights
    return rng.choices(vars_, weights=weights, k=1)[0]





import re
import json
from pathlib import Path

BUS_WIDTHS = [32, 64, 128, 256]

def parse_uint_array(text, array_name):
    # matches: static unsigned int NAME_data[] = { ... };
    # captures body between braces
    m = re.search(rf"static\s+unsigned\s+int\s+{re.escape(array_name)}\s*\[\]\s*=\s*{{(.*?)}};",
                  text, re.S)
    if not m:
        raise ValueError(f"Array not found: {array_name}")
    body = m.group(1)
    # grab hex like 0x..., allow commas/newlines/spaces
    vals = re.findall(r"0x[0-9a-fA-F]+", body)
    return [int(v, 16) for v in vals]

def parse_pattern_data_blocks(text):
    """
    Finds blocks like:
    static const struct PatternData walkingOnes = {
      "walkingOnes",
      walkingOnes_data,
      (sizeof walkingOnes_data / sizeof walkingOnes_data[0]) - 1,
      {1, 1, 2, 1}
    };
    """
    pat = re.compile(
        r"static\s+const\s+struct\s+PatternData\s+(\w+)\s*=\s*{\s*"
        r"\"([^\"]+)\"\s*,\s*"
        r"(\w+)\s*,\s*"
        r".*?\s*,\s*"
        r"{\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*}\s*"
        r"}\s*;",
        re.S
    )
    out = {}
    for m in pat.finditer(text):
        sym, name, data_sym, w0, w1, w2, w3 = m.groups()
        out[sym] = {
            "name": name,
            "data_sym": data_sym,
            "weights": [int(w0), int(w1), int(w2), int(w3)]
        }
    if not out:
        raise ValueError("No PatternData blocks found. File may be incomplete.")
    return out

def parse_pattern_array_order(text):
    # matches:
    # static const struct PatternData pattern_array[] = { a, b, c, ... };
    m = re.search(r"pattern_array\[\]\s*=\s*{\s*(.*?)\s*};", text, re.S)
    if not m:
        raise ValueError("pattern_array[] not found")
    body = m.group(1)
    syms = [s.strip() for s in body.split(",") if s.strip()]
    return syms

def build_actions(pattern_data, pattern_order, arrays):
    actions = []
    order_idx = 0
    for sym in pattern_order:
        pd = pattern_data[sym]
        data = arrays[pd["data_sym"]]
        for inv in [False, True]:
            for bi, bw in enumerate(BUS_WIDTHS):
                w = pd["weights"][bi]
                # weight controls sampling; if 0, that variant never selected
                if w <= 0:
                    continue
                actions.append({
                    "id": order_idx,
                    "family_sym": sym,
                    "name": pd["name"],
                    "buswidth": bw,
                    "invert": inv,
                    "weight": w,
                    # sequence length used by that PatternData (count in C++ is len-1)
                    "sequence_len": max(0, len(data) - 1),
                    "data_words_u32": data,  # full sequence (includes last sentinel)
                })
                order_idx += 1
    return actions

def top64_by_weight(actions):
    # sort by weight desc, then stable tie-break by (name,buswidth,invert)
    ranked = sorted(actions, key=lambda a: (-a["weight"], a["name"], a["buswidth"], a["invert"]))
    top = ranked[:64]
    # reassign compact action_id 0..63 for RL/baseline
    for i, a in enumerate(top):
        a["action_id"] = i
    return top

def main():
    src = Path("pattern.cc").read_text(errors="ignore")

    pattern_data = parse_pattern_data_blocks(src)
    pattern_order = parse_pattern_array_order(src)

    # gather all *_data arrays referenced
    arrays = {}
    for sym, pd in pattern_data.items():
        arrays[pd["data_sym"]] = parse_uint_array(src, pd["data_sym"])

    actions = build_actions(pattern_data, pattern_order, arrays)
    print(f"Total selectable variants with weight>0: {len(actions)}")

    gsat64 = top64_by_weight(actions)
    print(f"Selected GSAT64 actions: {len(gsat64)}")

    Path("gsat64_actions.json").write_text(json.dumps(gsat64, indent=2))
    print("Wrote gsat64_actions.json")

if __name__ == "__main__":
    main()
