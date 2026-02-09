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
