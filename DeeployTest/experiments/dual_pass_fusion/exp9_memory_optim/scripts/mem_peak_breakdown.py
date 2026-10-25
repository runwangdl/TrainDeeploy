#!/usr/bin/env python3
# exp9 memsim: L2 peak of the ZO-family recipes broken down by role (tensors alive at the peak step of
# deeployStates/memory_alloc.html; aliases sharing one address range counted once) + static constants + ECO momentum
# (harness, from run.log). -- QW
#   python3 mem_peak_breakdown.py            (reads ../memsim/<cfg>/, prints a markdown table)
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.join(HERE, "..", "memsim")
CFGS = [("zo", "ZO"), ("qzo", "QZO (unfused)"), ("eco", "QZO+ECO (unfused)"), ("eco_fus", "QZO+ECO (fused, cluster)"),
        ("eco_fus_ne16", "QZO+ECO (fused, NE16 b0-2)")]
CATS = ["block-0 activations (conv / BN output)", "conv input (transposed)", "im2col column (NE16)", "data window",
        "trainable parameters (master copy)", "perturbed parameter copies", "NE16 encoded weights + bias",
        "BN statistics", "label", "static constants", "ECO momentum"]

def cat(n):
    if n == "input_0": return "data window"
    if n == "input_1": return "label"
    if re.fullmatch(r"input_\d+", n): return "trainable parameters (master copy)"
    if "ne16enc" in n or "ne16add2" in n: return "NE16 encoded weights + bias"
    if "_col_" in n: return "im2col column (NE16)"
    if "running_" in n or "saved_" in n: return "BN statistics"
    if "conv_output" in n or "Conv__0_tensor" in n or "BatchNormalization__0_tensor" in n:
        return "block-0 activations (conv / BN output)"
    if "input_quant" in n: return "conv input (transposed)"
    return "perturbed parameter copies"

def peak(cfg):
    h = open(os.path.join(M, cfg, "deeployStates", "memory_alloc.html")).read()
    i = h.find('id="plot-L2"'); j = h.find("var fig = ", i); k = h.find(";\n", j); fig = json.loads(h[j + 10:k])
    ts = [t for t in fig["data"] if t.get("x") and t.get("fill") == "toself" and (t.get("line") or {}).get("color") != "red"]
    const = [t for t in ts if t.get("fillcolor") == "lightblue"]; var = [t for t in ts if t.get("fillcolor") != "lightblue"]
    alive = lambda t, s: min(t["x"]) <= s <= max(t["x"])
    def live(s):
        iv = sorted((min(t["y"]), max(t["y"])) for t in var if alive(t, s)); tot, cur = 0, None
        for a, b in iv:
            if cur is None or a > cur[1]: tot += (cur[1] - cur[0]) if cur else 0; cur = [a, b]
            else: cur[1] = max(cur[1], b)
        return tot + ((cur[1] - cur[0]) if cur else 0)
    steps = int(max(max(t["x"]) for t in ts) + 1); s = max(range(steps), key = live)
    out = dict.fromkeys(CATS, 0); seen = set()
    for t in sorted((t for t in var if alive(t, s)), key = lambda t: min(t["y"])):
        rng = (min(t["y"]), max(t["y"]))
        if rng in seen: continue          # alias (e.g. a Reshape view of the same buffer)
        seen.add(rng); out[cat(t.get("name", ""))] += rng[1] - rng[0]
    out["static constants"] = sum(max(t["y"]) - min(t["y"]) for t in const)
    log = open(os.path.join(M, cfg, "run.log"), errors = "ignore").read()
    m = re.search(r"\[ECO\].*?(\d+) B momentum", log); out["ECO momentum"] = int(m.group(1)) if m else 0
    return s, live(s), out

res = {c: peak(c) for c, _ in CFGS}
print("| L2 at the peak | " + " | ".join(l for _, l in CFGS) + " |"); print("|---|" + "--:|" * len(CFGS))
for k in CATS:
    print(f"| {k} | " + " | ".join(f"{res[c][2][k]:,}" if res[c][2][k] else "—" for c, _ in CFGS) + " |")
print("| **total** | " + " | ".join(f"**{sum(res[c][2].values()):,}**" for c, _ in CFGS) + " |")
print("| peak step / live variables | " + " | ".join(f"{res[c][0]} / {res[c][1]:,}" for c, _ in CFGS) + " |")
