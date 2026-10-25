#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# exp9 memsim: pie charts of the L2 peak (tensors alive at the peak step of Deeploy's memory plan + static constants
# + harness buffers) for chosen recipes, with the tensors of every slice listed under the pie. -- QW
#   python3 mem_peak_pies.py zo qzo        (needs matplotlib; writes ../memsim/plots/l2_peak_<cfgs>.{png,pdf})
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "..", "memsim")
LABEL = {"zo": "ZO (fp32)", "qzo": "QZO (int8, unfused)", "eco": "QZO+ECO (unfused)",
         "eco_fus": "QZO+ECO (fused, cluster)", "eco_fus_ne16": "QZO+ECO (fused, NE16 blocks 0-2)"}
# slice order and colours (same slice = same colour in every pie)
SLICES = [
    ("Block-0 activations", "#c0392b"),
    ("Conv input (HWC copy)", "#e67e22"),
    ("im2col column (NE16)", "#d35400"),
    ("Data window + label", "#2980b9"),
    ("Trainable parameters (master)", "#27ae60"),
    ("Perturbed parameter copies", "#8e44ad"),
    ("NE16 encoded weights + bias", "#7f8c8d"),
    ("BN auxiliary outputs (block 0)", "#95a5a6"),
    ("Static constants", "#bdc3c7"),
    ("ECO momentum (harness)", "#f1c40f"),
]
COLOR = dict(SLICES)

# parameter tensors by graph-input index (SpeechNet QZO/ZO training graphs, inputs 2..23)
_BLK = re.compile(r"blocks_?(\d)")


def slice_of(n):
    if n in ("input_0", "input_1"):
        return "Data window + label"
    if re.fullmatch(r"input_\d+", n):
        return "Trainable parameters (master)"
    if "ne16enc" in n or "ne16add2" in n:
        return "NE16 encoded weights + bias"
    if "_col_" in n:
        return "im2col column (NE16)"
    if "running_" in n or "saved_" in n:
        return "BN auxiliary outputs (block 0)"
    if "conv_output" in n or "Conv__0_tensor" in n or "BatchNormalization__0_tensor" in n:
        return "Block-0 activations"
    if "input_quant" in n:
        return "Conv input (HWC copy)"
    return "Perturbed parameter copies"


def describe(n, nbytes):
    """Human-readable name of one tensor."""
    if n == "input_0":
        return f"data window, fp32 14x700 ({nbytes:,} B)"
    if n == "input_1":
        return f"label ({nbytes} B)"
    if "BatchNormalization__0_tensor" in n:
        return f"BN0 output, fp32 14x701x8, HWC ({nbytes:,} B)"
    if "Conv__0_tensor" in n:
        return f"Conv0 output, fp32 14x701x8, HWC ({nbytes:,} B)"
    if "conv_output" in n and "dual" in n:
        return f"Conv0 output [A+ | A-], int8 14x701x16, HWC ({nbytes:,} B)"
    if "conv_output" in n:
        return f"Conv0 output, int8 14x701x8, HWC ({nbytes:,} B)"
    if "input_quant" in n and "blocks_1" in n:
        return f"block 1's conv input [x+ | x-], int8 14x87x16, HWC ({nbytes:,} B)"
    if "input_quant" in n:
        return f"Conv0 input, int8 14x700, HWC copy ({nbytes:,} B)"
    return None


def peak(cfg):
    h = open(os.path.join(M, cfg, "deeployStates", "memory_alloc.html")).read()
    i = h.find('id="plot-L2"')
    j = h.find("var fig = ", i)
    k = h.find(";\n", j)
    fig = json.loads(h[j + 10:k])
    ts = [t for t in fig["data"] if t.get("x") and t.get("fill") == "toself" and (t.get("line") or {}).get("color") != "red"]
    const = [t for t in ts if t.get("fillcolor") == "lightblue"]
    var = [t for t in ts if t.get("fillcolor") != "lightblue"]

    def alive(t, s):
        return min(t["x"]) <= s <= max(t["x"])

    def live(s):
        iv = sorted((min(t["y"]), max(t["y"])) for t in var if alive(t, s))
        tot, cur = 0, None
        for a, b in iv:
            if cur is None or a > cur[1]:
                tot += (cur[1] - cur[0]) if cur else 0
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        return tot + ((cur[1] - cur[0]) if cur else 0)

    steps = int(max(max(t["x"]) for t in ts) + 1)
    s = max(range(steps), key = live)
    parts = {name: [] for name, _ in SLICES}
    seen = set()
    for t in sorted((t for t in var if alive(t, s)), key = lambda t: min(t["y"])):
        rng = (min(t["y"]), max(t["y"]))
        if rng in seen:  # alias (Reshape view of one buffer)
            continue
        seen.add(rng)
        parts[slice_of(t.get("name", ""))].append((t.get("name", ""), rng[1] - rng[0]))
    cb = sum(max(t["y"]) - min(t["y"]) for t in const)
    if cb:
        parts["Static constants"].append((f"{len(const)} compile-time constant tensors (multipliers, shifts)", cb))
    log = open(os.path.join(M, cfg, "run.log"), errors = "ignore").read()
    m = re.search(r"\[ECO\].*?(\d+) B momentum", log)
    if m:
        parts["ECO momentum (harness)"].append(("22 fp32 buffers, one value per trainable parameter", int(m.group(1))))
    return s, parts


def slice_text(name, items):
    tot = sum(b for _, b in items)
    if name == "Trainable parameters (master)":
        return (f"{len(items)} tensors = the graph's trainable inputs (input_2..23): conv W and b, BN gamma and beta of "
                f"blocks 0-4, FC W and b; live for the whole step (the update graph changes them in place)")
    if name == "Perturbed parameter copies":
        n_all = 22
        miss = "" if len(items) == n_all else (f"; the {n_all - len(items)} copies of block 0's conv W/b are already "
                                               f"freed (block 0's conv has run)")
        if len(items) < 15:  # fused graphs: the conv weights are perturbed inside the conv kernels (L1 scratch)
            return (f"{len(items)} tensors: BN gamma/beta of blocks 0-4 as stacked [+; -] copies and the FC W/b as "
                    f"+ and - copies; the conv weights are perturbed inside the fused conv kernels (L1 scratch), so "
                    f"they have no L2 copy")
        return (f"{len(items)} tensors = theta + eps*z of each trainable tensor (conv W stored transposed to HWC), "
                f"all created at the start of the schedule{miss}")
    descs = [describe(n, b) or n for n, b in items]
    if name == "BN auxiliary outputs (block 0)":
        return (f"{len(items)} extra outputs written by block 0's BatchNormInternal (running mean/var out, saved "
                f"mean/inv_std), unused, alive only around that node, {tot:,} B")
    if name == "im2col column (NE16)":
        return (f"block 0's im2col column: 9,814 output pixels x K = 4 bytes, built by the cluster for the NE16 1x1 conv "
                f"({tot:,} B; its Reshape view is the same buffer)")
    if name == "NE16 encoded weights + bias":
        return (f"{len(items)} tensors: [W+; W-] of blocks 0-2 in NE16 bit-plane format (block 1, 2: 4,096 B each; "
                f"block 0: 256 B) + their corrected requant bias; prepared at the start of the schedule, {tot:,} B")
    if name == "ECO momentum (harness)":
        return f"22 fp32 buffers, one value per trainable parameter (15,489), allocated by the C harness, {tot:,} B"
    return "; ".join(descs)


def main():
    cfgs = sys.argv[1:] or ["zo", "qzo"]
    data = {c: peak(c) for c in cfgs}
    fig, axes = plt.subplots(1, len(cfgs), figsize = (7.5 * len(cfgs), 8.2))
    if len(cfgs) == 1:
        axes = [axes]
    for ax, c in zip(axes, cfgs):
        step, parts = data[c]
        rows = [(n, sum(b for _, b in parts[n])) for n, _ in SLICES if parts[n]]
        total = sum(v for _, v in rows)
        wedges, _ = ax.pie([v for _, v in rows], colors = [COLOR[n] for n, _ in rows], startangle = 90,
                           counterclock = False, wedgeprops = dict(edgecolor = "white", linewidth = 1))
        for w, (n, v) in zip(wedges, rows):  # percentage on slices >= 4 %
            if v / total >= 0.04:
                import math
                ang = math.radians((w.theta1 + w.theta2) / 2)
                ax.text(0.68 * math.cos(ang), 0.68 * math.sin(ang), f"{100 * v / total:.0f}%", ha = "center",
                        va = "center", color = "white", fontsize = 11, fontweight = "bold")
        ax.set_title(f"{LABEL.get(c, c)}\nL2 peak {total:,} B  (schedule step {step})", fontsize = 13)
        legend = [f"{n}: {v:,} B ({100 * v / total:.1f}%)" for n, v in rows]
        ax.legend(wedges, legend, loc = "upper center", bbox_to_anchor = (0.5, -0.02), fontsize = 9.5, frameon = False)
        import textwrap
        detail = "\n".join(textwrap.fill(f"- {n}: {slice_text(n, parts[n])}", 92, subsequent_indent = "   ")
                            for n, _ in rows)
        ax.text(0.0, -0.40, detail, transform = ax.transAxes, ha = "left", va = "top", fontsize = 8)
    fig.suptitle("SpeechNet, one update step on GAP9 (GVSoC), board L1/L2 settings: what occupies L2 at the peak",
                 fontsize = 14, y = 0.99)
    fig.subplots_adjust(top = 0.90)
    out = os.path.join(M, "plots")
    os.makedirs(out, exist_ok = True)
    base = os.path.join(out, "l2_peak_" + "_".join(cfgs))
    fig.savefig(base + ".png", dpi = 150, bbox_inches = "tight")
    fig.savefig(base + ".pdf", bbox_inches = "tight")
    for c in cfgs:
        step, parts = data[c]
        print(f"== {c} (step {step})")
        for n, _ in SLICES:
            for t, b in parts[n]:
                print(f"   {n:32s} {b:>9,}  {t}")
    print(base + ".png")


if __name__ == "__main__":
    main()
