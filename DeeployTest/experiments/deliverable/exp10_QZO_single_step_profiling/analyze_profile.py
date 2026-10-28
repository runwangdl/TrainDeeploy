# SPDX-License-Identifier: MIT
"""exp10 — parse --profileTiling logs (float ZO exp6 vs QZO exp10 single step) and produce the
latency breakdown: compute (Kernel) vs overhead (Pre+Post-Kernel = DMA/tiling) cycles, per
operator class and per node. Outputs results/breakdown.json, comparison PNGs, and a summary.

Log grammar (per tile):
  ===== Profiling <node> =====
  [<node>][SB][N ops][Tile k] Pre-Kernel :  C cycles
  [<node>][SB][N ops][Tile k] Kernel     :  C cycles
  [<node>][SB][N ops][Tile k] Post-Kernel:  C cycles
Pre/Post = tiling + DMA in/out around the kernel; Kernel = compute.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOAT_LOG = HERE.parent / "exp6_ZO_single_step_latency/logs/profiletiling.log"
QZO_LOG = HERE / "logs/profiletiling.log"

LINE = re.compile(r"^\[([^\]]+)\]\[SB\]\[\s*\d+ ops\]\[Tile\s+\d+\]\s+"
                  r"(Pre-Kernel|Kernel|Post-Kernel)\s*:\s*(\d+) cycles")

CLASSES = [  # (keyword (lowercase), class) — first match wins
    ("perturbrademacher", "Perturb"),
    ("rqsperturb", "Perturb"),
    # transpose BEFORE the rqsp_/pert_ prefixes: "..._pert_tensor_transpose" nodes are the
    # NHWC data-movement of the perturbed weight, not perturb compute -- QW
    ("transpose", "Transpose"),
    # QW: QZO node names — rqsp_* (int8 weight / int32 bias RQSPerturb, incl. rqsp_upd_*) and
    #     pert_* (fp32 BN/fc perturbs, incl. pert_upd_*). Without these they were silently
    #     absorbed into Conv ("...conv...") and Other — found via exp10a review. -- QW
    ("rqsp_", "Perturb"),
    ("pert_", "Perturb"),
    ("requantshift", "RequantShift"),
    ("requant", "RequantShift"),
    ("dequant", "Quant/Dequant"),
    ("quant", "Quant/Dequant"),
    ("batchnorm", "BatchNorm"),
    ("_bn_", "BatchNorm"),
    ("maxpool", "MaxPool"),
    ("averagepool", "GlobalAvgPool"),
    ("globalaverage", "GlobalAvgPool"),
    ("softmax", "Loss(SCE)"),
    ("crossentropy", "Loss(SCE)"),
    ("sce", "Loss(SCE)"),
    ("gemm", "Gemm/FC"),
    ("matmul", "Gemm/FC"),
    ("conv", "Conv"),
    ("relu", "ReLU"),
    ("add", "Add"),
    ("mul", "Mul"),
    ("div", "Div"),
    ("pad", "Pad"),
    ("reshape", "Reshape"),
]


def classify(node: str) -> str:
    n = node.lower()
    for kw, cls in CLASSES:
        if kw in n:
            return cls
    return "Other"


def parse(path: Path):
    per_node = defaultdict(lambda: dict(kernel=0, overhead=0, tiles=0))
    for line in open(path, errors="replace"):
        m = LINE.match(line.strip())
        if not m:
            continue
        node, kind, cyc = m.group(1), m.group(2), int(m.group(3))
        if kind == "Kernel":
            per_node[node]["kernel"] += cyc
            per_node[node]["tiles"] += 1
        else:
            per_node[node]["overhead"] += cyc
    per_class = defaultdict(lambda: dict(kernel=0, overhead=0, nodes=0))
    for node, d in per_node.items():
        c = classify(node)
        per_class[c]["kernel"] += d["kernel"]
        per_class[c]["overhead"] += d["overhead"]
        per_class[c]["nodes"] += 1
    return per_node, per_class


# normalization: exp6 ran n_accum=1 (1 loss pair = 2 forwards); the exp10 profiled run
# n_accum=4 (4 pairs). Normalize everything to ONE loss pair for a fair comparison.
N_PAIRS = {"float_zo": 1, "qzo": 4}


def main():
    out = {}
    for tag, path in (("float_zo", FLOAT_LOG), ("qzo", QZO_LOG)):
        if not path.exists():
            print(f"({tag}: {path} missing, skipping)")
            continue
        per_node, per_class = parse(path)
        np_ = N_PAIRS[tag]
        for d in per_node.values():
            d["kernel"] //= np_; d["overhead"] //= np_
        for d in per_class.values():
            d["kernel"] //= np_; d["overhead"] //= np_
        tot_k = sum(d["kernel"] for d in per_class.values())
        tot_o = sum(d["overhead"] for d in per_class.values())
        out[tag] = dict(
            total_kernel=tot_k, total_overhead=tot_o, total=tot_k + tot_o,
            per_class={c: d for c, d in sorted(per_class.items(),
                                               key=lambda kv: -(kv[1]["kernel"] + kv[1]["overhead"]))},
            top_nodes=[dict(node=n, **d) for n, d in sorted(
                per_node.items(), key=lambda kv: -(kv[1]["kernel"] + kv[1]["overhead"]))[:15]])
        print(f"\n== {tag}: total {tot_k+tot_o:,} cyc = {tot_k:,} compute ({100*tot_k/(tot_k+tot_o):.1f}%) "
              f"+ {tot_o:,} DMA/tiling overhead ({100*tot_o/(tot_k+tot_o):.1f}%) ==")
        print(f"  {'class':14s} {'kernel':>12s} {'overhead':>12s} {'total':>12s} {'%':>6s} {'nodes':>6s}")
        for c, d in out[tag]["per_class"].items():
            t = d["kernel"] + d["overhead"]
            print(f"  {c:14s} {d['kernel']:12,d} {d['overhead']:12,d} {t:12,d} "
                  f"{100.0*t/(tot_k+tot_o):5.1f}% {d['nodes']:6d}")
    json.dump(out, open(HERE / "results/breakdown.json", "w"), indent=1)

    if len(out) == 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        classes = sorted(set(out["float_zo"]["per_class"]) | set(out["qzo"]["per_class"]),
                         key=lambda c: -(out["qzo"]["per_class"].get(c, {"kernel": 0, "overhead": 0})["kernel"]
                                         + out["qzo"]["per_class"].get(c, {"kernel": 0, "overhead": 0})["overhead"]))
        x = np.arange(len(classes)); w = 0.38
        fig, ax = plt.subplots(figsize=(13, 5.5))
        for i, (tag, color_k, color_o) in enumerate((("float_zo", "#1f77b4", "#aec7e8"),
                                                     ("qzo", "#d62728", "#ff9896"))):
            ks = [out[tag]["per_class"].get(c, {"kernel": 0})["kernel"] / 1e6 for c in classes]
            os_ = [out[tag]["per_class"].get(c, {"overhead": 0}).get("overhead", 0) / 1e6 for c in classes]
            ax.bar(x + (i - 0.5) * w, ks, w, color=color_k, label=f"{tag} compute")
            ax.bar(x + (i - 0.5) * w, os_, w, bottom=ks, color=color_o, label=f"{tag} DMA/tiling")
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_ylabel("Mcycles (single ZO update step)"); ax.set_yscale("log")
        ax.set_title("Single-step ZO latency per operator class — float (exp6) vs QZO (exp10)")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(HERE / "results/per_class_comparison.png", dpi=130)

        fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
        tags = ["float_zo", "qzo"]
        ks = [out[t]["total_kernel"] / 1e6 for t in tags]
        os_ = [out[t]["total_overhead"] / 1e6 for t in tags]
        ax2.bar(tags, ks, 0.5, label="compute (Kernel)", color="#1f77b4")
        ax2.bar(tags, os_, 0.5, bottom=ks, label="DMA/tiling overhead", color="#aec7e8")
        for i, t in enumerate(tags):
            ax2.text(i, (ks[i] + os_[i]) * 1.01, f"{ks[i]+os_[i]:.0f}M", ha="center")
        ax2.set_ylabel("Mcycles"); ax2.set_title("Total single-step cycles: compute vs DMA/tiling")
        ax2.legend(); fig2.tight_layout(); fig2.savefig(HERE / "results/total_comparison.png", dpi=130)
        print("\nplots written: results/per_class_comparison.png, results/total_comparison.png")


if __name__ == "__main__":
    main()
