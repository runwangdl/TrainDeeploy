# SPDX-License-Identifier: MIT
"""exp10a — single-step latency comparison: float ZO (exp6) vs QZO on the FULLY-FIXED stack
(corrected per-layer RequantShift constants + parallel fp32 Quant/Dequant kernels), profiled
fresh on the exp9_corr fixture at the EXACT exp6 configuration (n_steps=1, n_accum=1:
1 update step, 1 loss pair, 1 window) — no per-pair normalization involved.

(The earlier n_accum=4 profiled run is archived as logs/profiletiling_qzo_fixed.log.gz; its
per-pair-normalized totals agree with the 1x1 run to ~3%, the residual being update
amortization.)

Outputs: results/comparison.json, results/overall_latency.png, results/per_operator.png
Run in agitated_hugle from the exp10a directory.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analyze_profile import parse  # noqa: E402  (per-tile log parser + op classifier)

FLOAT_LOG = HERE.parent.parent / "exp6_ZO_single_step_latency/logs/profiletiling.log"
QZO_LOG = HERE / "logs/profiletiling_qzo_1x1.log"   # 1 step x 1 accum = exp6-identical config
N_PAIRS = {"float ZO": 1, "QZO (fixed)": 1}        # both 1x1 -> NO normalization needed


def load(tag, path):
    per_node, per_class = parse(path)
    np_ = N_PAIRS[tag]
    cls = {c: dict(kernel=d["kernel"] // np_, overhead=d["overhead"] // np_)
           for c, d in per_class.items()}
    tot_k = sum(d["kernel"] for d in cls.values())
    tot_o = sum(d["overhead"] for d in cls.values())
    return dict(per_class=cls, total_kernel=tot_k, total_overhead=tot_o, total=tot_k + tot_o)


def main():
    data = {tag: load(tag, p) for tag, p in (("float ZO", FLOAT_LOG), ("QZO (fixed)", QZO_LOG))}
    json.dump(data, open(HERE / "results/comparison.json", "w"), indent=1)
    f, q = data["float ZO"], data["QZO (fixed)"]
    print(f"float ZO   : {f['total']:>12,d} cyc/pair ({f['total_kernel']:,} compute + "
          f"{f['total_overhead']:,} DMA/tiling)")
    print(f"QZO (fixed): {q['total']:>12,d} cyc/pair ({q['total_kernel']:,} compute + "
          f"{q['total_overhead']:,} DMA/tiling)")
    print(f"speedup    : {f['total']/q['total']:.2f}x  (QZO vs float ZO)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- plot 1: overall latency ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    tags = list(data)
    ks = [data[t]["total_kernel"] / 1e6 for t in tags]
    os_ = [data[t]["total_overhead"] / 1e6 for t in tags]
    ax.bar(tags, ks, 0.5, label="compute (kernel)", color="#1f77b4")
    ax.bar(tags, os_, 0.5, bottom=ks, label="DMA / tiling overhead", color="#aec7e8")
    for i, t in enumerate(tags):
        ax.text(i, (ks[i] + os_[i]) * 1.02, f"{ks[i]+os_[i]:.1f}M", ha="center",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("Mcycles per ZO loss pair (2 forwards + update)")
    ax.set_title(f"Single-step ZO latency — QZO is "
                 f"{f['total']/q['total']:.1f}× faster than float ZO")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "results/overall_latency.png", dpi=140)

    # ---- plot 2: per-operator breakdown --------------------------------------------------
    classes = sorted(set(f["per_class"]) | set(q["per_class"]),
                     key=lambda c: -(q["per_class"].get(c, {"kernel": 0, "overhead": 0})["kernel"]
                                     + q["per_class"].get(c, {"kernel": 0, "overhead": 0})["overhead"]))
    x = np.arange(len(classes)); w = 0.38
    fig2, ax2 = plt.subplots(figsize=(12.5, 5.2))
    for i, (tag, ck, co) in enumerate((("float ZO", "#1f77b4", "#aec7e8"),
                                       ("QZO (fixed)", "#d62728", "#ff9896"))):
        d = data[tag]["per_class"]
        kk = [d.get(c, {}).get("kernel", 0) / 1e6 for c in classes]
        oo = [d.get(c, {}).get("overhead", 0) / 1e6 for c in classes]
        ax2.bar(x + (i - 0.5) * w, kk, w, color=ck, label=f"{tag} compute")
        ax2.bar(x + (i - 0.5) * w, oo, w, bottom=kk, color=co, label=f"{tag} DMA/tiling")
    ax2.set_xticks(x); ax2.set_xticklabels(classes, rotation=30, ha="right")
    ax2.set_ylabel("Mcycles per loss pair (log)"); ax2.set_yscale("log")
    ax2.set_title("Per-operator latency — float ZO (exp6) vs QZO on the fixed stack (exp10a)")
    ax2.legend(ncols=2); ax2.grid(axis="y", alpha=0.3, which="both")
    fig2.tight_layout(); fig2.savefig(HERE / "results/per_operator.png", dpi=140)
    print("plots: results/overall_latency.png, results/per_operator.png")


if __name__ == "__main__":
    main()
