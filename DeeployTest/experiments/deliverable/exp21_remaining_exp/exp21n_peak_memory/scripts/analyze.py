# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""exp21n: peak memory per recipe from the generated static memory plans (codegen/<recipe>/*Network.c, gen.sh).

peak = activations + static (data window + label, weights, gradient accumulators, optimizer / ECO state, constants)
       + tile buffer (the L1 arena, counted once)
Every allocation is read from the generated C; all arenas are allocated once at init and never freed, so the
reserved total IS the peak. Graph inputs (window, label, weights, BP's accumulators) live for the whole schedule
and sit inside MEMORYARENA_L2 (their pointers are asserted to be arena offsets), so
activations = MEMORYARENA_L2 - graph inputs. The ECO momentum is the harness's pi_l2_malloc
(deeploymezotest.c:636, 4 B per trainable parameter). Not counted (off-chip, identical role for every recipe): the
54-window pool and the initial-weight copies in L3.
    /opt/anaconda3/bin/python3.11 analyze.py   -> results/peak_memory.{md,json}, plots/peak_memory.{png,pdf}
"""
import json, os, re

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.dirname(HERE)
BX = os.path.abspath(os.path.join(E, "..", "..", "..", "Board_exp"))
FIX = {"bp": f"{BX}/BP/fixtures/speechnet_train_BP_r1", "zo": f"{BX}/ZOfam/fixtures/speechnet_train_ZO_r1",
       "eco": f"{BX}/ZOfam/fixtures/speechnet_train_QZO_r1"}
NAME = {"bp": "BP (fp32)", "zo": "ZO (fp32)", "eco": "QZO-ECO (int8, fused NE16)"}
L2_SHARED, L1_CLUSTER = 1_572_864, 131_072   # GAP9 shared L2 (1.5 MiB; + 64 KB private bank) and cluster L1 (128 KiB)
BP_HEAP_RUN = 1_377_216   # L2 heap left for BP's arena on the board: Run, silentwear-zo/notes/README.md:252 ("1,478,772 > 1,377,216")
ELEM = {1: 4, 2: 1, 3: 1, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4}
CSIZE = {"int8_t": 1, "uint8_t": 1, "int16_t": 2, "uint16_t": 2, "int32_t": 4, "uint32_t": 4, "float32_t": 4,
         "float": 4, "int64_t": 8, "uint64_t": 8, "double": 8}


def nbytes(i):
    t = i.type.tensor_type
    return int(np.prod([d.dim_value for d in t.shape.dim])) * ELEM[t.elem_type]


def category(name):
    if name in ("input", "label", "labels"):
        return "data window + label"
    if "reset" in name.lower():                          # BP lazy_reset_grad flag
        return "other inputs"
    if "grad.accumulation.buffer" in name:               # BP InPlaceAccumulatorV2 buffers
        return "gradient accumulators"
    return "weights"


def main():
    out = {}
    for r in ("bp", "zo", "eco"):
        cg = os.path.join(E, "codegen", r)
        tr = open(os.path.join(cg, "TrainingNetwork.c")).read()
        op = open(os.path.join(cg, "OptimizerNetwork.c")).read()
        l2 = [int(x) for x in re.findall(r"MEMORYARENA_L2 = \(int8_t \*\)pi_l2_malloc\(sizeof\(int8_t\) \* (\d+)\)", tr)]
        l1 = [int(x) for x in re.findall(r"MEMORYARENA_L1 = \(int8_t \*\)pi_l1_malloc\(\(void \*\)0, sizeof\(int8_t\) \* (\d+)\)", tr)]
        calls = re.findall(r"\b(\w*malloc\w*)\s*\(", tr + op)    # any allocator call, any form
        assert sorted(calls) == ["pi_l1_malloc", "pi_l2_malloc"] and len(l2) == 1 and len(l1) == 1, (r, calls, l2, l1)
        # whitespace-tolerant: clang-format wraps long declarations (review: 101/580 BP decls were missed before)
        decl = re.findall(r"static\s+PI_L2\s+(\w+)\s+\w+\s*\[\s*(\d+)\s*\]", tr + op)
        assert len(decl) == len(re.findall(r"static\s+PI_L2\b", tr + op)), (r, "unparsed static PI_L2 declaration")
        const = sum(CSIZE[t] * int(n) for t, n in decl)
        g = onnx.load(os.path.join(FIX[r], "network.onnx")).graph
        cats = {}
        offs = set(re.findall(r"DeeployNetwork_(input_\d+) = \([^)]*\)\(\(char \*\)DeeployNetwork_MEMORYARENA_L2 \+ \d+\)", tr))
        for k, i in enumerate(g.input):
            assert f"input_{k}" in offs, (r, k, i.name, "graph input not an arena offset")
            c = category(i.name)
            cats[c] = cats.get(c, 0) + nbytes(i)
        inputs = sum(cats.values())
        n_params = sum(int(np.prod([d.dim_value for d in i.type.tensor_type.shape.dim])) for i in g.input
                       if category(i.name) == "weights")
        row = {"L2 arena": l2[0], "L1 arena (tile buffer)": l1[0], "constants (static PI_L2)": const,
               "activations & intermediates": l2[0] - inputs, **cats}
        row["optimizer / ECO state"] = 4 * n_params if r == "eco" else 0
        row["L2 total"] = l2[0] + const + row["optimizer / ECO state"]
        row["peak (L2 total + L1 tile buffer)"] = row["L2 total"] + l1[0]
        row["trainable parameters"] = n_params
        out[r] = row
    os.makedirs(os.path.join(E, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(E, "results", "peak_memory.json"), "w"), indent=1)
    keys = ["activations & intermediates", "data window + label", "weights", "gradient accumulators", "other inputs",
            "optimizer / ECO state", "constants (static PI_L2)", "L2 total", "L1 arena (tile buffer)",
            "peak (L2 total + L1 tile buffer)"]
    md = ["| B | " + " | ".join(NAME[r] for r in out) + " |", "|---|" + "--:|" * len(out)]
    for k in keys:
        md.append(f"| {k} | " + " | ".join(f"{out[r].get(k, 0):,}" for r in out) + " |")
    pk = {r: out[r]["peak (L2 total + L1 tile buffer)"] for r in out}
    l2t = {r: out[r]["L2 total"] for r in out}
    md += ["", f"BP / ZO = {pk['bp'] / pk['zo']:.2f}x, BP / QZO-ECO = {pk['bp'] / pk['eco']:.2f}x (peak incl. L1); "
           f"L2 only: {l2t['bp'] / l2t['zo']:.2f}x / {l2t['bp'] / l2t['eco']:.2f}x. GAP9 shared L2 = {L2_SHARED:,} B; "
           f"BP L2 arena {out['bp']['L2 arena']:,} B > {BP_HEAP_RUN:,} B L2 heap left for it on the board (Run)."]
    open(os.path.join(E, "results", "peak_memory.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seg = [("activations & intermediates", "#4c72b0"), ("data window + label", "#dd8452"), ("weights", "#55a868"),
           ("gradient accumulators", "#c44e52"), ("optimizer / ECO state", "#8172b3"),
           ("constants (static PI_L2)", "#937860"), ("other inputs", "#da8bc3")]
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(8.2, 3.6), gridspec_kw={"width_ratios": [3, 1.25]})
    rs = list(out)
    for j, r in enumerate(rs):
        b = 0
        for k, col in seg:
            v = out[r].get(k, 0) / 1e6
            if v:
                ax.bar(j, v, bottom=b, color=col, width=0.55, label=k, edgecolor="white", linewidth=0.5)
                b += v
        ax.text(j, (b + 0.02) if b < 1.3 else 1.61, f"L2 {b:.2f} MB\npeak {out[r]['peak (L2 total + L1 tile buffer)'] / 1e6:.2f} MB",
                ha="center", fontsize=7)
        ax1.bar(j, out[r]["L1 arena (tile buffer)"] / 1e3, color="#bbbbbb", hatch="//", edgecolor="white", width=0.55)
        ax1.text(j, out[r]["L1 arena (tile buffer)"] / 1e3 + 2, f"{out[r]['L1 arena (tile buffer)'] / 1e3:.0f}",
                 ha="center", fontsize=7)
    ax.axhline(L2_SHARED / 1e6, color="k", ls="--", lw=0.8)
    ax.text(1.55, L2_SHARED / 1e6 + 0.02, "GAP9 shared L2 1.57 MB", ha="center", fontsize=7)
    ax.axhline(BP_HEAP_RUN / 1e6, color="#c44e52", ls=":", lw=0.9)
    ax.text(1.55, BP_HEAP_RUN / 1e6 + 0.02, "L2 heap left for BP on the board 1.38 MB", ha="center", fontsize=7,
            color="#c44e52")
    ax.set_ylim(0, 1.9)
    ax.set_xticks(range(len(rs)), [NAME[r] for r in rs], fontsize=7.5)
    ax.set_ylabel("L2 [MB]")
    ax.set_title("L2 (on-chip shared memory)", fontsize=9)
    h, lab = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, lab):
        seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), fontsize=6.3, loc="upper right", bbox_to_anchor=(1.0, 0.70))
    ax.grid(axis="y", alpha=0.3)
    ax1.axhline(L1_CLUSTER / 1e3, color="k", ls="--", lw=0.8)
    ax1.text(1.0, L1_CLUSTER / 1e3 + 2, "cluster L1 128 KiB", ha="center", fontsize=7)
    ax1.set_ylim(0, 150)
    ax1.set_xticks(range(len(rs)), ["BP", "ZO", "QZO-ECO"], fontsize=7.5)
    ax1.set_ylabel("L1 [kB]")
    ax1.set_title("tile buffer (L1 arena)", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.join(E, "plots"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(E, "plots", f"peak_memory.{ext}"), dpi=200)


if __name__ == "__main__":
    main()
