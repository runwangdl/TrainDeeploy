#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp21c task 1: savings estimates of the encode / prep optimisation ideas, from MEASURED cycles (no kernel changed).

Inputs: results/E_breakdown.json (fixture phase split, scripts/parse_blocks.py) and the whole-network profiles
(logs/net_ne16_prof.log, logs/net_eco_plain.log, logs/net_ne16_plain.log). n_accum = 4.

Ideas (cycles per window = one of the 4 training-graph runs of an update step):
  H   hoist: the whole NE16 prep (wenc + bias nodes) once per UPDATE instead of once per window (z and W are constant
      over the n_accum windows: deeploymezotest.c:666/695) -> saves (1 - 1/4) of it.
  H+  hoist every other per-window perturbation too (DualConv2In phase 1 of blocks 3-4, fp32 BN / FC perturbation).
  B   bias node without the W re-perturbation and row sums when input_offset == 0 (network blocks 1-2; the term is
      multiplied by 0), or with the row sums produced by the wenc phase 1 (block 0, input_offset 128).
  S   SWAR / bit-matrix-transpose encode: ENC_SWAR cycles per (element, sign) per core instead of the measured ~42.
Estimates are first order (they keep each node's fork / closure overhead). -- QW exp21c
  python3 estimate_savings.py  -> results/savings.md
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_network import parse, bench  # noqa: E402

RES = os.path.join(HERE, "..", "results")
LOGS = os.path.join(HERE, "..", "logs")
N_ACC = 4
ENC_SWAR = 6.0  # assumed cycles per (element, sign) per core of a word-wise 8x8 bit transpose (see README sec. 1.4)
SHAPE = {0: (1, 8, 4), 1: (8, 16, 128), 2: (16, 16, 128), 3: (16, 32, 112), 4: (32, 32, 224)}  # Ci, Co, K


def main():
    J = json.load(open(os.path.join(RES, "E_breakdown.json")))
    E = {int(k): v for k, v in J["E"].items()}
    D = {int(k): v for k, v in J["D"].items()}
    net = parse(os.path.join(LOGS, "net_ne16_prof.log"))
    pw = lambda pref: sum(c for n, c in net.items() if n.startswith(pref)) / N_ACC
    wenc = {0: pw("wrappedInnerForwardImplConv_ne16wenc"), 1: pw("wrappedInnerForwardImpl_1Conv_ne16wenc"),
            2: pw("wrappedInnerForwardImpl_2Conv_ne16wenc")}
    bias = {0: pw("wrappedInnerForwardImplConv_ne16bias"), 1: pw("wrappedInnerForwardImpl_1Conv_ne16bias"),
            2: pw("wrappedInnerForwardImpl_2Conv_ne16bias")}
    bnp = pw("pert_blocks")
    fcp = pw("pert_fc")
    d34 = D[3]["p1"] + D[4]["p1"]
    step_ne16 = bench(os.path.join(LOGS, "net_ne16_plain.log"))[0]
    step_eco = bench(os.path.join(LOGS, "net_eco_plain.log"))[0]
    win = step_ne16 / N_ACC
    L = ["# exp21c task 1 -- savings estimates (measured inputs, first-order model)\n"]
    L.append(f"Fused NE16 network (board recipe), per window: BENCH/4 = {win:,.0f} cycles "
             f"(`logs/net_ne16_plain.log`, {step_ne16:,} per update step). Unfused: {step_eco:,}.\n")
    L.append("Measured per-window prep in the network (`logs/net_ne16_prof.log`, wall clock / 4):\n")
    L.append("| block | wenc node (perturb + encode) | bias node (re-perturb + add + row sums) | sum |")
    L.append("|---|--:|--:|--:|")
    for b in (0, 1, 2):
        L.append(f"| {b} | {wenc[b]:,.0f} | {bias[b]:,.0f} | {wenc[b] + bias[b]:,.0f} |")
    tp = sum(wenc.values()) + sum(bias.values())
    L.append(f"| 0-2 | {sum(wenc.values()):,.0f} | {sum(bias.values()):,.0f} | {tp:,.0f} ({tp / win * 100:.1f} % of the window) |")
    L.append(f"\nOther per-window perturbation: DualConv2In phase 1 blocks 3+4 {d34:,} (fixture probe), fp32 BN {bnp:,.0f}, "
             f"fp32 FC {fcp:,.0f}.\n")

    # per-idea savings per window (blocks 0-2 NE16 as in the network)
    # phase fractions of the network nodes from the fixture split (blocks 1, 2); block 0 is tiny and taken whole
    def split(b):
        e = E[b]
        wn, bn = e["wenc"], e["bias"]
        return dict(pert = e["pert"] / wn, enc = e["enc"] / wn, wrep = (e["b_wrep"]) / bn, rows = e["b_rows"] / bn)

    sv = {}
    sv["H"] = (1 - 1 / N_ACC) * tp
    sv["H+"] = sv["H"] + (1 - 1 / N_ACC) * (d34 + bnp + fcp)
    sB = sum(bias[b] * (split(b)["wrep"] + split(b)["rows"]) for b in (1, 2))
    sv["B"] = sB
    sS = 0.0
    for b in (1, 2):
        ci, co, K = SHAPE[b]
        enc_now = wenc[b] * split(b)["enc"]
        enc_new = ENC_SWAR * 2 * co * K / 8
        sS += enc_now - enc_new
    sv["S"] = sS
    rest = tp - sB - sS
    sv["B+S"] = sB + sS
    sv["H+B+S"] = sB + sS + (1 - 1 / N_ACC) * rest
    sv["H+ +B+S"] = sv["H+B+S"] + (1 - 1 / N_ACC) * (d34 + bnp + fcp)
    L.append("## Savings in the network, per window and per update step\n")
    L.append("| idea | per window | % of window | per update step (x4) | new step | whole-step speedup vs unfused (now "
             f"{step_eco / step_ne16:.3f}x) |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for k, desc in (("B", "B: bias without W re-perturb / row sums (input_offset 0)"),
                    ("S", f"S: SWAR encode ({ENC_SWAR:.0f} cyc/elem/sign/core)"),
                    ("B+S", "B + S (still per window)"),
                    ("H", "H: hoist NE16 prep to once per update"),
                    ("H+B+S", "H + B + S"),
                    ("H+", "H+: hoist all per-window perturbation"),
                    ("H+ +B+S", "H+ + B + S")):
        s = sv[k]
        new = step_ne16 - N_ACC * s
        L.append(f"| {desc} | {s:,.0f} | {s / win * 100:.1f} | {N_ACC * s:,.0f} | {new:,.0f} | {step_eco / new:.3f}x |")
    L.append("")

    # per-block fixture view: would NE16 win blocks 3-4 with the ideas?
    L.append("## Per block (fixture node level, no boundary transposes): E vs D after the ideas\n")
    L.append("E' = conv node + prep', D' = DualConv2In node with its phase 1 hoisted (H) = node - 3/4 phase 1. "
             "prep' = perturb + encode(S) + wenc overhead + bias (B: only the add perturbation + overhead).\n")
    L.append("| block | E now | D now | E with H | E with B+S | E with H+B+S | D with H | winner now | winner with H+B+S vs D with H |")
    L.append("|---|--:|--:|--:|--:|--:|--:|---|---|")
    for b in (1, 2, 3, 4):
        e, d = E[b], D[b]
        ci, co, K = SHAPE[b]
        prep = e["wenc"] + e["bias"]
        enc_s = ENC_SWAR * 2 * co * K / 8
        bias_b = e["bias"] - e["b_wrep"] - e["b_rows"]
        prep_bs = e["pert"] + enc_s + e["wenc_ovh"] + bias_b
        e_now = prep + e["conv"]
        e_h = prep / N_ACC + e["conv"]
        e_bs = prep_bs + e["conv"]
        e_hbs = prep_bs / N_ACC + e["conv"]
        d_now = d["node"]
        d_h = d["node"] - (1 - 1 / N_ACC) * d["p1"]
        L.append(f"| {b} | {e_now:,.0f} | {d_now:,.0f} | {e_h:,.0f} | {e_bs:,.0f} | {e_hbs:,.0f} | {d_h:,.0f} | "
                 f"{'NE16' if e_now < d_now else 'cluster'} | {'NE16' if e_hbs < d_h else 'cluster'} ({d_h / e_hbs:.2f}x) |")
    L.append("")
    out = "\n".join(L)
    open(os.path.join(RES, "savings.md"), "w").write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
