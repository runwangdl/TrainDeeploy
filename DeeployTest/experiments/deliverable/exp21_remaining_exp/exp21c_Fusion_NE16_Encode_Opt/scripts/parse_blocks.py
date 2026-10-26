#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp21c task 2: breakdown of the fused NE16 dual conv (E) and the fused cluster dual conv (D), blocks 1-4.

Inputs (logs/, written by scripts/run_blocks.sh):
  b<i>_E_plain     Runtime (total, reference)            b<i>_E_mb       per-node wall (master core 8)
  b<i>_E_wall      -D NE16DUAL_PROBE_WALL=ON (pipelined)  b<i>_E_wallseq  the same, DEEPLOY_NE16_PIPELINE=0
  b<i>_E_seqplain  sequential, no probe                   b<i>_D_{plain,mb,probe}  fused cluster (probe: DUALCONV2IN_PROBE)
Writes results/E_breakdown.md (+ prints it).

Split of the E prep nodes (probe, worker core 0, barrier to barrier = wall incl. imbalance + barrier):
  wenc node = perturbation (phase 1) + encode (phase 2) + node overhead (fork, closure: mb node - phases)
  bias node = re-perturbation of W and of add (phase 1) + row sums / correction (phase 2) + node overhead
Conv node (master core, running counter): column build (exposed) + NE16 wait + rest (task set-up, dispatch, loop);
sequential mode gives NE16 compute alone (= its wait) and the column build alone. -- QW exp21c
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "logs")
RES = os.path.join(HERE, "..", "results")
REC = re.compile(r"=== Performance Statistics: (\S+) ===\s*\nCycles:\s+(\d+)")
SHAPE = {1: (8, 16, "1x16", 128, 1232), 2: (16, 16, "1x8", 128, 308), 3: (16, 32, "7x1", 112, 40),
         4: (32, 32, "7x1", 224, 10)}  # Ci, Co, kernel, K, output pixels per pass


def rd(i, tag):
    p = os.path.join(LOGS, f"b{i}_{tag}.log")
    if not os.path.exists(p):
        return None
    return open(p, errors = "ignore").read()


def recs(txt):
    d = {}
    for n, c in REC.findall(txt or ""):
        d.setdefault(n, int(c))  # first occurrence
    return d


def rt(txt):
    m = re.search(r"^Runtime:\s+(\d+)", txt or "", re.M)
    return int(m.group(1)) if m else None


def err(txt):
    m = re.search(r"Errors: (\d+) out of (\d+)", txt or "")
    return f"{m.group(1)}/{m.group(2)}" if m else "?"


def f(v):
    return "—" if v is None else f"{v:,}"


def main():
    rows, D, chk = {}, {}, []
    for i in (1, 2, 3, 4):
        ep, em, ew, es, ews = (rd(i, t) for t in ("E_plain", "E_mb", "E_wall", "E_seqplain", "E_wallseq"))
        dp, dm, dpr = (rd(i, t) for t in ("D_plain", "D_mb", "D_probe"))
        if not (rt(ep) and rt(ew) and em and "Runtime:" in em):
            continue
        mb, w, ws = recs(em), recs(ew), recs(ews)
        node = lambda s: mb.get(f"b{i}_{s}")
        r = dict(total = rt(ep), wall_total = rt(ew), seq_total = rt(es), wallseq_total = rt(ews))
        r["in_tr"] = (node("xp_hwc") or 0) + (node("xm_hwc") or 0)
        r["out_tr"] = node("out_nchw")
        r["wenc"], r["bias"], r["conv"] = node("dual_wenc"), node("dual_bias"), node("ne16_dualconv")
        r["pert"], r["enc"] = w.get("X21C_wenc_perturb_wall"), w.get("X21C_wenc_encode_wall")
        r["wenc_ovh"] = r["wenc"] - r["pert"] - r["enc"]
        r["b_pert"], r["b_rows"] = w.get("X21C_bias_perturb_wall"), w.get("X21C_bias_rowsum_wall")
        r["b_wrep"] = w.get("X21C_bias_wrepert_work")
        r["bias_ovh"] = r["bias"] - r["b_pert"] - r["b_rows"]
        r["col"], r["wait"], r["rest"], r["ctot"] = (w.get(f"N16PROBE_conv_{k}") for k in ("colbuild", "ne16wait", "rest", "total"))
        r["jobs"] = w.get("N16PROBE_conv_jobs")
        r["col_seq"], r["wait_seq"], r["rest_seq"], r["ctot_seq"] = (ws.get(f"N16PROBE_conv_{k}") for k in ("colbuild", "ne16wait", "rest", "total"))
        r["conv_ovh"] = r["conv"] - r["ctot"]
        r["nodes"] = r["in_tr"] + r["wenc"] + r["bias"] + r["conv"] + r["out_tr"]
        r["outside"] = r["total"] - r["nodes"]
        r["err"] = " ".join(err(t) for t in (ep, em, ew, es, ews))
        rows[i] = r
        if dp and dm and dpr and rt(dp) and rt(dpr):
            dmb, dprr = recs(dm), recs(dpr)
            p1 = max(v for k, v in dprr.items() if k.startswith("DC2PROBE_p1_core"))
            p2 = max(v for k, v in dprr.items() if k.startswith("DC2PROBE_p2_core"))
            dn = dmb.get(f"dualconv_blocks{i}")
            D[i] = dict(total = rt(dp), probe_total = rt(dpr), node = dn, p1 = p1, p2 = p2, ovh = dn - p1 - p2,
                        in_tr = dmb.get(f"dualconv_blocks{i}_input_0_transpose", 0) + dmb.get(f"dualconv_blocks{i}_input_3_transpose", 0),
                        out_tr = dmb.get(f"dualconv_blocks{i}_output_0_pre_transpose"), err = " ".join(err(t) for t in (dp, dm, dpr)))
            D[i]["outside"] = D[i]["total"] - D[i]["node"] - D[i]["in_tr"] - D[i]["out_tr"]
    L = []
    L.append("# exp21c task 2 -- fused NE16 (E) breakdown per block, both passes, GAP9 GVSoC cycles\n")
    L.append("Settings: exp11's (--l1 86000 --l2 1500000, GAP9_NETWORK_O3=ON, 8 cores), fixtures "
             "`QZO_DualConv2in_NE16P_B<i>` / `QZO_DualConv2in_B<i>` (copies in `fixtures/`). Node times = "
             "`--profileMicrobenchmark` (master core 8). Phase split = `-D NE16DUAL_PROBE_WALL=ON` (exp21c). "
             "All runs exact (errors listed per run).\n")
    L.append("## E: one row per block (node-level columns sum to the fixture total)\n")
    hdr = ["block", "perturbation (wenc ph.1)", "weight encode (wenc ph.2)", "wenc node overhead",
           "bias: re-perturb W + add (ph.1)", "bias: row sums + correction (ph.2)", "bias node overhead",
           "conv: column build (exposed)", "conv: NE16 wait", "conv: rest (task set-up, dispatch)", "conv node overhead",
           "input transposes", "output transpose", "outside nodes", "**total (Runtime)**"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" + "--:|" * (len(hdr) - 1))
    for i, r in rows.items():
        cells = [r["pert"], r["enc"], r["wenc_ovh"], r["b_pert"], r["b_rows"], r["bias_ovh"], r["col"], r["wait"],
                 r["rest"], r["conv_ovh"], r["in_tr"], r["out_tr"], r["outside"], r["total"]]
        L.append(f"| {i} | " + " | ".join(f(c) for c in cells) + " |")
    L.append("")
    L.append("Shares of the fixture total, and per weight element (2 x Co x K elements per step, both signs):\n")
    L.append("| block | weights / pass | perturb % | encode % | bias prep % | prep total (wenc+bias nodes) | prep % | "
             "conv node | conv % | transposes % | perturb cyc/elem (core) | encode cyc/elem (core) |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for i, r in rows.items():
        ci, co, k, K, px = SHAPE[i]
        t = r["total"]
        prep = r["wenc"] + r["bias"]
        n_el = 2 * co * K
        L.append(f"| {i} | {co * K:,} | {r['pert'] / t * 100:.1f} | {r['enc'] / t * 100:.1f} | {r['bias'] / t * 100:.1f} | "
                 f"{prep:,} | {prep / t * 100:.1f} | {r['conv']:,} | {r['conv'] / t * 100:.1f} | "
                 f"{(r['in_tr'] + r['out_tr']) / t * 100:.1f} | {r['pert'] / (n_el / 8):.1f} | {r['enc'] / (n_el / 8):.1f} |")
    L.append("")
    L.append("Conv node, pipelined vs sequential (probe, master core):\n")
    L.append("| block | jobs | pipe: col build | pipe: NE16 wait | pipe: rest | pipe: total | seq: col build | "
             "seq: NE16 compute (wait) | seq: rest | seq: total | hidden by pipelining |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for i, r in rows.items():
        hid = (r["ctot_seq"] - r["ctot"]) if r["ctot_seq"] else None
        L.append(f"| {i} | {f(r['jobs'])} | {f(r['col'])} | {f(r['wait'])} | {f(r['rest'])} | {f(r['ctot'])} | "
                 f"{f(r['col_seq'])} | {f(r['wait_seq'])} | {f(r['rest_seq'])} | {f(r['ctot_seq'])} | {f(hid)} |")
    L.append("")
    L.append("Probe check (Runtime of the probe build vs the unprobed build of the same mode):\n")
    L.append("| block | E plain | E probe | delta | E seq plain | E seq probe | delta | errors (plain mb wall seq wallseq) |")
    L.append("|---|--:|--:|--:|--:|--:|--:|---|")
    for i, r in rows.items():
        d1 = (r["wall_total"] - r["total"]) / r["total"] * 100
        d2 = ((r["wallseq_total"] - r["seq_total"]) / r["seq_total"] * 100) if (r["seq_total"] and r["wallseq_total"]) else float("nan")
        L.append(f"| {i} | {f(r['total'])} | {f(r['wall_total'])} | {d1:+.2f} % | {f(r['seq_total'])} | {f(r['wallseq_total'])} | "
                 f"{d2:+.2f} % | {r['err']} |")
    L.append("")
    L.append("Bias re-perturbation of W alone (core-0 work, `X21C_bias_wrepert_work`) vs the wenc perturbation (same "
             "routine, same data): " + ", ".join(f"b{i} {r['b_wrep']:,} vs {r['pert']:,}" for i, r in rows.items()) + "\n")
    if D:
        L.append("## D (fused cluster DualConv2In), same settings, for comparison\n")
        L.append("| block | perturbation (phase 1, max core) | conv (phase 2, max core) | node overhead | input transposes | "
                 "output transpose | outside | **total (Runtime)** | probe Runtime | errors (plain mb probe) |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|---|")
        for i, d in D.items():
            L.append(f"| {i} | {d['p1']:,} | {d['p2']:,} | {d['ovh']:,} | {d['in_tr']:,} | {d['out_tr']:,} | {d['outside']:,} | "
                     f"{d['total']:,} | {d['probe_total']:,} | {d['err']} |")
        L.append("")
        L.append("## Node level (what the network sees: no NCHW boundary transposes)\n")
        L.append("| block | E: wenc + bias + conv | of which prep | D: DualConv2In node | of which perturbation | E/D |")
        L.append("|---|--:|--:|--:|--:|--:|")
        for i in rows:
            if i in D:
                r, d = rows[i], D[i]
                e = r["wenc"] + r["bias"] + r["conv"]
                L.append(f"| {i} | {e:,} | {r['wenc'] + r['bias']:,} | {d['node']:,} | {d['p1']:,} | {e / d['node']:.2f} |")
        L.append("")
    os.makedirs(RES, exist_ok = True)
    open(os.path.join(RES, "E_breakdown.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    import json
    json.dump(dict(E = rows, D = D), open(os.path.join(RES, "E_breakdown.json"), "w"), indent = 1)


if __name__ == "__main__":
    main()
