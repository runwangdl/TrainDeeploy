#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp21c impl: tables of the per-block fixtures and the whole-network runs under the optimisation configs.

Inputs (impl/logs, written by run_blocks.sh / run_network.sh):
  blk_<lane>_<cfg>_<plain|wall>.log    lane = B0 | off0_B<i> | B<i>, cfg = base | B | BH | BHS
  net_<cfg>_<kind>_ne16_<blocks>.log   kind = plain | prof | multi | multi2
Old checkpoint logs (../logs) are used for the exp21c base numbers of the 128-offset fixtures and the base network
profile (net_ne16_prof.log = the same code as cfg base).
Writes impl/results/{blocks.md, network.md, bitexact.txt}.   python3 parse_impl.py   -- QW exp21c
"""
import collections
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL = os.path.join(HERE, "..")
LOGS = os.path.join(IMPL, "logs")
OLD = os.path.join(IMPL, "..", "logs")
RES = os.path.join(IMPL, "results")
REC = re.compile(r"=== Performance Statistics: (\S+) ===\s*\nCycles:\s+(\d+)")
CFGS = ("base", "B", "BH", "BHS")


def rd(p):
    return open(p, errors = "ignore").read() if os.path.exists(p) else None


def recs(t):
    d = {}
    for n, c in REC.findall(t or ""):
        d.setdefault(n, int(c))
    return d


def rt(t):
    m = re.search(r"^Runtime:\s+(\d+)", t or "", re.M)
    return int(m.group(1)) if m else None


def err(t):
    m = re.search(r"Errors: (\d+) out of (\d+)", t or "")
    return f"{m.group(1)}/{m.group(2)}" if m else "?"


def f(v):
    return "—" if v is None else f"{v:,}"


def blocks(out, bx):
    out.append("# Per-block fixtures (GAP9 GVSoC, --l1 86000 --l2 1500000, GAP9_NETWORK_O3, 8 cores)\n")
    out.append("Runtime = fixture total (plain build). cfg: base = exp21 checkpoint code (all opts off), B, BH = B+H, "
               "BHS = B+H+S. In a fixture the graph runs ONCE, so H only adds its cache fill (miss path); its saving "
               "shows in the network (network.md).\n")
    out.append("## Totals and bit-exactness\n")
    out.append("| fixture | input_offset | " + " | ".join(f"{c} Runtime" for c in CFGS) + " | BHS vs base | Errors (all runs) |")
    out.append("|---|--:|" + "--:|" * len(CFGS) + "--:|---|")
    lanes = ["B0"] + [f"off0_B{i}" for i in (1, 2, 3, 4)] + [f"B{i}" for i in (1, 2, 3, 4)]
    for lane in lanes:
        vals, errs = {}, []
        for c in CFGS:
            t = rd(os.path.join(LOGS, f"blk_{lane}_{c}_plain.log"))
            vals[c] = rt(t)
            if t:
                errs.append(f"{c}:{err(t)}")
            tw = rd(os.path.join(LOGS, f"blk_{lane}_{c}_wall.log"))
            if tw:
                errs.append(f"{c}-wall:{err(tw)}")
        if lane.startswith("B") and lane != "B0" and vals["base"] is None:  # exp21c checkpoint run of the same code
            t = rd(os.path.join(OLD, f"b{lane[1]}_E_plain.log"))
            vals["base"] = rt(t)
            if t:
                errs.append(f"base(exp21c log):{err(t)}")
        sp = f"{vals['base'] / vals['BHS']:.2f}x" if vals["base"] and vals["BHS"] else "—"
        off = 0 if lane.startswith("off0") else 128
        out.append(f"| {lane} | {off} | " + " | ".join(f(vals[c]) for c in CFGS) + f" | {sp} | {', '.join(errs)} |")
        for e in errs:
            bx.append(f"block fixture {lane} {e}")
    out.append("")
    out.append("## Phase split (probe build -D NE16DUAL_PROBE_WALL=ON, worker core 0 wall barrier to barrier)\n")
    out.append("| fixture | cfg | wenc: perturb | wenc: **encode** | bias: phase 1 (W re-perturb + add) | of which W "
               "re-perturb | bias: row sums | conv node total | probe Runtime |")
    out.append("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for lane in ["B0"] + [f"off0_B{i}" for i in (1, 2, 3, 4)]:
        for c in CFGS:
            t = rd(os.path.join(LOGS, f"blk_{lane}_{c}_wall.log"))
            if not t:
                continue
            r = recs(t)
            out.append(f"| {lane} | {c} | {f(r.get('X21C_wenc_perturb_wall'))} | {f(r.get('X21C_wenc_encode_wall'))} | "
                       f"{f(r.get('X21C_bias_perturb_wall'))} | {f(r.get('X21C_bias_wrepert_work'))} | "
                       f"{f(r.get('X21C_bias_rowsum_wall'))} | {f(r.get('N16PROBE_conv_total'))} | {f(rt(t))} |")
    out.append("")


# ---------------------------------------------------------------------------------------------- network
PAT = re.compile(r"^\[(\S+?)_L2\]\[SB\]\[\d+ ops\]\[Tile \d+\] (Pre-Kernel|Kernel     |Post-Kernel):\s*(\d+) cycles", re.M)
PREP = re.compile(r"wrappedInnerForwardImpl(?:_(\d))?Conv_(ne16wenc|ne16bias)__dual")


def bench(t):
    m = re.search(r"BENCH train_cycles_hi=(\d+) train_cycles_lo=(\d+) opt_cycles_hi=(\d+) opt_cycles_lo=(\d+)", t or "")
    if not m:
        return None, None
    a = [int(x) for x in m.groups()]
    return a[0] * 2**32 + a[1], a[2] * 2**32 + a[3]


def prof_prep(t):
    """per prep node: list of per-run wall (pre+kernel+post, one tile) in execution order (run k = window k)."""
    runs = collections.defaultdict(list)
    cur = {}
    for n, k, c in PAT.findall(t or ""):
        m = PREP.match(n)
        if not m:
            continue
        key = (int(m.group(1) or 0), m.group(2))
        cur[key] = cur.get(key, 0) + int(c)
        if k == "Post-Kernel":
            runs[key].append(cur.pop(key))
    return runs


def prof_total(t):
    tot = collections.OrderedDict()
    for n, _k, c in PAT.findall(t or ""):
        tot[n] = tot.get(n, 0) + int(c)
    return tot


def losses(t):
    return re.findall(r"(lp_bits=0x[0-9a-f]+|lm_bits=0x[0-9a-f]+)", t or "")


def wsums(t):
    return re.findall(r"\[WSUM s=(\S+)\] (0x[0-9a-f]+)", t or "")


def wdump(t):
    return re.findall(r"^\[WDUMP s=\d+ wi=\d+ n=\d+\].*$", t or "", re.M)


def network(out, bx):
    out.append("# Whole network, fused QZO+ECO (board settings --l1 114000 --l2 1340000, frozen BN, GAP9_NETWORK_O3, "
               "ECO SR, lr 3e-6)\n")
    out.append("## One update step (n_accum 4): training BENCH, update graph, bit-exactness\n")
    old = rd(os.path.join(OLD, "net_ne16_plain.log"))
    rows = [("base (exp21c checkpoint log)", "0,1,2", old)]
    for c in CFGS:
        if c == "base":  # two runs of the same build: run-to-run codegen noise (tiler random-max)
            for r in ("run1", "run2"):
                rows.append((f"base {r}", "0,1,2", rd(os.path.join(LOGS, f"net_base_plain_ne16_012_{r}.log"))))
            continue
        rows.append((c, "0,1,2", rd(os.path.join(LOGS, f"net_{c}_plain_ne16_012.log"))))
    for bl in ("0123", "01234"):
        for c in CFGS:
            t = rd(os.path.join(LOGS, f"net_{c}_plain_ne16_{bl}.log"))
            if t and bench(t)[0]:
                rows.append((c, ",".join(bl), t))
    bb = [bench(rd(os.path.join(LOGS, f"net_base_plain_ne16_012_{r}.log")))[0] for r in ("run1", "run2")]
    bb = [x for x in bb if x]
    b0 = sum(bb) / len(bb)  # mean of the same-tree base runs
    out.append(f"Base reference = mean of the same-tree base runs {[f(x) for x in bb]} = {b0:,.0f}.\n")
    out.append("| run | NE16 blocks | train cycles / update step | vs unfused 13,288,673 | vs base | Δ vs base | opt "
               "(update graph) | Errors | WSUM s=0 |")
    out.append("|---|---|--:|--:|--:|--:|--:|---|---|")
    ref_l, ref_w = losses(old), wsums(old)
    for name, bl, t in rows:
        if not t:
            continue
        tr, op = bench(t)
        ws = dict(wsums(t)).get("0")
        same = "identical losses+WSUM" if (losses(t) == ref_l and wsums(t) == ref_w) else "DIFFERENT"
        bx.append(f"network 1 step {name} ne16={bl}: Errors {err(t)}, WSUM s=0 {ws}, 8 losses+WSUM vs exp21c "
                  f"checkpoint log: {same}")
        out.append(f"| {name} | {bl} | {f(tr)} | {13288673 / tr:.3f}x | {b0 / tr:.3f}x | {f(round(tr - b0))} | {f(op)} | "
                   f"{err(t)} | {ws} ({same}) |" if tr else f"| {name} | {bl} | — | | | | | {err(t)} | |")
    out.append("")
    # prep node cycles per window
    out.append("## NE16 prep nodes per window (--profileTiling, node wall = pre + kernel + post, cluster master)\n")
    out.append("Window 1 of the update step computes (cache miss) and fills the cache; windows 2-4 hit.\n")
    out.append("| run | block | node | window 1 | window 2 | window 3 | window 4 | sum (update step) |")
    out.append("|---|--:|---|--:|--:|--:|--:|--:|")
    profs = [("base (exp21c log)", rd(os.path.join(OLD, "net_ne16_prof.log")))]
    for c in CFGS:
        for bl in ("012", "0123", "01234"):
            t = rd(os.path.join(LOGS, f"net_{c}_prof_ne16_{bl}.log"))
            if t:
                profs.append((f"{c} ne16={bl}", t))
    sums = {}
    for name, t in profs:
        if not t:
            continue
        pr = prof_prep(t)
        tot = 0
        for key in sorted(pr):
            v = pr[key]
            tot += sum(v)
            out.append(f"| {name} | {key[0]} | {key[1]} | " + " | ".join(f(x) for x in (v + [None] * 4)[:4]) +
                       f" | {f(sum(v))} |")
        sums[name] = (tot, sum(prof_total(t).values()))
        out.append(f"| {name} | all | **prep total** | | | | | **{f(tot)}** |")
    out.append("")
    out.append("| profile run | prep total / update step | all nodes (training + update graph) |")
    out.append("|---|--:|--:|")
    for k, (a, b) in sums.items():
        out.append(f"| {k} | {f(a)} | {f(b)} |")
    out.append("")
    # multi-step
    out.append("## Multi-step bit-exactness (losses of every window, WSUM, weight dumps after every update)\n")
    for kind, desc in (("multi", "3 update steps x n_accum 4"), ("multi2", "2 update steps x n_accum 2")):
        for bl in ("012", "0123"):
            ref = rd(os.path.join(LOGS, f"net_base_{kind}_ne16_{bl}.log"))
            for c in ("B", "BH", "BHS"):
                t = rd(os.path.join(LOGS, f"net_{c}_{kind}_ne16_{bl}.log"))
                if not (t and ref):
                    continue
                if bl == "0123":  # compare against the 0,1,2 base: NE16 and cluster are bit-identical
                    pass
                lr, lt, wr, wt, dr, dt = losses(ref), losses(t), wsums(ref), wsums(t), wdump(ref), wdump(t)
                line = (f"{desc}, ne16={bl}: {c} vs base: losses {len(lt)}/{len(lr)} "
                        f"{'identical' if lt == lr and lr else 'DIFFERENT'}; WSUM {wt} "
                        f"{'identical' if wt == wr and wr else 'DIFFERENT'}; WDUMP lines {len(dt)}/{len(dr)} "
                        f"{'identical' if dt == dr and dr else 'DIFFERENT'}; Errors {err(t)} (base {err(ref)}); "
                        f"train cycles {f(bench(t)[0])} vs base {f(bench(ref)[0])}")
                out.append("- " + line)
                bx.append(line)
        # 0123 against 012 base
        ref = rd(os.path.join(LOGS, f"net_base_{kind}_ne16_012.log"))
        t = rd(os.path.join(LOGS, f"net_BHS_{kind}_ne16_0123.log"))
        if t and ref:
            ok = losses(t) == losses(ref) and wsums(t) == wsums(ref) and wdump(t) == wdump(ref)
            line = f"{desc}: BHS ne16=0,1,2,3 vs base ne16=0,1,2: losses+WSUM+WDUMP {'identical' if ok else 'DIFFERENT'}"
            out.append("- " + line)
            bx.append(line)
    out.append("")


def main():
    os.makedirs(RES, exist_ok = True)
    bx = []
    ob, on = [], []
    blocks(ob, bx)
    network(on, bx)
    open(os.path.join(RES, "blocks.md"), "w").write("\n".join(ob) + "\n")
    open(os.path.join(RES, "network.md"), "w").write("\n".join(on) + "\n")
    open(os.path.join(RES, "bitexact.txt"), "w").write("\n".join(bx) + "\n")
    print("\n".join(ob + on))
    print("\n".join(bx))


if __name__ == "__main__":
    main()
