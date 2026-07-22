# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regret-Aware Greedy Tensor Promotion (offline analysis).
================================================================================

Re-selects the L2-resident (promoted) tensor set for a training graph using the
regret-aware greedy of Alg. 1, and reports the OFF-CHIP (L2<->L3) TRAFFIC SAVED.

The three inputs are static graph properties, priced exactly per the paper:
  |t|          bytes of tensor t                       (shape x dtype width)
  n_acc(t)     reads + writes of t                     (consumers + producer)
  L(t)         live step interval [def, last_use]      (topological order)

  Q(t) = n_acc(t) * |t|                    off-chip bytes saved by promoting t
  lambda_tau = (1/B) * sum_{t' live at tau} |t'|       normalised demand at step tau
  C(t) = |t| * sum_{tau in L(t)} lambda_tau            capacity charge (contended)
  S(t|P) = Q(t)/C(t) * (1 + gamma * p_block(t|P))      regret-boosted score

Feasibility is the REAL 2D allocator: MiniMalloc(P) <= B iff the minimalloc
binary packs P's (lower,upper,size) rectangles within capacity B.

Budget B = C_L2 - 2*C_L1, with C_L1 ~ 8% of L2 reserved for the double-buffered
tile pair the tiler stages (Sec. dma_sched).
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import subprocess
import tempfile

import onnx

MINIMALLOC_BIN = os.path.join(
    os.environ.get("MINIMALLOC_INSTALL_DIR", "/app/install/minimalloc"), "minimalloc")

# onnx elem_type -> byte width
_DT = {1: 4, 2: 1, 3: 1, 4: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}
# pointer-alias ops: promoting one side of the alias crashes at runtime (see pass _SKIP_OPS)
_SKIP_OPS = {"Reshape", "Squeeze", "Unsqueeze", "Flatten", "Identity",
             "BatchNormInternal", "BatchNormalizationGrad",
             "LayerNormalization", "LayerNormalizationGrad"}


# --------------------------------------------------------------- MiniMalloc

class MiniMalloc:
    """Real 2D allocator via the minimalloc binary. fits(blocks,B) -> bool.

    Memoised on the canonical (sorted (lower,upper,size)) key + capacity, and
    pre-filtered by the sweep-line lower bound (peak_lb > B => infeasible, no
    binary call needed) since minimalloc's packed peak >= sweep-line peak.
    """

    def __init__(self):
        self._cache = {}
        self._calls = 0

    @staticmethod
    def sweepline_peak(blocks):
        ev = []
        for _id, lo, hi, sz in blocks:
            ev.append((lo, sz)); ev.append((hi + 1, -sz))
        ev.sort()
        peak = live = 0
        for _, d in ev:
            live += d
            peak = max(peak, live)
        return peak

    def fits(self, blocks, capacity):
        if not blocks:
            return True
        if self.sweepline_peak(blocks) > capacity:      # exact lower bound -> prune
            return False
        key = (tuple(sorted((lo, hi, sz) for _i, lo, hi, sz in blocks)), capacity)
        if key in self._cache:
            return self._cache[key]
        self._calls += 1
        d = tempfile.mkdtemp()
        inp, out = os.path.join(d, "i.csv"), os.path.join(d, "o.csv")
        with open(inp, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")      # LF only -- minimalloc rejects CRLF
            w.writerow(["id", "lower", "upper", "size"])
            for i, (bid, lo, hi, sz) in enumerate(blocks):
                w.writerow([f"b{i}", lo, hi, sz])
        r = subprocess.run([MINIMALLOC_BIN, f"--capacity={capacity}",
                            f"--input={inp}", f"--output={out}"],
                           capture_output=True, text=True)
        ok = (r.returncode == 0)
        self._cache[key] = ok
        return ok


# --------------------------------------------------------------- graph model

class Tensor:
    __slots__ = ("name", "size", "n_acc", "lo", "hi", "is_const")

    def __init__(self, name, size, n_acc, lo, hi, is_const):
        self.name, self.size, self.n_acc = name, size, n_acc
        self.lo, self.hi, self.is_const = lo, hi, is_const

    @property
    def Q(self):                       # off-chip bytes saved if promoted
        return self.n_acc * self.size

    def block(self):
        return (self.name, self.lo, self.hi, self.size)


def load_tensors(onnx_path, min_bytes=256):
    """Static extraction of promotable tensors with size, n_acc, lifetime."""
    g = onnx.load(onnx_path).graph
    shape, dtype = {}, {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        tt = vi.type.tensor_type
        if tt.shape.dim:
            shape[vi.name] = [d.dim_value if d.HasField("dim_value") else 1 for d in tt.shape.dim]
        dtype[vi.name] = tt.elem_type
    init_names = set()
    for it in g.initializer:
        shape[it.name] = list(it.dims); dtype[it.name] = it.data_type; init_names.add(it.name)

    def nbytes(t):
        s = shape.get(t)
        if not s:
            return 0
        p = 1
        for d in s:
            p *= d if d > 0 else 1
        return p * _DT.get(dtype.get(t, 1), 4)

    producer_step, consumer_steps = {}, collections.defaultdict(list)
    skip = set()
    for i, nd in enumerate(g.node):
        for o in nd.output:
            if o:
                producer_step[o] = i
        for x in nd.input:
            if x:
                consumer_steps[x].append(i)
        if nd.op_type in _SKIP_OPS:                     # alias/multi-output: not promotable
            for t in list(nd.input) + list(nd.output):
                if t:
                    skip.add(t)

    N = len(g.node)
    tensors = []
    all_names = set(producer_step) | set(consumer_steps) | init_names
    for name in all_names:
        cons = consumer_steps.get(name, [])
        if not cons:                                    # dead / graph output only -> no reuse traffic
            continue
        if name in skip:
            continue
        size = nbytes(name)
        if size < min_bytes:
            continue
        is_const = name in init_names or name not in producer_step
        lo = 0 if is_const else producer_step[name]     # consts alive from start
        hi = max(cons)
        if hi < lo:
            hi = lo
        writes = 0 if is_const else 1
        n_acc = len(cons) + writes
        tensors.append(Tensor(name, size, n_acc, lo, hi, is_const))
    return tensors, N


def lambda_and_C(tensors, budget, n_steps):
    """Per-step normalised demand lambda_tau and capacity charge C(t)."""
    demand = [0] * (n_steps + 2)
    for t in tensors:
        for tau in range(t.lo, t.hi + 1):
            demand[tau] += t.size
    lam = [demand[tau] / budget for tau in range(n_steps + 2)]
    C = {}
    for t in tensors:
        C[t.name] = t.size * sum(lam[tau] for tau in range(t.lo, t.hi + 1))
    return C


# --------------------------------------------------------------- Algorithm 1

def regret_greedy(tensors, budget, mm, gamma=1.0, k=4, verbose=False):
    """Regret-Aware Greedy Tensor Promotion (Alg. 1). Returns promoted list."""
    n_steps = max((t.hi for t in tensors), default=0)
    C = lambda_and_C(tensors, budget, n_steps)

    allblocks = [t.block() for t in tensors]
    if mm.fits(allblocks, budget):
        return list(tensors)                            # everything fits -> promote all

    P = []
    Pblocks = []
    F = [t for t in tensors if mm.fits([t.block()], budget)]
    pb_stats = {"computed": 0, "nonzero": 0, "max": 0.0, "picks_changed": 0}

    def p_block(t, competitors):
        Nk = sorted((u for u in competitors if u is not t), key=lambda u: -u.Q)[:k]
        if not Nk:
            return 0.0
        hits = sum(0 if mm.fits(Pblocks + [u.block(), t.block()], budget) else 1 for u in Nk)
        return hits / len(Nk)

    while F:
        # Base score b(t)=Q/C needs no allocator. The regret boost is in
        # [1, 1+gamma], so S(t) in [b(t), b(t)(1+gamma)]. Only candidates with
        # b(t) >= b_max/(1+gamma) can possibly be the argmax -> compute the
        # expensive p_block (minimalloc calls) for those alone. Exact pruning.
        base = {t: t.Q / (C[t.name] or 1e-9) for t in F}
        bmax = max(base.values())
        thresh = bmax / (1.0 + gamma)
        best, best_s = None, -1.0
        base_best, base_best_v = None, -1.0
        for t in F:
            if base[t] >= thresh:
                pb = p_block(t, F)
                pb_stats["computed"] += 1
                if pb > 0:
                    pb_stats["nonzero"] += 1
                    pb_stats["max"] = max(pb_stats["max"], pb)
                s = base[t] * (1.0 + gamma * pb)
            else:
                s = base[t]
            if s > best_s:
                best_s, best = s, t
            if base[t] > base_best_v:
                base_best_v, base_best = base[t], t
        if best is not base_best:
            pb_stats["picks_changed"] += 1
        if mm.fits(Pblocks + [best.block()], budget):
            P.append(best); Pblocks.append(best.block())
            if verbose:
                print(f"  + {best.name[:40]:40s} Q={best.Q:>10} size={best.size:>8} S={best_s:.3g}")
        F = [t for t in F if t is not best and mm.fits(Pblocks + [t.block()], budget)]
    if verbose:
        print(f"  [regret diag] p_block computed={pb_stats['computed']} "
              f"nonzero={pb_stats['nonzero']} max={pb_stats['max']:.2f} "
              f"argmax_changed_by_regret={pb_stats['picks_changed']}")
    return P


# --------------------------------------------------------------- baselines

def greedy_by_score(tensors, budget, mm, score):
    """Plain greedy: sort by score desc, admit while MiniMalloc-feasible."""
    allblocks = [t.block() for t in tensors]
    if mm.fits(allblocks, budget):
        return list(tensors)
    P, Pblocks = [], []
    for t in sorted(tensors, key=score, reverse=True):
        if not mm.fits([t.block()], budget):
            continue
        if mm.fits(Pblocks + [t.block()], budget):
            P.append(t); Pblocks.append(t.block())
    return P


def report(name, tensors, promoted, total_Q):
    saved = sum(t.Q for t in promoted)
    print(f"  {name:16s}: promote {len(promoted):>3}/{len(tensors)}  "
          f"saved {saved/1024:>9.0f} KB off-chip  ({100*saved/total_Q:5.1f}% of {total_Q/1024:.0f} KB)")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--l2", type=int, default=1_024_000)
    ap.add_argument("--l1-frac", type=float, default=0.08, help="C_L1 as fraction of L2")
    ap.add_argument("--min-bytes", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    budget = int(a.l2 - 2 * a.l1_frac * a.l2)           # B = C_L2 - 2*C_L1
    tensors, N = load_tensors(a.onnx, min_bytes=a.min_bytes)
    total_Q = sum(t.Q for t in tensors)
    mm = MiniMalloc()
    print(f"graph: {N} steps, {len(tensors)} candidates, "
          f"total off-chip traffic if all spill = {total_Q/1024:.0f} KB")
    print(f"budget B = {budget/1024:.0f} KB  (L2={a.l2/1024:.0f}KB - 2x{a.l1_frac:.0%})")

    reg = regret_greedy(tensors, budget, mm, gamma=a.gamma, k=a.k, verbose=a.verbose)
    noreg = regret_greedy(tensors, budget, mm, gamma=0.0, k=a.k)   # Q/C base, no regret boost
    print("\n-- selection strategies (off-chip traffic saved) --")
    report(f"regret (g={a.gamma})", tensors, reg, total_Q)
    report("Q/C no-regret", tensors, noreg, total_Q)
    # ablation baselines (Sec. experiments)
    report("n_acc", tensors, greedy_by_score(tensors, budget, mm, lambda t: t.n_acc), total_Q)
    report("n_acc/ell", tensors, greedy_by_score(tensors, budget, mm,
           lambda t: t.n_acc / max(t.hi - t.lo + 1, 1)), total_Q)
    report("Q (traffic)", tensors, greedy_by_score(tensors, budget, mm, lambda t: t.Q), total_Q)
    report("largest", tensors, greedy_by_score(tensors, budget, mm, lambda t: t.size), total_Q)
    print(f"\nminimalloc binary calls: {mm._calls}")


if __name__ == "__main__":
    main()
