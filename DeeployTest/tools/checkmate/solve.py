# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Checkmate rematerialisation ILP, solved with CP-SAT.

Emits the schedule format ``testUtils.trainingUtils._loadRecomputeSchedule`` already
reads: ``{"seq": [[node_name, is_recompute], ...]}``, one entry per execution, so a
recomputed node appears more than once. Point ``--recompute-schedule`` at the result.

    solve.py <deployed.onnx> <m_max_KB> <base_sched> <ban_ops_csv> <time_limit_s> [floor_KB]
    m_max_KB == 0  ->  print MODELPEAK=<bytes>, the unconstrained liveness peak, and exit

``<deployed.onnx>`` must be the graph the deployer will actually schedule, i.e.
``deeployStates/backend_post_parsing.onnx`` from a prior run, not the exported model:
parsing fuses and renames nodes, and a schedule solved against the wrong node set cannot
be replayed.

Formulation: Jain et al., "Checkmate: Breaking the Memory Wall with Optimal Tensor
Rematerialization", MLSys 2020. R[t,i] computes node i at stage t; S[t,q] retains tensor
q into stage t; stage t is the stage that first produces node t, which pins the execution
order as an INPUT -- whatever memory the ordering alone can remove must be removed before
this runs, which is what the memory-minimising scheduler in trainingUtils does.

Retention is indexed by TENSOR while computation is indexed by NODE. That asymmetry is
the point: a node materialises all of its outputs when it runs, but its outputs die at
different times, and BatchNormInternal is the case that matters -- the backward reads
saved_mean and saved_inv_std (0.1KB each) and never touches the 64KB main output. A
node-indexed S cannot express "keep saved_mean, drop the main output", and every attempt
to compensate with a stage-dependent COEFFICIENT failed the same way: a coefficient
derived from the BASELINE lifetime cannot answer what THIS schedule needs, because
rematerialisation is precisely what moves a consumer past the baseline last use. On
ResNet8 at m_max=650 that produced a schedule freeing 64KB and costing 64KB -- the
recomputed Relu forced its BatchNorm input to be retained from stage 13 to 67 -- which
the device measured at exactly zero saving.
"""
import json
import os
import sys
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # DeeployTest
from testUtils.trainingUtils import _memoryMinimisingScheduler  # noqa: E402


def materializeSchedule(R, n):
    """Linearise (R,S) into the execution sequence a code emitter would produce.

    A stage assignment does not by itself say where a clone sits in the executed order;
    emitting by stage index and, within a stage, by node index puts every recompute
    before the node whose stage it belongs to.
    """
    seq, ran = [], set()
    for t in range(n):
        for k in range(t + 1):
            if R[t, k]:
                seq.append((k, k in ran))
                ran.add(k)
    return seq


dep = sys.argv[1]
mkb = int(sys.argv[2])
base_name = sys.argv[3]
ban_csv = sys.argv[4]
tl = float(sys.argv[5])
floor_kb = int(sys.argv[6]) if len(sys.argv) > 6 else 0
EXACT = os.environ.get('EXACT', '1') == '1'

g = gs.import_onnx(onnx.load(dep))
if base_name not in ('mem_min_schedule', 'memory_minimising'):
    raise SystemExit(f'unknown base schedule {base_name!r}; the repository provides the '
                     'memory-minimising list schedule only. Pass "mem_min_schedule".')
order = [grp[0] for grp in _memoryMinimisingScheduler(g)]
idx = {id(nd): k for k, nd in enumerate(order)}


def sz(t):
    if t is None or t.shape is None:
        return 0
    p = 1
    for d in t.shape:
        p *= d if isinstance(d, int) and d > 0 else 1
    return p * 4


producer = {o.name: idx[id(nd)] for nd in order for o in nd.outputs if o.name}
names = [nd.name for nd in order]

# ALIASING.  sizes[i] must be the bytes node i's output ACTUALLY costs, which is zero
# whenever the backend gives that output the same address as an input.  The operator
# whitelist below covers only the case that was found by hand; ALIAS_FILE supplies the
# real set, read out of a completed deployment by solver/derive_aliases.py.
#
# Getting this wrong is not a rounding error, it is wrong in two compounding ways.
# M(t) is inflated (CCT: 1122KB against a measured 834KB activation peak, +34.5%; with
# the real alias set, 739KB, and the residual changes SIGN).  Worse, the solver then
# spends recomputes on tensors that are already free: it drops an aliased Reshape,
# books the full saving, and the device saves nothing because that address belonged to
# the input anyway -- while the clone's output is a genuinely new buffer.  The model
# goes down and the realised peak goes UP.  Measured on two CCT schedules: the one
# with 10 aliased Reshape clones of 28 books 36% of its saving on tensors that do not
# exist; the schedule that realised well has zero aliased clones of 18.
#
# So an aliased node is BOTH zero-sized and banned from recompute (see `ban` below).
ALIAS_FREE = {'InPlaceAccumulatorV2'}
_alias_names = set()
_alias_file = os.environ.get('ALIAS_FILE')
if _alias_file:
    _alias_names = set(json.load(open(_alias_file))['aliased'])
    print(f'ALIAS_FILE={_alias_file}: {len(_alias_names)} nodes aliased by the backend')
_aliased = {k for k, nd in enumerate(order) if nd.op in ALIAS_FREE or nd.name in _alias_names}
sizes = [0 if k in _aliased else sum(sz(o) for o in nd.outputs) for k, nd in enumerate(order)]

# PER-TENSOR LIFETIME.  `sizes[i]` above is node-level: it sums every output of node i,
# and `last[i]` below takes the latest consumer of ANY of them. A node whose outputs die
# at different times is then charged its full size for the lifetime of its longest-lived
# output, however small that output is.
#
# BatchNormInternal is exactly that node. Its backward reads saved_mean and saved_inv_std
# (0.1KB each) and never touches the 64KB main output, so on the device the main output
# is freed two steps after it is produced while the two small ones survive to the
# backward. Node-level accounting drags all 64.2KB along for 95 of 103 stages. On ResNet8
# that is 9 nodes and 341.2KB of a 437KB over-charge, and it is what puts the model's
# unconstrained peak at 997KB against a measured 689KB (+44.7%).
#
# The fix keeps the formulation node-indexed -- R[t,i] and S[t,i] are untouched -- and
# only makes the memory COEFFICIENT stage-dependent: at stage t, node i is charged for
# the outputs whose own last use has not passed. sizeAt[t][i] <= sizes[i] always, with
# equality for the single-output nodes that are the majority.
_tprod = {o.name: k for k, nd in enumerate(order) for o in nd.outputs if o is not None and o.name}
_tlast = {name: k for name, k in _tprod.items()}
for k, nd in enumerate(order):
    for i in nd.inputs:
        if i is not None and i.name and i.name in _tlast:
            _tlast[i.name] = max(_tlast[i.name], k)
preds = [sorted({producer[i.name] for i in nd.inputs
                 if i is not None and i.name and i.name in producer and producer[i.name] < idx[id(nd)]})
         for nd in order]
n = len(order)

# TENSOR-INDEXED RETENTION.  S used to be S[t, node]: one bit for a whole node, so a
# node whose outputs have different lifetimes could not be expressed at all. Every patch
# before this one tried to compensate with a stage-dependent COEFFICIENT (sizeAt) and each
# failed the same way, because a coefficient derived from the BASELINE cannot answer what
# THIS schedule needs: rematerialisation is precisely what moves a consumer past the
# baseline last use. Concretely, on ResNet8 at m_max=650 the solver recomputed a Relu at
# stage 67, which forces its input -- BatchNorm's 64KB main output -- to be retained from
# stage 13 to 67; sizeAt priced those 84 retentions at 74.5KB total because the baseline
# says that output dies at stage 13, where the constant coefficient prices them at
# 5397KB. The solver took a recompute that frees 64KB and costs 64KB, and the device
# measured exactly zero saving.
#
# So S is indexed by TENSOR. R stays node-indexed, because a node is what executes.
tensors = []                      # (name, producer node, bytes)
tid = {}
for _i, _nd in enumerate(order):
    for _o in _nd.outputs:
        if _o is None or not _o.name:
            continue
        tid[_o.name] = len(tensors)
        tensors.append((_o.name, _i, 0 if _i in _aliased else sz(_o)))
nT = len(tensors)

# What a node materialises when it runs: all of its outputs.
sizes = [0 if _i in _aliased else sum(sz(_o) for _o in _nd.outputs if _o is not None)
         for _i, _nd in enumerate(order)]

# A node depends on TENSORS, not on producer nodes: BatchNorm's main output and its
# saved_mean are two independent requirements from here on.
in_tensors = [[tid[_x.name] for _x in _nd.inputs
               if _x is not None and _x.name and _x.name in tid]
              for _nd in order]
consumers_of = [[] for _ in range(nT)]
for _k, _q in ((k, q) for k, qs in enumerate(in_tensors) for q in qs):
    consumers_of[_q].append(_k)
print(f'tensor-indexed S: {nT} tensors over {n} nodes')

ALWAYS_BAN = {'InPlaceAccumulatorV2', 'Constant', 'Identity'}
BAN_OPS = ALWAYS_BAN | set(x for x in ban_csv.split(',') if x)
# Aliased nodes join the ban: recomputing one costs cycles and a fresh buffer while
# freeing nothing, so it can only ever make the realised peak worse.
ban = {k for k, nd in enumerate(order) if nd.op in BAN_OPS} | _aliased

# SCHEDULE-DERIVED PRUNING.  Every node is a decision variable at every stage, but
# most of them can never pay: if a tensor's LAST consumer is only a couple of stages
# away, retaining it costs at most that many stages, so rematerialising it cannot
# shorten any meaningful lifetime -- and a tensor of a few KB cannot move a peak
# measured in hundreds.  Fixing R[t,i]=0 (t!=i) for those shrinks the model by a
# third on these graphs (CCT-LoRA: 302 candidates -> 83 worth deciding, 213k -> 134k
# booleans) without touching the schedules that actually matter.
#
# This is the same KIND of restriction as BAN_OPS, but derived from the schedule
# rather than from the operator type, so it is far better targeted.  It is a
# near-dominance argument, NOT a proof: at a very tight budget the ILP could in
# principle want a short-span tensor dropped to shave its bytes off one peak stage.
# Validate by solving one budget both ways before trusting it on a whole sweep.
# PRUNE_BWD -- MEASURED HARMFUL, do not use.  Kept only so the negative result is not
# rediscovered.
#
# The argument for it: a forward activation is worth rematerialising if and only if a
# BACKWARD node consumes it; if only forward nodes read it, recomputing spends cycles
# and frees nothing.  That reads like a dominance argument rather than a heuristic, and
# on CCT-LoRA 76 of 114 forward nodes have no backward consumer at all.
#
# It is not a dominance argument, and it fails exactly the way PRUNE_SPAN does.  A node
# in the MIDDLE of a forward chain is consumed by the next node in that chain, which is
# also forward, so this criterion bans it -- but segment (cone) recompute regenerates
# precisely those interior nodes from a checkpoint boundary.  Both criteria apply
# single-node intuition to a structure that only pays off as a whole.  PRUNE_SPAN was
# already recorded as banning 20 of the 51 nodes a cone schedule recomputes; this bans
# the same kind.  That is not academic: on CCT the cone schedule BEATS the ILP
# (2829KB vs 2865KB), so banning what cone expresses removes the better family.
#
# Measured, CCT at m_max=561KB, feasibility only, one factor at a time:
#
#                     PRUNE_BWD=0            PRUNE_BWD=1
#     EXACT=0         OPTIMAL      38s       INFEASIBLE  24s
#     EXACT=1         UNKNOWN     312s       INFEASIBLE  64s
#
# It turns a solved budget INFEASIBLE in both accounting modes.  Note also the top
# right vs top left: EXACT=1 is UNKNOWN (time limit), NOT infeasible.  Reading that as
# INFEASIBLE is what made this pruning look necessary in the first place.
if os.environ.get('PRUNE_BWD') == '1':
    _pos = {id(nd): i for i, nd in enumerate(order)}

    def _is_grad(nd):
        return ('Grad' in nd.op) or ('backward' in nd.name) or ('grad' in nd.name.lower())

    _keep = set()
    for _i, _nd in enumerate(order):
        if _is_grad(_nd):
            _keep.add(_i)                      # backward nodes stay decidable
            continue
        _cons = [c for o in _nd.outputs if o is not None
                 for c in o.outputs if id(c) in _pos]
        if any(_is_grad(c) for c in _cons):
            _keep.add(_i)
    _drop = {i for i in range(n) if i not in _keep and i not in ban}
    ban = ban | _drop
    print(f'PRUNE_BWD=1: {len(_drop)} nodes without a backward consumer fixed to R=0, '
          f'{n - len(ban)} remain decidable')

_span_min = int(os.environ.get('PRUNE_SPAN', '0'))
_byte_min = int(os.environ.get('PRUNE_BYTES', '8192'))
if _span_min > 0:
    _pos = {id(nd): i for i, nd in enumerate(order)}
    _pruned = set()
    for _i, _nd in enumerate(order):
        if _i in ban:
            continue
        _cs = [_pos[id(c)] for o in _nd.outputs if o is not None
               for c in o.outputs if id(c) in _pos]
        _span = (max(_cs) - _i) if _cs else 0
        if _span <= _span_min or sizes[_i] < _byte_min:
            _pruned.add(_i)
    ban = ban | _pruned
    print(f'PRUNE_SPAN={_span_min} PRUNE_BYTES={_byte_min}: '
          f'{len(_pruned)} more nodes fixed to R=0, {n - len(ban)} remain decidable')

last = list(range(n))
for t in range(n):
    for j in preds[t]:
        last[j] = max(last[j], t)

if mkb == 0:
    # exact-mode natural peak: live set with intra-stage freeing = classic liveness
    _tl = {q: _p for q, (_, _p, _) in enumerate(tensors)}
    for _k, _nd in enumerate(order):
        for _x in _nd.inputs:
            if _x is not None and _x.name in tid:
                _tl[tid[_x.name]] = max(_tl[tid[_x.name]], _k)
    print('MODELPEAK=%d' % max(
        sum(b for q, (_, _p, b) in enumerate(tensors) if _p <= t <= _tl[q])
        for t in range(n)))
    print(f'NBAN={len(ban)}/{n} BANOPS={sorted(BAN_OPS)}')
    sys.exit(0)

budget = mkb * 1024 - floor_kb * 1024
if budget <= 0:
    print(f'm_max={mkb}KB: BUDGET<=0 after floor {floor_kb}KB')
    sys.exit(1)

# COST MODEL.  Checkmate's objective is sum(costs[i]*R[t,i]); with costs=1 it minimises
# the NUMBER of recomputed nodes, which is not latency -- 3 recomputed Convs cost far more
# than 9 recomputed LayerNorms, and that mismatch is what makes the realised latency curve
# non-monotone in m_max. Weight each node by its actual compute instead.
def _numel(t):
    if t is None or t.shape is None:
        return 0
    p = 1
    for d in t.shape:
        p *= d if isinstance(d, int) and d > 0 else 1
    return p
def _cost(nd):
    out = sum(_numel(o) for o in nd.outputs if o is not None)
    if nd.op in ('Conv', 'ConvGradX', 'ConvGradW'):
        w = [i for i in nd.inputs if i is not None and i.shape and len(i.shape) >= 2]
        k = 1
        if w:
            sh = [d for d in w[-1].shape if isinstance(d, int) and d > 0]
            k = 1
            for d in sh[1:]:
                k *= d
        return max(1, out * k)                       # MACs
    if nd.op in ('Gemm', 'MatMul'):
        w = [i for i in nd.inputs if i is not None and i.shape and len(i.shape) >= 1]
        k = 1
        if len(w) > 1:
            sh = [d for d in w[1].shape if isinstance(d, int) and d > 0]
            k = sh[0] if sh else 1
        return max(1, out * k)
    if nd.op in ('Softmax', 'LayerNormalization', 'BatchNormInternal', 'Gelu'):
        return max(1, out * 4)                       # multi-pass elementwise
    return max(1, out)                               # plain elementwise
costs = [_cost(nd) for nd in order]
# SCALE THE OBJECTIVE.  _cost returns MACs, which on ResNet8 span 10 .. 150,994,944 --
# a factor of 15 million -- so the objective's upper bound is ~6e10 while the model
# distinguishes only 35 distinct cost values. CP-SAT reasons over integer bounds, and a
# range that wide makes them nearly useless: the solver spends its time proving an
# objective bound rather than searching schedules. Measured on ResNet8, EXACT=0:
# m_max=896 closes in 7s, 796 in 55s, 750 does not close in 182s -- for a 103-node graph.
#
# Dividing by GRAIN and rounding up keeps every distinct value distinct as long as GRAIN
# stays well below the smallest gap between them, so the argmin is unchanged while the
# objective shrinks by three orders of magnitude. Set GRAIN=1 to disable.
_grain = int(os.environ.get('GRAIN', '1024'))
if _grain > 1:
    costs = [max(1, -(-c // _grain)) for c in costs]
    print(f'GRAIN={_grain}: kappa rescaled to {min(costs)}..{max(costs)}, '
          f'{len(set(costs))} distinct values')
model = cp_model.CpModel()
R = {(t, i): model.NewBoolVar(f'R_{t}_{i}') for t in range(n) for i in range(n)}
S = {(t, q): model.NewBoolVar(f'S_{t}_{q}') for t in range(n) for q in range(nT)}
for t in range(n):
    model.Add(R[t, t] == 1)
    for i in range(n):
        if i > t:
            model.Add(R[t, i] == 0)
        if i in ban and i != t:
            model.Add(R[t, i] == 0)

# A tensor cannot be retained before the stage that produces it.
for t in range(n):
    for q, (_, _p, _) in enumerate(tensors):
        if t <= _p:
            model.Add(S[t, q] == 0)

# Retention comes from having retained it or having produced it in the previous stage.
for t in range(1, n):
    for q, (_, _p, _) in enumerate(tensors):
        model.Add(S[t, q] <= S[t - 1, q] + R[t - 1, _p])

# DEPENDENCY, per input TENSOR rather than per predecessor node.  This is the fix: to
# recompute node i at stage t, every tensor it reads must be present at t, each on its
# own. Retaining BatchNorm's 64KB main output from stage 13 to 67 so a Relu can be
# recomputed there is now charged 64KB at every one of those stages, so the solver sees
# that the recompute frees exactly as much as it costs and declines it.
for t in range(n):
    for i in range(t + 1):
        for q in in_tensors[i]:
            model.Add(R[t, i] <= R[t, tensors[q][1]] + S[t, q])

# SAFE PRUNING ONLY.  Do NOT add "S[t,q]=0 for t past q's baseline last use" -- that is
# the exact shape of the bug this rewrite removes, since recompute is what puts a
# consumer after the baseline last use. It is sound only when no consumer of q can ever
# be recomputed, i.e. every one of them is banned: then each runs at its original stage
# and the baseline lifetime really is an upper bound.
_tlast = {q: _p for q, (_, _p, _) in enumerate(tensors)}
for _k, _nd in enumerate(order):
    for _x in _nd.inputs:
        if _x is not None and _x.name in tid:
            _tlast[tid[_x.name]] = max(_tlast[tid[_x.name]], _k)
_pruned_S = 0
for q in range(nT):
    cs = consumers_of[q]
    if cs and all(c in ban for c in cs):
        for t in range(_tlast[q] + 1, n):
            model.Add(S[t, q] == 0); _pruned_S += 1
if _pruned_S:
    print(f'safe S pruning: {_pruned_S} retentions fixed to 0 (all consumers banned)')

if EXACT:
    raise SystemExit('EXACT=1 is not ported to tensor-indexed S: the Free[t,i,k] recurrence '
                     'is written over node-indexed retention. It also does not solve at this '
                     'graph size (ResNet8, 103 nodes, UNKNOWN after 604s with a 5% gap).')

# MEMORY.  Built once and reused verbatim for the post-solve M(t), so the reported value
# cannot drift from the constrained one -- it did once, reporting 997KB at a budget of
# 689KB, which made the realisability check pass against a number the constraint makes
# impossible.
mem_terms = [[] for _ in range(n)]
for t in range(n):
    for i in range(t + 1):
        if sizes[i]:
            mem_terms[t].append((sizes[i], R[t, i]))          # production: all outputs
    for q, (_, _p, b) in enumerate(tensors):
        if b and _p < t:
            mem_terms[t].append((b, S[t, q]))                 # retention: one tensor
    model.Add(sum(c * v for c, v in mem_terms[t]) <= budget)

# Clones are the recomputations only: R[i,i] is the original execution of node i.
clone_terms = [R[t, i] for t in range(n) for i in range(t + 1) if t != i]
_maxclones = os.environ.get('MAXCLONES')
if _maxclones:
    model.Add(sum(clone_terms) <= int(_maxclones))
    print(f'MAXCLONES={_maxclones}')
_auxkb = float(os.environ.get('AUXKB', '0'))
obj = sum(costs[i] * R[t, i] for t in range(n) for i in range(t + 1))
if _auxkb:
    obj = obj + int(_auxkb) * sum(clone_terms)
    print(f'AUXKB={_auxkb}')
# FEASIBILITY-ONLY MODE.  Minimising sum(kappa_i * R) is what makes this model hard:
# at 327 nodes it times out at 300s and OOMs at 600s.  But the objective was never
# load-bearing -- the realised peak is measured, and the model's own cost has
# repeatedly failed to predict it (m_max 1200 realises 1802KB while the "better"
# m_max 1000 realises 1898KB).  What the sweep actually needs is ANY schedule that
# fits the budget.  Dropping the objective and stopping at the first solution turns
# each budget from a hard optimisation into a constraint-satisfaction problem.
#
# The cost is real and must be stated wherever these points are reported: the
# schedule is feasible, not cost-minimal, so its CYCLE count is an upper bound and
# should not be compared against optimally-solved points on the latency axis.
if os.environ.get('FEAS_ONLY') == '1':
    sv_first = True
    print('FEAS_ONLY=1: no objective, first feasible schedule wins')
else:
    sv_first = False
    model.Minimize(obj)
sv = cp_model.CpSolver()
sv.parameters.max_time_in_seconds = tl
sv.parameters.num_search_workers = int(os.environ.get('WORKERS', '8'))
# GAP -- stop once the incumbent is provably within this relative gap of the optimum.
# Finding a good schedule is fast; PROVING optimality is what runs out the clock (CCT
# at m_max=561: EXACT=0 closes in 38s, EXACT=1 is still UNKNOWN at 312s).  Spending
# that time is buying precision on the wrong quantity -- the objective counts recompute
# cycles, and the model's own cost has repeatedly failed to predict the realised peak,
# which is why the ILP is used as a candidate generator and the measurement arbitrates.
# Unlike FEAS_ONLY, which drops the objective entirely and makes cycle counts
# incomparable between points, this keeps every point on the same objective with a
# stated quality bound, so a curve mixing gap-limited points is still a curve.
_gap = float(os.environ.get('GAP', '0'))
if _gap:
    sv.parameters.relative_gap_limit = _gap
    print(f'GAP={_gap}: stopping at proven {_gap*100:.0f}% of optimum')
# Solution-pool mode: the model peak does not predict the realized peak, so the ILP
# is most useful as a DIVERSE CANDIDATE GENERATOR -- vary the seed to get different
# optima at the same m_max and let the realized measurement pick the winner.
sv.parameters.random_seed = int(os.environ.get('SEED', '0'))
if sv_first:
    sv.parameters.stop_after_first_solution = True
st = sv.Solve(model)
nm = {cp_model.OPTIMAL: 'OPTIMAL', cp_model.FEASIBLE: 'FEASIBLE',
      cp_model.INFEASIBLE: 'INFEASIBLE'}.get(st, 'UNKNOWN')
if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f'm_max={mkb}KB(budget {budget // 1024}KB) EXACT={int(EXACT)}: {nm}')
    sys.exit(1)
# REPORT THE MODEL'S OWN M(t) AT THE SOLUTION, not just the budget it was given.
# m_max is only an upper bound on it, so quoting "the model claimed <= m_max" against a
# measured activation understates the error by however much slack the solution left.
# It also tests the part of the formulation the baseline agreement never did: MODELPEAK
# 689.1KB against a measured 689.2KB was obtained with R diagonal and S pure liveness,
# i.e. with none of the recompute machinery active. Every claim being made is about
# schedules where it IS active.
# Same list the constraint was built from, so the two cannot diverge.
_mt = max(sum(c * sv.Value(v) for c, v in mem_terms[t]) for t in range(n))
assert _mt <= budget, f'M(t)={_mt} exceeds budget={budget} -- constraint and report disagree'
print(f'MODEL_MT={_mt} ({_mt/1024:.1f}KB) at budget {budget/1024:.0f}KB '
      f'-- slack {100*(1 - _mt/budget):.1f}%')

if os.environ.get('DUMP_RS'):
    json.dump({'n': n, 'nT': nT,
               'R': [[int(sv.Value(R[t, i])) for i in range(n)] for t in range(n)],
               'S': [[int(sv.Value(S[t, q])) for q in range(nT)] for t in range(n)],
               'tensors': tensors, 'sizes': sizes, 'names': names},
              open(os.environ['DUMP_RS'], 'w'))
    print(f"DUMP_RS -> {os.environ['DUMP_RS']}")

seq = materializeSchedule({(t, i): sv.Value(R[t, i])
                           for t in range(n) for i in range(n)}, n)
json.dump({'seq': [[names[k], int(r)] for k, r in seq]},
          open(os.environ.get('SEQ_PATH', 'recompute_checkmate.json'), 'w'))
# REALISABILITY CHECK.  M(t) is computed from the same (R,S) the constraints were
# written over, so an error in the memory model is self-consistent and invisible from
# inside it. This replays the MATERIALISED sequence with plain reference counting --
# allocate a fresh buffer per execution, free it when no later execution reads it -- and
# compares. It shares no code with the constraint logic.
#
# Measured on ResNet8 with no recompute the three agree: M(t) 689.1KB, replay 691.8KB,
# device 689.2KB. At m_max=300 with 12 recomputes they do not: M(t) 296.6KB against a
# replay of 565.3KB and a device activation of 562.7KB. At the model's peak stage, seven
# recomputed tensors totalling 368KB carry S=0 and R=0 while still being live in the
# executed order: the model books a drop that the materialised sequence never performs,
# because nothing forces the consumers between the drop and the recompute onto the clone.
# The gap therefore grows with the number of recomputes, which is exactly the regime
# every claim about this solver is made in.
_cur, _execs = {}, []
for _k, (_ni, _r) in enumerate(seq):
    _nd = order[_ni]
    _in = [_cur.get(x.name) for x in _nd.inputs if x is not None and x.name]
    _out = []
    for _o in _nd.outputs:
        if _o is None or not _o.name:
            continue
        _vid = (_o.name, _k)
        _cur[_o.name] = _vid
        _out.append((_vid, 0 if _nd.op in ALIAS_FREE else sz(_o)))
    _execs.append(([x for x in _in if x], _out))
# Every produced version must be registered, not only the consumed ones. A version that
# nothing reads never entered _rem, so the `_rem[v] == 0` release never fired for it and
# it stayed live to the end -- the replay's own version of the leak this whole
# investigation started from. Worth a constant 2.6KB here (48 BatchNorm saved_mean /
# saved_inv_std outputs at 0.2KB), which is exactly the offset the replay showed against
# the device on both the baseline and the m_max=300 schedule.
_rem = {}
for _i, _o in _execs:
    for _vid, _b in _o:
        _rem.setdefault(_vid, 0)
    for _v in _i:
        _rem[_v] = _rem.get(_v, 0) + 1
_live, _cs, _replay = {}, 0, 0
for _i, _o in _execs:
    for _vid, _b in _o:
        if _vid not in _live:
            _live[_vid] = _b; _cs += _b
    _replay = max(_replay, _cs)
    for _vid, _b in _o:                      # an output nobody reads dies immediately
        if _rem.get(_vid, 0) == 0 and _vid in _live:
            _cs -= _live.pop(_vid)
    for _v in _i:
        _rem[_v] -= 1
        if _rem[_v] == 0 and _v in _live:
            _cs -= _live.pop(_v)
_ratio = _replay / _mt if _mt else 0
print(f'REPLAY={_replay} ({_replay/1024:.1f}KB) vs MODEL_MT {_mt/1024:.1f}KB '
      f'-> {"REALISABLE" if _ratio <= 1.05 else f"NOT REALISABLE, {100*(_ratio-1):+.0f}%"}')

rc_cost = sum(costs[k] for k, r in seq if r)
base_cost = sum(costs)
print(f'm_max={mkb}KB(budget {budget // 1024}KB) EXACT={int(EXACT)}: {nm} '
      f'{len(seq)}exec {sum(1 for _, r in seq if r)}recomp '
      f'recompute_cost={rc_cost} (+{rc_cost/base_cost*100:.1f}% of one pass)')
