# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Solve a gradient-checkpointing schedule against the graph Deeploy actually deploys.

A schedule is keyed on node names and sized against a specific graph, so one solved
before a lowering change replays as the default order afterwards. That is not
hypothetical: every ResNet8 schedule solved before the models moved to channels-first
convolutions matches 67-75% of the current graph, which is below the coverage the
replay accepts.

This solves against `backend_post_binding.onnx` -- the graph after binding, which is
what the tiler and the allocator see -- using the per-operator cost trace registered
for the model, and writes a sequence the `--recomputeSchedule` flag can replay.

    python solve_recompute_schedule.py <net> <backend_post_binding.onnx> <frac> <out.json>

`frac` scales the unconstrained peak: 0.9 asks the solver for a schedule whose peak
is at most 90% of what the same graph needs with nothing recomputed.
"""

import json
import sys
import time

sys.path.insert(0, '/home/agent/UltraTorch')

from ortools.linear_solver import pywraplp
from ultratorch.checkmate import CostModel, load_onnx, replay_schedule, summarize


def solveLpRelaxation(graph, memoryBudget, timeLimitSeconds = 90):
    """LP relaxation of the Checkmate ILP.

    The exact formulation is an integer program and does not finish on graphs this
    size: CCT's 242 nodes time out past two minutes, and the exact solver's Free[t,i,k]
    variables exhaust memory outright. The relaxation is polynomial and rounds to a
    valid schedule, which `replay_schedule` then checks rather than assumes.
    """
    n = graph.n
    solver = pywraplp.Solver.CreateSolver('GLOP')
    solver.set_time_limit(int(timeLimitSeconds * 1000))

    R = {(t, i): solver.NumVar(0, 1, '') for t in range(n) for i in range(n)}
    S = {(t, i): solver.NumVar(0, 1, '') for t in range(n) for i in range(n)}

    for t in range(n):
        solver.Add(R[t, t] == 1)  # the step's own node always runs
        for i in range(n):
            if i > t:
                solver.Add(R[t, i] == 0)
            if i >= t:
                solver.Add(S[t, i] == 0)
    for i in range(n):
        solver.Add(S[0, i] == 0)
    for t in range(1, n):
        for i in range(n):
            solver.Add(S[t, i] <= S[t - 1, i] + R[t - 1, i])  # only carry what existed
    for t in range(n):
        for i in range(t + 1):
            for j in graph.preds[i]:
                solver.Add(R[t, i] <= R[t, j] + S[t, j])  # inputs present to recompute
        solver.Add(sum(graph.sizes[i] * (S[t, i] + R[t, i]) for i in range(t + 1)) <= memoryBudget)

    solver.Minimize(sum(graph.costs[i] * R[t, i] for t in range(n) for i in range(t + 1)))

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None
    return ({
        (t, i): R[t, i].solution_value() for t in range(n) for i in range(n)
    }, {
        (t, i): S[t, i].solution_value() for t in range(n) for i in range(n)
    })


def roundSchedule(graph, lpR, lpS):
    """Round the relaxation without letting recompute cascade.

    R is rounded by threshold, which keeps the cones the LP chose. S is then forced
    minimally by R: a tensor is carried only from its last production to a later use
    that does not recompute it. Rounding S independently would add recomputes the LP
    never priced, and those cascade.
    """
    from ultratorch.checkmate import Solution
    n = graph.n
    R = {(t, i): 0 for t in range(n) for i in range(n)}
    S = {(t, i): 0 for t in range(n) for i in range(n)}
    for t in range(n):
        R[t, t] = 1
        for i in range(t):
            if lpR[t, i] > 0.5:
                R[t, i] = 1
    # S is forced by R, never rounded on its own: carry a tensor from its last
    # production up to a use that does not recompute it, and no further.
    for t in range(n):
        for i in range(t + 1):
            if R[t, i] != 1:
                continue
            for j in graph.preds[i]:
                if R[t, j] == 1:
                    continue
                produced = [tt for tt in range(t) if R[tt, j] == 1]
                if not produced:
                    continue
                for tt in range(max(produced) + 1, t + 1):
                    S[tt, j] = 1
    objective = sum(graph.costs[i] * R[t, i] for t in range(n) for i in range(t + 1))
    singlePass = graph.single_pass_cost()
    return Solution(status = "ROUNDED",
                    objective = int(objective),
                    recompute_cost = int(objective - singlePass),
                    overhead = (objective - singlePass) / singlePass if singlePass else 0,
                    R = R,
                    S = S)


def emitSequence(graph, solution):
    """Flatten the R matrix into the run order the replay expects.

    One entry per execution, not per node: a node that is recomputed appears more than
    once, and every appearance after the first is flagged so the replay clones it.
    """
    seen = set()
    sequence = []
    for t in range(graph.n):
        for i in range(t + 1):
            if not solution.R.get((t, i)):
                continue
            sequence.append([graph.names[i], i in seen])
            seen.add(i)
    return sequence


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 1
    net, graphPath, frac, outPath = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]

    costModel, costSource = CostModel.for_model(net)
    if costSource.startswith('merged'):
        print(f"refusing to solve {net} against {costSource}: that is a cost model averaged "
              f"from other networks, and a schedule chosen on it is not this model's schedule")
        return 1
    graph = load_onnx(graphPath, chain_only = False, cost_model = costModel)

    start = time.time()
    relaxed = solveLpRelaxation(graph, sum(graph.sizes))
    if relaxed is None:
        print(f"{net} frac={frac}: unconstrained solve failed")
        return 1
    basePeak = summarize(graph, roundSchedule(graph, *relaxed))['peak_bytes']
    print(f"{net}: n={graph.n}, cost={costSource}, unconstrained peak={basePeak / 1024:.0f} KB "
          f"({time.time() - start:.0f}s)")

    start = time.time()
    relaxed = solveLpRelaxation(graph, int(basePeak * frac))
    if relaxed is None:
        print(f"  frac={frac}: INFEASIBLE")
        return 1
    solution = roundSchedule(graph, *relaxed)
    replay_schedule(graph, solution)  # raises if the rounded schedule is not runnable
    peak = summarize(graph, solution)['peak_bytes']
    sequence = emitSequence(graph, solution)
    recomputes = sum(1 for _, isRecompute in sequence if isRecompute)

    with open(outPath, 'w') as handle:
        json.dump({"seq": sequence}, handle)
    print(f"  frac={frac}: peak={peak / 1024:.0f} KB ({100 * peak / basePeak:.0f}% of base), "
          f"{recomputes} recomputes, {len(sequence)} entries, {time.time() - start:.0f}s -> {outPath}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
