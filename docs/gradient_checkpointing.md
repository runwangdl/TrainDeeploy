# Gradient checkpointing on GAP9

Checkpointing drops an activation after its forward use and regenerates it before the
backward pass reads it. Deciding *what* to drop is a solve over the training graph;
running that decision is a replay over the graph Deeploy actually deploys. Those are
two different graphs, and most of the difficulty lives in the gap between them.

## Why schedules expire

A schedule is a list of node names in run order. It is solved against one graph and
sized against that graph's tensors, so a lowering change invalidates it silently:
the names stop matching, every recompute is skipped, and what runs is the default
order under a name that claims to be checkpointed.

This is not hypothetical. Every ResNet8 schedule solved before the model moved to
channels-first convolutions matches 67-75% of the current deployed graph.
`recomputeScheduler` refuses anything below 90% for that reason — a schedule that
half-matches is worse than no schedule, because the test still passes.

Re-solve after any change that alters the graph: a new lowering pass, a kernel whose
binding changes operand types, or a flag like `--convChannelsFirst` that removes the
transposes entirely.

## Solving

    python scripts/solve_recompute_schedule.py <net> <backend_post_binding.onnx> <frac> <out.json>

Three things matter about the inputs.

**Use `backend_post_binding.onnx`, not `backend_post_parsing.onnx`.** Binding fixes
constant types and can rewrite nodes; the tiler and the allocator see the post-binding
graph, so that is the one whose peak means anything.

**Generate that graph with the model's CI configuration.** `--convChannelsFirst`,
`--num-data-inputs`, `--defaultMemLevel` and the L1 budget all change the graph or its
tensor sizes. A schedule solved against a different configuration will not replay.

**Check the cost source.** `CostModel.for_model` falls back to an average over other
networks when a model has no registered trace, and says so only in its returned
`source` string. The script refuses to solve against a `merged-*` model rather than
produce a schedule chosen on another network's operator costs.

`frac` scales the unconstrained peak: 0.9 asks for a schedule peaking at most 90% of
what the same graph needs with nothing recomputed.

## What the solver does and does not tell you

The formulation is Checkmate's, relaxed to an LP and rounded. The exact ILP does not
finish at this size — CCT's graph times out past two minutes and the exact solver's
`Free[t,i,k]` variables exhaust memory outright.

Rounding is asymmetric on purpose. `R` is rounded by threshold, which keeps the cones
the LP chose. `S` is then *forced* by `R`: carry a tensor from its last production up
to a use that does not recompute it, and no further. Rounding `S` independently adds
recomputes the LP never priced, and those cascade.

`replay_schedule` then checks the rounded schedule is runnable. That check is worth
having and easy to over-read: it validates the sequence inside the abstract model. It
does not deploy, does not compile, and does not run. **A solved schedule is a
prediction until gvsoc has run it.**

## Verifying

    python deeployTrainingRunner_tiled_gap9.py -t <model> -p GAP9 -s gvsoc \
      --l1 <budget> --defaultMemLevel L3 --cores 8 --n-steps 1 --n-accum 1 \
      --searchStrategy max --recomputeSchedule <out.json> [model's CI flags]

Look for `Errors: 0` and `BENCH train_cycles=`. Anything short of that is not a result.

Two traps when comparing numbers. CI trains four mini-batches and most hand runs use
one, so per-step figures from the two are not comparable — and the four-batch
accumulator buffers also take about 6 KB of L1, which is why on-chip entries carry a
lower L1 budget than the measurement that produced their cycle count. And builds go
through ccache regardless of `rm -rf TEST_GAP9`; export `CCACHE_DISABLE=1` for
anything where the answer depends on the build being clean.

## Measured

Solved against the current graphs with each model's CI configuration, verified on
gvsoc at one mini-batch:

| model | recomputes | peak vs unconstrained | Errors | cycles |
|---|---|---|---|---|
| MobileNetV1 | 16 | 2910 KB of 3198 KB (91%) | 0 | 52,727,331 |

For reference, the schedule this replaced was solved on an older graph: 19 recomputes
and 54,166,817 cycles. Fewer recomputes and less runtime, from re-solving rather than
from any change to the kernels.

Recompute count does not predict the saving. A 113-recompute ResNet8 schedule needed
more arena (1535196 B) than doing nothing at all, while an 86-recompute one for the
same model needed the least of any tried. What is recomputed matters; how much is
recomputed does not.
