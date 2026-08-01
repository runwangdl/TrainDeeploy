# Checkmate rematerialisation solver

Produces the schedules `--recompute-schedule` consumes. `trainingUtils` already knows how
to replay one (`_loadRecomputeSchedule`); this is the half that generates and validates it.

```
# 1. deploy once to get the graph the deployer will actually schedule
#    -> <dump>/deeployStates/backend_post_parsing.onnx
# 2. what does it cost with no rematerialisation at all?
solve.py <graph.onnx> 0 mem_min_schedule ConvGradW 10          # prints MODELPEAK

# 3. solve for a budget, in KB of activation
SEQ_PATH=recompute.json solve.py <graph.onnx> 400 mem_min_schedule ConvGradW 240

# 4. deploy with it
deeployTrainingRunner_tiled_gap9.py ... --recompute-schedule recompute.json
```

Solve at the **post-parsing** graph, never the exported model: parsing fuses and renames
nodes, and a schedule solved against the wrong node set cannot be replayed.

## Measured, ResNet8 on GAP9

`--l1 122000 --defaultMemLevel L3`, L3 peak, address-deduplicated. Peak is a liveness
sum; `arena` is the address MiniMalloc actually reaches and is what must be reserved.

| m_max | recomputes | peak | arena | cycles |
|---|---|---|---|---|
| — | 0 | 1313 KB | 1321 KB | 49.31 M |
| 400 | 12 | 1000 KB | 1041 KB | 56.03 M |
| 350 | 30 | 969 KB | 977 KB | 68.93 M |

Both solved points ran on gvsoc with `Errors: 0` and a bit-identical loss
(`3ff13b69`). Below m_max=300 the model is infeasible. 12 recomputes buy 23.8% of the
memory for 13.6% of the time; the next 18 buy 2.4% more for another 26 points of it.

## Three things worth knowing before changing this

**Retention is per TENSOR, computation is per NODE.** A node materialises all of its
outputs when it runs, but they die at different times: BatchNormInternal's backward reads
`saved_mean` and `saved_inv_std` (0.1 KB each) and never touches the 64 KB main output. A
node-indexed `S` cannot express "keep saved_mean, drop the main output", and compensating
with a stage-dependent coefficient does not work — a coefficient derived from the
*baseline* lifetime cannot answer what *this* schedule needs, because rematerialisation is
exactly what moves a consumer past the baseline last use. That version chose a schedule
freeing 64 KB and costing 64 KB, and the device measured zero saving.

**Do not prune `S[t,q] = 0` past a tensor's baseline last use.** Same trap. It is sound
only when every consumer of the tensor is banned from recompute, which `solve.py` already
applies.

**`replay.py` shares no code with the solver, deliberately.** `M(t)` is computed from the
same `(R,S)` the constraints were written over, so an error in the memory model is
self-consistent and invisible from inside it. The replay walks the materialised sequence
with plain reference counting; `solve.py` runs it after every solve and prints
`REALISABLE` or the percentage by which the schedule exceeds what the model claimed. A
solve that reports `NOT REALISABLE` should not be deployed and should not be believed.

## derive_aliases.py

Reads a completed deployment's allocation plot and lists the nodes whose output shares an
address *and* a lifetime with an input — Deeploy aliases far more than the operator
whitelist in `solve.py` knows about (on CCT, 47 Reshape nodes). Feed the result back via
`ALIAS_FILE` so the solver neither charges for bytes that do not exist nor spends a
recompute freeing them. Requiring the lifetimes to overlap is not optional: the allocator
reuses an address as soon as its occupant dies, so matching ranges alone conflates reuse
with aliasing.

This is a diagnostic, not a predictor — it needs one deployment to learn the aliases.
