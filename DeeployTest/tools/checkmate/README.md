# Checkmate rematerialisation solver

Produces the schedules `--recompute-schedule` consumes. `trainingUtils` already knows how
to replay one (`_loadRecomputeSchedule`); this is the half that generates and validates it.

## The three steps

Solve/enumerate, deploy for peak memory, gvsoc for latency. Reference values from this
repository are given at each step, so a wrong turn shows up where it was taken rather
than after a full sweep.

### 0. Deploy once, to get the graph the deployer will actually schedule

```bash
deeployTrainingRunner_tiled_gap9.py \
    -t Tests/Models/Training/CCT/cct_train -p GAP9 -s gvsoc --toolchain LLVM \
    --l1 122000 --defaultMemLevel L3 --cores 8 --n-steps 1 --n-accum 1 \
    --num-data-inputs 1
# -> <build>/Tests/Models/Training/CCT/cct_train/deeployStates/backend_post_parsing.onnx
```

> **That graph must come from a ZERO-recompute deployment**, and checking the node count
> is how you know. Enumerating against a deployment that already had one MobileNetV1
> block rematerialised -- 287 nodes with six clones in them rather than the baseline's
> 281 -- yields a baseline of 1704.2 KB instead of 1848.4, and errors nowhere.

### 1. Solve

```bash
export PYTHONPATH=<repo>          # for testUtils.trainingUtils

# MobileNetV1: group-level enumeration. See "Two axes" below for why this is the
# right tool there and the wrong one on the other two.
tools/checkmate/enumerate_groups.py <graph.onnx> blocks recompute_mnv1   # 2^13

# ResNet8 and CCT: node-level ILP. Second argument is the activation budget in KB.
# ResNet8: sweep 689/650/600/550/500/450/400/350, below 300 is infeasible.
SEQ_PATH=recompute_rn8.json \
tools/checkmate/solve.py <graph.onnx> 400 mem_min_schedule ConvGradW 240
```

Replay KB to check against -- if these do not match, the graph is wrong:

```
CCT          962.5  738.5  705.5  704.5  672.5  672.5  768.5
              ^0     ^1     ^2     ^3     ^4     ^6     ^7 <- rebounds
MobileNetV1 1848.4 1560.2 1343.7 1199.3 1090.3 1017.6  961.6  905.6  896.6
              ^0     ^1     ^2     ^3     ^4     ^5     ^6     ^7     ^8 <- floor
```

CCT's seventh group **rebounds**: adding stem&head takes the replay from 672.5 to 768.5.
Not a bug -- the frozen tokenizer chain has one 32 KB tensor the backward needs, and
recomputing it materialises the whole chain's ~512 KB of intermediates. Leaving it in the
enumeration is more useful than excluding it by hand: it is an explicable result rather
than a trap.

### 2. Deploy, and read the peak

```bash
deeployTrainingRunner_tiled_gap9.py ... --recomputeSchedule recompute_cct_3.json
```

**Read this line before any number:**

```
[Recompute] 358 scheduled executions, 114 recompute clones
```

Zero clones, or no such line, means the point is void. See the section below on why that
is the only evidence a schedule ran.

The peak comes from the deployment's own `deeployStates/memory_alloc.html` -- the
address-deduplicated sum of live blocks.

| CCT groups | surviving clones | activation | peak |
|---|---|---|---|
| 0 | 0 | 834.5 KB | 2899 KB |
| 1 | 53 | 674.5 | 2739 |
| 2 | 106 | 577.5 | 2642 |
| 3 | 114 | **545.5** | **2610** |
| 4 | 126 | 545.5 | 2610 (dominated by 3) |
| 6 | 158 | 578.5 | 2643 (**worse** than 3) |

Persistent memory is **2064.8 KB at every one of those points, byte for byte** -- it is
what rematerialisation cannot touch. If it moves between points, the split is wrong.

Six groups lands 33 KB above three while the replay has them tied at 672.5 KB: **each
surviving clone occupies memory the replay does not model.** This is the concrete cost of
trusting a prediction, and the reason every reported point has to be deployed.

### 3. gvsoc, for the latency

Same command (`-s gvsoc`). CCT at 370 MHz:

| groups | cycles | vs baseline |
|---|---|---|
| 0 | 71.17 M | -- |
| 1 | 74.13 M | +4.17% |
| 2 | 76.90 M | +8.05% |
| 3 | 77.09 M | +8.32% |

**There is a ~0.3% noise floor.** Two runs of an identical configuration gave 78.84 M and
78.62 M: codegen is not deterministic -- two identity-schedule runs emit C differing in
3685 lines of declaration ordering, and pinning `PYTHONHASHSEED` only reduces that to 744
-- which moves buffer layout and DMA behaviour. Memory peaks are stable to the kilobyte.
**No cycle difference below about 0.5% is readable.**

### Two switches that silently produce wrong numbers

`--convChannelsFirst` is a **per-network** decision, not a preference. On CCT it buys
**0 KB** (2899 with and without, same tree same tool one flag) and costs **9.8%** cycles;
it removes two transposes and neither is live at the peak. On MobileNetV1 it is
mandatory -- training does not fit GAP9 L1 without it.

A schedule is bound to the graph version it was solved against. `recomputeScheduler`
replays by node name and **raises** below 90% coverage rather than degrading to the
default order. Change a topology pass or a branch and the schedule must be re-solved; old
JSON cannot be reused.

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

## Two axes, and each network sits somewhere different

Rematerialisation methods differ along two independent axes: the granularity of the
decision (per node, or per repeated group) and the set of tensors eligible for
recomputation (forward activations only, or any intermediate). Checkmate takes the finest
choice on both and pays in solver time; Rockmate restricts to both, which is what makes
it tractable -- on models where its assumption holds, that the memory worth reclaiming is
forward activations kept alive across the backward pass.

**That assumption is a property of the network, not of the method.** Activation memory
after rematerialisation, GAP9, replay estimate:

| | baseline | forward only | all nodes | node-level ILP |
|---|---|---|---|---|
| MobileNetV1 | 1848.4 KB | **895.8 KB** (−51.5%) | — | no feasible solution |
| ResNet8 | 689.1 KB | 608.4 KB (−11.7%) | — | **344.6 KB** (−50.0%) |
| CCT | 962.5 KB | 961.5 KB (−0.1%) | 962.5 KB (0%) | 707.2 KB (−26.5%) |

MobileNetV1 behaves as the literature assumes, and the node-level ILP is not merely
slower there but returns no feasible schedule at all in 25 minutes on its 281 nodes.

ResNet8 does not. Forward-only reaches −11.7%; lifting the restriction at node
granularity reaches −50.0%. The group-level method is not approximating the node-level
one, it is solving a smaller and different problem.

CCT is the extreme case: recomputing forward activations reclaims **nothing**, with
either candidate set, and past five groups it makes the peak worse. Of the 170 nodes in
its backward pass, 113 are `Gemm`, `Transpose` or `Reshape` -- indistinguishable by
operator type from its forward pass -- and it is those intermediates that dominate what
can be reclaimed. Use `solve.py` on CCT.

```
# MobileNetV1: one group per repeated block, 2^13 candidates
enumerate_groups.py <graph.onnx> blocks recompute_mnv1
# CCT / ResNet8: node-level ILP
SEQ_PATH=out.json solve.py <graph.onnx> 400 mem_min_schedule ConvGradW 240
```

## Where a recompute is placed matters more than which nodes are chosen

A group's recomputes are anchored at the earliest **backward** node that reads any of the
group's outputs, found per tensor. Both qualifiers were learned the expensive way:

*It must be a backward consumer.* A block's output is read by the next block's forward
immediately, so anchoring at the first consumer of any kind lands right after the group.
Measured: every MobileNetV1 point collapses to the 1848.4 KB baseline. What
rematerialisation avoids is the forward-to-backward lifetime, not the consumption.

*It must be per tensor, not per group.* An earlier version anchored at "the group's first
backward node", which depends on how forward and backward are told apart -- and the two
plausible definitions disagree by up to 177 positions in a 244-step order. Anchoring 177
steps early keeps every recomputed value alive that much longer.

Fixing the anchor is worth more than any choice of candidate set:

| groups | anchored per group | anchored per tensor |
|---|---|---|
| 1 | 1560.2 KB | **1416.2 KB** |
| 2 | 1343.7 | **1127.7** |
| 3 | 1199.3 | **982.7** |
| 7 | 905.6 | **895.8** |

Three groups now reach what seven reached before: the same peak for roughly half the
recomputes, and therefore half the latency.

## A schedule is only as good as the proof that it ran

Rematerialisation is numerically neutral, so a schedule that never reached codegen still
produces `Errors: 0` and a bit-identical loss, and a cycle count that looks entirely
reasonable. Neither is evidence. The evidence is the **clone count** in
`[Recompute] N scheduled executions, M recompute clones`.

Two consequences worth internalising:

- Requested recomputes and surviving clones are different numbers. A CCT schedule asking
  for 106 got 44: the other 62 were placed in stages whose consumers linearise before the
  clone, so nothing reads them, and `_pruneDeadRecomputes` drops them. **Report the
  surviving count.**
- Three CCT points once came back within 0.3% of baseline and read as "recompute is
  nearly free on CCT". They were the baseline, measured three times, through an injection
  path that had silently stopped reaching codegen. Assert the clone count on every point.

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
