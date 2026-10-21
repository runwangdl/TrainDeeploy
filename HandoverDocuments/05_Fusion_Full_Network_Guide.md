# 05 — Dual-pass fusion from the per-block kernels to the full QZO(+ECO) network on GAP9 + NE16

Written 2026-10-14; updated 2026-10-21 with exp9, the memory optimisation of the fused graph (§4.8, §6.10, §7.7).
Part of the handover set indexed by `TrainDeeploy/HandoverDocuments/00_Overview.md`.
Paths are relative to `/Users/qiwenwu/ETH/`; in-container paths appear only inside command blocks.
This guide continues `04_FusedQZO_NE16_Guide.md`, which covers the block-0 kernels (exp5, exp6); read its §1 first
if the idea of fusing the two ε-passes is new.

**Summary.** A QZO step runs the network twice per window, at θ+εz and θ−εz. Guide 04 fused the two passes of the
first convolution (block 0) into one kernel. This guide documents the two steps that followed:

- **exp7** extends the fusion to the convolutions of blocks 1–4. These are harder: the two passes no longer share
  their input. It adds:
  - a two-input fused cluster kernel `PULP_DualConv2d_2in_i8_i8_i8_HWC`, which also fixes the weight-perturbation
    order for Cin > 1;
  - a pipelined fused NE16 node `NE16DualConv2in` for GAP9;
  - a measured per-block engine choice: NE16 for blocks 0–2, cluster for blocks 3–4.
- **exp8** makes the whole training step dual-pass. A Deeploy pass, `ZODualPassFusion`, rewrites the unchanged QZO
  training graph at compile time into one graph:
  - between the convolutions the two passes travel channel-stacked as [+ | −];
  - every non-conv node runs once on 2·C channels;
  - the graph returns L+ and L− as outputs 0 and 1;
  - the harness (`-D ZO_FUSED_DUAL`) runs it once per window instead of twice.

  The result is bit-exact against the unfused run on Siracusa, the GAP9 cluster and GAP9 + NE16 (identical losses and
  weights after 4 device steps).
- **exp9** (memory, branch `feat/memory_opt`, on top of Run's board work `fe1ffa2`) fuses the fp32 tail of each frozen-BN
  block into one HWC kernel. A lowering pass, `FuseFrozenBNTailPass`, replaces
  `BN → [MaxPool] → Transpose → Relu → Quant → Transpose` with one node, `FrozenBNPoolReluQuant` (int8 in, int8 out):
  - the fp32 tensors between BN and Quant are never stored;
  - the fused QZO+ECO training arena on GAP9 + NE16 drops from 854,452 to **265,612 B** (−68.9 %);
  - the training part of a step drops by 13.0 %, as a side effect;
  - the result is bit-exact on GAP9 and Siracusa (same losses, same weight hash).

**Speedups of the training cycles:**

| platform | speedup |
|---|---|
| Siracusa | 1.53× |
| GAP9 cluster | 1.42× |
| GAP9 with NE16 for blocks 0–2 | 1.78× |

The rest of a step is unchanged, which caps the gain. Per conv block, the network reproduces the fixture speedups.

**Fine-tuning round on GAP9 + NE16** (one update step with n_accum 4 = 4 fused dual passes + the update graph):

| | cycles per update step | 120-epoch round (1,620 steps) @ 370 MHz | unfused round |
|---|--:|--:|--:|
| QZO | 8,855,589 | 38.8 s | 68.4 s |
| QZO + ECO | 9,129,644 | 40.0 s | 69.6 s |

ECO (Run's error-feedback implementation, `-D ZO_ECO=ON`) composes with the fusion without code changes. It only
needed a correct size for its momentum buffers.

## Status as of 2026-10-21

| item | state |
|---|---|
| Two-input fused cluster kernel, blocks 1–4 (`RequantizedDualConv`, 7 inputs) | works on Siracusa and GAP9; all four blocks exact; Siracusa blocks 1–4 2.44× over 2× pulp-nn (wall clock, node level) |
| Single-pass control kernel `PULP_Conv4x2_i8_i8_i8_HWC` | works, bit-identical to pulp-nn; selectable with `DEEPLOY_RQCONV_KERNEL=4x2`; default stays pulp-nn |
| Fused NE16 node `NE16DualConv2in` (GAP9), blocks 1–4 | works, exact; pipelined (double-buffered column + 2-deep job queue); wins on blocks 1–2, loses on blocks 3–4 |
| Whole-network dual-pass graph (`--zo-fused-dual`) | works on Siracusa, GAP9 cluster, GAP9 + NE16 (`--zo-fused-dual-ne16 0,1,2`); 4 device steps identical to unfused (all 22 tensors, all losses) |
| QZO + ECO on the fused graph (GAP9 + NE16, n_accum 4, one update step) | works; all 4 mini-batches equal the host reference (`Errors: 0 out of 8`) |
| Fused graph over 14 update steps of round 1 (54 windows in L3, direct QZO) | works; `Errors: 0 out of 112` (§6.9) |
| Fixtures | per-block `TrainDeeploy/SpeechNet_Test/speechnet_{unfused,fused}_b{1..4}_{cluster,NE16}`; whole-network entry fixtures `speechnet_{train,update}_QZOECO_full_fused_{single_update_step,round1}` (§5) |
| Full fine-tuning round with the fused graph | not run on GVSoC (many hours). On the board Run ran it (`fe1ffa2`, before exp9): QZO 35.1 s, QZO+ECO 36.1 s per 120-epoch round @ 370 MHz (§9 item 3) |
| Frozen-BN tail fusion `FrozenBNPoolReluQuant` (exp9) | works on GAP9, GAP9 + NE16 and Siracusa, bit-exact; fused QZO+ECO L2 arena 854,452 → 265,612 B, training −13.0 % (§4.8, §6.10); **not yet run on the board** |
| Non-conv nodes (BN, Quant/Dequant, pool, ReLU) | Run folded Dequant into BN; exp9 fused BN + MaxPool + ReLU + Quant per block (for memory, arithmetic unchanged). Folding BN into the conv requant is still open (§9 item 1) |
| Exporter support | not needed: the fusion is a Deeploy pass over the ordinary QZO export |
| Checkpoint | all numbers derive from `artifacts_reference` weights (paper-comparable): the per-block kernel fixtures from `Tests/Models/Training/SpeechNet/speechnet_qzo19_train` (pooled@99.99 calibration), the whole-network runs from `SpeechNet_Test/speechnet_train_QZO_single_step` and the new QZOECO fixtures (pooled@99.9) |

---

## 1. Concept

### 1.1 One QZO step, and where the time goes

For every window the device runs:
1. the training graph with the trainable tensors perturbed by +εz, giving the loss L+;
2. the same graph again with −εz, giving L−.

z is a Rademacher vector (±1) regenerated from a seed. After `n_accum` windows the update graph regenerates the same z
and applies `θ ← θ − lr·g_proj·z`, with `g_proj = Σ(L+ − L−) / (2ε·n_accum)`.

For an int8 conv weight the perturbation is an integer: `w± = clamp(w ± round(mul·z / 2^S), −127, 127)` (node
`RQSPerturbRademacher`). The fp32 tensors (BatchNorm γ/β, classifier W/b) use `PerturbRademacher`. Guide 03 documents
the graph; guide 04 §1.1 documents the harness loop.

**Where the time goes in one unfused step.** Siracusa, n_accum 1, both passes, wall clock per node from
`--profileTiling`; source `dual_pass_fusion/exp8_whole_network/results/step0_siracusa_nodes.txt`:

| part | share of the training cycles |
|---|--:|
| conv | 48.3 % |
| conv weight / bias perturbation | 16.5 % |
| BatchNorm | 16.0 % |
| Quant / Dequant | 9.4 % |
| pooling | 5.5 % |
| rest | about 4 % |

Training (both passes) is 5,162,943 cycles; the update graph is 321,350.

### 1.2 The fusion idea

Compute both passes in one execution instead of two:
- draw z once;
- form W+ and W− inside the conv kernel, so the separate perturbation nodes and weight transposes disappear;
- share the pixel loop, the addressing and the padding tests between the passes.

For block 0 the two passes also share the input x. For blocks 1–4 they do not: x+ and x− are the outputs of two
differently perturbed upstream blocks. That is why blocks 1–4 need a different kernel, and why their gain comes mostly
from removing the perturbation nodes.

### 1.3 The bit-exactness contract

A fused kernel or graph is accepted only if it produces **the same z** as the unfused graph. This matters for two
reasons:
- the validation compares against the unfused reference;
- the update graph is not changed by the fusion. It regenerates z with the original nodes' scheme, so a forward pass
  with a different z would apply `g_proj` along an unrelated direction, and training would become a random walk.

Every perturbation node draws its noise per core:

```
chunk_seed = ((seed + perturb_seed_base) + NUM_CORES * idx + core_id) ^ (tile_seed_offset * 0x9E3779B1)
chunk      = (size >> log2(NUM_CORES)) + (size % NUM_CORES != 0);   core c owns [c*chunk, (c+1)*chunk)
```

`TrainDeeploy/Deeploy/Targets/PULPOpen/Templates/RQSPerturbRademacherTemplate.py:41`; chunking in
`TrainDeeploy/TargetLibraries/PULPOpen/inc/kernel/DualPerturb.h:32` (`dc_chunk`, the same split as the node).

Inside its chunk a core draws one Xorshift32 word per 32 elements and uses one bit per element, LSB first. So element
i's noise bit depends on four things:
- the node index `idx`;
- the core that owns i;
- i's position in that core's chunk;
- the element order of the tensor.

A fused kernel must therefore reproduce the node's split, seeds and order **exactly**, even where a different split
would suit the computation better. Both signs are derived from one draw. `dc_perturb_i8` forms `prod = r·m` once and
derives the − noise as `(rounding − prod) >> S`; computing `(-r)*m` is mis-evaluated by the PULP clang-15 build (a
comment at `DualPerturb.h:1-12` records this).

**The order bug fixed in exp7.** The reference perturbs the int8 weight in its ONNX order CHW `[co][ci][ky][kx]`, but
the conv kernels read HWC `[co][ky][kx][ci]`. The block-0 kernels (`DualConv.c`, `NE16DualPrep.c`) perturbed the
already-transposed HWC buffer. That is the same order only for Cin = 1, i.e. block 0. On block 1 it gave 950 of 2,048
wrong W+ values. exp7 passes the weight unperturbed in CHW (`weight_chw = 1`, so the layout pass leaves it
untransposed) and perturbs it in CHW order while writing each element to its HWC position (`dc_perturb_i8_chw`,
`DualPerturb.h:156`).

### 1.4 From one conv to the whole network

The full-network form (exp8) keeps the two passes **channel-stacked** between the convs: a tensor with C channels per
pass becomes one tensor with 2·C channels, [+ | −].

Why channels and not the batch dimension: BatchNorm's γ and β are themselves perturbed, so they differ per pass. With
channel stacking that becomes one BatchNorm with 2·C parameters ([γ+; γ−], [β+; β−]). Batch stacking would need
per-sample BN parameters, which no kernel offers. The other non-conv nodes need no change:
- Quant and Dequant have one scale;
- MaxPool, ReLU and the global pool work per channel.

```
unfused (graph run twice, sign = runtime flag)       fused (graph run once)
x -Quant-Conv(W±εz)-Dequant-BN(γ±,β±)-...            x -Quant- RequantizedDualConv(W, z) -> [A+ | A-]   (2·C channels)
                                                          |  Dequant / BN([γ+;γ-],[β+;β-]) / MaxPool / ReLU on 2·C
                                                          v
                                                     RequantizedDualConv(stacked input) ... (blocks 1-4)
                                                          v
                                                     GlobalAveragePool [1, 2·C] -> Reshape
                                                          |- DualHalf(0) -> Gemm(W+εz, b+εz) -> SCE -> loss        (output 0)
                                                          '- DualHalf(1) -> Gemm(W-εz, b-εz) -> SCE -> loss_minus  (output 1)
```

The classifier and the loss are split back into two chains after the global pool. They are 0.3 % of the step, so a
dual GEMM was not worth building.

---

## 2. How the work progressed

The work went block by block on single-node fixtures first, then into the full graph. Every row was checked exactly
before the next started. "Fixture" means a single-block test network whose reference comes from the unfused
computation through the Onnx4Deeploy oracle.

| stage | what was built | result (both passes, exact) | evidence |
|---|---|---|---|
| exp5 (guide 04) | block-0 fused cluster kernel `DualConv.c` | Siracusa 1.88× (fixture) | `dual_pass_fusion/exp5_fused_dual_conv/` |
| exp6 (guide 04) | block-0 fused NE16 (cluster im2col + one NE16 1×1 job) | GAP9 fixture 2.71× over 2× pulp-nn; node level 4.73× (§6.6) | `dual_pass_fusion/exp6_NE16_fusion/` |
| exp7 step 0 | unfused block-1 baseline fixture `QZO_Conv_B1` | one pass 469,934 cycles, conv at 6.29 MAC/cycle | `exp7_remaining_blocks/exp7a_block1/block1_unfused/results/summary.txt` |
| exp7 step 2a | single-pass control kernel Conv4x2 (pulp-nn's 4×2 register blocking + a flat pixel split, interior pixels read in place) | block 1 conv node 401,397 → 250,179 cycles; fixture 1.44× over pulp-nn, bit-identical | `exp7_remaining_blocks/exp7a_block1/block1_mid/results/summary.txt` |
| exp7 steps 3+4 | fused two-input kernel `DualConv2In.c` + Deeploy integration | block 1 fixture A 939,868 → B 653,328 → C 580,248 (1.62×) | `exp7_remaining_blocks/exp7a_block1/block1_fused_cluster/results/summary.txt` |
| exp7 step 6 | blocks 2–4 | blocks 1–4 2.01× (core-0 counters, step-3 kernel) | `exp7_remaining_blocks/results_step6.txt` |
| exp7 step 5 | kernel optimisation (faster CHW perturbation, channel split, word-wise gather), wall-clock metric | Siracusa blocks 1–4 **2.44×** | `exp7_remaining_blocks/results_step5.txt` |
| exp7 step 7 | GAP9: cluster and NE16 variants for each block; pipelined `NE16DualConv2in` | best per block **2.57×** (fixture totals), engine choice NE16 for blocks 1–2, cluster for 3–4 | `exp7_remaining_blocks/results_step7_gap9.txt` |
| exp8 steps 0–5 | `ZODualPassFusion` pass, `PerturbRademacherFixed`, `DualHalf`, stacked-input conv, harness | Siracusa, 4 steps identical to unfused | `exp8_whole_network/logs/step5_*`, `step8_siracusa_*` |
| exp8 steps 4, 6 | GAP9 cluster and GAP9 + NE16 (blocks 0–2) | 4 steps identical to unfused | `exp8_whole_network/logs/step4_*`, `step6_*` |
| exp8 step 8 | per-block network numbers vs fixtures; de-interleave fix in `DualConv2In.c` | Siracusa **1.53×**, GAP9 cluster **1.42×**, GAP9 NE16 **1.78×** | `exp8_whole_network/results/step8_per_block_vs_fixtures.txt` |
| exp8 step 9 | why block 0 is 4.74× in the network but 2.71× in exp6 | the exp6 number included the fixture's NCHW transposes | `exp8_whole_network/results/step9_block0_gap9.txt` |
| exp8 step 10 | GAP9 + NE16, n_accum 4, one update step | 8,855,589 cycles; round 38.8 s | `exp8_whole_network/results/step10_round_prediction.txt` |
| exp8 step 11 | the same step with ECO | 9,129,644 cycles; round 40.0 s | `exp8_whole_network/results/step11_eco_round_prediction.txt` |
| (Run, `eabbb0b`) | NE16 silicon fix (job width multiple of 3); `FoldDequantIntoFrozenBNPass` | fused arena 1,317,012 → 854,452 B; fused QZO+ECO fits the board L2; board rounds 35.1 s / 36.1 s | §9 item 3 |
| exp9 | `FuseFrozenBNTailPass` + `PULP_FrozenBNPoolReluQuantHWC_i8_i8`; `ZODualPassFusion` phantom-consumer fix | fused QZO+ECO arena 854,452 → **265,612 B**, train 7.54 M → 6.56 M cycles, bit-exact | `exp9_memory_optim/logs/`, `verify/`, `memcompare/summary.txt` (§6.10) |

### 2.1 Decisions and why

| decision | alternatives considered | reason |
|---|---|---|
| One generic two-input kernel for blocks 1–4; keep block 0's kernel | one kernel per block; generalise `DualConv.c` | block 0's shared-input structure is a different kernel; one kernel keeps the code small; only a run-time split choice survived as specialisation |
| Same op `RequantizedDualConv`; x− appended as input 6 (7 inputs) | insert x− at index 1; new op | code that assumes `inputs[0]` = activation and `inputs[1]` = weight (layout pass, tiler, parser helpers) keeps working; the input count selects the binding (6 = block 0, 7 = blocks 1–4) |
| Weight passed raw CHW and unperturbed (`weight_chw = 1`), perturbed inside in CHW order, written HWC | perturb the transposed weight (wrong for Cin > 1); keep the perturbation node | reproduces the reference noise order for any Cin and removes 3 nodes per conv per pass |
| Output channel-stacked [A+ \| A−] | two outputs | reuses block 0's output handling; the full network wants channel stacking anyway (BN) |
| Kernel takes pointers + pixel strides (`pInP, pInM, in_stride; pOutP, pOutM, out_stride`) | a fixed layout | a channel-stacked input became a template change in exp8, not a kernel rewrite |
| HWC for all blocks, also the 7×1 kernels of blocks 3–4 | CHW for blocks 3–4 | `sdotsp4` (4 int8 MACs per instruction) needs the reduction axis (ci, ky, kx) contiguous; in HWC a receptive field is Kh runs of Kw·Ci bytes (block 1 one 128 B run, block 3 seven 16 B runs) |
| A single-pass control kernel (Conv4x2) before fusing | fuse directly | separates "is our inner loop as good as pulp-nn" from the fusion gain; exp5a's FlatConv had been 1.54× slower than pulp-nn on block 1 |
| Phase-2 core split by shape: pixels, or output channels when the pixel split leaves cores idle | always pixels | block 4 has 10 output pixels (5 of 8 cores busy); a naive cost model also chose channels for block 3, measured wrong (pixels 37.6 k vs channels 40.0 k) |
| NE16 via the GAP9 SDK's im2col route (cluster builds a column, NE16 runs a 1×1 job) | per-tap streamin (exp16) | per-tap was 4.3–7.5× slower than the cluster and refuses padded convs; Cin·K = 128 / 128 / 112 / 224 fills NE16's 16-channel input tile |
| NE16 node with a word-wise column, 3×w jobs, double-buffered columns and a 2-deep job queue | build the whole column first (exp6) | exp6 spent 26 % building the column before any NE16 work; overlapping hides it |
| Engine per block on GAP9: NE16 for blocks 0–2, cluster for 3–4 | NE16 everywhere; split one conv across both engines | measured (§6.3); NE16's per-step weight preparation does not amortise on blocks 3–4 |
| Metric: wall clock per node (`--profileTiling`); fused node vs the unfused nodes it replaces (2 × [perturb W + perturb b + weight transpose + conv]) | core-0 counters; whole-fixture runtime | core-0 counters stop while core 0 waits in a barrier and hide load imbalance; the fixture's NCHW boundary transposes do not exist in the network |
| GAP9 kernel fixtures at `--l1 86000` | exp6's 92000 | on the current branch the GAP9 kernel harness keeps the cluster stacks in L1, and 92000 fails to allocate |
| New kernel sources on the GAP9 library's `-O3` list | SDK default `-Os` | under `-Os` the fused cluster kernel on block 1 was 914 k instead of 425 k cycles (node; GAP9 fixture 991,489 vs 502,274) |
| Whole network: a Deeploy rewrite pass on the existing training graph, opt-in by flag | export a new dual graph from Onnx4Deeploy | the exported graph stays the semantic reference; existing fixtures and their L+/L− references check the result; the update graph is untouched; without the flag nothing changes |
| Rewrite rules per node type; unknown node types with a stacked input raise | pattern-match whole blocks | generic, and fails loudly instead of producing a wrong graph |
| Every perturbation keeps its `seed / idx / eps / div`; the seed uses the `idx` attribute | renumber nodes | each half is bit-identical to what the original node writes in that pass, so the update graph's regenerated z stays valid |
| FC tail via a new `DualHalf` node | Deeploy `Slice`; zero-padded FC weights | tiled `Slice` mis-strides; zero padding was exact on Siracusa but not on GAP9 (§8) |
| De-interleave the stacked input tile once inside `DualConv2In` (1×K kernels) | per-tap strided reads; change the upstream layout | block 1 has 8 channels per pass = 2 words per tap, so per-tap reads pay the loop overhead every 2 words; one copy per tile restores the in-place path |
| ECO momentum sized per element | `inputs_bytes × 4` (Run's original) | 4× over-allocation for int32 / fp32 tensors no longer fitted next to the fused graph's larger L2 arena |
| exp9: fuse the BN tail as computation only, each step with the expression of the kernel it replaces | fold BN into the conv requant (fewer ops, but different fp32 arithmetic) | stays bit-exact against the unfused graph and the existing CI hashes; the memory gain does not need the arithmetic change |
| exp9: pass at the end of lowering, run again in the NE16 deployer | fuse in `ZODualPassFusion`; fuse before lowering | the chain only exists after lowering (Run's channels-last BN and Dequant fold, the layout transposes); NE16's own transposes are added after the PULP lowering chain has run |
| exp9: only frozen-stat BN with a folded Dequant (`DEEPLOY_BN_FROZEN_NHWC=1`, `dq_scale` present), zero points 0, 8 bit | a general BN tail | the only configuration the board uses; zero points 0 keep the fused `x·s + zp` rounding identical to the unfused code |
| exp9: output pixels split across cores, channel loop outside | channels across cores | same order as the BN kernel, so the per-channel constants stay in registers |

---

## 3. Part A — the per-block kernels (exp7)

### 3.1 Block shapes

| block | input (C, H, W) | weight | kernel, pads | output per pass | MACs per pass | weights | MACs per weight (= output pixels) |
|---|---|---|---|---|--:|--:|--:|
| 0 | 1, 14, 700 | 8×1×1×4 | 1×4, (0,2) | 8, 14, 701 | 314 k | 32 | 9,814 |
| 1 | 8, 14, 87 | 16×8×1×16 | 1×16, (0,8) | 16, 14, 88 | 2.52 M | 2,048 | 1,232 |
| 2 | 16, 14, 22 | 16×16×1×8 | 1×8, (0,4) | 16, 14, 23 | 659 k | 2,048 | 322 |
| 3 | 16, 14, 5 | 32×16×7×1 | 7×1, none | 32, 8, 5 | 143 k | 3,584 | 40 |
| 4 | 32, 8, 5 | 32×32×7×1 | 7×1, none | 32, 2, 5 | 72 k | 7,168 | 10 |

MACs per pass = Cout × H_out × W_out × Cin × Kh × Kw. For block 1: W_out = (87 + 8 + 8 − 16) + 1 = 88, so
16 × 14 × 88 × 8 × 1 × 16 = 2,523,136. The last column explains most of the engine decision (§6.3): NE16's fixed
per-step cost grows with the number of weights, while its advantage grows with the MACs per weight.

### 3.2 The fused cluster kernel `PULP_DualConv2d_2in_i8_i8_i8_HWC`

File `TrainDeeploy/TargetLibraries/PULPOpen/src/DualConv2In.c` (entry `:114`; the header comment `:1-37` describes
layouts and scratch). The kernel computes `A+ = requant(conv(x+, W+), add+)` and `A− = requant(conv(x−, W−), add−)` in
two phases separated by one cluster barrier.

**Phase 1** (`:147`): each core perturbs **its `dc_chunk` chunk** of the unperturbed raw-CHW weight, both signs, and
writes W+ / W− at their HWC positions:
- `dc_perturb_i8_chw` (`DualPerturb.h:156`) walks the chunk in CHW memory order, so noise bit i lands on CHW element i.
  It writes HWC index `j = co·K + (ky·Kw + kx)·Ci + ci`.
- Because z = ±1, a channel has only two noise values, `up = (m + rnd) >> S` and `dn = (rnd − m) >> S`. The noise bit
  selects between them without branches. The function is out of line, because inlined it reloaded its pointers from
  the stack for every element.
- The bias uses `dc_perturb_i32` (`DualPerturb.h:222`), with the same seed formula as the `RQSPerturbRademacher` node
  it replaces.
- Cost: about 26 cycles per element for both signs, against about 141 cycles per element per sign in the standalone
  node.

**Phase 2** (`:184`): Conv4x2's loop structure (`TrainDeeploy/TargetLibraries/PULPOpen/src/Conv4x2.c:75`), run once per
pass from **one copy** of the loop code:
- a flat range of output pixels per core, two pixels at a time;
- 4 output channels × 2 pixels per inner loop (8 accumulators, 6 loads feed 8 `sdotsp4`);
- requant `D2_RQ` exactly as `pulp_nn_bn_quant_i8` (`:48`).

Receptive fields:
- interior fields are read in place (`in_place`, `:187`);
- fields that touch the padding are gathered word-wise into a patch with zeros (`d2_gather`, `:53`, out of line).

The core split is chosen at run time (`:198-237`): output channels instead of pixels only if the pixel split leaves
cores idle **and** the channel split keeps more of them busy. `-D DUALCONV2IN_SPLIT=1|2` forces one split for
measurement.

What the two passes share: the pixel loop, the padding test and the addressing (same positions in x+ and x−). What they
cannot share: operands. x+·W+ and x−·W− have nothing in common, so phase 2 costs the same as two single passes. On
blocks 1–4 the fusion gain is the removed nodes (perturbation at ~141 cycles/element/sign, weight transposes) plus
the shared scaffolding.

**Scratch** (one L1 transient buffer). Its layout must match `DualConv2InTemplate.computeTransientBuffersSize`
(`TrainDeeploy/Deeploy/Targets/PULPOpen/Templates/DualConv2InTemplate.py:36`):

```
W+ (HWC) | W− (HWC) | add+ | add− | four K-byte patches per core | (stacked 1xK input only) de-interleaved x+ / x− tile
```

Preconditions: K = Ci·Kh·Kw and Kw·Ci are multiples of 4 (true for all SpeechNet blocks).

### 3.3 Deeploy integration of the cluster kernel

Every change is marked `-- QW` in the code.

| piece | location | what it does |
|---|---|---|
| parser | `TrainDeeploy/Deeploy/Targets/PULPOpen/Parsers.py:828` `PULPDualConv2D2InParser` | 7 inputs `[x+, W, mul, add, wmul, bmul, x−]`, `weight_chw = 1`; the block-0 parser `PULPDualConv2DParser` (`:715`) now requires 6 inputs |
| template | `Templates/DualConv2InTemplate.py:62` `referenceTemplate` | seeds = block 0's formula; stacked output (`pOutP = out`, `pOutM = out + Co`, stride 2·Co) |
| tile constraint | `TrainDeeploy/Deeploy/Targets/PULPOpen/TileConstraints/DualConv2InTileConstraint.py` | x− gets exactly x+'s tile; weight dims in CHW order; output channels never tiled (the perturbation needs the whole weight) |
| layout pass | `TrainDeeploy/Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/LoweringOptimizationPasses.py:326` | a `weight_chw` weight is not transposed (like `ne16_weight_preencoded`); x− is transposed like any activation |
| mappers | `TrainDeeploy/Deeploy/Targets/PULPOpen/Platform.py:178-183, :204`; `TrainDeeploy/Deeploy/Targets/GAP9/Platform.py:193-198, :334` | `RequantizedDualConv` → [block-0 mapper, 7-input mapper, stacked mapper] on both platforms |
| control-kernel switch | `TrainDeeploy/Deeploy/Targets/PULPOpen/Platform.py:104-115` | `DEEPLOY_RQCONV_KERNEL=pulpnn\|flat\|4x2` selects the kernel of every int8 `RequantizedConv` without editing code; default `pulpnn` |
| GAP9 build | `TrainDeeploy/TargetLibraries/GAP9/CMakeLists.txt:63-68` | `DualConv2In.c`, `Conv4x2.c`, `NE16DualPrep.c`, `NE16DualConv2in.c` compiled at `-O3` |

### 3.4 The fused NE16 node `NE16DualConv2in` (GAP9)

**Starting point.** Block 0 (exp6) fuses both passes into **one** NE16 job over a shared im2col column. From block 1 on,
the passes have different inputs, so there are two columns and two jobs. What the passes still share is the weight
preparation: one Rademacher draw produces both W+ and W−.

The exp6-style first version used two chains (column + job per pass). It reached only 0.82× of 2× pulp-nn on block 1.
The NE16 job itself was 3× faster than pulp-nn, but the byte-wise column builder cost 291 k cycles per pass.

**The current node** (`TrainDeeploy/Deeploy/Targets/NE16/DualConv2in.py`, kernel
`TrainDeeploy/TargetLibraries/GAP9/src/NE16DualConv2in.c:170` `PULP_NE16DualConv2in`) runs on the cluster master and
dispatches the NE16 jobs itself (`Deeploy/Targets/NE16/Engine.py:35-44`, `:143`):
- **Word-wise HWC column** (`n16d_build_col`, `:52`). x± are transposed to HWC. Per receptive-field row the valid taps
  are one contiguous run, copied as words with `^ 0x80808080` (signed → unsigned input offset); padding is
  `0x80808080`. The column order is (ky, kx, ci), and the weight is encoded in the same order
  (`NE16DualPerturbEncode` with `ne16_hwc = 1`).
- **Job geometry** (`n16d_job_geometry`, `:152`). One job = 3 × w consecutive output pixels of one pass, presented to
  NE16 as a 3×w block. One-row jobs had left two of NE16's three 3×3-subtile rows idle. Jobs are sized from an 11 kB
  per-buffer budget.
- **Double buffering + 2-deep queue.** The master dispatches job j, waits for a free queue slot (meaning job j−1 is
  done and its buffer is free), and forks the workers to build job j+1's column while NE16 runs job j.
  `ne16_pipeline = 0` runs the same jobs sequentially (for measurement).
- **Tiling.** Over the output height only (`TrainDeeploy/Deeploy/Targets/NE16/TileConstraints/NE16DualConv2inConstraint.py`).

**Preparation nodes** (`TrainDeeploy/TargetLibraries/GAP9/src/NE16DualPrep.c`):
- `PULP_NE16DualPerturbEncode` (`:40`): perturb + bit-serial encode of the weight;
- `PULP_NE16DualBias` (`:80`): requant bias of each pass.

Both have `ne16_sign_sel`: 0 = stacked [W+; W−] (block 0), 1 = W+ only, 2 = W− only. They use the same seeds and
chunking as the reference.

**No weight-order fix was needed on NE16.** The column is gathered in (ci, ky, kx) / (ky, kx, ci) order, and the
encoder folds the raw CHW weight in the matching order.

**Step-by-step on block 1** (fixture `Runtime`, logs in `exp7_remaining_blocks/exp7a_block1/block1_fused_NE16/logs/`):

| version | cycles | log |
|---|--:|---|
| exp6-style, two chains | 903,653 | `gap9_C_plain.log` |
| one node, word-wise column, one-row jobs, sequential | 574,651 | `gap9_C_S_plain_rowjobs.log` |
| 3×w jobs, sequential | 361,352 | `gap9_C_S_plain.log` |
| 3×w jobs, double buffering + queue (first bias prep) | 283,529 | `gap9_C_P_plain_biasv1.log` |
| the same with the faster bias prep | 277,119 (275,034 in an earlier build) | `gap9_C_P_plain.log`, `results/summary_optimised.txt` |

That is 2.68–2.70× over 2× pulp-nn. NE16 is now the bottleneck: the conv node is 157 k cycles pipelined vs 241 k
sequential (`results/summary_optimised.txt`), so about 84 k of column building is hidden.

### 3.5 Per-block fixtures and builders

All builders are in `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/`.

| builder | fixture | contents |
|---|---|---|
| `build_qzo_conv_fixture.py --block i` | `Tests/Kernels/Integer/QZO_Conv_B<i>` | unfused, **one pass**: perturbation nodes + weight transpose + `Conv`+`RequantShift` (pulp-nn) |
| `build_ne16_im2col_fixture.py --block i` | `QZO_Conv_NE16_B<i>` | the same single pass on NE16 (exp6 route), plus `col_golden.npz` for debugging the column |
| `build_dual_conv2in_fixture.py --block i` | `QZO_DualConv2in_B<i>` | fused cluster: one 7-input `RequantizedDualConv`; `reference_plus_minus.npz` holds A+ and A− separately |
| `build_ne16_dual2in_fixture.py --block i --single-node 1` | `QZO_DualConv2in_NE16P_B<i>` | fused NE16: `NE16DualPerturbEncode` + `NE16DualBias` + `NE16DualConv2in` (`ne16_pipeline = 1`) |

How the fixtures are built:
- Inputs are the real block inputs of the full QZO forward through the Onnx4Deeploy oracle (x+ and x− from the + and
  − passes; 1,136 of 9,744 elements differ on block 1).
- The fused reference is the unfused single-block graph run on x+ and on x− (with negated perturbation multipliers),
  then stacked.
- The activation inputs must be pinned to `int8_t`. Block i's input is ≥ 0 after ReLU/MaxPool, so the generator would
  infer `uint8_t` and pick `pulp_nn_conv_u8_i8_i8`, while the full network uses `pulp_nn_conv_i8_i8_i8` (§7).

---

## 4. Part B — the whole network (exp8)

### 4.1 The rewrite pass `ZODualPassFusion`

File `TrainDeeploy/Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/ZODualPassFusion.py`,
entry `zoDualPassFusion(graph, ne16_convs=None)` (`:54`).

**When it runs.** It is called from `TrainDeeploy/DeeployTest/testMVPTraining.py:99-104`, right after the existing
strip step and before code generation, when `--zo-fused-dual` is given. `--zo-fused-dual-ne16 0,1,2` lists the convs
lowered to the NE16 form. The MeZO runner forwards both flags and adds `-DZO_FUSED_DUAL=ON`
(`TrainDeeploy/DeeployTest/testUtils/deeployMezoRunner.py:100-103, :138-142`).

**What it does.** It walks the graph in topological order and keeps a set `dual` of channel-stacked tensor names:
- the graph input is pass-independent;
- every perturbation output is pass-dependent;
- pass-dependence propagates downstream;
- each node type has one rewrite rule.

| node | condition | rewrite | line |
|---|---|---|---|
| `Conv` + `RequantShift` | weight and requant add come from `RQSPerturbRademacher` | one `RequantizedDualConv` with `[x, W, mul, add, wmul, bmul]`; attrs kernel/pads/strides/dilations/group, `shift = log2(div)`, `seed_w/idx_w/div_w`, `seed_b/idx_b/div_b`, `div/n_levels/signed` from the RQS. If x is already stacked: `weight_chw = 1, stacked_input = 1`. The output gets 2·C channels and joins `dual`. If the conv index is in `ne16_convs`: NE16 form (§4.4) | `:82` |
| `Quant`, `Dequant`, `Relu`, `MaxPool`, `GlobalAveragePool` | input in `dual` | unchanged; output shape 2·C (`_ELEMENTWISE`, `:32`) | `:121` |
| `BatchNormInternal` | input in `dual` | γ, β from `PerturbRademacherFixed` mode 0 ([+; −] stacked); running mean / var constants concatenated with themselves | `:126` |
| `Reshape` | input in `dual` | new shape constant with 2·C | `:141` |
| `Gemm` (classifier) | input in `dual`, feeds the loss | `DualHalf` half 0 and half 1; the original Gemm re-wired to (half 0, W mode 1, b mode 1); a new Gemm (half 1, W mode 2, b mode 2) and a new `SoftmaxCrossEntropyLoss` → `loss_minus`; `graph.outputs = [loss, loss_minus]` | `:149` |
| anything else with a `dual` input | — | `NotImplementedError` | `:180` |

The pass does not touch the update graph. The exported ONNX file is not modified. Without `--zo-fused-dual` the
build is exactly the old one.

### 4.2 `PerturbRademacherFixed` — fp32 perturbation with a codegen-time sign

The original `PerturbRademacher` reads the runtime `perturbation_sign`. In the dual graph both signs are needed in one
execution. The new node's mode fixes the sign at code-generation time:
- **0** → `[w + εz ; w − εz]` (2·size; BN γ/β);
- **1** → `w + εz`;
- **2** → `w − εz` (classifier W/b).

The template (`TrainDeeploy/Deeploy/Targets/PULPOpen/Templates/FloatPerturbRademacherFixedTemplate.py:53`) uses the same
chunking, the same seed expression (with `node_id` = the node's `idx` attribute), the same eps and the same kernel
`ApplyRademacherPerturbation` as `FloatPerturbRademacherTemplate`. Only the sign is a constant, so each half is
bit-identical to the original node in that pass.

Other pieces:
- parser `PULPOpen/Parsers.py:799` `PerturbRademacherFixedParser`;
- untiled tile constraint in `TileConstraints/UntiledTileConstraint.py`;
- mappers `PULPOpen/Platform.py:161-163, :257` and `GAP9/Platform.py:164-166`.

The template always hoists a 4 B transient buffer, because the tiler looks up whatever `computeTransientBuffersSize`
names (§8). (A `pad_cols` path left over from the zero-padded classifier workaround was removed on 2026-10-14; it
used the strided per-core loop of §8 bug 4.)

### 4.3 `DualHalf` — the classifier input split

- What it computes: `out[r, c] = in[r, half·C + c]` for a stacked `[rows, 2·C]` input
  (`TrainDeeploy/Deeploy/Targets/PULPOpen/Templates/DualHalfTemplate.py:10`).
- Parallelisation: each core copies one **contiguous** chunk of columns (§8, bug 4).
- Why it matters: each pass's Gemm then sees exactly the unfused `[1, C]` input, which is required for bit-exactness
  on GAP9 (§8, bug 3).
- Other pieces: parser `PULPOpen/Parsers.py:778` `DualHalfParser`; untiled tile constraint; mappers
  `PULPOpen/Platform.py:164-166, :258` and `GAP9/Platform.py:167-169, :324`.

### 4.4 Stacked-input convolutions (blocks 1–4)

**Cluster form** (`RequantizedDualConv` with `stacked_input = 1`):
- **Parser** `PULPDualConv2DStackedParser` (`PULPOpen/Parsers.py:870`): 6 inputs; stores `ch_im_in` = 2·Ci,
  `in_ch` = Ci per pass, `in_stride` = 2·Ci.
- **Template** `stackedTemplate` (`DualConv2InTemplate.py:78`): x− = `data_in + in_ch`, pixel stride `in_stride`. The
  kernel interface from §3.2 is used unchanged.
- **Tile constraint**: drops x− for the stacked form; the weight's channel dim must equal `in_ch`; the input tile keeps
  all 2·Ci channels.
- **De-interleave** (`DualConv2In.c:157-177`, before phase 1's barrier at `:178`). With a stacked input and 1×K kernel rows, a row's taps are 2·Ci apart,
  so they are not contiguous, and every receptive field would be gathered. The kernel therefore copies the tile once
  into contiguous x+ / x− scratch and then uses the in-place path. The copy shares phase 1's barrier. The template
  reserves the input size in the scratch (`DualConv2InTemplate.py:36`, `stacked_input` and Kw ≠ 1).
- **Effect:** block 1 kernel fixture `QZO_DualConv2in_stacked_B1` 607,791 → 533,330 cycles, equal to the two-input
  kernel's 534,483. The 7×1 kernels of blocks 3–4 have one tap per row and were already read in place.

**NE16 form** (GAP9, `--zo-fused-dual-ne16`), built by `_ne16_dual_conv` (`ZODualPassFusion.py:192`):
- **Block 0** (input not stacked) uses exp6's form: `NE16Im2ColU8` + `NE16DualPerturbEncode` (sel 0) +
  `NE16DualBias` + a 1×1 `Conv` (`ne16_weight_preencoded = 1`, `weight_offset = −128`) + `RequantShift` with the
  multiplier duplicated for 2·Co. That is one NE16 job for both passes over the shared column.
- **Stacked convs** use a `Transpose` to HWC + encode (`ne16_hwc = 1`) + bias + **`NE16DualConv2in`**
  (`stacked_input = 1`, `ne16_pipeline = 1`). That is exp7's pipelined node; its column builder gained `in_stride`
  (`NE16DualConv2in.c:47, :85-93`). The stacked parser and template are in `Deeploy/Targets/NE16/DualConv2in.py:95, :161`.
- Each NE16 form ends with a `Transpose` back to the NCHW stacked tensor, so the rest of the graph is unchanged. In
  the GAP9 profile no separate output-transpose node appears for these convs (observed, not traced).

### 4.5 Harness and runner

**Harness.** `TrainDeeploy/DeeployTest/Platforms/Siracusa/src/deeploymezotest.c:392-427` and
`TrainDeeploy/DeeployTest/Platforms/GAP9/src/deeploymezotest.c:698-732`.

The unfused harness runs the training graph twice per mini-batch:
- with `perturbation_sign = 1`, reading L+ from output 0;
- with `perturbation_sign = 0`, reading L− from output 0.

Under `#ifdef ZO_FUSED_DUAL` the first run already computed both passes. L− is read from `DeeployNetwork_outputs[1]`
(memcpy if in L2, `ram_read` if in L3), and the second run is skipped. Everything after is unchanged: the
accumulation of L+ − L−, `g_proj`, the update graph, the weight dumps and `[WSUM]` / `[WCHK]`.

CMake: `if(ZO_FUSED_DUAL)` adds the define (`Platforms/Siracusa/CMakeLists.txt:85-86`,
`Platforms/GAP9/CMakeLists.txt:120-121`).

**Runner flags** (`deeployMezoRunner_tiled_{siracusa,GAP9,GAP9_w_NE16}.py`):

| flag | effect |
|---|---|
| `--zo-fused-dual` | run the rewrite pass + `-DZO_FUSED_DUAL=ON` |
| `--zo-fused-dual-ne16 0,1,2` | conv indices lowered to the NE16 form (only with the `_w_NE16` runner) |
| `-D BN_FROZEN_STATS=ON` | required for our graphs. With it, the runner also sets `DEEPLOY_BN_FROZEN_NHWC=1` (`deeployMezoRunner.py:130-133`), which runs frozen-stat BN channels-last (Run's optimisation) |
| `-D GAP9_NETWORK_O3=ON` | compiles the generated GAP9 network at `-O3` (`TrainDeeploy/DeeployTest/CMakeLists.txt:141`) |
| `-D ZO_ECO=ON` | Run's ECO update; `-D ZO_ECO_SR=ON` for stochastic rounding, `ZO_ECO_BETA` default 0.9 |
| `-D DUMP_WEIGHTS=ON -D DUMP_STEP_LO=0 -D DUMP_STEP_HI=3` | per-step weight dumps for `compare_dumps.py` |
| `--profileTiling` | per-node wall-clock profile (the metric of §6) |

### 4.6 ECO on the fused graph

ECO is Run's implementation (TrainDeeploy commit `886c948`: `ZOEco.c`, `ZORuntime.c/.h`, hooks in the perturbation
templates used by the update graph, `-D ZO_ECO=ON`). It is memory-free error feedback through a momentum buffer per
trainable tensor, and it acts only on the update graph. The fusion acts only on the training graph. So the two compose
without code changes.

One harness change was needed (`Platforms/GAP9/src/deeploymezotest.c:606-625`):
- **Before:** the momentum was allocated as `inputs_bytes[k] × 4` bytes, an upper bound that over-allocates int32 and
  fp32 tensors 4× (69,264 B for SpeechNet).
- **Why it broke:** the fused NE16 graph's L2 arena (`pi_l2_malloc` of 1,325,460 B in the generated
  `TrainingNetwork.c`) left too little heap, and the run stopped with `[ECO] momentum alloc failed (k=19)`.
- **Now:** one fp32 per element, `inputs_elems[k] × 4` = 61,956 B. Run's line is kept as a comment.
  `inputs_elems` is a new array that `TrainDeeploy/Deeploy/DeeployTypes.py:2957` emits next to `inputs_bytes`.
- **Why it is safe:** the update node's `idx` equals the optimizer input index for all 22 SpeechNet tensors (checked).

Training-graph cycles barely change with ECO (8,612,666 → 8,621,038; the difference is the run-to-run spread of the
tiler's placement). The update graph grows from 242,923 to 508,606 cycles.

### 4.7 Lowered graphs kept for reference

`TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp8_whole_network/graphs/` holds the Deeploy middleware graphs
before and after lowering. Open them in Netron to see exactly what the pass produced:

| file | configuration |
|---|---|
| `dual_pass_train_frontend.onnx` | the dual graph straight after the pass |
| `siracusa_unfused_{train,update}_middleware_post_lowering.onnx` | baseline |
| `siracusa_fused_train_middleware_{pre,post}_lowering.onnx` | Siracusa fused |
| `gap9_fused_cluster_train_middleware_{pre,post}_lowering.onnx` | GAP9, all convs on the cluster |
| `gap9_fused_ne16_b012_train_middleware_{pre,post}_lowering.onnx` | GAP9, blocks 0–2 on NE16 |

exp9 adds `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp9_memory_optim/graphs/`, both GAP9 + NE16
(blocks 0–2), fused QZO+ECO:

| file | configuration |
|---|---|
| `before_fused_ne16_b012_train_post_lowering.onnx` | Run's `fe1ffa2` (Dequant folded into BN) |
| `after_fused_ne16_b012_train_post_lowering.onnx` | with `FuseFrozenBNTailPass`: one `FrozenBNPoolReluQuant` per block 0–3 |

### 4.8 Memory: the frozen-BN tail fusion (exp9)

**The problem.** After Run's `FoldDequantIntoFrozenBNPass` (`eabbb0b`), every frozen-stat QZO block reads, after
lowering:

```
BN(int8 HWC -> fp32 HWC) -> [MaxPool HWC] -> Transpose(0,3,1,2) -> Relu -> Quant(int8) -> Transpose(0,2,3,1)
```

The input and output are int8 HWC, but up to four fp32 tensors lie in between. In block 0 the fp32 BN output of the
dual pass is 628,096 B, i.e. 73 % of the 857,732 B L2 peak. Every step is elementwise or a max over a window, so one
kernel can compute the chain per output element in registers, and the fp32 tensors never exist.

**The pass** `FuseFrozenBNTailPass` (`TrainDeeploy/Deeploy/Targets/PULPOpen/FrozenBNTail.py:97`):
- **When it runs:** at the end of lowering:
  - in the PULP deployer (`Deeploy/Targets/PULPOpen/Deployer.py:71`);
  - again in the NE16 deployer (`Deeploy/Targets/NE16/Deployer.py:61`), because NE16's own transposes are added
    after the PULP chain has run.
- **When it fires:** only if `DEEPLOY_BN_FROZEN_NHWC=1`, which the MeZO runner sets for `-D BN_FROZEN_STATS=ON`
  (§4.5).
- **Opt-out:** `DEEPLOY_FUSE_BN_TAIL=0` turns it off, for A/B runs.
- **Matcher** (`_match_tail`, `:61`), conditions:
  - the BN has `bn_nhwc = 1` and a folded Dequant (`dq_scale`);
  - its four statistics outputs are unused;
  - each link has exactly one consumer;
  - the MaxPool has stride = kernel, no padding and no dilation;
  - the transposes have exactly the perms shown above;
  - both zero points are 0 and the Quant is 8 bit.
- **Rewrite:** the chain becomes one node `FrozenBNPoolReluQuant(x, γ, β, mean, var) -> y [1, Ho, Wo, C] int8`.
- **Coverage:**
  - blocks 0–2 match with MaxPool, block 3 without;
  - block 4 (Relu → GlobalAveragePool, no Quant) does not match and is left unchanged.

**The kernel** `PULP_FrozenBNPoolReluQuantHWC_i8_i8` (`TrainDeeploy/TargetLibraries/PULPOpen/src/BatchNorm.c:502`). It
lives in the same file as the BN kernel, so it gets the same compile flags (the strict-fp32 list on Siracusa, `-O3` on
GAP9).
- **Bit-exactness:** every step uses the exact expression of the kernel it replaces:

  | step | expression | replaced kernel |
  |---|---|---|
  | BN | `dq = ((float)q − dq_zp)·dq_scale`, then `(dq − mean)·inv_std·γ + β` | `PULP_BatchNormInternalHWC_i8_fp32` |
  | pool | `fmax.s` over the window | `PULP_MaxPool2d_fp32_fp32_HWC` |
  | ReLU | `MAX(m, 0)` | `PULP_Relu_fp32_fp32` |
  | Quant | round half away from zero, then clamp | `PULPQuantTemplate` |

  - Empty `asm volatile` barriers after `dq` and after the BN result keep each a rounded fp32 value, as it was when
    it was stored. This stops clang from contracting across the former node boundaries.
  - The two transposes only moved data, so dropping them changes nothing.
- **Parallelisation:** cores split the output pixels; the channel loop is outermost, as in the BN kernel, so that γ, β,
  mean and `inv_std` stay in registers.
- **Tiling** (`FrozenBNPoolReluQuantTileConstraint`, `FrozenBNTail.py:218`):
  - only the H dimension is tiled; each input tile is `kh` × the output tile's rows, at full W and C;
  - the four per-channel vectors are loaded whole.
- **Registration:**
  - mappers in `PULPOpen/Platform.py` and `GAP9/Platform.py`;
  - bindings from `tilingReadyBindings()` (`FrozenBNTail.py:260`);
  - prototype in `PULPOpen/inc/kernel/BatchNorm.h` and `GAP9/inc/kernel/GAP9Kernels.h`.

**A fix in `ZODualPassFusion`** (`ZODualPassFusion.py:184`). The nodes the pass replaces (the original
`Conv` / `RequantShift`) were removed from `graph.nodes`, but onnx-graphsurgeon keeps a tensor's consumer list in sync
only with `node.inputs`.
- **Effect:** the removed nodes stayed in their input tensors' consumer lists as phantom consumers through lowering.
  `cleanup()` could not see them, but they made single-consumer matches fail.
- **Fix:** the pass now clears their inputs and outputs.
- **Only visible change in the lowered graph:** three per-channel `mul` constants are no longer duplicated
  (`..._DUPLICATE_FOR_...`); the values are the same.
- **Extra guard:** `FrozenBNTail._only_consumer` (`:48`) skips consumers that have no outputs.

**Where it fires and where it does not:**

| configuration | fires? |
|---|---|
| QZO / QZO+ECO, fused or unfused, MeZO runner with `-D BN_FROZEN_STATS=ON`; the fused pytest gates | yes |
| ZO fp32 (no Dequant, so no `dq_scale`) | no |
| BP (the flag is never set, no Quant) | no |
| unfused pytest MeZO gates (they do not set the flag) | no |

---

## 5. Fixtures in `TrainDeeploy/SpeechNet_Test/`

All per-block fixtures are byte-identical copies of the `Tests/Kernels/Integer/` fixtures that exp7 measured (copy
script: `SpeechNet_Test/scripts/export_fixtures.sh kernels_b1to4`).

| fixture | original | what it measures | runner |
|---|---|---|---|
| `speechnet_unfused_b<i>_cluster` (i = 1..4) | `QZO_Conv_B<i>` | **one pass** of the unfused QZO conv (perturbation nodes + transpose + pulp-nn); ×2 for both passes = baseline A | `deeployRunner_tiled_siracusa.py` / `deeployRunner_tiled_gap9.py` |
| `speechnet_unfused_b<i>_NE16` | `QZO_Conv_NE16_B<i>` | one pass on NE16 via the im2col route (B) | `deeployRunner_tiled_gap9_w_ne16.py` |
| `speechnet_fused_b<i>_cluster` | `QZO_DualConv2in_B<i>` | both passes, fused cluster kernel (C_cl) | `deeployRunner_tiled_siracusa.py` / `deeployRunner_tiled_gap9.py` |
| `speechnet_fused_b<i>_NE16` | `QZO_DualConv2in_NE16P_B<i>` | both passes, pipelined fused NE16 node (C_P), the form the network uses | `deeployRunner_tiled_gap9_w_ne16.py` |

The block-0 fixtures `speechnet_{unfused,fused}_b0_{cluster,NE16}` were already there (guide 04).

**Whole-network entry fixtures.** These are the **ordinary, unfused** QZO export (`-mode q-zo-train`, pooled@99.9
calibration). TrainDeeploy fuses them at compile time with `--zo-fused-dual`. Both train graphs and both update graphs
are byte-identical to `speechnet_{train,update}_QZO_single_step`; only the windows and the host reference differ.

| fixture pair | export | windows | host reference | use |
|---|---|---|---|---|
| `speechnet_{train,update}_QZOECO_full_fused_single_update_step` | `--n-steps 1 --n-accum 4` | 4 (the first 4 of the 54-window pool) | L+ / L− of the 4 mini-batches, one update | one update step with n_accum 4, the step the round prediction is built on; valid for QZO and QZO+ECO (ECO acts after these 4 forwards) |
| `speechnet_{train,update}_QZOECO_full_fused_round1` | `--n-steps 14 --n-accum 4` | all 54 (14 steps = 56 forwards wrap around the pool) | L+ / L− of the first 56 mini-batches (14 updates) | a round: run 1,620 steps (120 epochs × 54 windows / 4). Direct QZO can be checked on its first 14 steps (112 loss values); a QZO+ECO run diverges from the direct reference after the first update, so only its first step is checkable, and the round is judged by accuracy |

**Why the round fixture has only a 14-step reference.** The host reference for a full round (1,620 steps = 6,480
forward pairs) takes hours. Beyond the first update it is also a direct-QZO trajectory, which an ECO run does not
follow (the multi-step host reference cannot simulate ECO; Onnx4Deeploy's `--qzo-optimizer eco` refuses multi-step
references). This is the same recipe as Run's GAP9 board rounds (`silentwear-zo/rounds/export_qzo999_rounds.sh`).

The device cycles through the pool with window index `mb % TRAINING_DATA_SIZE` (`Platforms/GAP9/src/deeploymezotest.c:659`).

**How the harness checks losses.** `Errors: a out of b` compares `min(total forwards, N_LOSS_REFS)` mini-batches of L+
and L− against the reference with an absolute tolerance of 1e-3 (`ZO_TOLERANCE_ABS`, runner flag `--tolerance`). It
also counts a failed `[WSUM]` check. A pass therefore means "within 1e-3". **Bit-exactness** of the fused graph was
established separately by comparing the raw loss bits and the dumped weights with an unfused run (§6.1).

**Data placement on GAP9.** The 54-window pool is 2.1 MB and cannot sit in L2. On GAP9 the training code generator
stages the initial weights and the baked test windows in L3 (hex files loaded at boot; log line `[testData->L3]`,
`TrainDeeploy/DeeployTest/testUtils/codeGenerateTraining.py:653-665`). The harness copies each window L3 → L2 before
its forward; that copy is part of every step and has not been timed (§9).

Until 2026-10-14 this did not happen with the NE16 runner. The generator recognised GAP9 by the platform class name
containing "GAP9", but GAP9 + NE16 deploys through `MemoryNE16PlatformWrapper`. So every window was baked into L2,
and the 4-window fixture failed at boot: `Allocation failed for allocator 1`, because the fused arena needs
1,325,460 B and only 1,272,272 B were free. Fixed at `codeGenerateTraining.py:610-614` (also accept "NE16"; the
original line is kept as a comment). The earlier single-window runs (§6.4, §6.7) were not affected: their one window
fitted in L2, and `BENCH` does not time the data copies.

---

## 6. Results and how to read them

### 6.1 How the numbers are obtained

There are four instruments. Do not mix them.

1. **`Runtime: N cycles`** (kernel fixtures). The cluster timer around the whole fixture network, including the
   fixture's boundary transposes (its inputs and outputs are NCHW, while the kernels run HWC). These are the **fixture
   totals**.
2. **`BENCH train_cycles_lo=… opt_cycles_lo=…`** (training harness). The sum, over all mini-batches, of the cluster
   timer around each training-graph execution (2 per mini-batch unfused, 1 fused), and around each update-graph
   execution. Loss read-out, `g_proj` and data copies between executions are not included.
3. **`--profileTiling`.** Wall clock per node, printed by the code generator's profiling wrapper. This is the
   **node-level metric** used for every per-block comparison. "Unfused block" means the sum of the nodes the fused node
   replaces: 2 × (weight perturbation + bias perturbation + weight transpose + conv). Input transposes are reported
   separately.
4. **Core-0 counters** (`--profileMicrobenchmark`). Not used for final numbers. They stop while core 0 sleeps in a
   barrier, so a load-imbalanced kernel looks faster than it is. In exp7 an apparent "DMA growth" turned out to be
   imbalance.

**Correctness instruments:**
- kernel fixtures: exact int8 compare, `Errors: 0 out of N`;
- training: the loss check above, plus `compare_dumps.py <unfused.log> <fused.log>`
  (`exp8_whole_network/compare_dumps.py`). With `-D DUMP_WEIGHTS=ON -D DUMP_STEP_LO=0 -D DUMP_STEP_HI=3` it compares
  all 22 trainable tensors after each of 4 steps (88 dumps, 17,316 words) and all 8 loss bit patterns. It must report
  zero differences.

**Seconds** = cycles / 370 MHz (GAP9 cluster clock used on the board). GVSoC's own default is 50 MHz; the harnesses
set 370 MHz.

**Unless stated otherwise the 4-step runs use n_accum 1** and the single-window fixture
`speechnet_{train,update}_QZO_single_step`, run 4 times. That is enough to compare fused against unfused, but it is
not a fine-tuning recipe.

### 6.2 Per block on Siracusa (cluster)

Node level, wall clock, both passes. Fixture numbers: `exp7_remaining_blocks/results_step5.txt`. Network numbers:
`exp8_whole_network/results/step8_per_block_vs_fixtures.txt` (logs `exp8_whole_network/logs/step0_siracusa_prof.log`,
`step8_siracusa_fused_prof_deint.log`).

| block | fixture A (unfused) | fixture C (fused) | fixture A/C | network unfused | network fused | network speedup |
|---|--:|--:|--:|--:|--:|--:|
| 0 | — | — | — | 1,356,479 | 640,523 | 2.12× |
| 1 | 884,766 | 472,437 | 1.87× | 883,811 | 475,276 | 1.86× |
| 2 | 308,026 | 139,274 | 2.21× | 306,079 | 141,489 | 2.16× |
| 3 | 193,136 | 52,416 | 3.68× | 191,716 | 51,757 | 3.70× |
| 4 | 332,600 | 40,821 | 8.15× | 325,995 | 40,812 | 7.99× |
| 1–4 | 1,718,528 | 704,948 | **2.44×** | 1,707,601 | 709,334 | **2.41×** |
| 0–4 | | | | 3,064,080 | 1,349,857 | **2.27×** |

**Interpretation.**
- **Blocks 3–4 gain the most.** Their unfused cost is dominated by the standalone weight-perturbation node (~141
  cycles per element per sign; block 4 has 7,168 weights and only 72 k MACs). The fused phase 1 replaces it at ~26
  cycles per element for both signs.
- **Block 1 gains the least.** Its conv is MAC-bound, and pulp-nn is already efficient there (6.29 MAC/cycle).
- **The network reproduces the fixtures** within 2 %.
- **Block 0** has no exp7 fixture under this metric. exp5's fixture total (1.88×) includes boundary transposes.

### 6.3 Per block on GAP9: cluster vs NE16

Fixture totals, both passes, `--l1 86000`; source `exp7_remaining_blocks/results_step7_gap9.txt` (per-block logs
`exp7*_block*/block*_fused_NE16/logs/gap9_*_plain.log`).

| block | A = 2 × pulp-nn | B = 2 × NE16 single pass | C_cl = fused cluster | C_P = fused NE16 | best |
|---|--:|--:|--:|--:|---|
| 1 | 741,872 | 911,630 (2 × one-pass run) | 502,274 | **277,119** | NE16, 2.68× |
| 2 | 242,930 | 325,614 | 144,827 | **108,908** | NE16, 2.23× |
| 3 | 114,526 | 177,546 | **58,309** | 97,700 | cluster, 1.96× |
| 4 | 168,920 | 252,242 | **49,057** | 158,922 | cluster, 3.44× |
| 1–4, best per block | 1,268,248 | | | 493,393 | **2.57×** (node level 2.99×) |

**Why NE16 loses on blocks 3–4.** For every step and pass, NE16 needs its weights perturbed and encoded into its
bit-serial format, plus a bias prep. That cost grows with the **number of weights**. NE16's advantage grows with the
**MACs per weight**. Fused NE16 node breakdown (node times, from the same file):

| block | MACs per weight | encode | bias prep | NE16 conv node | NE16 total | fused cluster node |
|---|--:|--:|--:|--:|--:|--:|
| 1 | 1,232 | 32,156 | 12,371 | 158,800 | 203,327 | 425,211 |
| 2 | 322 | 32,724 | 12,362 | 49,474 | 94,560 | 128,380 |
| 3 | 40 | 55,069 | 18,827 | 17,872 | 91,768 | **50,362** |
| 4 | 10 | 105,903 | 31,972 | 16,611 | 154,486 | **42,877** |

On blocks 3–4 the NE16 conv itself is fast (17–18 k), but encode + bias (74 k / 138 k) already exceed the whole fused
cluster kernel (50 k / 43 k). The cluster kernel has no encode step: it perturbs while loading in phase 1. The unfused
single-pass NE16 route (B) is slower than pulp-nn on every block, because its byte-wise column dominates.

The bias prep grows from 12 k to 32 k cycles for only 16–32 output channels. That looks larger than it should be and
has not been investigated (§9). It does not change the decision: the encode alone loses on blocks 3–4.

**In the network** (GAP9, NE16 for blocks 0–2, cluster for 3–4; node level, wall clock;
`exp8_whole_network/results/step8_per_block_vs_fixtures.txt`, logs `exp8_whole_network/logs/step7_gap9_{unfused,fused_ne16}_prof.log`):

| block | engine | unfused | fused | speedup | fixture |
|---|---|--:|--:|--:|--:|
| 0 | NE16 | 1,209,647 | 255,321 | **4.74×** | 4.73× node level (§6.6) |
| 1 | NE16 | 656,162 | 229,387 | 2.86× | 2.70× |
| 2 | NE16 | 209,870 | 91,254 | 2.30× | 2.23× |
| 3 | cluster | 100,765 | 48,172 | 2.09× | 1.96× |
| 4 | cluster | 152,443 | 40,459 | 3.77× | 3.44× |
| 1–4 | | 1,119,240 | 409,272 | **2.73×** | 2.57× |
| 0–4 | | 2,328,887 | 664,593 | **3.50×** | |

The network numbers are slightly **above** the fixture numbers, because the GAP9 fixture numbers are totals that
include NCHW boundary transposes the network does not have. No per-node profile was taken of the GAP9 all-cluster
build; its per-block cluster numbers are the fixtures' C_cl column.

### 6.4 The whole training step

4 device steps, n_accum 1, `BENCH train_cycles` (training graph only). All fused runs are identical to the unfused run
of the same platform (`compare_dumps.py`: all 88 weight dumps, all 8 loss values).

| platform | unfused | fused, cluster only | fused, NE16 for blocks 0–2 |
|---|--:|--:|--:|
| Siracusa | 20,680,174 | 13,542,814 (**1.53×**) | — |
| GAP9 | 15,383,030 | 10,814,171 (**1.42×**) | 8,630,585 (**1.78×**) |

Logs:
- Siracusa: `exp8_whole_network/logs/step5_siracusa_unfused_4steps.log`, `step8_siracusa_fused_4steps_deint.log`;
- GAP9: `step4_gap9_unfused_4steps.log`, `step8_gap9_fused_cluster_4steps_deint.log`, `step6_gap9_fused_ne16_4steps.log`.

The update graph is unchanged by the fusion: about 1.30 M cycles on Siracusa and 0.97 M on GAP9 for 4 updates.

Our unfused GAP9 number per step (15,383,030 / 4 = 3.85 M) matches Run's newest published unfused GAP9 QZO
single-step figure, 3.87 M (`silentwear-zo/notes/README.md`, §4.13). Older figures in her top-level README (6.63 M)
predate her BN / MaxPool / ReLU speedups.

### 6.5 Why the step gains less than the convs (Amdahl)

Training graph, wall clock per node (`exp8_whole_network/results/step8_per_block_vs_fixtures.txt`):

| | Siracusa unfused → fused | GAP9 unfused → fused NE16 |
|---|---|---|
| all convs and their perturbation | 3.06 M → 1.35 M (2.27×) | 2.33 M → 0.66 M (3.50×) |
| everything else (BN, Quant/Dequant, pool, ReLU, fp32 perturbation, FC + loss, input transposes) | 1.77 M → 1.73 M (1.03×) | 1.14 M → 1.13 M (1.01×) |
| sum over nodes | 4.83 M → 3.08 M (1.57×) | 3.47 M → 1.80 M (1.93×) |
| `BENCH` (4 steps / 4) | 5.17 M → 3.39 M (1.53×) | 3.85 M → 2.16 M (1.78×) |

Three things cap the step speedup:

1. **Non-conv nodes stay at ~1.0×.** They do the same work on 2·C channels once as on C channels twice; BatchNorm even
   grows (+6 % / +11 %). After fusion they are 56 % (Siracusa) and 63 % (GAP9) of the node time.
2. **Overhead outside the nodes.** `BENCH` minus the node sum is about 0.31–0.37 M cycles per step (closures, cluster
   offload, harness) and does not shrink.
3. **On GAP9 + NE16 the convs are already small.** At 0.66 M they are 37 % of the node time, so the remaining lever is
   the non-conv nodes (§9, item 1).

### 6.6 Block 0 on GAP9: 4.74× in the network, 2.71× in exp6

The exp6 fixtures were re-run on today's build at `--l1 86000` with wall-clock profiling
(`exp8_whole_network/results/step9_block0_gap9.txt`, logs `exp8_whole_network/logs/step9_exp6_{A,C}_rerun_prof.log`).

| level | unfused | fused | ratio |
|---|---|---|--:|
| node level, fixture | 2 × (pulp-nn 601,859 + perturb 2,478 + weight transpose 454) = 1,209,582 | im2col 119,666 + NE16 130,067 + encode/bias 6,190 = 255,923 | **4.73×** |
| node level, network | 1,209,647 | 255,321 | **4.74×** |
| fixture total | A: 2 × 767,262 = 1,534,524 | C: 549,018 | 2.80× (exp6 reported 2.71×) |

The fixture's boundary transposes are 229,566 cycles for A (input NCHW→NHWC and output NHWC→NCHW, twice) and 201,020
for C (output only). The fused output transpose alone is as large as the fused conv, and it does not exist in the
network, which stays channels-last downstream. So **the network matches the fixture node for node**; exp6's headline
2.71× was a fixture total. Today's im2col is faster than exp6's (142,171 → 119,666); not isolated why.

### 6.7 One update step with n_accum 4, QZO and QZO + ECO, and the round prediction

GAP9 + NE16 (blocks 0–2), fused, one update step = 4 fused dual passes + the update graph. `Errors: 0 out of 2` (the
step-10/11 runs reused the single-window fixture, so only the first pair was checked; the new
`..._single_update_step` fixture checks all 4 pairs: `0 out of 8`, §6.9).
Sources: `exp8_whole_network/results/step10_round_prediction.txt`, `step11_eco_round_prediction.txt`; logs
`exp8_whole_network/logs/step10_gap9_ne16_accum4_1step.log`, `step11_gap9_ne16_accum4_1step_eco.log`.

| cycles | QZO | QZO + ECO (β 0.9, round-to-nearest) |
|---|--:|--:|
| training (4 fused dual passes) | 8,612,666 | 8,621,038 |
| update graph | 242,923 | 508,606 (2.09×) |
| **one update step** | **8,855,589** | **9,129,644** (+3.1 %) |

**Round prediction.** A 120-epoch round on 54 windows with n_accum 4 has 120 × 54 / 4 = 1,620 update steps
(6,480 mini-batches). Cycles per round = 1,620 × cycles per update step:

| | cycles per round | @ 370 MHz | @ 240 MHz |
|---|--:|--:|--:|
| QZO fused | 14.35 G | **38.8 s** | 59.8 s |
| QZO + ECO fused | 14.79 G | **40.0 s** | 61.6 s |
| QZO unfused (15,625,953 cycles per step) | 25.31 G | 68.4 s | |
| QZO + ECO unfused (unfused training + measured ECO update) | 25.75 G | 69.6 s | |

So the fusion saves 1.76× (QZO) and 1.74× (QZO + ECO) on a round.

**What the prediction leaves out:**
- the per-window L3 → L2 copy of the 2.1 MB window pool;
- fabric-controller and host overhead;
- evaluation;
- the run-to-run spread of the tiler's randomised placement (up to ~1.5 %).

**Cross-check with the board.** Run's measured board rounds (unfused, 120 epochs, `silentwear-zo/rounds/summary.md`):
QZO 25.47 G cycles (69 s), QZO + ECO 26.0 G (70 s). Our simulated unfused estimates are 25.31 G and 25.75 G. The board
numbers include the data copy and harness overhead; the gaps (0.6 % and 1.0 %) bound those costs for the unfused graph.

**Hyper-parameters.** The step-11 ECO run used lr 1e-5 and round-to-nearest. Run's best board arm is QZO + ECO at
lr 3e-6 with stochastic rounding (`-D ZO_ECO=ON -D ZO_ECO_SR=ON`): 4-round mean 87.08 % vs 85.42 % for direct QZO. The
rounding mode and lr do not change the cycle count of the training graph; SR may add a little to the update graph
(not measured).

### 6.8 Accuracy

The fusion is bit-exact, so the accuracy of a fused round equals that of the unfused round with the same recipe. No
fused round has been run yet. The reference accuracies are Run's unfused board rounds (above) and the round-1 results
in guide 03.

### 6.9 Re-run from `SpeechNet_Test` (2026-10-14)

Every new fixture was run on GVSoC from its `SpeechNet_Test` location (scripts `run_smoke_{siracusa,gap9}.sh`, logs
`TrainDeeploy/SpeechNet_Test/logs/smoke/`). All of them pass exactly. The kernel numbers are **fixture totals**
(`Runtime`, including the boundary transposes), not the node-level numbers of §6.2–6.3. Unfused kernel fixtures are
**one** pass, so ×2 gives the both-pass baseline.

**Kernel fixtures, GAP9** (`--l1 86000`; reference = exp7 `results_step7_gap9.txt`, block 0 = the exp6 fixtures
re-run in exp8 step 9):

| block | unfused cluster ×2 | unfused NE16 ×2 | fused cluster | fused NE16 | errors |
|---|--:|--:|--:|--:|---|
| 0 | 1,526,074 (step 9: 1,534,524) | 755,106 | 871,696 | 545,506 (step 9: 549,018) | 0 on all |
| 1 | 741,782 (741,872) | 911,630 (911,630) | 502,896 (502,274) | 279,527 (277,119) | 0 on all |
| 2 | 242,930 (242,930) | 325,614 (325,614) | 145,245 (144,827) | 109,731 (108,908) | 0 on all |
| 3 | 114,566 (114,526) | 177,682 (177,546) | 58,364 (58,309) | 98,275 (97,700) | 0 on all |
| 4 | 168,970 (168,920) | 252,142 (252,242) | 49,169 (49,057) | 158,596 (158,922) | 0 on all |

Every cell is within 1 % of the exp7 measurement. The tiler's randomised placement explains the rest. Block 0's
numbers moved from exp6's `--l1 92000` (exp6: A 2 × 734,449, C_cl 925,593, C 541,345) because the tiling at 86000 is
different.

**Kernel fixtures, Siracusa** (`--l1 128000`): unfused (one pass) / fused, all exact.

| block | unfused (one pass) | fused |
|---|--:|--:|
| 1 | 469,934 | 529,028 |
| 2 | 170,216 | 162,895 |
| 3 | 103,558 | 60,817 |
| 4 | 173,803 | 48,656 |

Block 1 unfused equals exp7 step 0 exactly (469,934). The fused block-1 / block-2 totals match step 5d's whole-fixture
526.7 k / 161.8 k within 0.6 %.

**Whole network, GAP9 + NE16 for blocks 0–2, fused** (logs `gap9_ne16_fused_{single_qzo,single_eco,round1_14}.log`):

| run | fixture | check | training cycles | update cycles | per update step |
|---|---|---|--:|--:|--:|
| QZO, 1 update step, n_accum 4 | `..._single_update_step` | **Errors: 0 out of 8** (4 mini-batches) | 8,622,361 | 244,879 | 8,867,240 |
| QZO + ECO, 1 update step | `..._single_update_step` | **Errors: 0 out of 8**; `[ECO] … 61956 B momentum` | 8,621,191 | 508,766 | 9,129,957 |
| QZO, 14 update steps | `..._round1` (54 windows in L3, `[testData->L3] 2117232 B`) | **Errors: 0 out of 112** (56 mini-batches) | 120,770,781 | 3,420,993 | 8,870,841 |

These runs confirm the step-10/11 numbers with 4 distinct windows (+0.1 %) and with the full 54-window pool in L3. The
14-step run is the strongest check of the fused graph: 56 consecutive loss pairs on real, changing data, after 13
weight updates, all within the reference tolerance. A 120-epoch round at the 14-step average is
1,620 × 8,870,841 = 14.37 G cycles = **38.8 s @ 370 MHz**, the same as the prediction. The L3 → L2 window copies
happen outside the `BENCH` timer, so they are still not included.

### 6.10 Memory: the frozen-BN tail fusion (exp9, 2026-10-15)

**Setting.** GVSoC, GAP9 + NE16 for blocks 0–2, fused QZO+ECO, one update step, n_accum 4, lr 3e-3. This is Run's
board configuration plus `--plotMemAlloc`:
- fixture: `..._QZOECO_full_fused_single_update_step`;
- flags: `--l1 114000 --l2 1340000 -D BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON ZO_ECO=ON ZO_EXPECT_WSUM=0xb11cc64f`.

"Before" is `fe1ffa2`, which equals a run with `DEEPLOY_FUSE_BN_TAIL=0`. Sources: logs
`exp9_memory_optim/logs/{before,after}_fused_ne16_eco_step.log`; memory plots `results/{before,after}_memory_alloc.html`
(local, not tracked).

| | before | after |
|---|--:|--:|
| training L2 arena (`pi_l2_malloc` in `TrainingNetwork.c`) | 854,452 B | **265,612 B** (−588,840, −68.9 %) |
| peak live L2 (incl. 3,344 B constants) | 857,732 B @ block-0 BN | **268,956 B** @ block-0 conv |
| L1 arena | 113,664 B | 113,728 B |
| training cycles per update step | 7,540,615 | **6,560,579** (−13.0 %) |
| update-graph cycles | 508,659 | 508,301 |
| Errors / `[WSUM]` | 0 / 8, `0xb11cc64f` | 0 / 8, `0xb11cc64f`; all 8 loss bit patterns identical |

**What makes up the peak:**

| | contents |
|---|---|
| before | BN output, fp32, dual pass 628,096 B + BN input int8 157,024 B + window 39,200 B + ~33 KB |
| after | block-0 conv output int8 157,024 B + im2col column 39,256 B + window 39,200 B + ~33 KB |

**All board configurations, code generation only** (`scripts/mem_compare.sh`, round-1 fixtures, lr 3e-6, ECO with SR;
`exp9_memory_optim/memcompare/summary.txt`):

| configuration | training L2 arena, tail0 | tail1 (exp9) |
|---|--:|--:|
| BP (`deeployTrainingRunner_tiled_gap9.py`, L3 default level) | 687,488 B (+ 1,005,648 B in L3) | pass does not fire |
| ZO fp32 | 791,216 B | pass does not fire |
| QZO+ECO unfused (cluster) | 466,464 B | **162,152 B** |
| QZO+ECO fused, NE16 blocks 0–2 | 854,452 B | **265,612 B** |

So with exp9 the fused QZO+ECO graph needs less L2 than the unfused one did before.

**Regression; all bit-exact.** `tail0` means `DEEPLOY_FUSE_BN_TAIL=0`. Scripts `scripts/verify_{gap9,siracusa}.sh`;
logs in `exp9_memory_optim/verify/` and `logs/`.

| run | result | L2 arena tail0 → tail1 | training cycles tail0 → tail1 |
|---|---|--:|--:|
| GAP9 + NE16 fused + ECO, 1 update step | 0 / 8, same WSUM, 8 loss bit patterns identical | 854,452 → 265,612 | 7,539,511 → 6,559,071 |
| GAP9 + NE16 fused, round-1 fixture, 14 update steps, direct QZO lr 1e-5 | 0 / 112; 112 loss bit patterns and `[WSUM]` identical | 854,452 → 265,612 | 105,515,019 → 91,854,554 (−12.9 %) |
| GAP9 unfused QZO, 1 step | 0 / 2, WSUM `0x97e57471` both | 466,464 → 162,152 | 3,662,191 → 3,332,852 |
| GAP9 unfused QZO+ECO, 5 steps | 0 / 2, WSUM `0x6706ea8a` (the CI value) | → 162,152 | → 16,726,310 |
| GAP9 ZO fp32 | 0 / 2, WSUM `0xc14819e6` (the CI value); pass does not fire | 791,216 | 6,724,646 |
| Siracusa fused, 4 steps, `-D DUMP_WEIGHTS=ON` (setting of §7.3) | losses + 110 WDUMP lines identical | 855,516 → 237,108 | 10,778,198 → 8,798,767 (−18.4 %) |
| Siracusa unfused QZO, 4 steps | losses + WDUMP identical | 466,464 → 162,152 | 18,245,535 → 15,999,369 |
| GAP9 CI `gap9_tiled and speechnet_gap9` | **6 / 6 pass** (both fused weight-hash gates included) | | |
| Siracusa CI `-k "siracusa and SpeechNet"` | **13 / 13 pass** (kernel fixtures at the exp7 cycles) | | |

No Siracusa CI test sets `BN_FROZEN_STATS`, so CI never runs the pass on Siracusa; the 4-step runs above are the
Siracusa check.

**Board estimate (inferred, not measured).** From the 14-step run: (91.85 M + 3.42 M update) / 14 × 1,620 = 11.0 G
cycles per round, against 12.6 G before. That is about −13 % on Run's measured 35.1 s board round, so roughly 31 s.
The board run is open (§9 item 3).

---

## 7. How to reproduce

Containers and conventions as in `00_Overview.md`:
- Siracusa: container `traindeeploy`, TrainDeeploy at `/app/ETH/TrainDeeploy`.
- GAP9: container `deeploy_gap9`, TrainDeeploy at `/app/Deeploy`; source the three SDK lines first.

Before every GVSoC run, kill orphaned simulators **by PID**:

```bash
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
```

Give each configuration its own build directory with `PYTEST_XDIST_WORKER=<tag>` (CMake caches `-D` flags), or
`rm -rf TEST_<PLATFORM>`.

### 7.1 Regenerate the fixtures (host, needs `agitated_hugle` for the exports)

```bash
cd /Users/qiwenwu/ETH/TrainDeeploy/SpeechNet_Test/scripts
bash export_fixtures.sh kernels_b1to4                     # 16 per-block fixtures, copied from DeeployTest/Tests/Kernels/Integer
bash export_fixtures.sh qzoeco_single_update qzoeco_round1  # whole-network entry fixtures (~1.5 min)
# the per-block fixtures themselves (container agitated_hugle; builders need the Onnx4Deeploy oracle):
docker exec agitated_hugle bash -lc "cd /app && export PYTHONPATH=/app/Onnx4Deeploy:/app/TrainDeeploy && \
  python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_qzo_conv_fixture.py --block 1 && \
  python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_dual_conv2in_fixture.py --block 1"
# NE16 fixtures copy their tensors from the two above; plain python with onnx + numpy suffices
python3 ../../DeeployTest/experiments/dual_pass_fusion/build_ne16_im2col_fixture.py --block 1
python3 ../../DeeployTest/experiments/dual_pass_fusion/build_ne16_dual2in_fixture.py --block 1 --single-node 1
bash export_fixtures.sh kernels_b1to4                     # then copy them into SpeechNet_Test again
```

### 7.2 Per-block kernels

From the host (the scripts wrap the container calls, write `SpeechNet_Test/logs/smoke/*.log`, and print `Errors` and
`Runtime`):

```bash
cd /Users/qiwenwu/ETH/TrainDeeploy/SpeechNet_Test/scripts
bash run_smoke_siracusa.sh b1 b2 b3 b4      # unfused (one pass) + fused cluster per block
bash run_smoke_gap9.sh b1 b2 b3 b4          # unfused cluster / unfused NE16 / fused cluster / fused NE16 per block
bash run_smoke_gap9.sh unfused_cluster fused_cluster unfused_NE16 fused_NE16    # block 0
```

One run by hand (GAP9, block 1, fused NE16). The activation inputs must be pinned to int8 and the requant bias to
int32:

```bash
# container deeploy_gap9, in /app/Deeploy/DeeployTest, after the three SDK source lines
A=/blocks_1_conv_input_quant_1/Clip_output_0; B=blocks.1.conv.bias_rqsadd
python3 deeployRunner_tiled_gap9_w_ne16.py -t ../SpeechNet_Test/speechnet_fused_b1_NE16 \
  --l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8 \
  --input-type-map $B=int32_t $A=int8_t ${A}_minus=int8_t --input-offset-map $B=0 $A=0 ${A}_minus=0
#  -> Errors: 0 out of 39424 ; Runtime: ~277 k cycles (exp7: 277,119)
# unfused fixtures take only the first two maps ($B and $A); Siracusa: deeployRunner_tiled_siracusa.py --l1 128000 --l2 1500000 --cores 8
# per-node wall clock: add --profileTiling (then compare node times, not Runtime)
```

The exp7 measurement scripts that produced the tables are `dual_pass_fusion/exp7_remaining_blocks/run_block.sh <i>`
(Siracusa: builds the fixtures and runs unfused / Conv4x2 / fused) and `run_gap9_block.sh <i> [A B C_cl C C_S C_P]`
(GAP9). Tables come from `wall_table.py` (wall clock) and `gap9_table.py`.

### 7.3 Whole network: fused vs unfused, 4 steps, bit-exact check

```bash
# Siracusa (container traindeeploy, /app/ETH/TrainDeeploy/DeeployTest)
C="-t ../SpeechNet_Test/speechnet_train_QZO_single_step --optimizer-dir ../SpeechNet_Test/speechnet_update_QZO_single_step \
   --n-steps 4 --n-accum 1 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
   --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8 -D BN_FROZEN_STATS=ON \
   -D DUMP_WEIGHTS=ON -D DUMP_STEP_LO=0 -D DUMP_STEP_HI=3"
PYTEST_XDIST_WORKER=unf python3 deeployMezoRunner_tiled_siracusa.py $C                > unfused.log 2>&1   # BENCH train 20,680,174
PYTEST_XDIST_WORKER=fus python3 deeployMezoRunner_tiled_siracusa.py $C --zo-fused-dual > fused.log 2>&1     # BENCH train 13,542,814
python3 experiments/dual_pass_fusion/exp8_whole_network/compare_dumps.py unfused.log fused.log            # must report no difference

# GAP9 (container deeploy_gap9, /app/Deeploy/DeeployTest)
C="-t ../SpeechNet_Test/speechnet_train_QZO_single_step --optimizer-dir ../SpeechNet_Test/speechnet_update_QZO_single_step \
   --n-steps 4 --n-accum 1 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
   --l1 114000 --l2 1500000 --defaultMemLevel L2 --cores 8 -D BN_FROZEN_STATS=ON -D GAP9_NETWORK_O3=ON \
   -D DUMP_WEIGHTS=ON -D DUMP_STEP_LO=0 -D DUMP_STEP_HI=3"
PYTEST_XDIST_WORKER=gu python3 deeployMezoRunner_tiled_GAP9.py         $C                                     > g_unf.log  # 15,383,030
PYTEST_XDIST_WORKER=gc python3 deeployMezoRunner_tiled_GAP9.py         $C --zo-fused-dual                     > g_cl.log   # 10,814,171
PYTEST_XDIST_WORKER=gn python3 deeployMezoRunner_tiled_GAP9_w_NE16.py  $C --zo-fused-dual --zo-fused-dual-ne16 0,1,2 > g_ne16.log # 8,630,585
# per-node profile for §6.2/6.5: --n-steps 1 (no dumps) plus --profileTiling; categorise with exp8_whole_network/categorize.py
```

### 7.4 Whole network: one update step with n_accum 4 (QZO and QZO + ECO), GAP9 + NE16

```bash
# container deeploy_gap9, /app/Deeploy/DeeployTest
T=../SpeechNet_Test
F="-t $T/speechnet_train_QZOECO_full_fused_single_update_step --optimizer-dir $T/speechnet_update_QZOECO_full_fused_single_update_step \
   --n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
   --l1 114000 --l2 1500000 --defaultMemLevel L2 --cores 8 -D BN_FROZEN_STATS=ON -D GAP9_NETWORK_O3=ON \
   --zo-fused-dual --zo-fused-dual-ne16 0,1,2"
PYTEST_XDIST_WORKER=q1 python3 deeployMezoRunner_tiled_GAP9_w_NE16.py $F              > qzo_1step.log 2>&1
PYTEST_XDIST_WORKER=e1 python3 deeployMezoRunner_tiled_GAP9_w_NE16.py $F -D ZO_ECO=ON > eco_1step.log 2>&1
grep -aE "Errors|BENCH|\[ECO\]" qzo_1step.log eco_1step.log
# cycles per update step = train_cycles_lo + opt_cycles_lo ; round = 1,620 x that ; seconds = cycles / 370e6
# unfused comparison: drop --zo-fused-dual --zo-fused-dual-ne16 0,1,2 (and use deeployMezoRunner_tiled_GAP9.py)
```

### 7.5 Whole network: a round

```bash
# check the fused graph on the round fixture's referenced steps (direct QZO): expect Errors: 0 out of 112
F="-t $T/speechnet_train_QZOECO_full_fused_round1 --optimizer-dir $T/speechnet_update_QZOECO_full_fused_round1 \
   --n-accum 4 --num-data-inputs 2 --eps 0.01 --q 1 --seed 42 \
   --l1 114000 --l2 1500000 --defaultMemLevel L2 --cores 8 -D BN_FROZEN_STATS=ON -D GAP9_NETWORK_O3=ON \
   --zo-fused-dual --zo-fused-dual-ne16 0,1,2"
PYTEST_XDIST_WORKER=r14 python3 deeployMezoRunner_tiled_GAP9_w_NE16.py $F --n-steps 14 --lr 1e-5 > round_14.log 2>&1
# a full QZO+ECO round (1,620 steps; on GVSoC many hours, see §9 before trying the board):
PYTEST_XDIST_WORKER=r1 python3 deeployMezoRunner_tiled_GAP9_w_NE16.py $F --n-steps 1620 --lr 3e-6 \
   -D ZO_ECO=ON -D ZO_ECO_SR=ON -D DUMP_WEIGHTS=ON > round1_eco.log 2>&1
# only the first update step is checked against the reference; evaluate the dumped weights on batch 2 as in guide 03
```

---

### 7.6 CI gates

| test | marker / workflow | what it checks |
|---|---|---|
| `test_gap9_tiled_speechnet_fused[SpeechNet_QZO_ECO_fused_NE16_single_update_step]` (`TrainDeeploy/DeeployTest/test_platforms.py`, config `SPEECHNET_FUSED_TESTS` in `TrainDeeploy/DeeployTest/test_gap9_tiled_config.py`) | `gap9_tiled and speechnet_gap9`; runs on every push (`.github/workflows/ci-speechnet.yml`) | the fused dual-pass graph on GAP9 + NE16 (blocks 0–2), QZO+ECO, one update step with n_accum 4 at lr 3e-3: all 4 loss pairs equal the host reference, and the hash of all 22 trained tensors (`-D ZO_EXPECT_WSUM`, Run's `[WSUM]` gate) equals `0xb11cc64f`, the value the **unfused** GAP9 cluster run of the same step produces. 21 of 22 tensors change, so the hash covers the whole update. lr 3e-3 is a gate value, not the recipe (QZO+ECO trains at 3e-6, where one step moves almost no int8 weight and the hash would test little); Run's `SpeechNet_QZO_ECO_5steps` gate uses 3e-3 for the same reason. ~2.5 min |
| `test_siracusa_tiled_qzo_blocks_kernels` (config `QZO_BLOCKS_KERNEL_TESTS`, `qzo_block_gen_args` in `TrainDeeploy/DeeployTest/test_siracusa_tiled_config.py`) | `siracusa_tiled and qzo_blocks`; manual workflow `ci-platform-siracusa-tiled.yml` | the 8 blocks 1–4 cluster kernel fixtures, exact int8 output; ~70 s |

One fused test is enough for the whole network. It exercises the pass, every new node type (`RequantizedDualConv` in
both NE16 and cluster forms, `PerturbRademacherFixed`, `DualHalf`), the single-run harness, ECO and the L3 test-data
path, and the weight hash compares against the unfused computation. A Siracusa fused step and the GAP9 NE16
per-block kernels are not gated; run them by hand (§7.2–7.3).

To refresh the expected hash after an intended numerical change, run the same step unfused with a placeholder
(`deeployMezoRunner_tiled_GAP9.py … --lr 3e-3 -D ZO_ECO=ON -D ZO_EXPECT_WSUM=0x0`), read `[WSUM] final 0x…`, and
confirm the fused run prints the same value.

The fused gate also covers exp9: it builds with frozen BN, so `FrozenBNPoolReluQuant` is in the gated graph, and the
hash it checks is unchanged (`0xb11cc64f`).

### 7.7 Memory: the frozen-BN tail fusion (exp9)

The scripts are in `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp9_memory_optim/scripts/`. Every run is
done twice, without (`DEEPLOY_FUSE_BN_TAIL=0`) and with the pass. Each run keeps three outputs: the log, the generated
`TrainingNetwork.c` and the lowered graph. Each script ends with the CI gates.

```bash
# GAP9 (container deeploy_gap9; the script sources the SDK itself) -> verify/*.log, *.TrainingNetwork.c, *.post_lowering.onnx
docker exec deeploy_gap9 bash -lc "bash /app/Deeploy/DeeployTest/experiments/dual_pass_fusion/exp9_memory_optim/scripts/verify_gap9.sh"
# Siracusa (container traindeeploy)
docker exec traindeeploy bash -lc "bash /app/ETH/TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp9_memory_optim/scripts/verify_siracusa.sh"
# L2 plans of the four board configurations, code generation only (--skipsim --plotMemAlloc) -> memcompare/<tag>/
docker exec deeploy_gap9 bash -lc "bash /app/Deeploy/DeeployTest/experiments/dual_pass_fusion/exp9_memory_optim/scripts/mem_compare.sh"
cd /Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp9_memory_optim
for t in bp zo eco_unf_t0 eco_unf_t1 eco_fus_t0 eco_fus_t1; do echo "== $t"; \
  python3 scripts/mem_plot_summary.py memcompare/$t/deeployStates/memory_alloc.html; done   # peak live bytes per level
```

**What to check:**
- the per-run line prints `Errors`, `[WSUM]`, `fused_calls` (the number of `PULP_FrozenBNPoolReluQuantHWC_i8_i8`
  calls in the generated C: 4 = blocks 0–3 for every QZO graph with tail1, 0 for tail0 and for ZO fp32), the
  `pi_l2_malloc` size and `train_cycles_lo`;
- tail0 and tail1 must print the same `Errors` and WSUM.

For a single run by hand, take §7.4's command and set the variable in front of it:
`DEEPLOY_FUSE_BN_TAIL=0` (before) or nothing (after); the pass is on by default whenever frozen BN is on. Add
`--plotMemAlloc` for the L2 plot in `TEST_GAP9_W_NE16/<fixture>/deeployStates/`.

## 8. Known pitfalls and lessons

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | Fused kernel exact on block 0 but 950/2,048 wrong W+ values on block 1 | the old fused kernels perturbed the HWC-transposed weight; the reference perturbs CHW | raw CHW weight (`weight_chw = 1`) + `dc_perturb_i8_chw` |
| 2 | Fused L+ 2.52 instead of ~0.04; logits wrong, everything up to the global pool exact | Deeploy's tiled `Slice` strides with the full input shape, but its tile constraint replaces `data_in_size` by the tile size → `memcpy(…, 0)` | do not use `Slice`; found with `exp8_whole_network/bisect_dual.py` (truncated dual graph vs the oracle) |
| 3 | GAP9 step-1 L− `0x3d98abd6` vs `0x3d98abc7` with zero-padded FC weights (Siracusa exact) | the PULP fp32 Gemm (transB) sums K in blocks of 6; padding K to 2·C changes the grouping and the rounding | `DualHalf`, so each Gemm sees the unfused input |
| 4 | With `DualHalf`, Siracusa wrong at indices 25–31 of both halves; GAP9 exact | the strided per-core loop `c = core_id; c += NUM_CORES` lost its last iteration on cores 1–7 (likely a hardware-loop trip-count miscompile by Siracusa's LLVM; not proven) | contiguous per-core chunks, as in the other PULP templates; use them in new templates |
| 5 | Blocks 1–2 slower in the network than in their fixtures (+98 k on Siracusa) | the stacked input makes 1×K kernel rows non-contiguous, so every receptive field was gathered | de-interleave the tile once (§4.4) |
| 6 | `[ECO] momentum alloc failed (k=19)` on the fused build | momentum over-allocated 4× for int32 / fp32 tensors; the fused L2 arena left too little heap | per-element sizing via `inputs_elems` (§4.6) |
| 7 | Code generation: transient buffer lookup error for `PerturbRademacherFixed` | the tiler looks up whatever `computeTransientBuffersSize` names, even when nothing is hoisted | always hoist (4 B placeholder) |
| 8 | Code generation: `'data_in_m' is not defined` | Mako runs strict: every name in a template must exist, even in an untaken `% if` branch | separate stacked parser, binding and template |
| 9 | The block-1 unfused fixture used the `u8` pulp-nn kernel | input type inferred from values (≥ 0) | `--input-type-map $A=int8_t` (and `${A}_minus` for fused fixtures) |
| 10 | Fused cluster kernel 2× slower on GAP9 than expected | the GAP9 library compiled new sources at the SDK's `-Os` | add them to the `-O3` list in `TargetLibraries/GAP9/CMakeLists.txt` |
| 11 | GAP9 kernel fixture: L1 arena allocation fails at `--l1 92000` | the GAP9 kernel harness keeps the cluster stacks in L1 on this branch | `--l1 86000` for kernel fixtures; the MeZO harness uses `--l1 114000` |
| 12 | A load-imbalanced kernel looked faster, "DMA grew" after fusion | core-0 counters stop in barriers | measure wall clock (`--profileTiling`) |
| 13 | Inner-loop rewrites made `DualConv2In` slower (pointer-advancing loop 497 k → 649 k; out-of-line per-pass function; ×2 unroll) | clang loses the hardware loop or the I-cache footprint grows | keep the step-3 inner loop; it is a local optimum for this compiler |
| 14 | `No space left on device` during builds | the Docker VM disk (59 GB) was full | `pip cache purge` in `agitated_hugle` freed 9.5 GB; watch `docker exec agitated_hugle df -h /` |
| 15 | Kernel-fixture cycles differ by up to ~1.5 % between builds | the tiler's `random-max` search strategy randomises buffer placement | compare node times, not single `Runtime` values; outputs stay exact |
| 16 | GAP9 + NE16 training build: `Allocation failed for allocator 1` at `InitTrainingNetwork` with a multi-window fixture | `codeGenerateTraining.py` detected GAP9 by `"GAP9" in type(Platform).__name__`; the NE16 wrapper `MemoryNE16PlatformWrapper` failed the test, so all windows and initial weights were baked into L2 | also match "NE16" (`codeGenerateTraining.py:610-614`); check the log for `[testData->L3]` |
| 17 | exp9: a single-consumer pattern did not match after `ZODualPassFusion` although the graph showed one consumer | the pass removed the replaced `Conv` / `RequantShift` from `graph.nodes` only; onnx-graphsurgeon keeps tensor consumer lists in sync with `node.inputs`, so they stayed as phantom consumers through lowering (invisible to `cleanup()`, gone at export) | clear the removed nodes' inputs and outputs (`ZODualPassFusion.py:184`); new matchers should skip consumers without outputs (`FrozenBNTail._only_consumer`) |
| 18 | exp9: kernel does not compile with a local variable `inf` | `inf` is a macro in `TargetLibraries/Generic/inc/macros.h` (`1.0f / 0.0f`) | do not name variables `inf`; use the macro |

---

## 9. Open items and next steps

1. **Non-conv nodes** (63 % of the fused node time on GAP9).
   - **Done:**
     - Run folded Dequant into the frozen BN;
     - exp9 fused BN + MaxPool + ReLU + Quant with the arithmetic kept (§4.8). That took the training part of a step
       down 13 %.
   - **Still open:** folding the frozen BN into the conv requant, which would remove the fp32 tail entirely.
     **A decision is needed** on the reference: that changes the fp32 arithmetic, so the result would no longer be
     bit-exact against today's unfused graph.
2. **The ~0.3 M cycles per step outside the nodes** (closures, cluster offload).
3. **Board status** (measured on a `-s board` build of the fused NE16 QZO+ECO round, 2026-10-14; build log
   `TrainDeeploy/SpeechNet_Test/logs/smoke/board_build_fused_ne16_eco_round1_l2_1340000.log`). An L2 malloc failure
   hangs silently on the board, so check these before the first run.
   - L2 heap for this binary: 1,372,464 B (`__heapl2ram_start` 0x1C040ED0 to 0x1C190000). The heap depends on each
     binary's code size and SDK configuration; the 1,343,856 B measured earlier on the AE/CCT binaries does not apply.
   - Fused QZO fits: L2 arena 1,325,460 B, 47 KB margin (minus small allocations of the SDK at boot).
   - Fused QZO+ECO is 14,952 B short (ECO momentum 61,956 B). Dropping `GAP9_NETWORK_O3` (~35 KB of code) would likely
     make it fit, at some speed cost; alternatives are keeping the momentum in L3 or moving arena tensors to L3.
   - Needs `--l2 1340000`: at the board scripts' `--l2 1300000` the arena cannot be allocated (MiniMalloc fails,
     `TrainDeeploy/SpeechNet_Test/logs/smoke/board_build_fused_ne16_eco_round1.log`).
   - The 54 windows are staged in L3 through readfs (`[testData->L3] 2117232 B`).
   - **First board runs (RW, 2026-10-15)**, round-1 fixture `..._QZOECO_full_fused_round1`, one update step, n_accum 4,
     `Errors` over the 4 loss pairs: fused cluster only 0/8 (10,721,307 train cycles); fused, NE16 for block 0 only 0/8
     (9,512,093); fused, NE16 for block 1 only **7/8** -- the very first L+ is 0.039707 vs 0.038565, before any update,
     and identical with `DEEPLOY_NE16_PIPELINE=0` (sequential jobs), so it is not the pipeline race: `NE16DualConv2in`
     (the stacked-input HWC column job) computes differently on the silicon than in GVSoC. Block 0's NE16 form (im2col +
     plain 1x1) is exact on the board. Until this is fixed the board configuration is `--zo-fused-dual-ne16 0`.
   - **Fixed (RW, eabbb0b2).** Block-1 kernel fixture on the board: 152 of 39,424 outputs +1 LSB; 137 of them in
     column 2 of the first 3x3 output subtile of every job after the first (job width 29), the rest in the 3 x 5
     remainder job. The silicon NE16 corrupts that column when a job is wider than 3 and its width is not a multiple
     of 3 (GVSoC does not model it). `job_w` is now a multiple of 3 and the remainder is split into 3 x 3k, 1 x 3m and
     1 x (<3) jobs: 0 / 39,424 on the board. **L2**: `FoldDequantIntoFrozenBNPass` lets the frozen BN read the int8
     conv output (the fp32 dequantized tensor was half of the peak): training arena 1,317,012 -> 854,452 B, so fused
     QZO+ECO fits. Board, 120-epoch round 1, `--zo-fused-dual-ne16 0,1,2`, weights == unfused board rounds:
     QZO 12.99 G cycles = **35.1 s**, QZO+ECO (SR) 13.37 G = **36.1 s** @ 370 MHz.
   - **exp9 on the board (open).** With `FuseFrozenBNTailPass` the fused QZO+ECO arena is 265,612 B, about 1.1 MB
     below the board heap, so L2 is no longer a constraint. The step is −13 % on GVSoC (§6.10). Not yet run on the
     board; first run round 1 for 14 steps (`Errors: 0 out of 112`, same `[WSUM]` as without the pass), then a full
     round.
4. **Stack depth.** Worst slave-stack chain 576 B of the 768 B L1 slave stack (`silentwear-zo/scripts/stack_depth.py`
   on the board ELF): `PULP_DualConv2d_2in_i8_i8_i8_HWC` [336] -> `dc_perturb_i8_chw` [96] -> `lrintf` [16] under a
   128 B closure. The NE16 prep chains are 304–336 B. The tool covers worker-core entries only; `NE16DualConv2in` runs
   on the cluster controller (4 KB stack).
   The exp9 kernel `PULP_FrozenBNPoolReluQuantHWC_i8_i8` has not been measured yet;
   run `stack_depth.py` on the exp9 board ELF before the first board run.
   NE16 has not yet run on the board (the board rounds so far used the cluster-only runner); the first board run
   should be fused QZO for 14 steps of the round-1 fixture (`Errors: 0 out of 112`, `[WCHK]`).
5. **Time the per-window L3 → L2 copy** of the 54-window pool, which the round prediction does not include.
   - **Next memory lever:** the block-0 conv peak (int8 output 157 KB + im2col column 39 KB + window 39 KB).
6. **Block-0 NE16 double buffering.** Block 0 still uses exp6's unpipelined form (im2col 120 k + NE16 130 k
   sequential); the exp7 pipeline would hide most of the im2col.
7. **NE16 preparation costs.** The encode is now the largest prep item (32 k on block 1). The bias prep grows to 32 k
   on block 4, which is more than 32 channels should need.
8. **A full fused round** (QZO + ECO, lr 3e-6, SR) on the board once items 3–4 are done, judged by batch-2 accuracy
   against Run's unfused board rounds.
9. **Cleanups:** decide whether `DEEPLOY_RQCONV_KERNEL=4x2` stays as a feature. (CI gates were added and the unused
   `pad_cols` path was removed on 2026-10-14.)

---

## 10. Evidence and artifacts

Paths are relative to `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/` unless they start with `TrainDeeploy/`.

| topic | tracked logs, results, scripts, fixtures |
|---|---|
| fixtures (this guide) | `TrainDeeploy/SpeechNet_Test/speechnet_{unfused,fused}_b{1..4}_{cluster,NE16}/`, `TrainDeeploy/SpeechNet_Test/speechnet_{train,update}_QZOECO_full_fused_{single_update_step,round1}/`; export and copy script `TrainDeeploy/SpeechNet_Test/scripts/export_fixtures.sh`; device checks `TrainDeeploy/SpeechNet_Test/scripts/run_smoke_{siracusa,gap9}.sh` → `TrainDeeploy/SpeechNet_Test/logs/smoke/`; export logs `TrainDeeploy/SpeechNet_Test/logs/export_qzoeco_*.log` |
| fixture originals and builders | `TrainDeeploy/DeeployTest/Tests/Kernels/Integer/QZO_{Conv,Conv_NE16,DualConv2in,DualConv2in_NE16P}_B{1..4}/` (also `QZO_DualConv2in_NE16{,S}_B<i>`, `QZO_DualConv2in_B1_same`, `QZO_DualConv2in_stacked_B{1,3}`); `build_{qzo_conv,dual_conv2in,ne16_im2col,ne16_dual2in}_fixture.py` |
| exp7 Siracusa per block | `exp7_remaining_blocks/results_step5.txt` (final, wall clock), `results_step5{a,b,c,d}_nodes.txt`, `results_step5d_wall.txt`, `results_step6.txt`; per block `exp7_remaining_blocks/exp7{a,b,c,d}_block*/block*_{unfused,mid,fused_cluster}/{logs,results,codegen}/`; scripts `run_block.sh`, `node_table.py`, `wall_table.py`, `probe_block.sh` |
| exp7 GAP9 per block | `exp7_remaining_blocks/results_step7_gap9.txt`; `exp7*_block*/block*_fused_NE16/{logs,results,codegen}/` (block 1 also `results/summary_optimised.txt`, `results/summary.txt` with the `-Os` comparison); scripts `run_gap9_block.sh`, `gap9_table.py` |
| exp8 whole network | `exp8_whole_network/results/step{0,3,8,9,10,11}_*.txt`; logs `exp8_whole_network/logs/` (`step0_*` baseline, `step4_*` / `step6_*` / `step8_*` 4-step runs, `step7_*` GAP9 profiles, `step9_*` exp6 re-runs, `step10_*` / `step11_*` n_accum 4, `bisect_*`, `regression_*`); tools `compare_dumps.py`, `bisect_dual.py`, `run_bisect.sh`, `categorize.py`, `oracle_check_dual.py` |
| lowered graphs | `exp8_whole_network/graphs/*.onnx`, `exp9_memory_optim/graphs/*.onnx` (§4.7) |
| exp9 memory | `exp9_memory_optim/logs/` (before / after step, 14-step round, Siracusa 4 steps, CI); `exp9_memory_optim/verify/*.log` (tail0 / tail1 per configuration, CI); `exp9_memory_optim/memcompare/summary.txt` (L2 plan of the four board configurations); scripts `exp9_memory_optim/scripts/{verify_gap9.sh, verify_siracusa.sh, mem_compare.sh, mem_plot_summary.py}`. Local only (too large, or `.html` ignored): `results/*_memory_alloc.html`, `verify/*.TrainingNetwork.c`, `verify/*.post_lowering.onnx`, `memcompare/<tag>/` |
| Run's GAP9 work this builds on | TrainDeeploy commits on `silentwear-zo/gap9-on-devel` (ECO `886c948`); board rounds and constraints in the `silentwear-zo` repository (`rounds/summary.md`, `rounds/export_qzo999_rounds.sh`, `scripts/stack_depth.py`, `notes/README.md` §4.13) |

**Code map** (all marked `-- QW`):

| layer | files |
|---|---|
| pass | `TrainDeeploy/Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/ZODualPassFusion.py`; call site `TrainDeeploy/DeeployTest/testMVPTraining.py:96-104` |
| parsers | `TrainDeeploy/Deeploy/Targets/PULPOpen/Parsers.py` (`PULPDualConv2DParser`, `DualHalfParser`, `PerturbRademacherFixedParser`, `PULPDualConv2D2InParser`, `PULPDualConv2DStackedParser`); `TrainDeeploy/Deeploy/Targets/NE16/DualConv2in.py` (`NE16DualConv2inParser`, `NE16DualConv2inStackedParser`) |
| templates | `TrainDeeploy/Deeploy/Targets/PULPOpen/Templates/{DualConv2InTemplate.py, Conv4x2Template.py, DualHalfTemplate.py, FloatPerturbRademacherFixedTemplate.py}`; NE16 templates in `Deeploy/Targets/NE16/DualConv2in.py` |
| tile constraints | `TrainDeeploy/Deeploy/Targets/PULPOpen/TileConstraints/{DualConv2InTileConstraint.py, UntiledTileConstraint.py}`; `TrainDeeploy/Deeploy/Targets/NE16/TileConstraints/NE16DualConv2inConstraint.py` |
| bindings, tiler, platform | `TrainDeeploy/Deeploy/Targets/PULPOpen/{Bindings.py, Tiler.py, Platform.py}`; `TrainDeeploy/Deeploy/Targets/GAP9/Platform.py`; `TrainDeeploy/Deeploy/Targets/NE16/{Engine.py, Tiler.py}`; layout pass `LoweringOptimizationPasses.py:326` |
| kernels | `TrainDeeploy/TargetLibraries/PULPOpen/src/{DualConv2In.c, Conv4x2.c}`, `TrainDeeploy/TargetLibraries/PULPOpen/inc/kernel/{DualPerturb.h, DualConv2In.h, Conv4x2.h}`; `TrainDeeploy/TargetLibraries/GAP9/src/{NE16DualConv2in.c, NE16DualPrep.c}` |
| exp9 (memory) | pass, parser, layer, template, tile constraint, bindings: `TrainDeeploy/Deeploy/Targets/PULPOpen/FrozenBNTail.py`; called from `TrainDeeploy/Deeploy/Targets/{PULPOpen,NE16}/Deployer.py`; mappers in `TrainDeeploy/Deeploy/Targets/{PULPOpen,GAP9}/Platform.py`; kernel `TrainDeeploy/TargetLibraries/PULPOpen/src/BatchNorm.c:502`, prototypes `TrainDeeploy/TargetLibraries/PULPOpen/inc/kernel/BatchNorm.h`, `TrainDeeploy/TargetLibraries/GAP9/inc/kernel/GAP9Kernels.h`; phantom-consumer fix `ZODualPassFusion.py:184` |
| harness, runner, build | `TrainDeeploy/DeeployTest/Platforms/{Siracusa,GAP9}/src/deeploymezotest.c`, `TrainDeeploy/DeeployTest/Platforms/{Siracusa,GAP9}/CMakeLists.txt`, `TrainDeeploy/DeeployTest/testUtils/deeployMezoRunner.py`, `TrainDeeploy/Deeploy/DeeployTypes.py` (`inputs_elems`), `TrainDeeploy/TargetLibraries/GAP9/CMakeLists.txt` (`-O3` list) |

**Commits** (branch `feat/fusion_full_network` on TKCW-W/TrainDeeploy, based on Run's `gap9-on-devel` at `a16a16d`):

| commit | content |
|---|---|
| `5d298d5` | exp7: fused two-input QZO conv for blocks 1–4 on the cluster (control kernel, CHW-order perturbation, optimisations; Siracusa 2.44×) |
| `0b21c7c` | exp7: GAP9 + NE16 for blocks 1–4, pipelined `NE16DualConv2in`, engine choice per block |
| `55e081f` | exp8: whole-network dual-pass graph (`ZODualPassFusion`, new nodes, harness), bit-exact on all targets, de-interleave, analyses |
| `596268e` | exp8: n_accum 4 step, 120-epoch round prediction, QZO+ECO on the fused graph |
| `7ec96b7` | this guide, SpeechNet_Test fixtures, NE16 L3 test-data fix, CI gates |
| `82dca6e` | clearer variable names in the kernels (machine code unchanged) |

Run's board work on top of it is on `silentwear-zo/gap9-on-devel` (`204a92a` … `fe1ffa2`: NE16 silicon fix, Dequant →
BN fold, board harness, power window). exp9 is on branch `feat/memory_opt` on TKCW-W/TrainDeeploy, based on
`fe1ffa2`:

| commit | content |
|---|---|
| `9f45baa` | exp9: `FuseFrozenBNTailPass` + `PULP_FrozenBNPoolReluQuantHWC_i8_i8`, `ZODualPassFusion` phantom-consumer fix, exp9 scripts and lowered graphs |
| (this update) | this guide's exp9 sections, exp9 logs and summaries |

Each commit message lists the details (files, decisions, bugs, verified numbers).

---

**Companion guides.** `00_Overview.md` (entry point, environment, checkpoint rule, experiment index),
`01_BP_Guide.md` (back-propagation), `02_ZO_Guide.md` (float zeroth-order), `03_QZO_Guide.md`
(quantized zeroth-order), `04_FusedQZO_NE16_Guide.md` (fused dual-pass kernels and NE16, block 0),
`05_Fusion_Full_Network_Guide.md` (this guide: blocks 1–4, the full dual-pass network and its memory optimisation).
