# 04 — Fused dual-pass QZO convolution kernels (PULP cluster) and the NE16 accelerator on GAP9

Written 2026-10-12. Part of the handover set indexed by `TrainDeeploy/HandoverDocuments/00_Overview.md`.
Paths are relative to `/Users/qiwenwu/ETH/`; in-container paths appear only inside command blocks.

**Summary.** A zeroth-order (ZO) training step evaluates the network twice, at θ+εz and θ−εz.
For SpeechNet's first convolution (block 0, the single most expensive node of a QZO step) the two
passes share their input, their im2col windows and all loop scaffolding; only the perturbation sign
differs. Part A of this guide documents a cluster kernel, `PULP_DualConv2d_i8_i8_i8_HWC`, that
computes both passes in one traversal and emits them channel-stacked as one tensor; it is bit-exact
against the two-pass reference and 1.88× faster on the fixture (2.16× on the conv node) than two
pulp-nn passes on Siracusa, and a control kernel (`FlatConv`) splits that gain into 1.49× structure
and 1.45× fusion. Part B documents the NE16 accelerator work on GAP9: the exp16a–c bring-up (per-tap
1×K route, dense 3×3 chunks, on-device bit-serial weight encoding, signed-input bias correction, all
five SpeechNet convs bit-exact but 4.3–7.5× slower than the cluster), and exp6, which ports the GAP9
SDK's im2col route (cluster-built column buffer plus one NE16 1×1 job) and fuses both ε-passes into
one NE16 job with Cout = 16: bit-exact, 2.71× over two pulp-nn passes on the GAP9 cluster and 1.71×
over the fused cluster kernel on the same chip.

## Status as of 2026-10-12

| item | state |
|---|---|
| Fused cluster kernel (`RequantizedDualConv`, Siracusa) | works, block 0 only; `Errors: 0 out of 157024`, 910,763 cycles (exp5) |
| Control kernel (`FlatConv`, switchable on every int8 `RequantizedConv`) | works; tree left on the pulp-nn default (exp5a) |
| Level-1 trace breakdowns, tiling profile, double-buffer test | done (exp5a_conv_breakdown, exp5_dualconv_breakdown, exp5b) |
| NE16 per-tap and dense-3×3 routes (exp16a/b), on-device encode and signed-input bias (exp16c) | all five SpeechNet convs bit-exact on NE16; full QZO step on GAP9 bit-identical to cluster-only; slower than the cluster |
| NE16 im2col route, single pass (exp6 task 1) | works, block 0: 379,969 cycles/pass, 0/78,512 |
| NE16 im2col route, fused (exp6 task 2) | works, block 0: 541,345 cycles both passes, 0/157,024 |
| Blocks 1–4 on either fused path | not started; the weight-perturbation order of both fused kernels is correct only for Cin = 1 (§8). **Update 2026-10-14: done, with the full dual-pass network — see `05_Fusion_Full_Network_Guide.md`** |
| Exporter support (Onnx4Deeploy emitting the fused / NE16 nodes) | absent; every fixture is hand-built by a builder script (§3) |
| Checkpoint | every number here derives from `Tests/Models/Training/SpeechNet/speechnet_qzo19_train` (artifacts_reference weights, paper-comparable), except the exp16 single-layer fixtures, which derive from `speechnet_qzo12_train` (stale `artifacts/` weights, cycle counts unaffected, accuracies not quoted) |

---

## 1. Concept

### 1.1 Why fuse two ε-passes

QZO on the device (`DeeployTest/Platforms/*/src/deeploymezotest.c`) runs `RunTrainingNetwork()`
twice per mini-batch, flipping the global `perturbation_sign` between the runs. For block 0 each
run executes four nodes: `RQSPerturbRademacher` on the int8 weight, `RQSPerturbRademacher` on the
int32 requant add, a weight `Transpose` (NCHW to NHWC) and the fused `RequantizedConv`
(`pulp_nn_conv_i8_i8_i8`, produced from `Conv + RequantShift` by `PULPConvRequantMergePass`).
Block 0's input is the quantized EMG window, which is the same in both passes; only the weight and
the requant add differ. So every per-pixel cost that does not depend on the weight set (gathering the
receptive field, the loop control, parameter reloads) is paid twice for identical data.

Two measurements decided the design (both on Siracusa, exp3 and exp4):

1. **The deployed kernel is fetch-bound, not compute-bound.** exp3
   (`TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp3_as-is_QZOConv_microbench/`)
   measured 22.8 % of the conv node's cycles (151,362 of 664,646) as L0 instruction-refill stalls
   (`logs/run_microbench.log`, `results/summary.txt`; the PCER field printed as "I-cache Misses" is
   stall *cycles*, "cycles waiting for instruction fetches", read on core 0 only).
   The cause, confirmed with GVSoC instruction and I-cache traces: Siracusa's cluster instruction
   cache (GVSoC `cluster.json`, `icache/config`) gives every worker core a private L0 of 16-byte
   lines, 8 sets × 4 ways = 32 lines = 512 B (refill latency 2), backed by a shared L1 of two
   2 KB banks; lines 128 B apart share a set. `pulp_nn_conv_i8_i8_i8` (2,340 B) calls
   `pulp_nn_matmul_i8_i8_i8` (2,272 B) once per two output pixels, plus a `noinline` im2col helper
   and `pulp_nn_bn_quant_i8`; the per-pixel-pair route touches 86 16-byte lines (64 of them the
   matmul) against the 32-line L0. In the warm state (pairs 100, 300, 500 of core 0's 700) 85 of the
   86 lines are refetched every pair, some two or three times within a pair: 112–121 refills and
   ≈380 cycles of waiting per pair, i.e. about 3.2 cycles per refill (the shared L1 and interconnect
   behind the private cache, with eight cores missing on the same lines). Control: the flat
   single-function kernel (`FlatConv`, §3.1) on the same fixture walks 23 lines per pixel, which fit,
   and a warm pixel refills nothing (511 refills on core 0 over the whole run against 123,394).
   Evidence: the cut trace windows `logs/debug/pair{100,300,500}.log` and
   `logs/debug/ctl_pixel{100,300,500}.log`, tallied by `logs/debug/tally_pair.py <disassembly> <window>`
   (the full traces are git-ignored; regenerate with `--trace=pe0/insn --trace=icache/l0_bank0` on the
   GVSoC command line of the build and cut between consecutive occurrences of the matmul's entry PC).
2. **Interleaving the two passes' arithmetic alone is worth about 3 %.** exp4's host microbenchmark
   (`DeeployTest/Platforms/Siracusa/src/dualpass_bench.c`, opt-in `-D DUALPASS_BENCH=ON` in
   `DeeployTest/Platforms/Siracusa/CMakeLists.txt`) measured 1.03× for block 0's shape (inner
   length 1, 8 accumulators; `experiments/dual_pass_fusion/exp4_interleave_dualpass/results/step1_microbench.txt`,
   row `BLOCK 0`, 1.031). The inner loop already has enough ILP; the prize is walking one
   call-free function once instead of a 4.7 KB call chain twice.

### 1.2 The DualConv algorithm

One ONNX node `RequantizedDualConv` takes the unperturbed weight W, the unperturbed requant add,
the requant multiplier `mul`, the two Rademacher multipliers `wmul` / `bmul`, and the two perturb
nodes' `seed` / `idx` / `div` attributes. The kernel runs in two phases.

**Phase 1 (per tile, every core).** Core `c` perturbs its chunk of W and of `add`, both signs from one
Rademacher draw, into an L1 scratch:

```
chunk    = (size >> log2(NUM_CORES)) + (size % NUM_CORES != 0)        per core [chunk*c, +chunk)
seed     = ((seed_attr + perturb_seed_base) + NUM_CORES*idx + core_id) ^ (tile_seed_offset * 0x9E3779B1)
rng      = seed*1664525 + 1013904223 ; one Xorshift32 draw per <= 32 elements, LSB first
r        = bit ? +1 : -1 ;  m = M[i / channel_width]
prod     = r*m
noise+   = (prod + 2^(S-1)) >> S ;   noise- = (2^(S-1) - prod) >> S
W+[i]    = clamp(W[i] + noise+, -127, 127) ;  W-[i] = clamp(W[i] + noise-, -127, 127)
add+[c]  = add[c] + noise+ ; add-[c] = add[c] + noise-        (int32, no clamp)
```

This is exactly the chunking, seeding, rounding and clamp of `ApplyPerturbQuantRademacher_CHW` /
`_i32` (`TargetLibraries/PULPOpen/src/RandomNoiseQuant.c`), so the perturbed weights are
bit-identical to the two standalone nodes'. A `pi_cl_team_barrier()` publishes the scratch.

**Phase 2 (one traversal).** The 9,814 output pixels are split as a flat index over the 8 cores
(ceil(N/8) each: 99.3 % balance versus pulp-nn's 4/4/4/2 row split at 87.4 %). Per pixel the kernel
gathers the K = ch_in·kh·kw receptive-field bytes once (zeros for padded taps), then for each output
channel runs `__builtin_pulp_sdotsp4` (SumDotps4) against W+[c] and against W−[c], requantises each
inline as `clips8((acc*mul[c] + add±[c]) >> shift)` in int32 with truncating shift (the arithmetic of
`pulp_nn_bn_quant_i8`), and writes 16 contiguous bytes: channels 0–7 = A+, 8–15 = A−. The output is
one tensor `[1, 16, 14, 701]`; stacking on the channel axis keeps Deeploy's single-output tiler on
its normal path (`wrapTilingSolution` asserts one output).

No arithmetic is shared between the passes. The integer alternative (one base accumulation X·W plus
a signed correction X·Δ) is exact only where the int8 clamp does not bite. For block 0's weight
(S = 15) the noise is a per-channel constant `q_c = round(M_c / 2^15)` = `[3,4,3,4,3,4,3,2]`,
symmetric in sign, so `W± = W ± q_c·r` except at the clamp: 8 of block 0's 32 weights sit at ±127
and cannot move outward, and which of them get pushed outward depends on the draw (in exp3's +ε draw
3 were clamped; `build_qzo_conv_fixture.py` prints "29/32 elements moved"). A correction scheme would
therefore need a fix-up on up to 8 saturated taps per pass, and the requant
`(acc·mul + add) >> 16` with int8 saturation is non-linear anyway, so each pass must be requantised
from its own accumulator. The kernel computes both dot products in full; that is what keeps it
exact. (A side fact from the same analysis: block 0's bias perturbation is a no-op at ε = 0.01,
because with S = 16 and every bias multiplier ≤ 12,866 < 2^15, `(±M + 32768) >> 16 = 0` for all
eight channels and both signs; the node still costs 2,084 cycles to add zero. Whether this holds for
blocks 1–4 was not checked.)

### 1.3 The NE16 version of the same idea

NE16 has three filter modes (1×1, 3×3, 3×3-depthwise), reads 16 input channels per pass (TP_IN) and
retires 32 output channels per pass (TP_OUT) in 3×3 spatial subtiles. SpeechNet's 1×K kernels are
not a mode. The GAP9 SDK's answer for a 2-D map with a 1×K filter (`Ker_MM_Conv1DSmallv2_NE16`,
`CNN_BasicKernels_NE16.c`) is to let the cluster
cores build a column buffer (for every output pixel, the Cin·K input values under the window as one
row) and let NE16 run one 1×1 convolution over Cin·K "channels": the taps become channels.
Everything NE16 cannot do is done while copying: padding (the cores write the pad value; NE16's 1×1
mode cannot pad and GVSoC's guard is commented out) and the +128 offset a signed activation needs
(NE16 reads inputs as unsigned; GAP9's NE16 has no signed-input bit).

**How the GAP9 SDK handles 1×K convolutions.** The `deeploy:gap9` image carries the full GreenWaves
SDK, including its NE16 code generator (AutoTiler) and hand-written kernel library, under
`/app/install/gap9-sdk/tools/autotiler_v3/`: `CNN_Generators_NE16/CNN_Generators_NE16.c` (layer
generators and the route decision), `CNN_Libraries_HWPE/CNN_BasicKernels_NE16.c` (the kernels),
`CNN_Libraries_HWPE/hal_ne16.h` (register HAL) and `ne16_archi.h` (`HW_SIZE 3`). The generator
`CNN_ConvolutionNE16()` (`CNN_Generators_NE16.c:1491`) chooses among three routes, in order:

| route | when | what |
|---|---|---|
| (a) linear | the filter covers the whole input, no padding | the conv becomes a fully-connected layer over `InFeat·Fcx·Fcy` inputs (channel folding in its extreme form) |
| (b) im2col ("MM") | `EnableIm2Col` set, conv neither depthwise nor pointwise | `CNN_MM_ConvolutionNE16`: cores build the column buffer, NE16 runs 1×1 over `Cin·K` channels. `Ker_MM_Conv1DSmallv2_NE16` (`CNN_BasicKernels_NE16.c:4681` → `:4551`) is registered for `Fx = any, Fy = 1`; `ColBuffSize = InFeat·Fx`, `Nb_KI = ceil(ColBuffSize/16)`; two column buffers of `HW_SIZE²·ColBuffSize` bytes alternate so the cores prepare job n+1 while NE16 computes job n (a `Seq` variant disables the overlap to save memory and power) |
| (c) native | otherwise; `CNN_ConvolutionNE16_Internal` (`:1169`) turns a conv with `Height == 1 && Fcy == 1 && Fcx > 1` into `KOP_CONV1D` | `KerConv1D_StrideS_NE16` (`CNN_BasicKernels_NE16.c:1643`): one NE16 job per tap, input pointer stepped by `Cin` per tap, weight pointer by one encoded tap block, `streamin` set on every job after the first, invariant registers written only for the first two jobs, jobs pipelined through the 2-deep queue |

Route (c) is algorithmically the same as our exp16a per-tap route (§4.4), found independently. Its
`Height == 1` guard means it applies only to genuine 1-D signals; SpeechNet's maps are 2-D (block 1
is 14×87 with a 1×16 kernel), so for our shapes the SDK takes route (b). That is the route exp6
ported (§4.5), and it is why it suits SpeechNet: folding the taps into channels fills NE16's 16 input
lanes, which our convs never do on their own.

| block | kernel | Cin | Cin·K | TP_IN groups after folding | per-tap jobs | folded jobs |
|---|---|---|---|---|---|---|
| 0 | 1×4 | 1 | 4 | 1 (¼ full) | 4 | 1 |
| 1 | 1×16 | 8 | 128 | 8, exactly full | 16 | 1 |
| 2 | 1×8 | 16 | 128 | 8, exactly full | 8 | 1 |
| 3 | 7×1 | 16 | 112 | 7, exactly full | 7 | 1 |
| 4 | 7×1 | 32 | 224 | 14, exactly full | 7 | 1 |

The price is the column build on the cores (and `9·Cin·K` bytes × 2 buffers of L1 in the SDK's
double-buffered form). A third SDK trick, remapping a genuinely 1-D W axis onto NE16's 3×3 output
grid to retire 9 pixels per job, does not apply to 2-D maps.

Fusing two passes on NE16 is stacking output channels: one job with Cout = 16 (rows 0–7 = W+,
8–15 = W−) over the same column buffer, with requant vectors `[mul; mul]` and `[add+'; add−']`. The
hardware retires 32 output lanes per pass whether 8 or 16 are used, so the second pass rides in idle
lanes: the fused NE16 job cost +30 % over the single-pass job (136,618 vs 104,822 cycles,
`exp6_NE16_fusion/logs/t2_run_microbench.log` and `t1_run_microbench.log`), not +100 %.

---

## 2. The graph and the fixtures

**No exporter emits these graphs.** A grep of `Onnx4Deeploy/onnx4deeploy` for `RequantizedDualConv`,
`NE16Im2ColU8`, `NE16DualPerturbEncode` or `NE16WeightEncode` finds nothing. Every fixture is a
single-layer kernel test built by a script from the QZO training fixture
`TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_qzo19_train` (artifacts_reference
weights; see `03_QZO_Guide.md`). Making Onnx4Deeploy emit the fused node is an open item (§8).

### 2.1 Builders and fixtures (all under `TrainDeeploy/DeeployTest/`)

| builder (`experiments/dual_pass_fusion/…`) | fixture (`Tests/Kernels/…`) | graph | reference | container |
|---|---|---|---|---|
| `build_zo_conv_fixture.py --block {0,1}` | `FP32/ZO_Conv`, `FP32/ZO_Conv_Block1` | fp32 `PerturbRademacher`×2 + `Conv` lifted from `speechnet_zo_train_b1_fold3_ref` | host mirror `zo_host_ref.py` (Xorshift32, LCG warm-up, per-node z-stream; self-test vs `Tests/Kernels/FP32/ZOPerturb_QW`) | host python (onnx, numpy) |
| `build_qzo_conv_fixture.py` | `Integer/QZO_Conv` | `RQSPerturbRademacher`×2 + `Conv` + `RequantShift` (the device merges the last two) | Onnx4Deeploy `run_onnx_graph`, asserted equal to the full graph's tensor | `agitated_hugle` |
| `build_dual_conv_fixture.py` | `Integer/QZO_DualConv` (+ `reference_plus_minus.npz`) | one `RequantizedDualConv` node (domain `mezo`), 6 inputs, output `[1,16,14,701]` | exp3's 4-node graph through the oracle twice (as-is for A+, `wmul`/`bmul` negated for A−); A+ asserted equal to exp3's `outputs.npz` | `agitated_hugle` |
| `build_ne16_im2col_fixture.py` | `Integer/QZO_Conv_NE16` (+ `col_golden.npz`) | `RQSPerturbRademacher`×2, `NE16SignedInputBias`, `NE16WeightEncode(ne16_fold=1)`, `NE16Im2ColU8`, `Conv(1×1, channels_first=0, ne16_weight_preencoded=1, weight_offset=-128)`, `RequantShift(channels_first=0)`, `Transpose(0,3,1,2)` | exp3's `outputs.npz` copied | host python |
| `build_ne16_dual_fixture.py` | `Integer/QZO_DualConv_NE16` | `NE16DualPerturbEncode`, `NE16DualBias`, `NE16Im2ColU8`, `Conv(1×1, Cout 16)`, `RequantShift(mul2, add2)`, `Transpose` | exp5's `outputs.npz` copied | host python |
| `experiments/deliverable/exp16_NE16_GAP9/build_block1_pw_fixture.py` | `Tests/Models/NE16/speechnet_b1_pw_rq` | block 1 reduced to 1×1, constant weight (STEP 2a plumbing test) | `run_onnx_graph` | `agitated_hugle` |

Three contracts every builder respects:

- the npz is consumed positionally by `testMVP.py` (`[inputs[x] for x in inputs.files]` zipped onto
  the graph inputs), so the write order is load-bearing;
- `RequantShift`'s `div` / `n_levels` / `signed` must stay TENSOR-typed attributes (Deeploy's
  `RQSParserInterface` unpacks them with `_unpack_const`), so the builders copy the source node's
  `AttributeProto`s verbatim;
- the requant add stays a runtime tensor (perturbed), never a baked constant: `PULPConvRequantMergePass`
  bakes `+div/2` only into a constant add (exp16c's `--variable-bias` lesson, §7).

### 2.2 The `Tests/Models/NE16/` family (exp16 single-layer fixtures)

Built by `experiments/deliverable/exp16_NE16_GAP9/exp16a_PW_single_layer/build_fixtures.py`
(block 1, 1×K only; names `b1_ref_1x{K}`, `b1_1x{K}_ne16`, `b1_1x{K}_devenc`, `b1_1x{K}_pertenc`,
`b1_1x{K}_dense3x3`, `b1_pw{K}_plain`, `b1_pw{K}_ne16`, plus a `--suffix`: `_s` = 4×24 crop,
`_nhwc`, `_dev`, `_lp`/`_ln` = L+/L− via `--neg-pmul`, `_d33`, `_k3`) and by
`exp16c_PW_single_layer/build_fixtures.py` (any block, both tap axes; names `b{N}_ref`, `b{N}_ne16`,
`b{N}_plain` with `--plain` = no hand-authored NE16 attributes, `_vb` = `--variable-bias`). Current
ones are those whose runs are logged in the exp16 folders (§9): `b1_1x16_ne16_nhwc`, `b1_1x16_dense3x3_d33`,
`b1_1x16_devenc_dev`, `b1_1x16_pertenc_lp/_ln`, `b{0,1,2,3,4}_ne16` / `_ref`, `b3_plain`, `b4_plain`,
`b3_plain_vb`, `b3_ref_vb`. The rest (`b1_ne16`, `b1_pw_*`, `b1_pw2_*`, `b1_pw3_*`, `_s` crops) are
earlier iterations kept for reproducibility. None is registered in a pytest config
(`DeeployTest/test_gap9_tiled_config.py` has no NE16 or QZO entries); they run through the CLI
runner only. All derive from `speechnet_qzo12_train` (stale weights; only cycles and bit-exactness
are quoted from them).

---

## 3. Part A — the cluster side, in pipeline order

### 3.1 Kernels (C)

| file | symbol | ours? | what |
|---|---|---|---|
| `TrainDeeploy/TargetLibraries/PULPOpen/src/DualConv.c` | `PULP_DualConv2d_i8_i8_i8_HWC` | ours (exp5) | phases 1 and 2 of §1.2; argument order mirrors `pulp_nn_conv_i8_i8_i8` (Deeploy's `dim_im_in_y` is the kernel's `dim_in_x`); scratch layout `W+ | W- | add+ (4-aligned) | add- | one Kpad-byte patch per core`; `DUALCONV_DEBUG_PRINT` dumps core 0's scratch |
| `TrainDeeploy/TargetLibraries/PULPOpen/inc/kernel/DualPerturb.h` | `dc_xorshift32`, `dc_chunk`, `dc_perturb_i8`, `dc_perturb_i32` | ours (exp5, moved out in exp6) | the phase-1 helpers, shared verbatim with the GAP9 NE16 preparation kernels; carries the clang-15 rule (§7) |
| `TrainDeeploy/TargetLibraries/PULPOpen/src/FlatConv.c` | `PULP_FlatConv2d_i8_i8_i8_HWC` | ours (exp5a) | the control kernel: DualConv's phase 2 with one weight set and no phase 1; a mechanical deletion, deliberately untuned (fairness rule in the file header) |
| `TrainDeeploy/TargetLibraries/PULPOpen/src/Im2ColU8.c` | `PULP_Im2ColU8_NCHW_to_HWC` | ours (exp6) | column-buffer builder for the NE16 route (§4.3); compiled into both `deeploypulp` and `deeploygap9` (the GAP9 `CMakeLists.txt` globs `../PULPOpen/src/**`) |
| `TrainDeeploy/TargetLibraries/PULPOpen/src/PerfRecords.c` | `perf_records[]`, `perf_record_count` | ours (exp6) | L2 storage for the `--profileMicrobenchmark` records replayed by the GAP9 fabric controller |
| `TrainDeeploy/TargetLibraries/PULPOpen/inc/kernel/{DualConv,FlatConv,Im2ColU8}.h` | prototypes | ours | included from `TrainDeeploy/TargetLibraries/PULPOpen/inc/DeeployPULPMath.h` (lines 29–31, `-- QW`) and from `TrainDeeploy/TargetLibraries/GAP9/inc/DeeployGAP9Math.h` (lines 33–36) |
| `TrainDeeploy/TargetLibraries/PULPOpen/inc/perf_utils.h` | `PERF_MASTER_CORE`, `perf_record_t`, `perf_bench_dump_records()` | upstream, modified (exp6) | `PERF_MASTER_CORE` = `ARCHI_CLUSTER_MASTER_CORE` (8) when defined (GAP9), else 0 (Siracusa); `perf_bench_print` now also appends to the record table |

Two things in `DualConv.c` worth reading once: the requant line
`(int32_t)((uint32_t)ap * (uint32_t)pMul[c] + (uint32_t)addp[c]) >> out_shift` (int32 wrap, arithmetic
shift, no rounding term, then `__builtin_pulp_clip_r(y, 127)`), and the gather, which walks
`(ky, kx, ci)` because the kernel consumes Deeploy's NHWC-transposed weight `[co][kh][kw][ci]`.

### 3.2 Deeploy PULPOpen integration (all `-- QW` marked, additive)

| stage | file | symbol | what |
|---|---|---|---|
| layout pass | `TrainDeeploy/Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/LoweringOptimizationPasses.py` lines 255, 290 | `_NCHWtoNHWC_fun` | `RequantizedDualConv` added to the conv lists, so the activation gets its NCHW→NHWC transpose, the weight its `[8,1,1,4]→[8,1,4,1]` transpose and the output its NHWC→NCHW transpose; inputs[1:] = weight, mul, add, wmul, bmul |
| parser | `TrainDeeploy/Deeploy/Targets/PULPOpen/Parsers.py` line 674 | `PULPDualConv2DParser(RQSConv2DParser)` | requires 6 inputs, `shift`, and `seed_w/idx_w/div_w/seed_b/idx_b/div_b`; computes `log2D_w`, `log2D_b`, records `ch_out_pass` = weight rows (the output tensor's `ch_im_out` is twice that) |
| layer | `TrainDeeploy/Deeploy/Targets/PULPOpen/Layers.py` line 46 | `PULPDualConvLayer(RQSConvLayer)` | `computeShapes` returns the graph's explicit shapes (the RQS conv layer would mis-propagate the 2× channel count onto mul/add); `computeOps` = 2× |
| binding | `TrainDeeploy/Deeploy/Targets/PULPOpen/Bindings.py` line 634 | `PULPDualConv2DBindings` | `PULPConvChecker` over `[int8, int8, int32, int32, int32, int32] -> [int8]`, `ForkTransformer` |
| template | `TrainDeeploy/Deeploy/Targets/PULPOpen/Templates/DualConvTemplate.py` | `_DualConvTemplate`, `referenceTemplate` | emits `seed_w = (seed_w + perturb_seed_base) + NUM_CORES*idx_w` (core_id added in the kernel), `tile_xor = tile_seed_offset*0x9E3779B1`, `eps_scale` from the `ZORuntime.h` globals, then one kernel call with `ch_im_out >> 1`; `computeTransientBuffersSize` sizes the scratch `off_addp + 8*cout + 8*Kpad` (must match `DualConv.c`) |
| tile constraint | `TrainDeeploy/Deeploy/Targets/PULPOpen/TileConstraints/DualConvTileConstraint.py` | `DualConv2DTileConstraint` | geometry of `RQConv2DTileConstraint` plus `outC == 2*wO`; policy pins `wO`, `outC`, `inC`, `wC` whole (A+ and A− must stay in one tile; seed reproduction assumes one chunk-set); `serializeTilingSolution` asserts `COffset == 0 and CSize == 2*wN`, reuses `Conv2DTileConstraint.computeInputCube` |
| tiler | `TrainDeeploy/Deeploy/Targets/PULPOpen/Tiler.py` line 267 | `PULPDualConv2DTilingReadyBindings` | binds the constraint |
| platform | `TrainDeeploy/Deeploy/Targets/PULPOpen/Platform.py` lines 152–156, 176 | `DualConv2DMapper`; `'RequantizedDualConv': _PULPDualConvLayer([DualConv2DMapper])` in `PULPMapping` | |
| control kernel | `Templates/FlatConvTemplate.py`, `Bindings.py` line 645 `PULPRQSConv2DFlatBindings`, `Tiler.py` line 273 `PULPRQSConv2DFlatTilingReadyBindings` (same `RQConv2DTileConstraint`, so identical tiling) | | bound to the ordinary `RequantizedConv` |
| profiling | `TrainDeeploy/Deeploy/Targets/PULPOpen/CodeTransformationPasses/PULPMicrobenchmark.py` | `PULPMicrobenchmark` | pre/post templates now guarded by `pi_core_id() == PERF_MASTER_CORE` (was `== 0`) |

**The KERNEL SWITCH (standing hazard).** `TrainDeeploy/Deeploy/Targets/PULPOpen/Platform.py` lines 95–101:

```python
Conv2DMapper = NodeMapper(PULPConv2DParser(), PULPRQSConv2DTilingReadyBindings)        # default: pulp-nn  -- QW exp5a
# Conv2DMapper = NodeMapper(PULPConv2DParser(), PULPRQSConv2DFlatTilingReadyBindings)  # exp5a: flat kernel -- QW exp5a
```

The current default is the pulp-nn line (verified in the tree). Flipping the comment changes the
kernel of every int8 `RequantizedConv` on the PULP target, i.e. every QZO network, not just the
fixture; it must never stay flipped. Revert proof: `QZO_Conv` at 857,891 cycles with
`pulp_nn_conv_i8_i8_i8` back in `Network.c` (`exp5a_breakdown/logs/run_plain_default_after_revert.log`).

### 3.3 Tiling of the fused node (what the tiler actually does)

On Siracusa at `--l1 128000` the fused conv runs as 2 tiles (536 and 165 output columns, 7,504 and
2,310 pixels) because the 157,024 B output no longer fits L1's headroom next to the input; the
output transpose runs as 3 tiles. exp3's single-pass conv was 1 tile, its transpose 2. The tile
constraint never cuts channels; it cuts W (the long axis). Per-tile cost of phase 1 plus setup is
about 1,600 cycles (exp5b, `F + c*pixels` fit: c = 62.8 cycles/px).

### 3.4 Harness, runners and test registration (Siracusa)

| file | role |
|---|---|
| `TrainDeeploy/DeeployTest/Platforms/Siracusa/src/deeploytest.c` | inference harness; int8 outputs are compared exactly (`if (diff)`), fp32 at `OUTPUT_TOL` (default 1e-4, overridable with `-D OUTPUT_TOL=`); prints `Runtime: N cycles` from the cluster timer and `Errors: E out of N` |
| `TrainDeeploy/DeeployTest/Platforms/Siracusa/src/dualpass_bench.c` | exp4's interleaving microbenchmark; own `main()`, enabled by `-D DUALPASS_BENCH=ON`; the runner then reports `-1 errors out of -1` by design |
| `TrainDeeploy/DeeployTest/deeployRunner_tiled_siracusa.py` | CLI entry (`testUtils/deeployRunner.py::main`, platform Siracusa, tiled); flags used here: `--l1`, `--l2`, `--cores`, `--profileMicrobenchmark`, `--profileTiling`, `--doublebuffer`, `--skipsim`, `--input-type-map`, `--input-offset-map` |
| `TrainDeeploy/DeeployTest/testMVP.py` | tiled code generator; `_resolveInputTypes` honours `--input-type-map` / `--input-offset-map` (added for these fixtures: value inference types an all-32768 int32 vector as `uint16_t`) |
| `TrainDeeploy/DeeployTest/testUtils/deeployRunner.py` lines 279–289 | `--cores` is forwarded to the generator on tiled runs too (the earlier `if not tiling:` scoping sized the im2col scratch for one core while eight wrote into it) |
| `TrainDeeploy/DeeployTest/test_siracusa_tiled_config.py` | registers `Kernels/FP32/ZO_Conv`, `ZO_Conv_Block1`, `Kernels/Integer/QZO_Conv`, `QZO_DualConv` at L1 128000; `KERNEL_CMAKE_ARGS` pins `OUTPUT_TOL=1e-3` on the fp32 pair; `KERNEL_GEN_ARGS` pins the int32 requant add on the QZO pair (`--input-type-map blocks.0.conv.bias_rqsadd=int32_t --input-offset-map blocks.0.conv.bias_rqsadd=0`) |

### 3.5 Breakdown tooling

| file | what |
|---|---|
| `experiments/dual_pass_fusion/exp5_fused_dual_conv/parse_microbench.py` (copies in exp3, exp5a, exp6) | turns a `--profileMicrobenchmark` log into `per_node.{csv,md}`; the exp6 copy keeps only the text after `[perf] records=` |
| `experiments/dual_pass_fusion/exp5a_breakdown/exp5a_conv_breakdown/conv_breakdown.py` | level-1 attribution of a GVSoC `pe0/insn` trace window: cost of an instruction = next stamp − own stamp; category by hand-read address range (`ADDR["FlatConv"]`, `ADDR["DualConv"]`) and, as cross-check, by C line from `llvm-objdump -d -l`; `ADDR_OFFSET=<bytes>` shifts the address map when the same kernel sits elsewhere (5012 for the exp5a microbench build) |
| `experiments/dual_pass_fusion/exp5_fused_dual_conv/exp5_dualconv_breakdown/conv_breakdown.py` | the same script, used with `ADDR_OFFSET=4088` for the exp5 microbench build; `extract_windows.sh` next to each copy cuts the windows from the raw trace |

---

## 4. Part B — the NE16 side, in pipeline order

### 4.1 Provenance of `TrainDeeploy/Deeploy/Targets/NE16/`

The directory was vendored verbatim from upstream Deeploy PR #183 (`runwangdl:gap9-ne16`, head
`45ca06c`) in commit `f3ff035` (2026-09-10) and then extended. (`git log --follow` shows older
"first commits" for the vendored files because it follows the near-identical `Targets/Neureka/`
files the PR was copied from.)

| file | status | notes |
|---|---|---|
| `Engine.py` | vendored, modified | `_weightAcceptable` (runtime weight with `ne16_weight_preencoded=1`), `is1xKConv` delegating to `ne16_1xkAdmissible`, `enable1xK` flag; mapping order `[3x3Chunk, 1xK, PW, DW, Dense]` for `Conv`; `canExecute` = 1×K (if enabled) or PW (always) or 3×3/DW (if `enable3x3`) |
| `Platform.py` | written at vendoring (ours) | `NE16Platform(GAP9Platform)` with engines `[NE16Engine("NE16"), GAP9ClusterEngine("GAP9Cluster")]`; uses the plain cluster include list (the PR's trimmed list is kept commented; the `NE16_REG_*` collision it avoided cannot occur in our tree); `MemoryNE16Platform`, `MemoryNE16PlatformWrapper`; `NE16Optimizer` = PULPOptimizer passes + `RequantizedGemmToPwPass` |
| `Deployer.py` | vendored, modified | inserts `NE16Prepare1xKPass` at index 1 (after the first `EngineColoringPass`, before `PULPNCHWtoNHWCPass`); appends `ConvEngineDiscolorationPass`, `NE16OptimizationPass`, then re-runs the transpose clean-up chain (PR #183's 1.48× fix) |
| `Parsers.py` | vendored, modified | `NE16Conv2DBaseParser` requires `channels_first == False`, `dilations == [1,1]` and a `weight_offset` attribute; six PR parsers (`NE16{PW,DW,Dense}Conv2DParser`, `NE16RQS*`); ours: `NE161xKConv2DParser` (keyed on `ne16_taps`, publishes `ne16_halo = taps-1`), `NE163x3ChunkConv2DParser` (keyed on `ne16_chunks`, `ne16_halo = 3*(chunks-1)+2`) |
| `Bindings.py` | vendored, modified | PR bindings for PW/DW/Dense (`PULPConvChecker` RQS variants with int32 mul/add, `ConvChecker` plain variants with int32 output); ours: `NE161xKConv2DBindings`, `NE163x3ChunkConv2DBindings` (2 inputs, uint8 pre-encoded weight, int32 output: streamin forces 32-bit) |
| `Tiler.py` | ours | tiling-ready bindings for all of the above plus `NE16WeightEncode`, `NE16SignedInputBias`, `NE16Im2ColU8`, `NE16DualPerturbEncode`, `NE16DualBias` |
| `Templates/ConvTemplate.py` | vendored (2 `-- QW` comments) | `NE16ConvTemplate` base; `NE162DPWConvTemplate` (`NE16_KERNEL_SHAPE 1`, `NE16_SUBTILE_OUTPUT_CHANNEL 32`, `getConf0` sets `2<<5` PW mode, `1<<15` layerwise weight offset, `2<<21` for 32-bit out, and still `1<<26` for `input_signed`, the phantom bit), `NE162DDWConvTemplate`, `NE162DDenseConvTemplate`; `getCounters` computes the ten subtile fields (`nKo = _getNumTiles(cout,32)`, `nKi = _getNumTiles(cin,16)`, 3×3 spatial, border remainders); the template emits one `ne16_task_t` then `ne16_nnx_dispatch_wait` / `ne16_nnx_dispatch` / `ne16_nnx_resolve_wait` (blocking, one job at a time). It still prints `// N-EUREKA Task Init` and carries the odd `task.data.cfg.weights_stride = (ne16_stride_t){...}` initialiser (line 352): an assignment expression written where a `.weights_stride = ...` designator belongs inside the `ne16_task_t` initialiser; it compiles under the relaxed GAP9 flags but is not what was meant, so fix it rather than copy it when extending the template |
| `Templates/Conv1xKTemplate.py` | ours (exp16a, exp16c, exp16c_SDK_port) | K dispatches from one `Conv(1×K)` node: tap j offsets `weights_addr` by one encoded tap block and `infeat_addr` by `ch_im_in` bytes (W taps) or by `dim_im_in_x_stride` (H taps, `ne16_tap_axis == 1`, a per-tile value); `conf0_streamin = conf0 | NE16_FLAG_STREAMIN (1<<14)` for j > 0; pipelined over the 2-deep queue (`dispatch_wait` per tap, one `resolve_wait` after the chain) |
| `Templates/Conv3x3ChunkTemplate.py` | ours (exp16b) | ceil(K/3) dense 3×3 dispatches with only the middle row populated, `infeat_addr += 3c*ch_im_in`; `ne16_filter_mask = 0` (masking was tried and made it worse) |
| `Templates/AllocateTemplate.py`, `OptimizationPasses/MemoryLevelAnnotationPasses.py` | vendored, dead | `ne16GenericGlobalInitTemplate`, `AnnotateNE16WeightMemoryLevel`: imported nowhere in `Deeploy/` or `DeeployTest/` (verified by grep); the `use_wmem` / `WeightMemory_SRAM` path is unreachable |
| `TileConstraints/NE16{Pointwise,Dense,Depthwise}Constraint.py`, `RequantHelpers.py` | vendored | geometry + `PerformanceHint` policies (H/W divisible by 3, ≤ 32 output channels whole) + per-tile subtile counters |
| `TileConstraints/NE161xKConstraint.py` | ours (exp16a/b) | halo `Wout = Win − halo` on the tapped axis with input overlap; pins the non-tapped axis when `ne16_chunks` is set; `serializeTilingSolution` uses `NE162DDenseConvTemplate.getCounters` with real bottom/right padding for the 3×3 variant and `NE162DPWConvTemplate.getCounters(…, 0, 0)` for the per-tap variant (the exp16b ±1-LSB bug) |
| `TileConstraints/NE16WeightEncodeConstraint.py` | ours (exp16c) | `NE16WeightEncodeTileConstraint`, `NE16SignedInputBiasTileConstraint`: every dimension pinned to `.Max()`, one tile |
| `TileConstraints/NE16Im2ColConstraint.py` | ours (exp6) | `NE16Im2ColU8TileConstraint` (row-tiled, columns whole, per-tile pads via `_span`), `NE16UntiledTileConstraint` base with `INPUT_KEYS` as a class attribute, `NE16DualPerturbEncodeTileConstraint`, `NE16DualBiasTileConstraint` |
| `TopologyOptimizationPasses/Passes.py` | vendored | `_weightEncode` (the host bit-serial encoder, reference for the C port), `NE16AdjustWeightMemoryLayoutPass` (constant weights only; returns early for a runtime weight), `NE16ReshapePointwiseConvolutionPass` (`NE16_SPATIAL_SUBTILE = 3`), `ConvEngineDiscolorationPass`, `NE16DwLayoutFixupPass`, `NE16OptimizationPass` |
| `TopologyOptimizationPasses/Prepare1xKPass.py` | ours (exp16c_SDK_port) | `ne16_1xkAdmissible` (1×K or K×1, group 1, dilation 1, stride 1, zero pads, input provably non-negative via `_inputIsNonNegative` or `ne16_unsigned_input=1`; `QW_NE16_ONLY` env var bisects by node name); `_ne16_prepare_1xk_fun` restacks a constant weight per tap on the host or inserts `NE16WeightEncode` for a runtime weight, sets `ne16_taps`, `weight_offset=-128`, `ne16_weight_preencoded=1`, and tags the consuming `RequantShift` `channels_first=1` |
| `WeightEncode.py` | ours (exp16c; `ne16_fold` in exp6) | `NE16WeightEncodeParser` (expects `(cout,cin,H,W)` with `H*W == ne16_taps`; with `ne16_fold=1` and `taps == 1` reinterprets `(cin,H,W)` as `cin*H*W` channels of a 1×1 weight; output `(taps*cout, cinMajor, 16)` uint8), template chunking `taps*cout` rows over the cores, `NE16SignedInputBiasParser`/template chunking `cout` |
| `Im2Col.py` | ours (exp6) | `NE16Im2ColU8Parser` (attrs `kernel_shape`, `pads` [t,l,b,r], `strides`, `ne16_input_offset`=128; derives `xor_mask = pad_value = offset & 0xFF`; checks `K == ch_in*kh*kw` and the output geometry), unsigned-output checker, template calling `PULP_Im2ColU8_NCHW_to_HWC`, `ForkTransformer` binding, layer |
| `DualPrep.py` | ours (exp6) | `NE16DualPerturbEncode` (inputs W_RAW, wmul; attrs seed_w/idx_w/div_w; output `(2*cout, cinMajor, 16)`), `NE16DualBias` (inputs W_RAW, wmul, add, bmul, mul; all six seed attrs + `ne16_input_offset`; output `(2*cout,)` int32); templates emit the same seed expression as `DualConvTemplate.py`; `computeTransientBuffersSize` is a static method (the tiler calls it on the class) and the parser stores `nodeName` because the symbolic representation reaches it before binding |

### 4.2 GAP9 target files

| file | what (all `-- QW` marked) |
|---|---|
| `TrainDeeploy/Deeploy/Targets/GAP9/Platform.py` lines 122–147, 236–247 | `GAP9Mapping` entries for `NE16WeightEncode`, `NE16SignedInputBias`, `NE16Im2ColU8`, `NE16DualPerturbEncode`, `NE16DualBias` (all run on the cluster cores; `NE16Engine.canExecute` only ever claims `Conv`/`RequantizedConv`) and for `RequantizedDualConv` (exp5's kernel, baseline C_cl); `_includeList` gains `perf_utils.h` (line 320) |
| `TrainDeeploy/Deeploy/Targets/GAP9/Bindings.py` lines 68, 88 | `PULPMicrobenchmark()` appended to `GAP9Transformer` and `GAP9ClusterTransformer` (no-op unless `--profileMicrobenchmark`); without it the NE16 and Transpose nodes had no counters |
| `TrainDeeploy/TargetLibraries/GAP9/src/NE16WeightEncode.c` | `NE16WeightEncode_i8_u8(src, dst, cout, cin, taps, row_start, row_count)`: for each row `j*cout + co` and each 16-lane group, `v = src[...] ^ 0x80` (lanes past `cin` are 0, not 128, matching numpy's pad after the offset), output byte `b*2 + k` collects bit `b` of lanes `8k..8k+7` LSB-first; `NE16SignedInputBias_i32(weight, mul, add, out, cout, cinTaps, offset, co_start, co_count)`: `out[co] = add[co] - offset * sum_k w[co][k] * mul[co]` |
| `TrainDeeploy/TargetLibraries/GAP9/src/NE16DualPrep.c` | `PULP_NE16DualPerturbEncode`: `dc_chunk` + `dc_perturb_i8` into a `[2*cout][K]` scratch (W+ rows then W− rows), barrier, then `NE16WeightEncode_i8_u8(scratch, enc, 2*cout, K, 1, r0, r1-r0)` chunked by rows; `PULP_NE16DualBias`: re-derives W±, `dc_perturb_i32` for add±, barrier, then per row `r` of the stacked matrix `addOut[r] = add±[c] - input_offset * sum_k W±[c][k] * mul[c]` |
| `TrainDeeploy/TargetLibraries/GAP9/inc/DeeployGAP9Math.h` | prototypes for the four kernels above plus `#include "kernel/{RandomNoise,ZORuntime,Im2ColU8,DualConv,FlatConv}.h"`; a missing prototype here compiles as an implicit-int call under GAP9 GCC and shifts every argument after a `float` (that was C_cl's "Invalid access" crash) |
| `TrainDeeploy/TargetLibraries/GAP9/CMakeLists.txt` | globs `src/**` and `../PULPOpen/src/**` (minus `dory_mem`) into `deeploygap9`; when `platform STREQUAL GAP9_w_NE16` sets `USE_NE16 ON`, adds `../third_party/pulp-nnx` and links it (lines 90–103); PR #183's exclusion of the SDK's `CNN_BasicKernels_NE16.c` is not needed because our GAP9 lib never compiled it |
| `TrainDeeploy/TargetLibraries/GAP9/inc/DeeployMchan.h` | `-- QW` notes only (assert fallback, channel-id macro); the 17-bit MCHAN bound is asserted in `TrainDeeploy/Deeploy/Targets/GAP9/DMA/MchanDma.py` line 79 (`cmd = (flags << 17) + size`), i.e. our tree still asserts rather than chunking as PR #183 does |
| `TrainDeeploy/DeeployTest/testUtils/platformMapping.py` lines 23–25, 73, 102 | `GAP9_w_NE16` → `NE16Platform` / `NE16Deployer` / `MemoryNE16PlatformWrapper`; the NE16 `isinstance` branch precedes the GAP9 one (subclass) |

pulp-nnx driver entry points used by the generated C (vendored at
`TrainDeeploy/TargetLibraries/third_party/pulp-nnx/`, untouched): `ne16_task_t` and
`ne16_task_data_t` (`ne16/hal/ne16_task.h` lines 100–118), `NE16_SUBTILE_INPUT_CHANNEL 16`,
`NE16_SUBTILE_OUTPUT_CHANNEL 32`, `NE16_SUBTILE_OUTPUT_HEIGHT 3`, `NE16_FLAG_STREAMIN (1<<14)`
(`ne16/hal/ne16_task_defs.h`), `NE16_TASK_QUEUE_SIZE (2)` (`ne16/hal/ne16.h` line 27), and
`ne16_nnx_init`, `ne16_nnx_dispatch_wait`, `ne16_nnx_dispatch`, `ne16_nnx_resolve_check`,
`ne16_nnx_resolve_wait` (`inc/pulp_nnx_ne16.h`, `src/pulp_nnx_ne16.c`). Under GVSoC
`ne16_nnx_resolve_check` returns `ne16_task_queue_empty(dev)` (a full drain), which is why the
per-tap route gained only 1.65 % from pipelining.

### 4.3 The NE16 hardware facts this code depends on

There is no paper describing the NE16 microarchitecture; the authoritative source is the SDK's GVSoC
behavioural model inside `deeploy_gap9`, `/app/install/gap9-sdk/gvsoc/gvsoc_gap/gap/ne16/src/`
(`ne16_regfile.cpp` = CONFIG0 decode, `ne16_matrixvec.cpp` = MAC array, `ne16_normquant.cpp` =
scale/bias/shift, `ne16_streamout.cpp` = output saturation, `ne16_load.cpp` = input fetch), plus the
register names in `tools/autotiler_v3/CNN_Libraries_HWPE/hal_ne16.h` (`NE16_REG_CONFIG 0x5c`) and the
pulp-nnx constants in `TrainDeeploy/TargetLibraries/third_party/pulp-nnx/ne16/hal/ne16_task_defs.h`.
Every fact below was read there.

- **Lanes.** 16 input channels per pass (TP_IN, `NE16_SUBTILE_INPUT_CHANNEL`), 32 output channels per
  pass (TP_OUT, `NE16_SUBTILE_OUTPUT_CHANNEL`), 3×3 output spatial subtile from a 5×5 input subtile;
  9×9 engines × 16 = 1,296 binary MAC/cycle, bit-serial in the weights, so 8-bit weights give
  162 MAC/cycle peak. A layer with Cin = 8 half-fills the input lanes, one with Cout = 8 uses a
  quarter of the output lanes.
- **Filter modes**, CONFIG0[6:5]: `00` 3×3, `01` 3×3-DW, `10` 1×1. No 1×K.
- **Bit-serial weight layout.** Weights are consumed one bit-plane at a time: per (cout, group of 16
  input channels) the 8 bit-planes of the 16 lanes take 8 × 2 bytes = 16 bytes, so the layout is
  `[cout][ceil(cin/16)][16]` uint8, and byte `b*2 + k` holds bit `b` of lanes `8k..8k+7`, LSB first
  (`_weightEncode` in `TrainDeeploy/Deeploy/Targets/NE16/TopologyOptimizationPasses/Passes.py` and its
  C port `NE16WeightEncode_i8_u8`, §4.2). The hardware treats weights as unsigned and subtracts a
  layer-wise `weight_offset`, implemented as `Wmin` (`__weightoffs()` in `ne16_matrixvec.cpp`; the
  `FIXME not implemented` on CONFIG0[15] refers only to that selector bit). We fix
  `weight_offset = −128`, so the encoding of a signed weight is `w ^ 0x80` followed by a
  data-independent bit permutation; lanes past `cin` are encoded as 0.
- **CONFIG0 map** (`ne16_regfile.cpp:200-225`): [4] outquant, [6:5] filter mode, [7] linear mode,
  [8] strided 2×2, [11:9] reserved, [13:12] norm bits, [14] streamin, [15] weight-offset selector,
  [20:16] quant right-shift, [22:21] quant bits (`2<<21` = 32-bit output), [23] quant norect,
  [24] norm shift, [25] norm bias. **Bit 26 is not decoded.** PR #183's `ConvTemplate.getConf0` sets
  `1 << 26` for `input_signed`; that bit never reaches the hardware, so int8 activations are read as
  uint8 (its one signed test failed with expected +127 vs actual −128, a sign error). Its `use_wmem`
  sets bit 9, inside the reserved field, also a no-op.
- **Inputs are unsigned.** Hence the +128 offset on the activation and the bias correction
  `add' = add − 128·mul·Σw` (exp16c, §4.4). The SDK also ships a second model, `ne16v2`
  (`gvsoc_gap/gap/ne16v2/`), whose CONFIG0 has a real `mode_signed` bit 28, `use_rounding` at bit 15
  and scale/bias broadcast bits; it would remove both workarounds, but it is not GAP9's NE16:
  `gvsoc_gap/gap/gap9/cluster.py` imports `gap.ne16.ne16` (v1) and instantiates it, and the
  `gap9_v2` chip directory selected by `gap9_evk_audio.sh` (`TARGET_NAME=gap9_v2`) only adds ROM,
  efuse and `apb_soc_ctrl` on top of `gap9/cluster.py`. PR #183's bit 26 matches neither model (in
  v2 bit 26 is `scale_broadcast`).
- **Truncating requant.** `(acc*scale + bias) >> shift`, no rounding term, saturate to [−128, 127]
  (`ne16_normquant.cpp:169`, `ne16_streamout.cpp:87`). This is the same arithmetic as the fused
  cluster path (`pulp_nn_bn_quant_i8`, a bare shift), while Deeploy's standalone `RequantShift`
  template rounds; round-to-nearest is carried in the add as `+div/2` (32768 for block 0), so NE16
  equals pulp-nn's `(acc*mul + add) >> 16` bit for bit and introduces no new arithmetic.
- `streamin` (CONFIG0[14]) requires 32-bit output (`fsm.cpp:49` assert); padding in 1×1 mode is
  invalid and its assert at `fsm.cpp:53` is commented out (silent wrong results).

### 4.4 exp16a–c: the bring-up routes (correct, slow)

Builders, fixtures' generated code, logs and `results/results.json` per sub-experiment:
`TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/` (`exp16a_PW_single_layer/`,
`exp16b_Dense_single_layer/`, `exp16c_PW_single_layer/`, `exp16c_SDK_port/`).

**exp16a, per-tap 1×K route.** One `Conv(1×K)` node is emitted by `Conv1xKTemplate.py` as K pointwise
dispatches, tap j = an address offset on `infeat_addr`, accumulating in NE16 via streamin, int32
output, `RequantShift` left on the cluster (`_merge_conv_rq_fun` in
`TrainDeeploy/Deeploy/Targets/PULPOpen/TopologyOptimizationPasses/Passes.py` line 177 skips a conv
carrying `ne16_taps`). The host supplies a pre-perturbed weight in bit-serial layout with
`weight_offset = −128` as a graph input (`ne16_weight_preencoded=1`). Block 1 at full 14×87:
`0/19712`, 1,567,096 cycles vs 337,657 on the cluster (4.64× slower). Why it is slow: Cin = 8
half-fills TP_IN; 16 taps × 4 tiles = 64 jobs; 15 of 16 taps round-trip int32 partial sums through
L1; the int32 intermediate is 4× the int8 one. Pipelining the dispatch loop (exp16c_SDK_port phase 1)
recovered only 1.65 % in total, which showed the cost is the streamin round-trips, not the register
writes: the route is dispatch-bound in the sense of per-dispatch accumulator traffic.

**exp16b, dense 3×3 chunks.** The same 1×16 as 6 dense 3×3 jobs with only the middle row populated
(`Conv3x3ChunkTemplate.py`): `0/19712`, 1,364,975 cycles, 1.15× faster than per-tap against a
prediction of about 3× slower (the model assumed the MAC array was the bottleneck; it is not at this
size). Price: 3.4× the weight bytes. The correctness bug found on the way was geometry, not
arithmetic: `NE161xKConstraint.serializeTilingSolution` emitted pointwise subtile counters for a 3×3
job, so the border subtile read two pixels short (±1 LSB, position-dependent).

**exp16c, on-device weight path.** `NE16WeightEncode` (device bit-serial encode; fixed `−128` offset
reduces it to `w ^ 0x80` plus a data-independent permutation; validated natively via ctypes against
`_weightEncode` for all five conv shapes before any device run) cost 26,189 cycles = 1.64 % of block
1; perturb + encode on device = 2.7 %. This withdrew the originally planned linearity decomposition
(`conv(w+δz,x) = conv(w,x) + δ·conv(z,x)`), which would have doubled the dispatch count to save that
1.64 %. Blocks 1–4 then passed on their first device runs; block 0 needed the signed-input fix:
feed `x_u = x + 128` as uint8 and correct `add' = add − 128·mul·Σw` (`NE16SignedInputBias_i32`),
because `add` lands after the multiply in `(acc*mul + add) >> s`. Range for block 0:
`|128·Σw·mul| ≤ 128·508·454 ≈ 3.0e7`, inside int32. Result table (per pass, NE16 vs cluster
`pulp_nn_conv`, fixtures on stale weights; `exp16c_PW_single_layer/results/results.json`, NE16 runs in
`logs/step4_block1_ne16.log` … `step4_block4_ne16.log` and `logs/step6_block0_signed_ne16.log`, cluster references in
`logs/step5_block1_cluster_ref.log` … `step5_block4_cluster_ref.log` and `logs/step7_block0_cluster_ref.log`; exp16a/b numbers
above in `exp16a_PW_single_layer/results/results.json` and `exp16b_Dense_single_layer/results/results.json`):

| block | kernel | Cin→Cout | out | dispatches | errors | NE16 cycles | cluster cycles | ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 1×4 | 1→8 | 14×701 | 4 | 0/78,512 | 5,517,302 | 732,178 | 7.54× |
| 1 | 1×16 | 8→16 | 14×88 | 16 | 0/19,712 | 1,608,939 | 337,999 | 4.76× |
| 2 | 1×8 | 16→16 | 14×23 | 8 | 0/5,152 | 420,743 | 98,723 | 4.26× |
| 3 | 7×1 | 16→32 | 8×5 | 7 | 0/1,280 | 183,651 | 26,835 | 6.84× |
| 4 | 7×1 | 32→32 | 2×5 | 7 | 0/320 | 145,990 | 30,329 | 4.81× |

**exp16c_SDK_port.** `NE16Prepare1xKPass` makes a plain exported graph compile to NE16 without
fixture surgery (`b3_plain` 0/1280, `b4_plain` 0/320). The full QZO step
(`deeployMezoRunner_tiled_GAP9_w_NE16.py … --enable-1xk`) runs with blocks 3 and 4 on NE16
(14 dispatches, 2 device encodes, 3 cluster convs) and is bit-identical to the cluster-only
baseline on all eight per-pass losses (`max |NE16 − cluster| = 0.00e+00`); padded blocks 0/1/2 stay
on the cluster (5.8 % of conv MACs reach NE16), and the step is 1.9 % slower than cluster-only
(`exp16c_SDK_port/results/results.json`; the eight `[PHASE] ±eps loss read OK: l{p,m}_bits=` lines of
`logs/qzo_full_FINAL_bitexact.log` against `logs/qzo_baseline.log`; generated code with the 14
`ne16_nnx_dispatch(` calls in `fixture/TrainingNetwork_qzo_ne16_FINAL.c`).
Reaching bit-identity took three pre-existing Deeploy bugs, all unreachable before NE16 because no
standalone `RequantShift` with a runtime bias had ever existed:

| # | file | defect | fix |
|---|---|---|---|
| 1 | `TrainDeeploy/Deeploy/Targets/PULPOpen/Templates/RequantShiftTemplate.py` and the Generic twin | `rounding = 1` hardcoded while the merge pass bakes `+div/2` only into a constant add → +0.5 LSB on every element with a runtime add | `rqs_rounding = int(hasattr(addBuffer, "values") and addBuffer.values is not None)` |
| 2 | `TrainDeeploy/Deeploy/Targets/PULPOpen/TileConstraints/RQSPerturbTileConstraint.py` (serialize) | `channelDim = dims[0]` on a `(1, cout)` broadcast bias → `channel_width = cout` → kernel reads `M[0]` for every channel | skip leading unit axes |
| 3 | same file (geometry) | input/output dims paired by raw index → 4-byte input tile against a 128-byte DMA → 124-byte L1 overlap (constant error per channel, channel 0 correct) | align from the first non-unit axis |

Plus two earlier ones: `RequantShiftLayer.computeShapes` ignoring `channels_first`, and
`UniformRequantShiftParser.parseNode` (`TrainDeeploy/Deeploy/Targets/Generic/Parsers.py`) raising
instead of declining on a runtime mul/add.

### 4.5 exp6: the SDK im2col route and the fused NE16 node (current)

Logs, generated code (`codegen/{A,Ccl,t1,t2}/Network.c`) and `results/summary.txt`:
`TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp6_NE16_fusion/`.

What exp6 deliberately does not use from `Targets/NE16/`: `NE16Prepare1xKPass` / `Conv1xKTemplate.py`
/ `--enable-1xk` (the per-tap route; left in the tree, simply not enabled, and with the flag off the
pass sees no NE16-coloured 1×K conv), `Conv3x3ChunkTemplate.py` (exp16b), and the exp16c fixture
workarounds of padding and +128-offsetting the activation on the host or setting
`ne16_unsigned_input` (in the im2col route the cores do both while writing the column). Interactions
checked when the route was added:

| risk | why it does not bite |
|---|---|
| does the NE16 engine claim the folded 1×1 conv? | yes, as intended: kernel `[1,1]`, strides/dilations 1, weight marked `ne16_weight_preencoded=1` → `isPWConv` → NE16, no flag needed |
| does the conv+requant merge lose NE16 attributes? | no: `attrs = {**conv.attrs, **rqs.attrs}`; the merge guard keys on `ne16_taps`, which the folded conv does not carry |
| does the NHWC pass permute the column or the encoded weight? | no: conv and `RequantShift` are declared `channels_first=0` with NHWC shapes; the weight is rank-3 and pre-encoded |
| does `NE16AdjustWeightMemoryLayoutPass` re-encode the runtime weight? | no: it returns early for a non-constant weight |
| does `NE16ReshapePointwiseConvolutionPass` re-factor 14×701? | it may (fewer subtiles); harmless and deterministic |
| does `NE16Prepare1xKPass` touch the graph? | only NE16-coloured `Conv` with kernel ≠ 1×1; ours is 1×1, so it is a no-op |
| is `RequantizedDualConv` available on GAP9 for the C_cl baseline? | not originally (exp5 registered it in `PULPMapping` only); exp6 added the mirror entry in `GAP9Mapping` (§4.2) |
| does the PW binding accept Cin = 4? | yes: `_getNumTiles(4, 16) = 1`, `bKi = 4`, uint8 input, runtime `add` as a pointer input, 8-bit output with norm/quant |

**Task 1, single pass** (fixture `Tests/Kernels/Integer/QZO_Conv_NE16`):

```
ACT [1,1,14,700] int8 NCHW --NE16Im2ColU8(1x4, pads 0/2/0/2, +128)--> col [1,14,701,4] uint8 NHWC
W_RAW [8,1,1,4] --RQSPerturbRademacher(verbatim)--> weight_pert [8,1,1,4]
     +-- NE16WeightEncode(ne16_taps=1, ne16_fold=1) --> weight_enc [8,1,16] uint8
     +-- NE16SignedInputBias(weight_pert, mul, bias_pert) --> add_corr [8] int32
B_RAW [8] --RQSPerturbRademacher(verbatim)--> bias_pert [8]
Conv(col, weight_enc; 1x1, pads 0, channels_first=0, ne16_weight_preencoded=1, weight_offset=-128)
RequantShift(conv_out, mul, add_corr; channels_first=0)        } merged by PULPConvRequantMergePass -> one NE16 job per tile
Transpose(0,3,1,2) --> OUT [1,8,14,701] == exp3's outputs.npz
```

Load-bearing details:

- `Im2ColU8.c` writes column index `(ci*ky_n + ky)*kx_n + kx`, i.e. `(ci, ky, kx)` order, which is the
  flattening order of the NCHW weight `[co][ci][ky][kx]`; the folded 1×1 weight is therefore the
  perturbed weight reshaped to `[8][4]` with no permutation. `DualConv.c` gathers `(ky, kx, ci)`
  because it consumes the NHWC-transposed weight; the two orders coincide only for Cin = 1 (§8).
- `^ 0x80` is `+128` reinterpreted as uint8; `pad_value = 128` is signed zero after the offset
  (writing 0 would mean −128 to the correction term).
- The perturb node's output keeps shape `[8,1,1,4]`; the fold happens in the encoder because
  `RQSPerturbTileConstraint` pairs input/output dims by index (declaring `[8,4,1,1]` made the tiler
  cut a 4-element tensor along the wrong axis).
- The column is declared NHWC-native and the conv `channels_first=0`, so `_NCHWtoNHWC_fun` inserts
  no transposes; the rank-3 pre-encoded weight is skipped by the same function
  (`LoweringOptimizationPasses.py` line 294).
- The NE16 job is the ordinary pointwise path (`NE16RqntPWConv2DMapper`, `NE162DPWConvTemplate`):
  `KoKi = nnx_concat_half(1,1)`, remainder `(8, 4)`, `conf0` with outquant, 32-bit scale, shift 16,
  no ReLU. 4 conv tiles per pass (`--l1 92000`), one `ne16_nnx_dispatch` per tile.
- The column node is tiled (1 tile) on purpose: an untiled node keeps its tensors at the default
  memory level (L2) and the cores then gather from L2 at about 10× the cost (the first run measured
  ~1.4 M cycles for the column). With the im2col tiled as 1 tile and the conv as 4 tiles under
  `--defaultMemLevel L2`, the column is written in L1, DMA'd to L2 and brought back per conv tile.
  (Inferred from the tiling schedule; verify against `exp6_NE16_fusion/codegen/t1/Network.c`.)
  Tile counts measured: NE16 conv 4 tiles in both tasks, im2col 1 tile, output transpose 2 tiles (B)
  or 4 (C); C_cl: conv 2 tiles, transpose 4 (`exp6_NE16_fusion/results/summary.txt`, "TILING").

**Task 2, fused** (fixture `Tests/Kernels/Integer/QZO_DualConv_NE16`):

```
W_RAW, wmul --NE16DualPerturbEncode(seed_w, idx_w, div_w)--> weight_enc [16,1,16]   (W+ rows 0-7, W- rows 8-15)
W_RAW, wmul, B_RAW, bmul, mul --NE16DualBias(all six seed attrs, ne16_input_offset)--> add2 [16]
ACT --NE16Im2ColU8--> col                                                              (once, shared)
Conv(col, weight_enc; 1x1, Cout 16) + RequantShift(mul2 = [mul; mul], add2) --> one NE16 job per tile
Transpose --> OUT [1,16,14,701] == exp5's outputs.npz (A+ channels 0-7, A- 8-15)
```

Two preparation nodes instead of one because Deeploy's tiler is single-output; `NE16DualBias`
re-derives W± deterministically (same seeds, same chunking) to form Σw per row, costing 3,760
cycles. Both nodes emit `DualConvTemplate.py`'s seed expression, so the device draws exp5's noise
stream and exp5's reference applies unchanged; the exp5 refactor that moved the helpers into
`DualPerturb.h` was re-verified bit- and cycle-identical
(`exp6_NE16_fusion/logs/exp5_regression_after_DualPerturb_refactor.log`: 910,763 cycles, 0/157,024).

---

## 5. How to run

Rebuild directories and kill orphan simulators before every run (CMake caches `-D` flags across runs;
orphaned `gvsoc_launcher` processes starve new simulations, and `killall`/`pkill` by name do not
reach them, so kill by PID: `pgrep -f "[g]vsoc_launcher" | xargs -r kill -9`).
`PYTEST_XDIST_WORKER=<tag>` gives a private build directory (`TEST_SIRACUSA/build_<tag>`,
`TEST_GAP9*/build_<tag>`).

**Where the fixtures are.** The four block-0 kernel fixtures are kept, byte-identical, in
`TrainDeeploy/SpeechNet_Test/` under the shared naming scheme; the original names under
`TrainDeeploy/DeeployTest/Tests/Kernels/Integer/` (used everywhere else in this guide and in the
registered pytest configs) still exist:

| `TrainDeeploy/SpeechNet_Test/` | original (`DeeployTest/Tests/Kernels/Integer/`) | builder | runs on |
|---|---|---|---|
| `speechnet_unfused_b0_cluster` | `QZO_Conv` | `build_qzo_conv_fixture.py` (exp3) | Siracusa, GAP9 cluster |
| `speechnet_fused_b0_cluster` (+ `reference_plus_minus.npz`) | `QZO_DualConv` | `build_dual_conv_fixture.py` (exp5) | Siracusa, GAP9 cluster |
| `speechnet_unfused_b0_NE16` | `QZO_Conv_NE16` | `build_ne16_im2col_fixture.py` (exp6 task 1) | GAP9 NE16 |
| `speechnet_fused_b0_NE16` | `QZO_DualConv_NE16` | `build_ne16_dual_fixture.py` (exp6 task 2) | GAP9 NE16 |

A fixture outside `DeeployTest/` makes the runner print "Test path … is outside base directory" and
generate into `TEST_<PLATFORM>/<fixture name>`; this is expected. All commands below were run from
these paths (logs in `TrainDeeploy/SpeechNet_Test/logs/smoke/`, drivers
`TrainDeeploy/SpeechNet_Test/scripts/run_smoke_{siracusa,gap9}.sh`).

### 5.1 Fixtures (host side)

```bash
# cluster fixtures needing the Onnx4Deeploy oracle: container agitated_hugle
docker exec agitated_hugle bash -lc 'cd /app && PYTHONPATH=/app/Onnx4Deeploy:/app/TrainDeeploy \
  python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_qzo_conv_fixture.py && \
  PYTHONPATH=/app/Onnx4Deeploy:/app/TrainDeeploy \
  python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_dual_conv_fixture.py'
# NE16 fixtures copy their tensors from the two above; plain python with onnx + numpy suffices
cd /Users/qiwenwu/ETH
python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_ne16_im2col_fixture.py
python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_ne16_dual_fixture.py
```

The builders write to `Tests/Kernels/Integer/<name>`; copy a rebuilt fixture to its
`SpeechNet_Test/` name if you run from there.

### 5.2 Siracusa (container `traindeeploy`, repo at `/app/ETH/TrainDeeploy`)

```bash
docker exec traindeeploy bash -lc 'cd /app/ETH/TrainDeeploy/DeeployTest && \
  pgrep -f "[g]vsoc_launcher" | xargs -r kill -9; rm -rf TEST_SIRACUSA/build_e5 && \
  M="--l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max --cores 8" && \
  PIN="--input-type-map blocks.0.conv.bias_rqsadd=int32_t --input-offset-map blocks.0.conv.bias_rqsadd=0" && \
  PYTEST_XDIST_WORKER=e5 python3 deeployRunner_tiled_siracusa.py -t ../SpeechNet_Test/speechnet_fused_b0_cluster $M $PIN'
#   -> Errors: 0 out of 157024 ; Runtime: 910763 cycles   (exp5: 910,763)
# baseline (one pass): -t ../SpeechNet_Test/speechnet_unfused_b0_cluster $M $PIN
#   -> Errors: 0 out of 78512 ; Runtime: 858185 cycles    (exp3: 858,185)
# the int32 pin (PIN) is required: value inference otherwise types the all-32768 add as uint16_t (§7 item 5)
# per-node counters: add --profileMicrobenchmark (then ignore its Runtime line, §6.2) and
#   python3 experiments/dual_pass_fusion/exp5_fused_dual_conv/parse_microbench.py <log> results/per_node
# per-tile profile: add --profileTiling ; double buffering: add --doublebuffer
# control kernel: flip the KERNEL SWITCH in Deeploy/Targets/PULPOpen/Platform.py, run the unfused fixture (-> 638519), flip back
# registered path (original names): pytest test_platforms.py -m "siracusa_tiled and kernels" -k "QZO_Conv or QZO_DualConv"
```

### 5.3 GAP9 and NE16 (container `deeploy_gap9`, repo at `/app/Deeploy`)

**Container setup.** GAP9 needs the private image `ghcr.io/runwangdl/deeploy:gap9` (13.7 GB). It
is a multi-target Deeploy development image: Ubuntu 22.04, Python 3.10, CMake 3.24, the full
GreenWaves GAP9 SDK at `/app/install/gap9-sdk` (SDK 5.21.1, commit `8c42b653…`; GAP9 GCC
`riscv32-unknown-elf`, the AutoTiler and NNTool, and GreenWaves' own GVSoC with the GAP9 chip
models), plus PULP LLVM (clang 15), the open-source PULP SDK and GVSoC, and other targets'
toolchains. Two GVSoC installations exist and are not interchangeable: `/app/install/gvsoc` is the
open-source PULP one (Siracusa); GAP9 runs must use `/app/install/gap9-sdk/install/workstation`.
The image is x86_64, so on Apple Silicon everything runs emulated, which dominates wall-clock time
of long GVSoC runs. Access to the image is granted per GitHub account by its owner; then:

```bash
docker logout ghcr.io                 # drop a cached login of another account, if any
docker login ghcr.io                  # user = GitHub user; password = a classic PAT with read:packages
docker pull ghcr.io/runwangdl/deeploy:gap9
# from the TrainDeeploy directory, mount it at /app/Deeploy:
docker run -dit --name deeploy_gap9 -v $(pwd):/app/Deeploy ghcr.io/runwangdl/deeploy:gap9 tail -f /dev/null
docker exec deeploy_gap9 bash -lc 'cd /app/Deeploy && pip install -e .'
```

Every shell in the container must first source the SDK environment (the three lines at the top of
the block below); `gap9_evk_audio.sh` selects `TARGET_NAME=gap9_v2`, `BOARD_NAME=gap9_evk`.
Because the repo is mounted at `/app/Deeploy`, `SpeechNet_Test` is at `/app/Deeploy/SpeechNet_Test`.

```bash
docker exec deeploy_gap9 bash -lc '
source /app/install/gap9-sdk/.gap9-venv/bin/activate
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
cd /app/Deeploy/DeeployTest
P="--l1 92000 --l2 1500000 --defaultMemLevel L2 --cores 8 --input-type-map blocks.0.conv.bias_rqsadd=int32_t --input-offset-map blocks.0.conv.bias_rqsadd=0"
PYTEST_XDIST_WORKER=t1 python3 deeployRunner_tiled_gap9_w_ne16.py -t ../SpeechNet_Test/speechnet_unfused_b0_NE16   $P  # -> Runtime: 379969 ; Errors: 0 out of 78512   (exp6 B)
PYTEST_XDIST_WORKER=t2 python3 deeployRunner_tiled_gap9_w_ne16.py -t ../SpeechNet_Test/speechnet_fused_b0_NE16     $P  # -> Runtime: 541345 ; Errors: 0 out of 157024  (exp6 C)
PYTEST_XDIST_WORKER=A  python3 deeployRunner_tiled_gap9.py        -t ../SpeechNet_Test/speechnet_unfused_b0_cluster $P  # -> Runtime: 733298 ; Errors: 0 out of 78512   (exp6 A: 734449)
PYTEST_XDIST_WORKER=C  python3 deeployRunner_tiled_gap9.py        -t ../SpeechNet_Test/speechnet_fused_b0_cluster   $P  # -> Runtime: 925593 ; Errors: 0 out of 157024  (exp6 C_cl)
'
# per-node counters on GAP9: add --profileMicrobenchmark; the log carries "[perf] records=N" followed by the
# fabric-controller replay; parse with experiments/dual_pass_fusion/exp6_NE16_fusion/parse_microbench.py
```

The unfused cluster run from `SpeechNet_Test` gives 733,298 cycles against exp6's 734,449 (0.16 %),
outputs bit-exact in both. The generated code differs from exp6's only in the L1 offsets of two
buffers (the weight and the requant bias swap places): the tiler's default search strategy
`random-max` is randomised, so placement can change from build to build. Treat sub-percent
differences on GAP9 cluster runs as placement noise.

The exp16 per-tap route (not used by exp6) is driven with `--enable-1xk` on the same runner against
`Tests/Models/NE16/<fixture>`; the full QZO step with NE16 engaged is

```bash
python3 deeployMezoRunner_tiled_GAP9_w_NE16.py \
  -t Tests/Models/Training/SpeechNet/speechnet_qzo12_train \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo12_update \
  --enable-1xk --n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
  --l1 110000 --l2 1500000 --defaultMemLevel L2 -D BN_FROZEN_STATS=ON
# check the engine split: grep -c "ne16_nnx_dispatch(" TEST_GAP9_W_NE16/.../TrainingNetwork.c  (14)
# compare device losses against speechnet_qzo12_update/outputs.npz; "Errors: 0 out of 8" checks only the update graph
```

**L1 budgets differ per harness.** `DeeployTest/Platforms/GAP9/src/deeploymezotest.c` relocates the
cluster slave stacks to an L2 array with `pi_cluster_task_stacks()` (lines 125, 300–451), so the
training runner fits `--l1 110000`. The inference harness `DeeployTest/Platforms/GAP9/src/deeploytest.c`
does not (`#define SLAVESTACKSIZE 3800`, no `#ifndef`, no CMake override; PR #183's SLAVESTACKSIZE
plumbing is not in our tree), so about 30 KB of L1 is stacks and kernel fixtures use `--l1 92000`.

---

## 6. Results

### 6.1 Siracusa, block 0, both ε-passes (exact int8 compare, artifacts_reference weights)

| | A: 2× pulp-nn (exp3) | B: 2× FlatConv (exp5a) | C: DualConv (exp5) | A/B rewrite | B/C fusion | A/C |
|---|--:|--:|--:|--:|--:|--:|
| fixture, cluster timer | 1,716,370 | 1,277,038 | 910,763 | 1.34× | 1.40× | 1.88× |
| conv node, core-0 counter | 1,329,292 | 890,056 | 614,979 | 1.49× | 1.45× | 2.16× |

Sources (all under `experiments/dual_pass_fusion/`): fixture runtimes in
`exp3_as-is_QZOConv_microbench/logs/run_plain.log` (858,185 per pass), `exp5a_breakdown/logs/run_plain.log`
(638,519 per pass), `exp5_fused_dual_conv/logs/run_plain.log` (910,763); node counters in the three
`logs/run_microbench.log`; the table itself in `exp5a_breakdown/results/summary.txt`. The fused kernel doing both passes (614,979) is faster than
pulp-nn doing one (664,646). Other exp5 figures: I-cache refill 302,724 → 5,076 cycles (60×); core-0
instructions 822,686 → 523,987; IPC 0.619 → 0.852; kernel code 4,694 B → 1,828 B; cluster
MAC/cycle 0.473 → 1.02; per-core MAC share 314,048/8 = 39,256 per pass and 78,512 for both (core 0's
exact count is 1,227 px × 32 × 2 = 78,528, `exp5_fused_dual_conv/results/summary.txt`; code sizes in `results/code_size.txt`); output
transpose 210,699 cycles = 25.2 % of the run, the new bottleneck. Where fusion's B − C = 366,275
cycles come from, by node (`exp5a_breakdown/results/summary.txt`): conv 275,077 (75 %), output
transpose 62,649 (one 2× tensor in 3 tiles instead of two tensors in 2 + 2), outside-node DMA/tiling
10,537, the two perturb nodes 9,162 (moved into phase 1), input transpose 8,115, weight transpose 735.
The rewrite's A − B = 439,332 is entirely the conv node; it wins on fetch stalls (108 → 1.5 cycles
per pixel), load balance (87.4 % → 99.8 %) and removed calls/spills (loads 71 → 47, stores 36 → 12
per pixel), not on instruction count (294 → 302 per pixel).

### 6.2 The three counting rules (read before comparing any two numbers)

Three instruments appear in this work and they count different things. Mixing them produced wrong
shares in early analyses, so every number in §6 is tagged with its instrument.

| instrument | what it counts | example (FlatConv fixture, exp5a) |
|---|---|---|
| `Runtime:` in a plain run | the **cluster timer** (`0x10200400`), started and stopped by the harness around the cluster task: wall-clock cycles of the whole network | 638,519 (`exp5a_breakdown/logs/run_plain.log`) |
| `Cycles:` per node in a `--profileMicrobenchmark` run | core 0's **PCER cycle counter** between the wrapper's two reads: cycles core 0 was *awake*; it stops while core 0 sleeps in the event unit (barrier waits, DMA waits, fork/join), so the fused node's counter hides its 6.2 k of barrier wait | 445,028 (`exp5a_breakdown/logs/run_microbench.log`) |
| trace span of an address range | core 0 **wall clock** between the first and last instruction of the range in a `pe0/insn` trace; `conv_breakdown.py` splits out the sleeps it can identify (barrier wait, inter-tile gap) | 443,784 (`exp5a_conv_breakdown/results/node_breakdown_microbuild.txt`) |

The rules that follow:

1. **The `Runtime:` line of a microbench log is meaningless.** Every wrapper resets and stops the same
   cluster timer the harness reads, so it reports only the last profiled node's wall time
   (`exp5a_breakdown/logs/run_microbench.log` prints 166,110 = the output transpose with its DMA waits).
   Quote plain runs for runtime; a profiled run also carries wrapper overhead.
2. **Compare like with like.** Node counters may be compared with node counters (same instrument,
   same core) and fixture totals with fixture totals, but a counter is never divided by a plain
   runtime (different rule, different build). The gap between a fixture's wall total and its counter
   sum (A 87 k, B 87 k, C 76 k) is time core 0 is asleep or between nodes (DMA waits, fork/join,
   sequencing); it is not overhead of any node and must not be assigned to one.
3. **Reconcile a counter with a trace only through the sleeps.** For FlatConv: counter 445,028 =
   kernel span 443,784 − barrier wait inside the kernel (4) + fork/join glue executed awake (≈1,250);
   between the wrapper reads 456,048 cycles elapse, of which 1,022 (fork) and 10,069 (join) are sleeps
   the counter skips. For DualConv: 614,980 = 618,253 − 6,209 barrier waits + ≈2,900 glue. Because
   only the fused kernel has real barrier waits, the wall-clock conv ratio B/C is 887,568 / 618,253 =
   1.44×, not the counter's 1.45×.

Also, the counter blocks are partial: the "Instruction Mix" counts (loads, stores, branches, jumps,
compressed) omit arithmetic and data-movement instructions (≈70 %), and instructions + counted
stalls leave 10.8 % (B) / 8.2 % (C) of cycles uncovered (taken-branch penalties, barrier and DMA
waits, multi-cycle instructions). The wrapper sits outside the tile loop, so a printed number covers
all tiles of a node. On GAP9 the wrappers run on the master core (core 8, §7 item 8): its cycles are
the node's wall time, but its instruction, IPC and stall columns describe an idle core.

### 6.3 Level-1 breakdowns (GVSoC instruction traces of core 0)

Method: record `--trace=pe0/insn` for the whole run, map every PC to its C source line with
`llvm-objdump -d -l` (the build has `-g`) and to a category by address range of `FlatConv.c` /
`DualConv.c`; the cost of an instruction is the cycle stamp of the next one minus its own, so a
load-use bubble, branch flush or fetch miss is charged to the instruction before the gap and nothing
is unaccounted. Warm pixels 100, 300, 500 agree to ±1 cycle. The microbench build moves the kernel
(by 5,012 B for FlatConv, 4,088 B for DualConv), hence `ADDR_OFFSET`; tracing either build gives the
same categories to ±0.1 %.

FlatConv, one pass, whole node 443,835 cycles, 370,629 instructions
(`exp5a_breakdown/exp5a_conv_breakdown/results/node_breakdown.txt`; microbench build in
`node_breakdown_microbuild.txt`, per-pixel windows in `pixel{100,300,500}_breakdown.txt`):

| category | cycles | share | cycles / pixel |
|---|--:|--:|--:|
| gather (bounds-tested byte copies) | 148,540 | 33.5 % | 121.1 |
| requant + store | 119,233 | 26.9 % | 97.2 |
| channel-loop control | 97,103 | 21.9 % | 79.1 |
| dot product | 59,637 | 13.4 % | 48.6 |
| pixel-loop control | 18,651 | 4.2 % | 15.2 |
| payload (dot + requant/store) | 178,870 | 40.3 % | |

DualConv, both passes, kernel-only 618,350 cycles (span 634,889 minus the 16,539 inter-tile gap),
523,490 instructions (`exp5_fused_dual_conv/exp5_dualconv_breakdown/results/node_breakdown.txt`
and `node_breakdown_microbuild.txt`): phase 1 (perturb
W, perturb b, barrier) 2.1 k cycles = 0.3 %; phase 2 pixel loop 608,488 = 98.4 %; end-of-tile barrier
wait 6,242 = 1.0 %. Against 2× FlatConv the pixel loop saves 277,840 cycles: gather −148,198,
channel-loop control −81,202, requant/store −22,204, dot −11,131, pixel control −15,105. Per warm
pixel: flat 302 instructions / 360 cycles (payload 40.6 %); fused 426 / 495 (payload 61.6 %). The
algebraic split S + P (single) and S + 2P (fused) gives S = 224.2 cycles pass-independent, P = 138.5
per pass (`exp5a_breakdown/results/summary.txt`): 62 % of a single-pass pixel does not depend on
the pass (gather, bounds tests, pixel loop, parameter reloads from the stack, per-tile setup), which is
what fusion shares. It is also the case for tightening the pixel loop (§8 item 5): S is overhead, not
arithmetic.

### 6.4 Output transpose and double buffering (exp5a, exp5b)

Every output-transpose tile costs 48,575 instructions ≈ 67 k cycles on core 0 whatever its size: in
B (2 tiles of 39 KB), in C (3 tiles of 52 KB) and under double buffering (6 tiles of 26 KB) alike.
Root cause, read from the generated loop (the `output_0_pre_transpose` closure in
`exp5_fused_dual_conv/codegen/Network.c`): the tiler cuts the transpose along the **channel** axis
(`dimLen_3 = {6, 6, 4}` channels per tile in C; 4 + 4 in B), and the transpose template then
parallelises each tile **over that same channel axis** (`chunk = dimLen_3 / NUM_CORES`). With 4 or 6
channels in a tile, cores 0–3 (or 0–5) each take one channel and the remaining cores take nothing;
each working core moves one 14×701 channel plane with a byte-wise strided loop, ≈5 instructions per
byte. So a tile costs one channel's worth of time regardless of how many (≤ 8) channels it holds,
the transpose runs at 4/8 or 6/8 core utilisation, and its total is `tiles × 67 k`: anything that
raises the tile count raises it linearly. Remedies, in increasing effort: tile the transpose along H
or W (every tile then has all channels and all 8 cores work) or parallelise a tile over pixels;
for the fused node, emit NCHW directly and drop the transpose (§8 item 3). On this fixture the
transpose is 21–23 % of the run; with all cores busy it would be ≈12 %. Per-tile stamps:
`exp5b_tiling_profile/{pulpnn,flat,fused,fused_doublebuffer}/tiling_profile.txt` and
`exp5b_tiling_profile/results/summary.txt`.
Double buffering (`--doublebuffer`) halves the per-tile budget, takes the transpose from 3 to 6
tiles (+216,587) while recovering only ≈54 k of DMA waits: the fused run gets 16 % slower
(910,763 → 1,057,946, `exp5b_tiling_profile/fused_doublebuffer/logs_plain/run.log`). Verdict: not useful until the transpose is removed or re-tiled along H/W.
The conv itself is compute-bound (DMA waits 1.7–3.4 %) in all three kernels; per pixel: pulp-nn
67.7 cycles/pass, flat 45.3, fused 31.6.

### 6.5 GAP9, block 0, both passes (exact compare, `exp6_NE16_fusion/results/summary.txt`; plain runs `logs/{A,t1,t2,Ccl}_run_plain.log`, per-node `logs/*_run_microbench.log`)

| | what | cycles | vs A |
|---|---|--:|--:|
| A | 2× pulp-nn on the GAP9 cluster (exp3 fixture; 734,449 per pass) | 1,468,898 | 1.00× |
| A′ | 2× per-tap NE16 route (exp16c_SDK_port phase 1, 5,480,672 per pass, cited; stale-weight fixture, cycles only) | 10,961,344 | 0.13× |
| B | 2× NE16 im2col route, single pass (379,969 per pass) | 759,938 | 1.93× |
| C_cl | exp5's DualConv on the GAP9 cluster | 925,593 | 1.59× |
| C | NE16 im2col route, fused | 541,345 | 2.71× |

Ratios: A/B 1.93 (SDK route vs pulp-nn), A′/B 14.4 (vs our per-tap path), B/C 1.40 (fusion on
NE16), C_cl/C 1.71 (fused NE16 vs fused cluster: the decision number). Per node in C (master-core
wall time): output transpose 258,351 = 47 %, column buffer 142,171 = 26 %, NE16 job 136,618 = 25 %,
preparation 8,591 = 2 %. B → C saves 218,593 cycles: column built once (−142,229), one Cout-16 job
instead of two Cout-8 jobs (−73,026), preparation (−467), transpose unchanged.

**GAP9 vs Siracusa for the same cluster kernels.** The fusion ratio A/C_cl is 1.59× on GAP9 against
1.88× on Siracusa. Two identified contributors: GAP9 builds with GCC and Siracusa with clang 15
(`DualConv` 1,200 B vs 1,828 B of code), and the output transpose runs as 4 tiles on GAP9 (L1
92,000) against 3 on Siracusa (128,000), at ≈ 67 k per tile. A GAP9 instruction trace to pin the
GCC-vs-clang share has not been taken (§8).

---

## 7. Known pitfalls and lessons

1. **clang-15 negate trap.** Symptom: A+ exact, A− off by ±1 on every tap. Cause: the PULP clang-15
   build mis-evaluates `(select ±1) → negate → multiply`, so `-r*m` came out as `+m` for `r = +1`.
   Fix: form `prod = r*m` once and derive the minus noise as `(rounding − prod) >> S`; never negate
   `r` (`DualPerturb.h` header; inherited by the GAP9 kernels through the shared header).
2. **KERNEL SWITCH left flipped.** Symptom: every QZO conv on PULP silently runs `FlatConv`. Cause: the
   two-line switch in `PULPOpen/Platform.py`. Fix: restore the pulp-nn line and confirm `Network.c`
   calls `pulp_nn_conv_i8_i8_i8` (857,891 cycles on `QZO_Conv`).
3. **CMake caches `-D` flags.** Symptom: a run without `-D OUTPUT_TOL` passes at the tolerance of an
   earlier run. Cause: cache variables persist across runner invocations. Fix: `rm -rf TEST_*` between
   fixture kinds and pin flags in `test_siracusa_tiled_config.py::KERNEL_CMAKE_ARGS`.
4. **The fixture reference is NUM_CORES-specific.** Symptom: `ZO_Conv` at `--cores 2` gives 77,577
   errors. Cause: the seed folds in `core_id` and the chunking depends on `NUM_CORES`. Fix: rebuild
   `outputs.npz` for any other core count; run the fixtures at 8 cores.
5. **Value inference mistypes a constant vector.** Symptom: `RequantizedConv` fails to bind on an
   int32 add of all 32768 (`uint16_t`) or a bfloat16 bias. Fix: `--input-type-map` /
   `--input-offset-map` (carried by `KERNEL_GEN_ARGS`).
6. **Baked bias hides the rounding branch.** Symptom: single-layer NE16 fixtures pass while the full
   network is 20–55 % wrong. Cause: a constant requant add is the branch where merged and un-merged
   agree by construction. Fix: keep the add a runtime tensor (`--variable-bias`; every exp6 builder
   does this).
7. **Harness verdict is not a forward-pass check.** Symptom: `Errors: 0 out of 8` while losses are
   wrong. Cause: the MeZO harness checks the update graph driven by the fixture's reference losses.
   Fix: compare device losses against `speechnet_qzo12_update/outputs.npz`.
8. **GAP9 cluster task runs on core 8.** Symptom: `--profileMicrobenchmark` printed nothing on GAP9.
   Cause: wrappers guarded by `pi_core_id() == 0`, a worker. Fix: `PERF_MASTER_CORE`.
9. **Cluster printf is invisible on GAP9 GVSoC.** Fix: records go to an L2 table (`PerfRecords.c`)
   replayed by the FC after the cluster task (`deeploytest.c`, `[perf] records=N`). Caveat: the
   master core's cycles are the node's wall time, but its instruction/IPC/stall columns describe an
   idle core and must not be compared with Siracusa's core-0 columns.
10. **An untiled node keeps its tensors in L2.** Symptom: the column build cost ~1.4 M cycles.
    Fix: give the node a real (even single-tile) constraint so the data moves to L1.
11. **Tile constraints are called on the class.** Symptom: constructor arguments vanish. Fix: carry
    per-op configuration as class attributes (`NE16UntiledTileConstraint.INPUT_KEYS`).
12. **`computeTransientBuffersSize` is static and receives the symbolic representation.** Fix: build it
    in `constructSymbolicNodeRep` (the untiled constraints return the parser's dict, which carries
    `nodeName`).
13. **Missing prototype in the platform math header = implicit-int call.** Symptom: C_cl crashed
    with "Invalid access" on GAP9. Cause: GCC warns only; a `float` argument then shifts every later
    argument. Fix: prototypes in `DeeployGAP9Math.h`.
14. **NE16 1×1 padding is silent.** Cause: the GVSoC assert is commented out. Fix: never ask NE16 to
    pad (`ne16_1xkAdmissible` refuses pads; the im2col route writes pads into the column).
15. **Pad value must be 128, not 0**, in the column buffer (signed zero after the offset).
16. **Column order must match the weight fold** (`(ci, ky, kx)`): wrong order is bit-exact garbage on
    Cin = 1 and wrong on Cin > 1.
17. **Engine coloring runs before the NE16 passes.** Cause: `EngineColoringDeployer` interleaves a
    coloring pass between lowering passes, and a node that `canExecute` declines never reaches the
    rewrite. Fix: one admissibility predicate shared by engine and pass; insert the pass at index 1.
18. **A long Deeploy codegen is a symptom to shrink.** exp16a's "30-minute hang" was exponential
    backtracking over an infeasible binding; at K = 2 the same failure surfaced in 3 s.
19. **Read the error signature before the code.** Two contiguous wrong runs a tile stride apart
    means a buffer overrun; a constant diff per channel with channel 0 correct means a corrupted
    per-channel multiplier; position-dependent ±1 LSB at subtile borders means geometry.

---

## 8. Open items and next steps

1. **CHW perturbation order for Cin > 1** (agreed
   2026-10-05, deferred). The standalone QZO graph and the host reference perturb the weight in
   NCHW memory order (noise bit i → NCHW element i) and transpose afterwards. `DualConv.c` phase 1
   and `NE16DualPrep.c` perturb the buffer they are handed in its memory order: on the cluster that
   is the NHWC-transposed weight, which coincides with NCHW only for Cin = 1 (`[8][1][1][4]` ==
   `[8][1][4][1]` byte for byte). Every block-0 number in exp5/5a/5b/6 is unaffected (phase 2 never
   touches this). This is the same mistake the fp32 ZO path already made and fixed: a shortcut that
   folded the NCHW→NHWC permutation into the perturb node (`_foldLayoutIntoPerturb`,
   `TrainDeeploy/Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/LoweringOptimizationPasses.py`
   line 186) generated the Rademacher z over the transposed buffer, was bit-exact on conv 0 and wrong
   on convs 1–4, and was disabled in favour of a runtime Transpose after the perturb (comment at
   lines 303–311 of the same file). The fused kernels repeat that shortcut inside phase 1. Before any
   block 1–4 run: feed the fused cluster node the raw CHW weight (mark the input so
   `PULPNCHWtoNHWCPass` leaves it alone, as it does for pre-encoded NE16 weights), walk the CHW buffer
   in memory order in phase 1, compute `j = ((co*Kh + ky)*Kw + kx)*Ci + ci` from
   `i = ((co*Ci + ci)*Kh + ky)*Kw + kx` and write `wp[j]`, `wm[j]` (one index computation per weight
   element per tile: 2,048 elements for block 1, under 1 % of the node); the weight-transpose node
   (811 cycles on block 0) then disappears. For `NE16DualPrep.c` the input is already CHW; check that
   the `(ci, ky, kx)` fold order (= the im2col column order) survives. Verify with a block-1 fixture
   built like `build_dual_conv_fixture.py` (reference = the standalone graph through the oracle,
   exact compare); a wrong order shows up immediately as mismatches.
2. **Blocks 1–4 on the im2col route.** Cin·K = 128 / 128 / 112 / 224 fill TP_IN exactly, which is
   where NE16 should finally win; in the fused case the inputs differ per pass (after ReLU/MaxPool),
   so two column buffers per pixel or the pass as a batch dimension; the byte-wise copy must become
   word-wise. Measure before modelling (the cost models were wrong twice in exp16).
3. **Remove the output transpose** (47 % of C on GAP9, 25 % of exp5 on Siracusa): emit NCHW directly
   from the fused node (strided byte stores are free in L1), or tile the transpose along H/W so all
   cores work; this also makes double buffering worthwhile.
4. **Column buffer**: word copies on interior pixels (the SDK's `CopySeq`), bounds tests only at row
   edges, or the SDK's ping-pong double buffering against NE16 (26 % of C).
5. **Tighten the DualConv pixel loop** on the cluster (hoist the patch word, precompute weight-row
   pointers, unroll the 8-channel loop; one-word gather for Cin = 1): about −300 k of 615 k.
6. **Exporter support**: an Onnx4Deeploy lowering pass that emits `RequantizedDualConv` (cluster) or
   the `NE16Im2ColU8` / `NE16DualPerturbEncode` / `NE16DualBias` / 1×1 `Conv` form (NE16) from the
   QZO graph, and graph-level dualisation of the ops between convs so a ZO step is one
   `RunTrainingNetwork` with channel-stacked A±.
7. **Port the slave-stack relocation** of `deeploymezotest.c` to `deeploytest.c` (or the PR #183
   `SLAVESTACKSIZE` override) so kernel fixtures on GAP9 can use `--l1 110000`.
8. **GAP9 instruction trace** of `DualConv` to pin the GCC-vs-clang share of the 1.59× vs 1.88× gap.
9. **MCHAN 17-bit bound** is still an assert in `Deeploy/Targets/GAP9/DMA/MchanDma.py`; a tile over
   131,071 B will fail codegen rather than chunk.
10. Dead vendored code (`AllocateTemplate.py`, `MemoryLevelAnnotationPasses.py`) and the phantom
    `1 << 26` in `ConvTemplate.getConf0` are harmless but should be cleaned before an upstream PR.

---

## 9. Evidence and artifacts

Measurement logs are in the experiment folders listed below. Paths are relative to
`TrainDeeploy/DeeployTest/experiments/` unless they start with `TrainDeeploy/`.

| topic | tracked logs, results, scripts, fixtures |
|---|---|
| fixtures (all four block-0 kernels) | `TrainDeeploy/SpeechNet_Test/speechnet_unfused_b0_cluster/`, `TrainDeeploy/SpeechNet_Test/speechnet_fused_b0_cluster/`, `TrainDeeploy/SpeechNet_Test/speechnet_unfused_b0_NE16/`, `TrainDeeploy/SpeechNet_Test/speechnet_fused_b0_NE16/`; originals `TrainDeeploy/DeeployTest/Tests/Kernels/Integer/{QZO_Conv,QZO_DualConv,QZO_Conv_NE16,QZO_DualConv_NE16}/`; builders `dual_pass_fusion/build_*_fixture.py`, host mirror `dual_pass_fusion/zo_host_ref.py`; device check from the new location `TrainDeeploy/SpeechNet_Test/logs/smoke/{siracusa,gap9}_*.log` |
| exp3 baseline and the I-cache root cause | `dual_pass_fusion/exp3_as-is_QZOConv_microbench/`: `logs/run_plain.log`, `logs/run_microbench.log`, `results/summary.txt`, `results/schedule.txt` (fixture vs full network), `codegen/Network.c`, `codegen/full_network_qzo19_TrainingNetwork.c`, trace windows and tally script `logs/debug/{pair,ctl_pixel}{100,300,500}.log`, `logs/debug/tally_pair.py` |
| exp4 interleaving microbenchmark | `dual_pass_fusion/exp4_interleave_dualpass/results/step1_microbench.txt`, source `TrainDeeploy/DeeployTest/Platforms/Siracusa/src/dualpass_bench.c` |
| exp5 fused kernel | `dual_pass_fusion/exp5_fused_dual_conv/`: `logs/run_plain.log`, `logs/run_microbench.log`, `results/summary.txt`, `results/code_size.txt`, `codegen/Network.c`, `algorithms.tex` (pseudocode of both phases), `parse_microbench.py` |
| exp5a control kernel and level-0/1 breakdowns | `dual_pass_fusion/exp5a_breakdown/`: `logs/run_plain.log`, `logs/run_microbench.log`, `logs/run_plain_default_after_revert.log`, `results/summary.txt`, `results/code_size.txt`, `logs/debug/{cycle_breakdown,branch_classify,timeline_segments}.py`; `exp5a_conv_breakdown/{conv_breakdown.py,extract_windows.sh,results/*.txt}`; `exp5_fused_dual_conv/exp5_dualconv_breakdown/{conv_breakdown.py,extract_windows.sh,results/*.txt}` |
| exp5b tiling profile and double buffering | `dual_pass_fusion/exp5b_tiling_profile/{pulpnn,flat,fused,fused_doublebuffer}/tiling_profile.txt` and `logs/`, `results/summary.txt` |
| exp6 NE16 im2col route and fusion | `dual_pass_fusion/exp6_NE16_fusion/`: `logs/{A,t1,t2,Ccl}_run_{plain,microbench}.log`, `logs/exp5_regression_after_DualPerturb_refactor.log`, `logs/siracusa_microbench_sanity_after_perf_guard.log`, `results/summary.txt`, `codegen/{A,Ccl,t1,t2}/Network.c`, `parse_microbench.py` |
| exp16a/b per-tap and dense-3×3 routes | `deliverable/exp16_NE16_GAP9/exp16a_PW_single_layer/{build_fixtures.py,logs/,results/results.json,fixture/Network_*.c}`, `exp16b_Dense_single_layer/{logs/,results/results.json,fixture/}`; STEP 2a plumbing test `deliverable/exp16_NE16_GAP9/{build_block1_pw_fixture.py,logs/step2a_block1_pw*.log,fixture/Network_step2a.c}`; bit-serial encoding walkthroughs `deliverable/exp16_NE16_GAP9/_explain_{bitserial,encode_reshape}.py` |
| exp16c device encoder and signed-input bias | `deliverable/exp16_NE16_GAP9/exp16c_PW_single_layer/build_fixtures.py`, `deliverable/exp16_NE16_GAP9/exp16c_PW_single_layer/results/results.json`, its `logs/step1`–`step8_*.log` and its `fixture/Network_block0_ne16.c` … `Network_block4_ne16.c` |
| exp16c_SDK_port: pipelining, plain-graph pass, full QZO step on NE16 | `deliverable/exp16_NE16_GAP9/exp16c_SDK_port/`: `logs/phase1_block0_pipelined.log` … `phase1_block4_pipelined.log`, `logs/{b4_plain,plain3}.log`, `logs/vbias_*.log` (the three pre-existing bugs, before/after), `logs/qzo_baseline.log`, `logs/qzo_full_FINAL_bitexact.log`, `results/results.json`, `fixture/TrainingNetwork_qzo_ne16_FINAL.c` |
| NE16 hardware model and SDK kernels (inside `deeploy_gap9`) | `/app/install/gap9-sdk/gvsoc/gvsoc_gap/gap/ne16/src/` (v1, GAP9), `.../gap/ne16v2/` (not GAP9), `.../gap/gap9/cluster.py`; `/app/install/gap9-sdk/tools/autotiler_v3/{CNN_Generators_NE16,CNN_Libraries_HWPE}/` |
| GAP9 container, SDK environment, board flow | `TrainDeeploy/README_GAP9.md`, §5.3 above |

Numbers not verifiable from the current tree are marked as such above; everything else was read
from the files named.

---

**Companion guides.** `00_Overview.md` (entry point, environment, checkpoint rule, experiment index),
`01_BP_Guide.md` (back-propagation), `02_ZO_Guide.md` (float zeroth-order), `03_QZO_Guide.md`
(quantized zeroth-order), `04_FusedQZO_NE16_Guide.md` (fused dual-pass kernels and NE16).
