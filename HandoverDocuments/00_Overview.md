# On-device fine-tuning of SpeechNet with Deeploy — handover overview

Written 2026-10-12 for the handover to the supervisor and to any future collaborator or agent; guide 05
added 2026-10-14. This is the entry point. It indexes the five implementation guides, records the facts that every
guide depends on (repositories, containers, checkpoint, protocol), lists the standing working rules,
and ends with an index of every experiment that was run, with its path and status.

## 1. What this project is

The goal is on-device fine-tuning of the SilentWear SpeechNet model (a 5-block 1-D CNN for
surface-EMG silent-speech classification, 9 classes, 14 channels × 700 samples per window) on PULP
microcontrollers (Siracusa on GVSoC, and GAP9 with its NE16 accelerator), compiled by our fork of the
Deeploy compiler. Ready-to-run fixtures for every experiment (one naming scheme, each cross-checked
against the fixture its experiment ran on and smoke-run on GVSoC) and their run commands are in
`TrainDeeploy/SpeechNet_Test/README.md`. The work went through five stages, each with its own guide:

| guide | stage | one-line result |
|---|---|---|
| [01_BP_Guide.md](01_BP_Guide.md) | back-propagation (first-order) fine-tuning | full model, frozen pretrained BatchNorm statistics; round 1 reaches 87.78 % on batch 2 of session 3 (+11.11 pp over no adaptation), 180/180 inferences bit-exact; one forward+backward pass ≈ 33 M cycles (a device step of 4 accumulated windows ≈ 140 M), L2 peak 1.51 MB with the argmax-mask optimisation |
| [02_ZO_Guide.md](02_ZO_Guide.md) | float zeroth-order (MeZO) fine-tuning | two forwards per step instead of forward + backward; bit-exact on device; L2 peak 0.79 MB; round 1 reaches 90.00 % on batch 2, the best on-device result on this cell; the host simulation predicted it exactly |
| [03_QZO_Guide.md](03_QZO_Guide.md) | quantized zeroth-order (int8 convs) | int8 conv weights and int32 requant biases perturbed on device, fp32 BN and classifier; step cost 8.4–8.7 M cycles, 4.3× cheaper than fp32 ZO; round 1 reaches 85.00 % on device (host executor 85.56 %); the LSB stall and its three answers (lr 1e-5, master weights, ECO error feedback) |
| [04_FusedQZO_NE16_Guide.md](04_FusedQZO_NE16_Guide.md) | fused dual-pass QZO kernels and NE16 | one kernel computing both ε-passes of the block-0 conv: 1.88× on the fixture, 2.16× on the node, bit-exact; on GAP9 the same fusion on the NE16 accelerator via the SDK's im2col route gives 2.71× over 2× pulp-nn and 1.71× over the fused cluster kernel |
| [05_Fusion_Full_Network_Guide.md](05_Fusion_Full_Network_Guide.md) | dual-pass fusion for blocks 1–4 and the whole QZO network | two-input fused cluster kernel and pipelined fused NE16 node for blocks 1–4 (Siracusa 2.44×, GAP9 best per block 2.57×; NE16 for blocks 0–2, cluster for 3–4); a Deeploy pass rewrites the QZO training graph into one channel-stacked dual-pass graph, bit-exact on Siracusa / GAP9 / GAP9 + NE16 (step 1.53× / 1.42× / 1.78×); QZO+ECO round on GAP9 + NE16 predicted at 40.0 s vs 69.6 s unfused; exp9 (memory): frozen-BN tail fused into one kernel, fused QZO+ECO L2 arena 854 → 266 KB, step −13 %, bit-exact |

The same experimental cell is used throughout: subject S01, condition vocalized, held-out session 3
(fold 3). Fine-tuning data are 54 stratified windows (30 % of batch 1, seed 42, drawn by the
exporter); evaluation is balanced accuracy on the whole next batch (180 windows).

## 2. Repositories, containers, checkpoint

All repositories live under `/Users/qiwenwu/ETH/`.

| directory | role | git branch |
|---|---|---|
| `TrainDeeploy/` | our fork of Deeploy: device compile, code generation, GVSoC runs, kernels, harnesses, experiments | `feat/ZO_2fwd_fusion` (feat/BP+ZO → feat/QZO → feat/GAP9_w_NE16 are merged forward into it) |
| `Onnx4Deeploy/` | PyTorch / Brevitas → ONNX exporter; host references; calibration; host QZO studies (`QZO_exp/`) | `feat/ZO_2fwd_fusion` |
| `SilentWear/` | dataset (`SilentWear_data/data_raw_and_filt`), checkpoints, the paper's training code, and the host PyTorch simulations of the on-device recipes (`SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/`) | not a repo of ours |
| `Deeploy/`, `Onnx4Deeploy_ZO/` | read-only upstream mirrors (Deeploy `devel`; the shipped ZO export branch `ZO`). Never edit. | — |

Three Docker containers, each with a different mount prefix:

| container | image | used for | repo path inside |
|---|---|---|---|
| `agitated_hugle` | python:3.10 (brevitas 0.13, onnxruntime-training) | exports, calibration, host PyTorch/Brevitas simulations | `ETH` → `/app` (so `/app/Onnx4Deeploy`, `/app/TrainDeeploy`, `/app/SilentWear`) |
| `traindeeploy` | deeploy:traindeeploy-dev (pulp-sdk, LLVM 15) | Siracusa builds and GVSoC runs | `ETH` → `/app/ETH` (so `/app/ETH/TrainDeeploy`) |
| `deeploy_gap9` | ghcr.io/runwangdl/deeploy:gap9 (private GAP9 SDK 5.21.1, GCC, GVSoC; amd64 under qemu) | GAP9 and NE16 builds and runs | `TrainDeeploy` → `/app/Deeploy` |

Every command in `deeploy_gap9` must first run

```bash
source /app/install/gap9-sdk/.gap9-venv/bin/activate
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
```

Before every GVSoC run, kill orphaned simulators by PID (name-based kills do not work) and clear
the build directory when the fixture kind changes, because CMake caches `-D` flags:

```bash
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
rm -rf TEST_SIRACUSA      # or TEST_GAP9 / TEST_GAP9_W_NE16
```

**Checkpoint rule.** The correct pretrained weights are

```
TrainDeeploy/SpeechNet_Test/pretrained/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
```

(a copy of SilentWear's `artifacts_reference` checkpoints for all subjects and conditions, shipped in
the repo; the lab-machine reproduction that matches the paper; no-adaptation accuracy on batch 2 = 76.67 %).
The tree `SilentWear/SilentWear/artifacts/` is a stale Mac reproduction: all 37 tensors differ and it
stopped 17 epochs earlier. Everything measured on it before 2026-09-14 is **not paper-comparable**.
The correct-weight device results are `TrainDeeploy/DeeployTest/experiments/deliverable/exp17_BP_round1_correct_weights`,
`exp18_ZO_round1_correct_weights`, `exp19_QZO_round1_correct_weights` and `exp20_QZO_ffast_math`; every older accuracy figure in the tree carries a stale-weights caveat in this handover.
Calibration for QZO must be regenerated whenever the weights change. The current activation calibration
is the pooled absolute percentile at **99.9**
(`TrainDeeploy/SpeechNet_Test/calibration/pooled_999_fold3_ref.json`); exp19's device round used 99.99
(`pooled_9999_fold3_ref.json`, kept for reproducing exp19). Why 99.9 and how to regenerate: guide 03 §3.

## 3. Standing working rules

These were agreed with the user over the project and apply to anyone continuing the work.

1. No faked or extrapolated results; every number cites its experiment folder and its checkpoint.
2. Comment out, do not delete, shipped or original SpeechNet code in TrainDeeploy and Onnx4Deeploy;
   implement alongside. Our additions are marked `-- QW` (Python) or `QW` in C comments.
3. Match the existing pipeline: new training modes are emitted natively by Onnx4Deeploy (weights as
   graph inputs, `BatchNormInternal`, canonical two-output SCE), not patched downstream.
4. Verify before diagnosing: run the ONNX through onnxruntime or the executor first; never claim a
   feature is absent from a grep, trace the flow entry to exit and inspect the artifacts.
5. Bit-exactness is a bring-up gate, accuracy is the acceptance criterion. The default build for every
   method (BP, ZO, QZO) is `-ffast-math`, set in `TrainDeeploy/cmake/pulp/pulp.cmake`; it is
   learning-neutral (QZO round 1 reached the same batch-2 accuracy as the host executor). Strict fp32
   (`-D DEEPLOY_STRICT_FP32=ON "DEEPLOY_STRICT_FP32_FILES=BatchNorm.c;Gemm.c"`) is an opt-in debugging
   mode for bit-exact comparison against the host reference; it costs +3.84 % per QZO step (exp20).
6. Never run an evaluation next to a live device run with an unscoped `rm -rf TEST_SIRACUSA` or a
   global GVSoC kill (exp9's harness destroyed an exp18 round at update 1566 of 2700; fixed in commit
   93458a0, but check any older harness before reuse).
7. The `KERNEL SWITCH` on the `Conv2DMapper` line of `Deeploy/Targets/PULPOpen/Platform.py` selects
   pulp-nn (default) or FlatConv for every int8 `RequantizedConv` on PULP. It must be on the default
   when committing.

## 4. The pipeline in one picture

```
Onnx4Deeploy (agitated_hugle)                     TrainDeeploy (traindeeploy / deeploy_gap9)
 Onnx4Deeploy.py -model SpeechNet -mode <m>        pack fixture -> DeeployTest/Tests/Models/Training/SpeechNet/<name>_{train,update}
   infer      : int/float inference graph           deeployTrainingRunner_tiled_siracusa.py   (BP)
   train      : forward+backward+optimizer (BP)     deeployMezoRunner_tiled_siracusa.py       (ZO / QZO, Siracusa)
   zo-train   : zo_train + zo_update graphs (fp32)  deeployMezoRunner_tiled_GAP9.py           (ZO / QZO, GAP9)
   q-zo-train : same, int8 convs (QZO)              deeployMezoRunner_tiled_GAP9_w_NE16.py    (QZO with NE16 colouring)
 + PyTorch / executor host reference                deeployRunner_tiled_{siracusa,gap9,gap9_w_ne16}.py (kernel fixtures)
 + calibration JSON (QZO)                           -> dump weights -> build int8 inference fixture -> evaluate batch 2
```

Each guide documents its mode end to end: exporter functions, graph contract, Deeploy parsers,
templates, tile constraints, kernels, harness, runner flags, and how to reproduce the numbers.

## 5. Shared documentation

The shared documentation consists of the six handover documents in `TrainDeeploy/HandoverDocuments/`
(this overview and guides 01–05), `TrainDeeploy/SpeechNet_Test/README.md`, and the upstream
documentation: `TrainDeeploy/README_TRAINING.md` (training-infrastructure contracts: fixture
layout, npz rules, runner flags), `TrainDeeploy/README_GAP9.md` (GAP9 container, SDK, board),
`TrainDeeploy/docs/`, `Onnx4Deeploy/README.md` and `Onnx4Deeploy/docs/` (for example
`Onnx4Deeploy/docs/training_graph_generation.md`, and `Onnx4Deeploy/docs/Quantization_Integration.md` §2 for the QCDQ →
`RequantizedConv` fold chain). Measurement logs, results and scripts are in the experiment folders
listed in §8.

## 6. Points not covered by the guides

- **Dataset and windowing.** One window per utterance, anchored at the onset of each label run (the
  window starts at the run's first sample; no sliding window, no stride or overlap, no centring):
  700 samples at 500 Hz (1.4 s; hence the `WIN_1400` folder name), the 14 filtered `Ch_*_filt`
  channels put in the model's `channel_order`, giving `(N, 14, 700)` float input with no input
  normalisation (the paper config leaves `normalize_input` false). The exporter's windowing was fixed
  from centre to onset in Onnx4Deeploy commit 77e269d; results before that fix are not comparable.
- **Host simulations.** The host PyTorch simulations of the device recipes are in
  `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/` (outside these repositories); they replaced
  the organically grown `exp1..exp21` folders. One definition of model, data, device-faithful
  primitives and protocol in `common/` (`speechnet.py`, `data.py`, `device_faithful.py`,
  `zeroth_order.py`, `device_rng.py` = the device's xorshift32 Rademacher stream with per-core seeds,
  `protocol.py` = the b1→b5 incremental protocol); each experiment injects only its training step.
  Zero-shot cells are validated against the paper's `ft_summary.csv` before any fine-tuned number is
  quoted. Four-subject (S01–S04), three-seed means (fold 3, vocalized, batches 2–5): no adaptation
  72.43, MeZO 78.02 ± 0.30, backprop 78.69 ± 0.66, paper recipe 82.67.
- **Three device behaviours the host must copy.** (a) The labelled fine-tuning draw comes from the
  exporter (`np.random.RandomState`, then shuffled), not from a host-side stratified draw: with the
  same seed 42 the two share only 16 of 54 windows, so the host must reproduce the exporter's selection
  byte for byte including order. (b) The device never shuffles: window index = `mb % n_windows` with
  `mb = update_step * n_accum + accum_step` (`speechnet_exporter.py:365`), so the host uses cyclic order.
  (c) ZO directions must come from the device PRNG; an independent z is statistically equivalent but
  follows a different trajectory (matching it moved the host ZO result by +0.67 pp).
- **Gradient accumulation conventions.** Deeploy's BP accumulator sums (`InPlaceAccumulatorV2`, no
  1/N); the ZO harness divides the projected gradient by `n_accum` (`deeploymezotest.c`). Copying one
  convention into the other silently rescales the step. The exporter bakes the learning rate into the
  optimizer graph.
- **BatchNorm.** `BatchNormInternal` is flag-gated: device training and inference of our graphs need
  `-D BN_FROZEN_STATS=ON`, otherwise BN normalises with single-window batch statistics. Batch-1 live
  BN is why naive full-model fine-tuning collapses on device: the forward batch is pinned to one
  window by the L2 budget, and `n_accum` enlarges only the optimizer's batch, never BN's.
- **Requant rounding, two sites.** The `+div/2` rounding constant is baked by `PULPConvRequantMergePass`
  only for a constant bias. Our perturbable bias skipped it, so the fused conv truncated; the floor's
  −0.5 LSB is not antithetic-symmetric and sign-flips the ZO gradient. Fixed in the exporter
  (`qzo_weight_integerize.py`) and, independently, found again on the device template side during the
  NE16 work (`RequantShiftTemplate.py`).
- **Counting cycles on PULP.** Three instruments must not be mixed: the cluster timer (`Runtime:` wall
  clock around the cluster task), the core-0 cycle counter printed per node by the microbenchmark
  wrapper (it stops while the core sleeps at barriers and DMA waits; the wrapper also resets the
  cluster timer, so a microbench log's `Runtime` is only the last profiled node), and address-range
  spans from a GVSoC instruction trace. Node counters are compared with node counters; never divided
  by a plain runtime from another build. Details in guide 04.
- **GAP9 memory.** Usable L1 is about 98 KB in the inference harness (`--l1 92000` for kernel fixtures)
  and 123 KB in the MeZO harness, whose `deeploymezotest.c` relocates the cluster slave stacks to L2
  (`--l1 110000`). `--l2 1500000 --defaultMemLevel L2`.
- **Fine-tuning data placement on Siracusa.** The 54 fine-tuning windows are 54 separate per-window
  buffers fed one at a time (effective batch 1), stored in the 4 MB `WEIGHTMEM_SRAM` bank together
  with the weights, not in the 2 MB L2 arena; only one window (38.3 KB) and its tiled activations
  occupy L2/L1 per step. About 55 % of a batch (~100 windows) fits on chip, so 30 % is a choice, not a
  memory limit; more would need the L3-offload path.

## 7. Open items across the project

1. ~~Blocks 1–4 through the fused kernels~~ — done 2026-10-14 (guide 05): the two-input kernel perturbs the raw CHW
   weight in CHW order and writes HWC; the NE16 path needed no fix.
2. ~~A lowering pass for the fused and NE16 graphs~~ — done differently 2026-10-14 (guide 05): a Deeploy pass
   (`ZODualPassFusion`, `--zo-fused-dual`) rewrites the ordinary QZO export at compile time; no exporter change.
3. ~~Remove the output transpose after the fused conv~~ — resolved by guide 05: it exists only at the fixture boundary;
   the full network stays channels-last. Open instead: the non-conv nodes (63 % of the fused GAP9 step), the board L2
   budget and stack depth of the fused graph, and a full fused round (guide 05 §9).
4. Port the slave-stack relocation to the GAP9 kernel harness; trace the GAP9 cluster kernels to pin the
   GCC-vs-clang explanation of the 1.59× vs 1.88× fused-cluster ratio.
5. Cross-fold robustness of QZO at lr 1e-5 (loses 4.86 pt to float ZO on fold 2) and ECO on device.

## 8. Experiment index

Paths are relative to `/Users/qiwenwu/ETH/` and name the experiment folder, followed where useful
by its main tracked log or results file. "Checkpoint" says which weights produced the numbers:
`ref` = `artifacts_reference` (paper-comparable), `stale` = `artifacts/` (not paper-comparable),
`fixture` = inherits a `ref` fixture. Status: CURRENT = cite it; SUPERSEDED = replaced by the named
experiment; context = useful for its method or root cause, not for its numbers. Host-simulation
rows point to code in `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/`, outside these repositories.

### 8.1 Back-propagation (guide 01)

| path | goal | result | checkpoint | status |
|---|---|---|---|---|
| `TrainDeeploy/DeeployTest/experiments/maxpool_numerical_drift/` (main files: speechnet_maxpool_90_argmax.log, speechnet_argmax_ort_ref.py) | device-vs-ORT loss drift in MaxPool fine-tuning | argmax tie-flips, bit-exact ~36 steps then discrete onset; not a precision deficit | stale | CURRENT (root cause) |
| `TrainDeeploy/DeeployTest/experiments/headonly_ondevice_finetune/` (main files: speechnet_accuracy_results.json, speechnet_100step_run.log) | first on-device recipe that beats zero-shot | +4.44 pp (ep40); naive FT fails because of batch-1 BN; fix = fold BN into the last conv | stale (older base) | SUPERSEDED by fixedwindow |
| `TrainDeeploy/DeeployTest/experiments/headonly_ondevice_ft_fixedwindow/` (main files: eval_b2_zs_summary.log, eval_b2_ft_summary.log) | head-only + BN-fold after the onset-windowing fix | b2 zero-shot 81.67 % = paper; head-only FT 89.44 % | stale (`inter_session_ft`) | CURRENT for the head-only recipe |
| `TrainDeeploy/DeeployTest/experiments/fulltraining_naccum8_regression/` (main files: speechnet_ft_full_naccum8_sweep.json, full8_e40_device.log) | does unfolded full-model FT help? | no: −17.78 pp, BN running-stat non-update dominates | stale | CURRENT (negative result) |
| `TrainDeeploy/DeeployTest/experiments/frozen_bn_kernel_finetune/` (main files: frozenbn_grid_results.txt, frozenbn_device_final.log) | full model with frozen pretrained BN stats + the `BatchNorm.c` kernel change | −17.78 → +5.56 pp on host grid; `BN_FROZEN_STATS` kernel bit-exact | stale | CURRENT (recipe S2) |
| `TrainDeeploy/DeeployTest/experiments/speechnet_ondevice_incremental_fold3_headonly/` (main files: logs/round{2,3,4}_accuracy.txt) | head-only chain b1→b5 vs PyTorch | mean b2–5 83.89 (PyTorch 85.42); b4 gap is a draw difference | stale | CURRENT (chain) |
| `TrainDeeploy/DeeployTest/experiments/speechnet_ondevice_incremental_fold3_lastblock/` (main files: k1_vs_headonly_vs_paper.csv) | chain with K=1 scope (last conv block + classifier trained) | mean 84.17; runner's `Errors 2160/2160` is a false failure | stale | CURRENT (K=1) |
| `TrainDeeploy/DeeployTest/experiments/reference/` (main files: paper_incremental_ft_s01_fold3.py and .log) | the paper's incremental fine-tuning protocol for S01 fold 3 reproduced on host; why BN and `n_accum` cannot be fixed by accumulation | training BN uses the current window's statistics while inference uses running stats; the device forward batch is 1, `n_accum` enlarges only the optimizer batch; resolution = fold or freeze BN | — | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/exp1/` (main files: ondevice_sim_S01_fold3/results/ondevice_vs_pytorch_S01_fold3.csv; optionB_ort_reference/) | correct ORT reference for frozen-BN training | Option A (PyTorch frozen reference) works; B-decompose does not compile on Siracusa | stale | CURRENT (reference fix); chain SUPERSEDED by exp17 |
| `TrainDeeploy/DeeployTest/experiments/exp2/` (main files: profiling_run/cycle_summary.txt, profiling_run/memory_alloc_training.html) | latency + memory of one BP step | 140.4 M cycles/step at n_accum 4 (fwd 70.7 M, bwd 69.4 M); 94.4 % compute; L1 99.9 % | fixture | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/exp3/` (main files: logs/) | argmax mask instead of recompute MaxPool backward | L2 1,793,800 → 1,511,308 B (−15.7 %), bit-exact | fixture | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/exp4/` (main files: logs/gap9_l2_1p5M.log) | does BP fit GAP9's 1.5 MB L2? | yes, L2 peak 1,482,636 B (read from the `--plotMemAlloc` plot, which is not tracked; re-run with that flag to regenerate it) | fixture | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp1_BP_argmax_off/` (main files: logs/sim.log, logs/profiletiling.log, results/) | single-step BP, recompute baseline, latency derivation | L2 peak 1,735,180 B; 95 % compute; 34.1 M profiled cycles | stale | CURRENT (latency) |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp2_BP_argmax_on/` (main files: logs/sim.log, results/memory_alloc_deeployStates.html) | argmax mask on | L2 1,511,308 B | stale | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp2a_BP_argmax_naccum1/` (main files: logs/sim.log) | n_accum 1 for comparison with ZO | 32,708,926 cycles/step vs ZO 35,643,810 | stale | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp4_BP_round1/` (main files: results/eval_b2_results.json) | full BP round 1 | b2 87.22 % | stale | SUPERSEDED by exp17 |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp17_BP_round1_correct_weights/` (main files: results/eval_b2_results.json, logs/phase3_gvsoc_train_round1.log, scripts/run_round1.sh) | BP round 1 on the correct weights | b2 87.78 %, 180/180 bit-exact, +11.11 pp | ref | CURRENT |
| fine-tuning data placement (see §6), measured on the head-only training fixture `speechnet_train_head_ep40_isft_fw` | where the FT window pool lives in Siracusa memory | `WEIGHTMEM_SRAM`, one window at a time; verified against the generated code and `link.ld` | — | CURRENT |
| host simulation code in `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp1_BP_simulation/` (outside these repositories) | host simulation of the device BP recipe | reproduces the device batch-2 cell exactly (87.78) | ref | CURRENT |
| host simulation code in `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp7_BP_ablation/` (outside these repositories) | attribute the 4.7 pp gap between device-faithful BP (77.95, seed 42, 4 subjects) and the paper's FT (82.67) to the eight recipe differences | planned; not run | ref | open |

### 8.2 Float zeroth-order (guide 02)

| path | goal | result | checkpoint | status |
|---|---|---|---|---|
| `TrainDeeploy/DeeployTest/experiments/zo_smoke/` (main files: logs/, gen_perturb_fixture.py) | ZO harness bring-up; step-0 bias | L1 overlap for the odd-length fc bias; overlap-safe perturb kernel; 0/32 errors | smoke | CURRENT (harness design) |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp3_ZO/` (main files: logs/sim.log, results/memory_alloc_deeployStates.html) | single-step MeZO | bit-exact 0/2; L2 789,688 B (−55 % vs BP); 35,643,810 cycles | stale | CURRENT (memory) |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp5_ZO_round1/` (main files: results/eval_b2_results.json, logs/round1_gvsoc_zo.log.gz) | full ZO round 1 | b2 88.33 %; 0/21,600 errors | stale | SUPERSEDED by exp18 |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp6_ZO_single_step_latency/` (main files: logs/profiletiling.log, results/) | where a ZO step spends cycles | 35.87 M: two forwards 99.6 %, update 0.44 %, perturbation 1.3 % | stale | CURRENT (latency baseline) |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp18_ZO_round1_correct_weights/` (main files: results/eval_b2_results.json, logs/phase3_gvsoc_zo_round1.log, scripts/run_round1.sh) | ZO round 1 on the correct weights | b2 90.00 %, 180/180 bit-exact; 385 G cycles; `TrainDeeploy/DeeployTest/experiments/deliverable/exp18_ZO_round1_correct_weights/logs/phase3_gvsoc_zo_round1_KILLED_at_1566.log.gz` is the run destroyed by a concurrent evaluation (rule 6) | ref | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp1_as-is_microbench/` (main files: logs/run_microbench.log, results/schedule.txt) | fp32 ZO_Conv block 0 utilisation | 7.3 M cycles, 2 of 8 cores active, instruction-bound im2col | ref fixture | context |
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp2_as-is_2ndConv_microbench/` (main files: logs/run_microbench.log) | same for block 1 | 0.414 MAC/cycle, 8/8 cores | ref fixture | context |
| host simulation code in `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp2_ZO_simulation/` (outside these repositories) | host simulation with the device PRNG | predicted 90.00 % before exp18 ran; device losses replayed to 2e-6 | ref | CURRENT |
| host simulation code in `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp5_acc_epo/`, `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp6_acc_epo_revised/`, `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp10_ZO_acc_epo_70%/`, `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp9_half_batch_simulation/` (outside these repositories) | accuracy vs epochs (ZO, QZO, QZO+ECO) and vs labelled-data size | accuracy plateaus by epoch ~50; a 50 % draw (90 windows) instead of 30 % gives BP +2.43 pp (4-subject mean 77.95 → 80.38), ZO +1.08 and QZO+ECO +0.91 (single seed, within noise) | ref | CURRENT |
| `Onnx4Deeploy/demo_bp_vs_zo/` (local, untracked on this branch; tracked on the Onnx4Deeploy branch `ZO`) | the LightweightCNN demo fixtures of the ZO port | — | — | context |

### 8.3 Quantized zeroth-order (guide 03)

Device (`TrainDeeploy/DeeployTest/experiments/deliverable/`):

| path | goal | result | checkpoint | status |
|---|---|---|---|---|
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp7_QZO_single_step/` (main files: logs/sim.log) | QZO single-step smoke | bit-exact 0/2 | stale | CURRENT (bring-up) |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp8_QZO_8steps_4accum/` (main files: brevitas_evidence/, brevitas_stall_study/) | two real updates | 0/16; int8 conv weights byte-identical (sub-LSB stall) | stale | CURRENT (stall evidence) |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1/` (main files: logs/) | first round 1 at lr 1e-5 | failed, 11,171/21,600 loss errors; its harness destroyed an exp18 run (rule 6) | stale | SUPERSEDED |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp10_QZO_single_step_profiling/` (main files: logs/profiletiling.log, exp10a/results/comparison.json) | why a QZO step was 10.8× a ZO step | Quant/Dequant 97.9 % of cycles; fixed kernels | stale | CURRENT (bottleneck) |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp11_QZO_round1_fix/` (main files: logs/round1_gvsoc_strict.log.gz, logs/eval_b2_device.log.gz) | round 1 on the fixed stack, strict fp32 | 7,098/7,112 pre-cascade losses identical; cascade at step 888; b2 89.44 % | stale | CURRENT for the strict-fp32 fix list |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp11_bitexact_SCE/` (main files: logs/sce_probe_tol0.log.gz), `TrainDeeploy/DeeployTest/experiments/deliverable/exp11_bitexact_SCE_verif/` (main files: compare.py, logs/device_to891.log.gz) | last residual and its causal proof | picolibc `expf`/`logf`; 1-ulp shared scalar at step 884 | — | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp12_QZO_clean_round_1/` (main files: logs/round1_gvsoc_ffastmath.log.gz, results/pytorch_ref.json) | clean-room round 1 with ffast-math | device 88.33 % = host executor | stale | SUPERSEDED by exp19 |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp13_QZO_GAP9_single_step/` (main files: logs/gap9_PASS_l1_110000_stacksL2.log) | GAP9 port | passes 0/8; `--l1 110000` + slave stacks to L2 | — | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp19_QZO_round1_correct_weights/` (main files: results/eval_b2_results.json, logs/phase3_gvsoc_qzo_round1.log.gz, fixture/pooled_9999_fold3_ref.json) | QZO round 1 on the correct weights, calibration regenerated | device 85.00 %, host executor 85.56 %, PyTorch 87.78 %; 89.5 G cycles. Calibrated at pooled@99.99; the current setting is 99.9 (guide 03 §3). | ref | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp20_QZO_ffast_math/` (main files: results/summary.txt) | cost of strict fp32 | +3.84 % per step (8,695,234 → 9,029,363 cycles); strict fp32 stays an opt-in debugging mode; the default build keeps `-ffast-math` | ref | CURRENT (newest) |

Host (`Onnx4Deeploy/QZO_exp/`):

| path | goal | result | status |
|---|---|---|---|
| `Onnx4Deeploy/QZO_exp/` (the folder as a whole; tracked scripts in its `exp1/` … `exp13/` subfolders) | the locked QZO design and iterations 1–20 up to bit-exact device validation | five bugs B1–B5 found and fixed (listed in guide 03) | CURRENT |
| `Onnx4Deeploy/QZO_exp/exp_calibration/` (main files: results.json, calib_pooled.py, run_study.py) | is calibration the cause of the stall? | pooled percentile 99.99 over 1800 windows gives 85.6 % zero-shot; calibration does not move the stall (stale weights). On the correct weights a host sweep favours 99.9 (guide 03 §3). | CURRENT |
| `Onnx4Deeploy/QZO_exp/exp3_lr1e-5_stability/` (main files: results.json, run_all.py) | lr 1e-5 across folds | −4.86 pt on fold 2 | CURRENT (caveat) |
| `Onnx4Deeploy/QZO_exp/exp4_ceiling/` (main files: results.json, run_ceiling.py) | force every update to ≥1 LSB | un-stalls and destroys accuracy (51.67 %) | CURRENT (negative) |
| `Onnx4Deeploy/QZO_exp/exp5_A_vs_B/` (main files: matched_z_losses.json, matched_z_check.py) | fake-quant vs integer datapath | device is not the problem | SUPERSEDED by exp10 |
| `Onnx4Deeploy/QZO_exp/exp10/` (main files: test_requant_rounding.py, device_eval_results.json) | root cause of the A-vs-B gap | requant truncation sign-flips the ZO gradient | CURRENT |
| `Onnx4Deeploy/QZO_exp/exp11/` (main files: fma_scan.py, fma_scan_strict.py, surrogate/ANALYSIS.txt) | device-faithful host executor | strict-fp32 round 0/21,600 | CURRENT |
| `Onnx4Deeploy/QZO_exp/exp12/` (main files: analysis.txt), `Onnx4Deeploy/QZO_exp/exp13/` (main files: analysis.txt, micro/probe_summary.txt) | lr 3e-6 control; conv frozen | the whole residual is the float path, activation Quant amplifies | CURRENT |
| `Onnx4Deeploy/QZO_exp/exp14_fold_BN/` (main files: results*.json, fold_bn_sim.py) | folded-BN design | direct int8 never beats zero-shot after folding | CURRENT |
| `Onnx4Deeploy/QZO_exp/exp15_QZO_eco/` (main files: results.json, run_eco.py, eco_summary.png) | ECO error feedback | 86.81 % without master weights; residual injection, not momentum, breaks the stall | CURRENT |
| `Onnx4Deeploy/QZO_exp/exp19_ref_full/` (local, untracked; regenerate with the export command in guide 03; its calibration is tracked as `TrainDeeploy/DeeployTest/experiments/deliverable/exp19_QZO_round1_correct_weights/fixture/pooled_9999_fold3_ref.json`) | the correct-weight QZO export used by every later device and kernel experiment | — | CURRENT fixture |
| `Onnx4Deeploy/QZO_exp/` subfolders `exp1`, `exp2`, `exp8`, `exp9*`, `exp11_*`, `exp12_clean`, `meeting_prep_fcfloat`, `exp_brevitas_stall`, `exp_masterweight_ft` | fixtures and early studies (tracked scripts and results: `Onnx4Deeploy/QZO_exp/exp1/`, `Onnx4Deeploy/QZO_exp/meeting_prep_fcfloat/`, `Onnx4Deeploy/QZO_exp/exp_brevitas_stall/results.json`, `Onnx4Deeploy/QZO_exp/exp_masterweight_ft/results.json`; the other subfolders hold local, untracked fixtures) | — | context |
| host simulation code in `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp3_QZO_simulation/`, `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp4_QZO_eco_simulation/`, `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp11_acc_epo_S02/`, `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp12_acc_epo_S03/`, `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp13_acc_epo_S04/` (outside these repositories) | host QZO and ECO simulations; epoch curves (ZO, QZO, QZO+ECO) on the other subjects | 4-subject mean b2–b5: QZO at lr 1e-5, pooled@99.99 = 75.56 (S01 85.56); QZO+ECO (lr 3e-6, memory-free error feedback, stochastic rounding, pooled@99.9) = 78.40 (S01 87.50) | CURRENT |

### 8.4 Fused dual-pass kernels and NE16 (guide 04)

`TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/` (all on the `ref` qzo19 fixture, exact integer compare):

| path | goal | result | status |
|---|---|---|---|
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp3_as-is_QZOConv_microbench/` (main files: logs/run_microbench.log, logs/run_plain.log) | baseline: int8 block-0 conv as deployed | fixture 858,185 cycles; conv node 664,646; 22.8 % I-cache refill stalls | CURRENT baseline |
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp4_interleave_dualpass/` (main files: results/step1_microbench.txt) | does interleaving alone help? | 1.03× at block-0 shape (host microbench only) | SUPERSEDED by exp5 |
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp5_fused_dual_conv/` (main files: results/summary.txt, logs/run_microbench.log) | the fused kernel | 910,763 vs 1,716,370 (1.88×); node 614,979 vs 1,329,292 (2.16×); 0/157,024 errors | CURRENT headline |
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp5_fused_dual_conv/exp5_dualconv_breakdown/` (main files: results/node_breakdown.txt) | level-1 trace breakdown of the fused node | phase 1 = 0.3 %; gather and loop control shared | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp5a_breakdown/` (main files: results/summary.txt) | rewrite vs fusion split with the FlatConv control | rewrite 1.49× × fusion 1.45×; the output transpose root cause and the cycle-counting rules are in guide 04 | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp5a_breakdown/exp5a_conv_breakdown/` (main files: results/node_breakdown.txt) | level-1 breakdown of FlatConv | gather 33.5 %, requant+store 26.9 %, loop control 21.9 %, dot 13.4 % | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp5b_tiling_profile/` (main files: results/summary.txt) | per-tile profiles, double buffering | compute-bound; double buffering 16 % slower because of the transpose | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp6_NE16_fusion/` (main files: results/summary.txt) | fusion on NE16 via the SDK im2col route | 2× pulp-nn 1,468,898; 2× NE16 im2col 759,938; fused cluster 925,593; fused NE16 541,345 (2.71×) | CURRENT (frontier) |

NE16 bring-up (`TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/`):

| path | goal | result | status |
|---|---|---|---|
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16a_PW_single_layer/` (main files: results/results.json, logs/) | block-1 1×16 conv as 16 pointwise NE16 jobs | bit-exact 0/19,712; 1,567,096 cycles | CURRENT (per-tap route) |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16b_Dense_single_layer/` (main files: results/results.json, logs/step7_dense3x3_K16_FULL_fixed.log) | dense 3×3 chunks | 1,364,975 cycles, 1.15× faster; dispatch-bound | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16c_PW_single_layer/` (main files: results/results.json, logs/) | on-device perturb + weight encode + conv | all 5 convs bit-exact; NE16 4.3–7.5× slower than the cluster | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16c_SDK_port/` (main files: results/results.json, logs/vbias_ne16_FINAL_pass.log, logs/qzo_full_after_rounding_fix.log) | full SpeechNet QZO on GAP9 with NE16 | losses bit-identical to cluster after three Deeploy fixes; 5.8 % of conv MACs reach NE16 | CURRENT |

The early NE16 steps (STEP 1 and 2a) have their logs in
`TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16a_PW_single_layer/logs/` and the
helper scripts at the top level of `TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/`.
`TrainDeeploy/DeeployTest/experiments/deliverable/exp_profiletiling_bug/logs/` holds two profiling
logs of a single BP step (default and recompute MaxPool backward).

### 8.5 Fusion for blocks 1–4 and the full network (guide 05)

| path | goal | result | status |
|---|---|---|---|
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp7_remaining_blocks/` (main files: results_step5.txt, results_step6.txt, results_step7_gap9.txt) | fused kernels for blocks 1–4 (cluster two-input kernel, Conv4x2 control kernel, pipelined NE16 node) and the per-block engine choice | Siracusa blocks 1–4 2.44× (wall clock, node level); GAP9 best per block 2.57× (NE16 blocks 1–2, cluster 3–4); all exact | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp8_whole_network/` (main files: results/step8_per_block_vs_fixtures.txt, results/step10_round_prediction.txt, results/step11_eco_round_prediction.txt, graphs/) | the whole QZO step as one dual-pass graph (`ZODualPassFusion`) | 4 steps identical to unfused; Siracusa 1.53×, GAP9 cluster 1.42×, GAP9 + NE16 1.78×; n_accum 4 step 8.86 M (QZO) / 9.13 M (QZO+ECO) cycles; 120-epoch round 38.8 s / 40.0 s @ 370 MHz vs 68.4 s / 69.6 s unfused | CURRENT |
| `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp9_memory_optim/` (main files: memcompare/summary.txt, logs/, verify/*.log, graphs/; branch `feat/memory_opt`) | memory of the fused graph: frozen BN + MaxPool + ReLU + Quant as one HWC kernel (`FuseFrozenBNTailPass`), guide 05 §4.8 | fused QZO+ECO training L2 arena 854,452 → 265,612 B, training −13.0 % per step; bit-exact on GAP9, GAP9 + NE16 and Siracusa; CI 6/6 + 13/13; not yet on the board | CURRENT |

