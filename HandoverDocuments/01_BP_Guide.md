# 01 — Back-propagation (first-order) on-device fine-tuning of SpeechNet with Deeploy

**Scope.** This guide documents the first-order (SGD back-propagation) on-device fine-tuning of the
SilentWear SpeechNet model on the Siracusa / GAP9 class of PULP SoCs, as implemented in our forks
`TrainDeeploy` (Deeploy compiler + GVSoC harness) and `Onnx4Deeploy` (PyTorch to ONNX exporter and host
references). It covers the method, the graph the exporter emits, every file on the Deeploy side in
pipeline order, how to run it, what was measured, and what is still open. The zeroth-order (ZO),
quantized-ZO and NE16 work are covered by the sibling guides; they reuse the harness, fixture layout and
frozen-BatchNorm machinery described here.

All paths are relative to `/Users/qiwenwu/ETH/` unless they appear inside a command block, where they are
in-container paths and the container is named. Every path, symbol and flag below was checked against the
tree on 2026-10-12 (branch `feat/ZO_2fwd_fusion` in both repos).

## Status as of 2026-10-12

| | |
|---|---|
| **Works** | Full-model fine-tuning (22 trainable tensors: 5 conv weight+bias, 5 BN gamma+beta, fc weight+bias) on GVSoC Siracusa, 8 cores, with BatchNorm normalised by the frozen pretrained running statistics (recipe "S2"), MaxPool backward via a stored argmax mask, gradient accumulation SUM over 4 windows, SGD lr 3e-4, 40 epochs over 54 stratified windows = 540 optimizer steps. Device weights are dumped, carried into an inference fixture and evaluated on device. |
| **Measured (correct weights)** | Round 1 S01 / vocalized / fold 3: batch-2 balanced accuracy **87.78 %** (158/180), +11.11 pp over the 76.67 % no-FT baseline; the paper's own recipe gives 88.89 %. Evidence: `TrainDeeploy/DeeployTest/experiments/deliverable/exp17_BP_round1_correct_weights/` (`logs/phase6_device_eval_b2.log.gz`, `results/eval_b2_results.json`, `results/reference_S01_fold3.txt`), checkpoint `TrainDeeploy/SpeechNet_Test/pretrained/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt`. The host simulation (in `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/`, not part of these repositories) reproduces the same cell exactly (87.78 %). |
| **Measured (cost)** | One device step (4 fwd+bwd + 1 SGD) is about 140 M cycles, 94 % compute-bound; one fwd+bwd is 32.7 M cycles with the argmax mask (34.1 M with the recompute kernel). Train-network L2 peak 1,511,308 B at a 2 MB budget, 1,482,636 B under a strict 1.5 MB (GAP9) budget. Evidence: `experiments/exp2/profiling_run/`, `experiments/deliverable/{exp1_BP_argmax_off,exp2_BP_argmax_on,exp2a_BP_argmax_naccum1}/logs/`, `experiments/exp3/logs/`, `experiments/exp4/logs/` (section 5 lists the file per number). |
| **Open** | Rounds 2–4 and subjects S02–S04 on the correct weights are not run on device; there is no BP training harness for the GAP9 platform (only the memory fit was proven on the Siracusa runner with GAP9 budgets); the device cycle counter is a `uint32` that overflows on a full round; the per-step device-vs-reference loss check is expected to breach after the MaxPool argmax tie-flip onset, so accuracy is the acceptance criterion. |

---

## 1. Concept

### 1.1 The protocol

SilentWear records a subject in sessions; each session has 5 batches of 180 windows (9 classes x 20).
The fine-tuning protocol is incremental: starting from the model pretrained on sessions 1 and 2
(`leave_one_session_out_fold_3.pt`), round r trains on 30 % of batch r of session 3 (54 windows, 6 per
class, stratified, seed 42) and is evaluated on the whole batch r+1 (180 windows, balanced accuracy).
The trained weights are carried into the next round. On device we have completed round 1 (b1 -> b2).

### 1.2 The optimisation step the device runs

Plain SGD with gradient accumulation by summation. For optimizer step u and accumulation step a the
window index is `mb = u * n_accum + a`, the window is `windows[mb % 54]` (cyclic, never shuffled), and

```
g_acc  = sum_{a=0..n_accum-1}  dL(w; x_mb, y_mb)/dw        (InPlaceAccumulatorV2, reset when a == 0)
w     <- w - lr * g_acc                                     (SGD node, attribute lr)
```

There is no 1/n_accum, no momentum, no weight decay and no schedule. With lr 3e-4 and n_accum 4 the
effective step relative to a mean-reduced batch is 1.2e-3. The whole 2160-pass schedule
(540 steps x 4) runs inside the C harness; the host only prepares the fixture.

### 1.3 BatchNorm with frozen statistics ("S2")

BatchNorm in training mode normalises each window by its own per-channel spatial mean and variance.
With batch size 1 this is a degenerate estimate, and the device never updates the running statistics,
so the inference graph (which uses the pretrained running stats) sees features that were trained under a
different normalisation. On device this cost -17.78 pp on batch 2 (measured in
`experiments/fulltraining_naccum8_regression/`, device logs `full8_e20_device.log` / `full8_e40_device.log`;
stale `artifacts/` weights, not paper-comparable, but the mechanism is the point). That experiment
(full model, live batch-1 BN, n_accum 8, lr 1e-3, 20 epochs = 135 SGD steps) evaluated the same batch-2
windows under four weight/statistics combinations, which isolates the cause:

| trained weights | BN running stats at inference | batch-2 | vs zero-shot 78.33 % |
|---|---|---|---|
| host (ORT) simulation | updated by ORT's training-mode BN | 81.11 % | +2.78 pp (what the host sweep reported) |
| host (ORT) simulation | frozen pretrained | 63.33 % | -15.00 pp |
| device | updated by ORT | 78.33 % | +0.00 pp |
| device | frozen pretrained (what the device really has) | 60.56 % | -17.78 pp |

Device weights paired with updated statistics are harmless (row 3), so the whole loss comes from the
statistics mismatch: the device kernel never updates `running_mean`/`running_var`, while the weights were
trained against per-window statistics. The host sweep had looked positive only because its evaluator used
the ORT-updated statistics, which the device does not produce.

S2 instead normalises with the frozen pretrained `running_mean` / `running_var` during training as well,
while gamma and beta stay trainable:

```
y   = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
dX  = gamma / sqrt(running_var + eps) * dY          (no batch-statistics Jacobian terms)
dgamma = sum dY * x_hat,   dbeta = sum dY
```

Train and inference then compute the same function, so batch 1 is no longer a problem. The folding
alternative (bake the frozen BN into the conv and train the folded conv) was rejected for full-model
training because the fold rescales conv weights by gamma/sigma. Even frozen, BN acts as a per-layer
learning-rate scale: EMG is unnormalised (running variances in the thousands), so
`gamma / sqrt(running_var + eps)` is about 0.02–0.03 and damps the conv gradients by that factor.
Folding removes the damping, and on the host the folded full-model recipe collapsed to 31.5 % at the S2 lr
of 3e-4; it only recovered at lr 3e-6 (about 100x lower) and then reached 83.33 % mean b2–b5 versus
85.65 % for unfolded S2 (host PyTorch, S01 vocalized, mean of the 3 folds, stale `artifacts/` weights). Keeping BN unfolded with a kernel switch
keeps gamma/beta at their natural scale, so the lr validated on the host transfers unchanged. Folding is
still used for the head-only recipe, where the feature extractor is frozen (section 1.5).

### 1.4 MaxPool backward with a stored argmax mask

MaxPool's backward scatters dY to the position that won the forward max. The stock Deeploy kernel
`PULP_MaxPoolGrad2d_fp32_fp32_HWC` recomputes the argmax from the forward input X at backward time,
which keeps the 314 KB block-0 activation alive from forward to backward. Our alternative emits, in the
forward pass, the within-window offset `p*Q + q` of the winner as a separate single-output op
(`MaxPoolArgmax`), and the backward (`MaxPoolGradMask`) reconstructs the absolute input position
locally from the output position and the offset. The offset is tile-position independent, the tie-break
(strict `>`, first maximum) is identical in both forward kernels, and every node stays single-output, so
no tiler-core change was needed. The mask is stored as fp32. The first design used a uint8 mask and
reused the `MaxPoolGrad` op with a dtype-dispatched binding; it hit type errors and was switched (TrainDeeploy
commit `e322b08`, Onnx4Deeploy `e6af342`) to an fp32 offset plus the distinct `MaxPoolGradMask` op. Both
changes were made together, so the exact blocker is not isolated; the likely causes are a uint8 tensor
flowing as an activation through the all-fp32 transpose/tiling pipeline, or the ambiguous dtype-dispatch
binding. Retrying uint8 is open item 5 in section 8.

The mask does not remove the device-vs-reference drift: both kernels take the argmax of the same X, and
the drift is caused by about 1e-6 reduction-order differences in X between the tiled 8-core device and
the untiled host flipping a near-tie. The per-step loss stays bit-exact until the first flip and then
diverges as a cumulative trajectory difference; the device trajectory remains a valid SGD trajectory.
The onset is a discrete jump (one argmax flips once the accumulated weight difference crosses a tie), the
difference then saturates instead of growing (about 0.02–0.06 in loss on the early 90-step runs, `Errors:
52 out of 90` in `experiments/maxpool_numerical_drift/speechnet_maxpool_90step_acc1_val.log`), and the
model keeps learning. A host experiment that injected a controlled per-step relative perturbation into a
PyTorch fp32 run showed two things: shrinking the perturbation delays the onset for both pooling types
(precision helps, it is not a hard wall), and MaxPool diverges about 3–4x earlier than AvgPool at every
perturbation size, because a discrete argmax flips where an average absorbs the change. The only way to
match a reference over arbitrarily many steps is to make the reference use the device's reduction order;
fewer cores or higher-precision accumulators only delay the onset. Larger n_accum lowers the magnitude of
the drift (max diff 0.064 at n_accum 1 vs 0.020 at n_accum 4 on those runs) but not its onset (about step
36 vs 32).

### 1.5 Recipe decision history

| step | recipe | evidence | outcome |
|---|---|---|---|
| 1 | naive full-model FT, live batch-1 BN, n_accum 8 | `experiments/fulltraining_naccum8_regression/` (stale `artifacts/inter_session` weights) | -17.78 pp on device; 100 % attributable to the BN running-stat mismatch |
| 2 | head-only (fc) + fold frozen BN into conv (`_fold_bn_into_conv`) | `experiments/headonly_ondevice_finetune/`, re-run `experiments/headonly_ondevice_ft_fixedwindow/` (`run_train_isft_fw.log`, `eval_b2_zs_summary.log`, `eval_b2_ft_summary.log`; stale `artifacts/inter_session_ft`, not paper-comparable) | bit-exact on device (0/2160), +7.78 pp on b2 (81.67 -> 89.44) on those stale weights; the shipped recipe for a while |
| 3 | frozen-stat BN kernel (`BN_FROZEN_STATS`) enabling full-model FT at batch 1 | `experiments/frozen_bn_kernel_finetune/` (`speechnet_fullfrozen_validate.py`, `fullfrozen_validate.log`, `frozenbn_fair.log`), commit `e87291b` | kernel validated on a 20-step device run: the `[WDUMP]` weights match a host frozen-BN PyTorch run (worst relative deviation 0.000 %); the same run's per-step loss check reads `Errors: 20 out of 20` because the reference was still ORT live-BN (fixed in step 5). Host study, 8 draws x 2 seeds, each recipe at its best lr: full-FT +3.99 +/- 2.95 pp vs head-only +3.40 +/- 0.52 pp, full-FT wins 9/16, i.e. a tie with about 6x the variance |
| 4 | host comparison of 5 recipes, 3 folds, 10 seeds | host PyTorch study in `SilentWear/SilentWear/PyTorch_for_On_Device/` (not part of these repositories; stale `artifacts/`) | mean b2–b5, S01, 3 folds: head-only + BN fold 84.68, streaming AdaBN 85.42, S2 85.65, full on folded BN 83.33 (lr 3e-6), live batch-1 BN 60.74, paper recipe 88.24; over 10 seeds head-only 85.54 +/- 0.81 vs S2 84.76 +/- 0.70 (head-only wins 8/10). S2 chosen for the device chain because it exercises the full backward (the compiler/kernel deliverable) and is what the ZO/QZO guides compare against |
| 5 | S2 on device, reference fixed (Option A: PyTorch frozen-BN reference instead of ORT live-BN) | `experiments/exp1/logs/smoke_gvsoc_loss.log` (18 windows, 2 epochs, n_accum 4, lr 3e-4 = 36 passes, 9 updates) | device per-step loss bit-exact to the corrected reference (`Errors: 0 out of 36`, max diff about 4e-6); the smoke is below the drift onset, so it validates the BN fix only |
| 6 | S2 chain b1 -> b5 on device | `experiments/exp1/ondevice_sim_S01_fold3/` (`results/ondevice_vs_pytorch_S01_fold3.csv`, `logs/round{1..4}_gvsoc_train.log`; stale `artifacts/inter_session_ft`) | mean b2–b5 83.20 vs PyTorch 83.75 on the same draws; superseded by exp17 |
| 7 | argmax mask + transpose dedup; GAP9 L2 fit | `experiments/exp3/logs/`, `experiments/exp4/logs/gap9_l2_1p5M.log`, `experiments/deliverable/exp2_BP_argmax_on/results/memory_alloc_deeployStates.html` | L2 peak 1,793,800 -> 1,511,308 B; fits 1.5 MB at 1,482,636 B (section 6) |
| 8 | round 1 on the correct lab-machine checkpoint | `experiments/deliverable/exp17_BP_round1_correct_weights/` (logs and results) | **87.78 %** (current headline) |

---

## 2. The graph Onnx4Deeploy emits

### 2.1 Model

`Onnx4Deeploy/onnx4deeploy/models/pytorch_models/speechnet/speechnet.py`, class `SpeechNetDeploy`.
Input `(1, 1, 14, 700)` (14 filtered EMG channels, 700 samples = 1.4 s at 500 Hz).

| block | layer | pool |
|---|---|---|
| 0 | `Conv2d(1->8, k=(1,4), pad=(0,2))` -> BN -> ReLU | `MaxPool2d((1,8))` |
| 1 | `Conv2d(8->16, k=(1,16), pad=(0,8))` -> BN -> ReLU | `MaxPool2d((1,4))` |
| 2 | `Conv2d(16->16, k=(1,8), pad=(0,4))` -> BN -> ReLU | `MaxPool2d((1,4))` |
| 3 | `Conv2d(16->32, k=(7,1))` -> BN -> ReLU | `nn.Identity` |
| 4 | `Conv2d(32->32, k=(7,1))` -> BN -> ReLU | `nn.Identity` |
| head | `AdaptiveAvgPool2d(1,1)` -> `reshape(1, 32)` -> `Linear(32, 9)` | |

Two deployment choices: the paper's `(1,1)` pools are emitted as `nn.Identity` so the ONNX graph carries
no degenerate MaxPool/MaxPoolGrad pair (which trips `MaxPoolGradCTileConstraint`), and the head uses a
static `reshape(1, self._fc_in)` so no dynamic `Shape` op is exported. There is no Dropout. Parameter
names in the fixtures are `blocks_{i}_0_{weight,bias}` (conv), `blocks_{i}_1_{weight,bias,running_mean,running_var}` (BN) and `fc_{weight,bias}`.

### 2.2 Data: windowing and the 54-window draw

`Onnx4Deeploy/onnx4deeploy/data/silent_wear_datasource.py`, class `SilentWearDataSource`.

- `_load_windows()` reads `<data_path>/<subject>/<condition>/sess_<session>_batch_<batch>.h5`
  (key `emg`), takes the 14 `*_filt` columns, segments on `Label_int` changes and cuts exactly one
  700-sample window per segment anchored at the segment onset (windows overrunning the recording end are
  dropped). Onset anchoring is the paper's windowing: the window starts at the first sample of the
  labelled segment even when the labelled run is shorter than 700 samples (comments in
  `_load_windows`, lines 146–168). The earlier centre anchoring shifted every window off the paper's
  anchor; it was the bug fixed in Onnx4Deeploy commit `77e269d` (68.33 % -> 81.67 % zero-shot on the
  stale weights). Rest windows are downsampled with `np.random.RandomState(42)` to the smallest
  command-class count, giving 180 windows per batch.
- `load_batches(n_batches, input_shape, num_classes, seed=42)` draws the fine-tuning set with a local
  `np.random.RandomState(seed)`. With `stratified_split=True` (CLI `--stratified`) it takes
  `n_batches // n_classes` = 6 indices per class with `rng.choice`, concatenates them class by class and
  then `rng.shuffle`s the list. The order matters because accumulation sums 4 consecutive windows.
  This draw is the device's draw. A host draw with `np.random.default_rng(42)` (PCG64, returned sorted)
  instead of the exporter's `RandomState(42)` (Mersenne Twister, shuffled; line 263) shares only 16 of
  the 54 windows for S01 fold 3 batch 1, which is exactly chance overlap (9 x 6 x 6/20 = 16.2), and its
  order groups the windows differently into accumulation sums. A host simulation must reproduce the
  exporter's draw and order to match the device.

### 2.3 Export modes and the SpeechNet flags

`Onnx4Deeploy/Onnx4Deeploy.py`: `-model SpeechNet`, `-mode {infer, train, train_single_step, quant, zo-train, q-zo-train}`.
Flags used by this guide (all verified in the argparse block):

| flag | config key | effect |
|---|---|---|
| `--dataset silentwear --data-path P` | `dataset`, `data_path` | use `SilentWearDataSource` |
| `--pretrained-weights CKPT` | `pretrained_weights` (key `model_state_dict`) | strict `load_state_dict` |
| `--subject S01 --session 3 --batch 1 --condition vocalized` | | which `.h5` |
| `--stratified` | `stratified_sampling` | 6 windows per class |
| `--data-size 54` | `data_size` | distinct windows cycled |
| `--n-epochs 40` | | `n_batches = ceil(40*54/n_accum)*n_accum` = 2160 |
| `--n-accum 4` | `n_accum` | accumulation window (also must be passed to the device runner) |
| `--lr 0.0003` | `learning_rate` | overrides the exporter default `0.001` in `SpeechNetExporter.load_config()`; baked into the SGD graph |
| `--training-strategy full` | `training_strategy` | `get_trainable_params`: `full` = all; `last_layer` = names containing `fc`; `custom` = `--custom-trainable-params` |
| `--bn-frozen-stats` | `bn_frozen_stats` | S2 export (section 2.4) |
| `--maxpool-argmax-mask` | `maxpool_argmax_mask` | argmax-mask rewrite (section 2.5) |
| `-o DIR` | | output dir; **must contain `_train`** (section 2.6) |

`-mode infer` with the same data flags produces the evaluation fixture: `network.onnx` with weights as
initializers, `inputs.npz` with keys `input` (N,1,14,700) and `label`, `outputs.npz` with `output`
(ORT logits) for all 180 windows of the requested batch (`SpeechNetExporter._save_silentwear_eval_data`).

### 2.4 Training export (`BaseONNXExporter.export_training`)

`Onnx4Deeploy/onnx4deeploy/core/base_exporter.py`, in order:

1. `create_model()` (`speechnet_exporter.py`): builds `SpeechNetDeploy`, loads the checkpoint, and if
   `fold_bn` or `training_strategy == "last_layer"` calls `_fold_bn_into_conv` (QW): per block,
   `scale = gamma / sqrt(running_var + eps)`, `W *= scale`, `b = (b - running_mean) * scale + beta`,
   BN replaced by `nn.Identity`. Not applied for `full`.
2. BN running stats are snapshotted, `model.train()` is called, and with `bn_frozen_stats` every
   `_BatchNorm` module is switched back to `eval()` (QW). `_export_to_onnx` then exports with
   `torch.onnx.TrainingMode.PRESERVE` so BN exports as inference-mode BatchNormalization while conv and
   linear export in training mode (QW; `TRAINING` otherwise). The snapshotted stats are copied back over
   the initializers (the tracing forward would otherwise have EMA-corrupted them).
3. `run_inference_optimization` on `network_infer.onnx`, then
   `onnxruntime.training.artifacts.generate_artifacts(optimizer=SGD, loss=CrossEntropyLoss, requires_grad=..., frozen_params=...)`.
   BN buffers (`running_mean`, `running_var`, `num_batches_tracked`) go to `frozen_params`.
   Output renamed to `network_train.onnx`.
4. `run_training_optimization` -> `network_train_optim.onnx` -> copied to `network.onnx`
   (Deeploy-ready; ORT `com.microsoft` ops rewritten to the standard/Deeploy ops).
5. `_rewire_maxpoolgrad_recompute(network.onnx)` (QW, section 2.5). Only `network.onnx` is rewired;
   `network_train.onnx` keeps the ORT convention for the host reference.
6. `create_optimizer()` -> `onnx4deeploy/core/optimizer_onnx.py:create_optimizer_onnx`: one `SGD` node
   per `<param>_grad.accumulation.buffer` input of `network.onnx`, inputs `[param, grad_acc]`, output
   `<param>_updated`, attribute `lr`. Written to the sibling directory from `derive_optimizer_dir`
   (`_train` -> `_optimizer`; returns `None` and skips if the name has no `_train`).
7. `create_training_test_data()` (`speechnet_exporter.py`): draws the 54 windows (section 2.2), reads
   the initial weights from `network_infer.onnx`, and produces the reference. With `bn_frozen_stats` it
   calls `_frozen_pytorch_reference` (QW, "Option A"): rebuilds the
   torch model with the fixture's initial weights, `model.eval()` (frozen BN, gamma/beta trainable),
   `torch.optim.SGD(lr)`, `CrossEntropyLoss`, and runs the exact device loop (`mb % data_size`, SUM
   accumulation, one `opt.step()` per `n_accum`). Without the flag the original ORT loop on
   `network_train.onnx` runs (unchanged upstream path). ORT cannot serve as the frozen-BN reference
   because its `BatchNormInternal(training_mode=1)` always uses live batch statistics and
   `BatchNormalizationGrad` always applies the batch-statistics Jacobian; no ORT option changes this.
   Before Option A the reference therefore disagreed on every step (`Errors: 2160/2160` with a correct
   device). "Option B" tried to make ORT compute the frozen reference on the device's own graph by
   replacing each BN with a differentiable frozen affine in plain Add/Mul (`y = (x - mu) * inv_std *
   gamma + beta`, statistics baked as constants) before `generate_artifacts`. On ORT it matched the
   Option A reference to max |diff| 8.3e-7 over 36 steps (`experiments/exp1/optionB_ort_reference/validate_test1.py`),
   but it does not compile for Siracusa: the binary-op tile constraint (`BOPTileConstraint`) forces all
   operands to the same tile shape, so the per-channel `(1,C,1,1)` operand collapses the activation tile
   (assert at `TilingCodegen.py:537`), and the bound `FloatAdd`/`FloatMul` kernels have no per-channel
   broadcast, so even a tiler fix would compute wrong values. It was reverted (Onnx4Deeploy commit
   `c9285df`). A same-graph solution would need a fusion pass mapping the decomposed affine (forward and
   ORT's autodiff backward) onto the existing per-channel kernels `PULP_ChannelNormalize` and
   `PULP_BNGradReduce`/`PULP_BNGradNormalize` in `BatchNorm.c`; this is not done.

### 2.5 `_rewire_maxpoolgrad_recompute` (QW, both branches ours)

ORT emits `MaxPool -> [Y, Indices]` and `MaxPoolGrad(dY, Indices)`. Deeploy has no Indices tensor.

- Default: `MaxPoolGrad.input[1] <- MaxPool.input[0]` (recompute from X), the Indices output is dropped.
  This reuses the shipped kernel; it is only the exporter-to-kernel bridge.
- `--maxpool-argmax-mask`: for each MaxPool a `MaxPoolArgmax` node (same `kernel_shape`, `pads`,
  `strides`, `ceil_mode` forced to 0; input = the MaxPool's input; output `<Y>_argmax_u8`, declared
  FLOAT with the pooled shape) is inserted after it, and the consuming `MaxPoolGrad` becomes
  `MaxPoolGradMask(dY, mask)`. `MaxPool` stays single-output.

### 2.6 The fixture on disk

For `TrainDeeploy/SpeechNet_Test/speechnet_train_BP_round1` (byte-identical to the exp17 fixture
`TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3_ref`,
`SpeechNet_Test/logs/crosscheck.txt`; verified by loading it):

| file | content |
|---|---|
| `network.onnx` | training graph. Ops: `Conv` 5, `BatchNormInternal` 5, `Relu` 5, `MaxPool` 3, `MaxPoolArgmax` 3, `GlobalAveragePool` 1, `Gemm` 3, `Reshape` 2, `SoftmaxCrossEntropyLoss` 1, `SoftmaxCrossEntropyLossGrad` 1, `GlobalAveragePoolGrad` 1, `ReluGrad` 5, `BatchNormalizationGrad` 5, `ConvGrad` 5 (lowered to ConvGradX/W/B by Deeploy), `MaxPoolGradMask` 3, `ReduceSum` 1, `InPlaceAccumulatorV2` 22, `Constant` 10 (the frozen running_mean/var). 47 inputs: `input`, `labels`, 22 weights, 22 `<param>_grad.accumulation.buffer`, `lazy_reset_grad` (last). 23 outputs: `onnx::loss::2` first, then 22 `<param>_grad.accumulation.out`. |
| `inputs.npz` | 134 keys: `arr_0000..arr_0024` positional over the 25 non-grad-buffer inputs (window, label, 22 weights, `lazy_reset_grad`); `mb{1..53}_arr_0000/0001` (the other windows and labels); `meta_data_size=54`, `meta_n_batches=2160`, `meta_n_accum=4`. |
| `outputs.npz` | 33 keys: the 22 updated trainable tensors, the 10 unchanged BN stats, and `loss` (2160 per-pass reference losses). |
| `network_train.onnx`, `network_train_optim.onnx`, `network_infer.onnx`, `eval_model.onnx`, `optimizer_model.onnx`, `checkpoint/` | intermediates; not read by the runner |
| `../speechnet_optimizer_BP_round1/network.onnx` (exp17: `../speechnet_optimizer_fullfrozen_b1_fold3_ref/`) | `SGD` x 22; 44 inputs interleaved `[param, param_grad.accumulation.buffer]`, 22 outputs `<param>_updated` |

The optimizer directory must be the `_train` -> `_optimizer` sibling: the device runner resolves it the
same way (`resolve_optimizer_dir`), and if the name lacks `_train` it falls back to the training
directory itself and tries to tile the whole training graph as the optimizer ("Backtracking exhausted at
SoftmaxCrossEntropyLoss").

### 2.7 Fixture directories under `TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/`

The primary copies of the current fixtures are in `TrainDeeploy/SpeechNet_Test/` (section 4); the
directories below are the originals the experiments ran on.

| directory | role | status |
|---|---|---|
| `speechnet_train_fullfrozen_b1_fold3_ref`, `speechnet_optimizer_fullfrozen_b1_fold3_ref` | exp17 round-1 train fixture, `artifacts_reference` weights | **current** |
| `speechnet_infer_fullfrozen_b0_fold3_ref`, `speechnet_infer_fullfrozen_b1_fold3_ref` | exp17 zero-shot (batch 1) and post-FT (batch 2) eval fixtures | **current** |
| `speechnet_train_fullfrozen_b1_fold3`, `speechnet_optimizer_fullfrozen_b1_fold3`, `speechnet_infer_fullfrozen_b1_fold3` | exp1 chain / exp2 profiling / exp4_BP_round1, stale `artifacts/` weights | superseded for accuracy; still the exp2 profiling fixture |
| `deliverable_bp_train_recompute`, `deliverable_bp_optimizer_recompute` | deliverable exp1 (recompute MaxPool backward), 18 windows, 1 epoch | current for latency/memory (weights stale but cycles and allocation do not depend on weight values) |
| `deliverable_bp_train_argmaxmask`, `deliverable_bp_optimizer_argmaxmask` | deliverable exp2 / exp2a (argmax mask) | current for latency/memory |
| `speechnet_train_argmask_b1`, `speechnet_optimizer_argmask_b1` | exp3 / exp4 memory fixture (4 steps) | current for the memory numbers |
| `speechnet_train`, `speechnet_optimizer` | the CI fixture registered in `DeeployTest/test_siracusa_tiled_config.py` (`L2_SINGLEBUFFER_TRAINING_MODELS["Models/Training/SpeechNet/speechnet_train"] = [128000]`), 4-step MaxPool graph | current for CI; weight provenance not verified |
| `speechnet_train_head_ep10`, `speechnet_train_head_ep40` + optimizers | head-only fixtures from the shipped-recipe era | stale (the `_isft_fw` fixtures of `headonly_ondevice_ft_fixedwindow/` are no longer present) |
| `speechnet_train_maxpool_{20,2s,90,90_acc4,5ep_acc2,fbn,norm}` + optimizers | MaxPool drift studies | stale |
| `speechnet_train_preweights` | the fork-point fixture (commit `e6bfe27`) | historical |
| `deliverable_bp_argmax_{train,optimizer,mask}`, `deliverable_bp_original`, `regr_bp_argmax`, `deliv_speechnet_{train,optimizer}_argmask`, `speechnet_train_avgpool_cmp`, `speechnet_train_b1_ft` | provenance not documented | not verified; treat as scratch |

---

## 3. The Deeploy side, in pipeline order

### 3.1 What the training infrastructure provided before our fork

The upstream mirror `Deeploy/` (devel) has no training support at all: no `deeploytraintest.c`, no
training runner, no `Tests/Models/Training/`, and no `BatchNorm.c`/`ConvGrad.c` in
`TargetLibraries/PULPOpen/src/`. The training extension is the history of the `TrainDeeploy` fork before
our fork point `e6bfe27` (2026-06-08), documented in `TrainDeeploy/README_TRAINING.md`. It provides:

- the two-network compilation: `TrainingNetwork` (`InitTrainingNetwork`/`RunTrainingNetwork`: forward,
  loss, backward, `InPlaceAccumulatorV2`) and `OptimizerNetwork` (`InitOptimizerNetwork`/`RunOptimizerNetwork`: the SGD graph);
- the input layout contract: `[0 .. TRAINING_NUM_DATA_INPUTS-1]` data, then weights, then grad buffers
  from `TRAINING_GRAD_BUF_START_IDX`, then `lazy_reset_grad` (uint8) last; optimizer inputs interleaved
  `[weight_i, grad_acc_i]`, outputs `weight_i_updated`, copied back unless the buffers already alias;
- the fixture layout `Tests/Models/Training/<Model>/<name>_train/{network.onnx, inputs.npz (arr_NNNN, mb*_arr_*), outputs.npz (loss)}` plus `<name>_optimizer/network.onnx`;
- `InPlaceAccumulatorV2` SUM semantics (`accum = grad` when `lazy_reset_grad`, else `accum += grad`;
  `Deeploy/Targets/PULPOpen/Templates/FloatInPlaceAccumulatorV2Template.py`) and the in-place SGD
  (`Deeploy/Targets/PULPOpen/Templates/SGDTemplate.py`);
- the fp32 training kernels in `TargetLibraries/PULPOpen/src/` (`Convolution_fp32.c`, `ConvGrad.c`,
  `BatchNorm.c`, `MaxPool.c`, ...), the pulp-trainlib submodule used by `ConvGradW`, and the
  `TrainingSBTiler` single-buffer tiler with forward activations kept alive to the backward.

Everything we added or changed is tagged `QW` / `-- QW` in the source. The BP-relevant set is listed
per stage below; `git diff e6bfe27~1 HEAD --name-status` in `TrainDeeploy` lists the full change set.

### 3.2 Parsers and layers

`Deeploy/Targets/PULPOpen/Platform.py`, dict `PULPMapping`. The training ops SpeechNet uses:

| ONNX op | mapper / parser (inherited unless marked) |
|---|---|
| `Conv` | `ConvLayer([FPConv2DMapper, FPDWConv2DMapper])` |
| `ConvGradX` / `ConvGradW` / `ConvGradB` | `ConvGradXLayer([PWConvGradX2DMapper, DwConvGradxMapper, ConvGradXMapper])`, `ConvGradWLayer([...])`, `ConvGradBLayer([ConvGradBMapper])` |
| `BatchNormInternal` / `BatchNormalizationGrad` | `BatchNormInternalMapper = NodeMapper(BatchNormInternalParser(), PULPBatchNormInternalTilingReadyBindings)` / `BatchNormalizationGradMapper` |
| `MaxPool` / `MaxPoolGrad` | `MaxPool2DMapper` / `MaxPoolGrad2DMapper` |
| `MaxPoolArgmax` | **QW** `MaxPoolArgmaxMapper = NodeMapper(MaxPool2DParser(), PULPMaxPoolArgmaxTilingReadyBindings)`, layer `MaxPoolLayer` |
| `MaxPoolGradMask` | **QW** `MaxPoolGradMaskMapper = NodeMapper(MaxPoolGradParser(), PULPMaxPoolGradMaskTilingReadyBindings)`, layer `MaxPoolGradLayer` |
| `Relu` / `ReluGrad`, `GlobalAveragePool` / `GlobalAveragePoolGrad`, `Gemm`, `Reshape`, `ReduceSum`, `Transpose` | inherited |
| `SoftmaxCrossEntropyLoss` / `...Grad`, `SGD`, `InPlaceAccumulatorV2` | inherited |

The parsers for our two ops are reused (`MaxPool2DParser`, `MaxPoolGradParser` in
`Deeploy/Targets/Generic/Parsers.py`); no new parser class was written.

### 3.3 Type checkers, bindings, templates

- `Deeploy/Targets/PULPOpen/TypeCheckers.py`: `PULPMaxPoolArgmaxChecker` (QW; fp32 in, uint8 out) is
  defined but unused after the fp32-mask switch; the fp32 path reuses the MaxPool checker.
- `Deeploy/Targets/PULPOpen/Bindings.py`: `PULPMaxPoolArgmaxBindings` (QW, fp32 -> fp32 mask, template
  `argmaxTemplate`), `PULPMaxPoolGradMaskBindings` (QW, template `referenceGradMaskTemplate`);
  `PULPBatchNormInternalBindings` (5 fp32 inputs/outputs; inherited, the frozen behaviour is a runtime
  switch in the kernel, not a binding change). Conv bindings: `PULPFloatConv2DBindings` ->
  `FloatConvTemplate.reference2DIm2ColTemplate` (kernel `PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC`),
  `PULPFloatConvGradW2DBindings` -> `referenceConvGradW2DIm2ColTemplate` (kernel
  `PULP_ConvGradW2d_fp32_fp32_fp32_CHW_Im2Col`, which calls pulp-trainlib
  `pulp_conv2d_fp32_bw_param_grads_cl` with `USE_IM2COL = 1`), `PULPFloatConvGradX2DBindings` ->
  `referenceConvGradX2DTemplate` (kernel `PULP_ConvGradX2d_fp32_fp32_fp32_CHW_scatter_tiled`).
- `Deeploy/Targets/PULPOpen/Templates/FloatMaxPoolTemplate.py`: `referenceTemplate` (forward; QW fix
  commit `496a7f4` passes spatial dims as `(H,W)=(y,x)`, the base template passed `(x,y)` which is wrong
  for the 14x700 input), `referenceGradTemplate` (recompute; still `(x,y)`, see the `QW-TODO` in the
  file), `argmaxTemplate` (QW, calls `PULP_MaxPoolArgmax2d_fp..._HWC`), `referenceGradMaskTemplate`
  (QW, calls `PULP_MaxPoolGradMask2d_fp..._HWC`; input 1 is the pooled-shape mask so its pointer advances
  by the pooled size).

### 3.4 Tile constraints

`Deeploy/Targets/PULPOpen/Tiler.py`: `PULPMaxPoolArgmaxTilingReadyBindings` (QW) reuses the single-output
MaxPool channel-tiling constraint; `PULPMaxPoolGradMaskTilingReadyBindings` (QW) reuses the MaxPoolGrad
channel-tiling constraint (reads the actual shape of input 1). `PULPBatchNormInternalTilingReadyBindings`
uses `BatchNormInternalTileConstraint` (inherited). The tiler is an OR-Tools CP model
(`Deeploy/TilingExtension/TilerModel.py`) that maximises L1 fill; `--searchStrategy random-max` takes the
largest feasible tile and randomises packing order. Two structural asserts explain our design:
`Deeploy/TilingExtension/TileConstraint.py:134` (one output per tileable node; why `MaxPoolArgmax` is a
separate op) and `Deeploy/TilingExtension/TilingCodegen.py:537` (`offset should be zero when dims ==
reference`; the failure we hit before the layout passes existed).

### 3.5 Platform mapping and lowering passes

- `Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/LoweringOptimizationPasses.py`:
  `_NCHWtoNHWC_fun` op list extended with `"MaxPoolArgmax", "MaxPoolGradMask"` (QW); the second-input
  transpose special case for `MaxPoolGrad` extended to `MaxPoolGradMask` (QW); new passes
  `NCHWtoNHWCMaxPoolArgmaxPass`, `NCHWtoNHWCMaxPoolGradMaskPass` registered in the composite
  `PULPNCHWtoNHWCPass` (QW); and `MergeSiblingTransposesPass` (QW) which merges `Transpose` nodes with
  identical (input, perm) into one shared fan-out transpose.
- `Deeploy/Targets/PULPOpen/Deployer.py`: `MergeSiblingTransposesPass()` wired as the last lowering pass
  (line 64, QW), after the final `TransposeSplitPass`, so the merge is not re-split. This pass is what
  turned the argmax mask from a +1.8 % regression (three consumers each with a private 314 KB transpose
  of the block-0 activation) into the -15.7 % win; it is unconditional and also lowers the recompute
  path from 1,793,800 to 1,735,180 B.
- Layout fact: the forward `Conv` path is NHWC (im2col kernels `_HWC`); the backward
  `ConvGradX/W/B` stay NCHW (pulp-trainlib `_CHW`). The reason is visible in three places:
  `_NCHWtoNHWC_fun` (`LoweringOptimizationPasses.py:247`) and the passes built on it handle `Conv`, the
  pooling ops (MaxPool/MaxPoolGrad/AveragePool and our two ops) and a few elementwise ops, but no
  ConvGrad op, so the ConvGrad nodes are never transposed;
  the forward `ConvTileConstraint` reads C as the last dimension; and the gradient kernels set
  `conv_args.HWC = 0` (`TargetLibraries/PULPOpen/src/ConvGrad.c:74`), trainlib's native CHW layout.
  Consequences for tiling: the forward conv keeps whole input channels (required by im2col,
  `ConvTileConstraint.py:370`), tiles H/W with a `kernel-1` halo and tiles output channels, so each tile
  is a set of contiguous C-length lines and moves as one clean 2-D DMA. `ConvGradW` keeps N/H/W whole
  (they are the reduction dimensions of dW) and tiles output channels (`ConvGradConstraint.py:1579`);
  `ConvGradX` tiles the input-channel slice. Elementwise gradient ops inherit `BOPTileConstraint` (all
  dimensions equal). A new op that is left NCHW while the channel tiler assumes C last gets tiled on the
  wrong (W) axis, which is exactly the `TilingCodegen.py:537` assert of pitfall 17.

### 3.6 Kernels (C)

- `TargetLibraries/PULPOpen/src/BatchNorm.c` (QW parts): global `uint32_t g_bn_frozen_stats = 0u` and
  `g_bn_debug`. In `PULP_BatchNormInternal_fp32` the per-channel branch
  `if (g_bn_frozen_stats) { mean = running_mean[c]; inv_std = 1/sqrtf(running_var[c] + eps); }` replaces
  the batch-statistics computation; `saved_mean`/`saved_inv_std` are still written for the backward. In
  `PULP_BatchNormGrad_fp32` the frozen branch computes `dX = gamma * inv_std * dY` only; `dgamma`/`dbeta`
  are unchanged. Default 0 keeps every existing test unchanged. Validated bit-exact against the host in
  `experiments/frozen_bn_kernel_finetune/speechnet_fullfrozen_validate.py`.
- `TargetLibraries/PULPOpen/src/MaxPool.c`, `inc/kernel/MaxPool.h` (QW parts):
  `PULP_MaxPoolArgmax2d_fp32_fp32_HWC` (same window scan and strict-`>` tie-break as `PULP_MaxPool2d`,
  writes `(float32_t)(p*Q + q)` per pooled element) and `PULP_MaxPoolGradMask2d_fp32_fp32_HWC`
  (zeroes its channel slice of dX, then for each pooled element `off = (uint32_t)(mask + 0.5f)`,
  `p = off / Q`, `q = off % Q`, scatters dY to `window_origin + (p, q)`). Also an argmax-signature
  instrumentation in `PULP_MaxPoolGrad2d` (`g_maxpool_argmax_sig`, `_sig2`, `_en`), dormant unless the
  harness is built with `DUMP_ARGMAX`.
- `TargetLibraries/PULPOpen/third_party/pulp-trainlib` is pinned at `35ba1f6` (`git submodule status`
  confirms `35ba1f62afd460ffd24961b05fbe7fba80df3bce`). A later bump to `45740db` (commit `905859b`)
  deleted the tiled depthwise-conv backward kernel and broke the link of the training-kernel, DSCNN and
  MobileNet tests; the pin was restored in commit `95c01a5` ("Submodule pulp-trainlib points to 35ba1f6.
  Trial fix for CI tests"). Do not bump it.
- `TrainDeeploy/cmake/pulp/pulp.cmake` lines 12 and 16 compile everything with `-ffast-math`, and this is
  the default build for BP. It only matters for the last-ulp agreement with the host; the strict mode, a
  debugging option only,
  (`-D DEEPLOY_STRICT_FP32=ON`, `-D DEEPLOY_STRICT_FP32_FILES="BatchNorm.c;Gemm.c"`,
  `TargetLibraries/PULPOpen/CMakeLists.txt` lines 42–50 and `DeeployTest/CMakeLists.txt` lines 14–18, QW)
  was introduced for the QZO work and has not been used for a BP round.

### 3.7 Test harness

`DeeployTest/Platforms/Siracusa/src/deeploytraintest.c` (inherited loop; QW additions marked):

- `main`: prints `N_TRAIN_STEPS=... N_ACCUM_STEPS=... DATA_INPUTS=...`; `InitTrainingNetwork` on the
  cluster; zeroes the `TRAINING_NUM_GRAD_INPUTS` grad buffers (L2 `memset`, or `ram_write` zero pages
  for L3); `InitOptimizerNetwork`; copies `testInitWeights[wi]` into the weight inputs via
  `l3_aware_copy` (handles L2/L3 source and destination with `ram_read`/`ram_write`; `IS_L2(ptr)` is
  `>= 0x10000000`).
- `#ifdef BN_FROZEN_STATS` (QW): `extern uint32_t g_bn_frozen_stats; g_bn_frozen_stats = 1u;` and prints
  `[BN_FROZEN_STATS] training BN uses frozen running statistics (train==inference)`.
- Loop: for each `update_step`, `accum_step`: set `lazy_reset_grad = (accum_step == 0)`, copy
  `testDataVector[mb % TRAINING_DATA_SIZE]` into the data inputs, run `RunTrainingNetwork` on the
  cluster between `ResetTimer/StartTimer/StopTimer`, `g_train_cycles_acc += getCycles()`, store the loss
  (`DeeployNetwork_outputs[0]`); after the inner loop `run_optimizer_step()` (copies weight+grad pointers
  into `DeeployOptNetwork_inputs[2i]`, `[2i+1]` unless aliased, runs `RunOptimizerNetwork`, copies
  `weight_updated` back unless in place).
- `#ifdef DUMP_WEIGHTS` (QW): `dump_weights(step)` after the last optimizer step prints one line per
  trainable tensor `[WDUMP s=<step> wi=<i> n=<#floats>] <hex32> <hex32> ...` (raw bits, FPU-free, so the
  fabric controller can print it). `DUMP_STEP_LO/HI` windows exist in the MeZO harness only.
- `#ifdef DUMP_ARGMAX` (QW): enables the MaxPool argmax signature and prints `[AMSIG <mb>] <sig> <sig2>`
  per pass (drift evidence).
- Verification: `CompareLossesOnCluster` compares `stored_losses[i]` with `testLossRef[i]` under
  `TRAINING_TOLERANCE_ABS` (default 1e-3), printing `[loss i] computed=... ref=... diff=... TOL=...` and
  then `Errors: X out of Y`; the runner scrapes this line. A `BENCH train_cycles=... opt_cycles=...
  weight_sram=...` line follows. `g_train_cycles_acc` is `static unsigned int` (32-bit), so it wraps on
  a 540-step round (about 75.8 G cycles); only single-step runs give a quotable number.
- Data placement: `testinputs.h` (from `DeeployTest/testUtils/codeGenerateTraining.py`) puts every
  window as `testData_mb<k>_buf0[]` in the 4 MB `.weightmem_sram` section and defines `N_TRAIN_STEPS`,
  `N_ACCUM_STEPS`, `TRAINING_NUM_WEIGHT_INPUTS` (QW: emitted whenever weight inputs exist), etc.
  The Siracusa linker script (`install/pulp-sdk/rtos/pulpos/pulp/kernel/chips/siracusa/link.ld` in the
  container) gives `WEIGHTMEM_SRAM` 4 MB (`LENGTH = 0x003fffe0`), separate from the 2 MB L2 activation
  arena; it holds the weights (about 60 KB) and the window pool. Only one window (38.3 KB) and its tiled
  activations are in L2/L1 per pass, so the L1/L2 peak does not grow with the pool. 54 windows (2.02 MB)
  fit; the on-chip cap is about (4 MB - weights) / 38.3 KB = about 100 windows (about 55 % of a batch).
  70 % (126 windows, 4.71 MB) would need the pool in L3, which `l3_aware_copy` already supports at a
  per-window DMA cost. The 30 % fraction is therefore a protocol choice, not a memory limit.

CMake: `DeeployTest/Platforms/Siracusa/CMakeLists.txt` options `TRAINING`, `MEZO_TRAINING` (QW),
`DUALPASS_BENCH` (QW); in the `TRAINING` branch the pass-throughs `DUMP_WEIGHTS`, `DUMP_ARGMAX`,
`BN_FROZEN_STATS` (QW) become compile definitions. `DeeployTest/CMakeLists.txt:9` builds the training
and optimizer object libraries under `if(TRAINING OR MEZO_TRAINING)` (QW). CMake caches these in
`TEST_SIRACUSA/build_master`, hence the clean rule in section 4.

### 3.8 Runner scripts

- `DeeployTest/deeployTrainingRunner_tiled_siracusa.py` -> `testUtils/deeployTrainingRunner.py:main(tiling_enabled=True)`
  -> `testUtils/core/execution.py:run_complete_test` (generate -> `configure_cmake` -> `build_binary` ->
  `run_simulation`). `deeployTrainingRunner_siracusa.py` is the untiled variant.
  Flags (verified in `deeployTrainingRunner.py` and `testUtils/deeployRunner.py`): `-t DIR`,
  `--cores` (8), `--n-steps`, `--n-accum`, `--num-data-inputs` (all auto-detected from `inputs.npz`
  meta if omitted), `--optimizer-dir`, `--tolerance`, `--l1`, `--l2`, `--defaultMemLevel {L2,L3}`,
  `--memAllocStrategy {MiniMalloc,TetrisRandom,TetrisCo-Opt}`, `--searchStrategy {random-max,max,min}`,
  `--doublebuffer`, `--profileTiling`, `--profileNodes a,b`, `--plotMemAlloc`, `--promoteToL2*`,
  `--skipgen`, `--skipsim`, `-D NAME=VALUE ...` (forwarded to CMake), `-v/-vv`.
- Code generation: `testUtils/trainingUtils.py:run_training_codegen` runs `testMVPTraining.py -t <train>`
  (training graph) and `testMVPOptimizer.py -t <opt> --training-dir <train>` (SGD graph sharing the
  training network's weight and grad buffers). `testMVPTraining.py:generateTiledTrainingNetwork` loads
  `network.onnx`, strips undefined optional outputs, types every input (QW `_ONNX_ELEM_TO_PTR`: the
  ONNX-declared element type wins), builds `MemoryHierarchy([L3 64 MB, L2 --l2, L1 --l1])`, wraps the
  `PULPDeployer` in `MemoryDeployerWrapper` and `TilerDeployerWrapper(..., TrainingSBTiler)`, sets
  `memoryAllocStrategy`/`searchStrategy`, and calls `deployer.prepare()`. `add_training_cmake_flags`
  emits `-DTRAINING=ON -DN_TRAIN_STEPS -DN_ACCUM_STEPS -DTRAINING_NUM_DATA_INPUTS`.
  `_mockScheduler` wraps `deferPerturbNodes` (QW; a no-op for BP graphs).
- QW fix in `testUtils/deeployRunner.py`: `n_cores` from `--cores` is now applied to tiled runs too (it
  used to stay 1 silently).
- Accuracy evaluation: `DeeployTest/experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py --infer-dir DIR [--cores 8]`
  copies the infer fixture's `network.onnx` into `Tests/Models/speechnet_infer_eval_tmp`, writes one
  sample at a time into `inputs.npz`/`outputs.npz`, runs `deeployRunner_siracusa.py` (untiled inference
  harness) per window, parses `Logit[i]:` lines, counts `sim_errors`, and prints `Balanced accuracy :
  0.xxxx`. 180 GVSoC runs take roughly 30–40 minutes on the Mac host.
- Carry checkpoint: `DeeployTest/experiments/exp1/ondevice_sim_S01_fold3/extract_device_weights.py --gvsoc-log LOG --base-ckpt CKPT --out-carry OUT.pt`
  parses the last `[WDUMP]` block (22 tensors), value-matches them to names in `outputs.npz`, and writes
  them over the checkpoint's trainable tensors; BN running stats stay frozen.
- GAP9: there is no BP training harness for GAP9 (`DeeployTest/Platforms/GAP9/src/` holds `deeploytest.c`,
  `deeploymezotest.c`, `CycleCounter.c`; the runners are `deeployRunner_tiled_gap9.py`,
  `deeployMezoRunner_tiled_GAP9*.py`). The GAP9 statement for BP is the memory fit of section 6 only.

---

## 4. How to run

Containers: `agitated_hugle` mounts `ETH` at `/app` (so `/app/Onnx4Deeploy`, `/app/TrainDeeploy`,
`/app/SilentWear`); `traindeeploy` mounts `ETH` at `/app/ETH` (so `/app/ETH/TrainDeeploy`). Writing a
fixture to `/app/ETH/...` from `agitated_hugle` silently lands in that container's overlay and is invisible
everywhere else. `/tmp` is per container.

**Primary fixtures.** Every BP fixture is in `TrainDeeploy/SpeechNet_Test/` (see its `README.md`),
generated by `SpeechNet_Test/scripts/export_fixtures.sh` (targets `bp_round1`, `bp_single`, `bp_infer`)
and checked file by file against the fixture the original experiment ran on
(`SpeechNet_Test/scripts/crosscheck.py`, output `SpeechNet_Test/logs/crosscheck.txt`):

| fixture in `SpeechNet_Test/` | role | equivalent fixture used by exp17 (`DeeployTest/Tests/Models/Training/SpeechNet/`) | check |
|---|---|---|---|
| `speechnet_train_BP_round1` | round-1 training graph, 54 windows, 2160 reference losses | `speechnet_train_fullfrozen_b1_fold3_ref` | every file byte-identical |
| `speechnet_optimizer_BP_round1` | SGD graph (found by the `_train` -> `_optimizer` rename) | `speechnet_optimizer_fullfrozen_b1_fold3_ref` | byte-identical |
| `speechnet_infer_BP_round1` | batch-2 evaluation (180 windows) with exp17's device-trained weights | `speechnet_infer_fullfrozen_b1_fold3_ref` | byte-identical |
| `speechnet_train_BP_single_step`, `speechnet_optimizer_BP_single_step` | one fwd+bwd + one SGD step (n_accum 1) | first step of `..._b1_fold3_ref` | same graph, initial weights, first window and reference loss (0.01145147) |

There is no zero-shot (batch-1) fixture in `SpeechNet_Test/`; the gate of 4.1 uses
`speechnet_infer_fullfrozen_b0_fold3_ref`. The original exp17 flow, which writes to the old fixture names,
is scripted in `TrainDeeploy/DeeployTest/experiments/deliverable/exp17_BP_round1_correct_weights/scripts/run_round1.sh`
(`bash run_round1.sh [1-6|all]`, run from the host); blocks 4.1–4.5 below are its six phases, written
for the `SpeechNet_Test/` fixtures. In `traindeeploy` all runner paths are relative to
`/app/ETH/TrainDeeploy/DeeployTest`, so the fixtures are `../SpeechNet_Test/<name>`. A fixture outside
`DeeployTest/` makes the runner print an "outside base directory" warning and generate into
`TEST_SIRACUSA/<fixture name>`; this is expected.

### 4.1 Zero-shot gate (export in `agitated_hugle`, evaluate in `traindeeploy`)

```bash
# agitated_hugle
cd /app/Onnx4Deeploy
CKPT=/app/TrainDeeploy/SpeechNet_Test/pretrained/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
DATA=/app/SilentWear/SilentWear_data/data_raw_and_filt
python3 Onnx4Deeploy.py -model SpeechNet -mode infer \
  -o /app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_infer_fullfrozen_b0_fold3_ref \
  --dataset silentwear --data-path $DATA --pretrained-weights $CKPT \
  --subject S01 --session 3 --batch 1 --condition vocalized
```

```bash
# traindeeploy
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
  --infer-dir Tests/Models/Training/SpeechNet/speechnet_infer_fullfrozen_b0_fold3_ref
```

Expected: `Balanced accuracy : 0.7222` with `sim_errors=0` on every window. This equals the no-FT
reference for batch 1 in `experiments/deliverable/exp17_BP_round1_correct_weights/results/reference_S01_fold3.txt`
(the paper pipeline's own zero-shot number for this checkpoint), so it proves that the checkpoint loads
correctly and device inference is faithful (exp17 log: `logs/phase1_device_eval_zeroshot_b1.log.gz`).
It takes about 30–40 minutes and is the only phase that can fail for a setup reason rather than a
scientific one, so run it first in every new round.

### 4.2 Export the training fixture (`agitated_hugle`)

```bash
cd /app/Onnx4Deeploy
python3 Onnx4Deeploy.py -model SpeechNet -mode train \
  -o /app/TrainDeeploy/SpeechNet_Test/speechnet_train_BP_round1 \
  --dataset silentwear --data-path $DATA --pretrained-weights $CKPT \
  --subject S01 --session 3 --batch 1 --condition vocalized \
  --data-size 54 --n-epochs 40 --n-accum 4 --lr 0.0003 \
  --training-strategy full --bn-frozen-stats --stratified --maxpool-argmax-mask
```

(exp17 wrote the same fixture to `Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3_ref`.)
Expected lines (exp17 log `logs/phase2_export_train_b1.log.gz`): `BN_FROZEN_STATS: 5 BatchNorm layer(s) in eval mode`,
`Stratified split: 6 samples/class x 9 classes = 54 total`,
`Training sim: n_batches=2160  n_accum=4  n_steps=540  lr=0.0003`, `QW: inserted 3 MaxPoolArgmax node(s); rewired 3 MaxPoolGrad`,
`Building optimizer ONNX (lr=0.0003)`, `[exp1/Option A] frozen-BN PyTorch reference — 2160 losses`, `Export Complete!`.
The single-step fixture is the same command with `-o .../speechnet_train_BP_single_step --n-steps 1
--n-accum 1` in place of `--n-epochs 40 --n-accum 4` (`bp_single` in `export_fixtures.sh`); the exporter
still draws the full 54-window pool and keeps only the window the step consumes.
Drop `--maxpool-argmax-mask` for the recompute variant; use `--training-strategy last_layer` (BN is then
folded automatically) with `--lr 0.01` for the head-only recipe.

### 4.3 Train on device (`traindeeploy`; 540 steps is about 3 h 40 min on the Mac host)

```bash
docker exec -d traindeeploy bash -lc "
  pgrep -f '[g]vsoc_launcher' | xargs -r kill -9
  cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
  nohup python3 -u deeployTrainingRunner_tiled_siracusa.py \
    -t ../SpeechNet_Test/speechnet_train_BP_round1 \
    --n-steps 540 --n-accum 4 --cores 8 \
    --l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
    -D DUMP_WEIGHTS=ON BN_FROZEN_STATS=ON \
    > /app/ETH/TrainDeeploy/SpeechNet_Test/logs/bp_round1_train.log 2>&1 &"
```

(exp17 ran `-t Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3_ref` and logged to
`experiments/deliverable/exp17_BP_round1_correct_weights/logs/phase3_gvsoc_train_round1.log`.)
Expected: `[BN_FROZEN_STATS] ...`, `update 540/540`, 2160 `[loss i]` lines, `Errors: 1 out of 2160`
(exp17; any small count after the drift onset is normal, see section 7), 22 `[WDUMP s=539 ...]` lines,
`BENCH train_cycles=...` (wrapped, do not quote). The runner prints a `FAILED` banner and exits non-zero
whenever `Errors` is non-zero; for a full round that is expected. Device stdout is flushed only when
`main()` returns, so the log is empty until the end. For a memory-only check add `--skipsim --plotMemAlloc`
and read `memory_alloc.html` in the build's `deeployStates/` directory (and in the optimizer's). A short
smoke test is `--n-steps 4 --n-accum 4` (static allocation makes the footprint independent of
`--n-steps`).

### 4.4 Extract the device weights (`traindeeploy`; write to the bind mount, not `/tmp`)

```bash
cd /app/ETH/TrainDeeploy/DeeployTest/experiments/exp1/ondevice_sim_S01_fold3
python3 extract_device_weights.py \
  --gvsoc-log /app/ETH/TrainDeeploy/SpeechNet_Test/logs/bp_round1_train.log \
  --base-ckpt /app/ETH/TrainDeeploy/SpeechNet_Test/pretrained/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt \
  --out-carry /app/ETH/TrainDeeploy/SpeechNet_Test/carry_BP_round1.pt
```

Expected: `parsed [WDUMP] last step s=539: 22 tensors (expected 22)` (exp17: `logs/phase4_extract_weights.log`;
its carry checkpoint is `results/carry_fullfrozen_b1_fold3_ref.pt`). A correct carry changes all 22
trainable tensors and none of the 15 BN statistic tensors (`running_*`, `num_batches_tracked`); in exp17
the largest movement was 4.738e-2 in `blocks_0_0_weight`. For round r >= 2 pass the previous carry as
`--base-ckpt`.

### 4.5 Export the batch-2 inference fixture (`agitated_hugle`) and evaluate (`traindeeploy`)

`SpeechNet_Test/speechnet_infer_BP_round1` already carries exp17's device weights; regenerate it only
after a new device round (`bp_infer` in `export_fixtures.sh`, with the new carry checkpoint):

```bash
# agitated_hugle (same as 4.1 with the carry checkpoint and --batch 2)
cd /app/Onnx4Deeploy
python3 Onnx4Deeploy.py -model SpeechNet -mode infer \
  -o /app/TrainDeeploy/SpeechNet_Test/speechnet_infer_BP_round1 \
  --dataset silentwear --data-path $DATA \
  --pretrained-weights /app/TrainDeeploy/DeeployTest/experiments/deliverable/exp17_BP_round1_correct_weights/results/carry_fullfrozen_b1_fold3_ref.pt \
  --subject S01 --session 3 --batch 2 --condition vocalized
```

```bash
# traindeeploy
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
  --infer-dir ../SpeechNet_Test/speechnet_infer_BP_round1
```

(Equivalent fixture used by exp17: `Tests/Models/Training/SpeechNet/speechnet_infer_fullfrozen_b1_fold3_ref`.)
Expected: `Balanced accuracy : 0.8778  (87.78%)`, 158/180, `sim_errors=0` throughout (exp17:
`logs/phase6_device_eval_b2.log.gz`, `results/eval_b2_results.json`). The exporter's own ORT evaluation
of the same fixture prints `Balanced accuracy = 0.8778` at export time (`logs/phase5_export_infer_b2.log`).

### 4.6 Single-step latency and memory (`traindeeploy`)

```bash
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 deeployTrainingRunner_tiled_siracusa.py \
  -t ../SpeechNet_Test/speechnet_train_BP_single_step \
  --n-steps 1 --n-accum 1 --num-data-inputs 2 --cores 8 \
  --l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D BN_FROZEN_STATS=ON
```

`--num-data-inputs 2` is required: the runner infers the per-window input count from the second window's
`mb1_*` entries, which a one-window fixture does not have. Expected (`SpeechNet_Test/logs/smoke/siracusa_bp_single.log`):
`[loss 0] computed=0.011451  ref=0.011451  diff=0.000000`, `Errors: 0 out of 1`,
`BENCH train_cycles=32693752` (exp2a, same graph on the stale weights: 32,708,926). The latency and memory
studies of section 5 used the earlier fixtures `deliverable_bp_train_argmaxmask` /
`deliverable_bp_train_recompute` (2 MB L2 budget); add `--plotMemAlloc` for the allocation map and
`--profileTiling --profileNodes <ops>` for per-op cycles.

`--profileTiling` on the whole 78-node graph builds but crashes GVSoC right after `Initializing
TrainingNetwork...` with `Invalid access ... offset: 0x57575757` (`0x57` is the uninitialised-stack fill
byte): each instrumented node puts about six cycle-measurement arrays on the core stack, which overruns the
L1 stack (failed run: `experiments/exp2/profiling_run/train_profile.log`). Profile in disjoint
`--profileNodes` passes and merge; per-op cycles are deterministic across accumulation steps, so merging
is exact. The five passes of exp2 (each after `rm -rf TEST_SIRACUSA`, `--n-steps 1 --n-accum 4`):

```bash
--profileNodes Conv --plotMemAlloc                                    # also matches ConvGrad*  -> probe_conv.log
--profileNodes BatchNorm,Relu,MaxPool,GlobalAveragePool,AveragePool   # -> probe_bn_relu_pool.log
--profileNodes Gemm,Softmax,ReduceSum,Reshape                         # -> probe_gemm_loss.log
--profileNodes Accumulat                                              # the 22 accumulators -> probe_opt.log
--profileNodes sgd                                                    # matching is case-sensitive -> probe_sgd.log
```

Each node prints a cluster block (`Pre-Kernel` = L2->L1 DMA, `Kernel` = compute, `Post-Kernel` = L1->L2
DMA) and an L3 block whose `Kernel` is the whole cluster dispatch; count only the cluster block or the
compute is counted twice. `experiments/exp2/profiling_run/analyze_and_perfetto.py` merges the probe logs
(output `cycle_summary.txt`, `trace.json` for Perfetto); every pass's `BENCH train_cycles` line is the
aggregate the merged sum must reconcile to (within 0.2 %). `experiments/deliverable/exp1_BP_argmax_off/results/parse_profiletiling.py`
parses the per-tile lines of a single-pass log. Keep `DUMP_WEIGHTS` off while profiling.

### 4.7 Host simulation (Mac, CPU, `silent_wear` conda env)

The host simulation code lives in `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/` (not part of
these repositories). It needs no device code; it uses the `silent_wear` conda environment for the numpy
version that reads the HDF5 windows.

```bash
cd /Users/qiwenwu/ETH/SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/exp1_BP_simulation
PY=/opt/anaconda3/envs/silent_wear/bin/python
$PY validate_zero_shot.py        # all 20 zero-shot cells (S01-S04 x b1-b5) must match the paper pipeline's no-FT numbers to +0.00 pp
$PY run_experiment.py            # fold 3, S01-S04, vocalized; results/fold3_vocalized.{csv,md}
$PY run_experiment.py --seed 52 --out-prefix results/fold3_seed52
```

`common/device_faithful.py:DeviceRecipe` defaults are the device recipe (lr 3e-4, n_accum 4 SUM,
40 epochs, 6 per class, seed 42, `order="cyclic"`, `draw="device"` = the exporter's own `RandomState`
draw reproduced byte-for-byte, dropout 0). `SILENTWEAR_ARTIFACTS=artifacts` selects the stale checkpoint set.
The zero-shot gate is what licenses reading the fine-tuned numbers as a statement about the recipe: model
definition, strict checkpoint load, windowing and the balanced-accuracy metric all reproduce the paper
pipeline before any training. With `order="cyclic"` and `draw="device"` the simulation has no randomness
of its own; the seed only selects which 6 windows per class the exporter would have labelled, and the
evaluation set's rest-window downsampling stays pinned at 42.

---

## 5. Results

| quantity | value | checkpoint | source |
|---|---|---|---|
| device zero-shot, batch 1 | 72.22 % (= reference) | `artifacts_reference` | `experiments/deliverable/exp17_BP_round1_correct_weights/logs/phase1_device_eval_zeroshot_b1.log.gz` |
| no-FT reference, batch 2 | 76.67 % | `artifacts_reference` | `experiments/deliverable/exp17_BP_round1_correct_weights/results/reference_S01_fold3.txt` |
| **device BP round 1, batch 2** | **87.78 %** (158/180, 180/180 `sim_errors=0`), +11.11 pp | `artifacts_reference` | `experiments/deliverable/exp17_BP_round1_correct_weights/logs/phase6_device_eval_b2.log.gz`, `results/eval_b2_results.json` |
| paper recipe, batch 2 | 88.89 % (+12.22 pp; Adam, batch 32, 70 % data, 50 epochs, BN updated) | `artifacts_reference` | `experiments/deliverable/exp17_BP_round1_correct_weights/results/reference_S01_fold3.txt` |
| per-step loss breaches round 1 | 1/2160, first at step 1209, max diff 1.28e-3 (exp4 on stale weights: 416/2160, onset 133) | `artifacts_reference` | `experiments/deliverable/exp17_BP_round1_correct_weights/logs/phase3_gvsoc_train_round1.log` (`[loss 1209]`, `Errors: 1 out of 2160`); exp4: `experiments/deliverable/exp4_BP_round1/logs/round1_gvsoc_train.log` |
| host simulation, same cell | 87.78 % (identical to device); 4-subject mean b2–b5 77.95 (seed 42), 78.69 +/- 0.66 over 3 seeds; paper 82.67; no-FT 72.43 | `artifacts_reference` | host simulation in `SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/` (not part of these repositories) |
| cycles per device step (4 fwd+bwd + SGD), recompute MaxPool backward | 140.4 M (fwd 70.68 M, bwd 69.38 M, SGD 56 K, accumulation 48 K); 94.4 % compute, 5.6 % DMA | `artifacts/` fixture; cycle counts are weight-independent | `experiments/exp2/profiling_run/cycle_summary.txt`, `probe_conv.log` (`BENCH train_cycles=140390186 opt_cycles=56427`) |
| L1 / L2 peak (recompute, before transpose dedup) | 127,808 B (99.9 % of 128,000) / 1,793,800 B | same | `experiments/exp2/profiling_run/memory_alloc_training.html` |
| one fwd+bwd, recompute, n_accum 1 | `train_cycles = 34,084,097`; profiled sum 34,095,519; Conv 43.2 %, ConvGrad 30.4 %, MaxPoolGrad 12.5 % (4.25 M, recompute) | same | `experiments/deliverable/exp1_BP_argmax_off/logs/profiletiling.log` (parsed by `results/parse_profiletiling.py`) |
| train-net L2 peak, recompute, with dedup | 1,735,180 B | same | `experiments/deliverable/exp1_BP_argmax_off/results/memory_alloc_deeployStates.html` |
| train-net L2 peak, argmax mask + dedup, 2 MB budget | 1,511,308 B (-12.9 % vs 1,735,180; -15.7 % vs 1,793,800) | same | `experiments/deliverable/exp2_BP_argmax_on/results/memory_alloc_deeployStates.html`; bit-exact `Errors: 0 out of 16` in `experiments/exp3/logs/memtest_dedup.log` |
| train-net L2 peak, argmax mask, strict `--l2 1500000` | 1,482,636 B, `Errors: 0/16`, L1 fits at 128,000 (< GAP9's 131,072) | same | measured in `experiments/exp4/` (`logs/gap9_l2_1p5M.log`) |
| one fwd+bwd, argmax mask, n_accum 1 | `train_cycles = 32,708,926` (ZO single step 35,643,810: +9 %) | same | `experiments/deliverable/exp2a_BP_argmax_naccum1/logs/sim.log`; ZO: `experiments/deliverable/exp3_ZO/logs/sim.log` |
| full round estimate | 540 x (4 x 34.08 M + 56 K) = 73.65 G cycles = 3.3 min at 370 MHz (exp4 estimate 75.8 G) | | computed from `experiments/deliverable/exp1_BP_argmax_off/logs/profiletiling.log` (`results/flops_and_round.py`) |
| head-only + BN fold on device (shipped recipe of 2026-07) | b2 81.67 % -> 89.44 %, `Errors: 0/2160` | **stale `artifacts/inter_session_ft`, not paper-comparable** | `experiments/headonly_ondevice_ft_fixedwindow/` (`run_train_isft_fw.log`, `eval_b2_zs_summary.log`, `eval_b2_ft_summary.log`) |
| S2 chain b1 -> b5 on device | 80.56 / 87.22 / 81.11 / 82.78 / 81.67, mean b2–b5 83.20 vs PyTorch 83.75 | **stale `artifacts/inter_session_ft`** | `experiments/exp1/ondevice_sim_S01_fold3/results/ondevice_vs_pytorch_S01_fold3.csv` |

The 1.11 pp gap to the paper recipe is a single seed on a single cell; the measured draw spread of the
host simulation is +/- 1.58 pp for S01 (3 seeds), so it is not distinguishable from zero. The quantity
that is solid is the +11.11 pp over no-FT: on-device BP recovers 91 % of the paper recipe's gain while
using 54 instead of 126 windows, plain SGD at a static lr instead of Adam with ReduceLROnPlateau, 40
instead of 50 epochs, batch 1 (n_accum 4, SUM) instead of 32, and frozen instead of updated BN
statistics. Because all four deviations are present at once, this bounds their combined cost on this
cell; it does not isolate the BN term (open item 4). Across 4 subjects the host simulation recovers
+6.26 of the +10.24 pp that the paper recipe gains over no adaptation (61 %); every subject improves,
every subject stays below the paper recipe, and the S03/S04 gaps (-2.22 pp each) are inside their draw
spread.

**Why the checkpoint matters.** Every device experiment before exp17 started from
`SilentWear/SilentWear/artifacts/...`, a reproduction trained on the Mac that does not match the paper;
`artifacts_reference/` is the lab-machine reproduction that does. They differ in all 37 tensors: the
lab run trained 47 epochs (best epoch 38, best validation loss 0.2639, accuracy 0.9139) against 30 (best
21, 0.4001, 0.8972); `num_batches_tracked` differs by 765 = 17 extra epochs x 45 batches
(`experiments/deliverable/exp17_BP_round1_correct_weights/results/ckpt_provenance.json`). Numbers on
`artifacts/` are therefore not comparable to the paper.

**Drift on the better-converged checkpoint.** With an identical recipe, exp17 agrees with the reference
much more tightly than exp4 (stale weights): breaches 1 vs 416 of 2160, median |device - reference| 2.0e-6
vs 1.64e-4, p99 3.61e-4 vs 9.51e-3, max 1.28e-3 vs 3.49e-2, first breach at step 1209 vs 133. Because
the breaches are cumulative divergence after the first argmax flip, the error count mostly measures how
many steps remain after the onset; the onset is the meaningful quantity. A plausible reading is that a
better-converged model has larger margins between competing maxima, so near-ties are rarer; with one run
per condition this is an observation, not an established mechanism. The `BENCH` line of the exp17 round
(`train_cycles=1947274114 opt_cycles=30527702 weight_sram=61956`) has a wrapped `train_cycles`; `opt_cycles`
and `weight_sram` are in range and match exp4.

---

## 6. GAP9 memory fit

GAP9 has 128 KiB L1 (131,072 B) and 1.5 MB L2 (read pessimistically as 1,500,000 B; 1.5 MiB would be
1,572,864 B). exp4 built the argmax-mask fixture (`speechnet_train_argmask_b1`, 4 steps) on the Siracusa
tiled runner with `--l1 128000 --l2 1500000`: it compiles, runs bit-exact (`Errors: 0 out of 16`,
`experiments/exp4/logs/gap9_l2_1p5M.log`) and packs to 1,482,636 B of L2 (about 17 KB spare at the
decimal reading, about 90 KB at 1.5 MiB). The OR-Tools solver packs tighter when constrained: the same
graph used 1,511,308 B under the loose 2 MB budget. The recompute path (1,735,180 B) and the un-deduped mask (1,825,356 B) do not fit. This
proves a valid tiling exists within GAP9's memory; it is not a GAP9 build (no BP harness for that
platform, section 3.8). The remaining L2 peak is set in the forward pass by the `Conv_input_*_transposed`
buffers of blocks 2–3, not by any MaxPool buffer.

---

## 7. Known pitfalls and lessons

1. Live batch-1 BatchNorm destroys accuracy (-17.78 pp on device); cause: per-window statistics in
   training and frozen pretrained statistics at inference; fix: frozen-stat BN (`--bn-frozen-stats` +
   `BN_FROZEN_STATS=ON`) or fold BN into the conv for a frozen extractor.
2. `Errors: 2160/2160` with a correct device; cause: the ORT reference used live-BN while the device used
   frozen stats; fix: the PyTorch frozen-BN reference (`_frozen_pytorch_reference`), gated on
   `bn_frozen_stats`.
3. Loss breaches late in a round (exp17: step 1209; exp4: step 133); cause: a MaxPool argmax tie-flip
   from ~1e-6 reduction-order differences, after which the trajectories diverge cumulatively; fix: none
   needed, the acceptance criterion is accuracy from the `[WDUMP]` weights, and larger n_accum only
   shrinks the magnitude, not the onset (`experiments/maxpool_numerical_drift/`).
4. `n_accum` is not a free knob; cause: accumulation sums (`FloatInPlaceAccumulatorV2Template.py`:
   `accum_buffer = gradient` on reset, else `+= gradient`; `SGDTemplate.py`: `w -= lr * grad_acc`, no
   1/n_accum), so the effective step is `lr * n_accum`. A device-faithful host sweep (frozen-BN full-FT,
   stale weights, one favourable draw) at fixed lr 1e-3 gave 85.56 / 86.11 / 43.89 % for n_accum 1 / 4 / 8
   (collapse at effective 8e-3), but 85.56 / 87.22 / 86.67 % with the lr divided by n_accum; fix: keep
   `lr * n_accum` in about 1e-3 to 4e-3 and divide the lr by n_accum when changing n_accum. n_accum does
   nothing for BN (each forward is still one window).
5. Folding frozen BN into conv and training the folded conv collapses at lr 3e-4; cause: the fold removes
   the gamma/sigma damping (about 0.02–0.03) of the conv gradients; fix: keep BN unfolded under S2, or
   re-tune lr about 100x lower.
6. No optimizer graph, tiler fails at `SoftmaxCrossEntropyLoss`; cause: the `-o`/`-t` name lacks
   `_train`, so `derive_optimizer_dir`/`resolve_optimizer_dir` fall back to the training dir; fix: name
   the directory `<model>_train...`.
7. `--n-accum` (and `--n-steps`) must reach the device runner; cause: `N_ACCUM_STEPS` is a compile-time
   constant in `testinputs.h`; fix: pass them explicitly (auto-detection reads `meta_*` keys, but
   explicit is safer).
8. Inference run after a training run fails on every sample with `could not parse logits`; cause: CMake
   caches `TRAINING`/`MEZO_TRAINING` (and `BN_FROZEN_STATS`, `OUTPUT_TOL`, `NUM_CORES`) in
   `TEST_SIRACUSA/build_master`; fix: `rm -rf TEST_SIRACUSA` before every change of fixture kind or `-D` set.
9. `pgrep -f gvsoc_launcher | xargs kill -9` kills your own shell (exit 137, empty log); fix: bracket the
   first letter, `pgrep -f "[g]vsoc_launcher" | xargs -r kill -9`; kill orphans by PID before every run
   (`killall` does not reach `gvsoc_launcher`).
10. A multi-hour run dies silently; cause: the foreground `docker exec` streaming to the host was reaped
    under memory pressure; fix: `docker exec -d ... nohup python3 -u ... > log 2>&1 &` with the log on
    the bind mount.
11. Export "succeeds" but the fixture is nowhere; cause: `-o /app/ETH/...` inside `agitated_hugle`
    (which mounts at `/app`); fix: use the container's own prefix; same for `/tmp` carry checkpoints.
12. Running two GVSoC jobs in the same `DeeployTest` tree; cause: both share `TEST_SIRACUSA` and
    `Tests/Models/speechnet_infer_eval_tmp`, and a machine-wide gvsoc kill in one destroys the other (a
    ZO round-1 run was lost this way at update 1566/2700, about 7 h of simulation, by an evaluation
    harness whose cleanup killed every `gvsoc_launcher`; that cleanup was scoped in commit `93458a0`);
    fix: never run two device jobs concurrently in one checkout, or give each its own build directory
    with `PYTEST_XDIST_WORKER=<tag>`, and check any tool's startup block for a global kill before running
    it beside a long simulation.
13. Host simulation disagrees with the device by a few pp; cause: a `default_rng` stratified draw and a
    per-epoch shuffle, neither of which the device does; fix: reproduce the exporter's `RandomState`
    draw and iterate `mb % 54` cyclically, as the host simulation's `DeviceRecipe` defaults do
    (section 4.7).
14. `BENCH train_cycles` looks small for a 540-step round; cause: `unsigned int` accumulator wraps about
    17 times; fix: quote single-step `BENCH` x step count, or port the MeZO harness's 64-bit hi/lo print.
15. Full-graph `--profileTiling` crashes GVSoC at `Initializing TrainingNetwork...`; cause: per-node
    measurement arrays overrun the L1 stack; fix: `--profileNodes` in disjoint passes.
16. Argmax-mask build showed +1.8 % L2 instead of a reduction; cause: `TransposeSplitPass` gives each of
    MaxPool/MaxPoolArgmax/MaxPoolGradMask its own 314 KB input transpose; fix: `MergeSiblingTransposesPass`
    last in the lowering pipeline.
17. Argmax-mask build asserted in `TilingCodegen.py:537`; cause: the new ops had no NCHW-to-NHWC pass, so
    the channel tiler tiled the W axis; fix: the two layout passes in `LoweringOptimizationPasses.py`.
18. Linking fails with a missing tiled depthwise backward symbol; cause: a pulp-trainlib submodule bump;
    fix: keep the pin at `35ba1f6`.
19. Recomputing X from BN during the backward does not lower peak L2: on random-data 4-step fixtures
    (codegen-only `testMVPTraining.py ... --plotMemAlloc`), the recompute baseline peaked at 1,793,760 B,
    a variant that checkpoints a new ReLU' node at 1,946,208 B (worse) and a variant that fuses the ReLU
    into MaxPoolGrad at 1,793,760 B (equal); cause: the peak is forward-bound. The scratch fixtures and
    branches of this negative result were deleted.

---

## 8. Open items and next steps

1. Rounds 2–4 (b2 -> b5) on `artifacts_reference` with `run_round1.sh` adapted per round; about 4 h
   each. Round r trains on batch r with `--pretrained-weights` = the carry of round r-1 (round 1: the
   official checkpoint), writes the carry of round r, and evaluates batch r+1:

   | round r | train `--batch` | `--pretrained-weights` | carry out | eval `--batch` |
   |---|---|---|---|---|
   | 1 | 1 | `.../leave_one_session_out_fold_3.pt` | carry b1 | 2 |
   | 2 | 2 | carry b1 | carry b2 | 3 |
   | 3 | 3 | carry b2 | carry b3 | 4 |
   | 4 | 4 | carry b3 | carry b4 | 5 |

   Keep every carry on the bind mount, not in `/tmp`. The paper-pipeline reference for these rounds is in
   `results/reference_S01_fold3.txt` (no-FT 72.78 / 80.56 / 72.78, paper FT 82.78 / 88.89 / 87.78 for
   batches 3 / 4 / 5).
2. Subjects S02–S04 on device for a 4-subject table comparable to the host simulation (77.95 / 82.67).
3. Noise floor: 2–3 exporter seeds per cell before quoting any sub-2 pp difference (the exporter draw
   seed is hard-coded to 42 in `create_training_test_data`; exposing it is a small change).
4. BN ablation: frozen BN with everything else the paper's (batch 32, Adam 1e-3, 50 epochs, 70 % split),
   to isolate the BN term from batch size, data fraction and optimizer. Section 5 shows the combined
   penalty of all four deviations is small on this cell, which makes the isolated BN term the open
   question.
5. A true uint8 argmax mask (4x smaller long-lived mask; not peak-changing) and dedup/fusion of the
   forward `Conv_input_*_transposed` buffers (the current L2 peak driver) for more GAP9 headroom.
6. Latency: block-0/1 forward conv and ConvGrad are about 75 % of the step at 4.5 % FPU utilisation
   (12,174,848 conv MACs per fwd+bwd = 24.3 MFLOP in 34,084,097 cycles = 92.1 ms at 370 MHz, i.e.
   0.264 GFLOP/s against an 8-core fp32 peak of 5.92 GFLOP/s; `deliverable/exp1_BP_argmax_off/results/flops_and_round.py`).
   "94 % compute-bound" therefore does not mean efficient: block 0 runs at 23.6 cycles per MAC (one input
   channel, 700-wide rows, a tiny reduction depth). Candidates are a direct conv for the 1-input-channel
   block 0, Conv+BN+ReLU fusion of the frozen-BN affine (with frozen statistics BN is a per-channel
   `a*x + b` pass that could fold into the conv), and reusing the im2col buffer between forward and
   ConvGradW. Double buffering would recover at most the 5 % DMA share and does not fit the full L1.
7. A GAP9 BP harness (port `deeploytraintest.c` the way `deeploymezotest.c` was ported in
   `DeeployTest/Platforms/GAP9/src/`) and a 64-bit cycle counter in the Siracusa BP harness.
8. `FloatMaxPoolTemplate.py:referenceGradTemplate` still passes `(x,y)` (see its `QW-TODO`); the mask
   path uses the correct mapping, the recompute path has only been validated on SpeechNet's shapes.
9. CI covers only `speechnet_train` (4 steps, 128 KB L1) in `test_siracusa_tiled_config.py`; the
   frozen-BN and argmax-mask fixtures are not in CI.

---

## 9. Evidence and artifacts

Measurement logs, results and scripts are in the experiment folders below (paths relative to
`TrainDeeploy/DeeployTest/experiments/` unless stated otherwise).

| topic | tracked artifacts |
|---|---|
| inherited training extension | `TrainDeeploy/README_TRAINING.md` (memory layout contract, runner flags, fixture layout, adding a model); `Onnx4Deeploy/docs/training_graph_generation.md` (generic training export, npz layout, training strategies, gradient accumulation) |
| fixtures (primary) | `TrainDeeploy/SpeechNet_Test/` (`README.md`, `speechnet_{train,optimizer,infer}_BP_round1`, `speechnet_{train,optimizer}_BP_single_step`, `scripts/export_fixtures.sh`, `scripts/crosscheck.py`, `logs/crosscheck.txt`, `logs/export_bp_*.log`, `logs/smoke/siracusa_bp_single.log`, `logs/patchcheck_bp_round1.log`) |
| current accuracy result (exp17) | `deliverable/exp17_BP_round1_correct_weights/`: `scripts/run_round1.sh`; `logs/phase1_device_eval_zeroshot_b1.log.gz`, `phase1_export_zeroshot_b1.log`, `phase2_export_train_b1.log.gz`, `phase3_gvsoc_train_round1.log` (2160-step loss trace, `[WDUMP]`), `phase4_extract_weights.log`, `phase5_export_infer_b2.log`, `phase6_device_eval_b2.log.gz`; `results/eval_b2_results.json`, `reference_S01_fold3.txt`, `ckpt_provenance.json`, `carry_fullfrozen_b1_fold3_ref.pt` |
| superseded rounds (stale weights) | `deliverable/exp4_BP_round1/` (`logs/round1_gvsoc_train.log`, `results/eval_b2_results.json`, `results/carry_fullfrozen_b1_fold3.pt`); `exp1/ondevice_sim_S01_fold3/` (`extract_device_weights.py`, `logs/round{1..4}_gvsoc_train.log`, `results/ondevice_vs_pytorch_S01_fold3.csv`, `run_pytorch_fullfrozen_ondevicedata_chain.py`, `loss_device_vs_ref.png`) |
| frozen-BN reference (Option A / B) | `exp1/logs/smoke_gvsoc_loss.log`; `exp1/optionB_ort_reference/validate_test1.py`, `logs/test2_device_smoke.log`; code `Onnx4Deeploy/onnx4deeploy/models/speechnet_exporter.py:_frozen_pytorch_reference` |
| frozen-BN kernel and host recipe studies (stale weights) | `frozen_bn_kernel_finetune/` (`speechnet_fullfrozen_validate.py`, `fullfrozen_validate.log`, `frozenbn_fair.log`, `frozenbn_variance.log`, `frozenbn_grid_results.txt`, `speechnet_frozenbn_gridsearch.json`) |
| live-BN full-FT regression (stale weights) | `fulltraining_naccum8_regression/` (`full8_e20_device.log`, `full8_e40_device.log`, `origbn_sweep.log`, `speechnet_full_device_verify.py`, sweep scripts and JSON) |
| head-only recipe (stale weights) | `headonly_ondevice_finetune/` (`speechnet_accuracy_eval_untiled.py`, run logs), `headonly_ondevice_ft_fixedwindow/` (`run_train_isft_fw.log`, `eval_b{1,2}_*_summary.log`, `results_*.json`), `speechnet_ondevice_incremental_fold3_headonly/logs/` |
| MaxPool drift | `maxpool_numerical_drift/` (`speechnet_maxpool_90step_acc1_val.log`, `speechnet_maxpool_90step_acc4_val.log`, `speechnet_maxpool_90_argmax.log`, `speechnet_argmax_ort_ref.py`, `speechnet_drift_argmax_proof.png`) |
| paper-recipe reference run | `reference/paper_incremental_ft_s01_fold3.py`, `paper_incremental_ft_s01_fold3.log` |
| per-step cycles and memory (4 accumulation steps) | `exp2/profiling_run/` (`probe_*.log`, `analyze_and_perfetto.py`, `cycle_summary.txt`, `trace.json`, `memory_alloc_training.html`, `memory_alloc_optimizer.html`, `train_profile.log` = the stack-overflow failure) |
| single-step latency, recompute | `deliverable/exp1_BP_argmax_off/` (`logs/profiletiling.log`, `logs/sim.log`, `results/parse_profiletiling.py`, `flops_and_round.py`, `lat_fig*.png`, `mem_fig*.png`, `memory_alloc_deeployStates*.html`, `fixture/`) |
| argmax mask | `deliverable/exp2_BP_argmax_on/` (`logs/sim.log`, `results/memory_alloc_deeployStates.html`, `fixture/`), `deliverable/exp2a_BP_argmax_naccum1/logs/sim.log`, `exp3/logs/` (`memtest_afterfix.log`, `memtest_dedup.log`, `build_argmask_*.log`) |
| GAP9 memory fit | `exp4/logs/gap9_l2_1p5M.log` |
| BP vs ZO single step | `deliverable/exp3_ZO/logs/sim.log` (ZO side of the +9 % comparison) |

---

**Companion guides.** `00_Overview.md` (entry point, environment, checkpoint rule, experiment index),
`01_BP_Guide.md` (back-propagation), `02_ZO_Guide.md` (float zeroth-order), `03_QZO_Guide.md`
(quantized zeroth-order), `04_FusedQZO_NE16_Guide.md` (fused dual-pass kernels and NE16).
