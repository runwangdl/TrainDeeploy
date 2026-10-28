# 02 — Float (fp32) Zeroth-Order (MeZO) On-Device Fine-Tuning of SpeechNet

**Scope.** This guide documents our float32 zeroth-order (ZO, "MeZO") on-device fine-tuning of
SpeechNet: the method, the two ONNX graphs Onnx4Deeploy emits, every TrainDeeploy/Deeploy file that
makes them compile and run on Siracusa (and GAP9), the device harness and runner, how to reproduce
each measurement, and what is open. The quantized variant (QZO) has its own guide; this one covers
the fp32 path only. All paths are relative to `/Users/qiwenwu/ETH/` unless written inside a command
block for a named container. Every path, symbol and number below was checked against the tree on
2026-10-12 (branch `feat/ZO_2fwd_fusion` in both `TrainDeeploy` and `Onnx4Deeploy`).

**Summary.** MeZO replaces the backward pass by two perturbed forward passes along one shared
Rademacher direction `z` and updates the weights with a single scalar times `z`. We emit this as two
graphs (`zo_train`: perturb -> forward -> loss; `zo_update`: in-place weight step), run them from a
dedicated C harness on the fabric controller (`deeploymezotest.c`), and check every perturbed-forward
loss against a PyTorch reference that re-implements the device's own xorshift32 PRNG. Single-step
runs are bit-exact; the full 200-epoch round 1 on the correct checkpoint reaches **90.00 %** balanced
accuracy on batch 2 (exp18), which the host simulation (CleanUp exp2) predicted exactly, at
**384.95 G cycles** and an L2 peak of about **0.79 MB** (versus 1.51 MB for backprop).

> **Status as of 2026-10-12**
>
> - Works: export (`-mode zo-train`), packing, tiled codegen, build, GVSoC run on Siracusa;
>   single-step and multi-step bit-exact (0/2, 0/4, 0/32, 0/800); full round 1 (2700 steps,
>   21 600 forwards) with 5 marginal tolerance breaches (max 1.37x of 1e-3).
> - Measured: round-1 accuracy 90.00 % (exp18, `artifacts_reference` weights); step cost
>   35.9 M cycles (exp6); L2 789 688 B (exp3); host/device agreement to 2.15e-6 after 999
>   replayed updates (CleanUp exp2).
> - Open: rounds 2-4 and other subjects on device; a 100-epoch device round (host sim says
>   within ~0.4 pp of 200); `tile_seed_offset` path untested (no SpeechNet weight tiles); NaN loss
>   diff counts as a pass in the harness; `#pragma clang optimize off` still on the perturb kernel;
>   float ZO has not been run on GAP9 GVSoC (the GAP9 harness/runner exist and were validated with
>   the QZO fixture only).

---

## 1. Concept: MeZO as implemented

### 1.1 The estimator

For trainable parameters `theta` (a list of 22 tensors for SpeechNet), one update step draws one
Rademacher direction `z` (each element +1 or -1), evaluates the loss at two antithetic points and
moves along `z`:

```
z        ~ Rademacher(+-1), one draw per update step, shared by all tensors and all n_accum windows
L+       = L(theta + eps * z)                 # "+eps" forward
L-       = L(theta - eps * z)                 # "-eps" forward
g_proj   = sum_{i<n_accum} (L+_i - L-_i) / (2 * eps * n_accum)      # scalar, MEAN over the window
theta   <- theta - lr * g_proj * z            # SGD step along z, same z
```

Properties that shape the implementation:

- Forward only. No gradient tensors, no activation stash, no backward kernels. This is where the
  L2 saving comes from (section 7).
- The update is one scalar times a regenerable direction. The device never stores `z`; it
  regenerates it from a seed inside the perturb kernel, three times per step (+eps, -eps, update).
- `z` must be bit-identical between the two forwards, the update, and the host reference.
  Everything about the RNG (section 4.3) exists to guarantee that.
- Gradient accumulation takes the MEAN: `g_proj` divides by `n_accum`
  (`DeeployTest/Platforms/Siracusa/src/deeploymezotest.c`, `ComputeZOUpdateCoeffOnCluster`,
  `denom = 2*ZO_EPS*n_accum`). Our backprop harness SUMS accumulated gradients with no 1/N. Copying
  one convention into the other silently rescales the effective step.
- Hyper-parameters used everywhere: `eps = 0.01`, `lr = 3e-6`, `q = 1` direction per step,
  `n_accum = 4`, 200 epochs over 54 windows = 2700 update steps = 10 800 windows = 21 600 forwards.
  `lr = 1e-3` diverges to NaN by about step 11 and is used only for 1- and 2-step smoke tests.

### 1.2 What is trained

All 22 trainable tensors of SpeechNet: 10 conv weights/biases (`blocks_X_0_weight/bias`), 10
BatchNorm affine parameters (`blocks_X_1_weight` = gamma, `blocks_X_1_bias` = beta), 2 fc
(`fc_weight`, `fc_bias`); 15 489 scalars. BatchNorm running mean/var are frozen initializers and are
never perturbed. The device normalises with the pretrained running statistics
(`-D BN_FROZEN_STATS=ON`, kernel global `g_bn_frozen_stats`), and the host reference runs
`model.eval()`, so both sides implement "train gamma/beta, freeze the stats".

### 1.3 Data protocol

Subject S01, condition vocalized, held-out session 3 (fold 3). Fine-tune on 54 stratified windows
(6 per class) of batch 1, drawn by the exporter with `np.random.RandomState(42)`
(`Onnx4Deeploy/onnx4deeploy/data/silent_wear_datasource.py`, `load_batches`, line 263; called with
`seed=42` from `speechnet_exporter.py` line 680). The device never shuffles: window index is
`mb % data_size` with `mb = update_step*n_accum + accum_step` (harness line 352, reference
`speechnet_exporter.py` `_zo_pytorch_reference`, `test_inputs[mb % effective_data_size]`). Evaluate
balanced accuracy on all 180 windows of batch 2. The 54 windows are byte-identical across the BP
(exp17), ZO (exp18) and QZO (exp19) rounds (same `RandomState(42)` stratified draw in all three
export modes; verified by comparing the windows stored in the three fixtures' `inputs.npz`).

Correct checkpoint:
`TrainDeeploy/SpeechNet_Test/pretrained/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt`.
Its reference table (`.../w1400ms/ft_config_0/ft_summary.csv`, fold-3 rows; tracked copy
`TrainDeeploy/DeeployTest/experiments/deliverable/exp17_BP_round1_correct_weights/results/reference_S01_fold3.txt`)
gives batch-2 no-FT 76.67 % and paper-recipe FT 88.89 %. The older `artifacts/` tree is a stale Mac reproduction;
numbers measured on it (exp3, exp5, exp6, all `zo_smoke` runs) are structurally valid but not
paper-comparable.

---

## 2. The graphs Onnx4Deeploy emits

### 2.1 Export entry point

`Onnx4Deeploy/Onnx4Deeploy.py`:

| flag | effect |
|---|---|
| `-mode zo-train` | dispatches to `exporter.export_zo_training(noise_type=...)` (line 529-530). `q-zo-train` is the quantized sibling. |
| `--noise-type rademacher` | perturbation distribution; default `rademacher` (line 392, 669-671). Only `rademacher` is supported on device. |
| `--n-steps S` / `--n-epochs E` / `--n-batches N` | step count; priority `--n-batches > --n-steps > --n-epochs` (lines 434-454); `n_batches = S*n_accum` or `ceil(E*data_size/n_accum)*n_accum`. |
| `--n-accum`, `--data-size`, `--stratified`, `--lr`, `--bn-frozen-stats`, `--batch`, `--pretrained-weights`, `--dataset silentwear`, `--data-path`, `--subject`, `--session`, `--condition` | shared with the BP training export. |

ZO config block: `Onnx4Deeploy/onnx4deeploy/models/speechnet_exporter.py` line 62,
`"zo": {"epsilon": 0.01, "seed": 42, "exceptions": []}`; `learning_rate` default 0.001 (line 49), so
`--lr 3e-6` must be passed explicitly.

### 2.2 `export_zo_training` (`Onnx4Deeploy/onnx4deeploy/core/base_exporter.py`, line 529)

1. `ExportMode.ZO_TRAINING = "zo-train"` (line 104); `setup_paths` (line 360) adds
   `network_zo_train.onnx` and `network_zo_update.onnx` beside `network_infer.onnx`.
2. Builds the forward graph with BatchNorm UNFOLDED: `model.train()`, every `_BatchNorm` module set
   to `eval()` when `bn_frozen_stats` is on, then `_export_to_onnx(..., training_mode=True,
   bn_frozen_stats=...)`. Running stats are snapshotted before tracing and written back into the
   initializers afterwards. Folding BN into Conv would delete gamma/beta, which are exactly the
   parameters ZO trains.
3. `run_inference_optimization` on `network_infer.onnx` with `self._for_training = True`, then
   `infer_shapes_with_custom_ops`.
4. `generate_zo_graph(...)` -> `network_zo_train.onnx`; `generate_weight_update_graph(...)` ->
   `network_zo_update.onnx` (both in `onnx4deeploy/transform/zo_transform.py`).
5. `create_training_test_data_zo()` -> `inputs.npz`, `outputs.npz` (section 2.5).

### 2.3 `network_zo_train.onnx` (`zo_transform.py`)

Built by `inject_perturbation_nodes` (line 390; inner `modify_graph` line 438), then
`append_cross_entropy_loss` (line 886), then `_promote_initializers_to_inputs` (line 21).

- Op filter (line 467): `Conv, Gemm, MatMul, RequantShift, BatchNormalization`. For
  `BatchNormalization` only inputs 1 (gamma) and 2 (beta) are perturbed, never 3/4 (line 480).
- Each trainable initializer `w` gets one `PerturbRademacher` node with attributes
  `idx` (canonical node id), `seed` (42), `eps` (0.01), output edge named `f"{idx}_{w}"`, which
  feeds the original consumer. Note that `modify_graph` re-assigns `epsilon = 0.01` locally
  (line 450); the config's `zo.epsilon` is also 0.01, so this is consistent today, but changing
  eps requires touching that line.
- `idx` is the tensor's position in `network_infer`'s initializer list (`canonical_id`, line 457).
  The same map is used by the update graph, which fixes a pre-existing bug in the shipped ZO branch
  where the train counter advanced by 2 and the update counter by 1, so 21 of 22 tensors drew a
  different `z` in the update than in the forwards (fixed in commit `c1826f7`).
- `BatchNormalization` is re-emitted as `BatchNormInternal` (domain `com.microsoft`, 5 outputs,
  `training_mode=1`; lines 778-820), the same op the BP training graph carries.
- `append_cross_entropy_loss`: adds `label` (INT64, `[B,1]`) at input index 1 so data inputs come
  before weight inputs, appends a canonical two-output `SoftmaxCrossEntropyLoss` with outputs
  `[loss, log_prob]` and `reduction="mean"`, sets graph outputs to `loss` (scalar) and `log_prob`
  (`[B,K]`), and re-adds the logits `value_info` because ORT shape inference cannot see through the
  custom perturb op.
- The 22 perturbed initializers are promoted to graph inputs and removed from `graph.initializer`
  (line 852). The `mezo` opset is dropped (line 860); perturb ops live in the default domain.

Verified on `TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_zo_train_b1_fold3_ref/network.onnx`:
24 inputs (`input`, `label`, 22 weights), 10 initializers (BN running stats), 22 `PerturbRademacher`,
5 `Conv`, 5 `BatchNormInternal`, 1 `Gemm`, 1 `SoftmaxCrossEntropyLoss`, outputs `[loss, log_prob]`,
opsets `("",21) ("ai.onnx.contrib",1) ("com.microsoft",1)`.

### 2.4 `network_zo_update.onnx` (`generate_weight_update_graph`, line 109)

One `PerturbRademacher` per trainable tensor (selected by `weight`/`bias` in the name, line 124),
input `w`, output `f"{w}_updated"` (line 136), same `idx`/`seed`/`eps` as in the train graph. The 22
weights are graph inputs and the 22 `_updated` tensors are graph outputs (lines 358-375). Verified:
22 inputs, 22 outputs, 0 initializers.

The update graph reuses the perturb kernel as the SGD kernel: at runtime the harness overrides eps
with `-lr*g_proj` (section 4.2), so `w_updated = w + (-lr*g_proj) * z`.

### 2.5 Fixture contents (`speechnet_exporter.py`)

`create_training_test_data_zo` (line 642) and `_zo_pytorch_reference` (line 566):

- The reference is a PyTorch loop, not the pure-Python ONNX executor: loads the checkpoint,
  `model.eval()`, and for each update step draws `z` per tensor with
  `_perturb_rademacher(zeros, gseed, node_id, 1.0, 1)` where `gseed = seed + update_step*q + q_i`
  and `node_id` is read from the `idx` attribute of `zo_train`. For each of the `n_accum` windows it
  adds `+eps*z`, evaluates `L+` and `log_prob`, adds `-2*eps*z`, evaluates `L-`, adds `+eps*z` back,
  then applies `-lr*g_proj/q * z` in place. (Restoring by addition leaves a ~6e-7 float32 residue
  per probe pair; the CleanUp host simulation copies a saved baseline instead, see section 8.)
- `inputs.npz`: `arr_0000` = window 0 `(1,1,14,700)`, `arr_0001` = label `(1,1)` int64,
  `arr_0002..arr_0023` = the 22 initial weights in graph-input order, `mb{k}_arr_0000/0001` for the
  remaining windows, `meta_data_size`, `meta_n_batches`, `meta_n_accum`, `meta_zo_eps`,
  `meta_zo_seed`, `meta_zo_lr`, `meta_zo_q`. For the exp18 fixture: 137 keys, data_size 54,
  n_batches 10800, n_accum 4.
- `outputs.npz`: the 22 final weights (keyed by name), `log_prob` (all windows), `loss_plus` and
  `loss_minus` (one per window; 10 800 each for exp18). The device compares its per-window losses
  against `loss_plus`/`loss_minus`.

This departs from the upstream (shipped) ZO reference mechanism. There, `outputs.npz` holds a single
tensor: the `log_prob` vector `(1, N_classes)` of ONE +-eps-perturbed forward, computed by the
pure-Python `run_onnx_graph` executor on a random input and a random label; there are no loss pairs,
no update replay and no steps, and the shipped device test (`mezo_training` run mode) runs that one
forward and compares the one vector. No multi-step reference exists upstream; ours (PyTorch for
float ZO, `run_onnx_graph` for QZO) is our own infrastructure.

### 2.6 Host executor and RNG replica (`Onnx4Deeploy/onnx4deeploy/utils/onnx_node_implementations.py`)

- `NUM_CORES = 8` (line 34), `_scramble` (line 37, `seed*1664525 + 1013904223` mod 2^32),
  `_xorshift32` (line 42), `_perturb_rademacher` (line 118): per core, `chunk = (size>>3) +
  (size&7 != 0)`, seed `_scramble(global_seed + 8*node_id + core_id)`, 32 signs per xorshift word
  consumed LSB first, a trailing partial group takes one more word. This is the host replica of the
  C kernel and is used by both the exporter reference and the single-node kernel fixture.
- `run_onnx_graph` (line 1037) is the pure-Python executor; it has a `BatchNormInternal` handler
  (line 578) and a `SoftmaxCrossEntropyLoss` handler (line 846) rewritten in exp11 to match the
  device operation order (originals kept as `# [exp11-orig]` comments). The float ZO export only
  uses it indirectly; the QZO path uses it for its reference.
- `onnx4deeploy/operators/perturb{rademacher,normal,uniform,triangle,eggroll}.py` and
  `rqsperturb{rademacher,uniform}.py` are `BaseOperatorTest` classes for `-operator` tests, not
  export code. `zo_transform` dispatches on `noise_type` (`gaussian` -> `PerturbNormal`, `uniform` ->
  `PerturbUniform`, `rademacher` -> `PerturbRademacher`, ...), but TrainDeeploy registers only
  `PerturbRademacher` (float) and `RQSPerturbRademacher` (int8), so only `rademacher` deploys.

### 2.7 Packing into TrainDeeploy

`TrainDeeploy/DeeployTest/experiments/zo_smoke/pack_2step_fixture.py <export_dir> <dest> <train_name> <update_name>`
copies `network_zo_train.onnx` -> `<dest>/<train_name>/network.onnx`, `network_zo_update.onnx` ->
`<dest>/<update_name>/network.onnx`, and the same `inputs.npz`/`outputs.npz` into both. It is a pure
split; no graph surgery.

---

## 3. Fixtures in the tree (current vs stale)

**Primary fixtures: `TrainDeeploy/SpeechNet_Test/`** (README there). One fixture per experiment,
regenerated by `SpeechNet_Test/scripts/export_fixtures.sh` and cross-checked against the original
experiment's fixture by `scripts/crosscheck.py` (`logs/crosscheck.txt`):

| fixture | what | equivalent fixture used by the original experiment |
|---|---|---|
| `speechnet_train_ZO_round1`, `speechnet_update_ZO_round1` | round 1: 24-input `zo_train` / 22-in-22-out `zo_update`, 10 800 windows, n_accum 4, lr 3e-6, `artifacts_reference` | `speechnet_zo_{train,update}_b1_fold3_ref` (exp18); every file byte-identical |
| `speechnet_infer_ZO_round1` | batch-2 inference graph carrying exp18's device-trained weights | `speechnet_zo_infer_b1_fold3_ref` (exp18); byte-identical |
| `speechnet_train_ZO_single_step`, `speechnet_update_ZO_single_step` | `--n-steps 1 --n-accum 1` (lr 3e-6) from the same pool, weights and first window; L+ 0.00336818 / L- 0.04362458 equal round 1's first pair | round 1 above |

Older fixtures, `TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/`:

| fixture pair | form | checkpoint | status |
|---|---|---|---|
| `speechnet_zo_{train,update}_b1_fold3_ref` | 24-input, 10 800 windows, lr 3e-6 | `artifacts_reference` | CURRENT (byte-identical to `SpeechNet_Test/speechnet_{train,update}_ZO_round1`). exp18 round-1 fixture; also the source of the `ZO_Conv*` kernel fixtures and of the host simulation's matched draw. |
| `speechnet_zo_infer_b1_fold3_ref` | inference graph with exp18 carry weights | `artifacts_reference` | CURRENT (exp18 phase 5/6 eval fixture; = `SpeechNet_Test/speechnet_infer_ZO_round1`). |
| `speechnet_zo_{train,update}_b1_fold3` | same recipe | `artifacts/` | stale weights (exp5). |
| `speechnet_zo_{train,update}_{1step,2step,8step,exp5,exp6,deliv,fresh1step}`, `speechnet_fzo_{train,update}` | 24-input smoke fixtures (1 step lr 1e-3; 2 step lr 1e-3; 8 step n_accum 2 lr 3e-6; exp5 = 400 windows n_accum 4) | `artifacts/` | stale weights; still valid as structural bit-exactness smoke tests. `_fzo` is identical in shape to `_exp6`/`_deliv`. |
| `speechnet_zo_{train,update}` | initializer form: 2 inputs, 32 initializers | `artifacts/` | STALE FORM. Predates commit `e8b687e`; weight buffers are not shareable, so the update does not propagate across steps. Do not use. |
| `speechnet_baked_{train,update}`, `speechnet_baked200_{train,update}` | 12 `PerturbRademacher` (conv+fc only, BN folded), 25 initializers, no `meta_zo_*` keys | not verified | Undocumented. No builder script in the tree names them (grep on 2026-10-12); treat as an abandoned folded-BN probe, not a deliverable. |

Ladder fixtures (`{fc,conv,mp,b2,sn0,sn1,snc,bn,m4}_zo_{train,update}`) are NOT under
`Tests/Models`; they live in the git-ignored `Onnx4Deeploy/onnx/model/` (listed there on
2026-10-12) with their run logs in `TrainDeeploy/DeeployTest/experiments/zo_smoke/logs/`
(`fc_zo.log`, `conv_zo.log`, `sn1_zo.log`, `sn1_fix.log`, `speechnet_full_fix.log`, ...).

Kernel fixtures (`TrainDeeploy/DeeployTest/Tests/Kernels/FP32/`):

- `ZOPerturb_QW/PerturbRademacher/` — single-node `[128,48]` perturb, `seed=42, eps=0.01, idx=0`,
  generated by `experiments/zo_smoke/gen_perturb_fixture.py`; the isolated RNG gate (0/6144 errors).
  `experiments/dual_pass_fusion/zo_host_ref.py` self-tests against it.
- `ZO_Conv/`, `ZO_Conv_Block1/` — block-0 and block-1 conv with their two perturb nodes lifted
  verbatim from `speechnet_zo_train_b1_fold3_ref` by
  `experiments/dual_pass_fusion/build_zo_conv_fixture.py --block {0,1}` (fp32 microbench baselines
  for the fusion work, `artifacts_reference`).

---

## 4. The Deeploy side, in pipeline order

Everything in this section is tagged `-- QW` in source unless marked inherited. The perturb primitive
(kernel, template, tile constraint, parser/layer/checker/binding) was ported from the upstream Deeploy
`zo-support` branch and is marked "Ported verbatim from Deeploy zo-support" in each file; the
harness, runner, codegen glue and bug fixes are ours.

### 4.1 Parser -> layer -> type checker -> binding -> tiler -> platform

| stage | file | symbol | what it does |
|---|---|---|---|
| parser | `TrainDeeploy/Deeploy/Targets/Generic/Parsers.py` line 3069 | `PerturbRademacherParser` | accepts 1 input / 1 output with attrs `seed`, `eps`, `idx`; fills `data_in`, `data_out`, `seed`, `size` (product of the flat shape; 1-D bare-int shapes wrapped), `nodeIdx = idx`, `eps`. |
| layer | `Deeploy/Targets/Generic/Layers.py` line 666 | `PerturbRademacherLayer` | `computeOps = size`. |
| checker | `Deeploy/Targets/Generic/TypeCheckers.py` line 718 | `PerturbZOChecker` | nLevels passthrough, output signed. |
| binding | `Deeploy/Targets/PULPOpen/Bindings.py` line 488 | `PULPPerturbRademacherBindings` | `float32_t -> float32_t`, template `FloatPerturbRademacherTemplate.referenceTemplate`, `ForkTransformer` (runs on all cluster cores). |
| tiling binding | `Deeploy/Targets/PULPOpen/Tiler.py` line 221 | `PULPPerturbRademacherTilingReadyBindings` | binds `PerturbTileConstraint`. |
| mapper | `Deeploy/Targets/PULPOpen/Platform.py` line 146, 228 | `PerturbRademacherMapper`, `'PerturbRademacher'` in `PULPMapping` | registers the op for the PULP platform (GAP9 inherits via `Deeploy/Targets/GAP9/Platform.py`). |

`Deeploy/Targets/PULPOpen/Parsers.py` and `Deeploy/Targets/PULPOpen/TypeCheckers.py` carry no float-ZO
change (their `-- QW` edits are for the int8 DualConv, the square-padding guard and MaxPoolArgmax).
`Deeploy/DeeployTypes.py` (around line 2906) changed only by a comment: the generated per-network
header no longer declares the perturb control globals; they are the externs in `ZORuntime.h`.

### 4.2 Codegen template: `Deeploy/Targets/PULPOpen/Templates/FloatPerturbRademacherTemplate.py`

`alignToContext` sets `node_id = nodeIdx` (the ONNX `idx` attribute, not the schedule position) and
`tile_seed_offset = 0`. The emitted C for one node (`${size}` elements, `NUM_CORES` = 8):

```c
chunk       = (size >> log2(NUM_CORES)) + ((size & (NUM_CORES-1)) != 0);
chunk_start = MIN(chunk*core_id, size);  chunk_stop = MIN(chunk_start+chunk, size);
chunk_seed  = ((${seed} + perturb_seed_base) + NUM_CORES*${node_id} + core_id) ^ (${tile_seed_offset} * 0x9E3779B1u);
eps         = perturb_eps_use_override ? perturb_eps_override : ${eps}f;
ApplyRademacherPerturbation(&data_in[chunk_start], &data_out[chunk_start], chunk_seed, perturbation_sign, local_size, eps);
```

`USE_SEQUENTIAL_RADEMACHER` selects an alternative kernel; no build defines it, so the packed-bit
kernel is what compiles.

### 4.3 Runtime controls and kernel

`TargetLibraries/PULPOpen/src/ZORuntime.c` / `inc/kernel/ZORuntime.h` define five globals, linked into
every binary, with neutral defaults so inference and isolated kernel tests are unaffected:
`perturbation_sign = 1` (1: +eps, 0: -eps), `perturb_seed_base = 0` (added to each node's baked
seed), `perturb_eps_override = 0.0f`, `perturb_eps_use_override = 0`, `perturb_eps_baked = 1.0f`
(used only by the integer kernels of QZO).

`TargetLibraries/PULPOpen/src/RandomNoise.c` (header `inc/kernel/RandomNoise.h`):

- Line 9: `#pragma clang optimize off` for the whole file (inherited; see section 9).
- `Xorshift32`: `s ^= s<<13; s ^= s>>17; s ^= s<<5`.
- `ApplyRademacherPerturbation(src, dst, seed, dir, size, eps)` (line 337): warm-up
  `rng_state = seed*1664525u + 1013904223u`; `if (dir == 0) eps = -eps`; per 32 elements one
  xorshift, bits consumed LSB first (unrolled by 2); leftover elements take one more word;
  `dst[i] = src[i] + r*eps`.
- Overlap-safe prologue (our fix, commit `186a974`): if `dst - src` is in `(0, 128)` bytes, each
  core copies its chunk into a private `tmp[128]`, `pi_cl_team_barrier(0)`, then perturbs from the
  copy. The decision is identical on every core because the offset is a constant, so there is no
  barrier divergence; the RNG stream is untouched, so the result is bit-identical to the fast path.
- `ApplySequentialRademacherPerturbation` (line 401) is present but unused.

Host mirrors of this exact arithmetic exist in three places and agree with the kernel bit-for-bit:
`onnx_node_implementations._perturb_rademacher` (exporter reference),
`TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/zo_host_ref.py` (`perturb_rademacher`,
with the device-source citations on each function), and the RNG module `device_rng.py` of the
host-simulation code in SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/ (not part of these
repositories; functions `xorshift32`, `rademacher_chunk`, `tensor_direction`, and a `NODE_TABLE`
read from the generated `TrainingNetwork.c`, see section 8).

### 4.4 Tile constraint: `Deeploy/Targets/PULPOpen/TileConstraints/PerturbTileConstraint.py`

Subclass of `UnaryTileConstraint`; `serializeTilingSolution` adds a per-tile replacement
`tile_seed_offset` = flat element offset of the tile's first element (`calculateFlatOffset`), so a
tensor that tiles would get a different stream per tile. For SpeechNet no perturbed tensor tiles
(largest is `blocks_4_0_weight`, 7168 floats), so every emitted `chunk_seed` carries
`^ (0 * 0x9E3779B1u)` and the host replicas assume offset 0. The non-zero path is implemented but
untested (section 9).

### 4.5 Lowering pass: `Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/LoweringOptimizationPasses.py`

`_foldLayoutIntoPerturb` (line 186) can fold the NCHW->NHWC conv-weight permutation into a perturb
node's base weight, but only when that base is a `gs.Constant`. With weights as graph inputs it
returns `False` and the pass falls back to a runtime `Transpose` after the perturb, exactly as the BP
path does. This matters: perturbing a transposed buffer re-orders `z` for every conv with
`in_ch > 1` (section 5, bug 4). CleanUp exp2 proved the deployed order is NCHW by a negative control
(an NHWC host model fails to reproduce the device losses by 1.17).

### 4.6 Scheduling: `DeeployTest/testUtils/trainingUtils.py`

`deferPerturbNodes` (line 137) re-orders the toposorted node list so each perturb node runs just
before its earliest consumer instead of being hoisted to the front with the other constant-only
nodes; `_mockScheduler` applies it to every (BP or ZO) graph. At a 2 MB L2 the saving is small
(train-network L2 peak 788 840 B with vs 791 180 B without, measured on the single-step fixture in
the `experiments/zo_smoke/` runs `logs/singlestep_memalloc.log` / `logs/singlestep_nodefer.log`);
it is kept as a safety valve for tight budgets.

### 4.7 Codegen glue

- `DeeployTest/testMVPTraining.py`: `--zo` (line 346) re-emits `testoutputs.h` with
  `generateZOTestOutputsHeader` (`loss_plus`/`loss_minus` references, lines 285-297); the weight
  input fix at lines 247-258 (when the graph has no gradient buffers, every input after the data
  inputs is an initial weight, `grad_buf_start_idx = len(inputs)`); ONNX `elem_type` -> pointer type
  for every graph input (lines 109-113, mirrors `testMVPOptimizer`).
- `DeeployTest/testMVPOptimizer.py`: `_ONNX_ELEM_TO_PTR` (line 79: FLOAT->`float32_t`,
  INT8->`int8_t`, INT32->`int32_t`, UINT8->`uint8_t`) replaces the former hard-coded `float32_t` for
  every optimizer-graph input (bug 1 in section 5); `--training-dir` -> `build_shared_buffer_maps`
  (line 168).
- `DeeployTest/testUtils/codeGenerateTraining.py`: `TRAINING_NUM_WEIGHT_INPUTS` is emitted whenever
  `grad_buf_start_idx > num_data` (line 82) so the harness copies `testInitWeights[]`;
  `generateZOTestOutputsHeader` (line 226) emits `ZO_TOLERANCE_ABS`, `N_LOSS_REFS`,
  `testLossPlusRef[]`, `testLossMinusRef[]`; `build_shared_buffer_maps` (line 530) matches the update
  graph's inputs and `_updated` outputs (suffix stripped, line 571) to the train graph's weight
  inputs; `_patch_shared_buffers` (line 579) rewrites the generated `OptimizerNetwork.c` so those
  buffers alias `DeeployNetwork_inputs[...]`. This is what makes the update land where the next
  forward reads.
- `DeeployTest/testUtils/zo_graph_prep.py`: a DEAD END. It rewrote the early inference-style ZO
  graphs (mezo domain, 5-in/1-out BN, single-output SCE) into training-frontend form at deploy time.
  Native emission (section 2.3) made it unnecessary; nothing imports it (grep, 2026-10-12). Leave it.

### 4.8 Device harness: `DeeployTest/Platforms/Siracusa/src/deeploymezotest.c`

Whole file ours (523 lines). Selected by CMake `MEZO_TRAINING`. The fabric controller (FC) runs
`main()` and has no FPU, so every float operation is a cluster task and the FC only moves raw bits.

```
InitTrainingNetwork (zo_train) ; InitOptimizerNetwork (zo_update)          # cluster tasks
copy testInitWeights[wi] -> DeeployNetwork_inputs[TRAINING_NUM_DATA_INPUTS + wi]   (TRAINING_NUM_WEIGHT_INPUTS)
perturb_eps_baked = ZO_EPS ; g_bn_frozen_stats = 1 (BN_FROZEN_STATS)
for update_step in 0..N_TRAIN_STEPS-1:
  seed_base = update_step * ZO_Q                                    # line 339
  acc_bits = 0
  for accum_step in 0..N_ACCUM_STEPS-1:
    mb = update_step*N_ACCUM_STEPS + accum_step
    load testDataVector[mb % TRAINING_DATA_SIZE] into inputs[0..DATA-1]   # cyclic, no shuffle
    perturbation_sign=1; perturb_seed_base=seed_base; perturb_eps_use_override=0
    RunTrainingNetwork()            -> lp_bits (memcpy of outputs[0])     # +eps
    perturbation_sign=0
    RunTrainingNetwork()            -> lm_bits                            # -eps
    AccumulateZODiffOnCluster: acc += (L+ - L-)                            # cluster, FPU
  ComputeZOUpdateCoeffOnCluster: g_proj = acc/(2*ZO_EPS*N_ACCUM); perturb_eps_override = -ZO_LR*g_proj
  perturb_eps_use_override=1; perturbation_sign=1; perturb_seed_base=seed_base
  RunOptimizerNetwork()                                               # in place via shared buffers
  perturb_eps_use_override=0
dump_zo_weights(N_TRAIN_STEPS-1)                                    # DUMP_WEIGHTS: "[WDUMP s= wi= n=] hex..."
CompareLossesOnCluster(stored_loss_plus vs testLossPlusRef), same for minus  # |diff| > ZO_TOLERANCE_ABS counts
printf("Errors: %u out of %u")                                       # 2 * min(TOTAL_FWD_PASSES, N_LOSS_REFS)
printf("BENCH train_cycles_hi=.. train_cycles_lo=.. opt_cycles_hi=.. opt_cycles_lo=..")   # uint64 split
```

Details: `ZTRACE` prints with `fflush` (device stdout otherwise flushes only at `main()` return);
every loss is logged as `lp_bits=0x..` / `lm_bits=0x..` (decode with
`struct.unpack('<f', struct.pack('<I', bits))`); `DUMP_STEP_LO`/`DUMP_STEP_HI` dump weights after
each step in a window (exp11 verification); `MAINSTACKSIZE 12000`, `SLAVESTACKSIZE 3800`. The
comparison treats a NaN diff as a pass (`diff > tol` is false for NaN) — open item.

`DeeployTest/Platforms/GAP9/src/deeploymezotest.c` is the same harness with three GAP9 changes:
`void(void*)` wrappers around the generated `Init/Run*Network(core_id, n_cores)` entry points; the
cluster slave stacks relocated from L1 to a static L2 buffer `cluster_slave_stacks[12*SLAVESTACKSIZE]`
via `pi_cluster_task_stacks()` (frees about 30 KB of L1 so `--l1 110000` fits); `ZTRACE` without
`fflush` (pmsis printf). It was validated with the QZO fixture (deliverable/exp13); a float-ZO run on
GAP9 GVSoC is not recorded anywhere in the tree.

### 4.9 Build: CMake

`DeeployTest/Platforms/Siracusa/CMakeLists.txt`: `option(MEZO_TRAINING ...)` (line 7; takes
precedence over `TRAINING`), cache vars `N_TRAIN_STEPS`, `N_ACCUM_STEPS`, `ZO_EPS "0.01f"`,
`ZO_LR "3e-6f"`, `ZO_Q 1`, `ZO_SEED 0`; the `MEZO_TRAINING` branch compiles `src/deeploymezotest.c`
and defines `N_TRAIN_STEPS, N_ACCUM_STEPS, TRAINING_NUM_DATA_INPUTS, ZO_EPS, ZO_LR, ZO_Q, ZO_SEED`,
plus `DUMP_WEIGHTS` and `BN_FROZEN_STATS` when passed. `DeeployTest/CMakeLists.txt` builds the
`training_network` (= zo_train) and `optimizer_network` (= zo_update) object libraries under
`if(TRAINING OR MEZO_TRAINING)`. `Platforms/GAP9/CMakeLists.txt` mirrors this (lines 8-32).
CMake caches these `-D` values per build directory: always `rm -rf TEST_SIRACUSA` (or the
`build_<worker>` subdirectory) when switching fixture kinds.

### 4.10 Runner

`DeeployTest/testUtils/deeployMezoRunner.py::main(tiling_enabled, default_platform, default_simulator)`
with thin entry points `DeeployTest/deeployMezoRunner_tiled_siracusa.py` (Siracusa),
`DeeployTest/deeployMezoRunner_tiled_GAP9.py` (GAP9), `DeeployTest/deeployMezoRunner_tiled_GAP9_w_NE16.py`
(GAP9 + NE16 engine). The platform is the file's `default_platform`; `args.platform` is ignored.

Flags (all ours unless inherited from `DeeployRunnerArgumentParser`):

| flag | meaning |
|---|---|
| `-t <dir>` | the `zo_train` fixture dir (inherited `--dir`). |
| `--optimizer-dir <dir>` | the `zo_update` fixture dir. Default: `resolve_zo_update_dir` replaces `_zo_train` by `_zo_update` in the `-t` basename. Always pass it explicitly. |
| `--n-steps`, `--n-accum`, `--num-data-inputs` | `N_TRAIN_STEPS`, `N_ACCUM_STEPS`, `TRAINING_NUM_DATA_INPUTS` (2 = input + label); auto-detected from `inputs.npz` meta if omitted. |
| `--eps` (0.01), `--lr` (3e-6), `--q` (1), `--seed` (0) | `ZO_EPS`, `ZO_LR`, `ZO_Q`, `ZO_SEED`. The perturbation magnitude is baked into the ONNX node; `--eps` only enters the `g_proj` denominator, so it must equal the exported eps. `--seed` is informational on device (the graph bakes 42). |
| `--tolerance` | `ZO_TOLERANCE_ABS`; forwarded to `testMVPTraining.py --tolerance` (dest `tolerance_abs`, defined in `testUtils/trainingUtils.py` line 226, default 1e-3). |
| `--cores` (8) | `NUM_CORES`; the RNG chunking depends on it, so the reference assumes 8. |
| `--l1`, `--l2`, `--defaultMemLevel`, `--memAllocStrategy`, `--searchStrategy`, `--profileTiling`, `--profileNodes`, `--plotMemAlloc`, `--promoteToL2*` | tiler/codegen passthrough (inherited). |
| `-D KEY=VAL ...` | extra CMake defines (inherited `--cmake`); used for `BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON`. `add_mezo_cmake_flags` already emits `-DBN_FROZEN_STATS=ON` for every ZO build. |
| `--skipgen`, `--skipsim`, `--simulator`, `--toolchain`, `--input-type-map`, `--input-offset-map` | inherited. |
| `--enable-1xk`, `--enable-3x3` | GAP9_w_NE16 only. |

Environment: `PYTEST_XDIST_WORKER=<name>` selects `TEST_SIRACUSA/build_<name>` so two runs can
coexist (the generation directory is keyed on the fixture name already).

Flow: `DeeployTestConfig(mezo=True, training=False, zo_eps/zo_lr/zo_q/zo_seed, ...)`
(`testUtils/core/config.py` lines 36-42) -> `run_complete_test` -> `generate_network` calls
`run_zo_codegen` (`testUtils/core/execution.py` line 31) -> stage 1 `testMVPTraining.py --zo`,
stage 2 `testMVPOptimizer.py --training-dir=<train>` -> `configure_cmake` appends
`add_mezo_cmake_flags` (line 117) -> build -> GVSoC. `run_zo_codegen` supports the tiled path only.

`testMVP.py --run_mode mezo_training` (the upstream single-forward ZO test mode that compares one
perturbed forward's `log_prob` vector, section 2.5) is not present in `TrainDeeploy/DeeployTest/testMVP.py`
and is not used by our flow.

---

## 5. The four bit-exactness bugs (and the isolation ladder that found them)

Device logs of every step of these trails are in `TrainDeeploy/DeeployTest/experiments/zo_smoke/logs/`:
bug 4 before/after `sn1_zo.log` (`Errors: 2 out of 2`) / `sn1_fix.log` (`0 out of 2`), full model
after the fixes `speechnet_full_fix.log` (`0 out of 2`); the fc_bias bug below `twostep_run.log`
(`2 out of 4`) / `twostep_kfix.log` (`0 out of 4`); bug 3 the `mezo_1step_*` series.

1. **Optimizer-graph input typing.** Symptom: `zo_update` failed type-check ("no adequate
   mapping"). Cause: `testMVPOptimizer.py` typed every optimizer input `float32_t`. Fix:
   `_ONNX_ELEM_TO_PTR` keyed on the ONNX `elem_type` (`testMVPOptimizer.py` line 79). Harmless for
   fp32 but the single fix that also unblocked the int8 QZO update graph.
2. **Missing perturbation sign in the shipped quantized kernel.** Symptom: `L+ == L-`, `g_proj = 0`.
   Cause: `ApplyPerturbQuantRademacher_CHW/_i32` had no `dir` argument. Fix: added `dir`, threaded
   `perturbation_sign` through the templates (`RandomNoiseQuant.c`, `RQSPerturbRademacher*Template.py`).
   The float kernel already had `dir`; listed here because it belongs to the same bring-up.
3. **Float op on the FPU-less fabric controller.** Symptom: the sim looked hung / ~500x slow; a
   killed run showed an empty log. Cause: `float lp; stored[mb] = lp;` in `main()` compiled to
   `flw/fsw`, trapping into `pos_illegal_instr` (a jump-to-self). Fix: the FC handles losses only as
   `uint32_t` bits; `AccumulateZODiffOnCluster` and `ComputeZOUpdateCoeffOnCluster` do the float math
   on the cluster. Commit `430990d`. General PULP gotcha, not ZO-specific; `printf("%f")` is the same
   trap.
4. **`_foldLayoutIntoPerturb` re-ordered `z`.** Symptom: conv0 (in_ch = 1) matched, conv1-4
   (in_ch > 1) did not. Cause: folding the NCHW->NHWC permutation into the perturb's constant base
   weight made the kernel draw `z` over the transposed flat buffer while the reference perturbs in
   NCHW order. Fix: perturb in NCHW, then runtime `Transpose` (the BP path's `_appendTranspose`);
   with weights as inputs the fold no longer applies at all (section 4.5). Commit `430990d`.

A fifth, multi-step-only bug: **fc_bias L1 buffer overlap.** Symptom: in the 2-step test, step 0 showed
a ~0.002 bias in both losses although no update had happened. Diagnosis by differential tracing of
the fc Gemm's actual inputs: bias `C` corrupted on odd indices only. Cause: `fc_bias` (9 floats) is
the only tensor whose length is not a multiple of 8; the tiler placed the perturb's L1 output 4 bytes
into its input (the 1-D output shape is stored as a bare int, `UnaryTileConstraint` ties dims
loosely), so `dst[i]` overwrote `src[i+1]` before it was read. Fix: the overlap-safe prologue in
`ApplyRademacherPerturbation` (section 4.3), commit `186a974`; a tiler-level fix (full-size perturb
tiles) was infeasible because the constraint solver returns infeasible for biases feeding tiled
consumers.

Isolation ladder (supervisor's method: smallest graph first, then one layer or one perturbed tensor
at a time, comparing device loss bits to a fresh reference; fixtures in `Onnx4Deeploy/onnx/model/`,
logs in `experiments/zo_smoke/logs/`):

| rung (fixture) | perturbed | before fixes | after |
|---|---|---|---|
| fc only, no tiling (`fc_zo`) | fc | pass | - |
| conv0 + frozen BN0, tiled (`conv_zo`) | conv0 | pass | - |
| + MaxPool (`mp_zo`) | conv0 | pass | - |
| two blocks (`b2_zo`) | conv0 | pass | - |
| full 5-block forward (`sn0_zo`) | conv0 | pass | - |
| BN0 gamma/beta (`bn_zo`) | BN affine | pass | - |
| conv0 + BN0, four perturbs (`m4_zo`) | 4 | pass | - |
| all five conv weights (`snc_zo`) | 10 | FAIL | pass |
| conv1 only (`sn1_zo`) | conv1 | FAIL | pass |
| full SpeechNet (`speechnet_zo_*`) | 22 | FAIL | bit-exact |

The conv0-passes / conv1-fails contrast (in_ch 1 vs 8) localised bug 4 immediately.

Dead ends recorded so they are not re-tried: `zo_graph_prep.py` (section 4.7); running the ZO forward
through the inference harness `deeploytest.c` (BatchNormInternal is a training op and gave garbage on
the NHWC inference path); the "~500x slowdown" attributed to weights-as-inputs / runtime transposes
(it was bug 3).

---

## 6. How to run

Containers: `agitated_hugle` (ETH mounted at `/app`; export), `traindeeploy` (ETH at `/app/ETH`;
Siracusa build + GVSoC), `deeploy_gap9` (TrainDeeploy at `/app/Deeploy`; GAP9). The primary fixtures
are the ones in `TrainDeeploy/SpeechNet_Test/` (section 3); the `Tests/Models/Training/SpeechNet/`
names are given as the equivalent fixture used by exp18. Expected lines are quoted from the logs
named in each subsection.

### 6.1 Export the fixture (agitated_hugle)

The fixtures already exist; regenerate them with
`bash TrainDeeploy/SpeechNet_Test/scripts/export_fixtures.sh zo_round1 | zo_single | zo_infer`
(export + pack in one go; logs `SpeechNet_Test/logs/export_zo_*.log`, `logs/pack.log`). By hand:

```bash
docker exec agitated_hugle bash -lc 'ln -sfn /app/SilentWear/SilentWear_data /app/SilentWear_data'   # once
CKPT=/app/TrainDeeploy/SpeechNet_Test/pretrained/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
DATA=/app/SilentWear/SilentWear_data/data_raw_and_filt
# full round 1 (exp18 phase 1): 200 epochs x 54 windows / n_accum 4 = 2700 steps, 10800 windows
docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && python3 Onnx4Deeploy.py -model SpeechNet -mode zo-train \
  --noise-type rademacher --bn-frozen-stats --dataset silentwear --data-path $DATA \
  --pretrained-weights $CKPT --subject S01 --session 3 --condition vocalized --batch 1 \
  --n-epochs 200 --data-size 54 --stratified --n-accum 4 --lr 3e-6 -o ./onnx/model/speechnet_zo_b1_fold3_ref"
# single step: same command with --n-steps 1 --n-accum 1 (lr 3e-6) instead of --n-epochs 200 --n-accum 4
```

Expected (`SpeechNet_Test/logs/export_zo_round1.log`): `ZO training sim: n_batches=10800 n_accum=4
n_steps=2700 q=1 lr=3e-06 eps=0.01 seed=42`, then `outputs.npz: 22 final weights + log_prob (10800, 9)
+ loss_plus/minus (10800 step-losses)`. Export time is dominated by the PyTorch reference (21 600
forwards).

### 6.2 Pack (traindeeploy)

```bash
docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && \
  python3 experiments/zo_smoke/pack_2step_fixture.py /app/ETH/Onnx4Deeploy/onnx/model/speechnet_zo_b1_fold3_ref \
    ../SpeechNet_Test speechnet_train_ZO_round1 speechnet_update_ZO_round1"
```

Expected (`SpeechNet_Test/logs/pack.log`): `train : ... (network.onnx 24 inputs / 10 inits)` and
`update : ... (network.onnx 22 inputs / 22 outputs)`.

### 6.3 Single-step smoke on Siracusa (traindeeploy)

```bash
docker exec traindeeploy bash -lc "pgrep -f '[g]vsoc_launcher' | xargs -r kill -9; \
  cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA && \
  python3 deeployMezoRunner_tiled_siracusa.py \
    -t ../SpeechNet_Test/speechnet_train_ZO_single_step \
    --optimizer-dir ../SpeechNet_Test/speechnet_update_ZO_single_step \
    --n-steps 1 --n-accum 1 --num-data-inputs 2 --eps 0.01 --lr 3e-6 --q 1 --seed 42 \
    --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
    --cores 8 -D BN_FROZEN_STATS=ON --plotMemAlloc"
```

(`SpeechNet_Test/scripts/run_smoke_siracusa.sh zo_single` runs the same, without `--plotMemAlloc`.) A fixture outside
`DeeployTest/` makes the runner warn "outside base directory" and generate into
`TEST_SIRACUSA/<fixture name>`; this is expected.

Expected (`SpeechNet_Test/logs/smoke/siracusa_zo_single.log`):
`[loss+ 0] computed=0.003368 ref=0.003368 diff=0.000000`, `[loss- 0] computed=0.043624 ref=0.043625
diff=0.000000`, `Errors: 0 out of 2`, `BENCH train_cycles_hi=0 train_cycles_lo=35634779 opt_cycles_hi=0
opt_cycles_lo=143951`. The older stale-weight equivalent (`speechnet_zo_{train,update}_deliv`,
`--lr 0.001`, exp3) gives `computed=0.016488` / `0.069274`, `Errors: 0 out of 2`, 35 643 810 train
cycles (`experiments/deliverable/exp3_ZO/logs/sim.log`; that older build prints `BENCH train_cycles=`
without the hi/lo split). Memory plots land in `TEST_SIRACUSA/<fixture path>/deeployStates{,_optimizer}/memory_alloc.html`.
Add `--profileTiling` for the per-node cycle breakdown (exp6).

### 6.4 Full round 1 on Siracusa (exp18, about 12-18 h on this host)

```bash
docker exec -d traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA/build_zo && \
  PYTEST_XDIST_WORKER=zo nohup python3 -u deeployMezoRunner_tiled_siracusa.py \
    -t ../SpeechNet_Test/speechnet_train_ZO_round1 \
    --optimizer-dir ../SpeechNet_Test/speechnet_update_ZO_round1 \
    --n-steps 2700 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 3e-6 --q 1 --seed 42 \
    --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max --cores 8 \
    -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON > <log on the bind mount> 2>&1 &"
```

Equivalent fixture used by exp18: `-t Tests/Models/Training/SpeechNet/speechnet_zo_train_b1_fold3_ref
--optimizer-dir Tests/Models/Training/SpeechNet/speechnet_zo_update_b1_fold3_ref` (byte-identical).
exp18's own six-phase driver is
`TrainDeeploy/DeeployTest/experiments/deliverable/exp18_ZO_round1_correct_weights/scripts/run_round1.sh [1-6|all]`
(phase 3 is the command above on the old fixture names).

Expected (`exp18_ZO_round1_correct_weights/logs/phase3_gvsoc_zo_round1.log`): `update 2700/2700`,
`Errors: 5 out of 21600`, 22 `[WDUMP s=2699 ...]` lines,
`BENCH train_cycles_hi=89 train_cycles_lo=2697639026 opt_cycles_hi=0 opt_cycles_lo=389289212`.
Launch detached with the log on the bind mount: a foreground `docker exec` streams stdout through a
host process that memory pressure can kill, and the `[WDUMP]` block is only printed at the end.

Phases 4-6: `experiments/deliverable/exp5_ZO_round1/extract_zo_weights.py --gvsoc-log <log>
--base-ckpt <ckpt> --out-carry <carry.pt>` (parses `[WDUMP]`, `NAME_MAP` = the 22 names in
weight-input order, writes BN stats unchanged; expected `parsed [WDUMP] last step s=2699: 22 tensors`,
max movement 1.214e-02, `logs/phase4_extract_weights.log`); `Onnx4Deeploy.py -model SpeechNet -mode infer
--pretrained-weights <carry.pt> --batch 2 ...` (or `export_fixtures.sh zo_infer`, which writes
`SpeechNet_Test/speechnet_infer_ZO_round1`); then
`experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py --infer-dir ../SpeechNet_Test/speechnet_infer_ZO_round1`
(expected `Balanced accuracy : 0.9000`, `162/180`; exp18 `results/eval_b2_results.json`).

### 6.5 GAP9 (deeploy_gap9) — harness and runner exist; validated with QZO only

```bash
docker exec deeploy_gap9 bash -lc '
  source /app/install/gap9-sdk/.gap9-venv/bin/activate
  source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
  export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
  pgrep -f "[g]vsoc_launcher" | xargs -r kill -9 ; rm -rf /app/Deeploy/DeeployTest/TEST_GAP9
  cd /app/Deeploy/DeeployTest
  python3 deeployMezoRunner_tiled_GAP9.py -t <zo_train dir> --optimizer-dir <zo_update dir> \
    --n-steps 1 --n-accum 1 --num-data-inputs 2 --eps 0.01 --lr 0.001 --q 1 --seed 42 \
    --l1 110000 --l2 1500000 --defaultMemLevel L2 -D BN_FROZEN_STATS=ON'
```

`--l1 110000` (not 128000) and the L2 slave stacks are mandatory on GAP9 (usable L1 is about 93 KB
otherwise). This exact command with the float fixture has no recorded run; expect to debug.

### 6.6 Host simulation (Mac, CPU)

The device-faithful host simulation is the host-simulation code in
SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/ (not part of these repositories), experiment
`exp2_ZO_simulation`, run with the `silent_wear` conda environment. It is gated by four checks,
each of which must pass before its accuracy numbers are trusted:

1. zero-shot: the 20 no-FT cells reproduce the reference table `ft_summary.csv` exactly;
2. estimator: the central difference `(L+ - L-)/(2 eps)` along `z` matches the autograd directional
   derivative in float64;
3. device RNG: the host `z` reproduces the device's logged `lp_bits`/`lm_bits` of exp18's phase-3
   log at update step 1, and still at step 1000 after replaying 999 updates;
4. negative control: the same check with an NHWC-ordered `z` must FAIL (it does, by 1.17 in loss).

The fine-tuning run takes its 54 labelled windows directly from
`TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_zo_train_b1_fold3_ref/inputs.npz`
(= `SpeechNet_Test/speechnet_train_ZO_round1/inputs.npz`), so host and device train on byte-identical data.

---

## 7. Results

Paths in the source column are relative to `TrainDeeploy/DeeployTest/experiments/`.

| quantity | value | checkpoint | source |
|---|---|---|---|
| Single-step device vs reference | `Errors: 0 out of 2`, diff 0.000000 | `artifacts_reference` (SpeechNet_Test) / `artifacts/` (exp3) | `TrainDeeploy/SpeechNet_Test/logs/smoke/siracusa_zo_single.log`; `deliverable/exp3_ZO/logs/sim.log` |
| 2-step (n_accum 1, lr 1e-3) | step 0 bit-exact, step 1 within 1e-6, `0/4` | stale | `zo_smoke/logs/twostep_nativeinput.log` |
| 8-step (n_accum 2, lr 3e-6) | `0/32`, max diff 3e-6 | stale | `zo_smoke/logs/eightstep_native.log` |
| 100 epochs on 4 windows (n_accum 4) | `0/800`, max diff 3.5e-5 | stale | `zo_smoke/logs/exp5_zo.log` |
| Isolated perturb kernel | `0/6144` | n/a | `Tests/Kernels/FP32/ZOPerturb_QW`; `zo_smoke/logs/perturb_full_sim.log`, `perturb_full_sim_after_zoruntime.log` |
| Step cost, n_accum 1 | train 35 643 810 + opt 144 403 cycles (stale); 35 634 779 + 143 951 (`artifacts_reference`) | both | `deliverable/exp3_ZO/logs/sim.log`; `SpeechNet_Test/logs/smoke/siracusa_zo_single.log` |
| Step breakdown (`--profileTiling`) | 35 869 629 cycles (= train 35 712 108 + opt 157 521): +eps fwd 49.8 %, -eps fwd 49.8 %, update 0.44 %; all perturbation 457 992 cycles = 1.28 %; conv0 + conv1 = 75.5 % | stale | `deliverable/exp6_ZO_single_step_latency/logs/profiletiling.log`, `logs/runner.log` (per-node lines + `BENCH`) |
| Matched BP step, n_accum 1 | 32 708 926 cycles (ZO is +9 %) | stale | `deliverable/exp2a_BP_argmax_naccum1/logs/sim.log` |
| L2 peak, train net | 789 688 B, L1 127 232 B (exp3); 788 840 B with perturb deferral (section 4.6) vs BP 1 511 308 B (argmax mask) | stale | `deliverable/exp3_ZO/results/memory_alloc_deeployStates.html`; `deliverable/exp2_BP_argmax_on/results/memory_alloc_deeployStates.html` |
| Round 1, old weights (exp5) | b2 88.33 %, `0/21600`, 384.92 G train + 388.4 M opt (385.3 G total) | stale | `deliverable/exp5_ZO_round1/logs/round1_gvsoc_zo.log.gz`, `results/eval_b2_results.json` |
| **Round 1, correct weights (exp18)** | **b2 90.00 %** (162/180), `Errors: 5 out of 21600` (max 1.37e-3 = 1.37x tol, p99.9 below tol), 180/180 eval windows `sim_errors=0`, **384.95 G train cycles** + 389.3 M opt, max weight move 1.214e-2 | `artifacts_reference` | `deliverable/exp18_ZO_round1_correct_weights/logs/phase3_gvsoc_zo_round1.log`, `logs/phase4_extract_weights.log`, `logs/phase6_device_eval_b2.log.gz`, `results/eval_b2_results.json` |
| Context on the same cell | no-FT 76.67 %, paper FT 88.89 %, BP exp17 87.78 % (~75.8 G), QZO exp19 85.00 % (89.5 G) | `artifacts_reference` | `deliverable/exp17_BP_round1_correct_weights/results/reference_S01_fold3.txt` (no-FT / paper FT), `exp17_.../results/eval_b2_results.json`, `deliverable/exp19_QZO_round1_correct_weights/results/eval_b2_results.json` |
| Host prediction of exp18 | 90.00 % for S01 b2, recorded while the device run was at ~1300/2700 steps | `artifacts_reference` | host simulation (CleanUp exp2, outside these repositories) |
| Host/device direction stream | worst diff 1.43e-6 at step 1, 2.15e-6 at step 1000 after replaying 999 updates; NHWC control fails (1.17) | `artifacts_reference` | host simulation (CleanUp exp2), checked against `deliverable/exp18_ZO_round1_correct_weights/logs/phase3_gvsoc_zo_round1.log` |
| Host ZO vs BP, 4 subjects, 3 draws | ZO 78.02 +- 0.30 vs BP 78.69 +- 0.66 mean b2-b5 (difference 0.67 pp, within the combined spread) | `artifacts_reference` | host simulation (CleanUp exp2) |

exp18's 384.95 G train cycles are `89*2^32 + 2 697 639 026` (about 17.3 min at 370 MHz); the harness
prints the uint64 counter as hi/lo halves. It matches exp5's 384.92 G (same graph and step count,
different weights), confirming nothing structural changed with the checkpoint swap. Per-method cost on
this cell: BP ~75.8 G, QZO 89.5 G, ZO 385 G; ZO buys its accuracy and halved L2 with ~5x BP's
cycles. exp18's training-loss diff statistics over the 21 600 forwards: median 3.40e-5, mean
9.13e-5, p99 5.94e-4, p99.9 8.74e-4, max 1.369e-3; 965 (4.5 %) are exactly 0.

Reading the exp6 breakdown: perturbation is 1.3 % of a step today because the fp32 forward
dominates; once the convs move to an int8 datapath the same perturbation work becomes 15-30 % of the
step, which is the motivation for the QZO and fusion work. The forward shrinks but the perturbation
element count does not: the same 15 489 parameters are perturbed three times per step on the cluster
cores (about 0.46 M cycles today: perturbed forwards ~0.31 M + update ~0.15 M), at about 7 cycles per
element on large tensors (unoptimised, memory-bound) plus a fixed 1 500-2 000 cycles per node (RNG
seed, tiling/DMA setup, launch) that dominates for the many tensors of at most 16 elements.

Reading exp18's five breaches: ZO has no MaxPool backward, so none of BP's argmax tie-flip drift,
but "bit-exact by construction" was too strong; the correct statement is "essentially bit-exact:
5 marginal breaches in 21 600, none above 1.4x tolerance, p99.9 still below tolerance". All five are
on the `loss-` pass in the last quarter of the trace (steps 6483, 9772, 10096, 10265, 10778; diffs
1.02-1.37x tol) and none persists. A genuine BP-style tie-flip moves the weights permanently and
produces errors orders of magnitude past tolerance that persist (exp17 BP on the old weights reached
3.49e-2 = 35x tol); nothing here does that.

---

## 8. Host simulation agreement (CleanUp exp2)

The host-simulation code in SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/ (not part of these
repositories; its ZO module `zeroth_order.py`, `ZeroOrderRecipe` / `finetune`) reproduces
`deeploymezotest.c` statement for statement: `seed_base = update_step`, one `z` per update step
shared by the `n_accum` windows, MEAN accumulation, cyclic window order, frozen BN (`model.eval()`),
Rademacher `z` from its `device_rng.py` (`rng="device"`, default), which mirrors
`ApplyRademacherPerturbation` exactly as `zo_host_ref.py` does in this repository. Theta is
restored by copying a saved baseline, which matches the device (the perturb node writes to a
separate buffer and never touches the master weights). The exporter's own reference restores by
addition instead (section 2.5), which is why the two host references differ at the 1e-7 level while
both pass the 1e-3 device tolerance.

That RNG module carries a `NODE_TABLE`: the `(node_id, parameter, size)` triples read from the generated
`TrainingNetwork.c` of the deployed fixture (any host replica needs the same table) (node ids 0,1,2,3,6,7,8,9,12,...,30,31 — Deeploy's
scheduling index, not recoverable from the ONNX alone). If the architecture or the schedule changes,
re-read this table.

Matched to the device on four axes, each verified: the labelled draw (byte-identical to the
exporter's), the cyclic order, the MEAN accumulation, and the direction stream (checked against the
device's logged `lp_bits`/`lm_bits` to 2.15e-6 after 999 replayed updates). With `rng="device"`,
`--seed` affects only which 54 windows are labelled. Matching the stream moved the earlier host result
by +0.67 pp and closed what had looked like a ZO-vs-BP deficit.

---

## 9. Known pitfalls and lessons

- Symptom: `Errors` nonzero everywhere after switching fixture kinds. Cause: CMake caches
  `TRAINING`, `MEZO_TRAINING`, `BN_FROZEN_STATS`, `NUM_CORES`, `OUTPUT_TOL` per build dir. Fix:
  `rm -rf TEST_SIRACUSA` (or `TEST_SIRACUSA/build_<worker>`) before each run.
- Symptom: a new sim looks hung. Cause: orphaned `gvsoc_launcher` processes starve it;
  `killall`/`pkill` by name do not reliably kill them. Fix: `pgrep -f "[g]vsoc_launcher" | xargs -r
  kill -9` (bracketed so it does not match its own shell), verify zero remain.
- Symptom: a long device round dies with a bare `gmake ... Error 2`, no `BENCH`, no `[WDUMP]`.
  Cause: another tool ran `rm -rf TEST_SIRACUSA` and a bare `pgrep -f gvsoc | kill -9`; exp9's
  `qzo_accuracy_eval_untiled.py` did exactly this and destroyed the first exp18 round at update
  1566/2700 (~7 h). `PYTEST_XDIST_WORKER` isolates only the runner's build dir, not another script's
  wipe. Fix: commit `93458a0` scopes both to the worker and the eval's own temp name; read the
  startup block of any harness before running it beside a live round. The killed attempt's log is
  `exp18_ZO_round1_correct_weights/logs/phase3_gvsoc_zo_round1_KILLED_at_1566.log.gz`.
- Symptom: the runner loads the perturbed-forward graph as the update graph and generation fails.
  Cause: `-t` basename lacks `_zo_train`, so `resolve_zo_update_dir` resolves to the same dir. Fix:
  keep the `<name>_zo_train_* / <name>_zo_update_*` pair and pass `--optimizer-dir` explicitly.
- Symptom: NaN losses after ~11 steps. Cause: `lr = 1e-3`. Fix: `lr = 3e-6` for anything beyond a
  2-step smoke test. The harness counts a NaN diff as a pass, so check the log, not only `Errors`.
- Symptom: device losses differ from the reference by a constant small bias at step 0. Cause:
  tiler under-allocation of an odd-length 1-D perturb output (fc_bias). Fix: shipped in
  `ApplyRademacherPerturbation`; if a new model has other odd-length tensors with offsets >= 128 B,
  the `tmp[128]` guard does not cover them (the comment documents the bound).
- Symptom: `L+ == L-` (quantized path). Cause: no `dir` argument in the kernel. Fix: section 5 bug 2.
- Symptom: run shows nothing and never ends. Cause: float op on the FC. Fix: section 5 bug 3; never
  write float arithmetic or `printf("%f")` in `main()`.
- Symptom: device z differs from host for `in_ch > 1` convs. Cause: layout fold into the perturb
  base. Fix: section 5 bug 4; keep weights as graph inputs.
- Symptom: BP-style accuracy numbers from ZO or vice versa. Cause: BP sums accumulated gradients, ZO
  takes the mean. Fix: never copy `lr` or `n_accum` semantics across the two harnesses.
- Symptom: `BENCH train_cycles` far too small on a long run. Cause: older builds used a `uint32`
  counter (exp5-era wrap). Fix: the harness now prints `hi/lo` halves; reassemble `hi*2^32 + lo`.
- Symptom: `--eps` changes nothing. Cause: eps is baked in the ONNX node; the flag only feeds the
  `g_proj` denominator. Fix: re-export to change eps and pass the same value to the runner.
- Symptom: export cannot find the dataset. Cause: the exporter expects `/app/SilentWear_data`. Fix:
  `ln -sfn /app/SilentWear/SilentWear_data /app/SilentWear_data` in `agitated_hugle`.
- Symptom: QZO numbers (or stale-weight exp3/exp5/exp6 numbers) quoted as float-ZO accuracy.
  Fix: the float-ZO accuracy result is exp18 (90.00 %, `artifacts_reference`); use section 7.

---

## 10. Open items / next steps

1. Device rounds 2-4 for S01/fold 3 (about 12 h each on this host) and subjects S02-S04; the host
   simulation (section 8) already has all cells.
2. A 100-epoch device round: an accuracy-vs-epochs study in the host simulation finds float ZO
   within ~0.4 pp of 200 epochs at 100; that would halve the 385 G cycles.
3. Characterise exp18's five tolerance breaches to the ulp from the logged raw bits (no re-run
   needed).
4. `tile_seed_offset` for perturbed tensors that tile: implemented in `PerturbTileConstraint`, but
   the host replicas assume offset 0 and no fixture exercises it. Before deploying a model with a
   large perturbed weight, either model the offset in the host replicas (`_perturb_rademacher`,
   `zo_host_ref.py`, the host simulation's RNG module) or
   assert `numTiles == 1` per perturb at codegen.
5. Flag NaN diffs as errors in `CompareLossesOnCluster`.
6. Remove `#pragma clang optimize off` from `RandomNoise.c` and re-validate bit-exactness (the
   kernel runs unoptimised today; exp6's `logs/profiletiling.log` gives about 7 cycles per element
   on large tensors and a 1 500-2 000 cycle fixed cost per node, section 7).
7. Run the float fixture on GAP9 GVSoC with `deeployMezoRunner_tiled_GAP9.py` (section 6.5) to
   confirm the harness port for fp32 kernels.
8. Strip the residual `ZTRACE` phase markers and loss-bit dumps once no longer needed for
   diagnosis (the host simulation's device-RNG check reads them, so keep the `lp_bits`/`lm_bits`
   lines or update that check).
9. Decide whether the exporter's PyTorch reference should restore theta by copy rather than
   addition, to match the device and the host simulation exactly.
10. Document or delete the `speechnet_baked*` fixtures.

---

## 11. Evidence and artifacts

Measurement logs are in the experiment folders listed below (paths relative to `TrainDeeploy/`
unless stated).

| topic | tracked artifacts |
|---|---|
| Current fixtures | `SpeechNet_Test/speechnet_{train,update}_ZO_round1`, `speechnet_infer_ZO_round1`, `speechnet_{train,update}_ZO_single_step`; `SpeechNet_Test/scripts/{export_fixtures.sh,crosscheck.py,run_smoke_siracusa.sh}`; `SpeechNet_Test/logs/{export_zo_round1,export_zo_single,export_zo_infer,pack,crosscheck}.*`, `logs/smoke/siracusa_zo_single.log`; README `SpeechNet_Test/README.md` |
| Exporter | `Onnx4Deeploy/Onnx4Deeploy.py`, `onnx4deeploy/core/base_exporter.py` (`export_zo_training`), `onnx4deeploy/transform/zo_transform.py`, `onnx4deeploy/models/speechnet_exporter.py` (`create_training_test_data_zo`, `_zo_pytorch_reference`), `onnx4deeploy/utils/onnx_node_implementations.py` (`_perturb_rademacher`) |
| Deeploy / harness / runner | files in section 4; `DeeployTest/Platforms/{Siracusa,GAP9}/src/deeploymezotest.c`, `DeeployTest/testUtils/deeployMezoRunner.py` |
| Packing and kernel fixtures | `DeeployTest/experiments/zo_smoke/{pack_2step_fixture.py,gen_perturb_fixture.py}`; `DeeployTest/Tests/Kernels/FP32/ZOPerturb_QW/`, `ZO_Conv/`, `ZO_Conv_Block1/`; `DeeployTest/experiments/dual_pass_fusion/{zo_host_ref.py,build_zo_conv_fixture.py}` |
| Bit-exactness bring-up and isolation ladder | `DeeployTest/experiments/zo_smoke/logs/` (`fc_zo`, `conv_zo`, `mp_zo`, `b2_zo`, `sn0_zo`, `bn_zo`, `m4_zo`, `snc_zo`, `sn1_zo`, `sn1_fix`, `speechnet_full_fix`, `twostep_*`, `eightstep_native`, `exp5_zo`, `perturb_full_sim*`, `mezo_1step_*`) |
| Single step, memory (stale weights) | `DeeployTest/experiments/deliverable/exp3_ZO/` (`fixture/`, `logs/sim.log`, `results/memory_alloc_deeployStates{,_optimizer}.html`) |
| Single-step latency breakdown (stale weights) | `DeeployTest/experiments/deliverable/exp6_ZO_single_step_latency/` (`fixture/`, `logs/{profiletiling,runner}.log`, `results/lat_fig*.{py,png}`) |
| BP comparison points | `DeeployTest/experiments/deliverable/exp2a_BP_argmax_naccum1/logs/sim.log`, `exp2_BP_argmax_on/results/memory_alloc_deeployStates.html` |
| Round 1, old weights | `DeeployTest/experiments/deliverable/exp5_ZO_round1/` (`extract_zo_weights.py`, `logs/round1_gvsoc_zo.log.gz`, `results/eval_b2_results.json`, `results/carry_zo_b1_fold3.pt`, `plot_loss_BP_vs_ZO.py`) |
| Round 1, correct weights | `DeeployTest/experiments/deliverable/exp18_ZO_round1_correct_weights/` (`scripts/run_round1.sh`, `logs/phase{1..6}_*`, `logs/phase3_gvsoc_zo_round1_KILLED_at_1566.log.gz`, `results/carry_zo_b1_fold3_ref.pt`, `results/eval_b2_results.json`) |
| Cell reference accuracies | `DeeployTest/experiments/deliverable/exp17_BP_round1_correct_weights/results/reference_S01_fold3.txt` |
| Host simulation | the host-simulation code in SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp/ (not part of these repositories), section 8 |

---

**Companion guides.** `00_Overview.md` (entry point, environment, checkpoint rule, experiment index),
`01_BP_Guide.md` (back-propagation), `02_ZO_Guide.md` (float zeroth-order), `03_QZO_Guide.md`
(quantized zeroth-order), `04_FusedQZO_NE16_Guide.md` (fused dual-pass kernels and NE16).
