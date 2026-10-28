# 06 — Further optimisation of the fused QZO(+ECO) step, BP fixes, and the board rounds

Written 2026-10-28. Part of the handover set indexed by `TrainDeeploy/HandoverDocuments/00_Overview.md`.
Paths are relative to `/Users/qiwenwu/ETH/`; in-container paths appear only inside command blocks.
This guide continues `05_Fusion_Full_Network_Guide.md`. It is the entry point of the branch
`feat/further_optimisation`, which is `feat/memory_opt` (PR #83, exp9) plus everything that followed it.

**Summary.** After exp9 (guide 05 §4.8) the fused QZO+ECO step on GAP9 + NE16 was 6.56 M cycles per update step.
This guide documents what came next:

- **exp11** — a per-block breakdown of blocks 1–4 on GAP9 with a new control kernel, `PerturbConv4x2`
  (one pass, perturbation inside the kernel). It separates the gain of the dual-pass fusion from the gain of the
  kernel rewrite and explains why NE16 wins on blocks 1–2 but loses on blocks 3–4 (§2).
- **exp21c** — three optimisations of the NE16 preparation (B, H, S): −0.27 M cycles per update step, bit-exact,
  each with an opt-out switch (§3).
- **exp21e** — two Siracusa BP bugs found by bisection and fixed (ConvGradW tile padding; a frozen-BN per-core loop),
  plus opt-in strict-fp32 builds that make complete BP rounds bit-exact against a C-order host reference (§4).
- **Merge gates** of the combined tree `92027907` on GAP9 and Siracusa (§5).
- **Measurements on this tree** — the speed-up of the fusion alone (exp21m), static peak memory of BP / ZO /
  QZO-ECO (exp21n), and four incremental rounds of every recipe on the GAP9 board (Board_exp) (§6).

**Headline numbers** (GAP9 @ 370 MHz, one update step = 4 training-graph runs + 1 update-graph run, QZO-ECO with SR):

| build | cycles per step (GVSoC) | speed-up |
|---|--:|--:|
| unfused, pulp-nn | 13,763,138 | |
| unfused control (FlatConv block 0 + PerturbConv4x2 blocks 1–4) | 11,232,224 | 1.225× (rewrite + in-kernel perturbation) |
| fused, cluster | 9,129,656 | **1.230× (fusion alone)** |
| fused, NE16 blocks 0–2 (board configuration) | 6,850,086 | **1.333× (accelerator)** |

The chain multiplies to 2.0×, and the board confirms it: QZO-ECO round of 1890 steps 71.2 s unfused vs **35.7 s**
fused NE16 (6.99 M cycles per step, 2 % above GVSoC), identical weight dumps in all four rounds.

---

## 1. Branch, commits and how they fit together

`feat/further_optimisation` starts from Run's `silentwear-zo/gap9-on-devel` @ `fe1ffa28`. Commits in order:

| commit | content | section |
|---|---|---|
| `9f45baa3`, `f535fb9c` | exp9 `FuseFrozenBNTailPass` + guide 05 update (**= PR #83**) | guide 05 §4.8 |
| `9a1050f4` | exp11: `Conv4x2` step-5 optimisations, new `PerturbConv4x2`, probes, per-block GAP9 breakdown | §2 |
| `eb1fe3ba` | merge exp11 into exp9 (only `Deeploy/Targets/{GAP9,PULPOpen}/Platform.py` overlap; both mappers kept) | |
| `31bb400e` | exp9 memsim: single-step GAP9 L2/L3 memory of all recipes at board settings | §6.2 |
| `bb7deadb` | exp21 checkpoint: results of exp21a–f; default-off probes `NE16DUAL_PROBE_WALL`, `ARGMAX_MASK_SIG` | §7 |
| `b2cc3c75`, `e7ed4d06`, `8f9f6fb4`, `5f218820` | exp21c opts B, H, S and their results | §3 |
| `edc15ec1`, `4f409ce1` | BP regression fixes A and B | §4 |
| `cbf97226`, `8c31aa89`, `a693bbf6`, `5630361d`, `c9dbae1a` | strict-fp32 opt-ins, C-order BP and out-of-place ZO fixtures, exp21e/fix evidence | §4 |
| `92027907` | merge of the exp21c branch (the tree every number in §5–§6 ran on) | |
| this commit | guide 06, overview, evidence of exp21m / exp21n / merge gates / Board_exp | §5–§6 |

Every optimisation is **on by default** and **bit-exact**; each has a CMake or environment switch to turn it off for
A/B runs. No default build changes numerically: GAP9 CI hashes (`ZO_EXPECT_WSUM`) are unchanged.

---

## 2. exp11 — per-block breakdown of blocks 1–4 and the single-pass control kernel

**Question.** Guide 05 measured fused vs unfused per block, but the fused kernel differs from pulp-nn in two ways at
once: a different conv loop (Conv4x2-style), and the two ε-passes sharing one perturbation and one weight read. exp11
separates them.

**Code** (`9a1050f4`):
- `TargetLibraries/PULPOpen/src/Conv4x2.c`: the step-5 optimisations of `DualConv2In` (word-wise border gather,
  pixel/channel split) moved into the single-pass control kernel; the old step-2a kernel kept under
  `-D CONV4X2_STEP2A`. Selected with `DEEPLOY_RQCONV_KERNEL=4x2` (`Deeploy/Targets/PULPOpen/Platform.py:104`,
  `Deeploy/Targets/GAP9/Platform.py:112`).
- `TargetLibraries/PULPOpen/src/PerturbConv4x2.c` (new): **one** pass of the QZO conv with the perturbation inside the
  kernel (`dc_perturb1_i8_chw`, `dc_perturb1_i32`, single-sign versions of the fused kernel's helpers) followed by the
  Conv4x2 loop. In the graph it is `RequantizedDualConv` with `single_pass = 1`
  (`Deeploy/Targets/PULPOpen/Parsers.py:871`, template `Templates/PerturbConv4x2Template.py`, bindings
  `Bindings.py:701`, mappers on PULPOpen and GAP9). Fixtures `QZO_PerturbConv_B1..4` (+ pass) and
  `QZO_PerturbConvM_B1..4` (− pass, `force_sign = 0`) from `build_perturb_conv_fixture.py` in
  `DeeployTest/Tests/Kernels/Integer/`.
- Probes (default off): `DUALCONV2IN_PROBE` (per-core phase work cycles), `NE16DUAL_PROBE` (NE16 prep phases and the
  `NE16DualConv2in` column / wait / rest split). `NE16WeightEncode.c` and `PerturbConv4x2.c` added to the GAP9 `-O3`
  list.

**Results** (GAP9 GVSoC, state `fe1ffa2`, `--l1 86000 --l2 1500000`, all exact). Speed-up of perturb + encode +
transposes + conv vs 2 × pulp-nn, blocks 1 / 2 / 3 / 4:

| variant | block 1 | block 2 | block 3 | block 4 |
|---|--:|--:|--:|--:|
| 2 × Conv4x2 | 1.42 | 1.30 | 1.02 | 1.17 |
| 2 × PerturbConv4x2 (in-kernel perturbation) | 1.51 | 1.60 | 1.90 | 3.05 |
| fused cluster | 1.57 | 1.70 | 2.21 | 3.87 |
| fused NE16 | 2.76 | 2.10 | 1.18 | 1.08 |

The dual-pass fusion alone (PerturbConv4x2 → fused cluster) is worth 1.04–1.27× per block: the perturbation gets
1.27–1.33× (weight read and random draw once), the conv saves the per-invocation fork / barrier / DMA.

**Why NE16 loses on blocks 3–4** (`results/ne16_vs_cluster_blocks34.txt`): the NE16 preparation (weight encode,
a second perturbation of W for the bias row sums) scales with the weight size (Co × K); the NE16 gain scales with
output pixels × weight. Blocks 3–4 have the largest weights and the fewest pixels (40 / 10 per pass), so the prep is
paid in full and amortised over almost no work. Balance (conv saving − NE16-only prep): block 1 +187 k, block 2
+28 k, block 3 −40 k, block 4 −104 k cycles. This analysis motivated exp21c.

Evidence: `TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp11_b1to4_breakdown/` (`results/breakdown.md`,
`results/ne16_vs_cluster_blocks34.txt`, `scripts/run_{breakdown,perturbconv,ne16_probe,encode_o3}.sh`, `logs/`).

---

## 3. exp21c — NE16 dual-prep optimisations B, H, S

All three live in `TargetLibraries/GAP9/src/NE16DualPrep.c`, `NE16WeightEncode.c` and `Deeploy/Targets/NE16/DualPrep.py`.
Opt-outs: `-D NE16DUAL_NOOPT_B=ON`, `NE16DUAL_NOOPT_H=ON`, `NE16DUAL_NOOPT_S=ON` (`TargetLibraries/GAP9/CMakeLists.txt:117`).

- **B** (`b2cc3c75`) — `NE16DualBias` skips the second perturbation of W and the row sums when `input_offset == 0`.
  The signed-input correction is `add2 = add − input_offset · ΣW · mul`; for blocks 1–4 of the fused network the
  stacked input comes after a ReLU, so `input_offset = 0` (`ZODualPassFusion.py:232`) and the term is exactly 0.
  Block 0 has offset 128 and keeps the full path.
- **H** (`e7ed4d06`) — hoist the prep to **once per update step**. z changes per update step, not per window
  (the harness sets `seed_base` outside the accumulation loop, `DeeployTest/Platforms/GAP9/src/deeploymezotest.c`),
  and W / bias change only in the update graph. So `NE16DualPerturbEncode` and `NE16DualBias` produce identical
  outputs in the n_accum windows of one step. Each node gets a static L2 cache (header + output, emitted by
  `DualPrep.py:_hoistPrepCache`), keyed on `(seed_w, seed_b, tile_xor, eps bits, zo_weight_version, bytes)`
  (`NE16DualPrep.c:98–112`). The runtime global `zo_weight_version` (`TargetLibraries/PULPOpen/src/ZORuntime.c:16`) is
  bumped by the GAP9 MeZO harness after every `zo_update` and after `ZO_RESUME` (`deeploymezotest.c:615`, `:797`).
  **Contract:** any code that changes weights outside the update graph must bump `zo_weight_version`.
- **S** (`8f9f6fb4`) — word-wise weight encoder `NE16WeightEncode_i8_u8_t1` (`NE16WeightEncode.c:129`): four word
  loads per 16-lane subtile, XOR `0x80808080`, two 8×8 bit transposes, byte interleave, four word stores; partial or
  unaligned cases fall back to the bit loop. Same bytes as the bit loop (host check on 5.65 M random elements and
  every value in every lane).

**Results** (whole network, one update step, board settings; `impl/SUMMARY.txt`):

| config | NE16 blocks | training cycles | vs base |
|---|---|--:|--:|
| base | 0,1,2 | 6,558,466 | 1.000× |
| B | 0,1,2 | 6,483,614 | 1.012× |
| B+H | 0,1,2 | 6,318,369 | 1.038× |
| **B+H+S** | **0,1,2** | **6,292,263** | **1.042×** |
| base | 0,1,2,3 | 6,719,509 | 0.976× |
| B+H+S | 0,1,2,3 | 6,234,177 | 1.052× |
| B+H+S | 0,1,2,3,4 | 6,224,111 | 1.054× |

- Prep nodes per step (blocks 0–2): 348,676 → 83,468 cycles. Encode phase 3.7–4.4× faster per block.
- Bit-exact: every block fixture `Errors 0`; network `Errors 0/8` with `[WSUM s=0] 0xfa1f7dd2` (= the checkpoint);
  3 steps × n_accum 4 and 2 × 2 multi-step runs: losses, WSUM and weight dumps identical to the unoptimised build.
- Memory: training L2 arena unchanged (265,612 B); cache statics outside the arena +8,936 B (blocks 0–2).
- Run-to-run noise from the tiler's random-max placement is about ±0.15 %.

**Block 3 on NE16.** After B+H+S, block 3 on NE16 is 58 k cycles per step faster (−0.85 %); before them it was
slower. exp21c recommended 0,1,2,3. **The board recipe stays at `--zo-fused-dual-ne16 0,1,2`**: all board rounds,
memory plans and paper numbers were measured with it, and the gain is below 1 %. Switching is safe (bit-identical).
Block 4 on NE16 is a tie within noise.

Not implemented: **H+** (hoisting the remaining per-window perturbations of the cluster and fp32 nodes) — at most
~0.11 M cycles per step (~1.7 %), in shared PULPOpen kernels with a new interface.

Evidence: `TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21c_Fusion_NE16_Encode_Opt/`
(`impl/SUMMARY.txt`, `impl/results/`, `impl/logs/`, `impl/scripts/`, `impl/fixtures/QZO_DualConv2in_NE16P_off0_B<i>`;
the analysis that preceded the implementation in `results/`, `logs/`).

---

## 4. exp21e — BP regression on Siracusa, and bit-exact BP rounds

**Symptom.** A 2-step BP run on Siracusa no longer matched exp17 (forward 4: 0.229024 vs 0.228693 and worse,
depending on stale L1). GAP9 BP at the board settings was **not** affected (exact, unchanged by the fixes).

**Bisection** over 85 first-parent commits found two independent bugs:

- **A — ConvGradW tiling** (`edc15ec1`, introduced by the devel merge `ce75da00` / `599f3c1e`).
  `computeInputTileFromGradOutTile` gave a dY tile the full ONNX pad only if it was a boundary tile of dY, else 0.
  In block 0 (k = 4, pad 2/2, W_out = 701, dY tiles 175, 175, 175, 175, 1) tile 3 needs `X[523..701)` of a 700-wide
  X, i.e. pad_right 1, but got pad 0. pulp-trainlib's im2row then built 174 columns while the GEMM read 175, so a
  stale column entered dW (blocks_0_0 / blocks_1_0 gradients 12 % / 2 % wrong). Fix: per-tile pad = exactly what the
  clamp cut off (`Deeploy/Targets/PULPOpen/TileConstraints/ConvGradConstraint.py`, around line 1581). Original rule
  kept as a comment.
- **B — frozen BN saved statistics** (`4f409ce1`, introduced by `da8cb63f`). The element-parallel frozen path wrote
  `saved_mean` / `saved_inv_std` in `for (c = core_id; c < C; c += NUM_CORES)`. Siracusa's LLVM lowers it to a
  hardware loop with trip count `(C − core_id) >> 3` instead of the ceiling, so cores 1–7 skip their last channel and
  BatchNormGrad reads stale L1. BP only (ZO / QZO never read these stats). Fix: contiguous per-core chunk
  (`TargetLibraries/PULPOpen/src/BatchNorm.c`, around line 66). Same miscompile as guide 05 pitfall 4: **do not use
  core-strided loops in new PULP kernels.**

Neither fix costs measurable cycles. The BP single-step CI gate could not have caught either bug: it checks only the
pre-update loss.

**Bit-exact BP reference.** A C-order host emulator (`fix/scripts/bp_emu.py`: device reduction order including the
tile-wise ConvGradW partial sums, one rounding per op) plus the device's own `expf` / `logf` transliterated from the
image (`fix/scripts/libm_emu.c`; picolibc for Siracusa, newlib for GAP9), against an opt-in strict-fp32 device build
(`-D DEEPLOY_STRICT_FP32=ON DEEPLOY_STRICT_FP32_ALL=ON`, `cbf97226` / `8c31aa89`; default off, +5.3 % / +6.8 % BP
cycles on Siracusa / GAP9):

- Siracusa, complete BP round (540 steps, 2160 forwards): `Errors: 0 out of 2160`, all loss bits and all 15,489
  final weight words equal.
- GAP9, board settings, complete round: same, 2160/2160 and 15,489/15,489.
- Default (fast-math) build vs the PyTorch fixture: `Errors 0/2160`, max 9.56e-4.

**ZO reference drift.** The old ZO host reference perturbed in place and drifted linearly over a round (max 9.7e-4).
With the out-of-place reference (Onnx4Deeploy branch `fix/zo_ref_out_of_place`, `71dacbc`) a complete Siracusa ZO
round (1620 steps) gives `Errors 0/12960`, max 6.2e-6, flat.

Fixtures (`a693bbf6`): `DeeployTest/Tests/Models/Training/speechnet_{train,optimizer}_fullfrozen_b1_fold3_corder`,
`SpeechNet_Test/speechnet_{train,optimizer}_BP_round1_corder_gap9`, `SpeechNet_Test/speechnet_{train,update}_ZO_round1_oop`
(+ `SpeechNet_Test/exports/zo_round1_oop_120ep`).

Evidence: `TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift/fix/`
(`SUMMARY.txt`, `logs/`, `results/`, `scripts/`, including `scripts/run_gates_siracusa.sh`).

---

## 5. Merge gates of `92027907`

Scripts and logs: `TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/merge_gates_2026-10-27/`.

| gate | result |
|---|---|
| GAP9 fused QZO+ECO step, NE16 0,1,2 (`scripts/run_gap9_gates.sh`) | `Errors 0/8`, train 6,290,864 + update 559,222 = **6,850,086** cycles, `[WSUM s=0] 0xfa1f7dd2` |
| GAP9 fused QZO+ECO step, NE16 0,1,2,3 | `Errors 0/8`, 6,234,218 + 558,528 = 6,792,746, same WSUM |
| GAP9 CI `-m "gap9_tiled and speechnet_gap9"` | 6 passed (exact `ZO_EXPECT_WSUM`) |
| Siracusa exp21e gates (zo / qzo single, full-net QZO+ECO fused / unfused, block-0 kernels, BP single) | `gate_merged_*` logs identical to the exp21e branch's `gate_fixed_*` |
| Siracusa CI `-k "siracusa and (SpeechNet or speechnet)"` | 13 passed |
| Siracusa CI `-m zo_training` | 2 passed |

The fork's GitHub CI cannot run the GAP9 jobs (private image; fails at "Initialize containers"), so these local
gates are the check.

---

## 6. Measurements on this tree

### 6.1 Speed-up of the fusion alone (exp21m)

`exp21_remaining_exp/exp21m_fusion_speedup_cluster/`. Three cluster-only builds of one QZO-ECO update step at the
board settings (`--l1 114000 --l2 1340000`, frozen BN, `GAP9_NETWORK_O3`, ECO + SR); NE16 off.

| build | block 0 | blocks 1–4 | step cycles |
|---|---|---|--:|
| unfused, vendor | `pulp_nn_conv_i8_i8_i8` | `pulp_nn_conv_i8_i8_i8` | 13,763,138 |
| unfused control | `PULP_FlatConv2d_i8_i8_i8_HWC` (`DEEPLOY_RQCONV_FLAT=1`) | `PULP_PerturbConv4x2_i8_i8_i8_HWC` | 11,232,224 |
| fused, cluster | `PULP_DualConv2d_i8_i8_i8_HWC` | `PULP_DualConv2d_2in_i8_i8_i8_HWC` | 9,129,656 |

All three `Errors 0/8`, `[WSUM s=0] 0xfa1f7dd2`. Fusion alone (control → fused) = **1.230×** per step. The control
graph comes from `scripts/build_ctrl_fixture.py` (exp11's PerturbConv rewrite on blocks 1–4 of the full graph).
Block 0 has no in-kernel-perturbation control; its standalone perturbation is 2.5 % of block 0's fusion gain (exp5a),
so the bias is small and against the control. Rerun: `bash scripts/run.sh` in `deeploy_gap9`.

### 6.2 Memory

- **exp9 memsim** (`dual_pass_fusion/exp9_memory_optim/memsim/summary.txt`, `31bb400e`): single step per recipe with
  Run's board settings, `--plotMemAlloc`. Fused QZO+ECO with NE16 0–2: L2 arena 265,612 B + constants + 61,956 B ECO
  momentum = 330,912 B; ZO 792,064 B; BP 1,024,552 B L2 + 1,006,480 B in L3 (Run's L3 + promotion build).
- **exp21n peak memory** (`exp21_remaining_exp/exp21n_peak_memory/`): static plans of the round-1 Board_exp fixtures
  with every tensor in L2, no promotion, no double buffering, each recipe's board `--l1`; read from the generated C
  (`scripts/analyze.py` asserts the two arenas are the only allocations).

  | B | BP (fp32) | ZO (fp32) | QZO-ECO (int8, fused NE16) |
  |---|--:|--:|--:|
  | L2 total (arena + constants + ECO momentum) | 1,481,989 | 792,710 | 341,175 |
  | L1 arena (tile buffer) | 121,984 | 113,844 | 113,728 |
  | **peak** | **1,603,973** | **906,554** | **454,903** |

  BP / QZO-ECO = 3.5× peak. BP's 1.48 MB L2 arena does not fit next to its program image on the board
  (Run: "1,478,772 > 1,377,216"), which is why the deployed BP spills to HyperRAM. The L2 arenas of the ZO recipes
  include the window and the weights (graph inputs); only the ECO momentum is outside.

### 6.3 Board rounds (Board_exp)

`TrainDeeploy/DeeployTest/experiments/Board_exp/`. GAP9 EVK @ 370 MHz / 0.8 V, tree `92027907`. S01, fold 3,
vocalized, seed 42, 4 incremental rounds; round r+1 starts from round r's **board** weights (`[WDUMP]` →
`ZOfam/scripts/carry_weights.py`, Run's script unchanged). Settings from the exp21l selection rule: BP lr 3e-5 ×
40 epochs (540 steps); ZO and QZO-ECO lr 3e-6 × 140 epochs (1890 steps); QZO lr 1e-5 × 140 epochs.

| recipe | build | cycles / step | time / round | energy / round (Run's PPK2 power) | next-batch acc., 4-round mean |
|---|---|--:|--:|--:|--:|
| BP | L3 + double buffering + promotion (Run's flags) | 82.4 M | 120.6 s | 8.89 J | 84.86 % |
| ZO fp32 | unfused cluster | 27.5 M | 140.4 s | 11.28 J | 86.81 % |
| QZO | fused, NE16 0–2 | 6.70 M | 34.2 s | 2.55 J | 85.28 % |
| QZO-ECO | fused, NE16 0–2 | 6.99 M | **35.7 s** | **2.64 J** | **87.50 %** |
| QZO-ECO | unfused cluster | 13.8–14.0 M | 71.2 s | (5.28 J) | 87.50 % |

- Power was measured once per recipe by Run (2026-10-15) on the same recipes; energy = power × board time. For the
  current QZO-ECO build (with the BN-tail fold and exp21c) it is an estimate.
- **Fused NE16 vs unfused on silicon:** identical weight dumps and WSUM in all 4 rounds; 2.0× faster. This closes
  guide 05 §9 items 3 (exp9 on the board) and 8 (a full fused round on the board).
- BP and ZO reproduce the host chain exactly in every round. QZO / QZO-ECO are not bit-identical to the host after
  1890 integer steps (expected, guide 03): within 0.6 / 1.1 pp on the 4-round mean.
- Accuracy = host evaluation of the board-dumped weights (`ZOfam/scripts/eval_round.py`, `BP/scripts/eval_round.py`,
  container `agitated_hugle`).

**Layout and rerun.**
- `scripts/board_run.sh`: build + readfs pad + UART capture (`/dev/ttyUSB2`) + gapy flash / run (replaces the
  missing `gap9_board_serial_run.sh`).
- `BP/scripts/{export_bp_round.sh, run_chain.sh, eval_round.py}`;
  `ZOfam/scripts/{export_round.sh, run_chain.sh <zo|qzo_fused|eco_fused|eco_unfused>, eval_round.py, carry_weights.py, check_state_map.py}`.
- Raw UART logs `runs/<arm>_r<r>/uart.log`; accuracies `BP/results/round*.json`, `ZOfam/results/<arm>/round*.json`.
- Board access from macOS (README_GAP9 path): host USB/IP server (`pyusbip`, 127.0.0.1:3240, needs brew `libusb`) →
  `usbip-devmgr` container attaches the FTDI `0403:6011` into the Docker VM → privileged board container
  `deeploy_gap9_board` from image `deeploy:gap9-openocd` (`ghcr.io/runwangdl/deeploy:gap9` + openocd `1449af5bd`
  built with the SDK's `make openocd.all`).
- Fixtures and exports (2 MB `inputs.npz` each) are not committed; `export_round.sh` / `export_bp_round.sh`
  regenerate them.

---

## 7. Other exp21 results on this branch (checkpoint `bb7deadb`)

| folder (`exp21_remaining_exp/`) | goal |
|---|---|
| `exp21a_BP_L2L3Mem` | BP memory with L2 only vs L2 + L3: single-level joint peak 1,478,772 B (cross-check of exp21n) |
| `exp21b_QZO_ECO_Abalation` | QZO error-feedback ablation: optimizer (SGD / SGDM / Adam) × EF form (none / carry / ECO memory-free / exact) × rounding (RTN / SR), 4 subjects |
| `exp21c_Fusion_NE16_Encode_Opt` | §3 |
| `exp21d_70%data` | host: 70 % instead of 30 % of each batch at the same number of update steps (helps BP only) |
| `exp21e_numerical_drift` | §4 |
| `exp21f_BP_ep` | host: BP epoch sweep of the deployed recipe |

The later host studies that fixed the fine-tuning settings (exp21g–l: loss curves, lr × epoch sweeps, selection
rule) and their plots are not part of this branch.

---

## 8. Pitfalls and lessons

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | A gain estimate for hoisting (H) was too optimistic | a cache hit is not free: node fork, tiled DMA of W in / output out, the copy (1.2–10.5 k per node) | measure in the network, never in kernel fixtures (fixtures run the graph once, H is always a miss) |
| 2 | Stale-cache bugs are invisible in a one-step fixture | staleness appears from update step 2 | multi-step runs (3 × 4, 2 × 2) with weight dumps |
| 3 | Same code, different cycles (±0.15 %) | the tiler's random-max placement | compare means or many nodes, not one `Runtime` |
| 4 | BP wrong only on Siracusa | core-strided loop miscompiled by Siracusa LLVM (hardware-loop trip count) | contiguous per-core chunks (also guide 05 pitfall 4) |
| 5 | BP single-step CI passed with wrong gradients | the gate checks only the pre-update loss | 2-step oracle with weights; add one to CI |
| 6 | A non-boundary conv tile can still need padding | a window can reach past X even when the tile is not at the dY border | pad = what the clamp cut off |
| 7 | Leftover `TEST_GAP9_W_NE16/<fixture>/TrainingNetwork.c` misread as another run's code | a later run with the same fixture overwrote it | read the generation arguments in that run's log |

---

## 9. Open items

1. **Stack depth** of `PULP_FrozenBNPoolReluQuantHWC_i8_i8` (exp9) and of `PerturbConv4x2` was never measured with
   `silentwear-zo/scripts/stack_depth.py`; the board runs were fine, but measure before shrinking stacks.
2. **Block 3 on NE16** (−0.85 %, bit-identical) is ready to switch on if the board numbers are re-measured.
3. **H+**: remaining per-window perturbations (≤ ~1.7 %), see §3.
4. Guide 05 §9 items 1 (fold frozen BN into the conv requant; needs a reference decision), 2 (the ~0.3 M cycles
   outside the nodes), 5 (L3 → L2 window copy), 6 (block-0 NE16 double buffering) remain open.
5. Host-vs-device bias semantics: the device keeps conv biases in RequantShift-add units (1/mul of the host grid) and
   updates them there, so the host's bias step differs from the device's. Small effect on accuracy (§6.3), but a
   fidelity difference to check. `Board_exp/ZOfam/scripts/eval_round.py` evaluates the device's exact bias values.
6. Add a BP 2-step-with-weights gate to CI (pitfall 5).
