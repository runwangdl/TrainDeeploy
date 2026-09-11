# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

PLATFORM_NAME = "GAP9"
SIMULATOR = "gvsoc"
DEFAULT_CORES = 8
DEFAULT_L2 = 1024000
DEFAULT_MEM_ALLOC_STRATEGY = "MiniMalloc"
DEFAULT_SEARCH_STRATEGY = "random-max"

L2_SINGLEBUFFER_KERNELS = {
    "Kernels/Integer/MatMul/Regular": [64000, 32000, 16000],
    "Kernels/Integer/Conv/Regular_2D_RQ": [8000, 6000, 4000],
    "Kernels/Integer/Conv/StriddedPadded_2D_RQ": [600],
    "Kernels/Integer/Conv/DW_2D_RQ": [2561],
    "Kernels/Integer/Softmax/Regular": [800, 500, 300],
    "Kernels/Integer/Concat": [32000, 16000, 8000],
    "Kernels/Integer/Hardswish/Regular": [750],
    "Kernels/FP32/GEMM/Regular": [8000],
    "Kernels/FP32/Conv/Regular_2D_NoBias": [6600],
    "Kernels/FP32/Conv/Regular_2D_ZeroValuedBias": [6600],
    "Kernels/FP32/Conv/DW_2D_Bias": [7200],
    "Kernels/FP32/Conv/DW_2D_NoBias": [7200],
    "Kernels/FP32/Conv/DW_2D_ZeroValuedBias": [7200],
    "Kernels/FP32/LayerNorm": [2000],
    "Kernels/FP32/MaxPool/Regular_2D": [2000],
    "Kernels/FP32/MatMul": [2000],
    "Kernels/FP32/ReLU": [2000],
    "Kernels/FP32/Reshape/SkipConnection": [1400],
    "Kernels/FP32/Softmax/Regular": [4000],
    "Kernels/FP32/Transpose": [2000],
    "Kernels/FP32/Mul": [2000],
    "Kernels/Integer/GEMM/Batch_RQ": [20000],
    "Kernels/Integer/MatMul/Batch": [20000],
}

L2_DOUBLEBUFFER_KERNELS = {
    "Kernels/Integer/MatMul/Regular": [64000, 32000, 16000],
    "Kernels/Integer/Conv/Regular_2D_RQ": [8000, 6000, 5000],
    "Kernels/Integer/Conv/DW_2D_RQ": [5121],
    "Kernels/Integer/Softmax/Regular": [1600, 1000, 600],
    "Kernels/Integer/Concat": [64000, 32000, 16000],
    "Kernels/Integer/Hardswish/Regular": [750],
    "Kernels/FP32/GEMM/Regular": [8000],
    "Kernels/FP32/Conv/Regular_2D_NoBias": [8800],
    "Kernels/FP32/Conv/Regular_2D_ZeroValuedBias": [8800],
    "Kernels/FP32/Conv/DW_2D_Bias": [9800],
    "Kernels/FP32/Conv/DW_2D_NoBias": [10000],
    "Kernels/FP32/Conv/DW_2D_ZeroValuedBias": [9800],
    "Kernels/FP32/LayerNorm": [2000],
    "Kernels/FP32/MaxPool/Regular_2D": [5000],
    "Kernels/FP32/MatMul": [5000],
    "Kernels/FP32/ReLU": [20],
    "Kernels/FP32/Reshape/SkipConnection": [2600],
    "Kernels/FP32/Softmax/Regular": [8000],
    "Kernels/FP32/Transpose": [2000],
    "Kernels/FP32/Mul": [2000],
}

L2_SINGLEBUFFER_MODELS = {
    "Models/miniMobileNet": [60000, 12000, 6000, 3000],
    "Models/miniMobileNetv2": [60000, 16000, 12000, 8000],
    "Models/MLPerf/KeywordSpotting": [64000],
    "Models/MLPerf/ImageClassification": [64000],
    "Models/MLPerf/AnomalyDetection": [64000],
}

L2_DOUBLEBUFFER_MODELS = {
    "Models/miniMobileNet": [60000, 24000, 12000, 6000],
    "Models/miniMobileNetv2": [60000, 32000, 24000, 16000],
    "Models/MLPerf/KeywordSpotting": [64000],
    "Models/MLPerf/ImageClassification": [64000],
    "Models/MLPerf/AnomalyDetection": [64000],
}

L3_SINGLEBUFFER_MODELS = {
    "Models/miniMobileNet": [60000, 12000, 6000],
    "Models/miniMobileNetv2": [60000, 16000, 12000, 8000],
    "Models/CCT/FP32/CCT_2_32_32_128": [128000],
}

L3_DOUBLEBUFFER_MODELS = {
    "Models/miniMobileNet": [60000, 24000, 12000, 6000],
    "Models/miniMobileNetv2": [60000, 32000, 24000, 16000],
}

# Training models — GAP9 L2 size is 1 MB (DEFAULT_L2 = 1024000).
# L2 models: Autoencoder and DSCNN fit within 1 MB L2.
L2_SINGLEBUFFER_TRAINING_MODELS = {
    "Models/Training/SimpleMLP/simplemlp_train": [64000],
    "Models/Training/Autoencoder/autoencoder_train": [128000],
    "Models/Training/DSCNN/dscnn_train": [128000, 64000],
    # ResNet8 on-chip. It only fits channels-first: the NHWC path materialises a
    # transposed copy of every conv weight -- 75 of them, 2950 KB in total -- because
    # the pass that eliminates W^T covers Linear layers and not Conv. CHW kernels need
    # no transpose at all, which takes the peak from 1410 KB to 1278 KB and brings it
    # inside L2 with no recompute, no promotion and no double buffering.
    # Measured at one mini-batch: Errors: 0, 43,531,388 cycles, L2_shared 164928 B.
    # 116000, not 122000: CI trains four, and their accumulator buffers leave the L1
    # allocator 118 KB.
    "Models/Training/ResNet8/resnet8_train": [116000],
    # CCT-QLoRA on-chip. The frozen backbone is int8 and its Dequant is folded into
    # the MatMul/Gemm/Conv, so the dequantised weights are never materialised:
    # weight_sram is 48 KB and the arena needs 923 KB, which fits GAP9's real 1.5 MB
    # L2 but not the 1000 KB runner default -- hence the l2 override below.
    # 116000, not the 122000 the L3 entry uses: CI trains 4 mini-batches, whose
    # accumulator buffers leave the L1 allocator 118 KB, and a 122000 arena does not
    # fit alongside them.
    "Models/Training/CCT_QLORA_FT/cct_qlorar1_train": [116000],
}

# L3 models: ResNet8, MobileNetV1, CCT exceed 1 MB L2 — weights spill to L3.
# L1 budgets are the validated best-performing values with the scatter ConvGradX
# binding + -O3 kernels (see TargetLibraries/GAP9/CMakeLists.txt). Per-model
# cc_stack (TRAINING_MODEL_OVERRIDES) keeps arena+cc_stack within the ~127 KB L1
# pool. MobileNetV1 runs channels-first (CHW kernels, no NCHW<->NHWC transpose).
L3_SINGLEBUFFER_TRAINING_MODELS = {
    "Models/Training/ResNet8/resnet8_train": [122000],
    "Models/Training/MobileNetV1/mobilenetv1_train": [116000],
    "Models/Training/CCT/cct_train": [122000],
    # Rank-4 LoRA on CCT-2 at the official spec (mlp_ratio=1), adapters on attention
    # and FFN. 2292 KB peak, 54.5M cycles -- faster than full fine-tuning because the
    # frozen base weights need no weight gradients.
    "Models/Training/CCT_LoRA_R1/cct_lorar1_train": [122000],
    # The same model with its frozen backbone quantised to int8. Exercises the folded
    # Dequant path: without in-kernel dequantisation this model does not fit at all
    # (minimalloc fails), and with it the weights reach the kernels as int8.
    "Models/Training/CCT_QLORA_FT/cct_qlorar1_train": [122000],
    "Models/Training/SleepConViT/sleepconvit_train": [122000],
    "Models/Training/TSDR/tsdr_train": [122000],
    "Models/Training/MCUNet/mcunet_train": [116000],
}

# Gradient-checkpointed training. Each entry replays a solved recompute schedule
# instead of keeping every forward activation live across the backward pass: an
# activation is dropped after its forward use and regenerated just before the
# gradient node reads it, trading cycles for peak memory.
#
# Only schedules verified end to end on gvsoc belong here. A schedule is keyed on
# node names, so one produced from a different graph would replay as the default
# order under a name claiming to be checkpointed; the replay refuses below 90%
# name coverage rather than report that as a pass.
#
#   model                  schedule                       Errors  cycles/step
#   CCT (exact ILP)        recompute_checkmate.json       0       68.63M vs 64.24M
#                                                                 baseline (+6.8%)
L3_RECOMPUTE_TRAINING_MODELS = {
    # Solved by scripts/solve_recompute_schedule.py against the current deployed
    # graph. They cannot be carried across a lowering change: see
    # docs/gradient_checkpointing.md.
    #
    #   model         recomputes  Errors  cycles (1 step)
    #   CCT                    6       0   68,634,810
    #   MobileNetV1           16       0   52,727,331
    "Models/Training/CCT/cct_train": {
        "l1": 122000,
        "schedule": "Tests/Models/Training/CCT/cct_train/recompute_checkmate.json",
    },
    "Models/Training/MobileNetV1/mobilenetv1_train": {
        "l1": 116000,
        "schedule": "Tests/Models/Training/MobileNetV1/mobilenetv1_train/recompute_checkmate.json",
    },
}

# L3 double-buffered training. Only the L3<->L2 hop is double-buffered
# (TrainingDBOnlyL3Tiler) so the L2 staging budget doesn't double.
# ResNet8's strided ConvGrad InPlaceAccumulatorV2 is forced to coeff=1 (SB)
# by TrainingDBOnlyL3Tiler. MobileNetV1: multi-tile Transpose + ConvGradW
# opted out of DB (tilingUtils.DB_OPT_OUT_OPS, same numerical fix as
# Siracusa); its CHW im2col forward conv runs the async DB hop directly
# now that AnydimAsyncDmaTransferAdapter serializes decomposed sub-transfers
# on the shared per-tensor request handle.
L3_DOUBLEBUFFER_TRAINING_MODELS = {
    "Models/Training/CCT/cct_train": [122000],
    "Models/Training/ResNet8/resnet8_train": [122000],
    "Models/Training/MobileNetV1/mobilenetv1_train": [116000],
}

# Training + PromoteTensorsToL2 (singlebuffer). test path ->
# list of (l1, strategy, includeActivations).
# ResNet8 uses "smallest" (cycle-aware promotes 0 tensors for ResNet8 — its
# heuristic finds no positive-cycle-benefit candidate). Both pass thanks to the
# InitNetwork pi_l2_malloc-before-cl_ram_malloc hoist (see codeGenerateTraining
# _hoistL2AllocsBeforeL3): without it, promotion's PROMOTED_POOL_L2 pi_l2_malloc
# interleaved with the FC-delegated cl_ram_malloc loop races the pulp-os L2
# allocator freelist -> FC os_evt_release corruption at init.
L3_SINGLEBUFFER_TRAINING_PROMOTE_MODELS = {
    "Models/Training/CCT/cct_train": [(122000, "smallest", True),],
    "Models/Training/ResNet8/resnet8_train": [(122000, "smallest", True),],
    "Models/Training/MobileNetV1/mobilenetv1_train": [(116000, "smallest", True),],
}

# Training + PromoteTensorsToL2 + double-buffering combined.
# Both CCT and ResNet8 pass thanks to the InitNetwork pi_l2_malloc-before-
# cl_ram_malloc hoist (codeGenerateTraining._hoistL2AllocsBeforeL3), which fixes
# the FC/CC pulp-os L2-allocator freelist race that previously corrupted the FC
# RTOS event list -> os_evt_release deadlock at init (this is why ResNet8
# promote+DB "never completed" and why CCT smallest used to hang). Per-model
# strategy: CCT cycle-aware (~336M/4-step), ResNet8 smallest (~247.6M/4-step;
# cycle-aware promotes 0 for ResNet8). headroom 700000 (set in the test) leaves
# enough L2 for the doubled DB staging buffers.
L3_DOUBLEBUFFER_TRAINING_PROMOTE_MODELS = {
    "Models/Training/CCT/cct_train": [(122000, "cycle-aware", True),],
    "Models/Training/ResNet8/resnet8_train": [(122000, "smallest", True),],
    "Models/Training/MobileNetV1/mobilenetv1_train": [(116000, "smallest", True),],
}

TRAINING_MODEL_OVERRIDES = {
    # Slave stacks live in L1 (SDK default); we just shrink them to 512B. Small
    # L1 stacks are a big win over parking them in L2 (cyc/step, L1 vs L2 below).
    "Models/Training/Autoencoder/autoencoder_train": {
        "slave_stack": 512,  # 0.48M vs 0.78M cyc/step (-38.5%)
    },
    "Models/Training/DSCNN/dscnn_train": {
        "slave_stack": 512,  # 0.80M vs 1.21M cyc/step (-33.7%)
    },
    "Models/Training/ResNet8/resnet8_train": {
        "conv_channels_first": True,  # CHW convs; see the on-chip entry above
        "l2": 1400000,  # arena 1278 KB plus the static section, inside GAP9's 1.5 MB
        "cc_stack": 4096,  # conv-light backward -> small CC stack, frees L1 for arena
        # arena 122000 + cc 4096 + slave 512*8 = 130192 < 131072 -> L1 stacks fit.
        # L1 vs L2 stacks: 47.8M vs 62.9M cyc/step (-23.9%, SB). With the gather
        # ConvGradX kernel + promote+DB the best config is 44.1M/step.
        "slave_stack": 512,
    },
    "Models/Training/MobileNetV1/mobilenetv1_train": {
        "conv_channels_first": True,  # CHW convs; the NHWC-transpose tiling is infeasible
        "cc_stack": 8192,  # -O3 cut the CC-stack need from 16384; frees L1 for a bigger arena
        # 700000 (was 920000): testData is now hex-loaded to L3 (not baked into L2
        # .data, freeing ~432KB), so MNV1 can promote a larger pool (~316KB). 700000
        # keeps it below the runtime L2-staging cliff (DB doubles staging: promote+DB
        # fails ≥~500KB, promote-SB ≥~800KB) -> promote+DB ~-6.8% vs SB.
        "promote_headroom": 700000,
        # arena 116000 + cc 8192 + slave 512*8 = 128288 < 131072 -> L1 stacks fit.
        # L1 vs L2 stacks: 53.1M vs 68.7M cyc/step (-22.7%).
        "slave_stack": 512,
    },
    "Models/Training/CCT/cct_train": {
        "tolerance": 5e-3,
        # cc_stack 4096 (was 8192): with promote_headroom 700000 the CC closure
        # chain no longer overflows at 4096 (the old 4096->os_evt_release deadlock
        # was a tighter-headroom scenario, since fixed). Dropping to 4096 frees the
        # L1 the slave stacks need: arena 122000 + cc 4096 + slave 512*8 = 130192
        # < 131072 -> L1 stacks fit, no arena cut, no tiling penalty. Verified
        # (build memcheck + sim) on SB, DB and promote+DB.
        "cc_stack": 4096,
        # L1 slave stacks: measured cyc/step (N=2) on the CI best (promote+DB):
        #   cc8192 + L2 stacks            87.1M  (previous config)
        #   cc4096 + L1 stacks (this)     66.8M  -> -23.4%
        # Also helps single-buffer (95.8M -> 75.3M). promote+DB+L1 is CCT's best.
        "slave_stack": 512,
    },
    "Models/Training/CCT_LoRA_R1/cct_lorar1_train": {
        "tolerance": 5e-3,
        "cc_stack": 4096,
        "slave_stack": 512,
    },
    "Models/Training/CCT_QLORA_FT/cct_qlorar1_train": {
        "l2": 1572864,  # GAP9's real 1.5 MB; the arena needs 923 KB of it
        # arena 122000 + cc 4096 + slave 512*8 = 130192 < 131072, so the cluster
        # stacks stay in L1. Without these the SDK defaults overflow it and gvsoc
        # exits before producing any output.
        "cc_stack": 4096,
        "slave_stack": 512,
        # int8 weights dequantised in-kernel; the tolerance covers per-tensor
        # symmetric quantisation of the frozen backbone, not a looser kernel.
        "tolerance": 5e-2,
        "cc_stack": 4096,
        "slave_stack": 512,
    },
    "Models/Training/SleepConViT/sleepconvit_train": {
        "tolerance": 5e-3,
        "cc_stack": 4096,  # transformer backward; small CC stack frees L1 for the arena
    },
    "Models/Training/TSDR/tsdr_train": {
        "tolerance": 5e-3,
        "cc_stack": 4096,  # spectrogram transformer; same profile as SleepConViT
        "conv_channels_first": True,  # CHW patch-embed conv (no NCHW<->NHWC transpose)
    },
    "Models/Training/MCUNet/mcunet_train": {
        "tolerance": 5e-3,
        "cc_stack": 8192,  # MnasNet-style; deep DW/PW chain needs a larger CC stack
        "conv_channels_first": True,  # CHW convs; NHWC-transpose tiling is infeasible
    },
}
