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
    "Models/Training/CCT_LoRA/cct_lora_train": [40000],
}

# L3 double-buffered training. Only the L3<->L2 hop is double-buffered
# (TrainingDBOnlyL3Tiler) so the L2 staging budget doesn't double.
# CCT + ResNet8: ResNet8's strided ConvGrad InPlaceAccumulatorV2 is forced to
# coeff=1 (SB) by TrainingDBOnlyL3Tiler, so on GAP9 it takes the blocking
# gap9L3DmaHack path (safe for strided 2D ConvGrad transfers) — no UDMA DB
# deadlock. MobileNetV1: multi-tile Transpose + ConvGradW opted out of DB
# (tilingUtils.DB_OPT_OUT_OPS, same numerical fix as Siracusa) AND its CHW
# im2col forward conv routed through GAP9ClusterBlockingDBTransformer (blocking
# DB hop) so its per-channel strided transfers don't crash the gvsoc UDMA model.
L3_DOUBLEBUFFER_TRAINING_MODELS = {
    "Models/Training/CCT/cct_train": [122000],
    "Models/Training/ResNet8/resnet8_train": [122000],
    "Models/Training/MobileNetV1/mobilenetv1_train": [116000],
}

# Training + PromoteTensorsToL2 (singlebuffer). test path ->
# list of (l1, strategy, includeActivations). CCT only for now.
L3_SINGLEBUFFER_TRAINING_PROMOTE_MODELS = {
    "Models/Training/CCT/cct_train": [(122000, "smallest", True),],
}

# Training + PromoteTensorsToL2 + double-buffering combined. CCT only.
# Strategy is "cycle-aware", NOT "smallest": "smallest" promotes ~138 KB of
# const tensors (vs cycle-aware's ~40 KB), which deepens the init-time setup so
# the CC master stack high-water overflows the razor-thin L1 margin (arena
# 122000 + cc_stack 8192 = 130192, only ~864 B free) -> corrupts the RTOS event
# list -> os_evt_release deadlock at "Initializing TrainingNetwork" (CI job
# never completes). cycle-aware promotes fewer consts -> init fits -> CCT
# promote+DB passes at ~336M/4-step.
# ResNet8 promote+DB on GAP9 did not complete within a 25-min build+sim budget
# (heavier promote+DB codegen, possible hang) — left out pending investigation.
# Plain ResNet8 DB (L3_DOUBLEBUFFER_TRAINING_MODELS) passes at ~196M/step.
L3_DOUBLEBUFFER_TRAINING_PROMOTE_MODELS = {
    "Models/Training/CCT/cct_train": [(122000, "cycle-aware", True),],
}

TRAINING_MODEL_OVERRIDES = {
    "Models/Training/ResNet8/resnet8_train": {
        "cc_stack": 4096,  # conv-light backward -> small CC stack, frees L1 for arena
    },
    "Models/Training/MobileNetV1/mobilenetv1_train": {
        "conv_channels_first": True,  # CHW convs; the NHWC-transpose tiling is infeasible
        "cc_stack": 8192,  # -O3 cut the CC-stack need from 16384; frees L1 for a bigger arena
    },
    "Models/Training/CCT/cct_train": {
        "num_data_inputs": 1,
        "tolerance": 5e-3,
        # 8192 (not 4096): promote+DB deepens the CC closure chain and DB doubles
        # L1 arena pressure. cc_stack=4096 -> CC master stack overflows into the
        # RTOS event list -> os_evt_release deadlock at init. cc_stack=16384 ->
        # arena(122000)+stack > TCDM(131072) -> overlap -> wild-pointer crash.
        # 8192: 122000+8192=130192 < 131072 -> fits AND stack deep enough.
        "cc_stack": 8192,
    },
    "Models/Training/CCT_LoRA/cct_lora_train": {
        "num_data_inputs": 1,
    },
}
