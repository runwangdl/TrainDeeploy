# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

PLATFORM_NAME = "Siracusa"
SIMULATOR = "gvsoc"
DEFAULT_CORES = 8
DEFAULT_L2 = 1024000
DEFAULT_MEM_ALLOC_STRATEGY = "MiniMalloc"
DEFAULT_SEARCH_STRATEGY = "random-max"

L2_SINGLEBUFFER_KERNELS = {
    "Kernels/FP32/ReLU": [2000],
    "Kernels/FP32/Softmax/Regular": [4000],
    "Kernels/FP32/Add/Large": [220000],
    "Kernels/FP32/Conv/DW_2D_Bias": [7200],
    "Kernels/FP32/Conv/DW_2D_NoBias": [7200],
    "Kernels/FP32/Conv/DW_2D_ZeroValuedBias": [7200],
    "Kernels/FP32/Conv/Regular_2D_Bias": [6600],
    "Kernels/FP32/Conv/Regular_2D_NoBias": [1600],
    "Kernels/FP32/Conv/Regular_2D_ZeroValuedBias": [6600],
    # ZO prototype: SpeechNet block-0 Conv + its two PerturbRademacher nodes, lifted
    # verbatim out of the float-ZO training graph. L1 = 128000 is the ZO runner's own
    # setting, which reproduces the ZO network's tiling for this node exactly (single
    # buffer only -- the double-buffer dict is deliberately left alone). -- QW
    "Kernels/FP32/ZO_Conv": [128000],
    "Kernels/FP32/ZO_Conv_Block1": [128000],
    # QZO prototype: block-0 QUANTIZED conv (Conv + RequantShift -> fused RequantizedConv) with
    # its two RQSPerturbRademacher nodes, lifted verbatim from speechnet_qzo19_train. -- QW
    "Kernels/Integer/QZO_Conv": [128000],
    # QW exp5: the FUSED dual-pass version of QZO_Conv (one node, both eps signs, 2*ch_out output). -- QW
    "Kernels/Integer/QZO_DualConv": [128000],
    "Kernels/FP32/GEMM/Regular": [8000],
    "Kernels/FP32/MatMul": [2000],
    "Kernels/FP32/MaxPool/Regular_2D": [2000],
    "Kernels/FP32/Mul": [2000],
    "Kernels/FP32/LayerNorm": [2000],
    "Kernels/FP32/ReduceMean/KeepDims/Add_ReduceMean": [8000],
    "Kernels/FP32/ReduceMean/KeepDims/Add_ReduceMean_Add": [8000],
    "Kernels/FP32/ReduceMean/KeepDims/AllAxes": [50000],
    "Kernels/FP32/ReduceMean/KeepDims/Axes1_2_3": [50000],
    "Kernels/FP32/ReduceMean/KeepDims/Axes1_3": [5000, 50000],
    "Kernels/FP32/ReduceMean/KeepDims/Axes2_1": [6200, 50000],
    "Kernels/FP32/ReduceMean/KeepDims/Axis0": [8400, 50000],
    "Kernels/FP32/ReduceMean/KeepDims/Axis2": [8400, 50000],
    "Kernels/FP32/ReduceMean/KeepDims/ReduceMean_Add": [8000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Add_ReduceMean": [8000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Add_ReduceMean_Add": [8000],
    "Kernels/FP32/ReduceMean/NoKeepDims/AllAxes": [50000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Axes1_2_3": [50000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Axes1_3": [5000, 50000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Axes2_1": [6200, 50000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Axis0": [8400, 50000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Axis2": [8400, 50000],
    "Kernels/FP32/ReduceMean/NoKeepDims/ReduceMean_Add": [8000],
    "Kernels/FP32/Reshape/SkipConnection": [1400],
    "Kernels/FP32/Transpose": [2000],
    "Kernels/Integer/Hardswish/Regular": [750],
    "Kernels/Integer/Softmax/Regular": [800, 500, 300],
    "Kernels/Integer/Concat": [32000, 16000, 8000],
    "Kernels/Integer/MatMul/Batch": [20000],
    "Kernels/Integer/MatMul/Regular": [64000, 32000, 16000],
    "Kernels/Integer/RMSNorm": [2048, 1024, 512],
    "Kernels/Integer/Conv/Regular_2D_RQ": [8000, 6000, 4000],
    "Kernels/Integer/Conv/DW_2D_RQ": [2561],
    "Kernels/Integer/Conv/StriddedPadded_2D_RQ": [600],
    "Kernels/Integer/GEMM/Batch_RQ": [20000],
    "Kernels/Integer/Hardswish/Regular_RQ": [750],
}

# Per-kernel extra generation arguments (same shape as TRAINING_MODEL_OVERRIDES below):
# test path -> list of flags appended to the generator's command line.
#
# ZO_Conv_Block1: SpeechNet block-1's conv bias is |b| < 1e-5, and testMVP.py types inputs by
# VALUE (`minimalFloatType` walks bfloat16 -> float16 -> float32), so it is inferred as bfloat16
# and PerturbRademacher -- which is float32-only -- then has no binding and parsing fails. The
# ONNX declares FLOAT, but 32 Integer kernel tests declare FLOAT while holding int8/int64 data,
# so the declared type can't simply be trusted here the way testMVPTraining.py trusts it. Pin
# the one tensor instead. -- QW
# Per-kernel extra CMake defines (NAME=VALUE, same form as the runner's -D).
#
# The ZO_Conv* fixtures are lifted out of the float-ZO training graph, so they are held to the
# tolerance that flow uses: ZO_TOLERANCE_ABS / TRAINING_TOLERANCE_ABS default to 1e-3
# (testUtils/codeGenerateTraining.py:191,229). deeploytest.c's own OUTPUT_TOL default is 1e-4,
# which these fixtures inherit only by living under Tests/Kernels/.
#
# 1e-4 is the wrong bar for block 0 specifically: its conv output reaches |y| ~ 2615, and the
# comparison is ABSOLUTE. On a clean tree 10 of 78512 elements miss by exactly 1.22e-4 -- one fp32
# ulp in the [1024, 2048) binade -- i.e. the default sits just *below* one ulp. Which elements land
# on which side is decided by clang's -ffast-math contraction, not by correctness. 1e-3 clears the
# observed max by 8.2x and is still ~4e-7 RELATIVE at that magnitude, far stricter than 1e-3 is on
# the scalar losses the ZO harness applies it to.
#
# Block 1 passes at the 1e-4 default on its own (outputs max at ~28, an ulp is ~2e-6); it is pinned
# to the same 1e-3 so the two experiments are built identically and the comparison has no
# confound. -- QW
KERNEL_CMAKE_ARGS = {
    "Kernels/FP32/ZO_Conv": ["OUTPUT_TOL=1e-3"],
    "Kernels/FP32/ZO_Conv_Block1": ["OUTPUT_TOL=1e-3"],
}

KERNEL_GEN_ARGS = {
    # QZO_Conv: the requant bias is int32 but every entry is 32768, which value inference types as
    # uint16_t -- and the RQSPerturbRademacher_i32 / RequantizedConv bindings want int32_t. Pin it.
    # Output is int8, compared EXACTLY by deeploytest.c's integer path: no OUTPUT_TOL needed. -- QW
    "Kernels/Integer/QZO_Conv": [
        "--input-type-map", "blocks.0.conv.bias_rqsadd=int32_t",
        "--input-offset-map", "blocks.0.conv.bias_rqsadd=0",
    ],
    "Kernels/Integer/QZO_DualConv": [  # -- QW exp5: same int32 pin for the unperturbed requant add
        "--input-type-map", "blocks.0.conv.bias_rqsadd=int32_t",
        "--input-offset-map", "blocks.0.conv.bias_rqsadd=0",
    ],
    "Kernels/FP32/ZO_Conv_Block1": [
        "--input-type-map", "blocks_1_0_bias=float32_t",
        "--input-offset-map", "blocks_1_0_bias=0",
    ],
}

L2_DOUBLEBUFFER_KERNELS = {
    "Kernels/FP32/ReLU": [20],
    "Kernels/FP32/Softmax/Regular": [8000],
    "Kernels/FP32/Conv/DW_2D_Bias": [10000],
    "Kernels/FP32/Conv/DW_2D_NoBias": [9800],
    "Kernels/FP32/Conv/DW_2D_ZeroValuedBias": [9800],
    "Kernels/FP32/Conv/Regular_2D_Bias": [8800],
    "Kernels/FP32/Conv/Regular_2D_NoBias": [2000],
    "Kernels/FP32/Conv/Regular_2D_ZeroValuedBias": [8800],
    "Kernels/FP32/GEMM/Regular": [8000],
    "Kernels/FP32/MatMul": [5000],
    "Kernels/FP32/MaxPool/Regular_2D": [5000],
    "Kernels/FP32/Mul": [2000],
    "Kernels/FP32/LayerNorm": [2000],
    "Kernels/FP32/ReduceMean/KeepDims/Add_ReduceMean": [8000],
    "Kernels/FP32/ReduceMean/KeepDims/Add_ReduceMean_Add": [8000],
    "Kernels/FP32/ReduceMean/KeepDims/AllAxes": [100000],
    "Kernels/FP32/ReduceMean/KeepDims/Axes1_2_3": [100000],
    "Kernels/FP32/ReduceMean/KeepDims/Axes1_3": [10000, 50000],
    "Kernels/FP32/ReduceMean/KeepDims/Axes2_1": [13000, 50000],
    "Kernels/FP32/ReduceMean/KeepDims/Axis0": [17000, 50000],
    "Kernels/FP32/ReduceMean/KeepDims/Axis2": [17000, 50000],
    "Kernels/FP32/ReduceMean/KeepDims/ReduceMean_Add": [8000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Add_ReduceMean": [8000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Add_ReduceMean_Add": [8000],
    "Kernels/FP32/ReduceMean/NoKeepDims/AllAxes": [100000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Axes1_2_3": [100000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Axes1_3": [10000, 50000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Axes2_1": [13000, 50000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Axis0": [17000, 50000],
    "Kernels/FP32/ReduceMean/NoKeepDims/Axis2": [17000, 50000],
    "Kernels/FP32/ReduceMean/NoKeepDims/ReduceMean_Add": [8000],
    "Kernels/FP32/Reshape/SkipConnection": [2600],
    "Kernels/FP32/Transpose": [2000],
    "Kernels/Integer/Hardswish/Regular": [750],
    "Kernels/Integer/Softmax/Regular": [1600, 1000, 600],
    "Kernels/Integer/Concat": [64000, 32000, 16000],
    "Kernels/Integer/MatMul/Regular": [64000, 32000, 16000],
    "Kernels/Integer/RMSNorm": [4096, 2048, 1024],
    "Kernels/Integer/Conv/Regular_2D_RQ": [8000, 6000, 5000],
    "Kernels/Integer/Conv/DW_2D_RQ": [5121],
    "Kernels/Integer/Hardswish/Regular_RQ": [800],
}

L2_SINGLEBUFFER_MODELS = {
    "Models/CNN_Linear2": [45000, 30000, 15000],
    "Models/miniMobileNet": [60000, 12000, 6000, 3000],
    "Models/miniMobileNetv2": [60000, 16000, 12000, 8000],
    "Kernels/Integer/Attention": [60000, 10000, 5000],
    "Models/microLlama/microLlama1": [60000, 10000, 5000],
    "Models/microLlama/microLlama8": [60000, 10000, 5000],
    "Models/microLlama/microLlama8_parallel": [60000, 10000, 5000],
    "Models/MLPerf/KeywordSpotting": [64000],
    "Models/MLPerf/ImageClassification": [64000],
    "Models/MLPerf/AnomalyDetection": [64000],
    "Models/CCT/FP32/CCT_1_16_16_8": [64000],
    "Models/TinyViT/Demo": [4000],
}

L2_DOUBLEBUFFER_MODELS = {
    "Models/CNN_Linear2": [60000, 45000, 30000],
    "Models/miniMobileNet": [60000, 24000, 12000, 6000],
    "Models/miniMobileNetv2": [60000, 32000, 24000, 16000],
    "Kernels/Integer/Attention": [60000, 20000, 10000, 5000],
    "Models/microLlama/microLlama1": [60000, 20000, 10000],
    "Models/microLlama/microLlama8": [60000, 20000, 10000],
    "Models/microLlama/microLlama8_parallel": [60000, 20000, 10000],
    "Models/MLPerf/KeywordSpotting": [128000],
    "Models/MLPerf/ImageClassification": [128000],
    "Models/MLPerf/AnomalyDetection": [128000],
    "Models/CCT/FP32/CCT_1_16_16_8": [128000],
    "Models/TinyViT/Demo": [8000],
}

L3_SINGLEBUFFER_MODELS = {
    "Models/CNN_Linear2": [45000, 30000, 16000],
    "Models/miniMobileNet": [60000, 12000, 6000],
    "Models/miniMobileNetv2": [60000, 16000, 12000, 8000],
    "Kernels/Integer/Attention": [60000, 10000, 5000, 2500],
    "Models/Transformer": [60000, 30000, 15000],
    "Models/microLlama/microLlama1": [60000, 10000, 5000],
    "Models/CCT/FP32/CCT_2_32_32_128": [128000],
    "Models/TinyViT/Demo": [4000],
}

L3_DOUBLEBUFFER_MODELS = {
    "Models/CNN_Linear2": [60000, 45000, 30000],
    "Models/miniMobileNet": [60000, 24000, 12000, 6000],
    "Models/miniMobileNetv2": [60000, 32000, 24000, 16000],
    "Kernels/Integer/Attention": [60000, 20000, 10000, 5000],
    "Models/Transformer": [60000, 30000, 15000],
    "Models/microLlama/microLlama1": [60000, 20000, 10000],
    "Models/microLlama/microLlama8": [60000, 20000, 10000],
    "Models/microLlama/microLlama8_parallel": [60000, 20000, 10000],
    "Models/CCT/FP32/CCT_2_32_32_128": [128000],
    "Models/TinyViT/Demo": [4000],
}

# Training-enabled tiled models. Maps test path -> list of L1 sizes (bytes).
# L2 size is fixed by the runner at 2_000_000 to match the validated local run.
L2_SINGLEBUFFER_TRAINING_MODELS = {
    "Models/Training/SimpleMLP/simplemlp_train": [64000],
    # The Autoencoder has moved to the L3 list below. As the MLperf Tiny AD
    # reference (640 -> 128x4 -> 8 -> 128x4 -> 640, 267,928 params) its
    # graph-input tensors alone are 2,148,548 B -- 1046.6 KB of weights plus an
    # equally large gradient-accumulation buffer -- against Siracusa's 2 MB
    # (2,097,152 B) L2. It is over the whole level by 51 KB before a single
    # activation, so no --l2 value recovers it; minimalloc fails at the runner's
    # 1,991,744 B capacity.
    "Models/Training/DSCNN/dscnn_train": [128000],
    "Models/Training/SpeechNet/speechnet_train": [128000],
}

# Training-enabled tiled models that need L3 spill (weights/activations don't
# fit in L2). Same shape: test path -> list of L1 sizes (bytes).
L3_SINGLEBUFFER_TRAINING_MODELS = {
    # Autoencoder: L3 is the only level that fits it (see the L2 list above).
    "Models/Training/Autoencoder/autoencoder_train": [128000],
    "Models/Training/ResNet8/resnet8_train": [128000],
    "Models/Training/MobileNetV1/mobilenetv1_train": [128000],
    "Models/Training/CCT/cct_train": [128000],
}

# Double-buffered training models. Start narrow: only SimpleMLP until DB+alias
# path is validated end-to-end.
# L2 DB at L1=128 KB → almost all ops are 1-tile (tensors fit comfortably);
# DB pass triggers but has nothing to pipeline. The 32 KB autoencoder variant
# used to be here to force ~8 of 55 ops to 2-4 tiles so DB pipelining actually
# fired; that reasoning was written against the old 29,120-parameter fixture
# (55 ops) and does not carry over -- the MLperf reference is 205 nodes and no
# longer fits L2 at all. DSCNN is structurally DB-unfriendly at L2
# (depthwise/pointwise Conv weights are tiny, only ~1 of 97 ops multi-tiles even
# at L1=16 KB).
L2_DOUBLEBUFFER_TRAINING_MODELS = {
    "Models/Training/SimpleMLP/simplemlp_train": [64000],
    "Models/Training/DSCNN/dscnn_train": [128000],
}

# L3 DB training: only DB the L3↔L2 hop (TrainingDBOnlyL3Tiler) so the L2
# staging budget doesn't double.
# ResNet8 works now that InPlaceAccumulatorV2 is routed through the blocking L3
# DMA (BlockingForkTransformer): its conv-weight-grad accumulator emits strided
# 2D pi_cl_ram_copy_2d that would trip the gvsoc UDMA hyper_v3 transfer_splitter
# leak under async — blocking waits each transfer inline so it is safe.
# MobileNetV1 DB now passes: its multi-tile Transpose (NHWC<->NCHW) and
# ConvGradW were computed wrong under DB and are opted out of DB (see
# tilingUtils.DB_OPT_OUT_OPS). Localized via an SB-vs-DB per-op checksum
# node-diff; CCT/ResNet8 are unaffected by those opt-outs.
L3_DOUBLEBUFFER_TRAINING_MODELS = {
    "Models/Training/CCT/cct_train": [128000],
    "Models/Training/ResNet8/resnet8_train": [128000],
    "Models/Training/MobileNetV1/mobilenetv1_train": [128000],
}

# Per-model overrides for training tests.
#
# - num_data_inputs: required when inputs.npz has only one mini-batch (no
#   mb1_arr_* entries) so the runner can't auto-detect.
# - tolerance: per-model TRAINING_TOLERANCE_ABS bump for tests with known FP
#   precision drift past the 1e-3 default. Both numbers were established
#   locally; matching values pass on TP TrainingPlatform with the same
#   model artifacts.
TRAINING_MODEL_OVERRIDES = {
    "Models/Training/CCT/cct_train": {},
    # conv_channels_first: run the convs natively in NCHW. In NHWC the conv
    # nets emit hundreds of NCHW<->NHWC Transpose ops (ResNet8 forward alone has
    # ~647) that stream through L3 HyperRAM — cheap on GAP9's memory but ~4x the
    # cycles on Siracusa. CHW eliminates them, bringing ResNet8/MobileNetV1 back
    # to the expected ~80-100M cyc/step.
    "Models/Training/ResNet8/resnet8_train": {
        "conv_channels_first": True,
    },
    "Models/Training/MobileNetV1/mobilenetv1_train": {
        # Pretrained MLPerf Tiny VWW checkpoint (vww_96.h5): max diff 3.1e-5
        # across all 4 steps — default 1e-3 tolerance is fine.
        "conv_channels_first": True,
        # Bigger promote headroom than CCT (500K): MNV1's larger activations
        # over-commit L2 at 500K -> init crash. 900K shrinks the promoted pool.
        "promote_headroom": 900000,
    },
}

# Training models tested with PromoteTensorsToL2.
# Same shape as the inference promote dict: test path -> list of
# (l1, strategy, includeActivations). "off" = baseline (no promotion).
L3_SINGLEBUFFER_TRAINING_PROMOTE_MODELS = {
    "Models/Training/ResNet8/resnet8_train": [(128000, "cycle-aware", True),],
    "Models/Training/MobileNetV1/mobilenetv1_train": [(128000, "cycle-aware", True),],
    "Models/Training/CCT/cct_train": [(128000, "smallest", True),],
}

# Training models tested with PromoteTensorsToL2 AND double-buffering together.
# CCT + ResNet8: both run end-to-end under L3 DB (ResNet8's strided
# InPlaceAccumulatorV2 grad-accumulate is routed through the blocking L3 DMA
# adapter, so it no longer trips the gvsoc UDMA strided-DMA deadlock).
# MobileNetV1 is excluded for the same reason as L3_DOUBLEBUFFER_TRAINING_MODELS
# (DB numerically wrong from forward step 0 — under investigation).
L3_DOUBLEBUFFER_TRAINING_PROMOTE_MODELS = {
    "Models/Training/CCT/cct_train": [(128000, "smallest", True),],
    "Models/Training/ResNet8/resnet8_train": [(128000, "cycle-aware", True),],
    "Models/Training/MobileNetV1/mobilenetv1_train": [(128000, "cycle-aware", True),],
}

# Inference models tested with PromoteTensorsToL2.
# Each entry maps test path -> list of (l1, strategy, includeActivations).
# "off" strategy = no promotion (baseline for cycle comparison).
L3_SINGLEBUFFER_PROMOTE_MODELS = {
    "Models/CCT/FP32/CCT_2_32_32_128": [(128000, "greedy-score", True),],
    "Models/MLPerf/AnomalyDetection": [(128000, "cycle-aware", True),],
}

# -- QW 2026-10-12: SpeechNet single-step CI gates. Fixtures live in TrainDeeploy/SpeechNet_Test (see its
#    README); paths are relative to DeeployTest/. Each entry reproduces the command-line smoke run that
#    passed on GVSoC (SpeechNet_Test/logs/smoke/siracusa_*.log), with the cycle count it gave.
#
#    MeZO single step: one antithetic forward pair (L+, L-) + one update, compared exactly with the
#    host reference (Errors: 0 out of 2). lr / eps / seed must match the values baked at export.
ZO_SINGLE_STEP_TESTS = {
    "SpeechNet_ZO_single_step": {  # float ZO, lr 3e-6; smoke run: 35,634,779 train cycles
        "train_dir": "../SpeechNet_Test/speechnet_train_ZO_single_step",
        "update_dir": "../SpeechNet_Test/speechnet_update_ZO_single_step",
        "lr": 3e-6,
        "eps": 0.01,
        "seed": 42,
        "l1": 128000,
        "l2": 2000000,
    },
    "SpeechNet_QZO_single_step": {  # quantized ZO, lr 1e-5; smoke run: 8,311,241 train cycles
        "train_dir": "../SpeechNet_Test/speechnet_train_QZO_single_step",
        "update_dir": "../SpeechNet_Test/speechnet_update_QZO_single_step",
        "lr": 1e-5,
        "eps": 0.01,
        "seed": 42,
        "l1": 128000,
        "l2": 2000000,
    },
}

#    Block-0 QZO conv kernels on the cluster (exact int8 compare). The requant add of block 0 is int32
#    but every entry is 32768, which value inference would type as uint16_t: pin it (as for
#    Kernels/Integer/QZO_Conv above). fused = both eps-passes in one kernel (DualConv), unfused = one
#    pass through pulp-nn (the deployed path). Smoke runs: 858,185 (unfused) / 910,763 (fused) cycles.
QZO_BLOCK0_KERNEL_TESTS = {
    "SpeechNet_unfused_b0_cluster": "../SpeechNet_Test/speechnet_unfused_b0_cluster",
    "SpeechNet_fused_b0_cluster": "../SpeechNet_Test/speechnet_fused_b0_cluster",
}
QZO_BLOCK0_GEN_ARGS = [
    "--input-type-map", "blocks.0.conv.bias_rqsadd=int32_t",
    "--input-offset-map", "blocks.0.conv.bias_rqsadd=0",
]

#    QW 2026-10-14: blocks 1-4 QZO conv kernels on the cluster (exp7; HandoverDocuments/05_Fusion_Full_Network_Guide.md).
#    unfused = ONE pass (perturb nodes + transpose + pulp-nn), fused = both passes in RequantizedDualConv (7 inputs,
#    PULP_DualConv2d_2in_i8_i8_i8_HWC). The activation inputs are >= 0, so they must be pinned to int8 (otherwise the
#    generator types them uint8 and picks the u8 pulp-nn kernel); the fused fixtures have a second activation x-.
#    Smoke runs (fixture Runtime): unfused 469,934 / 170,216 / 103,558 / 173,803; fused 529,028 / 162,895 / 60,817 / 48,656.
QZO_BLOCKS_KERNEL_TESTS = {
    f"SpeechNet_{kind}_b{i}_cluster": (i, kind, f"../SpeechNet_Test/speechnet_{kind}_b{i}_cluster")
    for i in (1, 2, 3, 4) for kind in ("unfused", "fused")
}


def qzo_block_gen_args(i: int, fused: bool) -> list:
    a, b = f"/blocks_{i}_conv_input_quant_1/Clip_output_0", f"blocks.{i}.conv.bias_rqsadd"
    types, offsets = [f"{b}=int32_t", f"{a}=int8_t"], [f"{b}=0", f"{a}=0"]
    if fused:
        types.append(f"{a}_minus=int8_t")
        offsets.append(f"{a}_minus=0")
    return ["--input-type-map", *types, "--input-offset-map", *offsets]
