# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for Siracusa platform with RedMulE accelerator (tiled)."""

# Siracusa + RedMulE platform with tiling support
# Default configuration: 8 cores, gvsoc simulator

DEFAULT_CORES = 8

# L2 single-buffer kernel tests
# Format: dict of {test_name: [L1_sizes]}
L2_SINGLEBUFFER_KERNELS = {
    "Kernels/FP32/GEMM/Regular": [8000],
    "Kernels/FP32/GEMM/TransB": [8000],
    # Pointwise (1x1) ConvGradX from the MobileNet-style training backward
    # path: binds to RedMulE via PWConvGradX2DRedmuleMapper inserted into
    # PULPCluster's ConvGradXLayer in RedmulePlatform.__init__.  L1=8000
    # mirrors the GEMM kernel budget so the tiler picks similar tile shapes.
    # ConvGradW_PW is intentionally NOT in the matrix: the analogous
    # PWConvGradW2DRedmuleMapper exists in tree but is currently not
    # registered (see comment in Platform.py).
    "Kernels/FP32/ConvGradX_PW_block_11": [8000],
}

# L2 double-buffer kernel tests
L2_DOUBLEBUFFER_KERNELS = {
    "Kernels/FP32/GEMM/Regular": [8000],
}

# L3 single-buffer training models. Mirrors the Siracusa-only tiled training
# CI; on Siracusa_w_redmule the forward GEMM / MatMul / Conv ops bind to
# RedMulE (FP32) while every other op (LayerNorm/GELU/*Grad/Softmax/...) falls
# back to PULPCluster via the second engine in RedmulePlatform.  ResNet8 and
# MobileNetV1 are mostly Conv-heavy (the forward Conv kernel is RedMulE-bound
# as of 4517cc9); CCT is mostly transformer matmul/gemm.  Together they
# bracket the workload mix RedMulE actually sees in training.
L3_SINGLEBUFFER_TRAINING_MODELS = {
    "Models/Training/ResNet8/resnet8_train": [128000],
    "Models/Training/MobileNetV1/mobilenetv1_train": [128000],
    "Models/Training/CCT/cct_train": [128000],
}

# Match the per-model overrides used in test_siracusa_tiled_config so the
# RedMulE training run inherits the same num_data_inputs and tolerance
# (CCT step-0 forward drift ~1.5e-3, see comment in that file).
TRAINING_MODEL_OVERRIDES = {
    "Models/Training/CCT/cct_train": {
        "num_data_inputs": 1,
        "tolerance": 5e-3,
    },
}
