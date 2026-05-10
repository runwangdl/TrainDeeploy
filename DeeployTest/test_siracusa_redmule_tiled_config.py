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
}

# L2 double-buffer kernel tests
L2_DOUBLEBUFFER_KERNELS = {
    "Kernels/FP32/GEMM/Regular": [8000],
}

# L3 single-buffer training models. Same fixtures used by the Siracusa-only
# tiled training CI; on Siracusa_w_redmule the GEMM/Matmul ops are bound to
# RedMulE while the rest of the graph (LayerNorm/GELU/Conv-grad/...) falls
# back to PULPCluster kernels via the second engine in RedmulePlatform.
L3_SINGLEBUFFER_TRAINING_MODELS = {
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
