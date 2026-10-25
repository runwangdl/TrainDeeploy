# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE SOURCES
  "src/**"
)

# RW: Include PULPOpen sources but exclude dory_mem related files
file(GLOB_RECURSE PULPOPEN_SOURCES "../PULPOpen/src/**")
list(FILTER PULPOPEN_SOURCES EXCLUDE REGEX ".*dory_mem.*")
list(APPEND SOURCES ${PULPOPEN_SOURCES})

add_deeploy_library(deeploygap9 STATIC ${SOURCES})
target_include_directories(deeploygap9
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/inc
)

target_compile_options(deeploygap9 PUBLIC
    -DNUM_CORES=${NUM_CORES}
  )

target_compile_options(deeploygap9 PRIVATE
  -Wno-sign-conversion
  -Wno-sign-compare
  -Wno-type-limits
  -Wno-attributes
  -Wno-maybe-uninitialized  # RW: -O3 flags perf_stats_t in RandomNoise.c profiling (benign)
)

# The GAP9 SDK injects -Os globally, which compiles the compute kernels for size.
# Siracusa builds the same trainlib/conv sources at -O3, and the forward+backward
# conv/matmul kernels dominate training (~95% of cycles). Compile the perf-critical
# kernel files at -O3 (appended last so it wins over the SDK's -Os on the compile
# line), leaving init/util/scheduler code at -Os. Measured: ~2.3x per kernel; on
# ResNet8 training the whole step drops from -Os to faster-than-Siracusa.
set(_KERNEL_O3_FILES
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/Convolution_fp32.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/DWConvolution_fp32.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/ConvGrad.c  # backward dX/dW: ~70% of training cycles
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/DWConvGrad.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/PWConvGrad.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/Gemm.c       # transformer (CCT) fwd+bwd Gemm
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/Matmul.c     # attention matmul
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/Layernorm.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/Softmax.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/GELU.c
  # SpeechNet ZO/QZO kernels (RW 2026-10-12)
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/BatchNorm.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/AvgPool.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/MaxPool.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/GlobalAveragePool.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/Relu.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/RandomNoise.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/RandomNoiseQuant.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/RequantShift.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/UniformRequantShift.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/DualConv.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/FlatConv.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/Im2ColU8.c
  # QW (dual_pass_fusion exp7): the blocks 1-4 kernels and exp6's NE16 prep, at -O3 like DualConv.c / Im2ColU8.c
  #    (under the SDK's -Os the fused cluster kernel ran 914 k vs 451 k on Siracusa). -- QW
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/DualConv2In.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/Conv4x2.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/src/PerturbConv4x2.c  # QW exp11: single-pass in-kernel-perturbation conv
  ${CMAKE_CURRENT_LIST_DIR}/src/NE16DualPrep.c
  ${CMAKE_CURRENT_LIST_DIR}/src/NE16DualConv2in.c
  # QW (exp11, 2026-10-23): the on-device NE16 weight encode (exp16c), the one compute kernel of the NE16 path left at
  #    the SDK's -Os (5.3-5.5 cycles per weight element; block 4: 76 k of the step). Same output (integer bit packing);
  #    +56 B code, +32 B worker stack (64 -> 96 B), measured with the board flags. -- QW
  ${CMAKE_CURRENT_LIST_DIR}/src/NE16WeightEncode.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_matmul_fp32.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_conv2d_fp32.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_im2col_fp32.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_conv_pw_fp32.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_conv_dw_fp32.c
)
set_source_files_properties(${_KERNEL_O3_FILES} PROPERTIES COMPILE_OPTIONS "-O3")

# -- QW (2026-10-22, per-block breakdown on GAP9): the same measurement switches as TargetLibraries/PULPOpen for the
#    PULPOpen kernels compiled into this library. DUALCONV2IN_PROBE=ON (core-0 cycles of phase 1 / phase 2 of
#    DualConv2In.c), DUALCONV2IN_SPLIT=1|2, CONV4X2_STEP2A=ON (original step-2a Conv4x2), CONV4X2_SPLIT=1|2. -- QW
if(DUALCONV2IN_PROBE)
  target_compile_definitions(deeploygap9 PUBLIC DUALCONV2IN_PROBE)
  message(STATUS "[QW] DUALCONV2IN_PROBE: DualConv2In.c prints per-phase cycles")
endif()
if(DUALCONV2IN_SPLIT)
  target_compile_definitions(deeploygap9 PUBLIC DUALCONV2IN_SPLIT=${DUALCONV2IN_SPLIT})
  message(STATUS "[QW] DUALCONV2IN_SPLIT=${DUALCONV2IN_SPLIT}")
endif()
# -- QW exp11: NE16DUAL_PROBE=ON: per-core phase cycles of NE16DualPerturbEncode / NE16DualBias and the master-core split
#    of NE16DualConv2in (column builds / NE16 waits / rest), as perf records. -- QW
if(NE16DUAL_PROBE)
  target_compile_definitions(deeploygap9 PUBLIC NE16DUAL_PROBE)
  message(STATUS "[QW] NE16DUAL_PROBE: NE16 prep / conv node phase cycles")
endif()
if(CONV4X2_STEP2A)
  target_compile_definitions(deeploygap9 PUBLIC CONV4X2_STEP2A)
  message(STATUS "[QW] CONV4X2_STEP2A: Conv4x2.c is the original step-2a kernel")
endif()
if(CONV4X2_SPLIT)
  target_compile_definitions(deeploygap9 PUBLIC CONV4X2_SPLIT=${CONV4X2_SPLIT})
  message(STATUS "[QW] CONV4X2_SPLIT=${CONV4X2_SPLIT}")
endif()

target_link_libraries(deeploygap9 PUBLIC pmsis)

# Add pulp-trainlib integration (used by ConvGrad family), mirroring PULPOpen.
target_include_directories(deeploygap9 PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/include
)

# Add necessary pulp-trainlib sources for ConvGradW / ConvGradX
file(GLOB PULP_TRAINLIB_CONV_SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_conv2d_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_im2col_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_conv_dw_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_conv_pw_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_matmul_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_train_utils_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_conv_naive_fp32.c"
)
target_sources(deeploygap9 PRIVATE ${PULP_TRAINLIB_CONV_SOURCES})

# Fix compilation issues with pulp-trainlib sources
# Undefine USE_DMA and define float16alt to avoid conflicts
set_source_files_properties(${PULP_TRAINLIB_CONV_SOURCES}
  PROPERTIES
  COMPILE_FLAGS "-UUSE_DMA -Dfloat16alt=float -Wno-error -Wno-strict-aliasing -Wno-unused-variable -Wno-unused-function -Wno-sign-compare"
)

#RW: Link PULP-NN
#RW: Set PULP-NN version and bitwidth for pulp-nn-mixed
set(PULPNNVERSION XPULPV2)
set(PULPNNBITWIDTH 32)

# Option to use prebuilt pulp-nn library (default: ON for faster builds)
option(USE_PREBUILT_PULPNN "Use prebuilt pulp-nn-mixed librar with 8 cores" ON)

# Check if NUM_CORES matches prebuilt library requirement.
# FORCE_PREBUILT_PULPNN keeps the 8-core prebuilt even at other core counts:
# safe for pure-FP32 (training) networks that never call the integer pulp-nn
# kernels, and it sidesteps the source build (which needs GAP9-v2 integer SIMD
# builtins absent from this LLVM).
option(FORCE_PREBUILT_PULPNN "Keep prebuilt pulp-nn even when NUM_CORES!=8" OFF)
if(USE_PREBUILT_PULPNN AND NOT NUM_CORES EQUAL 8 AND NOT FORCE_PREBUILT_PULPNN)
  message(WARNING "[Deeploy GAP9] Prebuilt pulp-nn library is compiled for 8 cores, but NUM_CORES=${NUM_CORES}.")
  message(WARNING "[Deeploy GAP9] Disabling prebuilt library and building from source instead.")
  set(USE_PREBUILT_PULPNN OFF)
endif()

if(USE_PREBUILT_PULPNN)
  # Use prebuilt pulp-nn library
  set(PREBUILT_PULPNN_DIR ${CMAKE_CURRENT_LIST_DIR}/prebuilt)

  if(EXISTS ${PREBUILT_PULPNN_DIR}/libpulp-nn-mixed.a)
    message(STATUS "[Deeploy GAP9] Using prebuilt pulp-nn-mixed library")

    # Create imported target for prebuilt library
    add_library(pulp-nn-mixed STATIC IMPORTED)
    set_target_properties(pulp-nn-mixed PROPERTIES
      IMPORTED_LOCATION ${PREBUILT_PULPNN_DIR}/libpulp-nn-mixed.a
      INTERFACE_INCLUDE_DIRECTORIES ${PREBUILT_PULPNN_DIR}/include
    )
  else()
    message(WARNING "[Deeploy GAP9] Prebuilt pulp-nn library not found at ${PREBUILT_PULPNN_DIR}")
    message(WARNING "[Deeploy GAP9] Falling back to building from source")
    set(USE_PREBUILT_PULPNN OFF)
  endif()
endif()

if(NOT USE_PREBUILT_PULPNN)
  # Build pulp-nn from source
  message(STATUS "[Deeploy GAP9] Building pulp-nn-mixed from source")

  add_subdirectory(../third_party/pulp-nn-mixed ${CMAKE_CURRENT_BINARY_DIR}/pulp-nn-mixed)

  #RW: GCC not recognizing -Wno-typedef-redefinition defined in PULP-NN CMakelist
  target_compile_options(pulp-nn-mixed PRIVATE -Wno-error)
  target_compile_definitions(pulp-nn-mixed PUBLIC NUM_CORES=${NUM_CORES})
  target_link_libraries(pulp-nn-mixed PUBLIC pmsis)
endif()

target_link_libraries(deeploygap9 PUBLIC pulp-nn-mixed)

# QW: NE16 accelerator (via pulp-nnx) — GAP9_w_NE16 only. Modelled on upstream Deeploy PR #183's
#     TargetLibraries/GAP9/CMakeLists.txt, MINUS its CNN_BasicKernels_NE16.c exclusion: that
#     exclusion exists because the PR's GAP9 target compiles the GAP9-SDK NE16 AutoTiler kernels,
#     whose hal_ne16.h redefines the NE16_REG_* macros that pulp-nnx's ne16_task_defs.h also
#     defines (-Werror redefinition). Our GAP9 target never compiled those SDK kernels in, so
#     there is nothing to exclude. Codegen emits the pulp-nnx HAL calls (ne16_nnx_init /
#     ne16_nnx_dispatch), which live here. -- QW
if(platform STREQUAL GAP9_w_NE16)  # -- QW
  set(USE_NE16 ON CACHE BOOL "Use the NE16 accelerator." FORCE)  # -- QW
  add_subdirectory(../third_party/pulp-nnx ${CMAKE_CURRENT_BINARY_DIR}/pulp-nnx)  # -- QW
  target_link_libraries(pulp-nnx PUBLIC pmsis)  # -- QW
  target_compile_options(pulp-nnx PRIVATE  # -- QW
    -Wno-error
    -Wno-implicit-int-conversion
    -Wno-sign-conversion
    -Wno-typedef-redefinition
    -Wno-unused-parameter
    -Wno-incompatible-pointer-types-discards-qualifiers
  )
  target_link_libraries(deeploygap9 PUBLIC pulp-nnx)  # -- QW
endif()  # -- QW

target_link_libraries(deeploygap9 PUBLIC m)

