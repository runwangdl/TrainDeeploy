# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE SOURCES
  "src/**"
)

if(NOT DEFINED ENV{PULP_SDK_HOME})
  message(FATAL_ERROR "Environment variable PULP_SDK_HOME not set.")
endif()

if(platform STREQUAL "Siracusa" OR platform STREQUAL "Siracusa_w_neureka")
  include(cmake/pulp-sdk-siracusa.cmake)
elseif(platform STREQUAL "PULPOpen")
  include(cmake/pulp-sdk-pulp-open.cmake)
else()
  message(FATAL_ERROR "Unrecognized platform: ${platform}")
endif()

add_deeploy_library(deeploypulp STATIC ${SOURCES})
target_include_directories(deeploypulp
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
)

target_compile_options(deeploypulp PRIVATE
  -Wno-implicit-function-declaration
  -Wno-implicit-int-conversion
  -Wno-sign-conversion
  -Wno-sign-compare
  -Wno-typedef-redefinition
  -Wno-unused-parameter
  -Wno-unused-function
  -Wno-unused-variable
  -Wno-uninitialized
)

target_include_directories(deeploypulp PUBLIC ${PULP_SDK_INCLUDES})
target_compile_options(deeploypulp PUBLIC ${PULP_SDK_COMPILE_FLAGS})

# -- QW exp13/14 (2026-09-07): opt-in STRICT fp32 for selected fp32 kernels. Under the PULP SDK's -ffast-math clang
#    fuses mul+add into fmadd.s and reassociates (BatchNorm.c: 13 fused, Gemm.c: 154), so the device's fp32 result
#    is not the C source order and a host reference cannot match it bit-for-bit. Only -fno-fast-math -ffp-contract=off
#    together give fused=0 (measured). Per-source (appended after the target flags, so it wins), OFF by default.
#    Usage: -D DEEPLOY_STRICT_FP32_FILES="BatchNorm.c;Gemm.c"  (basenames under src/). -- QW
if(DEEPLOY_STRICT_FP32_FILES)
  foreach(_f IN LISTS DEEPLOY_STRICT_FP32_FILES)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/${_f} PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    message(STATUS "[QW strict-fp32] ${_f}: -fno-fast-math -ffp-contract=off")
  endforeach()
endif()

# -- QW (dual_pass_fusion exp7 step 5b): opt-in phase timing in DualConv2In.c (core 0 prints the cycles of phase 1
#    and phase 2, each including its closing barrier). Usage: -D DUALCONV2IN_PROBE=ON. OFF by default. -- QW
if(DUALCONV2IN_PROBE)
  target_compile_definitions(deeploypulp PUBLIC DUALCONV2IN_PROBE)
  message(STATUS "[QW] DUALCONV2IN_PROBE: DualConv2In.c prints per-phase cycles")
endif()
# -- QW (exp7 step 5c): measurement override of DualConv2In.c's phase-2 split: -D DUALCONV2IN_SPLIT=1 (pixels) | 2
#    (output channels). Unset = automatic (cost model). -- QW
if(DUALCONV2IN_SPLIT)
  target_compile_definitions(deeploypulp PUBLIC DUALCONV2IN_SPLIT=${DUALCONV2IN_SPLIT})
  message(STATUS "[QW] DUALCONV2IN_SPLIT=${DUALCONV2IN_SPLIT}")
endif()
# -- QW (2026-10-22, per-block breakdown): Conv4x2.c switches. -D CONV4X2_STEP2A=ON builds the original step-2a
#    kernel (byte gather, pixel split only); -D CONV4X2_SPLIT=1 (pixels) | 2 (output channels) overrides the split
#    of the default (step-5) kernel. Unset = step-5 kernel with the automatic split. -- QW
if(CONV4X2_STEP2A)
  target_compile_definitions(deeploypulp PUBLIC CONV4X2_STEP2A)
  message(STATUS "[QW] CONV4X2_STEP2A: Conv4x2.c is the original step-2a kernel")
endif()
if(CONV4X2_SPLIT)
  target_compile_definitions(deeploypulp PUBLIC CONV4X2_SPLIT=${CONV4X2_SPLIT})
  message(STATUS "[QW] CONV4X2_SPLIT=${CONV4X2_SPLIT}")
endif()

add_subdirectory(../third_party/pulp-nn-mixed ${CMAKE_CURRENT_BINARY_DIR}/pulp-nn-mixed)
target_include_directories(pulp-nn-mixed PUBLIC ${PULP_SDK_INCLUDES})
target_compile_options(pulp-nn-mixed PUBLIC ${PULP_SDK_COMPILE_FLAGS})

# Add pulp-trainlib integration (used by ConvGrad family)
target_include_directories(deeploypulp PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/include
)

# Add necessary pulp-trainlib sources for ConvGradW / ConvGradX
file(GLOB PULP_TRAINLIB_CONV_SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv2d_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_im2col_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv_dw_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv_pw_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_matmul_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_train_utils_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv_naive_fp32.c"
)
target_sources(deeploypulp PRIVATE ${PULP_TRAINLIB_CONV_SOURCES})

# Fix compilation issues with pulp-trainlib sources
# Undefine USE_DMA and define float16alt to avoid conflicts
set_source_files_properties(${PULP_TRAINLIB_CONV_SOURCES}
  PROPERTIES
  COMPILE_FLAGS "-UUSE_DMA -Dfloat16alt=float"
)

target_link_libraries(deeploypulp PUBLIC pulp-nn-mixed)
target_link_libraries(deeploypulp INTERFACE pulp-sdk)
target_sources(deeploypulp INTERFACE $<TARGET_OBJECTS:pulp-sdk>)

set(PULP_NNX_PLATFORMS "Siracusa;Siracusa_w_neureka")
if (platform IN_LIST PULP_NNX_PLATFORMS)
  if (platform STREQUAL "Siracusa" OR platform STREQUAL "Siracusa_w_neureka")
    set(USE_NEUREKA ON)
  else()
    message(FATAL_ERROR "Missing accelerator flags for platform ${platform}")
  endif()
  add_subdirectory(../third_party/pulp-nnx ${CMAKE_CURRENT_BINARY_DIR}/pulp-nnx)
  target_include_directories(pulp-nnx PUBLIC ${PULP_SDK_INCLUDES})
  target_compile_options(pulp-nnx PUBLIC ${PULP_SDK_COMPILE_FLAGS})
    target_compile_options(pulp-nnx PRIVATE
    -Wno-implicit-int-conversion
    -Wno-sign-conversion
    -Wno-typedef-redefinition
    -Wno-unused-parameter
    -Wno-incompatible-pointer-types-discards-qualifiers
    )
  target_link_libraries(deeploypulp INTERFACE pulp-nnx)
endif()
