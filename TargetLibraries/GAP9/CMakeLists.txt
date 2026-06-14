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
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_matmul_fp32.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_conv2d_fp32.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_im2col_fp32.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_conv_pw_fp32.c
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib/lib/sources/pulp_conv_dw_fp32.c
)
set_source_files_properties(${_KERNEL_O3_FILES} PROPERTIES COMPILE_OPTIONS "-O3")

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

target_link_libraries(deeploygap9 PUBLIC m)

