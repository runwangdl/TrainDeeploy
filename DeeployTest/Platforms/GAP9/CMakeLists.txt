# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})

option(TRAINING "Use training harness instead of inference harness" OFF)

# Compile-time training parameters (override via -D on cmake command line)
set(N_TRAIN_STEPS "1" CACHE STRING "Number of optimizer steps")
set(N_ACCUM_STEPS "1" CACHE STRING "Number of mini-batches per optimizer step")
set(TRAINING_NUM_DATA_INPUTS "2" CACHE STRING "Number of data inputs per mini-batch")

if(TRAINING)
    file(GLOB_RECURSE SOURCES
        src/CycleCounter.c
        src/deeploytraintest.c
    )
    set(NETWORK_LIB training_network)
else()
    file(GLOB_RECURSE SOURCES
        src/CycleCounter.c
        src/deeploytest.c
    )
    set(NETWORK_LIB network)
endif()

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})
target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inc)

if(TRAINING)
    target_link_libraries(${ProjectId} PRIVATE ${NETWORK_LIB} optimizer_network deeploylib)
else()
    target_link_libraries(${ProjectId} PRIVATE ${NETWORK_LIB} deeploylib)
endif()
target_compile_options(${ProjectId} INTERFACE ${NETWORK_LIB})

if(TRAINING)
    # N_TRAIN_STEPS, N_ACCUM_STEPS, TRAINING_NUM_DATA_INPUTS are defined in
    # the generated testinputs.h; do not redefine them here.
endif()

add_gvsoc_emulation(${ProjectId} "gap9.evk")
add_board_deployment(${ProjectId} "gap9.evk")

if(POWER_MEASUREMENT)
  target_compile_definitions(${ProjectId} PRIVATE POWER_MEASUREMENT)
  # Operating point (FREQ_FC/CL/PE, optional VOLTAGE) is configured at the
  # board-deployment level — see add_board_deployment() in cmake/gap9/gap9_board.cmake.
  # Debug-only: -DPOWER_BISECT emits GPIO pulse bursts at each main() checkpoint
  # to locate where on-board execution hangs before the ROI.
  if(POWER_BISECT)
    target_compile_definitions(${ProjectId} PRIVATE POWER_BISECT)
  endif()
endif()

if(CC_STACK_SIZE)
  target_compile_definitions(${ProjectId} PRIVATE CC_STACK_SIZE=${CC_STACK_SIZE})
endif()

# RW: Waive sign comparison warnings from pulp_nn_utils.h
target_compile_options(${NETWORK_LIB} PRIVATE
    -Wno-sign-compare
    -Wno-pointer-sign
    -Wno-unknown-pragmas
    -Wno-error
  )

target_link_options(${ProjectId} PRIVATE
    -Wl,--print-memory-usage
)
link_compile_dump(${TESTNAME})
