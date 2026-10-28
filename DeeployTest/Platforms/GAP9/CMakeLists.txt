# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})

option(TRAINING "Use training harness instead of inference harness" OFF)  # -- QW
option(MEZO_TRAINING "Use MeZO (ZO) training harness instead of inference/BP harness" OFF)  # -- QW

# QW: MeZO and BP training are mutually exclusive; MEZO takes precedence if both set. -- QW
if(MEZO_TRAINING AND TRAINING)  # -- QW
    message(WARNING "Both MEZO_TRAINING and TRAINING set; MEZO_TRAINING takes precedence, disabling TRAINING.")  # -- QW
    set(TRAINING OFF)  # -- QW
endif()  # -- QW

# Compile-time training parameters (override via -D on cmake command line)  # -- QW
set(N_TRAIN_STEPS "1" CACHE STRING "Number of optimizer steps")  # -- QW
set(N_ACCUM_STEPS "1" CACHE STRING "Number of mini-batches per optimizer step")  # -- QW
set(TRAINING_NUM_DATA_INPUTS "2" CACHE STRING "Number of data inputs per mini-batch")  # -- QW

# QW: MeZO (ZO) compile-time hyperparameters (override via -D on cmake command line) -- QW
set(ZO_EPS "0.01f" CACHE STRING "MeZO perturbation epsilon")   # -- QW
set(ZO_LR "3e-6f" CACHE STRING "MeZO learning rate")           # -- QW
set(ZO_Q "1" CACHE STRING "MeZO number of random directions per step")  # -- QW
set(ZO_SEED "0" CACHE STRING "MeZO base seed (informational)") # -- QW

set(${SDKCONFIG_FILE} ${CMAKE_CURRENT_LIST_DIR}/sdk.config)

if(MEZO_TRAINING)  # QW: MeZO (ZO) harness — parallel branch alongside BP TRAINING -- QW
    file(GLOB_RECURSE SOURCES
        src/CycleCounter.c
        src/deeploymezotest.c
    )
    set(NETWORK_LIB training_network)
elseif(TRAINING)
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

if(TRAINING OR MEZO_TRAINING)  # QW: MeZO links the same two OBJECT libs as BP -- QW
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

# RW: training harnesses run gvsoc at 370 MHz by default; -DGVSOC_DEFAULT_FREQ=ON keeps the SDK clocks.
if(GVSOC_DEFAULT_FREQ)
  target_compile_definitions(${ProjectId} PRIVATE GVSOC_DEFAULT_FREQ)
endif()

if(CC_STACK_SIZE)
  target_compile_definitions(${ProjectId} PRIVATE CC_STACK_SIZE=${CC_STACK_SIZE})
endif()

# Per-model cluster slave-stack size (see deeploytraintest.c). The stacks live in
# L1 (SDK default); this only shrinks them. Optional override per model.
if(SLAVESTACKSIZE)
  target_compile_definitions(${ProjectId} PRIVATE SLAVESTACKSIZE=${SLAVESTACKSIZE})
endif()

# Untiled baseline: "L1" buffers come from FC L2 and the mchan transfers become
# memcpy, so the cluster computes over the L2 fabric instead of out of TCDM.
# Both branches are emitted by codegen; this define selects the untiled one.
if(DEEPLOY_L1_AS_L2)
  target_compile_definitions(${ProjectId} PRIVATE DEEPLOY_L1_AS_L2)
  message(STATUS "[cMake  ]   DEEPLOY_L1_AS_L2       = ON (untiled: compute from L2)")
endif()

if(MEZO_TRAINING)  # QW: MeZO (ZO) compile-time defines — parallel block alongside BP TRAINING -- QW
    target_compile_definitions(${ProjectId} PRIVATE
        N_TRAIN_STEPS=${N_TRAIN_STEPS}
        N_ACCUM_STEPS=${N_ACCUM_STEPS}
        TRAINING_NUM_DATA_INPUTS=${TRAINING_NUM_DATA_INPUTS}
        ZO_EPS=${ZO_EPS}
        ZO_LR=${ZO_LR}
        ZO_Q=${ZO_Q}
        ZO_SEED=${ZO_SEED}
    )  # -- QW
    if(DUMP_WEIGHTS)  # QW: passthrough on-device weight dump -- QW
        target_compile_definitions(${ProjectId} PRIVATE DUMP_WEIGHTS)  # -- QW
    endif()  # -- QW
    if(DEFINED DUMP_STEP_LO AND DEFINED DUMP_STEP_HI)  # QW: per-step WDUMP window -- QW
        target_compile_definitions(${ProjectId} PRIVATE DUMP_STEP_LO=${DUMP_STEP_LO} DUMP_STEP_HI=${DUMP_STEP_HI})  # -- QW
    endif()  # -- QW
    if(BN_FROZEN_STATS)  # QW: passthrough frozen-stat BN training -- QW
        target_compile_definitions(${ProjectId} PRIVATE BN_FROZEN_STATS)  # -- QW
    endif()  # -- QW
    if(ZO_FUSED_DUAL)  # QW exp8: dual-pass training graph, one execution per window -- QW
        target_compile_definitions(${ProjectId} PRIVATE ZO_FUSED_DUAL)  # -- QW
    endif()  # -- QW
    if(ZO_QUIET)  # RW: quiet UART for long on-board rounds
        target_compile_definitions(${ProjectId} PRIVATE ZO_QUIET)
    endif()
    if(MEZO_STACKS_L2)  # RW: old behaviour -- cluster slave stacks in an L2 buffer (default: L1)
        target_compile_definitions(${ProjectId} PRIVATE MEZO_STACKS_L2)
    endif()
    if(ZO_EXPECT_WSUM)  # RW 2026-10-13: CI gate -- expected hash of the trained weights
        target_compile_definitions(${ProjectId} PRIVATE ZO_EXPECT_WSUM=${ZO_EXPECT_WSUM})
    endif()
    if(ZO_PERSIST)  # RW 2026-10-13: write the trained weights to a reserved flash region
        target_compile_definitions(${ProjectId} PRIVATE ZO_PERSIST)
    endif()
    if(ZO_RESUME)  # RW 2026-10-13: start from the weights saved there by the previous round
        target_compile_definitions(${ProjectId} PRIVATE ZO_RESUME)
    endif()
    if(ZO_ECO)  # RW 2026-10-13: ECO error feedback in the ZO update (ZORuntime.h); ZO_ECO_BETA / ZO_ECO_SR optional
        target_compile_definitions(${ProjectId} PRIVATE ZO_ECO)
        if(ZO_ECO_BETA)
            target_compile_definitions(${ProjectId} PRIVATE ZO_ECO_BETA=${ZO_ECO_BETA})
        endif()
        if(ZO_ECO_SR)
            target_compile_definitions(${ProjectId} PRIVATE ZO_ECO_SR=1u)
        endif()
    endif()
elseif(TRAINING)
    target_compile_definitions(${ProjectId} PRIVATE
        N_TRAIN_STEPS=${N_TRAIN_STEPS}
        N_ACCUM_STEPS=${N_ACCUM_STEPS}
        TRAINING_NUM_DATA_INPUTS=${TRAINING_NUM_DATA_INPUTS}
    )
    if(DUMP_WEIGHTS)
        target_compile_definitions(${ProjectId} PRIVATE DUMP_WEIGHTS)
    endif()
    if(BN_FROZEN_STATS)
        target_compile_definitions(${ProjectId} PRIVATE BN_FROZEN_STATS)
    endif()
else()  # QW: inference harness — allow frozen-stat BN for the ZO single perturbed forward -- QW
    if(BN_FROZEN_STATS)  # -- QW
        target_compile_definitions(${ProjectId} PRIVATE BN_FROZEN_STATS)  # -- QW
    endif()  # -- QW
endif()

# RW: Waive sign comparison warnings from pulp_nn_utils.h
target_compile_options(${NETWORK_LIB} PRIVATE  # -- QW: generalized from hardcoded 'network' to support training_network
    -Wno-sign-compare
    -Wno-pointer-sign
    -Wno-unknown-pragmas
    -Wno-error
  )

target_link_options(${ProjectId} PRIVATE
    -Wl,--print-memory-usage
)
link_compile_dump(${TESTNAME})
