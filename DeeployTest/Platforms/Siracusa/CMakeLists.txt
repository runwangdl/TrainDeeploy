# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})

option(TRAINING "Use training harness instead of inference harness" OFF)
option(MEZO_TRAINING "Use MeZO (ZO) training harness instead of inference/BP harness" OFF)  # QW -- QW
# QW (dual_pass_fusion exp4): standalone instruction-scheduling microbenchmark. Replaces the harness
#     source only; the generated network is still linked but never invoked, so any tiny kernel test
#     can serve as the build vehicle. -- QW
option(DUALPASS_BENCH "Use the dual-pass interleaving microbenchmark harness" OFF)  # QW -- QW

# QW: MeZO and BP training are mutually exclusive; MEZO takes precedence if both set. -- QW
if(MEZO_TRAINING AND TRAINING)  # QW -- QW
    message(WARNING "Both MEZO_TRAINING and TRAINING set; MEZO_TRAINING takes precedence, disabling TRAINING.")  # QW -- QW
    set(TRAINING OFF)  # QW -- QW
endif()  # QW -- QW

# Compile-time training parameters (override via -D on cmake command line)
set(N_TRAIN_STEPS "1" CACHE STRING "Number of optimizer steps")
set(N_ACCUM_STEPS "1" CACHE STRING "Number of mini-batches per optimizer step")
set(TRAINING_NUM_DATA_INPUTS "2" CACHE STRING "Number of data inputs per mini-batch")

# QW: MeZO (ZO) compile-time hyperparameters (override via -D on cmake command line) -- QW
set(ZO_EPS "0.01f" CACHE STRING "MeZO perturbation epsilon")   # QW -- QW
set(ZO_LR "3e-6f" CACHE STRING "MeZO learning rate")           # QW -- QW
set(ZO_Q "1" CACHE STRING "MeZO number of random directions per step")  # QW -- QW
set(ZO_SEED "0" CACHE STRING "MeZO base seed (informational)") # QW -- QW

if(DUALPASS_BENCH)  # QW exp4: scheduling microbenchmark; own main(), does not call the network -- QW
    file(GLOB_RECURSE SOURCES
        src/CycleCounter.c
        src/dualpass_bench.c
    )
    set(NETWORK_LIB network)
elseif(MEZO_TRAINING)  # QW: MeZO (ZO) harness — parallel branch alongside BP TRAINING -- QW
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

if(MEZO_TRAINING)  # QW: MeZO (ZO) compile-time defines — parallel block alongside BP TRAINING -- QW
    target_compile_definitions(${ProjectId} PRIVATE
        N_TRAIN_STEPS=${N_TRAIN_STEPS}
        N_ACCUM_STEPS=${N_ACCUM_STEPS}
        TRAINING_NUM_DATA_INPUTS=${TRAINING_NUM_DATA_INPUTS}
        ZO_EPS=${ZO_EPS}
        ZO_LR=${ZO_LR}
        ZO_Q=${ZO_Q}
        ZO_SEED=${ZO_SEED}
    )  # QW
    if(DUMP_WEIGHTS)  # QW: passthrough on-device weight dump -- QW
        target_compile_definitions(${ProjectId} PRIVATE DUMP_WEIGHTS)  # QW
    endif()  # QW
    if(DEFINED DUMP_STEP_LO AND DEFINED DUMP_STEP_HI)  # QW exp11_bitexact_SCE_verif: per-step WDUMP window -- QW
        target_compile_definitions(${ProjectId} PRIVATE DUMP_STEP_LO=${DUMP_STEP_LO} DUMP_STEP_HI=${DUMP_STEP_HI})  # QW
    endif()  # QW
    if(BN_FROZEN_STATS)  # QW: passthrough frozen-stat BN training -- QW
        target_compile_definitions(${ProjectId} PRIVATE BN_FROZEN_STATS)  # QW
    endif()  # QW
    if(ZO_FUSED_DUAL)  # QW exp8: dual-pass training graph, one execution per window (L+ / L- = outputs 0 / 1) -- QW
        target_compile_definitions(${ProjectId} PRIVATE ZO_FUSED_DUAL)  # QW
    endif()  # QW
elseif(TRAINING)
    target_compile_definitions(${ProjectId} PRIVATE
        N_TRAIN_STEPS=${N_TRAIN_STEPS}
        N_ACCUM_STEPS=${N_ACCUM_STEPS}
        TRAINING_NUM_DATA_INPUTS=${TRAINING_NUM_DATA_INPUTS}
    )
    if(DUMP_WEIGHTS)  # QW: opt-in on-device weight dump (-D DUMP_WEIGHTS=ON) -- QW
        target_compile_definitions(${ProjectId} PRIVATE DUMP_WEIGHTS)
    endif()  # QW
    if(DUMP_LOSS_HEX)  # -- QW exp21e: opt-in raw loss bits ([losshex i] device ref)
        target_compile_definitions(${ProjectId} PRIVATE DUMP_LOSS_HEX)
    endif()
    if(DUMP_ARGMAX)  # QW: opt-in MaxPoolGrad argmax-signature dump (drift evidence) -- QW
        target_compile_definitions(${ProjectId} PRIVATE DUMP_ARGMAX)
    endif()  # QW
    if(BN_FROZEN_STATS)  # QW: opt-in frozen-stat BN training (-D BN_FROZEN_STATS=ON) -- QW
        target_compile_definitions(${ProjectId} PRIVATE BN_FROZEN_STATS)
    endif()  # QW
else()  # QW: inference harness — allow frozen-stat BN for the ZO single perturbed forward (-D BN_FROZEN_STATS=ON) -- QW
    if(BN_FROZEN_STATS)  # QW
        target_compile_definitions(${ProjectId} PRIVATE BN_FROZEN_STATS)  # QW
    endif()  # QW
    if(BN_DEBUG)  # QW: frozen-BN kernel value dump -- QW
        target_compile_definitions(${ProjectId} PRIVATE BN_DEBUG)  # QW
    endif()  # QW
    if(DEFINED OUTPUT_TOL)  # QW exp13: output compare tolerance, e.g. -D OUTPUT_TOL=0.0f for bit-level probes -- QW
        target_compile_definitions(${ProjectId} PRIVATE OUTPUT_TOL=${OUTPUT_TOL})  # QW
    endif()  # QW
    if(OUTPUT_BITS)  # QW exp13: print raw fp32 bits of mismatching outputs -- QW
        target_compile_definitions(${ProjectId} PRIVATE OUTPUT_BITS)  # QW
    endif()  # QW
endif()

add_gvsoc_emulation(${ProjectId} "siracusa")
link_compile_dump(${TESTNAME})
