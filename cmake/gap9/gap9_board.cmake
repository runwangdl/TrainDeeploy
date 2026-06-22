# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# add_board_deployment(name target)
#
# Adds two cmake targets for on-board deployment:
#
#   image_<name>   — generate mram.bin + flash.bin in board_workdir only
#                    (no cable required; use this when the JTAG cable is on a
#                    remote machine and you scp the images manually)
#
#   board_<name>   — image + flash + run in one shot via gapy
#                    (requires JTAG cable accessible from this machine;
#                    set GAPY_OPENOCD_CABLE in the environment to the correct
#                    openocd cable .cfg, e.g. the olimex or gapuino config)
#
# Readfs files are passed via GAPY_RUNNER_ARGS as a list of
#   --flash-property=<path>@flash:readfs_flash:files
# entries before calling this macro.
#
# Power measurement operating point (compiled in only under -DPOWER_MEASUREMENT=1):
#   -DFREQ_FC=370 -DFREQ_CL=370 -DFREQ_PE=370 -DVOLTAGE=800
macro(add_board_deployment name target)

    if(NOT DEFINED GVSOC_INSTALL_DIR)
        message(FATAL_ERROR "Environment variable GVSOC_INSTALL_DIR not set")
    endif()

    set(GAP9_SDK_HOME $ENV{GAP_SDK_HOME})
    if(NOT GAP9_SDK_HOME)
        message(FATAL_ERROR "Environment variable GAP_SDK_HOME not set")
    endif()

    message(STATUS "[Deeploy GAP9] Board deployment target: ${name}")

    set(BOARD_WORKDIR ${CMAKE_BINARY_DIR}/board_workdir)
    set(DEEPLOY_BINARY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}")
    set(GAPY "${GAP9_SDK_HOME}/utils/gapy_v2/bin/gapy")
    set(FLASH_LAYOUT "${GAP9_SDK_HOME}/utils/layouts/default_layout_multi_readfs.json")
    set(FSBL_BINARY "${GAP9_SDK_HOME}/install/target/bin/fsbl")
    set(SSBL_BINARY "${GAP9_SDK_HOME}/install/target/bin/ssbl")

    # JTAG cable: prefer environment variable set by the board's gap9.sh
    # (e.g. GAPY_OPENOCD_CABLE=/path/to/olimex.cfg on federi).
    # Fall back to the gapuino cable shipped with the SDK.
    if(DEFINED ENV{GAPY_OPENOCD_CABLE})
        set(OPENOCD_CABLE $ENV{GAPY_OPENOCD_CABLE})
    else()
        set(OPENOCD_CABLE "${GAP9_SDK_HOME}/utils/openocd_tools/tcl/gapuino_ftdi.cfg")
        message(STATUS "[Deeploy GAP9] GAPY_OPENOCD_CABLE not set, using default gapuino cable")
    endif()

    make_directory(${BOARD_WORKDIR})

    # Power measurement operating point
    set(FREQ_FC "240" CACHE STRING "FC frequency in MHz (board power measurement)")
    set(FREQ_CL "240" CACHE STRING "Cluster frequency in MHz (board power measurement)")
    set(FREQ_PE "240" CACHE STRING "Periph frequency in MHz (board power measurement)")
    if(POWER_MEASUREMENT)
        target_compile_definitions(${name} PRIVATE
            FREQ_FC=${FREQ_FC} FREQ_CL=${FREQ_CL} FREQ_PE=${FREQ_PE})
        if(VOLTAGE)
            target_compile_definitions(${name} PRIVATE VOLTAGE=${VOLTAGE})
        endif()
        message(STATUS "[Deeploy GAP9] Power point: FC=${FREQ_FC} CL=${FREQ_CL} PE=${FREQ_PE} MHz")
    endif()

    # Common gapy arguments shared by both targets
    set(GAPY_COMMON
        ${GAPY}
        --target=gap9.evk
        --platform=board
        --target-property=boot.flash_device=mram
        --target-property=boot.mode=flash
        --target-dir=${GAP9_SDK_HOME}/utils/gapy_v2/targets
        --openocd-cable=${OPENOCD_CABLE}
        --openocd-script=${GAP9_SDK_HOME}/utils/openocd_tools/tcl/gap9revb.tcl
        --openocd-tools=${GAP9_SDK_HOME}/utils/openocd_tools
        --work-dir=${BOARD_WORKDIR}
        --multi-flash-content=${FLASH_LAYOUT}
        --flash-size=67108864
        --flash-property=${FSBL_BINARY}@mram:fsbl:binary
        --flash-property=${SSBL_BINARY}@mram:ssbl:binary
        --flash-property=${DEEPLOY_BINARY}@mram:app:binary
        ${GAPY_RUNNER_ARGS}
        --py-stack
    )

    # --- Target: image_<name> ---
    # Generates mram.bin + flash.bin in board_workdir without touching the board.
    # Use when the JTAG cable is on a remote machine (scp images, then flash manually).
    add_custom_target(image_${name}
        DEPENDS ${name}
        WORKING_DIRECTORY ${BOARD_WORKDIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${GAP9_SDK_HOME}/utils/efuse/GAP9/efuse_hyper_preload.data
            ${BOARD_WORKDIR}/chip.efuse_preload.data
        COMMAND ${GAPY_COMMON} image --binary=${DEEPLOY_BINARY}
        COMMENT "Generating flash images for ${name} in ${BOARD_WORKDIR}"
        USES_TERMINAL
        VERBATIM
    )

    # --- Target: board_<name> ---
    # Full image + flash + run via gapy. Requires JTAG cable on this machine.
    set(GAPY_CMD ${GAPY_COMMON} image flash run --binary=${DEEPLOY_BINARY})
    string(REPLACE ";" " " GAPY_CMD_STR "${GAPY_CMD}")

    add_custom_target(board_${name}
        DEPENDS ${name}
        WORKING_DIRECTORY ${BOARD_WORKDIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${GAP9_SDK_HOME}/utils/efuse/GAP9/efuse_hyper_preload.data
            ${BOARD_WORKDIR}/chip.efuse_preload.data
        COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
        COMMAND ${CMAKE_COMMAND} -E echo "[Deeploy GAP9] ${GAPY_CMD_STR}"
        COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
        COMMAND ${GAPY_CMD}
        COMMENT "Flashing and running ${name} on GAP9 board"
        USES_TERMINAL
        VERBATIM
    )
endmacro()
