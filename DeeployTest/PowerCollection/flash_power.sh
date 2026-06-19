#!/usr/bin/env bash
#
# flash_power.sh — flash a migrated Deeploy GAP9 build to the board via gapy,
# and (optionally) collect power with a Nordic PPK2.
#
# This is the "make/flash" step on the POWER-MEASUREMENT platform after copying
# a build folder over from the Deeploy build machine. Each <model> dir here
# contains everything gapy needs from the build machine:
#
#   PowerCollection/<Model>/
#     build_master/            full build dir (incl. the flashable ELF + board_workdir)
#     hex/                     L3 weight payload written to external flash (readfs)
#
# Everything else (SSBL, FSBL, openocd cable/scripts, gapy itself) comes from the
# test platform's own GAP SDK — NOT from the build machine. Point GAP_SDK_HOME at
# it. IMPORTANT: keep the build-machine SDK and the test-platform SDK on the same
# version (SSBL/boot/ABI are SDK-bound), or the board may not boot.
#
# The power window is delimited by a GPIO pulse (pin 89) that the training harness
# raises around the full training loop (see deeploytraintest.c, #ifdef
# POWER_MEASUREMENT). gapy's PPK2 mode synchronizes to that pulse to integrate
# average power over exactly the compute window — so the binary MUST have been
# built with -DPOWER_MEASUREMENT=ON (the builds in this folder were).
#
# Usage:
#   ./flash_power.sh <Model-dir> [--power]
#
#   <Model-dir>   path to a per-model dir (e.g. ./SleepConViT) holding build_master/ + hex/
#   --power       also run gapy's PPK2 power capture (needs a PPK2 on the test platform)
#
# Env:
#   GAP_SDK_HOME  test platform's GAP SDK (default: /app/install/gap9-sdk)
#
# Examples:
#   GAP_SDK_HOME=$HOME/gap_sdk ./flash_power.sh ./SleepConViT
#   GAP_SDK_HOME=$HOME/gap_sdk ./flash_power.sh ./MCUNet --power
#
set -euo pipefail

SDK="${GAP_SDK_HOME:-/app/install/gap9-sdk}"
GAPY="$SDK/utils/gapy_v2/bin/gapy"

MODELDIR="${1:?usage: ./flash_power.sh <Model-dir> [--power]}"
POWER="${2:-}"

BUILDDIR="$(cd "$MODELDIR/build_master" && pwd)"
HEXDIR="$(cd "$MODELDIR/hex" && pwd)"

# The flashable ELF in build_master is the bare (extensionless) executable, e.g.
# sleepconvit_train. Auto-detect it (skip .s/.map/etc.).
MODEL=""
for f in "$BUILDDIR"/*_train; do
  [ -f "$f" ] && { MODEL="$(basename "$f")"; break; }
done
[ -n "$MODEL" ] || { echo "ERROR: no *_train ELF found in $BUILDDIR" >&2; exit 1; }

echo "SDK      : $SDK"
echo "BUILDDIR : $BUILDDIR"
echo "HEXDIR   : $HEXDIR ($(ls "$HEXDIR"/*.hex 2>/dev/null | wc -l) hex files)"
echo "MODEL    : $MODEL"
echo "POWER    : ${POWER:-off}"

# Sanity: the binary must contain the power GPIO trigger for PPK2 sync.
if [ "$POWER" = "--power" ]; then
  if ! grep -qa "POWER_MEASUREMENT" "$BUILDDIR/$MODEL" 2>/dev/null; then
    echo "WARN: '$MODEL' may not be built with -DPOWER_MEASUREMENT=ON; PPK2 GPIO sync needs it." >&2
  fi
fi

args=(
  --target=gap9.evk --platform=board
  --target-property=boot.flash_device=mram --target-property=boot.mode=flash
  --target-dir="$SDK/utils/gapy_v2/targets"
  --openocd-cable="$SDK/utils/openocd_tools/tcl/gapuino_ftdi.cfg"
  --openocd-script="$SDK/utils/openocd_tools/tcl/gap9revb.tcl"
  --openocd-tools="$SDK/utils/openocd_tools"
  --work-dir="$BUILDDIR/board_workdir"
  --multi-flash-content="$SDK/utils/layouts/default_layout_multi_readfs.json"
  --flash-size=67108864
  --flash-property="$SDK/install/target/bin/fsbl@mram:fsbl:binary"
  --flash-property="$SDK/install/target/bin/ssbl@mram:ssbl:binary"
  --flash-property="$BUILDDIR/$MODEL@mram:app:binary"
)

# L3 weight payload -> external flash readfs partition.
shopt -s nullglob
for h in "$HEXDIR"/*.hex; do
  args+=(--flash-property="$h@flash:readfs_flash:files")
done
shopt -u nullglob

# Optional PPK2 power capture (synchronized to the pin-89 GPIO window).
if [ "$POWER" = "--power" ]; then
  args+=(--power --power-tool ppk2 --power-plot)
fi

# image: (re)build the flash image from the properties above
# flash: write it to the board over JTAG/openocd
# run:   start execution
args+=(--py-stack image flash run --binary="$BUILDDIR/$MODEL")

# gapy chdir's into the work-dir; openocd resolves its tcl includes relative to
# the SDK tools, so run from the work-dir to match the in-tree build behaviour.
cd "$BUILDDIR/board_workdir"
echo "+ gapy ${args[*]}"
exec "$GAPY" "${args[@]}"
