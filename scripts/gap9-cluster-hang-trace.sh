#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# gap9-cluster-hang-trace.sh
#
# Locate where a GAP9 (GVSoC) training/inference run dies on the cluster.
#
# Why this exists:
#   When a GAP9 cluster run corrupts a control pointer (e.g. a buffer/DMA
#   overrun clobbers the pi_cl fork-dispatch entry), a worker core jumps to a
#   garbage PC and spins on an illegal `c.unimp` instruction. Tracing the FC is
#   useless (the FC just idle-spins in the scheduler), so this script traces a
#   *cluster* core, watches for the PC-corruption trap loop, stops GVSoC, and
#   prints the last valid instructions + addr2line of the crash site — i.e. the
#   function/closure that was running when the corruption surfaced.
#
# It distinguishes three outcomes:
#   - PC-CORRUPTION (c.unimp trap loop)  -> the informative case; shows crash site
#   - FC "Invalid fetch"                 -> FC-side abort (GVSoC dies, trace lost)
#   - no corruption past the target cyc  -> ran clean (bug not triggered)
#
# Requirements:
#   - The test must already be generated+built (a build_master with a binary).
#   - GVSoC `--trace` needs the debug models dir on LD_LIBRARY_PATH (handled here),
#     otherwise it aborts with "mem_plug_debug.so: cannot open shared object".
#   - Run as the SAME user that built build_master (root/agent ownership of the
#     gvsoc_workdir artifacts must be writable, else gapy fails to dump flash).
#
# Usage:
#   scripts/gap9-cluster-hang-trace.sh [TEST_NAME] [CORE] [TEST_DIR]
# Examples:
#   scripts/gap9-cluster-hang-trace.sh                      # mobilenetv1_train, pe0
#   scripts/gap9-cluster-hang-trace.sh resnet8_train pe3
#
# Env overrides:
#   GAP9_SDK       (default /app/install/gap9-sdk)
#   GAP9_GCC       (default /app/install/gcc/gap9)
#   PASS_CYCLES    (default 40000000) cycles past which "no c.unimp" => ran clean

set +e

TEST_NAME="${1:-mobilenetv1_train}"
CORE="${2:-pe0}"
TEST_DIR="${3:-TEST_GAP9}"
GAP9_SDK="${GAP9_SDK:-/app/install/gap9-sdk}"
GAP9_GCC="${GAP9_GCC:-/app/install/gcc/gap9}"
PASS_CYCLES="${PASS_CYCLES:-40000000}"
GARBAGE='c.unimp'   # illegal-instruction marker of the PC-corruption trap loop

# --- locate the repo's DeeployTest build dir ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD="$REPO/DeeployTest/$TEST_DIR/build_master"
ELF="$BUILD/$TEST_NAME"
A2L="$GAP9_GCC/bin/riscv32-unknown-elf-addr2line"
OUT="/tmp/gap9_hang_${TEST_NAME}_${CORE}.trace"

[ -x "$ELF" ] || { echo "ERROR: binary not found: $ELF (generate+build the test first)"; exit 1; }

# --- extract the GVSoC gapy run command emitted by CMake for this target ---
GVMAKE="$BUILD/DeeployTest/Platforms/GAP9/CMakeFiles/gvsoc_${TEST_NAME}.dir/build.make"
CMD="$(grep -hoa "${GAP9_SDK}/utils/gapy_v2/bin/gapy --target=gap9.evk[^\"]*--binary=[^ \"]*${TEST_NAME}" "$GVMAKE" 2>/dev/null | head -1)"
[ -n "$CMD" ] || { echo "ERROR: could not extract gapy gvsoc command from $GVMAKE"; exit 1; }

# --- environment (debug models on LD_LIBRARY_PATH so --trace doesn't abort) ---
source "$GAP9_SDK/configs/gap9_evk_audio.sh" >/dev/null 2>&1
export CHIP_FAMILY=9 CHIP_VERSION=2 GVSOC_INSTALL_DIR="$GAP9_SDK/install/workstation"
export LD_LIBRARY_PATH="$GAP9_SDK/install/workstation/models/debug:$LD_LIBRARY_PATH"
unset PYTHONPATH
cd "$BUILD" || exit 1
# stale root-owned gvsoc artifacts block writing; drop the ones gapy regenerates
find gvsoc_workdir -maxdepth 1 \( -name 'chip.*' -o -name 'flash.bin' \) ! -writable -delete 2>/dev/null

echo ">>> tracing /chip/cluster/$CORE/insn for $TEST_NAME  (out: $OUT)"
rm -f "$OUT"
stdbuf -oL -eL $CMD --trace=/chip/cluster/$CORE/insn > "$OUT" 2>&1 &
GVPID=$!
reason="?"
prevsz=-1; stall=0; STALL_NEEDED=12   # ~60s of no new pe trace => pe core idle/hung
trap 'reason=ctrl-c; kill -9 $GVPID 2>/dev/null; pkill -9 gvsoc_launcher 2>/dev/null' INT
while kill -0 $GVPID 2>/dev/null; do
  sleep 5
  sz=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  cyc=$(grep -a "$CORE/insn" "$OUT" 2>/dev/null | tail -1 | awk -F: '{print $2}' | tr -dc 0-9)
  hit=$(grep -ac "$GARBAGE" "$OUT" 2>/dev/null)
  free_gb=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc 0-9)
  [ "$sz" -eq "$prevsz" ] && stall=$((stall+1)) || stall=0; prevsz=$sz
  echo "[$(date +%H:%M:%S)] $((sz/1048576))MB cyc=${cyc:-?} c.unimp=${hit} stall=${stall}/${STALL_NEEDED} free=${free_gb}G"
  if [ "$hit" -gt 0 ]; then reason="PC-CORRUPTION (c.unimp trap loop) — crash site below"; kill -9 $GVPID 2>/dev/null; pkill -9 gvsoc_launcher; break; fi
  if grep -qa "Invalid fetch" "$OUT" 2>/dev/null; then reason="FC Invalid fetch — GVSoC aborted, cluster trace lost (try a build that crashes cluster-side)"; kill -9 $GVPID 2>/dev/null; pkill -9 gvsoc_launcher; break; fi
  if [ "$stall" -ge "$STALL_NEEDED" ]; then reason="$CORE STALLED at cyc ${cyc} — this core stopped executing (idle during a master/optimizer phase, OR a hang). Trace another core / run the full training to tell which."; kill -9 $GVPID 2>/dev/null; pkill -9 gvsoc_launcher; break; fi
  if [ -n "$free_gb" ] && [ "$free_gb" -lt 8 ]; then reason="ABORT: low disk"; kill -9 $GVPID 2>/dev/null; pkill -9 gvsoc_launcher; break; fi
done
# loop fell through without a detected crash -> GVSoC exited by itself
[ "$reason" = "?" ] && reason="GVSoC exited on its own — run finished, NO hang/crash detected"

echo ""
echo "===================== RESULT: $reason ====================="
LN=$(grep -an "$GARBAGE" "$OUT" | head -1 | cut -d: -f1)
[ -z "$LN" ] && LN=$(grep -an "$CORE/insn" "$OUT" | tail -1 | cut -d: -f1)
echo "anchor line ${LN:-none}  last_cycle=$(grep -a "$CORE/insn" "$OUT" 2>/dev/null | tail -1 | awk -F: '{print $2}')  pe_trace_lines=$(grep -ac "$CORE/insn" "$OUT" 2>/dev/null)"
if [ -n "$LN" ]; then
  S=$((LN-40)); [ $S -lt 1 ] && S=1
  echo "--- ~40 instructions before the crash anchor ---"
  sed -n "${S},${LN}p" "$OUT" | sed -E 's/\x1b\[[0-9]*m//g' | sed -E 's#.*'"$CORE"'/insn *\] *##' | cut -c1-95
  echo "--- distinct functions near the crash (call sequence) ---"
  grep -a "$CORE/insn" "$OUT" | tail -6000 | sed -E 's/\x1b\[[0-9]*m//g' | grep -oE '\] [A-Za-z0-9_]+:[0-9]' | awk '{print $2}' | awk '!s[$0]++' | tail -25
  echo "--- addr2line of the 12 PCs before the anchor ---"
  for pc in $(sed -n "$((LN-12)),$((LN-1))p" "$OUT" | sed -E 's/\x1b\[[0-9]*m//g' | grep -oE ' M [0-9a-f]+ ' | awk '{print $2}'); do
    printf "0x%s -> " "$pc"; "$A2L" -f -e "$ELF" "0x$pc" | tr '\n' ' '; echo
  done
fi
grep -a "Invalid fetch" "$OUT" 2>/dev/null | tail -1 | sed -E 's/\x1b\[[0-9]*m//g'
