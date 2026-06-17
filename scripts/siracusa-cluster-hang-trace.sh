#!/usr/bin/env bash
# Siracusa adaptation of gap9-cluster-hang-trace.sh (PR #35).
# Traces a Siracusa (GVSoC) cluster core to locate where a run hangs/crashes.
#   Usage: siracusa-cluster-hang-trace.sh [TEST_NAME] [CORE]
set +e
TEST_NAME="${1:-cct_train}"
CORE="${2:-pe0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
GVSOC_INSTALL_DIR="${GVSOC_INSTALL_DIR:-$ROOT/install/gvsoc}"
# riscv binutils built for ubuntu glibc 2.33+ won't run on older hosts (RHEL8);
# prefer the host-runnable LLVM addr2line, fall back to the gap9 gcc one.
A2L="${A2L:-$ROOT/install/llvm/bin/llvm-addr2line}"
[ -x "$A2L" ] || A2L="$ROOT/install/gcc/gap9/bin/riscv32-unknown-elf-addr2line"
BUILD="$ROOT/DeeployTest/TEST_SIRACUSA/build_master"
ELF="$BUILD/bin/$TEST_NAME"
GVMAKE="$BUILD/DeeployTest/Platforms/Siracusa/CMakeFiles/gvsoc_${TEST_NAME}.dir/build.make"
OUT="/tmp/sira_hang_${TEST_NAME}_${CORE}.trace"
GARBAGE='c.unimp'
STALL_NEEDED="${STALL_NEEDED:-10}"   # ~50s no new pe trace => core idle/hung

[ -x "$ELF" ] || { echo "ERROR: binary not found: $ELF"; exit 1; }
CMD="$(grep -oaE "[^ ]*gvsoc/bin/gvsoc --target=siracusa .*image flash run" "$GVMAKE" 2>/dev/null | head -1)"
[ -n "$CMD" ] || { echo "ERROR: could not extract gvsoc command"; exit 1; }
# insert --trace before the trailing "image flash run"
TRACED="${CMD% image flash run} --trace=/chip/cluster/$CORE/insn image flash run"

export GVSOC_INSTALL_DIR
export LD_LIBRARY_PATH="$GVSOC_INSTALL_DIR/models/debug:$LD_LIBRARY_PATH"
unset PYTHONPATH
cd "$BUILD" || exit 1

echo ">>> tracing /chip/cluster/$CORE/insn for $TEST_NAME (out: $OUT)"
rm -f "$OUT"
stdbuf -oL -eL bash -c "$TRACED" > "$OUT" 2>&1 &
GVPID=$!
reason="?"; prevsz=-1; stall=0
while kill -0 $GVPID 2>/dev/null; do
  sleep 5
  sz=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  cyc=$(grep -a "$CORE/insn" "$OUT" 2>/dev/null | tail -1 | awk -F: '{print $2}' | tr -dc 0-9)
  hit=$(grep -ac "$GARBAGE" "$OUT" 2>/dev/null)
  [ "$sz" -eq "$prevsz" ] && stall=$((stall+1)) || stall=0; prevsz=$sz
  echo "[$(date +%H:%M:%S)] $((sz/1048576))MB cyc=${cyc:-?} c.unimp=${hit} stall=${stall}/${STALL_NEEDED}"
  if [ "$hit" -gt 0 ]; then reason="PC-CORRUPTION (c.unimp trap loop)"; kill -9 $GVPID 2>/dev/null; pkill -9 gvsoc_launcher; break; fi
  if grep -qa "Invalid fetch\|Invalid access" "$OUT" 2>/dev/null; then reason="Invalid fetch/access — core aborted"; kill -9 $GVPID 2>/dev/null; pkill -9 gvsoc_launcher; break; fi
  if [ "$stall" -ge "$STALL_NEEDED" ]; then reason="$CORE STALLED at cyc ${cyc} (idle/hung)"; kill -9 $GVPID 2>/dev/null; pkill -9 gvsoc_launcher; break; fi
done
[ "$reason" = "?" ] && reason="GVSoC exited on its own — NO hang detected"
echo ""
echo "===================== RESULT: $reason ====================="
LN=$(grep -an "$GARBAGE" "$OUT" | head -1 | cut -d: -f1)
[ -z "$LN" ] && LN=$(grep -an "$CORE/insn" "$OUT" | tail -1 | cut -d: -f1)
echo "last_cycle=$(grep -a "$CORE/insn" "$OUT" 2>/dev/null | tail -1 | awk -F: '{print $2}')  pe_trace_lines=$(grep -ac "$CORE/insn" "$OUT" 2>/dev/null)"
if [ -n "$LN" ]; then
  echo "--- distinct functions near the stop (call sequence tail) ---"
  grep -a "$CORE/insn" "$OUT" | tail -8000 | sed -E 's/\x1b\[[0-9]*m//g' | grep -oE '\] [A-Za-z0-9_]+:[0-9]' | awk '{print $2}' | awk '!s[$0]++' | tail -30
  echo "--- addr2line of the 14 PCs before the stop ---"
  for pc in $(sed -n "$((LN-14)),$((LN-1))p" "$OUT" | sed -E 's/\x1b\[[0-9]*m//g' | grep -oE ' [0-9a-f]{6,8} ' | awk '{print $1}' | tail -14); do
    printf "0x%s -> " "$pc"; "$A2L" -f -e "$ELF" "0x$pc" 2>/dev/null | tr '\n' ' '; echo
  done
fi
