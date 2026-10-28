#!/usr/bin/env bash
# Reproduce inputs/ from the raw core-0 trace of exp5a's own build (TEST_SIRACUSA/build_e5a). -- QW
# Run on the Mac (paths under /Users/qiwenwu/ETH) or in the traindeeploy container (/app/ETH).
set -euo pipefail
ROOT=${ROOT:-/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest}
D=$ROOT/experiments/dual_pass_fusion/exp5a_breakdown/logs/debug     # e5a_trace_pe0.log lives here (git-ignored)
B=$ROOT/TEST_SIRACUSA/build_e5a/bin
W=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W/inputs"

# 0. the raw trace, if absent: record it from build_e5a (container only; ~20 min, ~110 MB)
#    cd $ROOT/TEST_SIRACUSA/build_e5a/gvsoc_workdir && /app/install/gvsoc/bin/gvsoc --target=siracusa \
#      --binary ../bin/QZO_Conv --work-dir=. --trace=pe0/insn --trace=icache/l0_bank0 image flash run 2>&1 \
#      | grep -E "pe0/insn|l0_bank0" > $D/e5a_trace_pe0.log

# 1. the kernel's address range, from the disassembly: start = its symbol, end = the next symbol
START=$(grep -oE "^[0-9a-f]{8} <PULP_FlatConv2d_i8_i8_i8_HWC>:" "$B/QZO_Conv.s" | cut -c1-8)
END=$(grep -oE "^[0-9a-f]{8} <" "$B/QZO_Conv.s" | cut -c1-8 | awk -v s="$START" '$0 > s' | head -1)
echo "kernel range [$START, $END)"
cp "$B/QZO_Conv.s" "$W/inputs/QZO_Conv.s"

# 2. every instruction core 0 executed inside that range = the conv node, all tiles (string compare: same-length hex)
grep "pe0/insn" "$D/e5a_trace_pe0.log" \
  | awk -v s="$START" -v e="$END" '{for(i=1;i<=NF;i++) if($i=="M"){pc=$(i+1); break}} pc>=s && pc<e' \
  > "$W/inputs/e5a_conv_only.log"

# 3. the per-pixel landmark: the PC executed exactly once per pixel of core 0 (1227 pixels)
L=$(grep -oE " M [0-9a-f]{8}" "$W/inputs/e5a_conv_only.log" | sort | uniq -c | awk '$1==1227{print $3; exit}')
echo "landmark $L"
for N in 100 300 500; do
  a=$(grep -n "M $L " "$W/inputs/e5a_conv_only.log" | sed -n "${N}p" | cut -d: -f1)
  b=$(grep -n "M $L " "$W/inputs/e5a_conv_only.log" | sed -n "$((N+1))p" | cut -d: -f1)
  sed -n "${a},$((b-1))p" "$W/inputs/e5a_conv_only.log" > "$W/inputs/e5a_pixel$N.log"
done

# 4. the line-annotated disassembly of the kernel (needs the LLVM objdump: container path shown)
OD=${OD:-/app/install/llvm/bin/llvm-objdump}
if [ -x "$OD" ]; then
  "$OD" -d -l --no-show-raw-insn "$B/QZO_Conv" \
    | awk '/<PULP_FlatConv2d_i8_i8_i8_HWC>:/{p=1} p && /^$/{exit} p' > "$W/inputs/e5a_lines.dis"
fi
echo "inputs ready: $(wc -l < "$W/inputs/e5a_conv_only.log") instructions in the conv node"
