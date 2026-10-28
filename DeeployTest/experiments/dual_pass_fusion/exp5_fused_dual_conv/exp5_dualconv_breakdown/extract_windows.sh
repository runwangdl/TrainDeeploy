#!/usr/bin/env bash
# Reproduce inputs/ from the raw core-0 trace of exp5's own build (TEST_SIRACUSA/build_dual). -- QW
# Run on the Mac (ROOT default) or in the traindeeploy container (ROOT=/app/ETH/TrainDeeploy/DeeployTest).
set -euo pipefail
ROOT=${ROOT:-/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest}
D=$ROOT/experiments/dual_pass_fusion/exp5a_breakdown/logs/debug     # dual_trace_pe0.log lives here (git-ignored)
B=$ROOT/TEST_SIRACUSA/build_dual/bin
W=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W/inputs"

# 0. raw trace, if absent (container; ~20 min):
#    cd $ROOT/TEST_SIRACUSA/build_dual/gvsoc_workdir && /app/install/gvsoc/bin/gvsoc --target=siracusa \
#      --binary ../bin/QZO_DualConv --work-dir=. --trace=pe0/insn --trace=icache/l0_bank0 image flash run 2>&1 \
#      | grep -E "pe0/insn|l0_bank0" > $D/dual_trace_pe0.log

# 1. kernel address range from the disassembly
START=$(grep -oE "^[0-9a-f]{8} <PULP_DualConv2d_i8_i8_i8_HWC>:" "$B/QZO_DualConv.s" | cut -c1-8)
END=$(grep -oE "^[0-9a-f]{8} <" "$B/QZO_DualConv.s" | cut -c1-8 | awk -v s="$START" '$0 > s' | head -1)
echo "kernel range [$START, $END)"
cp "$B/QZO_DualConv.s" "$W/inputs/QZO_DualConv.s"

# 2. the conv node = every core-0 instruction inside that range (both tiles; the inter-tile DMA gap is
#    charged to tile 0's last instruction and must be subtracted for kernel-only time, see Findings §1)
grep "pe0/insn" "$D/dual_trace_pe0.log" \
  | awk -v s="$START" -v e="$END" '{for(i=1;i<=NF;i++) if($i=="M"){pc=$(i+1); break}} pc>=s && pc<e' \
  > "$W/inputs/dual_conv_only.log"

# 3. per-pixel landmark (executed exactly 1227x = core 0's pixels over both tiles) and three warm pixels
L=$(grep -oE " M [0-9a-f]{8}" "$W/inputs/dual_conv_only.log" | sort | uniq -c | awk '$1==1227{print $3; exit}')
echo "landmark $L"
for N in 100 300 500; do
  a=$(grep -n "M $L " "$W/inputs/dual_conv_only.log" | sed -n "${N}p" | cut -d: -f1)
  b=$(grep -n "M $L " "$W/inputs/dual_conv_only.log" | sed -n "$((N+1))p" | cut -d: -f1)
  sed -n "${a},$((b-1))p" "$W/inputs/dual_conv_only.log" > "$W/inputs/dual_pixel$N.log"
done

# 4. per-tile prefixes: kernel entry -> first loop-back of that tile (phase 1 + partition + pixel 0's body)
E2=$(grep -n "M $START " "$W/inputs/dual_conv_only.log" | sed -n 2p | cut -d: -f1)
P1=$(grep -n "M $L " "$W/inputs/dual_conv_only.log" | sed -n 1p | cut -d: -f1)
P2=$(grep -n "M $L " "$W/inputs/dual_conv_only.log" | sed -n 939p | cut -d: -f1)     # tile 0 = 938 pixels
sed -n "1,$((P1-1))p" "$W/inputs/dual_conv_only.log" > "$W/inputs/dual_tile0_prefix.log"
sed -n "${E2},$((P2-1))p" "$W/inputs/dual_conv_only.log" > "$W/inputs/dual_tile1_prefix.log"

# 5. line-annotated disassembly of the kernel (LLVM objdump: container path)
OD=${OD:-/app/install/llvm/bin/llvm-objdump}
if [ -x "$OD" ]; then
  "$OD" -d -l --no-show-raw-insn "$B/QZO_DualConv" \
    | awk '/<PULP_DualConv2d_i8_i8_i8_HWC>:/{p=1} p && /^$/{exit} p' > "$W/inputs/dual_lines.dis"
fi
echo "inputs ready: $(wc -l < "$W/inputs/dual_conv_only.log") instructions in the conv node"
