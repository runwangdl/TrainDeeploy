#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
#
# exp7 step 6: the block-1 pipeline (steps 0, 2a, 3+4) for any block i, run from the host.
#   bash run_block.sh <i>         i = 1..4
# Writes exp7<letter>_block<i>/{block<i>_unfused, block<i>_mid, block<i>_fused_cluster}/{logs,results,codegen}.
#   1. fixtures (agitated_hugle): QZO_Conv_B<i> (unfused, one pass) and QZO_DualConv2in_B<i> (fused, x+ / x-)
#   2. GVSoC runs (traindeeploy), each plain (valid Runtime) + --profileMicrobenchmark (per-node counters):
#        unfused  pulp_nn_conv_i8_i8_i8              (default mapping)
#        mid      PULP_Conv4x2_i8_i8_i8_HWC          (DEEPLOY_RQCONV_KERNEL=4x2)
#        fused    PULP_DualConv2d_2in_i8_i8_i8_HWC   (7-input RequantizedDualConv)
#   3. per-node tables via exp3's parse_microbench.py
set -euo pipefail
i=${1:?block 1..4}
letter=$(printf "\\x$(printf %x $((96 + i)))")          # 1 -> a, 2 -> b, ...
H=$(cd "$(dirname "$0")" && pwd)
REL=DeeployTest/experiments/dual_pass_fusion/exp7_remaining_blocks/exp7${letter}_block${i}
OUT_HOST=$H/exp7${letter}_block${i}
for v in unfused mid fused_cluster; do mkdir -p "$OUT_HOST/block${i}_$v"/{logs,results,codegen}; done

A="/blocks_${i}_conv_input_quant_1/Clip_output_0"
B="blocks.${i}.conv.bias_rqsadd"
MEM="--l1 128000 --l2 1500000 --cores 8"
FLAGS1="$MEM --input-type-map $B=int32_t $A=int8_t --input-offset-map $B=0 $A=0"
FLAGS2="$MEM --input-type-map $B=int32_t $A=int8_t ${A}_minus=int8_t --input-offset-map $B=0 $A=0 ${A}_minus=0"

echo "== block $i: fixtures"
docker exec agitated_hugle bash -lc "cd /app && export PYTHONPATH=/app/Onnx4Deeploy:/app/TrainDeeploy && \
  python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_qzo_conv_fixture.py --block $i 2>&1 \
    | grep -v -E 'cudart|warnings.warn|Custom ONNX|Inputs:|Outputs:|Internal ops' > /app/TrainDeeploy/$REL/block${i}_unfused/logs/build_fixture.log && \
  python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_dual_conv2in_fixture.py --block $i 2>&1 \
    | grep -v -E 'cudart|warnings.warn|Custom ONNX|Inputs:|Outputs:|Internal ops' > /app/TrainDeeploy/$REL/block${i}_fused_cluster/logs/build_fixture.log"

run() {  # $1 variant dir, $2 test, $3 env, $4 flags
  local L=/app/ETH/TrainDeeploy/$REL/block${i}_$1
  for mode in plain microbench; do
    local extra=""; [ $mode = microbench ] && extra="--profileMicrobenchmark"
    docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA && \
      (pgrep -f '[g]vsoc_launcher' | xargs -r kill -9 || true); \
      $3 python3 deeployRunner_tiled_siracusa.py -t Tests/Kernels/Integer/$2 $4 $extra > $L/logs/run_$mode.log 2>&1; \
      [ $mode = plain ] && cp TEST_SIRACUSA/Tests/Kernels/Integer/$2/Network.c $L/codegen/Network.c || true"
    echo "   $1 $mode: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' "$H/exp7${letter}_block${i}/block${i}_$1/logs/run_$mode.log" | head -1)  $(grep -aE '^Runtime:' "$H/exp7${letter}_block${i}/block${i}_$1/logs/run_$mode.log" | head -1)"
  done
  python3 "$H/../exp3_as-is_QZOConv_microbench/parse_microbench.py" "$OUT_HOST/block${i}_$1/logs/run_microbench.log" \
    "$OUT_HOST/block${i}_$1/results/per_node" > /dev/null
}

echo "== block $i: runs"
run unfused QZO_Conv_B$i "" "$FLAGS1"
run mid QZO_Conv_B$i "DEEPLOY_RQCONV_KERNEL=4x2" "$FLAGS1"
run fused_cluster QZO_DualConv2in_B$i "" "$FLAGS2"
