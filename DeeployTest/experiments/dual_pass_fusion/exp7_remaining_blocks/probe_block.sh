#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
#
# exp7 step 5: run block i's fused fixture once with -D DUALCONV2IN_PROBE=ON (per-phase core-0 cycles), and
# optionally the plain + profiled runs of the current kernel into results/per_node<tag>.csv.
#   bash probe_block.sh <i> <tag> [full]      e.g. bash probe_block.sh 4 _5b full
set -euo pipefail
i=${1:?block}; tag=${2:?tag}; full=${3:-}
letter=$(printf "\\x$(printf %x $((96 + i)))")
H=$(cd "$(dirname "$0")" && pwd)
REL=DeeployTest/experiments/dual_pass_fusion/exp7_remaining_blocks/exp7${letter}_block${i}/block${i}_fused_cluster
A="/blocks_${i}_conv_input_quant_1/Clip_output_0"; B="blocks.${i}.conv.bias_rqsadd"
FLAGS="--l1 128000 --l2 1500000 --cores 8 --input-type-map $B=int32_t $A=int8_t ${A}_minus=int8_t --input-offset-map $B=0 $A=0 ${A}_minus=0"
one() {  # $1 log name, $2 extra flags
  docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA && \
    (pgrep -f '[g]vsoc_launcher' | xargs -r kill -9 || true); \
    python3 deeployRunner_tiled_siracusa.py -t Tests/Kernels/Integer/QZO_DualConv2in_B$i $FLAGS $2 > /app/ETH/TrainDeeploy/$REL/logs/$1.log 2>&1" || true
  local f="$H/../../../../$REL/logs/$1.log"
  echo "  block $i $1: $(grep -aE 'Errors: [0-9]+ out of' "$f" | head -1)  $(grep -aE '^Runtime:' "$f" | head -1)  $(grep -aE '\[DC2PROBE\]' "$f" | head -1)"
}
one "run_probe$tag" "-D DUALCONV2IN_PROBE=ON ${EXTRA:-}"   # EXTRA: more -D flags, e.g. EXTRA="-D DUALCONV2IN_SPLIT=2"
if [ -n "$full" ]; then
  one "run_plain$tag" ""
  one "run_microbench$tag" "--profileMicrobenchmark"
  python3 "$H/../exp3_as-is_QZOConv_microbench/parse_microbench.py" "$H/../../../../$REL/logs/run_microbench$tag.log" \
    "$H/../../../../$REL/results/per_node$tag" > /dev/null
fi
