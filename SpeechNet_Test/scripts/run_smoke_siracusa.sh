#!/usr/bin/env bash
# GVSoC smoke runs of the short fixtures on Siracusa, from their SpeechNet_Test location.
# Run from the HOST; needs the `traindeeploy` container. Each run gets a private build dir.
#   bash run_smoke_siracusa.sh [bp_single zo_single qzo_single unfused_cluster fused_cluster]
#   bash run_smoke_siracusa.sh b1 b2 b3 b4      blocks 1-4 conv kernels, unfused (one pass) and fused (both passes)
set -uo pipefail
H=/Users/qiwenwu/ETH/TrainDeeploy/SpeechNet_Test
T=../SpeechNet_Test                      # relative to /app/ETH/TrainDeeploy/DeeployTest
MEM="--l1 128000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max --cores 8"
PIN="--input-type-map blocks.0.conv.bias_rqsadd=int32_t --input-offset-map blocks.0.conv.bias_rqsadd=0"
mkdir -p "$H/logs/smoke"

run() {  # $1 = name, $2 = command run inside DeeployTest
  docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA/build_st_$1 &&
    PYTEST_XDIST_WORKER=st_$1 $2" > "$H/logs/smoke/siracusa_$1.log" 2>&1
  echo "$1: $(grep -aE 'Runtime: [0-9]+|Errors: [0-9]+ out of [0-9]+|PASSED|FAILED' "$H/logs/smoke/siracusa_$1.log" | tail -3 | tr '\n' ' ')"
}

bp_single()   { run bp_single  "python3 deeployTrainingRunner_tiled_siracusa.py -t $T/speechnet_train_BP_single_step \
  --n-steps 1 --n-accum 1 --num-data-inputs 2 --l2 1500000 $MEM -D BN_FROZEN_STATS=ON"; }
zo_single()   { run zo_single  "python3 deeployMezoRunner_tiled_siracusa.py -t $T/speechnet_train_ZO_single_step \
  --optimizer-dir $T/speechnet_update_ZO_single_step --n-steps 1 --n-accum 1 --num-data-inputs 2 \
  --eps 0.01 --lr 3e-6 --q 1 --seed 42 --l2 2000000 $MEM -D BN_FROZEN_STATS=ON"; }
qzo_single()  { run qzo_single "python3 deeployMezoRunner_tiled_siracusa.py -t $T/speechnet_train_QZO_single_step \
  --optimizer-dir $T/speechnet_update_QZO_single_step --n-steps 1 --n-accum 1 --num-data-inputs 2 \
  --eps 0.01 --lr 1e-5 --q 1 --seed 42 --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8 -D BN_FROZEN_STATS=ON"; }
unfused_cluster() { run unfused_cluster "python3 deeployRunner_tiled_siracusa.py -t $T/speechnet_unfused_b0_cluster \
  --l2 1500000 $MEM $PIN"; }
fused_cluster()   { run fused_cluster   "python3 deeployRunner_tiled_siracusa.py -t $T/speechnet_fused_b0_cluster \
  --l2 1500000 $MEM $PIN"; }

# blocks 1-4 (exp7, 2026-10-14): settings of exp7 run_block.sh; reference cycles in exp7 results_step5.txt
blk() {  # $1 = block i
  local i=$1 A="/blocks_${1}_conv_input_quant_1/Clip_output_0" B="blocks.${1}.conv.bias_rqsadd"
  local M="--l1 128000 --l2 1500000 --cores 8"
  local T1="--input-type-map $B=int32_t $A=int8_t --input-offset-map $B=0 $A=0"
  local T2="--input-type-map $B=int32_t $A=int8_t ${A}_minus=int8_t --input-offset-map $B=0 $A=0 ${A}_minus=0"
  run b${i}_unfused_cluster "python3 deeployRunner_tiled_siracusa.py -t $T/speechnet_unfused_b${i}_cluster $M $T1"
  run b${i}_fused_cluster   "python3 deeployRunner_tiled_siracusa.py -t $T/speechnet_fused_b${i}_cluster $M $T2"
}
b1() { blk 1; }; b2() { blk 2; }; b3() { blk 3; }; b4() { blk 4; }

targets=("$@"); [ ${#targets[@]} -eq 0 ] && targets=(unfused_cluster fused_cluster bp_single zo_single qzo_single)
docker exec traindeeploy bash -lc 'pgrep -f "[g]vsoc_launcher" | xargs -r kill -9' || true
for t in "${targets[@]}"; do $t; done
