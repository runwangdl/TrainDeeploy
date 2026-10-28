#!/usr/bin/env bash
# GVSoC runs of the block-0..4 kernel fixtures on GAP9 (cluster and NE16), from SpeechNet_Test.
# Run from the HOST; needs the `deeploy_gap9` container (TrainDeeploy mounted at /app/Deeploy).
# Settings = exp6 (kernel harness deeploytest.c: usable L1 ~98 KB, hence --l1 92000).
#   bash run_smoke_gap9.sh [unfused_cluster fused_cluster unfused_NE16 fused_NE16]      block 0
#   bash run_smoke_gap9.sh b1 b2 b3 b4                                                  blocks 1-4, all four variants each
# QW 2026-10-14: --l1 86000. On the current branch the GAP9 kernel harness keeps the cluster stacks in L1, and 92000
#   no longer fits (the L1 arena allocation fails, also for exp6's own block-0 fixture); 86000 = Run's kernel setting.
set -uo pipefail
H=/Users/qiwenwu/ETH/TrainDeeploy/SpeechNet_Test
# P="--l1 92000 --l2 1500000 --defaultMemLevel L2 --cores 8 --input-type-map blocks.0.conv.bias_rqsadd=int32_t --input-offset-map blocks.0.conv.bias_rqsadd=0"
P="--l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8 --input-type-map blocks.0.conv.bias_rqsadd=int32_t --input-offset-map blocks.0.conv.bias_rqsadd=0"
ENV='source /app/install/gap9-sdk/.gap9-venv/bin/activate; source /app/install/gap9-sdk/configs/gap9_evk_audio.sh;
     export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation'
mkdir -p "$H/logs/smoke"

run() {  # $1 = name, $2 = runner, $3 = fixture
  docker exec deeploy_gap9 bash -lc "$ENV; cd /app/Deeploy/DeeployTest && rm -rf TEST_GAP9/build_sg_$1 TEST_GAP9_W_NE16/build_sg_$1 &&
    PYTEST_XDIST_WORKER=sg_$1 python3 $2 -t ../SpeechNet_Test/$3 $P" > "$H/logs/smoke/gap9_$1.log" 2>&1
  echo "$1: $(grep -aE 'Runtime: [0-9]+|Errors: [0-9]+ out of [0-9]+' "$H/logs/smoke/gap9_$1.log" | tail -2 | tr '\n' ' ')"
}
unfused_cluster() { run unfused_cluster deeployRunner_tiled_gap9.py         speechnet_unfused_b0_cluster; }  # exp6 A    734449
fused_cluster()   { run fused_cluster   deeployRunner_tiled_gap9.py         speechnet_fused_b0_cluster;   }  # exp6 C_cl 925593
unfused_NE16()    { run unfused_NE16    deeployRunner_tiled_gap9_w_ne16.py  speechnet_unfused_b0_NE16;    }  # exp6 B    379969
fused_NE16()      { run fused_NE16      deeployRunner_tiled_gap9_w_ne16.py  speechnet_fused_b0_NE16;      }  # exp6 C    541345

# blocks 1-4 (exp7): the activation input(s) are int8 and must be pinned with the requant bias.
#   unfused = ONE pass (x2 for both passes); fused = both passes. Reference cycles: exp7 results_step7_gap9.txt.
blk() {  # $1 = block i
  local i=$1 A="/blocks_${1}_conv_input_quant_1/Clip_output_0" B="blocks.${1}.conv.bias_rqsadd"
  local M="--l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8"
  local T1="--input-type-map $B=int32_t $A=int8_t --input-offset-map $B=0 $A=0"
  local T2="--input-type-map $B=int32_t $A=int8_t ${A}_minus=int8_t --input-offset-map $B=0 $A=0 ${A}_minus=0"
  P="$M $T1"; run b${i}_unfused_cluster deeployRunner_tiled_gap9.py        speechnet_unfused_b${i}_cluster
  P="$M $T1"; run b${i}_unfused_NE16    deeployRunner_tiled_gap9_w_ne16.py speechnet_unfused_b${i}_NE16
  P="$M $T2"; run b${i}_fused_cluster   deeployRunner_tiled_gap9.py        speechnet_fused_b${i}_cluster
  P="$M $T2"; run b${i}_fused_NE16      deeployRunner_tiled_gap9_w_ne16.py speechnet_fused_b${i}_NE16
}
b1() { blk 1; }; b2() { blk 2; }; b3() { blk 3; }; b4() { blk 4; }

targets=("$@"); [ ${#targets[@]} -eq 0 ] && targets=(unfused_NE16 fused_NE16 unfused_cluster fused_cluster)
docker exec deeploy_gap9 bash -lc 'pgrep -f "[g]vsoc_launcher" | xargs -r kill -9' || true
for t in "${targets[@]}"; do $t; done
