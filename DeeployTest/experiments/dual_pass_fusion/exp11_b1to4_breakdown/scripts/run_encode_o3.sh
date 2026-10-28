#!/usr/bin/env bash
# exp11 step 3: NE16WeightEncode.c on the GAP9 -O3 list. Re-measure the fused NE16 row (C)
# plain + microbench, and the C probe (pipelined). Fresh build directories (the -O3 list is a CMake change).
# Logs: logs/o3enc_b<i>_<v>_{plain,mb}.log, logs/o3enc_b<i>_Cprobe_pipe.log. -- QW
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest
O=experiments/dual_pass_fusion/exp11_b1to4_breakdown/logs
M="--l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8 -D GAP9_NETWORK_O3=ON"
blocks=("$@"); [ ${#blocks[@]} -eq 0 ] && blocks=(1 2 3 4)
r() { # log worker fixture typeflags extra
  pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
  PYTEST_XDIST_WORKER=$2 python3 deeployRunner_tiled_gap9_w_ne16.py -t Tests/Kernels/Integer/$3 $M $4 $5 > $O/$1.log 2>&1
  echo "$1: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $O/$1.log | head -1) $(grep -aE '^Runtime: [0-9]+' $O/$1.log | head -1)"
}
for i in "${blocks[@]}"; do
  A="/blocks_${i}_conv_input_quant_1/Clip_output_0"; B="blocks.${i}.conv.bias_rqsadd"
  T1="--input-type-map $B=int32_t $A=int8_t --input-offset-map $B=0 $A=0"
  T2="--input-type-map $B=int32_t $A=int8_t ${A}_minus=int8_t --input-offset-map $B=0 $A=0 ${A}_minus=0"
  r o3enc_b${i}_C_plain  x11c_ne_plain QZO_DualConv2in_NE16P_B$i "$T2" ""
  r o3enc_b${i}_C_mb     x11c_ne_mb    QZO_DualConv2in_NE16P_B$i "$T2" "--profileMicrobenchmark"
  r o3enc_b${i}_Cprobe_pipe x11c_ne_probe QZO_DualConv2in_NE16P_B$i "$T2" "-D NE16DUAL_PROBE=ON"
done
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
