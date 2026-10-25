#!/usr/bin/env bash
# exp11 step 2: why NE16 loses on blocks 3-4. Fused NE16 fixture (C) with -D NE16DUAL_PROBE=ON:
#   NE16DualPerturbEncode / NE16DualBias: per-core work cycles of phase 1 (perturbation) and phase 2 (encode / row sums);
#   NE16DualConv2in: master-core cycles of column builds, NE16 waits and the rest (task set-up, dispatch).
# pipelined (shipped) and sequential (DEEPLOY_NE16_PIPELINE=0: column build and NE16 job strictly alternate, so the
# wait time = pure NE16 job time). Run inside deeploy_gap9. -- QW
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest
O=experiments/dual_pass_fusion/exp11_b1to4_breakdown/logs
M="--l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8 -D GAP9_NETWORK_O3=ON -D NE16DUAL_PROBE=ON"
blocks=("$@"); [ ${#blocks[@]} -eq 0 ] && blocks=(1 2 3 4)
for i in "${blocks[@]}"; do
  A="/blocks_${i}_conv_input_quant_1/Clip_output_0"; B="blocks.${i}.conv.bias_rqsadd"
  T2="--input-type-map $B=int32_t $A=int8_t ${A}_minus=int8_t --input-offset-map $B=0 $A=0 ${A}_minus=0"
  for mode in pipe seq; do
    pl=""; [ $mode = seq ] && pl=0
    pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
    log=$O/b${i}_Cprobe_${mode}.log
    env ${pl:+DEEPLOY_NE16_PIPELINE=$pl} PYTEST_XDIST_WORKER=x11a_ne_probe python3 deeployRunner_tiled_gap9_w_ne16.py \
      -t Tests/Kernels/Integer/QZO_DualConv2in_NE16P_B$i $M $T2 > $log 2>&1
    echo "b$i $mode: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $log | head -1) $(grep -aE '^Runtime: [0-9]+' $log | head -1)"
  done
done
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
