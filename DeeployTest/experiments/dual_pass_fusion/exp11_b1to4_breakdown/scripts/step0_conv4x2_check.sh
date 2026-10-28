#!/usr/bin/env bash
# exp11 step 0: the step-5 Conv4x2 (word gather + pixel/channel split, no fusion) is still bit-identical to pulp-nn.
# GAP9 (deeploy_gap9): blocks 1-4 unfused single-pass fixtures, kernel = 4x2 (step 5) and 4x2 step-2a.
# Run inside deeploy_gap9 from /app/Deeploy/DeeployTest. Logs -> exp11_b1to4_breakdown/logs/step0_*.log
source /app/install/gap9-sdk/.gap9-venv/bin/activate; source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest; O=experiments/dual_pass_fusion/exp11_b1to4_breakdown/logs; T=../SpeechNet_Test; S=${RUN_SUFFIX:-a}
for i in 1 2 3 4; do
  A="/blocks_${i}_conv_input_quant_1/Clip_output_0"; B="blocks.${i}.conv.bias_rqsadd"
  P="-t $T/speechnet_unfused_b${i}_cluster --l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8 --input-type-map $B=int32_t $A=int8_t --input-offset-map $B=0 $A=0 -D GAP9_NETWORK_O3=ON"
  for v in 4x2 4x2_step2a; do
    extra=""; [ $v = 4x2_step2a ] && extra="-D CONV4X2_STEP2A=ON"
    pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
    DEEPLOY_RQCONV_KERNEL=4x2 PYTEST_XDIST_WORKER=x11s0${S}_b${i}_$v python3 deeployRunner_tiled_gap9.py $P $extra > $O/step0_b${i}_$v.log 2>&1
    echo "b$i $v: $(grep -aE 'Errors: [0-9]+ out of [0-9]+|Runtime: [0-9]+' $O/step0_b${i}_$v.log | tr '\n' ' ') calls=$(grep -ac 'PULP_Conv4x2_i8_i8_i8_HWC(' TEST_GAP9/speechnet_unfused_b${i}_cluster/Network.c 2>/dev/null)"
  done
done
