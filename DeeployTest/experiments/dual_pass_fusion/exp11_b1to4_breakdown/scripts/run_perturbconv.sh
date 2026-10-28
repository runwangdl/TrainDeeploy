#!/usr/bin/env bash
# exp11 step 4: single-pass conv with in-kernel perturbation (QZO_PerturbConv_B<i>, PULP_PerturbConv4x2_i8_i8_i8_HWC),
# the unfused reference with every non-fusion optimisation. GAP9 cluster, plain + microbench + probe (phase split).
# Same settings as run_breakdown.sh. Logs: logs/b<i>_Bpp_{plain,mb,probe,minus_plain}.log. -- QW
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest
E=experiments/dual_pass_fusion/exp11_b1to4_breakdown; O=$E/logs; CG=$E/codegen
M="--l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8 -D GAP9_NETWORK_O3=ON"
blocks=("$@"); [ ${#blocks[@]} -eq 0 ] && blocks=(1 2 3 4)
W=${WTAG:-x11d}
for i in "${blocks[@]}"; do
  A="/blocks_${i}_conv_input_quant_1/Clip_output_0"; B="blocks.${i}.conv.bias_rqsadd"
  T1="--input-type-map $B=int32_t $A=int8_t --input-offset-map $B=0 $A=0"
  for mode in plain mb probe; do
    extra=""; [ $mode = mb ] && extra="--profileMicrobenchmark"; [ $mode = probe ] && extra="-D DUALCONV2IN_PROBE=ON"
    pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
    log=$O/b${i}_Bpp_${mode}.log
    PYTEST_XDIST_WORKER=${W}_g9_$mode python3 deeployRunner_tiled_gap9.py -t Tests/Kernels/Integer/QZO_PerturbConv_B$i $M $T1 $extra > $log 2>&1
    d=$(ls -d TEST_GAP9/Tests/Kernels/Integer/QZO_PerturbConv_B$i 2>/dev/null | head -1)
    [ $mode = plain ] && [ -n "$d" ] && cp $d/Network.c $CG/b${i}_Bpp_Network.c
    echo "b$i Bpp $mode: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $log | head -1) $(grep -aE '^Runtime: [0-9]+' $log | head -1)"
  done
  # the - pass (attr force_sign = 0; input x-, reference A- from QZO_DualConv2in_B<i>): correctness check only
  pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
  log=$O/b${i}_Bpp_minus_plain.log
  PYTEST_XDIST_WORKER=${W}_g9_plain python3 deeployRunner_tiled_gap9.py -t Tests/Kernels/Integer/QZO_PerturbConvM_B$i $M $T1 > $log 2>&1
  echo "b$i Bpp minus: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $log | head -1) $(grep -aE '^Runtime: [0-9]+' $log | head -1)"
done
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
