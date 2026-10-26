#!/usr/bin/env bash
# exp21c task 2: per-block breakdown of the fused NE16 dual conv (E) and the fused cluster dual conv (D), blocks 1-4,
# GAP9 GVSoC, exp11 settings (--l1 86000 --l2 1500000, GAP9_NETWORK_O3=ON, 8 cores). Run inside deeploy_gap9:
#   bash run_blocks.sh [blocks...]      (VARIANTS="E_plain E_mb ..." to run a subset)
# Fixtures are COPIES of Tests/Kernels/Integer/QZO_DualConv2in{,_NE16P}_B<i> in ../fixtures, so the generated code lands
# in TEST_GAP9*/experiments/deliverable/.../fixtures/<name> and never collides with another agent's run of the same test.
#   E_plain    fused NE16, default build                 -> Runtime (reference total)
#   E_mb       fused NE16, --profileMicrobenchmark       -> per-node wall (master core 8)
#   E_wall     fused NE16, -D NE16DUAL_PROBE_WALL=ON     -> phase split; Runtime must stay ~= E_plain
#   E_seqplain fused NE16, DEEPLOY_NE16_PIPELINE=0       -> sequential total (reference for E_wallseq)
#   E_wallseq  sequential + probe                        -> NE16 compute alone (wait) and column build alone
#   D_plain / D_mb / D_probe   fused cluster DualConv2In, the probe = -D DUALCONV2IN_PROBE=ON (exp11 Cpr)
# gvsoc is killed only by PID: gvsoc_launcher processes running in this script's own build directories (KILLPAT, default
# build_x21c_(ne|g9)_; set KILLPAT=build_x21c_ne_ / build_x21c_g9_ when an E-only and a D-only instance run concurrently). -- QW exp21c
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest
E=experiments/deliverable/exp21_remaining_exp/exp21c_Fusion_NE16_Encode_Opt; O=$E/logs; F=$E/fixtures; CG=$E/codegen
blocks=("$@"); [ ${#blocks[@]} -eq 0 ] && blocks=(1 2 3 4)
variants=(${VARIANTS:-E_plain E_mb E_wall E_seqplain E_wallseq D_plain D_mb D_probe})
M="--l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8 -D GAP9_NETWORK_O3=ON"

kill_mine() {  # only gvsoc_launcher processes whose working directory is one of THIS script's build dirs (build_x21c_(ne|g9)_*);
  # gvsoc_launcher's command line carries no build path, its cwd (<build>/gvsoc_workdir) does. Never by name. -- QW exp21c
  for p in $(pgrep -f "[g]vsoc_launcher" 2>/dev/null); do
    if readlink /proc/$p/cwd 2>/dev/null | grep -qE "${KILLPAT:-build_x21c_(ne|g9)_}"; then kill -9 $p 2>/dev/null; fi
  done
}

run() {  # tag block runner fixture worker extra-args...  (env passed via ENVS)
  local tag=$1 i=$2 runner=$3 fix=$4 wk=$5; shift 5
  local log=$O/b${i}_${tag}.log
  kill_mine
  env ${ENVS:-} PYTEST_XDIST_WORKER=$wk timeout 3000 python3 $runner -t $F/$fix $M "$@" > $log 2>&1
  kill_mine
  local d; d=$(ls -d TEST_GAP9*/$F/$fix 2>/dev/null | head -1)
  [[ $tag == *plain ]] && [ -n "$d" ] && cp $d/Network.c $CG/b${i}_${tag}_Network.c
  echo "b$i $tag: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $log | head -1) $(grep -aE '^Runtime: [0-9]+' $log | head -1)"
}

for i in "${blocks[@]}"; do
  A="/blocks_${i}_conv_input_quant_1/Clip_output_0"; B="blocks.${i}.conv.bias_rqsadd"
  T2="--input-type-map $B=int32_t $A=int8_t ${A}_minus=int8_t --input-offset-map $B=0 $A=0 ${A}_minus=0"
  NE=deeployRunner_tiled_gap9_w_ne16.py; G9=deeployRunner_tiled_gap9.py
  for v in "${variants[@]}"; do
    case $v in
      E_plain)    ENVS=""                        run E_plain    $i $NE QZO_DualConv2in_NE16P_B$i x21c_ne_plain   $T2 ;;
      E_mb)       ENVS=""                        run E_mb       $i $NE QZO_DualConv2in_NE16P_B$i x21c_ne_mb      $T2 --profileMicrobenchmark ;;
      E_wall)     ENVS=""                        run E_wall     $i $NE QZO_DualConv2in_NE16P_B$i x21c_ne_wall    $T2 -D NE16DUAL_PROBE_WALL=ON ;;
      E_seqplain) ENVS="DEEPLOY_NE16_PIPELINE=0" run E_seqplain $i $NE QZO_DualConv2in_NE16P_B$i x21c_ne_seq     $T2 ;;
      E_wallseq)  ENVS="DEEPLOY_NE16_PIPELINE=0" run E_wallseq  $i $NE QZO_DualConv2in_NE16P_B$i x21c_ne_wallseq $T2 -D NE16DUAL_PROBE_WALL=ON ;;
      D_plain)    ENVS=""                        run D_plain    $i $G9 QZO_DualConv2in_B$i       x21c_g9_plain   $T2 ;;
      D_mb)       ENVS=""                        run D_mb       $i $G9 QZO_DualConv2in_B$i       x21c_g9_mb      $T2 --profileMicrobenchmark ;;
      D_probe)    ENVS=""                        run D_probe    $i $G9 QZO_DualConv2in_B$i       x21c_g9_probe   $T2 -D DUALCONV2IN_PROBE=ON ;;
    esac
  done
done
kill_mine
