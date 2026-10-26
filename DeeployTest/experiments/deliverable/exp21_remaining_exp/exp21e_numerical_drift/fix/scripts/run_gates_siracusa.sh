#!/usr/bin/env bash
# exp21e/fix: Siracusa SpeechNet gates (SpeechNet_Test/scripts/run_smoke_siracusa.sh settings) on a worktree, with
# DUMP_WEIGHTS so the trained weights can be compared bit for bit between two trees (Siracusa has no WSUM line):
#   zo_single, qzo_single (unfused MeZO steps), qzo_fused / qzo_unfused (full-network QZO+ECO update step, n_accum 4,
#   --zo-fused-dual cluster vs unfused), b0_unfused / b0_fused (block-0 QZO conv kernels), bp_single.
# Run INSIDE traindeeploy: bash run_gates_siracusa.sh <worktree> <tag> <logdir> [gate ...]          -- QW exp21e
WT=$1; TAG=$2; LD=$3; shift 3
export PYTHONPATH=$WT
cd $WT/DeeployTest
T=../SpeechNet_Test
MEM="--l1 128000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max --cores 8"
PIN="--input-type-map blocks.0.conv.bias_rqsadd=int32_t --input-offset-map blocks.0.conv.bias_rqsadd=0"
Q="-t $T/speechnet_train_QZOECO_full_fused_single_update_step --optimizer-dir $T/speechnet_update_QZOECO_full_fused_single_update_step"
run() { local n=$1; shift; PYTEST_XDIST_WORKER=x21e2_g_${TAG}_$n python3 -u "$@" > $LD/gate_${TAG}_$n.log 2>&1; echo "EXIT $?" >> $LD/gate_${TAG}_$n.log; }
gates=("$@"); [ ${#gates[@]} -eq 0 ] && gates=(zo_single qzo_single qzo_unfused qzo_fused b0_unfused b0_fused bp_single)
for g in "${gates[@]}"; do case $g in
  zo_single) run $g deeployMezoRunner_tiled_siracusa.py -t $T/speechnet_train_ZO_single_step --optimizer-dir $T/speechnet_update_ZO_single_step \
     --n-steps 1 --n-accum 1 --num-data-inputs 2 --eps 0.01 --lr 3e-6 --q 1 --seed 42 --l2 2000000 $MEM -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON ;;
  qzo_single) run $g deeployMezoRunner_tiled_siracusa.py -t $T/speechnet_train_QZO_single_step --optimizer-dir $T/speechnet_update_QZO_single_step \
     --n-steps 1 --n-accum 1 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8 \
     -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON ;;
  qzo_unfused) run $g deeployMezoRunner_tiled_siracusa.py $Q --n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 3e-3 --q 1 --seed 42 \
     --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8 -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON ;;
  qzo_fused) run $g deeployMezoRunner_tiled_siracusa.py $Q --n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 3e-3 --q 1 --seed 42 \
     --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8 --zo-fused-dual -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON ;;
  b0_unfused) run $g deeployRunner_tiled_siracusa.py -t $T/speechnet_unfused_b0_cluster --l2 1500000 $MEM $PIN ;;
  b0_fused) run $g deeployRunner_tiled_siracusa.py -t $T/speechnet_fused_b0_cluster --l2 1500000 $MEM $PIN ;;
  bp_single) run $g deeployTrainingRunner_tiled_siracusa.py -t $T/speechnet_train_BP_single_step --n-steps 1 --n-accum 1 --num-data-inputs 2 \
     --l2 1500000 $MEM -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON ;;
esac; done
