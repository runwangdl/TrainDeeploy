#!/usr/bin/env bash
# Board_exp / BP: run the incremental chain on the GAP9 board, rounds ROUNDS (default "2 3 4"; round 1 is run the
# same way). Per round r (HOST script; uses the agitated_hugle export container and the deeploy_gap9_board container):
#   1. export_bp_round.sh r   (round r > 1 starts from carry/carry_r<r-1>.pt = round r-1's BOARD weights)
#   2. board_run.sh bp_r<r>   (540 steps = 40 epochs, n_accum 4, lr 3e-5; Run's BP board flags; DUMP_WEIGHTS)
#   3. eval_round.py --round r  (-> carry/carry_r<r>.pt, results/round<r>.json, accuracy vs the host chain)
# Stops at the first failure. Log: logs/chain.log. -- QW Board_exp
set -o pipefail
H=/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/experiments/Board_exp/BP; cd $H
for r in ${ROUNDS:-2 3 4}; do
  echo "=== round $r  $(date '+%F %T')"
  [ "$r" = 1 ] || bash scripts/export_bp_round.sh $r || { echo "export r$r failed"; exit 1; }
  docker exec -e TMO=2400 deeploy_gap9_board bash /app/Deeploy/DeeployTest/experiments/Board_exp/scripts/board_run.sh \
    bp_r$r deeployTrainingRunner_tiled_gap9.py -t experiments/Board_exp/BP/fixtures/speechnet_train_BP_r$r \
    --n-steps 540 --n-accum 4 --num-data-inputs 2 --l1 122000 --l2 1300000 --defaultMemLevel L3 --doublebuffer \
    --promoteToL2 --promoteToL2Strategy=traffic-per-peak --promoteToL2Headroom=131072 --memAllocStrategy MiniMalloc \
    --cores 8 -- BN_FROZEN_STATS=ON CC_STACK_SIZE=4096 DUMP_WEIGHTS=ON || { echo "board r$r failed"; exit 1; }
  grep -aq "WDUMP s=539" ../runs/bp_r$r/uart.log || { echo "r$r: no final weight dump"; exit 1; }
  /opt/anaconda3/bin/python3.11 scripts/eval_round.py --round $r || { echo "eval r$r failed"; exit 1; }
done
echo "=== chain done $(date '+%F %T')"
