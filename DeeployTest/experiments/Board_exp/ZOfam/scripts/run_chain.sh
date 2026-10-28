#!/usr/bin/env bash
# Board_exp / ZO family: 4-round incremental chain of one arm on the GAP9 board (HOST script).
#   bash run_chain.sh <arm>        arms: zo | qzo_fused | eco_fused | eco_unfused        ROUNDS="1 2 3 4" (default)
# Selected settings (exp21l rule): 140 epochs = 1890 steps per round, n_accum 4, eps 0.01, q 1, seed 42;
#   zo          fp32 MeZO, lr 3e-6, unfused cluster                 (Run's ZO board flags, --l2 1300000)
#   qzo_fused   QZO plain, lr 1e-5, fused dual pass, NE16 blocks 0-2  (Run's best board config, --l2 1340000)
#   eco_fused   QZO-ECO (SR, memory-free EF, b .9), lr 3e-6, fused, NE16 blocks 0-2
#   eco_unfused QZO-ECO, lr 3e-6, unfused cluster (pulp-nn)        (same weights as eco_fused: WSUM must match)
# Per round r (Run's silentwear-zo rounds/run_rounds.sh, CARRY=host):
#   fixtures of round r (export_round.sh, paper init) are copied to fixtures_arm/<arm>/; for r > 1 the previous
#   round's BOARD weights ([WDUMP]) are written into them (carry_weights.py, Run's script, copied unchanged);
#   board_run.sh <arm>_r<r>; eval_round.py in agitated_hugle (exp21h model + metrics) -> results/<arm>/round<r>.json.
# Host protocol match: weights carried; ECO momentum, SR stream and z restart every round (fresh run). -- QW Board_exp
set -o pipefail
ARM=${1:?arm}
H=/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/experiments/Board_exp/ZOfam; C=experiments/Board_exp/ZOfam
cd $H
ZA="--n-steps 1890 --n-accum 4 --num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --l1 114000 --defaultMemLevel L2 --cores 8"
ZD="BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON ZO_QUIET=ON DUMP_WEIGHTS=ON"
FU="--zo-fused-dual --zo-fused-dual-ne16 0,1,2"
case $ARM in
  zo)          K=ZO;  M=zo;  R=deeployMezoRunner_tiled_GAP9.py;         X="--l2 1300000 --lr 3e-6";     D="$ZD" ;;
  qzo_fused)   K=QZO; M=qzo; R=deeployMezoRunner_tiled_GAP9_w_NE16.py;  X="--l2 1340000 --lr 1e-5 $FU"; D="$ZD" ;;
  eco_fused)   K=QZO; M=eco; R=deeployMezoRunner_tiled_GAP9_w_NE16.py;  X="--l2 1340000 --lr 3e-6 $FU"; D="$ZD ZO_ECO=ON ZO_ECO_SR=ON" ;;
  eco_unfused) K=QZO; M=eco; R=deeployMezoRunner_tiled_GAP9.py;         X="--l2 1340000 --lr 3e-6";     D="$ZD ZO_ECO=ON ZO_ECO_SR=ON" ;;
  *) echo "arm?"; exit 1 ;;
esac
prev=""
for r in ${ROUNDS:-1 2 3 4}; do
  echo "=== $ARM round $r  $(date '+%F %T')"
  A=fixtures_arm/$ARM; mkdir -p $A
  for k in train update; do rm -rf $A/speechnet_${k}_${K}_r$r; cp -R fixtures/speechnet_${k}_${K}_r$r $A/; done
  [ $r -gt 1 ] && prev=$H/../runs/${ARM}_r$((r-1))/uart.log
  if [ -n "$prev" ]; then
    for k in train update; do
      /opt/anaconda3/bin/python3.11 scripts/carry_weights.py $prev $A/speechnet_${k}_${K}_r$r/inputs.npz \
        $A/speechnet_train_${K}_r$r/network.onnx || { echo "carry r$r failed"; exit 1; }
    done
  fi
  docker exec -e TMO=3000 deeploy_gap9_board bash /app/Deeploy/DeeployTest/experiments/Board_exp/scripts/board_run.sh \
    ${ARM}_r$r $R -t $C/$A/speechnet_train_${K}_r$r --optimizer-dir $C/$A/speechnet_update_${K}_r$r $ZA $X -- $D \
    || { echo "board r$r failed"; exit 1; }
  grep -aq "WDUMP s=1889" ../runs/${ARM}_r$r/uart.log || { echo "$ARM r$r: no final weight dump"; exit 1; }
  docker exec agitated_hugle bash -lc "cd /app/TrainDeeploy/DeeployTest/experiments/Board_exp/ZOfam/scripts && \
    python3 eval_round.py --method $M --arm $ARM --round $r 2>&1 | grep -v -i warn | tail -5" || { echo "eval r$r failed"; exit 1; }
done
echo "=== $ARM chain done $(date '+%F %T')"
