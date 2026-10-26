#!/usr/bin/env bash
# exp21e/fix task 3: re-export the ZO round-1 fixture with the OUT-OF-PLACE reference (Onnx4Deeploy branch
# fix/zo_ref_out_of_place, worktree .worktrees/exp21e_zo_ref). Run from the HOST.
# Recipe = SpeechNet_Test/scripts/export_fixtures.sh zo_round1 (exp18) except --n-epochs 120 (deployed board setting,
# 1620 update steps, as Run's pw_sw_build_all.sh --n-steps 1620) instead of 200 (2700). -- QW exp21e
set -euo pipefail
O4D=/app/Onnx4Deeploy/.worktrees/exp21e_zo_ref                       # agitated_hugle view
C=/app/TrainDeeploy/.worktrees/exp21e_bp_debug/SpeechNet_Test         # agitated_hugle view
H=/Users/qiwenwu/ETH/TrainDeeploy/.worktrees/exp21e_bp_debug/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift/fix
CKPT=$C/pretrained/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
DATA=/app/SilentWear/SilentWear_data/data_raw_and_filt
CELL="--dataset silentwear --data-path $DATA --subject S01 --session 3 --condition vocalized"
EXP=$C/exports/zo_round1_oop_120ep
docker exec agitated_hugle bash -lc "cd $O4D && export PYTHONPATH=$O4D OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 && \
  python3 -c 'import onnx4deeploy;print(\"onnx4deeploy from\", onnx4deeploy.__file__)' && \
  python3 -u Onnx4Deeploy.py -model SpeechNet -mode zo-train --noise-type rademacher --bn-frozen-stats $CELL --batch 1 \
  --pretrained-weights $CKPT --data-size 54 --stratified --n-epochs 120 --n-accum 4 --lr 3e-6 -o $EXP" \
  > $H/logs/export_zo_round1_oop_120ep.log 2>&1
docker exec agitated_hugle bash -lc "python3 /app/TrainDeeploy/.worktrees/exp21e_bp_debug/DeeployTest/experiments/zo_smoke/pack_2step_fixture.py \
  $EXP $C speechnet_train_ZO_round1_oop speechnet_update_ZO_round1_oop" >> $H/logs/export_zo_round1_oop_120ep.log 2>&1
