#!/usr/bin/env bash
# Board_exp / BP: export the BP training fixture of ONE incremental round for the GAP9 board.
#   bash export_bp_round.sh <round 1..4> [start_ckpt.pt]
# Selected setting (exp21l rule, 4 subjects x 3 seeds, held-out current batch): constant lr 3e-5, 40 epochs,
# n_accum 4 (SUM), 54 stratified windows of batch <round> (30 %, seed 42 = the exporter's own draw, which the host
# runs reproduce with CleanUp draw="device", cyclic order), full model, frozen BN stats, argmax-mask MaxPool.
# Cell: S01, vocalized, fold 3 (session 3). Round 1 starts from the paper checkpoint (artifacts_reference, sha256
# 18ead42c...); round r > 1 starts from the carry checkpoint of round r-1's BOARD weights (carry/carry_r<r-1>.pt,
# made by extract_device_weights.py from the board's [WDUMP]). Same exporter command as
# silentwear-zo SpeechNet_Test/scripts/export_fixtures.sh bp_round1, except --lr / --batch / --pretrained-weights.
# Writes fixtures/speechnet_train_BP_r<round>/ (+ the optimizer graph next to it). Runs from the HOST, exports in
# agitated_hugle (ETH mounted at /app). -- QW Board_exp
set -euo pipefail
R=${1:?round}
H=/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/experiments/Board_exp/BP
C=/app/TrainDeeploy/DeeployTest/experiments/Board_exp/BP
PAPER=/app/TrainDeeploy/SpeechNet_Test/pretrained/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
if [ "$R" = 1 ]; then CKPT=${2:-$PAPER}; else CKPT=${2:-$C/carry/carry_r$((R-1)).pt}; fi
[ "$R" = 1 ] || [ -f "$H/carry/carry_r$((R-1)).pt" ] || [ -n "${2:-}" ] || { echo "missing carry_r$((R-1)).pt"; exit 1; }
DATA=/app/SilentWear/SilentWear_data/data_raw_and_filt
docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && PYTHONDONTWRITEBYTECODE=1 python3 -u Onnx4Deeploy.py \
  -model SpeechNet -mode train -o $C/fixtures/speechnet_train_BP_r$R \
  --dataset silentwear --data-path $DATA --subject S01 --session 3 --condition vocalized --batch $R \
  --pretrained-weights $CKPT --data-size 54 --stratified --n-epochs 40 --n-accum 4 --lr 3e-5 \
  --training-strategy full --bn-frozen-stats --maxpool-argmax-mask" > "$H/logs/export_bp_r$R.log" 2>&1
echo "round $R exported from $CKPT -> $(ls -d $H/fixtures/*_r$R)"
