#!/usr/bin/env bash
# Board_exp / ZO family: export the train + update fixtures of ONE round for the GAP9 board.
#   bash export_round.sh <zo|qzo> <round 1..4>
# Same exporter commands as TrainDeeploy/SpeechNet_Test (README rows 3 and 15) / silentwear-zo
# rounds/export_qzo999_rounds.sh, with --batch <round>:
#   zo  : -mode zo-train,  Rademacher,      frozen BN, fp32                         (ZO)
#   qzo : -mode q-zo-train, RQS Rademacher,  pooled@99.9 calibration (int8 convs)    (QZO and QZO-ECO share it)
# Cell S01 / vocalized / fold 3; 54 stratified windows of batch <round> (seed 42 = the exporter's draw, reproduced by
# the host runs); initial weights = paper checkpoint. A short host reference (14 update steps, 56 loss pairs) is
# enough: the board run sets its own step count (--n-steps) and lr (--lr); from round 2 on run_chain.sh replaces the
# initial weights by the previous round's BOARD weights (carry_weights.py), so the reference no longer applies.
# Output: fixtures/speechnet_{train,update}_<KIND>_r<round>/ (KIND = ZO | QZO). Runs from the HOST (agitated_hugle).
set -euo pipefail
K=${1:?zo|qzo}; R=${2:?round}
H=/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/experiments/Board_exp/ZOfam
C=/app/TrainDeeploy/DeeployTest/experiments/Board_exp/ZOfam
CKPT=/app/TrainDeeploy/SpeechNet_Test/pretrained/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
DATA=/app/SilentWear/SilentWear_data/data_raw_and_filt
THR=/app/TrainDeeploy/SpeechNet_Test/calibration/pooled_999_fold3_ref.json
PACK=/app/TrainDeeploy/DeeployTest/experiments/zo_smoke/pack_2step_fixture.py
CELL="--dataset silentwear --data-path $DATA --subject S01 --session 3 --condition vocalized --batch $R"
COMMON="$CELL --pretrained-weights $CKPT --data-size 54 --stratified --n-steps 14 --n-accum 4"
mkdir -p $H/logs $H/exports $H/fixtures
case $K in
  zo)  KIND=ZO;  ENV="";                              MODE="-mode zo-train --noise-type rademacher --bn-frozen-stats --lr 3e-6" ;;
  qzo) KIND=QZO; ENV="QZO_POOLED_THRESHOLDS=$THR";    MODE="-mode q-zo-train --noise-type rqs_rademacher --lr 1e-5" ;;
  *) echo "kind?"; exit 1 ;;
esac
E=$C/exports/${K}_b$R
docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && $ENV PYTHONDONTWRITEBYTECODE=1 python3 -u Onnx4Deeploy.py \
  -model SpeechNet $MODE $COMMON -o $E" > $H/logs/export_${K}_r$R.log 2>&1
docker exec agitated_hugle bash -lc "python3 $PACK $E $C/fixtures speechnet_train_${KIND}_r$R speechnet_update_${KIND}_r$R" \
  >> $H/logs/export_${K}_r$R.log 2>&1
echo "$K round $R -> $(ls -d $H/fixtures/speechnet_{train,update}_${KIND}_r$R | tr '\n' ' ')"
