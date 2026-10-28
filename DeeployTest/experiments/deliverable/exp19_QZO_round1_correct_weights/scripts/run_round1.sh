#!/usr/bin/env bash
# exp19 -- QZO (quantized ZO) round-1 on-device FT on the artifacts_reference base weights.
#   bash run_round1.sh [1|2|3|4|5|all]
#
# QZO differs from ZO by more than a checkpoint path: the int8 activation thresholds are CALIBRATED
# from the pretrained network, so new base weights => new quantization scales. Phase 1 regenerates them.
set -euo pipefail
EXP=/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp19_QZO_round1_correct_weights
EXPC=/app/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp19_QZO_round1_correct_weights  # traindeeploy
EXPH=/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp19_QZO_round1_correct_weights      # agitated_hugle
CKPT=/app/SilentWear/SilentWear/artifacts_reference/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
DATA=/app/SilentWear/SilentWear_data/data_raw_and_filt
SRC=QZO_exp/exp19_ref_full
TRAIN=Tests/Models/Training/SpeechNet/speechnet_qzo19_train
UPDATE=Tests/Models/Training/SpeechNet/speechnet_qzo19_update
PHASE="${1:-all}"; run(){ [[ "$PHASE" == all || "$PHASE" == "$1" ]]; }
mkdir -p "$EXP"/{logs,results,fixture,qinfer}

# 1 -- FRESH pooled@99.99 calibration + device-faithful PyTorch fc-float reference.
#      Needs QZO_ARTIFACTS_DIR (default artifacts_reference); set to "artifacts" to reproduce exp12.
run 1 && docker exec agitated_hugle bash -lc "cd $EXPH && python3 -u run_ref_and_calib.py" \
  > "$EXP/logs/ref_calib_stdout.log" 2>&1

# 2 -- export the QZO fixture with the FRESH thresholds.
#      --n-epochs 200 => n_batches=10800 => the FULL 21600-forward reference.
#      (exp12 used the default 1 epoch and therefore checked only 104 of 21600 forwards.)
run 2 && docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && \
  QZO_POOLED_THRESHOLDS=$EXPH/fixture/pooled_9999_fold3_ref.json python3 -u Onnx4Deeploy.py -model SpeechNet \
    -mode q-zo-train --noise-type rqs_rademacher --dataset silentwear --data-path $DATA \
    --pretrained-weights $CKPT --subject S01 --session 3 --condition vocalized --batch 1 \
    --n-epochs 200 --data-size 54 --stratified --n-accum 4 --lr 1e-5 -o $SRC" \
  > "$EXP/logs/phase2_export_qzo_fullref.log" 2>&1

# 3 -- pack + device QZO round, 2700 steps, ffast-math (NO DEEPLOY_STRICT_FP32). ~12 h, detached.
run 3 && docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && \
  python3 experiments/zo_smoke/pack_2step_fixture.py /app/ETH/Onnx4Deeploy/$SRC \
    Tests/Models/Training/SpeechNet speechnet_qzo19_train speechnet_qzo19_update" \
  2>&1 | tee "$EXP/logs/phase3_pack.log"
run 3 && docker exec -d traindeeploy bash -lc "
  cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA/build_qzo
  PYTEST_XDIST_WORKER=qzo nohup python3 -u deeployMezoRunner_tiled_siracusa.py \
    -t $TRAIN --optimizer-dir $UPDATE \
    --n-steps 2700 --n-accum 4 --num-data-inputs 2 \
    --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
    --l1 128000 --l2 2000000 --cores 8 \
    -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON > $EXPC/logs/phase3_gvsoc_qzo_round1.log 2>&1 &"

# 4 -- dumped weights -> rebuilt quantized inference fixture (offset-corrected injection)
run 4 && docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1 && \
  python3 extract_qzo_weights.py --gvsoc-log $EXPC/logs/phase3_gvsoc_qzo_round1.log \
    --train-onnx /app/ETH/Onnx4Deeploy/$SRC/network_zo_train.onnx \
    --out $EXPC/results/dumped_weights.npz" 2>&1 | tee "$EXP/logs/phase4_extract.log"
run 4 && docker exec agitated_hugle bash -lc "cd $EXPH && python3 -u \
  ../exp12_QZO_clean_round_1/build_qzo_infer_fixture11.py \
    --fixture-dir /app/Onnx4Deeploy/$SRC --dump-npz $EXPH/results/dumped_weights.npz \
    --out-dir $EXPH/qinfer" 2>&1 | tee "$EXP/logs/phase4_build_qinfer.log"

# 5 -- untiled device inference over all 180 batch-2 windows
run 5 && docker exec -d traindeeploy bash -lc "
  pgrep -f '[g]vsoc_launcher' | xargs -r kill -9
  cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA/build_qzoeval
  PYTEST_XDIST_WORKER=qzoeval nohup python3 -u experiments/deliverable/exp9_QZO_round1/qzo_accuracy_eval_untiled.py \
    --infer-dir $EXPC/qinfer --cores 8 > $EXPC/logs/phase5_eval_b2_device.log 2>&1 &"
