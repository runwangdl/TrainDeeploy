#!/usr/bin/env bash
# exp18 -- ZO (MeZO, forward-only) round-1 on-device FT on the artifacts_reference base weights.
# Run from the HOST with `agitated_hugle` and `traindeeploy` up.
#   bash run_round1.sh [1|2|3|4|5|6|all]
#
# Container path prefixes DIFFER:  agitated_hugle -> /app/TrainDeeploy ;  traindeeploy -> /app/ETH/TrainDeeploy
set -euo pipefail
EXP=/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp18_ZO_round1_correct_weights
EXPC=/app/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp18_ZO_round1_correct_weights   # traindeeploy view
CKPT=/app/SilentWear/SilentWear/artifacts_reference/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
DATA=/app/SilentWear/SilentWear_data/data_raw_and_filt
SRC=./onnx/model/speechnet_zo_b1_fold3_ref
TRAIN=Tests/Models/Training/SpeechNet/speechnet_zo_train_b1_fold3_ref
UPDATE=Tests/Models/Training/SpeechNet/speechnet_zo_update_b1_fold3_ref
CARRY=$EXPC/results/carry_zo_b1_fold3_ref.pt
PHASE="${1:-all}"; run(){ [[ "$PHASE" == all || "$PHASE" == "$1" ]]; }
mkdir -p "$EXP"/{logs,results}

# 1 -- export the ZO fixture. --n-epochs 200 gives the FULL 21600-forward reference (not a prefix).
run 1 && docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && python3 Onnx4Deeploy.py -model SpeechNet -mode zo-train \
  --noise-type rademacher --bn-frozen-stats --dataset silentwear --data-path $DATA \
  --pretrained-weights $CKPT --subject S01 --session 3 --condition vocalized --batch 1 \
  --n-epochs 200 --data-size 54 --stratified --n-accum 4 --lr 3e-6 -o $SRC" \
  > "$EXP/logs/phase1_export_zo_train.log" 2>&1

# 2 -- pack into the TWO-dir layout (train graph + update graph). Both paths are passed to the runner.
run 2 && docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && \
  python3 experiments/zo_smoke/pack_2step_fixture.py /app/ETH/Onnx4Deeploy/${SRC#./} \
    Tests/Models/Training/SpeechNet speechnet_zo_train_b1_fold3_ref speechnet_zo_update_b1_fold3_ref" \
  2>&1 | tee "$EXP/logs/phase2_pack.log"

# 3 -- device ZO round, 2700 steps (~12 h). DETACHED + private build dir so it can run beside exp19.
run 3 && docker exec -d traindeeploy bash -lc "
  cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA/build_zo
  PYTEST_XDIST_WORKER=zo nohup python3 -u deeployMezoRunner_tiled_siracusa.py \
    -t $TRAIN --optimizer-dir $UPDATE \
    --n-steps 2700 --n-accum 4 --num-data-inputs 2 \
    --eps 0.01 --lr 3e-6 --q 1 --seed 42 \
    --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max --cores 8 \
    -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON > $EXPC/logs/phase3_gvsoc_zo_round1.log 2>&1 &"

# 4 -- [WDUMP s=2699] -> carry ckpt. Write to the BIND MOUNT: /tmp is container-local.
run 4 && docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp5_ZO_round1 && \
  python3 extract_zo_weights.py --gvsoc-log $EXPC/logs/phase3_gvsoc_zo_round1.log \
    --base-ckpt $CKPT --out-carry $CARRY" 2>&1 | tee "$EXP/logs/phase4_extract_weights.log"

# 5 -- inference fixture for batch 2 with the device-trained weights
run 5 && docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && python3 Onnx4Deeploy.py -model SpeechNet -mode infer \
  -o /app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_zo_infer_b1_fold3_ref \
  --dataset silentwear --data-path $DATA \
  --pretrained-weights /app/TrainDeeploy/DeeployTest/experiments/deliverable/exp18_ZO_round1_correct_weights/results/carry_zo_b1_fold3_ref.pt \
  --subject S01 --session 3 --batch 2 --condition vocalized" > "$EXP/logs/phase5_export_infer_b2.log" 2>&1

# 6 -- device eval on batch 2 (180 windows). Fresh build tree: CMake caches TRAINING.
run 6 && docker exec -d traindeeploy bash -lc "
  pgrep -f '[g]vsoc_launcher' | xargs -r kill -9
  cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA/build_zoeval
  PYTEST_XDIST_WORKER=zoeval nohup python3 -u experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
    --infer-dir Tests/Models/Training/SpeechNet/speechnet_zo_infer_b1_fold3_ref \
    > $EXPC/logs/phase6_device_eval_b2.log 2>&1 &"
