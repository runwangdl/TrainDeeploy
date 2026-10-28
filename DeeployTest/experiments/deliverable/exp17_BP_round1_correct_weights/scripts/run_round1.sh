#!/usr/bin/env bash
# exp17 — BP round-1 on-device fine-tuning on the lab-machine ("reference") pretrained weights.
# Run from the HOST. Requires the `agitated_hugle` and `traindeeploy` containers to be up.
#
#   bash run_round1.sh [phase]        phase = 1|2|3|4|5|6|all   (default: all)
#
# NOTE the two containers mount the host tree at DIFFERENT prefixes:
#   agitated_hugle : /Users/qiwenwu/ETH -> /app        => TrainDeeploy is /app/TrainDeeploy
#   traindeeploy   : /Users/qiwenwu/ETH -> /app/ETH    => TrainDeeploy is /app/ETH/TrainDeeploy
set -euo pipefail

EXP=/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp17_BP_round1_correct_weights
CKPT=/app/SilentWear/SilentWear/artifacts_reference/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
DATA=/app/SilentWear/SilentWear_data/data_raw_and_filt
FIX_H=/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet          # agitated_hugle
FIX_T=Tests/Models/Training/SpeechNet                                        # traindeeploy (relative)
CARRY=/tmp/carry_fullfrozen_b1_fold3_ref.pt
mkdir -p "$EXP/logs" "$EXP/results"
PHASE="${1:-all}"
run() { [[ "$PHASE" == "all" || "$PHASE" == "$1" ]]; }

# --- phase 1: zero-shot b1 (gate) --------------------------------------------------------------
if run 1; then
  echo ">>> phase 1: export + device-eval zero-shot on batch 1"
  docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && python3 Onnx4Deeploy.py -model SpeechNet -mode infer \
    -o $FIX_H/speechnet_infer_fullfrozen_b0_fold3_ref \
    --dataset silentwear --data-path $DATA --pretrained-weights $CKPT \
    --subject S01 --session 3 --batch 1 --condition vocalized" > "$EXP/logs/phase1_export_zeroshot_b1.log" 2>&1
  docker exec traindeeploy bash -lc "pgrep -f '[g]vsoc_launcher' | xargs -r kill -9
    cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
    python3 experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
      --infer-dir $FIX_T/speechnet_infer_fullfrozen_b0_fold3_ref" > "$EXP/logs/phase1_device_eval_zeroshot_b1.log" 2>&1
  grep -E "Balanced accuracy" "$EXP/logs/phase1_device_eval_zeroshot_b1.log"
  echo "    expected 0.7222 == reference ft_summary.csv (S01/vocalized/fold 3, num_prev_ft_rounds=0)"
fi

# --- phase 2: export TRAIN fixture (recipe S2) --------------------------------------------------
if run 2; then
  echo ">>> phase 2: export train fixture (540 steps, 54 stratified windows)"
  docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && python3 Onnx4Deeploy.py -model SpeechNet -mode train \
    -o $FIX_H/speechnet_train_fullfrozen_b1_fold3_ref \
    --dataset silentwear --data-path $DATA --pretrained-weights $CKPT \
    --subject S01 --session 3 --batch 1 --condition vocalized \
    --data-size 54 --n-epochs 40 --n-accum 4 --lr 0.0003 \
    --training-strategy full --bn-frozen-stats --stratified \
    --maxpool-argmax-mask" > "$EXP/logs/phase2_export_train_b1.log" 2>&1
  grep -E "n_batches=|Stratified split" "$EXP/logs/phase2_export_train_b1.log"
fi

# --- phase 3: device TRAIN, 540 steps (MULTI-HOUR) ----------------------------------------------
if run 3; then
  echo ">>> phase 3: device train, 540 steps -- multi-hour; device stdout flushes only at main() return"
  docker exec traindeeploy bash -lc "pgrep -f '[g]vsoc_launcher' | xargs -r kill -9
    cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
    python3 deeployTrainingRunner_tiled_siracusa.py \
      -t $FIX_T/speechnet_train_fullfrozen_b1_fold3_ref \
      --n-steps 540 --n-accum 4 --cores 8 \
      --l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
      -D DUMP_WEIGHTS=ON BN_FROZEN_STATS=ON" > "$EXP/logs/phase3_gvsoc_train_round1.log" 2>&1
fi

# --- phase 4: device weights -> carry checkpoint -------------------------------------------------
if run 4; then
  echo ">>> phase 4: extract [WDUMP s=539] -> carry checkpoint"
  docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest/experiments/exp1/ondevice_sim_S01_fold3 && \
    python3 extract_device_weights.py \
      --gvsoc-log /app/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp17_BP_round1_correct_weights/logs/phase3_gvsoc_train_round1.log \
      --base-ckpt $CKPT --out-carry $CARRY" 2>&1 | tee "$EXP/logs/phase4_extract_weights.log"
fi

# --- phase 5: export INFER fixture for batch 2 with the carry weights ----------------------------
if run 5; then
  echo ">>> phase 5: export infer fixture for batch 2"
  docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && python3 Onnx4Deeploy.py -model SpeechNet -mode infer \
    -o $FIX_H/speechnet_infer_fullfrozen_b1_fold3_ref \
    --dataset silentwear --data-path $DATA --pretrained-weights $CARRY \
    --subject S01 --session 3 --batch 2 --condition vocalized" > "$EXP/logs/phase5_export_infer_b2.log" 2>&1
  grep -E "Balanced accuracy" "$EXP/logs/phase5_export_infer_b2.log"
fi

# --- phase 6: device EVAL on batch 2 -> the exp17 result -----------------------------------------
if run 6; then
  echo ">>> phase 6: device eval on batch 2 (180 windows)"
  docker exec traindeeploy bash -lc "pgrep -f '[g]vsoc_launcher' | xargs -r kill -9
    cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
    python3 experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
      --infer-dir $FIX_T/speechnet_infer_fullfrozen_b1_fold3_ref" > "$EXP/logs/phase6_device_eval_b2.log" 2>&1
  grep -E "Balanced accuracy|Overall accuracy" "$EXP/logs/phase6_device_eval_b2.log"
fi
