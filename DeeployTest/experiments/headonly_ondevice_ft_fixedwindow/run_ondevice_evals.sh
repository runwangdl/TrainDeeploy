#!/bin/bash
# Run the three on-device (GVSoC, untiled Siracusa) inference accuracy evals sequentially.
# Each eval = 180 per-sample GVSoC inferences via speechnet_accuracy_eval_untiled.py.
# Must run AFTER the training run has released TEST_SIRACUSA (they share build_master).
set -u
cd /app/ETH/TrainDeeploy/DeeployTest
EXP=experiments/headonly_ondevice_ft_fixedwindow
HARNESS=$EXP/speechnet_accuracy_eval_untiled.py
[ -f $HARNESS ] || HARNESS=experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py

for name in b1_zs b2_zs b2_ft; do
  dir=Tests/Models/speechnet_infer_${name}_isft_fw
  echo "================ EVAL $name  ($dir) ================"
  python experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
      --infer-dir $dir --cores 8 > $EXP/eval_${name}.log 2>&1
  cp -f speechnet_accuracy_results_untiled.json $EXP/results_${name}.json 2>/dev/null
  echo "---- $name summary ----"
  grep -iE "Balanced accuracy|Overall|correct" $EXP/eval_${name}.log | tail -3
done
echo "ALL EVALS DONE"
