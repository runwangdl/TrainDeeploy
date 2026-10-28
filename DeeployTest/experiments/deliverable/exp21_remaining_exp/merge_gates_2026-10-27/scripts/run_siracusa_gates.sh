#!/usr/bin/env bash
# merge gates 2026-10-27, Siracusa part. Run INSIDE traindeeploy:  bash run_siracusa_gates.sh
# 1) exp21e's Siracusa gates (zo/qzo single, full-net QZO+ECO fused/unfused, block-0 kernels, BP single step) on the
#    merged tree, tag "merged"; compared to the exp21e branch's own gate logs (gate_base_* = gate_fixed_* of exp21e)
# 2) Siracusa SpeechNet CI subset (pytest -k "siracusa and (SpeechNet or speechnet)") and zo_training         -- QW
WT=/app/ETH/TrainDeeploy
G=$WT/DeeployTest/experiments/deliverable/exp21_remaining_exp/merge_gates_2026-10-27; O=$G/logs
export PYTHONPATH=$WT
cd $WT/DeeployTest
python3 -c 'import Deeploy; assert Deeploy.__file__.startswith("'$WT'/"), Deeploy.__file__' || exit 9
bash experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift/fix/scripts/run_gates_siracusa.sh $WT merged $O
echo "gates done" >> $O/siracusa_summary.txt
PYTEST_XDIST_WORKER=mg1027_sir_ci python3 -m pytest test_platforms.py -v -k "siracusa and (SpeechNet or speechnet)" > $O/siracusa_ci_speechnet.log 2>&1
echo "EXIT $?" >> $O/siracusa_ci_speechnet.log
PYTEST_XDIST_WORKER=mg1027_sir_zo python3 -m pytest test_platforms.py -v -m zo_training > $O/siracusa_ci_zo_training.log 2>&1
echo "EXIT $?" >> $O/siracusa_ci_zo_training.log
for f in siracusa_ci_speechnet siracusa_ci_zo_training; do
  echo "$f: $(grep -aE '^=+ .*(passed|failed|error).* =+$' $O/$f.log | tail -1)" >> $O/siracusa_summary.txt
done
echo DONE >> $O/siracusa_summary.txt
