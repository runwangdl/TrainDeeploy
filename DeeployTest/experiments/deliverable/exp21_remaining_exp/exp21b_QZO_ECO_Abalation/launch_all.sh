#!/usr/bin/env bash
# exp21b launcher (main study: run_arm.py default = 120 epochs = 1620 steps/round, the deployed recipe) -- run INSIDE agitated_hugle:
#   docker exec -d agitated_hugle bash -lc 'cd /app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21b_QZO_ECO_Abalation && bash launch_all.sh'
# Runs every (arm, subject) job of run_arm.py, PARALLEL jobs at a time (each single-threaded, so
# results are deterministic and independent of the parallelism). Finished jobs (json exists) are
# skipped, so re-running resumes. NNPACK warnings (one per forward, ~170 MB/job) are filtered out.
set -u
cd "$(dirname "$0")"
PARALLEL=${PARALLEL:-6}   # another agent shares agitated_hugle
export ADAM_MATCHED_LR=${ADAM_MATCHED_LR:-7e-5}    # = 3e-6 x RMS(g_proj)=24.5, pilot S01 round 1
ARMS=${ARMS:-"A0_qzo_deployed A4_eco_sr A1_sgd_rtn A2_sgdm_rtn A3_adam_sr A5_sgd_sr A6_sgdm_sr A7_eco_rtn A8_adam_rtn A10_sgd_carry_rtn A11_sgd_carry_sr A12_adam_sr_matched A13_sgd_sr_lr1e-5 A14_adam_rtn_matched"}
mkdir -p logs results
echo "$(date '+%F %T') launch: PARALLEL=$PARALLEL ADAM_MATCHED_LR=$ADAM_MATCHED_LR" >> progress.txt
for a in $ARMS; do for s in S01 S02 S03 S04; do echo "$a $s"; done; done | \
xargs -P "$PARALLEL" -L 1 bash -c '
  a=$0; s=$1
  if [ -f results/$a/$s.json ]; then echo "$(date "+%F %T") skip $a $s" >> progress.txt; exit 0; fi
  echo "$(date "+%F %T") start $a $s" >> progress.txt
  python3 run_arm.py --arm $a --subject $s 2>&1 | grep -v NNPACK > logs/${a}_${s}.log
  echo "$(date "+%F %T") end   $a $s : $(grep DONE logs/${a}_${s}.log || echo FAILED)" >> progress.txt
'
echo "$(date '+%F %T') ALL DONE" >> progress.txt
