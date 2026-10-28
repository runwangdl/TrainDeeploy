#!/usr/bin/env bash
# exp21d -- one extra serial worker in agitated_hugle, started after the native BP/ZO set finished
# (keeps the total at 4 single-threaded jobs). Walks the optional p70_e120 quantized jobs in REVERSE
# order of launch.sh, so the two workers meet in the middle; a job both happen to run is simply
# recomputed (deterministic, identical JSON).
set -u
cd "$(dirname "$0")"
export TORCH_CPP_LOG_LEVEL=ERROR OMP_NUM_THREADS=1 CLEANUP_ROOT=/app/SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp
echo "$(date '+%F %T') launch extra (reverse optional arms)" >> progress.txt
for job in "qzo S04" "qzo S03" "qzo S02" "qzo S01" "eco S04" "eco S03" "eco S02" "eco S01"; do
  set -- $job; r=$1; s=$2; a=p70_e120
  if [ -f results/$r/$a/$s.json ] || [ -f logs/${r}_${a}_${s}.log ]; then echo "$(date "+%F %T") extra skip $r $a $s" >> progress.txt; continue; fi
  echo "$(date "+%F %T") start $r $a $s (extra)" >> progress.txt
  python3 -u run_job.py --recipe $r --arm $a --subject $s 2>&1 | grep -av NNPACK > logs/${r}_${a}_${s}.log
  echo "$(date "+%F %T") end   $r $a $s : $(grep -a DONE logs/${r}_${a}_${s}.log || echo FAILED)" >> progress.txt
done
echo "$(date '+%F %T') ALL DONE extra" >> progress.txt
