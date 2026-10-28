#!/usr/bin/env bash
# exp21b SR-noise repeats: re-run the SR arms with the SR uniform stream reseeded (sr_seed 1, 2).
# Everything else (z stream, labelled draw, data order, calibration) is identical, so the spread
# across sr_seed {42 (main run), 1, 2} is the run-to-run noise due to stochastic rounding alone.
set -u
cd "$(dirname "$0")"
PARALLEL=${PARALLEL:-6}
ARMS=${ARMS:-"A4_eco_sr A6_sgdm_sr A5_sgd_sr"}
echo "$(date '+%F %T') srseed launch" >> progress.txt
for k in 1 2; do for a in $ARMS; do for s in S01 S02 S03 S04; do echo "$a $s $k"; done; done; done | \
xargs -P "$PARALLEL" -L 1 bash -c '
  a=$0; s=$1; k=$2
  if [ -f results_srseed/${a}_sr$k/$s.json ]; then exit 0; fi
  echo "$(date "+%F %T") start $a sr$k $s" >> progress.txt
  python3 run_arm.py --arm $a --subject $s --sr-seed $k 2>&1 | grep -v NNPACK > logs/${a}_sr${k}_${s}.log
  echo "$(date "+%F %T") end   $a sr$k $s : $(grep DONE logs/${a}_sr${k}_${s}.log || echo FAILED)" >> progress.txt
'
echo "$(date '+%F %T') SRSEED DONE" >> progress.txt
