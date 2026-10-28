#!/usr/bin/env bash
# exp21f launcher: native host python3.11 (arm64), <=2 parallel single-threaded jobs (concurrency rule).
# Finished jobs (json exists) are skipped -> re-running resumes.
#   ./launch.sh            seed 42, full grid, S01-S04   (main sweep)
#   ./launch.sh noise      seeds 43, 44, full grid, S01-S04 (draw-noise floor)
set -u
cd "$(dirname "$0")"
export TORCH_CPP_LOG_LEVEL=ERROR OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ONNX4DEEPLOY_ROOT=/Users/qiwenwu/ETH/Onnx4Deeploy TRAINDEEPLOY_ROOT=/Users/qiwenwu/ETH/TrainDeeploy
GRID="0 5 10 15 20 30 40 50 60 80 100 120 160 200 300"
SEEDS=42; [ "${1:-}" = noise ] && SEEDS="43 44"
PARALLEL=${PARALLEL:-2}
mkdir -p logs results
echo "$(date '+%F %T') launch seeds=$SEEDS PARALLEL=$PARALLEL" >> progress.txt
for sd in $SEEDS; do for e in $GRID; do for s in S01 S02 S03 S04; do echo "$sd $e $s"; done; done; done |
xargs -P "$PARALLEL" -L 1 bash -c '
  sd=$0; e=$1; s=$2; f=$(printf "results/seed%s/e%03d/%s.json" $sd $e $s)
  if [ -f $f ]; then exit 0; fi
  L=logs/seed${sd}_e${e}_${s}.log
  python3.11 -u run_job.py --subject $s --epochs $e --seed $sd 2>&1 | grep -av NNPACK > $L
  echo "$(date "+%F %T") $(grep -a DONE $L || echo FAILED seed$sd e$e $s)" >> progress.txt
'
echo "$(date '+%F %T') ALL DONE seeds=$SEEDS" >> progress.txt
