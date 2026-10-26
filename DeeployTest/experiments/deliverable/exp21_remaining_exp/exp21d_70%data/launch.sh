#!/usr/bin/env bash
# exp21d launcher. Two job sets, because the quantized recipes need Brevitas, which only works with
# the torch inside `agitated_hugle` (native torch 2.13 + brevitas 0.13 fails: Tensor.rename, see
# logs/progress_failed_native_quant_attempt.txt):
#   native   : BP + ZO       -> ./launch.sh native        (host python3.11, arm64)
#   container: QZO + QZO_ECO -> docker exec -d agitated_hugle bash -lc \
#      'cd "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21d_70%data" && ./launch.sh container'
# PARALLEL single-threaded jobs (default 1 native + 3 container, i.e. <= half the 10-core Mac;
# exp21b runs concurrently). Finished jobs (json exists) are skipped -> re-running resumes.
# Core arms first, the optional "more steps" arms last.
set -u
cd "$(dirname "$0")"
MODE=${1:?native|container}
export TORCH_CPP_LOG_LEVEL=ERROR OMP_NUM_THREADS=1
if [ "$MODE" = native ]; then
  export ONNX4DEEPLOY_ROOT=/Users/qiwenwu/ETH/Onnx4Deeploy TRAINDEEPLOY_ROOT=/Users/qiwenwu/ETH/TrainDeeploy
  PY=python3.11; PARALLEL=${PARALLEL:-1}
  jobs () {
    for a in p30_e120 p70_eq1620; do for s in S01 S02 S03 S04; do echo "zo $a $s"; done; done
    for a in p30_e40 p70_eq540 p30_e120 p70_eq1620; do for s in S01 S02 S03 S04; do echo "bp $a $s"; done; done
    for s in S01 S02 S03 S04; do echo "zo p70_e120 $s"; done
    for s in S01 S02 S03 S04; do echo "bp p70_e40 $s"; done
  }
else
  export CLEANUP_ROOT=/app/SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp
  PY=python3; PARALLEL=${PARALLEL:-3}
  jobs () {
    for r in eco qzo; do for a in p30_e120 p70_eq1620; do for s in S01 S02 S03 S04; do echo "$r $a $s"; done; done; done
    for r in eco qzo; do for s in S01 S02 S03 S04; do echo "$r p70_e120 $s"; done; done
  }
fi
mkdir -p logs results
echo "$(date '+%F %T') launch $MODE PARALLEL=$PARALLEL" >> progress.txt
export PY
jobs | xargs -P "$PARALLEL" -L 1 bash -c '
  r=$0; a=$1; s=$2
  if [ -f results/$r/$a/$s.json ]; then echo "$(date "+%F %T") skip  $r $a $s" >> progress.txt; exit 0; fi
  echo "$(date "+%F %T") start $r $a $s" >> progress.txt
  $PY -u run_job.py --recipe $r --arm $a --subject $s 2>&1 | grep -av NNPACK > logs/${r}_${a}_${s}.log
  echo "$(date "+%F %T") end   $r $a $s : $(grep -a DONE logs/${r}_${a}_${s}.log || echo FAILED)" >> progress.txt
'
echo "$(date '+%F %T') ALL DONE $MODE" >> progress.txt
