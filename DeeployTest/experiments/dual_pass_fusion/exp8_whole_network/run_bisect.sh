#!/usr/bin/env bash
# exp8 debug: run a bisect fixture (bisect_dual.py) through the tiled Siracusa inference runner, frozen-BN channels-last
# as in the MeZO runner.  bash run_bisect.sh <tag> [<tag> ...]
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd)
R=/app/ETH/TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp8_whole_network
for tag in "$@"; do
  TM=""; OM=""
  for n in $(python3 -c "import numpy as np; print(' '.join(k for k in np.load('$H/bisect/$tag/inputs.npz').files if k.endswith('bias_rqsadd')))"); do
    TM="$TM $n=int32_t"; OM="$OM $n=0"; done
  docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && (pgrep -f '[g]vsoc_launcher' | xargs -r kill -9 || true); \
    rm -rf TEST_SIRACUSA/build_e8b_$tag; DEEPLOY_BN_FROZEN_NHWC=1 PYTEST_XDIST_WORKER=e8b_$tag \
    python3 deeployRunner_tiled_siracusa.py -t $R/bisect/$tag --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8 \
    -D BN_FROZEN_STATS=ON ${TM:+--input-type-map $TM --input-offset-map $OM} > $R/logs/bisect_$tag.log 2>&1"
  echo "$tag: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' "$H/logs/bisect_$tag.log" | head -1) $(grep -aiE 'traceback|error:' "$H/logs/bisect_$tag.log" | grep -v -E 'Werror|-Wno' | head -1)"
done
