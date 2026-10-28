# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Pooled activation calibration for the QZO export (container agitated_hugle).

For every activation-quantization site of the Brevitas SpeechNet, collect |activation| over all 1800
fold-3 pretraining windows (sessions 1 and 2, batches 1-5), take one percentile per site and write
{site: threshold} as JSON. The export reads it through QZO_POOLED_THRESHOLDS; scale = threshold / 128.
This is step (a) of DeeployTest/experiments/deliverable/exp19_QZO_round1_correct_weights/run_ref_and_calib.py
with the percentile as a parameter (exp19 used 99.99).

    python3 /app/TrainDeeploy/SpeechNet_Test/scripts/calibrate_pooled.py --percentile 99.9 \
        --out /app/TrainDeeploy/SpeechNet_Test/calibration/pooled_999_fold3_ref.json

Uses the calibration code in Onnx4Deeploy/QZO_exp (calib_pooled.py, stability_lib.py, run_study.py) and
the artifacts_reference checkpoint (QZO_ARTIFACTS_DIR, default artifacts_reference).
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp_calibration")
sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp3_lr1e-5_stability")
import calib_pooled as CP  # noqa: E402
import run_study as S  # noqa: E402
import stability_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--percentile", type = float, nargs = "+", required = True)
ap.add_argument("--out", nargs = "+", required = True, help = "one JSON path per percentile")
a = ap.parse_args()
assert len(a.out) == len(a.percentile)

S.SEED = 42
_, qm = L.build_qmodel(3)
pre = []
for sess in L.pretrain_sessions(3):
    for b in range(1, 6):
        xs, _ = S.make_exporter(sess, b).get_data_source()._load_windows()
        pre.append(np.concatenate(xs, 0).astype(np.float32))
preX = np.concatenate(pre, 0)
assert preX.shape[0] == 1800, preX.shape
coll, _ = CP.collect_pooled(qm, preX, batch_size = 64, cap = 40_000_000, seed = 0, log = print)
th = CP.thresholds_from_pool(coll, list(a.percentile))
for p, out in zip(a.percentile, a.out):
    d = {site: float(v["threshold"]) for site, v in th[p].items()}
    json.dump(d, open(out, "w"), indent = 1)
    print(f"pooled@{p}: {len(d)} sites -> {out}; block-0 input threshold {d['blocks.0.conv.input_quant']:.4f}")
