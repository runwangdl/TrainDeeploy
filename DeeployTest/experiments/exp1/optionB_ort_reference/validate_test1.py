# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""Test 1 validation — B-decompose frozen-BN ORT reference vs Option-A PyTorch reference.

Run inside the `agitated_hugle` container (Onnx4Deeploy + ORT/torch). First export the two smoke
fixtures with IDENTICAL config (only the BN-handling flag differs):

  CKPT=/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
  common="-model SpeechNet -mode train --dataset silentwear \
    --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt --pretrained-weights $CKPT \
    --subject S01 --session 3 --batch 1 --condition vocalized \
    --data-size 18 --n-epochs 2 --n-accum 4 --lr 0.0003 --training-strategy full --stratified"
  python3 Onnx4Deeploy.py $common -o /tmp/dec  --bn-decompose-frozen
  python3 Onnx4Deeploy.py $common -o /tmp/optA --bn-frozen-stats

  python3 validate_test1.py --decompose /tmp/dec --optionA /tmp/optA

Checks: (1) no BatchNormInternal + only device-bound ops in the decomposed graph; (2) ORT-decompose vs
Option-A-PyTorch per-step loss match; (3) γ/β/conv/fc train while the frozen stat buffers stay fixed.
"""
import argparse, collections
import numpy as np, onnx
from onnx import numpy_helper

BOUND = {"Add", "Mul", "ReduceSum", "Reshape", "Conv", "ConvGrad", "Relu", "ReluGrad", "MaxPool",
         "MaxPoolGrad", "Gemm", "GlobalAveragePool", "SoftmaxCrossEntropyLoss",
         "SoftmaxCrossEntropyLossGrad", "InPlaceAccumulatorV2", "Scale", "Shape", "Identity"}
FLAG = ("Sub", "Sqrt", "Reciprocal", "Neg", "BatchNormInternal", "BatchNormalizationGrad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decompose", required=True)
    ap.add_argument("--optionA", required=True)
    a = ap.parse_args()

    m = onnx.load(f"{a.decompose}/network_train.onnx")
    c = collections.Counter(n.op_type for n in m.graph.node)
    flagged = {k: c[k] for k in FLAG if k in c}
    unbound = {k: v for k, v in c.items() if k not in BOUND}
    print(f"[graph] decomposed op-types: {dict(c)}")
    print(f"[graph] BN/unwanted ops (expect none): {flagged or 'NONE'}")
    print(f"[graph] not-in-known-bound-set (Expand also appears in Option A): {unbound or 'NONE'}")

    lB = np.load(f"{a.decompose}/outputs.npz")["loss"]
    lA = np.load(f"{a.optionA}/outputs.npz")["loss"]
    md = float(np.max(np.abs(lA - lB)))
    print(f"[loss ] steps A={len(lA)} B={len(lB)}  max|Δ|={md:.3e}  mean|Δ|={np.mean(np.abs(lA-lB)):.3e}  "
          f"→ {'PASS' if md < 1e-3 else 'FAIL'}")

    init = {t.name.replace('.', '_'): numpy_helper.to_array(t)
            for t in onnx.load(f"{a.decompose}/network_infer.onnx").graph.initializer}
    o = np.load(f"{a.decompose}/outputs.npz")
    for tag, names in (("trained>0", ["blocks_0_1_weight", "blocks_0_1_bias", "fc_weight"]),
                       ("frozen=0", ["blocks_0_1_neg_running_mean", "blocks_0_1_inv_running_std"])):
        for n in names:
            if n in init and n in o.files:
                print(f"[weight] {tag:10s} {n:28s} max|final-init|={np.max(np.abs(o[n]-init[n])):.3e}")


if __name__ == "__main__":
    main()
