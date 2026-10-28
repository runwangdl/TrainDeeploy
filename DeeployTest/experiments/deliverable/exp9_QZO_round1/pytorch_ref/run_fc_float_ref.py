# SPDX-License-Identifier: MIT
"""exp9 PyTorch reference — device-faithful fc-FLOAT QZO, round 1 (ft b1 -> eval b2), fold 3.

The on-device QZO graph keeps the classifier head in fp32 (`build_int8_forward`,
qzo_weight_integerize.py:337-355): fc weight/bias are dequantized to fp32 (w_fp32=w_int8*s_w),
the fc input activation quant is BYPASSED (Gemm consumes fp32 GAP features), and fc is
float-perturbed like BN gamma/beta. So the device is "int8 conv datapath + fp32 head"
(QMCUNetZO). Our earlier run_incremental sim quantized fc as int8 (fc.weight was frozen anyway).

This script measures BOTH so we know what accuracy to expect on device:
  * fc-int8  : the exact run_incremental setting (should reproduce b2 = 90.00%)
  * fc-float : device-faithful — fc converted to a float Linear (dequant-quant init, fc_iq
               bypassed), fc weight+bias trained by float ZO like BN.
Everything else identical: conv int8 direct update lr 1e-5, act scales pooled@99.99, weight
scales abs-max, eps 0.01, 2700 steps, seed 42, 54 windows (b1) -> eval whole b2 (180).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EXP_CALIB = Path("/app/Onnx4Deeploy/QZO_exp/exp_calibration")
sys.path.insert(0, str(EXP_CALIB))
sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp3_lr1e-5_stability")

import run_study as S            # noqa: E402
import stability_lib as L        # noqa: E402

HERE = Path(__file__).resolve().parent
S._LOG = open(HERE / "run.log", "a")
LR = 1e-5


def to_float_fc(model, params):
    """Convert the int8 QuantLinear head to a device-faithful fp32 Linear: dequant-quant init,
    fc_iq bypassed, fc weight+bias become float params. Returns the new params list."""
    fcw = next(p for p in params if p["name"] == "fc.weight")
    fcb = next(p for p in params if p["name"] == "fc.bias")
    s_w = fcw["scale"].reshape(-1, 1)                 # (9,1)
    s_b = fcb["scale"].reshape(-1)                    # (9,)
    w_qd = (torch.round(fcw["init"] / s_w).clamp(-127, 127) * s_w).float()
    b_qd = (torch.round(fcb["init"].reshape(-1) / s_b) * s_b).float()
    model.fc_iq = nn.Identity()
    newfc = nn.Linear(model._fc_in, model.num_classes, bias=True)
    newfc.weight.data = w_qd.reshape(newfc.weight.shape)
    newfc.bias.data = b_qd.reshape(newfc.bias.shape)
    model.fc = newfc
    params = [p for p in params if p["name"] not in ("fc.weight", "fc.bias")]
    params.append(dict(name="fc.weight", mod=newfc, attr="weight", kind="float",
                       scale=None, init=newfc.weight.detach().clone()))
    params.append(dict(name="fc.bias", mod=newfc, attr="bias", kind="float",
                       scale=None, init=newfc.bias.detach().clone()))
    return params


def run(fc_float: bool):
    r = S.load_results()
    data = L.load_fold_data(3)
    trX = {1: data["trX1"]}; trY = {1: data["trY1"]}
    evX, evY = data["evX1"], data["evY1"]
    _, model = L.build_qmodel(3)
    # Use the CANONICAL stored pooled@99.99 thresholds (exp_calibration/results.json pools) —
    # identical to run_incremental (the 90.00% result) and to what we bake into the device
    # fixture. Re-deriving would drift ~1 window via subsample RNG.
    S.freeze_config(model, r, "pooled@99.99", None)
    params = S.build_params(model)
    tag = "fc-int8"
    if fc_float:
        params = to_float_fc(model, params)
        tag = "fc-float"
    # single-round streaming (reuse stability_lib.qzo_stream with a 1-round data dict)
    d1 = {"trX1": trX[1], "trY1": trY[1], "evX1": evX, "evY1": evY}
    # qzo_stream expects rounds 1..4; make a 1-round variant inline
    conv_w = [p for p in params if p["kind"] == "quant" and p["name"].endswith(".conv.weight")]
    n_convw = sum(int(p["init"].numel()) for p in conv_w)
    state = {p["name"]: (S.q_int(p["init"], p) if p["kind"] == "quant" else p["init"].clone())
             for p in params}
    init_int = {p["name"]: state[p["name"]].clone() for p in conv_w}
    Xt, Yt = torch.from_numpy(trX[1]), torch.from_numpy(trY[1])
    with torch.no_grad():
        S.install(params, state, "direct")
        ab, _, _ = S.balanced(S.logits_of(model, evX), evY)
    import torch.nn.functional as F
    STEPS = 200 * S.N_TRAIN // S.N_ACCUM
    for u in range(STEPS):
        z = S.draw_z(params, u)
        idx = [(u * S.N_ACCUM + a) % S.N_TRAIN for a in range(S.N_ACCUM)]
        xb, yb = Xt[idx], Yt[idx]
        di_p, di_m, df_p, df_m = {}, {}, {}, {}
        for p in params:
            n = p["name"]
            if p["kind"] == "quant":
                dz = p["dz_int"] * z[n]; di_p[n], di_m[n] = dz, -dz
            else:
                dz = S.EPS * z[n]; df_p[n], df_m[n] = dz, -dz
        with torch.no_grad():
            S.install(params, state, "direct", di_p, df_p)
            Lp = float(F.cross_entropy(model(xb), yb, reduction="sum"))
            S.install(params, state, "direct", di_m, df_m)
            Lm = float(F.cross_entropy(model(xb), yb, reduction="sum"))
        coeff = -LR * (Lp - Lm) / (2.0 * S.EPS * S.N_ACCUM)
        for p in params:
            n = p["name"]
            if p["kind"] == "float":
                state[n] = state[n] + coeff * z[n]
            else:
                state[n] = torch.clamp(state[n] + torch.round(coeff * z[n] / p["scale"]),
                                       p["lo"], p["hi"])
    with torch.no_grad():
        S.install(params, state, "direct")
        aa, _, _ = S.balanced(S.logits_of(model, evX), evY)
    moved = sum(int((state[p["name"]] != init_int[p["name"]]).sum()) for p in conv_w)
    S.log(f"[{tag}] round1 b2: {ab:.2f} -> {aa:.2f}%  conv-moved={100.0*moved/n_convw:.1f}%")
    return dict(tag=tag, acc_before=ab, acc_after=aa, conv_moved_pct=100.0 * moved / n_convw)


def main():
    S.SEED = 42
    S.log("==== exp9 PyTorch reference: fc-int8 vs fc-float (device-faithful), round1 b2 ====")
    res = {}
    res["fc_int8"] = run(fc_float=False)
    res["fc_float"] = run(fc_float=True)
    import json
    json.dump(res, open(HERE / "results.json", "w"), indent=2)
    S.log(f"  fc-int8  b2 = {res['fc_int8']['acc_after']:.2f}%  (expect ~90.00)")
    S.log(f"  fc-float b2 = {res['fc_float']['acc_after']:.2f}%  <- device-faithful reference")
    S.log("==== exp9 reference complete ====")


if __name__ == "__main__":
    main()
