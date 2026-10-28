# SPDX-License-Identifier: MIT
"""exp12 step 1 (clean, no reused artifacts): (a) regenerate pooled@99.99 activation thresholds
fresh on the fold-3 pretraining data, save the JSON to bake into the export; (b) run the
device-faithful fc-FLOAT PyTorch reference for round-1 (ft session-3 batch1 -> eval whole batch2)
at lr 1e-5, giving the reference accuracy.

Reuses only PIPELINE CODE (calib_pooled, stability_lib, run_fc_float_ref.to_float_fc), not any
stored results/thresholds.
"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp_calibration")
sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp3_lr1e-5_stability")
sys.path.insert(0, "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1/pytorch_ref")
import run_study as S, stability_lib as L, calib_pooled as CP
import run_fc_float_ref as RF

HERE = Path(__file__).resolve().parent
S._LOG = open(HERE / "logs/ref_calib.log", "a")
S.SEED = 42
STEPS = 200 * S.N_TRAIN // S.N_ACCUM   # 2700

# (a) FRESH pooled@99.99 calibration on fold-3 pretraining (sessions 1+2, 1800 windows)
_, qm = L.build_qmodel(3)
pre = []
for sess in L.pretrain_sessions(3):
    for b in range(1, 6):
        e = S.make_exporter(sess, b); xs, _ = e.get_data_source()._load_windows()
        pre.append(np.concatenate(xs, 0).astype(np.float32))
preX = np.concatenate(pre, 0); assert preX.shape[0] == 1800, preX.shape
coll, _ = CP.collect_pooled(qm, preX, batch_size=64, cap=40_000_000, seed=0, log=S.log)
th = CP.thresholds_from_pool(coll, [99.99])[99.99]
thresholds = {site: float(d["threshold"]) for site, d in th.items()}
json.dump(thresholds, open(HERE / "fixture/pooled_9999_fold3_fresh.json", "w"), indent=1)
CP.freeze_act_thresholds(qm, thresholds, log=S.log)
S.log(f"[calib] fresh pooled@99.99 saved ({len(thresholds)} sites); "
      f"block0-in threshold={thresholds['blocks.0.conv.input_quant']:.4f} "
      f"-> scale={thresholds['blocks.0.conv.input_quant']/128:.6f}")

# (b) fc-FLOAT device-faithful round-1 reference accuracy
params = S.build_params(qm)
params = RF.to_float_fc(qm, params)
d = L.load_fold_data(3)
trX, trY = torch.from_numpy(d["trX1"]), torch.from_numpy(d["trY1"])
evX, evY = d["evX1"], d["evY1"]
state = {p["name"]: (S.q_int(p["init"], p) if p["kind"] == "quant" else p["init"].clone()) for p in params}
with torch.no_grad():
    S.install(params, state, "direct"); zs, _, _ = S.balanced(S.logits_of(qm, evX), evY)
for u in range(STEPS):
    z = S.draw_z(params, u); idx = [(u*S.N_ACCUM+a) % S.N_TRAIN for a in range(S.N_ACCUM)]
    xb, yb = trX[idx], trY[idx]
    dip, dim, dfp, dfm = {}, {}, {}, {}
    for p in params:
        n = p["name"]
        if p["kind"] == "quant": dz = p["dz_int"]*z[n]; dip[n], dim[n] = dz, -dz
        else: dz = S.EPS*z[n]; dfp[n], dfm[n] = dz, -dz
    with torch.no_grad():
        S.install(params, state, "direct", dip, dfp); Lp = float(F.cross_entropy(qm(xb), yb, reduction="sum"))
        S.install(params, state, "direct", dim, dfm); Lm = float(F.cross_entropy(qm(xb), yb, reduction="sum"))
    coeff = -1e-5*(Lp-Lm)/(2.0*S.EPS*S.N_ACCUM)
    for p in params:
        n = p["name"]
        if p["kind"] == "float": state[n] = state[n] + coeff*z[n]
        else: state[n] = torch.clamp(state[n] + torch.round(coeff*z[n]/p["scale"]), p["lo"], p["hi"])
with torch.no_grad():
    S.install(params, state, "direct"); aa, _, _ = S.balanced(S.logits_of(qm, evX), evY)
res = dict(zero_shot=zs, after=aa, lr=1e-5, steps=STEPS, fc="float(device-faithful)")
json.dump(res, open(HERE / "results/pytorch_ref.json", "w"), indent=1)
S.log(f"[ref] fc-float round-1: zero-shot b2 {zs:.2f}% -> after {aa:.2f}%  (THE reference accuracy)")
print(f"REFERENCE fc-float b2: {zs:.2f} -> {aa:.2f}%")
