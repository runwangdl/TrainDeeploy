# SPDX-License-Identifier: MIT
"""
QUANTIZED zeroth-order (MeZO) fine-tuning of SpeechNet WITH MASTER WEIGHTS — PyTorch/Brevitas simulation.

Question: what batch-2 balanced accuracy does quantized ZO fine-tuning reach when the sub-LSB
"weight stall" is fixed with a latent fp32 master copy, vs the direct-int8 (memoryless) update the
device does today?

Model  : the real Brevitas QuantSpeechNetDeploy built exactly like `-mode q-zo-train`
         (SpeechNetExporter.create_brevitas_model + real-data PTQ calibration), i.e.
         int8 per-channel conv/fc weights (Int8WeightPerChannelFloat), int32 bias (Int32Bias),
         int8 per-tensor activations (Int8ActPerTensorFloat), fp32 UNFOLDED BatchNorm.
Scales : FROZEN after calibration.  The Brevitas weight `scaling_impl` is replaced by a constant
         module holding the post-calibration per-output-channel s_w, so abs-max re-tracking of the
         latent weight can never move the grid.  Activation scales are frozen by `model.eval()`
         (verified: 0.0 drift over the whole run).
Train  : 54 stratified windows (6/class, seed 42) from S01 / session 3 / vocalized / BATCH 1.
Eval   : the WHOLE of batch 2 (S01 / session 3 / vocalized) = 180 windows, 20/class.
Recipe : 200 epochs x 54 windows / n_accum 4 = 2700 update steps, q=1, eps=0.01, seed 42.

Two regimes (identical scales, identical z per step, identical window order):
  MASTER : latent fp32 kept for the quantized params; W_latent += coeff*z UNROUNDED; every forward
           re-quantizes with the FROZEN scale.
  DIRECT : the int grid is the state; delta = round(coeff*z/s); int += delta (clamped). Sub-LSB
           discarded each step -> memoryless (today's device).
BN gamma/beta are plain fp32 in both regimes.

Writes results.json / accuracy_summary.png / run.log next to this file.
Creates no new files elsewhere and modifies no existing source file.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

CKPT = ("/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/"
        "speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt")
DATA_PATH = "/app/SilentWear/SilentWear_data/data_raw_and_filt"

EPS = 0.01
N_ACCUM = 4
SEED = 42
EPOCHS = 200
N_TRAIN = 54
N_STEPS = EPOCHS * N_TRAIN // N_ACCUM          # 2700 — exactly 200 passes over the 54 windows
LOSS_EVERY = 50
ACC_EVERY = 270
CALIB_SAMPLES = 8

# float-ZO reference points (exp5 device / exp18 PyTorch sim / float zero-shot)
REF = {"float_ZO_device": 88.33, "float_ZO_sim": 87.36, "float_zero_shot": 80.56}

# EXTRA="direct@1e-5,direct@3e-5" re-opens results.json, runs only those extra regimes,
# merges them in and re-plots (run.log is appended to, never truncated, in that mode).
EXTRA = os.environ.get("EXTRA", "").strip()
_LOG = open(HERE / "run.log", "a" if EXTRA else "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG.write(s + "\n")
    _LOG.flush()


# --------------------------------------------------------------------------------------
# Frozen weight-scale module
# --------------------------------------------------------------------------------------
class ConstScale(torch.nn.Module):
    """Replaces Brevitas' StatsFromParameterScaling so s_w can never re-track the latent weight."""

    def __init__(self, v: torch.Tensor):
        super().__init__()
        self.register_buffer("v", v.detach().clone())

    def forward(self, *a, **k):
        return self.v


# --------------------------------------------------------------------------------------
# Build + calibrate + freeze
# --------------------------------------------------------------------------------------
def make_exporter(batch: int):
    from onnx4deeploy.models.speechnet_exporter import SpeechNetExporter
    e = SpeechNetExporter()
    e._config_overrides = dict(
        dataset="silentwear", data_path=DATA_PATH, pretrained_weights=CKPT,
        pretrained_key="model_state_dict", subject="S01", session=3, batch=batch,
        condition="vocalized", batch_size=1, num_classes=9,
        calib_samples=CALIB_SAMPLES, stratified_sampling=True)
    e.config = e.load_config()
    return e


def build():
    """Returns (model, params, train_X, train_y, eval_X, eval_y, act_scales_after_calib)."""
    from brevitas.graph.calibrate import calibration_mode
    import brevitas.nn as qnn

    e1 = make_exporter(1)
    ishape = e1.get_input_shape()
    X, Y = e1.get_data_source().load_batches(N_TRAIN, (1,) + tuple(ishape[1:]),
                                             e1.config["num_classes"], seed=SEED)
    trX = np.concatenate([np.asarray(a, np.float32) for a in X], 0)
    trY = np.asarray([int(np.asarray(y).reshape(-1)[0]) for y in Y], np.int64)
    log(f"train windows {trX.shape}  label dist "
        f"{ {int(k): int(v) for k, v in zip(*np.unique(trY, return_counts=True))} }")

    # WHOLE batch 2 — SilentWearDataSource._load_windows() returns every window of the batch,
    # not a stratified sub-draw.  Same segmentation/windowing/rest-downsampling as the exporter's
    # own `_save_silentwear_eval_data`.
    e2 = make_exporter(2)
    i2, l2 = e2.get_data_source()._load_windows()
    evX = np.concatenate(i2, 0)
    evY = np.concatenate(l2, 0)
    log(f"eval (whole batch 2) {evX.shape}  label dist "
        f"{ {int(k): int(v) for k, v in zip(*np.unique(evY, return_counts=True))} }")

    model = e1.create_brevitas_model()
    model.eval()
    with torch.no_grad(), calibration_mode(model):
        model(torch.from_numpy(trX[:CALIB_SAMPLES]))
    model.eval()   # freezes BN running stats + activation observers

    # ---- collect quantized params and FREEZE the weight scales ------------------------
    params = []          # ordered list of dicts, defines the z draw order
    qlayers = [(n, m) for n, m in model.named_modules()
               if isinstance(m, (qnn.QuantConv2d, qnn.QuantLinear))]
    for name, m in qlayers:
        s_w = m.quant_weight().scale.detach().clone()            # [Cout,1,1,1] / [Cout,1]
        s_in = m.input_quant.scale().detach().clone()
        m.weight_quant.tensor_quant.scaling_impl = ConstScale(s_w)   # <-- freeze
        params.append(dict(name=f"{name}.weight", mod=m, attr="weight", kind="quant",
                           scale=s_w, lo=-127, hi=127,
                           init=m.weight.detach().clone()))
        s_b = (s_in * s_w).reshape(-1)                            # int32 bias grid = s_in * s_w
        params.append(dict(name=f"{name}.bias", mod=m, attr="bias", kind="quant",
                           scale=s_b, lo=-(2 ** 31) + 1, hi=2 ** 31 - 1,
                           init=m.bias.detach().clone()))
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            params.append(dict(name=f"{name}.weight", mod=m, attr="weight", kind="float",
                               scale=None, init=m.weight.detach().clone()))
            params.append(dict(name=f"{name}.bias", mod=m, attr="bias", kind="float",
                               scale=None, init=m.bias.detach().clone()))
    log(f"trainable params: {len(params)}  "
        f"({sum(p['kind']=='quant' for p in params)} quantized / "
        f"{sum(p['kind']=='float' for p in params)} fp32)")
    for p in params:
        if p["kind"] == "quant":
            dz = torch.round(EPS / p["scale"])
            p["dz_int"] = dz
            log(f"   {p['name']:22s} kind=quant  n={p['init'].numel():6d}  "
                f"round(eps/s) in [{int(dz.min())},{int(dz.max())}]")
        else:
            log(f"   {p['name']:22s} kind=float  n={p['init'].numel():6d}")

    return model, params, trX, trY, evX, evY, act_scales(model)


def act_scales(model):
    import brevitas.nn as qnn
    out = {}
    for n, m in model.named_modules():
        if isinstance(m, (qnn.QuantConv2d, qnn.QuantLinear, qnn.QuantIdentity)):
            for attr in ("input_quant", "output_quant", "act_quant"):
                q = getattr(m, attr, None)
                if q is not None and getattr(q, "is_quant_enabled", False):
                    try:
                        out[f"{n}.{attr}"] = float(q.scale())
                    except Exception:
                        pass
    return out


# --------------------------------------------------------------------------------------
# State <-> model
# --------------------------------------------------------------------------------------
def q_int(v, p):
    return torch.clamp(torch.round(v / p["scale"]), p["lo"], p["hi"])


def install(params, state, mode, delta_int=None, delta_float=None):
    """Install a state into the live model.

    mode == "master": state[name] is the latent fp32 tensor -> int = clamp(round(latent/s)).
    mode == "direct": state[name] is the int grid tensor.
    delta_int   : optional dict name -> integer offset added to the int grid (the device's
                  RQSPerturbRademacher: int += round(eps/s)*z, then clamped).
    delta_float : optional dict name -> fp32 offset for the fp32 (BN) params.
    """
    for p in params:
        n = p["name"]
        if p["kind"] == "quant":
            iv = q_int(state[n], p) if mode == "master" else state[n]
            if delta_int is not None and n in delta_int:
                iv = torch.clamp(iv + delta_int[n], p["lo"], p["hi"])
            getattr(p["mod"], p["attr"]).data = (iv * p["scale"]).reshape(p["init"].shape).float()
        else:
            v = state[n]
            if delta_float is not None and n in delta_float:
                v = v + delta_float[n]
            getattr(p["mod"], p["attr"]).data = v.clone()


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, X, Y, chunk=60):
    outs = []
    for i in range(0, X.shape[0], chunk):
        outs.append(model(torch.from_numpy(X[i:i + chunk])).numpy())
    logits = np.concatenate(outs, 0)
    pred = logits.argmax(-1)
    classes = np.unique(Y)
    rec = {int(c): float((pred[Y == c] == c).mean()) for c in classes}
    return dict(balanced_accuracy=float(np.mean(list(rec.values())) * 100.0),
                overall_accuracy=float((pred == Y).mean() * 100.0),
                per_class_recall=rec)


@torch.no_grad()
def train_loss(model, X, Y, chunk=54):
    tot, n = 0.0, 0
    for i in range(0, X.shape[0], chunk):
        o = model(torch.from_numpy(X[i:i + chunk]))
        t = torch.from_numpy(Y[i:i + chunk])
        tot += float(F.cross_entropy(o, t, reduction="sum"))
        n += t.numel()
    return tot / n


# --------------------------------------------------------------------------------------
# MeZO training
# --------------------------------------------------------------------------------------
def draw_z(params, u, cache={}):
    """Rademacher +-1 per trainable tensor, seed = SEED + u. Cached so both regimes share it."""
    key = u
    if key in cache:
        return cache[key]
    rng = np.random.RandomState(SEED + u)
    z = {p["name"]: torch.from_numpy(
        (rng.randint(0, 2, size=tuple(p["init"].shape)).astype(np.float32) * 2.0 - 1.0))
        for p in params}
    cache.clear()          # only ever need the current step
    cache[key] = z
    return z


def run_regime(model, params, trX, trY, evX, evY, mode, lr, n_steps=N_STEPS, tag=""):
    t0 = time.time()
    # Dedicated RNG for stochastic rounding, so it cannot disturb the shared z draw
    # (z must stay identical across regimes for the comparison to be apples-to-apples).
    g_sr = torch.Generator().manual_seed(SEED + 12345)
    conv_w = [p for p in params if p["kind"] == "quant" and p["name"].startswith("blocks")
              and p["name"].endswith(".conv.weight")]
    n_convw = sum(int(p["init"].numel()) for p in conv_w)

    if mode == "master":
        state = {p["name"]: p["init"].clone() for p in params}
    else:
        state = {p["name"]: (q_int(p["init"], p) if p["kind"] == "quant" else p["init"].clone())
                 for p in params}
    init_int = {p["name"]: q_int(p["init"], p) for p in params if p["kind"] == "quant"}

    def cur_int(n, p):
        return q_int(state[n], p) if mode == "master" else state[n]

    prev_int = {p["name"]: cur_int(p["name"], p).clone() for p in conv_w}

    hist = dict(step=[], train_loss=[], g_proj=[], coeff=[],
                pct_moved_step=[], pct_moved_cum=[], acc_step=[], acc_bal=[])
    zero_move_steps = 0
    Xt = torch.from_numpy(trX)
    Yt = torch.from_numpy(trY)

    for u in range(n_steps):
        z = draw_z(params, u)
        idx = [(u * N_ACCUM + a) % N_TRAIN for a in range(N_ACCUM)]
        xb = Xt[idx]
        yb = Yt[idx]

        di_p, di_m, df_p, df_m = {}, {}, {}, {}
        for p in params:
            n = p["name"]
            if p["kind"] == "quant":
                d = p["dz_int"] * z[n]                       # device: round(eps/s) LSBs, +-
                di_p[n], di_m[n] = d, -d
            else:
                d = EPS * z[n]
                df_p[n], df_m[n] = d, -d

        with torch.no_grad():
            install(params, state, mode, di_p, df_p)
            Lp = float(F.cross_entropy(model(xb), yb, reduction="sum"))
            install(params, state, mode, di_m, df_m)
            Lm = float(F.cross_entropy(model(xb), yb, reduction="sum"))

        g_proj = (Lp - Lm) / (2.0 * EPS * N_ACCUM)
        coeff = -lr * g_proj

        # ---- update -------------------------------------------------------------------
        for p in params:
            n = p["name"]
            if p["kind"] == "float":
                state[n] = state[n] + coeff * z[n]
            elif mode == "master":
                state[n] = state[n] + coeff * z[n]           # UNROUNDED latent fp32
            elif mode == "stoch":
                # STOCHASTIC ROUNDING: int grid is the state (no fp32 anywhere), but the
                # sub-LSB step is applied in EXPECTATION instead of being discarded:
                #   x = coeff*z/s ;  delta = floor(x) + Bernoulli(x - floor(x))
                #   => E[delta] = x exactly. Zero extra memory (device kernel already has an RNG).
                x = coeff * z[n] / p["scale"]
                fl = torch.floor(x)
                u01 = torch.rand(x.shape, generator=g_sr, device=x.device, dtype=x.dtype)
                delta = fl + (u01 < (x - fl)).to(x.dtype)
                state[n] = torch.clamp(state[n] + delta, p["lo"], p["hi"])
            else:
                delta = torch.round(coeff * z[n] / p["scale"])
                state[n] = torch.clamp(state[n] + delta, p["lo"], p["hi"])

        # ---- movement stats on the int8 conv weights ----------------------------------
        moved = 0
        moved_cum = 0
        for p in conv_w:
            n = p["name"]
            ci = cur_int(n, p)
            moved += int((ci != prev_int[n]).sum())
            moved_cum += int((ci != init_int[n]).sum())
            prev_int[n] = ci.clone()
        if moved == 0:
            zero_move_steps += 1

        if (u % LOSS_EVERY == 0) or (u == n_steps - 1):
            with torch.no_grad():
                install(params, state, mode)
                tl = train_loss(model, trX, trY)
            hist["step"].append(u)
            hist["train_loss"].append(tl)
            hist["g_proj"].append(g_proj)
            hist["coeff"].append(coeff)
            hist["pct_moved_step"].append(100.0 * moved / n_convw)
            hist["pct_moved_cum"].append(100.0 * moved_cum / n_convw)
        if (u % ACC_EVERY == 0) or (u == n_steps - 1):
            with torch.no_grad():
                install(params, state, mode)
                a = evaluate(model, evX, evY)
            hist["acc_step"].append(u)
            hist["acc_bal"].append(a["balanced_accuracy"])
            log(f"  [{tag}] step {u:5d}/{n_steps}  loss={hist['train_loss'][-1]:.4f}  "
                f"bal_acc={a['balanced_accuracy']:.2f}%  cum_moved={100.0*moved_cum/n_convw:.3f}%  "
                f"coeff={coeff:+.3e}  ({time.time()-t0:.0f}s)")

    # ---- final ------------------------------------------------------------------------
    with torch.no_grad():
        install(params, state, mode)
        final = evaluate(model, evX, evY)
        final_loss = train_loss(model, trX, trY)

    # movement summary over ALL quantized params + the conv weights specifically
    def mv(sel):
        tot = sum(int(p["init"].numel()) for p in sel)
        ch = sum(int((cur_int(p["name"], p) != init_int[p["name"]]).sum()) for p in sel)
        return 100.0 * ch / tot if tot else 0.0

    allq = [p for p in params if p["kind"] == "quant"]
    stats = dict(
        pct_steps_zero_conv_weight_movement=100.0 * zero_move_steps / n_steps,
        cum_pct_convw_int8_changed=mv(conv_w),
        cum_pct_all_quant_int_changed=mv(allq),
        n_conv_weight_elems=n_convw,
    )
    if mode == "master":
        drift = {p["name"]: float(((state[p["name"]] - p["init"]).abs() / p["scale"]).max())
                 for p in allq}
        stats["max_latent_drift_LSB_per_param"] = drift
        stats["max_latent_drift_LSB"] = max(drift.values())
        stats["max_latent_drift_LSB_convw"] = max(drift[p["name"]] for p in conv_w)
        stats["mean_latent_drift_LSB_convw"] = float(np.mean(
            [float(((state[p["name"]] - p["init"]).abs() / p["scale"]).mean()) for p in conv_w]))
    else:
        stats["max_int_move_convw"] = max(
            float((state[p["name"]] - init_int[p["name"]]).abs().max()) for p in conv_w)

    res = dict(mode=mode, lr=lr, n_steps=n_steps, eps=EPS, n_accum=N_ACCUM, seed=SEED,
               final_train_loss=final_loss, weight_movement=stats, history=hist, **final)
    log(f"[{tag}] DONE  bal_acc={final['balanced_accuracy']:.2f}%  "
        f"overall={final['overall_accuracy']:.2f}%  loss={final_loss:.4f}  "
        f"zero-move steps={stats['pct_steps_zero_conv_weight_movement']:.1f}%  "
        f"cum convW moved={stats['cum_pct_convw_int8_changed']:.3f}%  ({time.time()-t0:.0f}s)")
    return res


# --------------------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_steps = int(os.environ.get("N_STEPS", N_STEPS))
    lrs_master = [float(v) for v in os.environ.get("LRS", "3e-6,1e-5,3e-5").split(",")]

    if os.environ.get("PLOT_ONLY"):
        plot(json.load(open(HERE / "results.json")))
        return

    if EXTRA:
        results = json.load(open(HERE / "results.json"))
        model, params, trX, trY, evX, evY, acts0 = build()
        log(f"\n### supplementary runs: {EXTRA}")
        for spec in EXTRA.split(","):
            mode, lr = spec.split("@")
            lr = float(lr)
            tag = f"{mode}@{lr:g}"
            log(f"\n=== {tag} (supplementary) ===")
            results["runs"][tag] = run_regime(model, params, trX, trY, evX, evY,
                                              mode, lr, n_steps, tag)
        out = results["runs"]
        results["gates"]["gate3_direct_stall"] = {
            t: dict(pct_steps_zero_conv_weight_movement=out[t]["weight_movement"]
                    ["pct_steps_zero_conv_weight_movement"],
                    cum_pct_convw_int8_changed=out[t]["weight_movement"]
                    ["cum_pct_convw_int8_changed"])
            for t in out if t.startswith("direct")}
        with open(HERE / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        log(f"\nwrote {HERE/'results.json'} (supplemented)")
        plot(results)
        return

    model, params, trX, trY, evX, evY, acts0 = build()
    results = {"config": dict(eps=EPS, n_accum=N_ACCUM, seed=SEED, epochs=EPOCHS,
                              n_train=N_TRAIN, n_steps=n_steps, ckpt=CKPT,
                              reference_points=REF)}

    # ---- gate 1/2: zero-shot + calibration sanity -------------------------------------
    install(params, {p["name"]: p["init"].clone() for p in params}, "master")
    zs = evaluate(model, evX, evY)
    zs_loss = train_loss(model, trX, trY)
    log(f"\n[GATE 1] quantized zero-shot batch-2: bal_acc={zs['balanced_accuracy']:.2f}%  "
        f"overall={zs['overall_accuracy']:.2f}%   (float-ZO ref zero-shot {REF['float_zero_shot']}%)")
    log(f"[GATE 1] per-class recall: {zs['per_class_recall']}")
    log(f"[GATE 2] mean training loss on the 54 calibration/training windows = {zs_loss:.4f}")

    # float (non-quantized) zero-shot, for context
    e1 = make_exporter(1)
    fm = e1.create_model()
    fm.eval()
    with torch.no_grad():
        fo = np.concatenate([fm(torch.from_numpy(evX[i:i + 1])).numpy()
                             for i in range(evX.shape[0])], 0)
    fp = fo.argmax(-1)
    frec = {int(c): float((fp[evY == c] == c).mean()) for c in np.unique(evY)}
    fbal = float(np.mean(list(frec.values())) * 100)
    log(f"[ctx]    float SpeechNet zero-shot batch-2 bal_acc = {fbal:.2f}%")

    results["zero_shot"] = dict(quantized=zs, quantized_train_loss=zs_loss,
                                float_model_balanced_accuracy=fbal)
    results["gates"] = {
        "gate1_zero_shot_close_to_float_ref": dict(
            value=zs["balanced_accuracy"], float_ref=REF["float_zero_shot"],
            float_model_measured=fbal,
            pass_=bool(zs["balanced_accuracy"] > 70.0)),
        "gate2_calibration_sane_loss": dict(value=zs_loss, pass_=bool(0.1 < zs_loss < 2.0)),
    }

    runs = []
    for lr in lrs_master:
        runs.append(("master", lr))
    best = None

    out = {}
    for mode, lr in runs:
        tag = f"{mode}@{lr:g}"
        log(f"\n=== {tag} ===")
        out[tag] = run_regime(model, params, trX, trY, evX, evY, mode, lr, n_steps, tag)
        if best is None or out[tag]["balanced_accuracy"] > out[best]["balanced_accuracy"]:
            best = tag
    best_lr = out[best]["lr"]
    log(f"\nbest MASTER lr = {best_lr:g} ({out[best]['balanced_accuracy']:.2f}%)")

    direct_lrs = sorted({best_lr, 3e-6})
    for lr in direct_lrs:
        tag = f"direct@{lr:g}"
        log(f"\n=== {tag} ===")
        out[tag] = run_regime(model, params, trX, trY, evX, evY, "direct", lr, n_steps, tag)

    # sanity: activation scales unchanged over the whole run
    acts1 = act_scales(model)
    drift = max(abs(acts0[k] - acts1[k]) for k in acts0) if acts0 else 0.0
    log(f"\n[check] activation-scale drift over the whole run = {drift:.3e} "
        f"({len(acts0)} activation quantizers)")
    results["activation_scale_drift"] = drift

    results["runs"] = out
    results["gates"]["gate3_direct_stall"] = {
        t: dict(pct_steps_zero_conv_weight_movement=out[t]["weight_movement"]
                ["pct_steps_zero_conv_weight_movement"],
                cum_pct_convw_int8_changed=out[t]["weight_movement"]["cum_pct_convw_int8_changed"])
        for t in out if t.startswith("direct")}
    results["gates"]["gate4_master_moves"] = {
        t: dict(cum_pct_convw_int8_changed=out[t]["weight_movement"]["cum_pct_convw_int8_changed"],
                max_latent_drift_LSB_convw=out[t]["weight_movement"].get("max_latent_drift_LSB_convw"),
                max_latent_drift_LSB=out[t]["weight_movement"].get("max_latent_drift_LSB"))
        for t in out if t.startswith("master")}

    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nwrote {HERE/'results.json'}")
    plot(results)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = results["runs"]
    zs = results["zero_shot"]["quantized"]["balanced_accuracy"]
    def by_lr(pfx):
        return sorted([t for t in out if t.startswith(pfx)], key=lambda t: out[t]["lr"])
    tags = ["zero-shot"] + by_lr("master") + by_lr("direct")
    vals = [zs] + [out[t]["balanced_accuracy"] for t in tags[1:]]
    cols = ["#8a8f98"] + ["#2f6fdb" if t.startswith("master") else "#d1495b" for t in tags[1:]]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    b = ax[0].bar(range(len(tags)), vals, color=cols)
    for r, v in zip(b, vals):
        ax[0].text(r.get_x() + r.get_width() / 2, v + 0.4, f"{v:.2f}", ha="center", fontsize=9)
    for k, c, ls in [("float_ZO_device", "#1b7f4b", "-"), ("float_ZO_sim", "#1b7f4b", "--"),
                     ("float_zero_shot", "#a06010", ":")]:
        ax[0].axhline(REF[k], color=c, ls=ls, lw=1.2,
                      label=f"{k.replace('_',' ')} {REF[k]}%")
    ax[0].set_xticks(range(len(tags)))
    ax[0].set_xticklabels(tags, rotation=25, ha="right", fontsize=9)
    ax[0].set_ylabel("batch-2 balanced accuracy (%)")
    ax[0].set_title("Quantized ZO fine-tuning of SpeechNet — whole batch 2 (180 windows)")
    ax[0].set_ylim(min(min(vals), 70) - 5, max(max(vals), 90) + 4)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(axis="y", alpha=.3)

    for t in tags[1:]:
        h = out[t]["history"]
        ax[1].plot(h["acc_step"], h["acc_bal"], marker="o", ms=3,
                   ls="-" if t.startswith("master") else "--",
                   label=t)
    ax[1].axhline(zs, color="#8a8f98", lw=1, label=f"quantized zero-shot {zs:.2f}%")
    ax[1].axhline(REF["float_ZO_device"], color="#1b7f4b", lw=1,
                  label=f"float ZO device {REF['float_ZO_device']}%")
    ax[1].set_xlabel("MeZO update step")
    ax[1].set_ylabel("batch-2 balanced accuracy (%)")
    ax[1].set_title("Accuracy during quantized ZO fine-tuning")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(HERE / "accuracy_summary.png", dpi=140)
    log(f"wrote {HERE/'accuracy_summary.png'}")


if __name__ == "__main__":
    main()
