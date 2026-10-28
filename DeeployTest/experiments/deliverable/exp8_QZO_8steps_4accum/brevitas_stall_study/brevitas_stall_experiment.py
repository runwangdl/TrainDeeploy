# SPDX-License-Identifier: MIT
"""
Brevitas "weight LSB stall" experiment for quantized zeroth-order (MeZO) fine-tuning
of SpeechNet.

Tests: is the on-device QZO int8 weight "stall" (0/N weights move) a REAL property of
DIRECT-INT8 MeZO updates, and does it DISAPPEAR under "master weights" (a latent fp32
copy re-quantized each step, = standard Brevitas QAT fine-tuning)?  And is the weight
scale s_w data-free (independent of activation calibration)?

Three regimes, identical everything else (same frozen s_w, same z per step, same windows,
same g_proj formula). The ONLY difference is the UPDATE RULE:
  Regime A (MASTER):    maintain W_latent fp32, update W_latent += coeff*z (unrounded),
                        re-quantize to int8 each step.        [4 B/weight extra state]
  Regime B (DIRECT):    maintain int8 directly, delta = round(coeff*z / s_w),
                        int8 += delta (sub-LSB discarded -> memoryless).   [0 B, stalls]
  Regime C (STOCHASTIC) maintain int8 directly, delta = floor(x) + Bernoulli(frac(x)),
                        x = coeff*z/s_w. E[delta] = x exactly -> unbiased, so sub-LSB
                        steps are applied in expectation.     [0 B, pays in variance]

Writes results.json, cumulative_pct_moved.png, run.log to this directory.

DOES NOT modify any existing source file. Only reuses the exporter's public helpers.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # /app/Onnx4Deeploy
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Config (mimics Onnx4Deeploy.py -mode q-zo-train construction for SpeechNet)
# ---------------------------------------------------------------------------
CKPT = "/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt"
DATA_PATH = "/app/SilentWear/SilentWear_data/data_raw_and_filt"

# MeZO recipe (match exp8)
LR_MAIN = 1e-5
EPS = 0.01
N_ACCUM = 4
Q = 1
SEED = 42
N_STEPS = 100
N_SWEEP_STEPS = 30
LR_SWEEP = [1e-5, 1e-4, 1e-3]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Build + calibrate the Brevitas QuantSpeechNet exactly like _export_qzo_training
# ---------------------------------------------------------------------------
def build_model_and_data():
    from onnx4deeploy.models.speechnet_exporter import SpeechNetExporter
    from brevitas.graph.calibrate import calibration_mode

    exporter = SpeechNetExporter()
    exporter._config_overrides = {
        "dataset": "silentwear",
        "data_path": DATA_PATH,
        "pretrained_weights": CKPT,
        "pretrained_key": "model_state_dict",
        "subject": "S01",
        "session": 3,
        "batch": 1,
        "condition": "vocalized",
        "batch_size": 1,
        "num_classes": 9,
        "calib_samples": 8,
        "stratified_sampling": True,
    }
    # load_config() populates both self.model_config and (returned) self.config
    exporter.config = exporter.load_config()

    ishape = exporter.get_input_shape()  # (1,1,14,700)
    log(f"input shape = {ishape}")

    # Real SilentWear windows (stratified) for calibration + the MeZO forwards.
    ds = exporter.get_data_source()
    # request enough windows to cover the study; stratified sampling covers all 9 classes
    n_windows_req = 54
    X, Y = ds.load_batches(n_windows_req, (1,) + tuple(ishape[1:]),
                           exporter.config["num_classes"], seed=SEED)
    windows = np.concatenate([np.asarray(a, np.float32) for a in X], 0)  # (N,1,14,700)
    labels = np.asarray([int(np.asarray(y).reshape(-1)[0]) for y in Y], np.int64)
    log(f"loaded {windows.shape[0]} real windows; label dist = "
        f"{dict(zip(*np.unique(labels, return_counts=True)))}")

    # -- Model A: the calibrated model used for BOTH regimes' forwards -----------
    model = exporter.create_brevitas_model()
    model.eval()

    calib = windows[: exporter.config.get("calib_samples", 8)].astype(np.float32)
    with torch.no_grad(), calibration_mode(model):
        model(torch.from_numpy(calib))
    model.eval()  # freeze BN running stats + activation scales

    return exporter, model, windows, labels, ishape


# ---------------------------------------------------------------------------
# Extract the quantized conv/fc layers and their FROZEN per-channel weight scales
# ---------------------------------------------------------------------------
class ConstScale(torch.nn.Module):
    """Replaces Brevitas' StatsFromParameterScaling so s_w can NEVER re-track the latent weight.

    Brevitas re-derives the weight scale from the parameter on every quant_weight() call
    (StatsFromParameterScaling, brevitas/core/scaling/runtime.py:25). During fine-tuning that
    means the grid follows the weights -- exactly the "that's cheating" case. Swapping the
    scaling_impl for this constant module pins s_w to its post-calibration value while leaving
    the rest of Brevitas' quantization path (round -> clamp -> *scale) fully in charge.
    """

    def __init__(self, v: torch.Tensor):
        super().__init__()
        self.register_buffer("v", v.detach().clone())

    def forward(self, *a, **k):
        return self.v


def collect_quant_layers(model):
    """Return list of (name, module, s_w tensor, W_latent_init tensor) for conv+fc.

    ALSO freezes each layer's weight scale by module surgery, so that from here on Brevitas
    itself performs the quantization with a fixed s_w (faithful methodology: we do NOT
    pre-quantize and inject; we hand Brevitas the latent weight and let it quantize).
    """
    import brevitas.nn as qnn
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, (qnn.QuantConv2d, qnn.QuantLinear)):
            qw = m.quant_weight()
            s_w = qw.scale.detach().clone()          # per-output-channel [Cout,1,1,1] or [Cout,1]
            W_init = m.weight.detach().clone()        # latent fp32 weight
            # FREEZE: pin the scale so it cannot re-track the latent weight during training.
            m.weight_quant.tensor_quant.scaling_impl = ConstScale(s_w)
            layers.append({"name": name, "module": m, "s_w": s_w,
                           "W_init": W_init,
                           "is_conv": isinstance(m, qnn.QuantConv2d)})
    return layers


def fake_quant(W, s_w):
    """round(W/s_w)*s_w, clamp int to [-127,127]. s_w broadcasts over output channel."""
    q = torch.clamp(torch.round(W / s_w), -127, 127)
    return q * s_w


def to_int8(W, s_w):
    return torch.clamp(torch.round(W / s_w), -127, 127)


# ---------------------------------------------------------------------------
# Quantized forward-loss L(weights-per-layer, x, y)
# ---------------------------------------------------------------------------
def loss_with_weights(model, layers, weight_override, x, y):
    """
    Run the model with each conv/fc's LATENT fp32 weight temporarily set to the given tensor,
    and let BREVITAS perform the quantization.

    Methodology note (this is the faithful path, and it matters):
      We install the RAW latent weight into `m.weight.data` -- NOT a pre-quantized one -- and
      Brevitas' own forward then computes round(W/s_w) -> clamp -> *s_w via
      quant_layer.py:146 -> mixin/parameter.py:49 -> core/quant/int_base.py:54-74.
      Because collect_quant_layers() pinned scaling_impl to a ConstScale, that s_w is the
      frozen post-calibration value and cannot re-track the weight.

      (An earlier version pre-quantized with a local fake_quant() and injected the result,
      intending to bypass abs-max re-tracking. That was NOT a no-op: Brevitas re-derived a
      slightly finer scale from the injected tensor and re-rounded, shifting some elements by
      1 LSB. Freezing scaling_impl is the correct fix and lets Brevitas stay in charge.)

    weight_override: dict name -> latent fp32 W to install.
    Returns scalar CE loss (fp32).
    """
    saved = {}
    for L in layers:
        m = L["module"]
        saved[L["name"]] = m.weight.data
        m.weight.data = weight_override[L["name"]]   # LATENT weight; Brevitas quantizes it
    try:
        with torch.no_grad():
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
        return float(loss.item())
    finally:
        for L in layers:
            L["module"].weight.data = saved[L["name"]]


# ---------------------------------------------------------------------------
# The two regimes
# ---------------------------------------------------------------------------
def run_regimes(model, layers, windows, labels, lr, n_steps, eps=EPS,
                n_accum=N_ACCUM, seed=SEED, trainable_conv_only=True,
                verbose=False):
    """
    Run Regime A (master) and Regime B (direct int8) with SHARED z per step, SHARED
    windows, SHARED g_proj. Returns per-step measurement dicts for A and B.

    trainable_conv_only: restrict the perturbed/updated params to the 5 conv weights
    (where the stall lives). fc + biases stay at their init fake-quant values for the
    forward (frozen).
    """
    dev = next(model.parameters()).device
    # which layers are trainable (perturbed + updated)
    trainable = [L for L in layers if (L["is_conv"] if trainable_conv_only else True)]

    # frozen (non-trainable) weights: use init fake-quant, never change
    frozen_override = {}
    for L in layers:
        frozen_override[L["name"]] = L["W_init"].clone()

    # Regime A state: latent fp32 per trainable layer, init = pretrained latent weight
    A_latent = {L["name"]: L["W_init"].clone() for L in trainable}
    A_latent_init = {L["name"]: L["W_init"].clone() for L in trainable}
    # Regime B state: int8 grid per trainable layer, init = round(W_init/s_w)
    B_int8 = {L["name"]: to_int8(L["W_init"], L["s_w"]) for L in trainable}
    B_int8_init = {k: v.clone() for k, v in B_int8.items()}

    # Regime C state: int8 grid, updated with STOCHASTIC ROUNDING (no fp32 state at all).
    #   delta = floor(x) + Bernoulli(frac(x))   with x = coeff*z/s_w
    #   -> E[delta] = x exactly, so a sub-LSB update is applied in expectation rather than
    #      discarded. Costs ZERO extra memory (the device kernel already owns an RNG), at the
    #      price of variance. This is the third escape from the stall, alongside master
    #      weights (A) and error feedback. -- QW
    C_int8 = {L["name"]: to_int8(L["W_init"], L["s_w"]) for L in trainable}
    C_int8_init = {k: v.clone() for k, v in C_int8.items()}

    # cache initial int8 for A too (all regimes start from the SAME int8 grid)
    A_int8_prev = {L["name"]: to_int8(L["W_init"], L["s_w"]) for L in trainable}
    B_int8_prev = {k: v.clone() for k, v in B_int8.items()}
    C_int8_prev = {k: v.clone() for k, v in C_int8.items()}

    # total trainable conv-weight element count
    tot = sum(int(L["W_init"].numel()) for L in trainable)

    rng = np.random.RandomState(seed)
    n_win = windows.shape[0]

    resA = {"pct_moved_step": [], "pct_moved_cumulative": [],
            "max_abs_latent_drift": [], "mean_abs_latent_drift": [], "g_proj": [],
            "coeff": [], "L_plus_mean": [], "L_minus_mean": []}
    resB = {"pct_moved_step": [], "pct_moved_cumulative": [], "g_proj": [], "coeff": []}
    resC = {"pct_moved_step": [], "pct_moved_cumulative": [], "g_proj": [], "coeff": [],
            "expected_lsb_step": [], "realized_lsb_step": []}

    def build_override(state, is_latent):
        """Merge frozen (non-trainable) + trainable state into a full latent-weight override."""
        ov = dict(frozen_override)
        for L in trainable:
            if is_latent:
                ov[L["name"]] = state[L["name"]]                 # fp32 latent (A)
            else:
                ov[L["name"]] = state[L["name"]] * L["s_w"]      # int8*s_w -> latent-equiv (B)
        return ov

    for u in range(n_steps):
        # draw z (Rademacher +/-1) once, shared by A and B, per trainable layer
        z = {}
        for L in trainable:
            zt = torch.from_numpy(
                rng.choice([-1.0, 1.0], size=tuple(L["W_init"].shape)).astype(np.float32)
            ).to(dev)
            z[L["name"]] = zt

        # ---- accumulate g_proj (shared windows) --------------------------------
        accA = 0.0
        accB = 0.0
        accC = 0.0
        lpA = lmA = 0.0
        for a in range(n_accum):
            idx = (u * n_accum + a) % n_win
            x = torch.from_numpy(windows[idx:idx + 1]).to(dev)
            y = torch.tensor([int(labels[idx])], dtype=torch.long, device=dev)

            # ---- Regime A: perturb the fp32 latent by +/- eps*z --------------
            ovA_p = build_override(A_latent, is_latent=True)
            ovA_m = dict(ovA_p)
            for L in trainable:
                ovA_p[L["name"]] = A_latent[L["name"]] + eps * z[L["name"]]
                ovA_m[L["name"]] = A_latent[L["name"]] - eps * z[L["name"]]
            Lp_A = loss_with_weights(model, layers, ovA_p, x, y)
            Lm_A = loss_with_weights(model, layers, ovA_m, x, y)
            accA += (Lp_A - Lm_A)
            lpA += Lp_A
            lmA += Lm_A

            # ---- Regime B: perturb the int8 grid via integer perturb ----------
            # mirror the device: perturb the int8 by round(eps/s_w)*z (integer grid),
            # dequant to latent-equiv = (int8 + dz_int)*s_w.
            ovB_p = build_override(B_int8, is_latent=False)
            ovB_m = dict(ovB_p)
            for L in trainable:
                dz_int = torch.round(torch.tensor(eps) / L["s_w"]) * z[L["name"]]
                ovB_p[L["name"]] = (B_int8[L["name"]] + dz_int) * L["s_w"]
                ovB_m[L["name"]] = (B_int8[L["name"]] - dz_int) * L["s_w"]
            Lp_B = loss_with_weights(model, layers, ovB_p, x, y)
            Lm_B = loss_with_weights(model, layers, ovB_m, x, y)
            accB += (Lp_B - Lm_B)

            # ---- Regime C: identical forward to B (int8 grid, integer perturb) -
            # C differs from B ONLY in the UPDATE rule (stochastic vs nearest rounding),
            # so its probe path is the same integer perturbation.
            ovC_p = build_override(C_int8, is_latent=False)
            ovC_m = dict(ovC_p)
            for L in trainable:
                dz_int = torch.round(torch.tensor(eps) / L["s_w"]) * z[L["name"]]
                ovC_p[L["name"]] = (C_int8[L["name"]] + dz_int) * L["s_w"]
                ovC_m[L["name"]] = (C_int8[L["name"]] - dz_int) * L["s_w"]
            Lp_C = loss_with_weights(model, layers, ovC_p, x, y)
            Lm_C = loss_with_weights(model, layers, ovC_m, x, y)
            accC += (Lp_C - Lm_C)

        gA = accA / (2.0 * eps * n_accum)
        gB = accB / (2.0 * eps * n_accum)
        gC = accC / (2.0 * eps * n_accum)
        coeffA = -lr * gA
        coeffB = -lr * gB
        coeffC = -lr * gC

        # ---- Regime A update: W_latent += coeff*z (unrounded fp32) --------------
        for L in trainable:
            A_latent[L["name"]] = A_latent[L["name"]] + coeffA * z[L["name"]]
        A_int8_now = {L["name"]: to_int8(A_latent[L["name"]], L["s_w"]) for L in trainable}

        # ---- Regime B update: delta_int8 = round(coeff*z / s_w) ----------------
        for L in trainable:
            delta = torch.round(coeffB * z[L["name"]] / L["s_w"])
            B_int8[L["name"]] = torch.clamp(B_int8[L["name"]] + delta, -127, 127)
        B_int8_now = {k: v.clone() for k, v in B_int8.items()}

        # ---- Regime C update: STOCHASTIC ROUNDING (unbiased, zero extra memory) -
        #   x     = coeff*z/s_w                       (the desired step, in LSB; ~0.07 here)
        #   delta = floor(x) + Bernoulli(x - floor(x))
        #   => E[delta] = x  exactly, so sub-LSB steps are applied in expectation instead of
        #      being discarded. Handles negative x correctly via floor (not trunc).
        exp_lsb = 0.0
        real_lsb = 0.0
        for L in trainable:
            x = coeffC * z[L["name"]] / L["s_w"]
            fl = torch.floor(x)
            prob = x - fl                                       # in [0,1)
            u01 = torch.from_numpy(
                rng.random_sample(size=tuple(x.shape)).astype(np.float32)).to(x.device)
            delta = fl + (u01 < prob).to(x.dtype)               # stochastic round
            C_int8[L["name"]] = torch.clamp(C_int8[L["name"]] + delta, -127, 127)
            exp_lsb += float(x.abs().sum())
            real_lsb += float(delta.abs().sum())
        C_int8_now = {k: v.clone() for k, v in C_int8.items()}

        # ---- measurements -------------------------------------------------------
        # A
        moved_step_A = sum(int((A_int8_now[k] != A_int8_prev[k]).sum()) for k in A_int8_now)
        moved_cum_A = sum(int((A_int8_now[k] != B_int8_init[k]).sum()) for k in A_int8_now)
        drift_max = max(float((A_latent[k] - A_latent_init[k]).abs().max()) for k in A_latent)
        drift_mean = float(np.mean([float((A_latent[k] - A_latent_init[k]).abs().mean())
                                    for k in A_latent]))
        resA["pct_moved_step"].append(100.0 * moved_step_A / tot)
        resA["pct_moved_cumulative"].append(100.0 * moved_cum_A / tot)
        resA["max_abs_latent_drift"].append(drift_max)
        resA["mean_abs_latent_drift"].append(drift_mean)
        resA["g_proj"].append(gA)
        resA["coeff"].append(coeffA)
        resA["L_plus_mean"].append(lpA / n_accum)
        resA["L_minus_mean"].append(lmA / n_accum)
        A_int8_prev = A_int8_now

        # B
        moved_step_B = sum(int((B_int8_now[k] != B_int8_prev[k]).sum()) for k in B_int8_now)
        moved_cum_B = sum(int((B_int8_now[k] != B_int8_init[k]).sum()) for k in B_int8_now)
        resB["pct_moved_step"].append(100.0 * moved_step_B / tot)
        resB["pct_moved_cumulative"].append(100.0 * moved_cum_B / tot)
        resB["g_proj"].append(gB)
        resB["coeff"].append(coeffB)
        B_int8_prev = B_int8_now

        # C (stochastic rounding)
        moved_step_C = sum(int((C_int8_now[k] != C_int8_prev[k]).sum()) for k in C_int8_now)
        moved_cum_C = sum(int((C_int8_now[k] != C_int8_init[k]).sum()) for k in C_int8_now)
        resC["pct_moved_step"].append(100.0 * moved_step_C / tot)
        resC["pct_moved_cumulative"].append(100.0 * moved_cum_C / tot)
        resC["g_proj"].append(gC)
        resC["coeff"].append(coeffC)
        resC["expected_lsb_step"].append(exp_lsb / tot)     # mean |desired step| in LSB
        resC["realized_lsb_step"].append(real_lsb / tot)    # mean |applied  step| in LSB
        C_int8_prev = C_int8_now

        if verbose and (u < 5 or u % 20 == 0):
            log(f"  step {u:3d} lr={lr:.0e}  "
                f"A: step%={resA['pct_moved_step'][-1]:.3f} cum%={resA['pct_moved_cumulative'][-1]:.3f} "
                f"drift_max={drift_max:.2e}  |  "
                f"B: step%={resB['pct_moved_step'][-1]:.3f} cum%={resB['pct_moved_cumulative'][-1]:.3f}  "
                f"coeff={coeffB:.2e}")

    return resA, resB, resC, tot


# ---------------------------------------------------------------------------
# Calibration-hypothesis check: s_w is data-free
# ---------------------------------------------------------------------------
def data_free_check(exporter):
    """Build a FRESH pretrained model WITHOUT calibration, compute s_w = max|W[c]|/127,
    and compare to the post-calibration quant_weight scale."""
    import brevitas.nn as qnn
    m_nocal = exporter.create_brevitas_model()
    m_nocal.eval()  # NO calibration_mode forward at all
    out = {}
    for name, m in m_nocal.named_modules():
        if isinstance(m, (qnn.QuantConv2d, qnn.QuantLinear)):
            W = m.weight.detach()
            dims = tuple(range(1, W.dim()))
            absmax = W.abs().amax(dim=dims) / 127.0            # data-free
            s_from_quant = m.quant_weight().scale.detach().flatten()
            maxdiff = float((absmax.flatten() - s_from_quant).abs().max())
            out[name] = {"s_absmax_head": absmax.flatten()[:3].tolist(),
                         "s_quant_head": s_from_quant[:3].tolist(),
                         "max_abs_diff": maxdiff}
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    exporter, model, windows, labels, ishape = build_model_and_data()
    layers = collect_quant_layers(model)

    # calibrated loss sanity (should be ~1.2-1.5, not ~9)
    with torch.no_grad():
        losses = []
        for i in range(min(16, windows.shape[0])):
            x = torch.from_numpy(windows[i:i + 1])
            y = torch.tensor([int(labels[i])])
            logits = model(x)
            losses.append(float(torch.nn.functional.cross_entropy(logits, y)))
    calibrated_loss = float(np.mean(losses))
    log(f"\n== calibrated mean CE loss over {len(losses)} real windows = {calibrated_loss:.4f} "
        f"(expect ~1.2-1.5; ~9 => failed calibration) ==")

    # record s_w for all conv + fc
    s_w_record = {}
    for L in layers:
        s_w_record[L["name"]] = {"shape": list(L["s_w"].shape),
                                 "values": L["s_w"].flatten().tolist(),
                                 "is_conv": L["is_conv"]}
    log("\n== frozen per-channel weight scales s_w ==")
    for L in layers:
        s = L["s_w"].flatten()
        log(f"  {L['name']:16s} n={s.numel():3d}  s_w[min,mean,max]="
            f"[{float(s.min()):.5f},{float(s.mean()):.5f},{float(s.max()):.5f}]")

    # sanity: s_w == abs-max/127 on the CALIBRATED model too
    log("\n== s_w == max|W[c]|/127 sanity (calibrated model) ==")
    for L in layers:
        W = L["module"].weight.detach()
        dims = tuple(range(1, W.dim()))
        absmax = (W.abs().amax(dim=dims) / 127.0).flatten()
        md = float((absmax - L["s_w"].flatten()).abs().max())
        log(f"  {L['name']:16s} max|absmax/127 - s_w| = {md:.3e}")

    # ---- data-free calibration-hypothesis check --------------------------------
    log("\n== DATA-FREE CHECK: s_w WITHOUT any calibration vs WITH calibration ==")
    df = data_free_check(exporter)
    df_maxdiff = 0.0
    for name, d in df.items():
        # compare no-cal absmax to the calibrated frozen s_w
        cal_s = next(L for L in layers if L["name"] == name)["s_w"].flatten().numpy()
        diff = float(np.max(np.abs(np.array(d["s_quant_head"]) - cal_s[:3])))
        df_maxdiff = max(df_maxdiff, d["max_abs_diff"], diff)
        log(f"  {name:16s} |nocal_absmax - nocal_quant|={d['max_abs_diff']:.3e}  "
            f"|nocal_quant - cal_s (head)|={diff:.3e}")
    log(f"  => max |s_w(no-cal) - s_w(cal)| across all layers = {df_maxdiff:.3e}")

    # ---- step-0 round-trip check -----------------------------------------------
    log("\n== step-0 int8 round-trip: round(W_init/s_w)*s_w re-quant == itself ==")
    rt_ok = True
    for L in layers:
        i8 = to_int8(L["W_init"], L["s_w"])
        Wq = i8 * L["s_w"]
        i8b = to_int8(Wq, L["s_w"])
        ok = bool(torch.equal(i8, i8b))
        rt_ok = rt_ok and ok
        if not ok:
            log(f"  {L['name']}: ROUND-TRIP FAIL")
    log(f"  round-trip stable for all layers: {rt_ok}")

    # ---- FREEZE GATE: prove Brevitas' own quantizer now uses the pinned s_w -------
    # (a) perturbing the latent weight must NOT move the scale, and
    # (b) Brevitas' quant_weight() must equal our reference round(W/s_w)*s_w exactly.
    log("\n== freeze gate: Brevitas quantizes with the PINNED s_w (no re-tracking) ==")
    freeze_ok, max_scale_drift, max_val_err = True, 0.0, 0.0
    for L in layers:
        m = L["module"]
        saved = m.weight.data
        try:
            # a deliberately large latent perturbation: would move an abs-max scale a lot
            W_test = L["W_init"] * 0.3 + 0.7 * L["s_w"] * torch.sign(torch.randn_like(L["W_init"]))
            m.weight.data = W_test
            s_now = m.quant_weight().scale.detach()
            drift = float((s_now - L["s_w"]).abs().max())
            # Brevitas' quantized value vs our frozen-scale reference
            ref = to_int8(W_test, L["s_w"]) * L["s_w"]
            err = float((m.quant_weight().value.detach() - ref).abs().max())
        finally:
            m.weight.data = saved
        max_scale_drift = max(max_scale_drift, drift)
        max_val_err = max(max_val_err, err)
        if drift > 0.0 or err > 1e-9:
            freeze_ok = False
            log(f"  {L['name']}: scale drift={drift:.3e}  value err={err:.3e}")
    log(f"  max scale drift = {max_scale_drift:.3e} (must be 0.0)")
    log(f"  max |brevitas_quant_weight - frozen-scale reference| = {max_val_err:.3e} (must be 0)")
    log(f"  FREEZE GATE: {'PASS' if freeze_ok else 'FAIL'}")

    # ---- MAIN RUN: 100 steps at lr=1e-5 ----------------------------------------
    log(f"\n{'='*70}\nMAIN RUN: {N_STEPS} steps, lr={LR_MAIN:.0e}, eps={EPS}, n_accum={N_ACCUM}, seed={SEED}\n{'='*70}")
    resA, resB, resC, tot = run_regimes(model, layers, windows, labels, LR_MAIN, N_STEPS,
                                  trainable_conv_only=True, verbose=True)
    log(f"\ntrainable conv-weight elements = {tot}")
    log(f"Regime B (device)  lr=1e-5: cum% moved @100 = {resB['pct_moved_cumulative'][-1]:.4f}  "
        f"max step% = {max(resB['pct_moved_step']):.4f}")
    log(f"Regime A (master)  lr=1e-5: cum% moved @100 = {resA['pct_moved_cumulative'][-1]:.4f}  "
        f"final latent drift_max = {resA['max_abs_latent_drift'][-1]:.3e}")

    # first step where A / B first move
    def first_move(res):
        for i, v in enumerate(res["pct_moved_cumulative"]):
            if v > 0:
                return i
        return None
    log(f"  Regime A first int8 move at step = {first_move(resA)}")
    log(f"  Regime B first int8 move at step = {first_move(resB)}")

    # ---- lr SWEEP --------------------------------------------------------------
    log(f"\n{'='*70}\nLR SWEEP: {N_SWEEP_STEPS} steps each, lr in {LR_SWEEP}\n{'='*70}")
    sweep = {}
    for lr in LR_SWEEP:
        rA, rB, rC, _ = run_regimes(model, layers, windows, labels, lr, N_SWEEP_STEPS,
                                trainable_conv_only=True, verbose=False)
        sweep[f"{lr:.0e}"] = {
            "A_cum_pct": rA["pct_moved_cumulative"],
            "B_cum_pct": rB["pct_moved_cumulative"],
            "C_cum_pct": rC["pct_moved_cumulative"],
            "C_final_cum": rC["pct_moved_cumulative"][-1],
            "A_final_cum": rA["pct_moved_cumulative"][-1],
            "B_final_cum": rB["pct_moved_cumulative"][-1],
            "A_final_drift_max": rA["max_abs_latent_drift"][-1],
        }
        log(f"  lr={lr:.0e}  A cum%@{N_SWEEP_STEPS}={rA['pct_moved_cumulative'][-1]:8.4f}  "
            f"B cum%@{N_SWEEP_STEPS}={rB['pct_moved_cumulative'][-1]:8.4f}")

    # ---- validation checkpoints ------------------------------------------------
    B_first5_max = max(resB["pct_moved_step"][:5])
    chk = {
        "B_stalled_first5steps_lr1e5": bool(B_first5_max < 1e-9),   # ~0% for first several steps
        "A_eventually_moves_lr1e5": bool(resA["pct_moved_cumulative"][-1] > 0),
        "step0_roundtrip_ok": bool(rt_ok),
        "s_w_data_free_maxdiff_lt_1e6": bool(df_maxdiff < 1e-6),
        # Brevitas itself does the quantization, with the scale pinned to its calibrated value
        "C_stochround_moves_lr1e5": bool(resC["pct_moved_cumulative"][-1] > 0),
        "freeze_gate_brevitas_uses_pinned_s_w": bool(freeze_ok),
        "freeze_max_scale_drift": float(max_scale_drift),
        "freeze_max_value_err": float(max_val_err),
    }
    log(f"\n{'='*70}\nVALIDATION CHECKPOINTS\n{'='*70}")
    for k, v in chk.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")

    # ---- dump results.json -----------------------------------------------------
    results = {
        "config": {"lr_main": LR_MAIN, "eps": EPS, "n_accum": N_ACCUM, "q": Q,
                   "seed": SEED, "n_steps": N_STEPS, "n_sweep_steps": N_SWEEP_STEPS,
                   "lr_sweep": LR_SWEEP, "trainable_conv_only": True,
                   "input_shape": list(ishape), "ckpt": CKPT},
        "calibrated_loss": calibrated_loss,
        "trainable_conv_weight_elements": tot,
        "s_w": s_w_record,
        "data_free_check": {"per_layer": df, "max_abs_diff_nocal_vs_cal": df_maxdiff},
        "regime_A_master_lr1e5": resA,
        "regime_B_direct_lr1e5": resB,
        "regime_C_stochround_lr1e5": resC,
        "lr_sweep": sweep,
        "validation_checkpoints": chk,
        "first_move": {"A": first_move(resA), "B": first_move(resB)},
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nwrote {HERE / 'results.json'}")

    # ---- plot ------------------------------------------------------------------
    make_plot(resA, resB, resC, sweep)
    log(f"wrote {HERE / 'cumulative_pct_moved.png'}")

    return results


def make_plot(resA, resB, resC, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    steps = np.arange(1, len(resA["pct_moved_cumulative"]) + 1)
    ax.plot(steps, resA["pct_moved_cumulative"], "-o", ms=3, label="Regime A (master weights)")
    ax.plot(steps, resB["pct_moved_cumulative"], "-s", ms=3, label="Regime B (direct int8, device)")
    ax.plot(steps, resC["pct_moved_cumulative"], "-^", ms=3, label="Regime C (stochastic rounding)")
    ax.set_xlabel("MeZO update step")
    ax.set_ylabel("cumulative % int8 conv-weights moved vs init")
    ax.set_title(f"Cumulative % moved  (lr={LR_MAIN:.0e}, eps={EPS}, n_accum={N_ACCUM})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for lr in LR_SWEEP:
        s = sweep[f"{lr:.0e}"]
        st = np.arange(1, len(s["A_cum_pct"]) + 1)
        ax2.plot(st, s["A_cum_pct"], "-", label=f"A master lr={lr:.0e}")
        ax2.plot(st, s["B_cum_pct"], "--", label=f"B direct lr={lr:.0e}")
        ax2.plot(st, s["C_cum_pct"], ":", label=f"C stoch lr={lr:.0e}")
    ax2.set_xlabel("MeZO update step")
    ax2.set_ylabel("cumulative % int8 conv-weights moved")
    ax2.set_title("lr sweep: master (solid) / direct (dashed) / stochastic (dotted)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(HERE / "cumulative_pct_moved.png", dpi=130)


if __name__ == "__main__":
    main()
