# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""exp21b -- QZO vs QZO+ECO ablation: one (arm, subject) job of the CleanUp incremental protocol.

Everything except the update rule is the exp4 (chosen QZO_ECO) harness:
    fold 3, vocalized, b1->b5 incremental (evaluate on b, fine-tune on 6/class of b, carry forward),
    int8 SpeechNet (Brevitas) conv + float fc, pooled@99.9 activation calibration recomputed per
    subject (passed EXPLICITLY -- CleanUp's constant default is 99.99), eps 0.01 (integer-step probe),
    n_accum 4 (MEAN), seed 42, device z stream, cyclic window order, device labelled draw.
    Epochs: DEFAULT 120 x 54 windows / 4 = 1620 update steps per round = the DEPLOYED recipe
    (silentwear-zo/kb/docs/SilentWearZO/rounds_summary.md:3; scripts/pw_sw_build_all.sh --n-steps 1620).
    --epochs 200 (2700 steps) = exp4's older setting, kept only as a secondary check (results_e200/).
The update rule comes from CleanUp/common/qzo_ablation.py (additive module).

MUST RUN IN `agitated_hugle` (Brevitas + Onnx4Deeploy QZO helpers):
    docker exec agitated_hugle bash -lc 'cd /app/TrainDeeploy/DeeployTest/experiments/deliverable/\
exp21_remaining_exp/exp21b_QZO_ECO_Abalation && python3 run_arm.py --arm A4_eco_sr --subject S01'
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

torch.set_num_threads(1)
CLEANUP = os.environ.get("CLEANUP_ROOT",
                         "/app/SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp")
sys.path.insert(0, CLEANUP)

from common import data, quantized                                    # noqa: E402
from common.protocol import BATCHES, mean_adapted                     # noqa: E402
from common.qzo_ablation import AblationRecipe, finetune              # noqa: E402
from common.quantized_zeroth_order import initial_state               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PERCENTILE = 99.9          # exp4's run (CleanUp default constant is 99.99; exp4 passed 99.9)
N_CHECKPOINTS = 10               # per round (+ step 0): every steps/10 update steps

# Matched Adam lr: Adam's per-element step is ~lr*|m_hat|/sqrt(v_hat) (dimensionless ratio), SGDM's is
# lr*|m|. With z Rademacher, sqrt(v_hat) = RMS(g_proj). So lr_adam = 3e-6 * RMS(g_proj) gives the same
# typical real-domain step as SGDM at 3e-6. RMS(g_proj) measured in the pilot (README step 3).
ADAM_MATCHED_LR = float(os.environ.get("ADAM_MATCHED_LR", "nan"))

# arm id -> (recipe overrides, short label). Shared defaults: AblationRecipe (lr 3e-6, beta .9, ...).
ARMS = {
    "A1_sgd_rtn":        (dict(optimizer="sgd",  rounding="rtn", ef="none"),       "SGD + RTN (direct int8 QZO)"),
    "A2_sgdm_rtn":       (dict(optimizer="sgdm", rounding="rtn", ef="none"),       "SGDM + RTN"),
    "A3_adam_sr":        (dict(optimizer="adam", rounding="sr",  ef="none"),       "Adam + SR"),
    "A4_eco_sr":         (dict(optimizer="sgdm", rounding="sr",  ef="memoryfree"), "SGDM + SR + EF (QZO_ECO, chosen)"),
    "A5_sgd_sr":         (dict(optimizer="sgd",  rounding="sr",  ef="none"),       "SGD + SR"),
    "A6_sgdm_sr":        (dict(optimizer="sgdm", rounding="sr",  ef="none"),       "SGDM + SR"),
    "A7_eco_rtn":        (dict(optimizer="sgdm", rounding="rtn", ef="memoryfree"), "SGDM + RTN + EF"),
    "A8_adam_rtn":       (dict(optimizer="adam", rounding="rtn", ef="none"),       "Adam + RTN"),
    # A0 = the DEPLOYED QZO (direct int8, lr 1e-5) -- the main baseline. (Named A9 in results_e200/.)
    "A0_qzo_deployed":   (dict(optimizer="sgd",  rounding="rtn", ef="none", learning_rate=1e-5),
                          "QZO deployed: SGD + RTN @ lr 1e-5"),
    "A10_sgd_carry_rtn": (dict(optimizer="sgd",  rounding="rtn", ef="carry"),      "SGD + RTN + EF(carry)"),
    "A11_sgd_carry_sr":  (dict(optimizer="sgd",  rounding="sr",  ef="carry"),      "SGD + SR + EF(carry)"),
    "A12_adam_sr_matched": (dict(optimizer="adam", rounding="sr", ef="none", learning_rate=ADAM_MATCHED_LR),
                            "Adam + SR @ matched lr"),
    "A13_sgd_sr_lr1e-5": (dict(optimizer="sgd",  rounding="sr",  ef="none", learning_rate=1e-5),
                          "SGD + SR @ lr 1e-5"),
    # reference point: the calibrated int8 model with NO fine-tuning (0 update steps every round)
    "Z0_no_ft":          (dict(optimizer="sgd", rounding="rtn", ef="none", epochs=0), "no fine-tuning (int8 zero-shot)"),
    "A14_adam_rtn_matched": (dict(optimizer="adam", rounding="rtn", ef="none", learning_rate=ADAM_MATCHED_LR),
                             "Adam + RTN @ matched lr"),
}


@torch.no_grad()
def evaluate(model, X, y):
    """Balanced accuracy (same computation as device_faithful.balanced_accuracy) + mean CE loss."""
    model.eval()
    logits = [model(torch.from_numpy(X[i:i + 1])) for i in range(len(y))]
    logits = torch.cat(logits, 0)
    pred = logits.argmax(1).numpy()
    loss = float(torch.nn.functional.cross_entropy(logits, torch.from_numpy(y.astype(np.int64))))
    classes = sorted(set(y.tolist()))
    acc = 100.0 * float(np.mean([float((pred[y == c] == c).mean()) for c in classes]))
    return acc, loss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--subject", required=True)
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--condition", default="vocalized")
    ap.add_argument("--epochs", type=int, default=120, help="120 = deployed (1620 steps)")
    ap.add_argument("--rounds", type=int, default=4, help="<4 only for pilots")
    ap.add_argument("--no-curves", action="store_true")
    ap.add_argument("--sr-seed", type=int, default=-1,
                    help="repeat runs: reseed ONLY the SR uniform stream (z, data, draw unchanged)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    overrides, label = ARMS[args.arm]
    overrides = dict(overrides)
    epochs = overrides.pop("epochs", args.epochs)
    recipe = dataclasses.replace(AblationRecipe(), epochs=epochs, sr_seed=args.sr_seed, **overrides)
    recipe.validate()
    assert not np.isnan(recipe.learning_rate), "set ADAM_MATCHED_LR for the matched-Adam arms"
    out = args.out or os.path.join(HERE, "results" if args.epochs == 120 else f"results_e{args.epochs}",
                                   args.arm, f"{args.subject}.json")
    if args.sr_seed >= 0 and not args.out:
        out = os.path.join(HERE, "results_srseed", f"{args.arm}_sr{args.sr_seed}", f"{args.subject}.json")
    checkpoint_every = None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    progress = os.path.splitext(out)[0] + ".progress"

    def note(msg):
        with open(progress, "a") as h:
            h.write(f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")
        print(msg, flush=True)

    started = time.time()
    note(f"{args.arm} {args.subject}: {recipe.describe()}")
    _, model = quantized.build_quantized(args.subject, args.fold, args.condition)
    thresholds = quantized.calibrate(model, args.subject, args.fold, args.condition,
                                     log=lambda *a: None, percentile=CALIB_PERCENTILE)
    params = quantized.collect_parameters(model, float_fc=True)
    state = initial_state(params, recipe.mode)
    base_int = {p["name"]: state[p["name"]].clone() for p in params if p["kind"] == "quant"}
    conv_names = [p["name"] for p in params if p["kind"] == "quant" and not p["name"].endswith(".bias")]
    n_conv = sum(base_int[n].numel() for n in conv_names)

    accuracies, rounds = [], []
    eval_sets = {b: data.evaluation_windows(args.subject, args.fold, b, args.condition) for b in BATCHES}
    for r, batch in enumerate(BATCHES):
        X, y = eval_sets[batch]
        acc, loss = evaluate(model, X, y)
        accuracies.append(acc)
        note(f"  b{batch} eval: acc={acc:.2f} loss={loss:.4f}")
        if batch == BATCHES[-1] or r >= args.rounds:
            continue
        X_train, y_train = data.labelled_subset(X, y, recipe.per_class, seed=recipe.seed,
                                                draw=recipe.draw)
        X_next, y_next = eval_sets[batch + 1]
        checkpoint_every = recipe.update_steps(len(y_train)) // N_CHECKPOINTS   # 162 @ 120 epochs

        def on_ck(step, _state):
            tr_acc, tr_loss = evaluate(model, X_train, y_train)
            nx_acc, nx_loss = evaluate(model, X_next, y_next)
            if step % (checkpoint_every * 5) == 0:
                note(f"    round{batch} step {step:4d}: train acc {tr_acc:.2f} loss {tr_loss:.4f} | "
                     f"next(b{batch + 1}) acc {nx_acc:.2f}")
            return {"train_acc": tr_acc, "train_loss": tr_loss, "next_acc": nx_acc, "next_loss": nx_loss}

        stats = {}
        state = finetune(model, X_train, y_train, recipe, params, state, stats,
                         checkpoint_every=0 if args.no_curves else checkpoint_every,
                         on_checkpoint=None if args.no_curves else on_ck)
        cum = sum(int((state[n] != base_int[n]).sum()) for n in conv_names) / n_conv
        stats["frac_conv_differ_from_base"] = cum
        stats["round"] = batch
        rounds.append(stats)
        note(f"  round {batch} done: moved_steps={100 * stats['moved_step_fraction']:.1f}% "
             f"differ(round)={100 * stats['frac_conv_differ_from_round_start']:.2f}% "
             f"ever={100 * stats['frac_conv_ever_moved']:.2f}% cum={100 * cum:.2f}% "
             f"flips={stats['total_flips']} [{time.time() - started:.0f}s]")

    result = {
        "experiment": "exp21b_QZO_ECO_Abalation", "arm": args.arm, "label": label,
        "subject": args.subject, "fold": args.fold, "condition": args.condition,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recipe": recipe.as_dict(), "calib_percentile": CALIB_PERCENTILE,
        "thresholds": thresholds, "artifacts": data.ARTIFACTS_DIR,
        "accuracies": accuracies,
        "mean_b2_b5": mean_adapted(accuracies) if len(accuracies) == 5 else None,
        "rounds": rounds, "seconds": time.time() - started,
        "checkpoint_every": checkpoint_every, "torch": torch.__version__,
    }
    with open(out + ".tmp", "w") as h:
        json.dump(result, h)
    os.replace(out + ".tmp", out)
    note(f"DONE {args.arm} {args.subject}: " + " ".join(f"b{b}={a:.2f}" for b, a in zip(BATCHES, accuracies))
         + (f" mean(b2-b5)={result['mean_b2_b5']:.2f}" if result["mean_b2_b5"] is not None else "")
         + f" [{result['seconds']:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
