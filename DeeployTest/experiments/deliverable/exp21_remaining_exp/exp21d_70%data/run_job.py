# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""exp21d -- more fine-tuning data at EQUAL update steps (= equal on-device latency).

One job = (recipe, arm, subject): one full incremental chain (rounds 1..4, fold 3, vocalized),
built from CleanUp/common WITHOUT modifying it. The only new code is a step-budget cap:

    CappedRecipe.update_steps(n) = min(epochs * n // n_accum, max_steps)

so the 70 % arm stops at exactly the baseline's update-step count even though that is not an integer
number of 70 % epochs (1620 steps = 51.43 epochs over 126 windows). The window visit order is the
device's cyclic `mb % n_windows`, so the cap simply truncates the device loop after `max_steps`
updates -- exactly what setting N_TRAIN_STEPS on device does.

The per-round metrics come from common.epoch_sweep.run_chain (val_next = accuracy on the NEXT batch,
val_in = held-out part of the current batch, train), i.e. the same code that produced exp6/11/12/13.

Run natively (python3.11 has torch 2.13 + brevitas 0.13):
    ONNX4DEEPLOY_ROOT=/Users/qiwenwu/ETH/Onnx4Deeploy TRAINDEEPLOY_ROOT=/Users/qiwenwu/ETH/TrainDeeploy \
    python3.11 run_job.py --recipe eco --arm p70_eq1620 --subject S01
"""
import argparse, dataclasses, json, os, sys, time
from datetime import datetime, timezone

import numpy as np
import torch

torch.set_num_threads(1)
CLEANUP = os.environ.get("CLEANUP_ROOT",
                         "/Users/qiwenwu/ETH/SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp")
sys.path.insert(0, CLEANUP)

from common import data                                   # noqa: E402
from common.epoch_sweep import run_chain, ROUNDS          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PERCENTILE = 99.9          # pooled@99.9 == epoch_sweep_driver default == exp6/11/12/13 (Table 3)

# arm -> (per_class, epochs, max_steps or 0). per_class 6 = 54 windows (30 %), 14 = 126 (70 %).
ARMS = {
    # ZO recipes (deployed: 120 epochs = 1620 steps at 30 %)
    "p30_e120":    (6, 120, 0),        # 1620 steps -- baseline
    "p70_eq1620":  (14, 52, 1620),     # 52 ep would be 1638 -> capped at 1620 (= 51.43 epochs)
    "p70_e120":    (14, 120, 0),       # 3780 steps -- "more steps", 2.33x latency
    # BP (deployed: 40 epochs = 540 steps at 30 %)
    "p30_e40":     (6, 40, 0),         # 540 steps -- BP deployed baseline
    "p70_eq540":   (14, 18, 540),      # 18 ep would be 567 -> capped at 540 (= 17.14 epochs)
    "p70_e40":     (14, 40, 0),        # 1260 steps -- "more steps", 2.33x latency
}


def capped(cls):
    @dataclasses.dataclass(frozen=True)
    class Capped(cls):
        max_steps: int = 0

        def update_steps(self, n_windows: int) -> int:
            steps = self.epochs * n_windows // self.n_accum
            return min(steps, self.max_steps) if self.max_steps else steps

        def describe(self) -> str:
            return super().describe() + f" | max_steps={self.max_steps}"
    Capped.__name__ = f"Capped{cls.__name__}"
    return Capped


def build(recipe_name, subject, fold, condition, per_class, seed, epochs, max_steps, log_round):
    """(base_recipe, start, train_round, model_of). log_round(dict) receives per-round step info."""
    if recipe_name in ("bp", "zo"):
        from common.speechnet import load_pretrained
        if recipe_name == "bp":
            from common.device_faithful import DeviceRecipe as R, finetune
        else:
            from common.zeroth_order import ZeroOrderRecipe as R, finetune
        base = capped(R)(seed=seed, per_class=per_class, epochs=epochs, max_steps=max_steps)
        ckpt = data.pretrained_checkpoint(subject, fold, condition)

        def train_round(model, X, y, _epochs):
            log_round({"n_windows": len(y), "update_steps": base.update_steps(len(y))})
            finetune(model, X, y, base)
            return model
        return base, (lambda: load_pretrained(ckpt)), train_round, (lambda m: m)

    from common import quantized
    from common.quantized_zeroth_order import initial_state
    if recipe_name == "qzo":
        from common.quantized_zeroth_order import QuantizedZeroOrderRecipe as R, finetune
    else:
        from common.eco import EcoRecipe as R, finetune
    base = capped(R)(seed=seed, per_class=per_class, epochs=epochs, max_steps=max_steps)
    _, model = quantized.build_quantized(subject, fold, condition)
    quantized.calibrate(model, subject, fold, condition, log=lambda *a: None,
                        percentile=CALIB_PERCENTILE)
    params = quantized.collect_parameters(model, float_fc=True)
    run_study, _, _ = quantized._qzo_modules()

    def start():
        state = initial_state(params, base.mode)
        with torch.no_grad():
            run_study.install(params, state, base.mode)
        return state

    def train_round(state, X, y, _epochs):
        info = {"n_windows": len(y), "update_steps": base.update_steps(len(y))}
        if recipe_name == "eco":
            stats = {}
            state = finetune(model, X, y, base, params, state, stats=stats)
            info.update(stats)
        else:
            state = finetune(model, X, y, base, params, state)
        log_round(info)
        return state

    def model_of(state):
        with torch.no_grad():
            run_study.install(params, state, base.mode)
        return model

    return base, start, train_round, model_of


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=("bp", "zo", "qzo", "eco"))
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--subject", required=True)
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--condition", default="vocalized")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = os.path.join(HERE, "results", args.recipe, args.arm, f"{args.subject}.json")
    if os.path.exists(out):
        print(f"skip {out}")
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)

    per_class, epochs, max_steps = ARMS[args.arm]
    rounds_info = []
    base, start, train_round, model_of = build(args.recipe, args.subject, args.fold,
                                               args.condition, per_class, args.seed, epochs,
                                               max_steps, rounds_info.append)
    print(f"{args.recipe} {args.arm} {args.subject}: {base.describe()}", flush=True)
    t0 = time.time()
    res = run_chain(args.subject, args.fold, args.condition, start, train_round, model_of,
                    per_class, args.seed, epochs)
    steps = [r["update_steps"] for r in rounds_info]
    n_accum = base.n_accum
    if args.recipe == "bp":
        fwd = [s * n_accum for s in steps]          # 1 forward + 1 backward per window visit
    else:
        fwd = [s * n_accum * 2 for s in steps]      # 2 forwards (theta +/- eps z) per window visit
    record = {
        "experiment": "exp21d_70%data", "recipe": args.recipe, "arm": args.arm,
        "subject": args.subject, "fold": args.fold, "condition": args.condition,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recipe_settings": base.as_dict(), "artifacts": data.ARTIFACTS_DIR,
        "calib_percentile": CALIB_PERCENTILE if args.recipe in ("qzo", "eco") else None,
        "per_round": {str(r): {**res[r], **rounds_info[i], "train_forwards": fwd[i]}
                      for i, r in enumerate(ROUNDS)},
        "update_steps_per_round": steps, "train_forwards_per_round": fwd,
        "mean_val_next": float(np.mean([res[r]["val_next"] for r in ROUNDS])),
        "mean_val_in": float(np.mean([res[r]["val_in"] for r in ROUNDS])),
        "seconds": time.time() - t0,
    }
    with open(out, "w") as h:
        json.dump(record, h, indent=1)
    print(f"DONE {args.recipe} {args.arm} {args.subject}: mean val_next "
          f"{record['mean_val_next']:.2f}  val_in {record['mean_val_in']:.2f}  steps {steps}  "
          f"({record['seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
