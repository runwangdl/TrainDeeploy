# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""exp21f -- BP epoch sweep (convergence of the deployed backprop recipe).

One job = (subject, seed, epochs): ONE full incremental chain (rounds 1..4, fold 3, vocalized) where
EVERY round trains `epochs` epochs and round r+1 starts from round r's model trained with that same
epoch count (common.epoch_sweep.run_chain -- separate chain per epoch setting, NOT checkpoints of one
long run). Everything else is the deployed BP recipe = common.device_faithful.DeviceRecipe defaults
(SGD lr 3e-4, n_accum 4 SUM, per_class 6 = 54 windows, cyclic order, device draw, dropout 0,
frozen BN stats, batch 1, artifacts_reference). CleanUp/common is imported, never modified.

    python3.11 run_job.py --subject S01 --epochs 40 [--seed 42]
"""
import argparse, json, os, sys, time
from datetime import datetime, timezone

import numpy as np
import torch

torch.set_num_threads(1)
CLEANUP = os.environ.get("CLEANUP_ROOT",
                         "/Users/qiwenwu/ETH/SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp")
sys.path.insert(0, CLEANUP)

from common import data                                        # noqa: E402
from common.device_faithful import DeviceRecipe, finetune      # noqa: E402
from common.epoch_sweep import run_chain, ROUNDS               # noqa: E402
from common.speechnet import load_pretrained                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--condition", default="vocalized")
    args = ap.parse_args()

    out = os.path.join(HERE, "results", f"seed{args.seed}", f"e{args.epochs:03d}", f"{args.subject}.json")
    if os.path.exists(out):
        print(f"skip {out}")
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)

    recipe = DeviceRecipe(epochs=args.epochs, seed=args.seed)   # ONLY epochs (and seed) differ
    ckpt = data.pretrained_checkpoint(args.subject, args.fold, args.condition)
    steps = []

    def train_round(model, X, y, _epochs):
        steps.append(recipe.update_steps(len(y)))
        finetune(model, X, y, recipe)
        return model

    print(f"bp e{args.epochs} seed{args.seed} {args.subject}: {recipe.describe()}", flush=True)
    t0 = time.time()
    res = run_chain(args.subject, args.fold, args.condition, lambda: load_pretrained(ckpt),
                    train_round, lambda m: m, recipe.per_class, args.seed, args.epochs)
    if args.epochs == 0:
        steps = [0] * len(ROUNDS)
    record = {
        "experiment": "exp21f_BP_ep", "recipe": "bp", "subject": args.subject,
        "fold": args.fold, "condition": args.condition, "epochs": args.epochs, "seed": args.seed,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recipe_settings": recipe.as_dict(), "artifacts": data.ARTIFACTS_DIR, "checkpoint": ckpt,
        "per_round": {str(r): {**res[r], "update_steps": steps[i]} for i, r in enumerate(ROUNDS)},
        "update_steps_per_round": steps,
        "mean_val_next": float(np.mean([res[r]["val_next"] for r in ROUNDS])),
        "mean_val_in": float(np.mean([res[r]["val_in"] for r in ROUNDS])),
        "mean_train": float(np.mean([res[r]["train"] for r in ROUNDS])),
        "seconds": time.time() - t0,
    }
    with open(out, "w") as h:
        json.dump(record, h, indent=1)
    print(f"DONE bp e{args.epochs} seed{args.seed} {args.subject}: mean val_next "
          f"{record['mean_val_next']:.2f}  val_in {record['mean_val_in']:.2f}  steps {steps}  "
          f"({record['seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
