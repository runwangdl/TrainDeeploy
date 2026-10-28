# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch fold-3 head-only incremental baseline — the reference the on-device (GVSoC) simulation is
compared against. Identical recipe to exp7 (lr 0.01, n_accum 4, 40 ep, 30% data, incremental fc carry,
BN folded/frozen at pretrained stats), but restricted to S01 vocalized FOLD 3 only (the fold the
on-device sim runs), reporting per-batch balanced accuracy for b1..b5.

CLI:  python3 pytorch_fold3_baseline.py
"""
import os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model                # noqa: E402
from windowing import load_windows, stratified_draw       # noqa: E402
from ondevice_ft import balanced_accuracy, finetune_head  # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND, FOLD = "S01", "vocalized", 3
LR, N_ACCUM, EPOCHS = 0.01, 4, 40


def main():
    ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
            f"leave_one_session_out_fold_{FOLD}.pt")
    base = make_model(ckpt); m = make_model(ckpt)
    # NOTE: balanced_accuracy() already returns a percentage (0..100), do NOT ×100.
    print(f"# S01 vocalized fold 3 — head-only incremental (lr {LR}, n_accum {N_ACCUM}, {EPOCHS} ep, 30% data)")
    print(f"{'batch':>5} {'no_ft':>8} {'headonly_ft':>12}")
    rows = []
    for b in range(1, 6):
        Xe, ye = load_windows(DATA, SUBJECT, FOLD, b, COND, downsample_rest=True)
        no_ft = balanced_accuracy(base, Xe, ye)
        zs = balanced_accuracy(m, Xe, ye)                 # carried head-FT'd model + pretrained stats
        tag = "(zero-shot)" if b == 1 else "(FT on b1..b%d)" % (b - 1)
        print(f"{b:>5} {no_ft:>8.2f} {zs:>12.2f}   {tag}", flush=True)
        rows.append(dict(batch=b, no_ft=round(no_ft, 2), headonly_ft=round(zs, 2), note=tag))
        if b != 5:
            Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
            finetune_head(m, Xtr, ytr, lr=LR, n_accum=N_ACCUM, epochs=EPOCHS)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "pytorch_fold3_headonly.csv"), index=False)


if __name__ == "__main__":
    main()
