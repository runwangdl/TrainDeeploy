# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""PyTorch matched reference for the on-device full-training-frozen-BN chain (S01 vocalized fold 3).

For each round r, extract the EXACT 54 FT windows from the Onnx4Deeploy train fixture
(speechnet_train_fullfrozen_b{r}_fold3/inputs.npz) — same windows the device trains on — run the S2 recipe
in PyTorch (model.eval() = frozen BN, full model, SGD lr3e-4 no-momentum, SUM accumulation n_accum=4,
40 epochs), carry weights forward, and evaluate batch r+1. This removes the data-draw confound so the
on-device vs PyTorch accuracy comparison is apples-to-apples.
CLI:  python3 run_pytorch_fullfrozen_ondevicedata_chain.py
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
SILENT = "/app/SilentWear/SilentWear/PyTorch_for_On_Device"
sys.path.insert(0, SILENT)
sys.path.insert(0, os.path.join(SILENT, "exp13_progressive_unfreeze"))
from adabn_full_training import make_model            # noqa: E402  (loads official ckpt, BN model)
from windowing import load_windows                    # noqa: E402
from ondevice_ft import balanced_accuracy             # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
PRETRAINED = ("/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/"
              "speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt")
SN = "/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet"
HERE = os.path.dirname(os.path.abspath(__file__))
LR, N_ACCUM, EPOCHS = 3e-4, 4, 40
# per round: (batch trained on, train fixture with its 54 windows, eval batch)
ROUNDS = [(r, f"{SN}/speechnet_train_fullfrozen_b{r}_fold3", r + 1) for r in range(1, 5)]


def extract_windows(fixture, n=54):
    d = np.load(os.path.join(fixture, "inputs.npz"))
    Xs = [d["arr_0000"]]; ys = [np.atleast_1d(d["arr_0001"])]
    for i in range(1, n):
        Xs.append(d[f"mb{i}_arr_0000"]); ys.append(np.atleast_1d(d[f"mb{i}_arr_0001"]))
    return np.concatenate(Xs, 0).astype(np.float32), np.concatenate(ys).astype(np.int64)


def train_full_frozen(model, X, y):
    """S2: full model, model.eval() frozen BN, SGD lr no-momentum, SUM accumulation, EPOCHS, fixed order."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    N = len(y)
    for _ in range(EPOCHS):
        opt.zero_grad(); c = 0
        for j in range(N):                                  # fixed order (matches device data order)
            crit(model(Xt[j:j + 1]), yt[j:j + 1]).backward(); c += 1
            if c % N_ACCUM == 0:
                opt.step(); opt.zero_grad()
        if c % N_ACCUM:
            opt.step(); opt.zero_grad()
    model.eval()
    return model


def main():
    import pandas as pd
    m = make_model(PRETRAINED)
    Xb1, yb1 = load_windows(DATA, "S01", 3, 1, "vocalized", downsample_rest=True)
    rows = [dict(batch=1, note="zero-shot", pytorch_fullfrozen=round(balanced_accuracy(m, Xb1, yb1), 2))]
    print(f"{'batch':>5} {'note':>14} {'PyTorch full-frozen (matched)':>30}")
    print(f"{1:>5} {'zero-shot':>14} {rows[0]['pytorch_fullfrozen']:>30.2f}")
    for r, fixture, eb in ROUNDS:
        if not os.path.exists(os.path.join(fixture, "inputs.npz")):
            print(f"  [skip round {r}] fixture not found yet: {fixture}"); continue
        X, y = extract_windows(fixture)
        train_full_frozen(m, X, y)
        Xe, ye = load_windows(DATA, "S01", 3, eb, "vocalized", downsample_rest=True)
        acc = round(balanced_accuracy(m, Xe, ye), 2)
        rows.append(dict(batch=eb, note=f"FT through b{r}", pytorch_fullfrozen=acc))
        print(f"{eb:>5} {('FT through b'+str(r)):>14} {acc:>30.2f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "pytorch_fullfrozen_ondevicedata.csv"), index=False)
    print("\nsaved results/pytorch_fullfrozen_ondevicedata.csv", flush=True)


if __name__ == "__main__":
    main()
