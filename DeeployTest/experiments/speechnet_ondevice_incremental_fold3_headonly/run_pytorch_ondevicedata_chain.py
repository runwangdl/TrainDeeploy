# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Show the on-device (GVSoC) b1->b5 chain is bit-exact to the PyTorch simulation, by running the PyTorch
head-only incremental chain on the SAME 54-window draws Onnx4Deeploy prepared for each on-device round
(extracted from each fixture's inputs.npz), in fixture order, carrying fc forward. Since on-device
training + inference are bit-exact to ORT, feeding PyTorch the identical Onnx4Deeploy data must reproduce
the on-device per-batch accuracies exactly (removing the stratified-draw confound that separates the
seed-42 PyTorch baseline from on-device).

S01 vocalized fold 3, head-only lr 0.01 / n_accum 4 / 40 ep, incremental (fc carried).
CLI:  python3 run_pytorch_ondevicedata_chain.py
"""
import os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model                # noqa: E402
from windowing import load_windows                        # noqa: E402
from ondevice_ft import balanced_accuracy, finetune_head  # noqa: E402

SN = "/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet"
DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
PRETRAINED = ("/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/"
              "speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt")

# per round r: (fixture with the 54 Onnx4Deeploy FT windows for batch r, eval batch, on-device GVSoC acc)
ROUNDS = [
    (1, f"{SN}/speechnet_train_head_ep40_isft_fw", 2, 89.44),
    (2, f"{SN}/speechnet_train_b2_fold3", 3, 80.00),
    (3, f"{SN}/speechnet_train_b3_fold3", 4, 83.89),
    (4, f"{SN}/speechnet_train_b4_fold3", 5, 82.22),
]


def extract_windows(fixture):
    """The 54 unique FT windows Onnx4Deeploy stored, in fixture order (mb0..mb53) = on-device cycle order."""
    d = np.load(os.path.join(fixture, "inputs.npz"))
    Xs = [d["arr_0000"]]; ys = [np.atleast_1d(d["arr_0001"])]
    for i in range(1, 54):
        Xs.append(d[f"mb{i}_arr_0000"]); ys.append(np.atleast_1d(d[f"mb{i}_arr_0001"]))
    return np.concatenate(Xs, 0).astype(np.float32), np.concatenate(ys).astype(np.int64)


def main():
    import pandas as pd
    m = make_model(PRETRAINED)
    Xb1, yb1 = load_windows(DATA, "S01", 3, 1, "vocalized", downsample_rest=True)
    rows = [dict(batch=1, note="zero-shot", pytorch_on_ondevice_data=round(balanced_accuracy(m, Xb1, yb1), 2),
                 ondevice_gvsoc=80.56)]
    print(f"{'batch':>5} {'PyTorch(on-device data)':>24} {'on-device GVSoC':>16} {'match':>6}")
    print(f"{1:>5} {rows[0]['pytorch_on_ondevice_data']:>24.2f} {80.56:>16.2f} {'=':>6}   (zero-shot)")
    for r, fixture, eb, dev in ROUNDS:
        X, y = extract_windows(fixture)                    # exact Onnx4Deeploy 54 windows for this round
        finetune_head(m, X, y, lr=0.01, n_accum=4, epochs=40)   # head-only, carry fc forward
        Xe, ye = load_windows(DATA, "S01", 3, eb, "vocalized", downsample_rest=True)
        acc = balanced_accuracy(m, Xe, ye)
        ok = "=" if abs(acc - dev) < 0.01 else f"{acc-dev:+.2f}"
        rows.append(dict(batch=eb, note=f"FT through b{r}",
                         pytorch_on_ondevice_data=round(acc, 2), ondevice_gvsoc=dev))
        print(f"{eb:>5} {acc:>24.2f} {dev:>16.2f} {ok:>6}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "pytorch_ondevicedata_vs_gvsoc.csv"), index=False)
    n_match = int((abs(df.pytorch_on_ondevice_data - df.ondevice_gvsoc) < 0.01).sum())
    print(f"\n{n_match}/{len(df)} batches match exactly (PyTorch on Onnx4Deeploy data == on-device GVSoC)",
          flush=True)


if __name__ == "__main__":
    main()
