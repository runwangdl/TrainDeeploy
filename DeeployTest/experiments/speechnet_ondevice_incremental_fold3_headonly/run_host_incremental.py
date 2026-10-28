# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Host incremental chain that PREDICTS the on-device (GVSoC) head-only b1->b5 result exactly.

Rationale: on-device head-only training is bit-exact to the Onnx4Deeploy ORT reference (verified:
loss diff 0, device fc == ORT fc < 1e-4), and on-device inference is bit-exact to ORT. So the on-device
per-batch accuracy EQUALS the host accuracy computed with the SAME data draw as the export. This driver
runs that matched-draw chain on the host in minutes (the GVSoC run only needs to CONFIRM bit-exactness).

Per round r=1..4: export a head-only training fixture for batch r (Onnx4Deeploy, matched stratified
draw, fc initialised from round r-1's trained fc) -> read the ORT-trained fc from outputs.npz -> build
a carry checkpoint (frozen pretrained conv/BN + trained fc) -> evaluate it on batch r+1 (bit-exact
inference). Round 1 initialises from the pretrained fold-3 checkpoint. b1 is zero-shot.

S01 vocalized fold 3. lr 0.01, n_accum 4, 40 ep, 54 windows (30%), last_layer (BN folded).
CLI:  python3 run_host_incremental.py
"""
import os, sys, subprocess
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model            # noqa: E402
from windowing import load_windows                    # noqa: E402
from ondevice_ft import balanced_accuracy             # noqa: E402

ONNX = "/app/Onnx4Deeploy"
SN = "/app/ETH/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet"
DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
PRETRAINED = ("/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/"
              "speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt")


def eval_on_batch(ckpt_path, b):
    m = make_model(ckpt_path)
    X, y = load_windows(DATA, "S01", 3, b, "vocalized", downsample_rest=True)
    return balanced_accuracy(m, X, y)


def export_train(batch, carry_ckpt, outdir):
    subprocess.run(
        ["python3", "Onnx4Deeploy.py", "-model", "SpeechNet", "-mode", "train", "-o", outdir,
         "--dataset", "silentwear", "--data-path", DATA, "--pretrained-weights", carry_ckpt,
         "--subject", "S01", "--session", "3", "--batch", str(batch), "--condition", "vocalized",
         "--data-size", "54", "--n-accum", "4", "--n-epochs", "40", "--lr", "0.01",
         "--training-strategy", "last_layer", "--stratified"],
        cwd=ONNX, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.load(os.path.join(outdir, "outputs.npz"))


def main():
    base = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    base_state = base.get("model_state_dict", base)

    rows = [dict(batch=1, note="zero-shot", ondevice_pred=round(eval_on_batch(PRETRAINED, 1), 2))]
    carry = PRETRAINED
    for r in range(1, 5):
        outdir = f"{SN}/speechnet_train_b{r}_fold3"
        out = export_train(r, carry, outdir)
        new_state = {k: (v.clone() if torch.is_tensor(v) else torch.as_tensor(v)) for k, v in base_state.items()}
        new_state["fc.weight"] = torch.from_numpy(out["fc_weight"]).float()
        new_state["fc.bias"] = torch.from_numpy(out["fc_bias"]).float()
        carry = f"/tmp/carry_fold3_b{r}.pt"
        torch.save({"model_state_dict": new_state}, carry)
        acc = eval_on_batch(carry, r + 1)
        rows.append(dict(batch=r + 1, note=f"FT on b1..b{r}", ondevice_pred=round(acc, 2)))
        print(f"round {r}: FT on b{r} -> eval b{r+1} = {acc:.2f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "ondevice_predicted_fold3.csv"), index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
