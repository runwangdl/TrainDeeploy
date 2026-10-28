# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""Board_exp / BP: board weights of one round -> carry checkpoint + accuracy, same metric as the host selection.

    /opt/anaconda3/bin/python3.11 eval_round.py --round 1

1. Parse the board's last-step [WDUMP] (runs/bp_r<r>/uart.log), 22 trainable tensors, mapped by weight-buffer index
   with extract_device_weights.NAME_MAP (exp17), onto the round's start checkpoint (round 1: paper checkpoint,
   round r > 1: carry_r<r-1>.pt) -> carry/carry_r<r>.pt (the start of round r+1, see export_bp_round.sh).
2. Check: board weights vs the exporter's PyTorch reference after the same steps (fixture outputs.npz) -> confirms the
   index mapping and that the board trained what the host trains (fp32: last-bit differences only).
3. Accuracy with CleanUp's own code (epoch_sweep.run_chain's metrics, balanced accuracy, frozen BN):
   val_next = all of batch r+1, val_in = held-out 70 % of batch r, train = the 54 windows.
   Compared with the host chain exp21j const lr 3e-5, 40 epochs, seed 42, S01 (same windows, order, checkpoint).
Writes results/round<r>.json. CleanUp/common is imported, never modified. -- QW Board_exp
"""
import argparse, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BP = os.path.dirname(HERE)
TD = os.path.abspath(os.path.join(BP, "..", "..", "..", ".."))
CLEANUP = "/Users/qiwenwu/ETH/SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp"
sys.path.insert(0, CLEANUP)
sys.path.insert(0, os.path.join(TD, "DeeployTest", "experiments", "exp1", "ondevice_sim_S01_fold3"))
from common import data                                   # noqa: E402
from common.device_faithful import balanced_accuracy      # noqa: E402
from common.speechnet import load_pretrained              # noqa: E402
from extract_device_weights import NAME_MAP, parse_wdump_last_step  # noqa: E402

PAPER = os.path.join(TD, "SpeechNet_Test", "pretrained", "inter_session_ft", "S01", "vocalized", "speechnet",
                     "w1400ms", "model_1", "leave_one_session_out_fold_3.pt")
HOST = os.path.join(TD, "DeeployTest", "experiments", "deliverable", "exp21_remaining_exp", "exp21j_BP_ep_lr",
                    "results", "const_lr3e-05", "seed42", "e040", "S01.json")
SUBJ, FOLD, COND, PER_CLASS, SEED = "S01", 3, "vocalized", 6, 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--run", default=None, help="run dir (default runs/bp_r<round>)")
    a = ap.parse_args()
    r = a.round
    run = a.run or os.path.join(BP, "..", "runs", f"bp_r{r}")
    start = PAPER if r == 1 else os.path.join(BP, "carry", f"carry_r{r - 1}.pt")

    # 1. board weights -> carry checkpoint
    last, dev = parse_wdump_last_step(os.path.join(run, "uart.log"))
    assert set(dev) == set(NAME_MAP), f"dumped tensors {sorted(dev)}"
    base = torch.load(start, map_location="cpu", weights_only=False)
    sd = {k: v.clone() for k, v in base.get("model_state_dict", base).items()}
    ref = np.load(os.path.join(BP, "fixtures", f"speechnet_train_BP_r{r}", "outputs.npz"))
    worst = 0.0
    for wi, oname in NAME_MAP.items():
        t = oname.replace("_", ".")
        w = torch.from_numpy(dev[wi].reshape(tuple(sd[t].shape))).float()
        rd = float(np.abs(dev[wi] - ref[oname].ravel()).max())     # board vs exporter's PyTorch reference
        worst = max(worst, rd)
        sd[t] = w
    out_ckpt = os.path.join(BP, "carry", f"carry_r{r}.pt")
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save({"model_state_dict": sd}, out_ckpt)
    print(f"round {r}: [WDUMP] s={last}, 22 tensors -> {out_ckpt}; max |board - PyTorch ref| = {worst:.3e}")

    # 2. accuracy, CleanUp metrics (= epoch_sweep.run_chain)
    model = load_pretrained(out_ckpt)
    X, y = data.evaluation_windows(SUBJ, FOLD, r, COND)
    Xt, yt = data.labelled_subset(X, y, PER_CLASS, seed=SEED, draw="device")
    taken = {w.tobytes() for w in Xt.reshape(len(yt), -1)}
    keep = [i for i, w in enumerate(X.reshape(len(y), -1)) if w.tobytes() not in taken]
    Xn, yn = data.evaluation_windows(SUBJ, FOLD, r + 1, COND)
    res = {"val_next": balanced_accuracy(model, Xn, yn), "val_in": balanced_accuracy(model, X[keep], y[keep]),
           "train": balanced_accuracy(model, Xt, yt)}
    host = json.load(open(HOST))["per_round"][str(r)]
    for k in res:
        print(f"  {k:8s} board {res[k]:6.2f}   host chain {host[k]:6.2f}")
    os.makedirs(os.path.join(BP, "results"), exist_ok=True)
    json.dump({"round": r, "start_ckpt": start, "carry": out_ckpt, "wdump_step": last,
               "max_abs_board_minus_ref": worst, "board": res, "host_seed42": {k: host[k] for k in res}},
              open(os.path.join(BP, "results", f"round{r}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
