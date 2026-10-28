# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Process one completed on-device GVSoC training round of the head-only incremental chain:
  1. parse the device fc from the runner log's [WDUMP] lines (FPU-free hex reconstruct),
  2. validate device fc == ORT reference (outputs.npz) within fp32,
  3. build the carry checkpoint = frozen pretrained conv/BN + this round's device fc,
  4. evaluate the carry model on the next batch (bit-exact host inference; inference is bit-exact to
     GVSoC per speechnet_infer_original = 70.56% match) -> the on-device FT accuracy for that batch,
  5. save device_fc_r<N>.npy and the carry checkpoint for the next round.

CLI: python3 process_round.py --round 2 \
       --fixture /app/.../speechnet_train_b2_fold3 \
       --gvsoc-log logs/round2_gvsoc_train.log \
       --eval-batch 3 --out-carry /tmp/carry_b2_device_fold3.pt
"""
import argparse, os, re, struct, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model            # noqa: E402
from windowing import load_windows                    # noqa: E402
from ondevice_ft import balanced_accuracy             # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
PRETRAINED = ("/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/"
              "speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt")


def parse_wdump(logpath):
    """Return {wi: float32 array} from the last-step [WDUMP s= wi= n=] hex lines."""
    pat = re.compile(r"\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)")
    dumps = {}
    for line in open(logpath):
        m = pat.search(line)
        if not m:
            continue
        s, wi, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
        words = m.group(4).split()
        arr = np.array([struct.unpack("<f", struct.pack("<I", int(w, 16)))[0] for w in words],
                       dtype=np.float32)
        dumps[(s, wi)] = arr
    laststep = max(s for s, _ in dumps)
    return {wi: dumps[(s, wi)] for (s, wi) in dumps if s == laststep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--gvsoc-log", required=True)
    ap.add_argument("--eval-batch", type=int, required=True)
    ap.add_argument("--out-carry", required=True)
    a = ap.parse_args()
    logdir = os.path.join(HERE, "logs"); os.makedirs(logdir, exist_ok=True)

    dev = parse_wdump(a.gvsoc_log)
    fc_w = dev[0].reshape(9, 32); fc_b = dev[1].reshape(9)
    ref = np.load(os.path.join(a.fixture, "outputs.npz"))
    dw = float(np.max(np.abs(fc_w - ref["fc_weight"]))); db = float(np.max(np.abs(fc_b - ref["fc_bias"])))
    valid = "VALID (device==ORT<1e-4)" if dw < 1e-4 and db < 1e-4 else "WARNING device diverges"
    print(f"round {a.round}: device fc extracted; max|device-ORT| w={dw:.2e} b={db:.2e}  {valid}", flush=True)
    np.save(os.path.join(logdir, f"device_fc_r{a.round}_weight.npy"), fc_w)
    np.save(os.path.join(logdir, f"device_fc_r{a.round}_bias.npy"), fc_b)

    base = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    st = base.get("model_state_dict", base)
    ns = {k: (v.clone() if torch.is_tensor(v) else torch.as_tensor(v)) for k, v in st.items()}
    ns["fc.weight"] = torch.from_numpy(fc_w).float(); ns["fc.bias"] = torch.from_numpy(fc_b).float()
    torch.save({"model_state_dict": ns}, a.out_carry)

    m = make_model(a.out_carry)
    Xe, ye = load_windows(DATA, "S01", 3, a.eval_batch, "vocalized", downsample_rest=True)
    acc = balanced_accuracy(m, Xe, ye)
    print(f"round {a.round}: FT through b{a.round} -> eval b{a.eval_batch} = {acc:.2f}  "
          f"(on-device, bit-exact inference)", flush=True)
    with open(os.path.join(logdir, f"round{a.round}_accuracy.txt"), "w") as f:
        f.write(f"round {a.round}: eval b{a.eval_batch} = {acc:.2f}\n")


if __name__ == "__main__":
    main()
