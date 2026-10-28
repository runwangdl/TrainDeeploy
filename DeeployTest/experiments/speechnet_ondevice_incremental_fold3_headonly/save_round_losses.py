# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Persist the per-round fine-tuning loss traces of the on-device head-only incremental flow, so the loss
data survives beyond the ephemeral GVSoC runner output.

For each round it saves:
  - the ORT reference per-step loss (from the fixture's outputs.npz['loss'], length = 40 ep x 54 = 2160),
  - the per-epoch mean loss (2160 -> 40),
  - if a GVSoC runner log is given, the on-device computed losses parsed from '[loss k] computed=...'
    and their max |computed - ref| (bit-exactness check).

CLI:  python3 save_round_losses.py --round 2 \
        --fixture /app/.../Training/SpeechNet/speechnet_train_b2_fold3 \
        [--gvsoc-log /path/to/round2_gvsoc_train.log]
Outputs: logs/round<N>_losses.csv (+ logs/round<N>_gvsoc_train.log copy if provided).
"""
import argparse, os, re, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "logs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--gvsoc-log", default=None)
    ap.add_argument("--n-accum", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(LOGDIR, exist_ok=True)
    import pandas as pd

    ref = np.load(os.path.join(a.fixture, "outputs.npz"))["loss"].astype(np.float64)
    n = len(ref)

    dev = None
    if a.gvsoc_log and os.path.exists(a.gvsoc_log):
        dst = os.path.join(LOGDIR, f"round{a.round}_gvsoc_train.log")
        if os.path.abspath(a.gvsoc_log) != os.path.abspath(dst):
            shutil.copy(a.gvsoc_log, dst)
        pat = re.compile(r"\[loss\s+(\d+)\]\s+computed=([-\d.eE]+)\s+ref=([-\d.eE]+)")
        got = {}
        for line in open(a.gvsoc_log):
            m = pat.search(line)
            if m:
                got[int(m.group(1))] = float(m.group(2))
        if got:
            dev = np.array([got.get(i, np.nan) for i in range(n)])

    df = pd.DataFrame({"step": np.arange(n), "ort_ref_loss": np.round(ref, 6)})
    if dev is not None:
        df["gvsoc_loss"] = np.round(dev, 6)
        df["abs_diff"] = np.round(np.abs(dev - ref), 6)
    df.to_csv(os.path.join(LOGDIR, f"round{a.round}_losses.csv"), index=False)

    # per-epoch mean (assumes 54 windows/epoch -> n/54 epochs)
    per_ep = 54
    n_ep = n // per_ep
    ep_mean = ref[: n_ep * per_ep].reshape(n_ep, per_ep).mean(axis=1)
    pd.DataFrame({"epoch": np.arange(1, n_ep + 1), "mean_ort_loss": np.round(ep_mean, 6)}).to_csv(
        os.path.join(LOGDIR, f"round{a.round}_epoch_mean_loss.csv"), index=False)

    print(f"round {a.round}: {n} steps, first loss {ref[0]:.4f} -> last {ref[-1]:.4f}, "
          f"epoch-mean {ep_mean[0]:.4f} -> {ep_mean[-1]:.4f}")
    if dev is not None:
        print(f"  GVSoC bit-exactness: max|computed-ref| = {np.nanmax(np.abs(dev - ref)):.2e} "
              f"over {np.sum(~np.isnan(dev))} logged steps")


if __name__ == "__main__":
    main()
