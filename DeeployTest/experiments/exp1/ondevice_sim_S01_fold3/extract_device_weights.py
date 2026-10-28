# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""Extract the 22 full-model trainable tensors dumped by the device ([WDUMP], last step) and build the
carry checkpoint for the next round of the on-device full-training-frozen-BN chain.

Mapping is by WEIGHT-BUFFER ORDER (the wi index = the grad.accumulation.buffer input order in
network_train.onnx), NOT value-proximity — because the MaxPool argmax drift makes device weights diverge
from any reference, which would break value-matching. Order = drift-proof.

Carry = base checkpoint with the 22 trainable tensors overwritten by the device-dumped values; BN running
stats and everything else stay frozen (unchanged).

CLI: python3 extract_device_weights.py --gvsoc-log logs/round1_gvsoc_train.log \
       --base-ckpt <official-or-prev-carry.pt> --out-carry /tmp/carry_fullfrozen_b1_fold3.pt
"""
import argparse, os, re, struct
import numpy as np
import torch

# wi index -> ONNX trainable-param name (grad.accumulation.buffer input order in network_train.onnx)
NAME_MAP = {
    0: "blocks_0_0_weight", 1: "blocks_0_0_bias", 2: "blocks_0_1_weight", 3: "blocks_0_1_bias",
    4: "blocks_1_0_weight", 5: "blocks_1_0_bias", 6: "blocks_1_1_weight", 7: "blocks_1_1_bias",
    8: "blocks_2_0_weight", 9: "blocks_2_0_bias", 10: "blocks_2_1_weight", 11: "blocks_2_1_bias",
    12: "blocks_3_0_weight", 13: "blocks_3_0_bias", 14: "blocks_3_1_weight", 15: "blocks_3_1_bias",
    16: "blocks_4_0_weight", 17: "blocks_4_0_bias", 18: "blocks_4_1_weight", 19: "blocks_4_1_bias",
    20: "fc_weight", 21: "fc_bias",
}


def parse_wdump_last_step(logpath):
    pat = re.compile(r"\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)")
    dumps = {}
    for line in open(logpath):
        m = pat.search(line)
        if not m:
            continue
        s, wi = int(m.group(1)), int(m.group(2))
        arr = np.array([struct.unpack("<f", struct.pack("<I", int(w, 16)))[0]
                        for w in m.group(4).split()], dtype=np.float32)
        dumps[(s, wi)] = arr
    if not dumps:
        raise RuntimeError(f"no [WDUMP] lines found in {logpath}")
    last = max(s for s, _ in dumps)
    return last, {wi: dumps[(s, wi)] for (s, wi) in dumps if s == last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvsoc-log", required=True)
    ap.add_argument("--base-ckpt", required=True, help="round-1: official ckpt; round>1: previous carry")
    ap.add_argument("--out-carry", required=True)
    a = ap.parse_args()

    last, dev = parse_wdump_last_step(a.gvsoc_log)
    print(f"parsed [WDUMP] last step s={last}: {len(dev)} tensors (expected 22)")

    base = torch.load(a.base_ckpt, map_location="cpu", weights_only=False)
    sd = base.get("model_state_dict", base)
    carry = {k: (v.clone() if torch.is_tensor(v) else torch.as_tensor(v)) for k, v in sd.items()}

    assert set(dev.keys()) == set(NAME_MAP.keys()), f"wi set {sorted(dev)} != 0..21"
    maxmove = 0.0
    for wi, oname in NAME_MAP.items():
        tname = oname.replace("_", ".")                    # ONNX underscores -> torch dots
        assert tname in carry, f"{tname} not in base ckpt"
        want = carry[tname].numel()
        got = dev[wi].size
        assert got == want, f"wi={wi} {oname}: dumped {got} floats != target {want}"
        newv = torch.from_numpy(dev[wi].reshape(tuple(carry[tname].shape))).float()
        mv = float((newv - carry[tname]).abs().max())
        maxmove = max(maxmove, mv)
        print(f"  wi={wi:2d} {oname:20s} shape={tuple(carry[tname].shape)}  max|dev-base|={mv:.3e}")
        carry[tname] = newv
    print(f"max weight movement over all trainable tensors: {maxmove:.3e}")

    torch.save({"model_state_dict": carry}, a.out_carry)
    print(f"saved carry -> {a.out_carry}")


if __name__ == "__main__":
    main()
