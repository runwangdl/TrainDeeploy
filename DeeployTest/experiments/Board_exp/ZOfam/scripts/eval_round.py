# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""Board_exp / ZO family: accuracy of one round's BOARD weights, with exp21h's own host model and metrics.

Run in agitated_hugle (brevitas):   python3 eval_round.py --method <zo|eco|qzo> --arm <arm> --round <r>

The board dumps the 22 trainable tensors in the fixture's graph-input order ([WDUMP s=<last> wi=<i>], wi = input
2+i; the same order carry_weights.py uses). They are loaded into exp21h's Method (same checkpoint, same pooled@99.9
calibration, same float fc) and scored with exp21h round_data / balanced_accuracy:
  val_next = all of batch r+1, val_in = held-out 70 % of batch r, train = the 54 windows.
Mapping (checked on the round-1 fixture, scripts/check_state_map.py): BN affine and fc (fp32) and conv weights (int8)
one-to-one by name; conv bias: device bias_rqsadd = host int bias * requant mul (per channel) + 32768 (the baked
rounding offset) -> host bias = (rqsadd - 32768) / mul, asserted exact. For zo every tensor is fp32 and named as
the torch state dict.
Host comparison: exp21h results/<method>/seed42/e140/S01.json (same windows, order, z and SR streams).
Writes results/<arm>/round<r>.json. -- QW Board_exp
"""
import argparse, json, os, re, sys

import numpy as np
import onnx
import torch
from onnx import numpy_helper

E21H = "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21h_ZO_QZO_ep"
sys.path.insert(0, E21H)
import run_job as h                                         # noqa: E402  (exp21h; CleanUp on its path)
from common.device_faithful import balanced_accuracy       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ZF = os.path.dirname(HERE)


def wdump_last(path):
    pat = re.compile(r"\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)")
    d = {}
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if m:
            d[(int(m.group(1)), int(m.group(2)))] = np.array([int(x, 16) for x in m.group(4).split()], "<u4")
    last = max(s for s, _ in d)
    return last, {wi: v for (s, wi), v in d.items() if s == last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["zo", "eco", "qzo"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    r = a.round
    kind = "ZO" if a.method == "zo" else "QZO"
    fx = os.path.join(ZF, "fixtures", f"speechnet_train_{kind}_r{r}")
    g = onnx.load(os.path.join(fx, "network.onnx")).graph
    ins = list(g.input)[2:]
    last, dev = wdump_last(os.path.join(ZF, "..", "runs", f"{a.arm}_r{r}", "uart.log"))
    assert sorted(dev) == list(range(22)), sorted(dev)
    ELEM = {1: np.float32, 3: np.int8, 6: np.int32}
    vals = {}
    for wi, i in enumerate(ins):
        dt = ELEM[i.type.tensor_type.elem_type]
        shape = tuple(x.dim_value for x in i.type.tensor_type.shape.dim)
        n = int(np.prod(shape))
        v = dev[wi].view(np.uint8).view(np.int8)[:n] if dt is np.int8 else dev[wi].view(dt)[:n]
        vals[i.name] = v.reshape(shape).copy()

    m = h.Method(a.method, "S01", 3, "vocalized", 42)
    if a.method == "zo":
        sd = m.model.state_dict()
        for k, v in vals.items():
            t = k.replace("_", ".")                      # ONNX blocks_0_0_weight -> torch blocks.0.0.weight
            assert t in sd and tuple(sd[t].shape) == v.shape, k
            sd[t] = torch.from_numpy(v.astype(np.float32))
        m.model.load_state_dict(sd)
    else:
        inits = {x.name: numpy_helper.to_array(x) for x in g.initializer}
        st = {k: v.clone() for k, v in m.state.items()}
        for k, v in vals.items():
            if k.endswith("weight_int8"):
                hk = k.replace("_int8", "")
                st[hk] = torch.from_numpy(v.astype(np.float32)).to(st[hk].dtype).reshape(st[hk].shape)
            elif k.endswith("bias_rqsadd"):
                b = int(k.split(".")[1])
                mul = [x for n_, x in inits.items() if f"blocks_{b}_conv_output_quant" in n_ and n_.endswith("mul_pc")]
                mul = np.asarray(mul[0]).astype(np.int64).ravel()
                # The device keeps the bias in RequantShift-add units (1/div of an output LSB, = host bias grid / mul)
                # and updates it THERE, so trained values fall between host grid points (the first version asserted
                # divisibility -- true only at init, where every bias is 0). Device semantics: bias = (rqsadd - 32768)
                # / mul host-grid units, kept fractional (install multiplies by the host bias scale s_in*s_w).
                off = v.astype(np.int64) - 32768
                hk = f"blocks.{b}.conv.bias"
                st[hk] = torch.from_numpy((off / mul).astype(np.float32)).to(st[hk].dtype).reshape(st[hk].shape)
            else:
                hk = k.replace("_fp32", "")
                st[hk] = torch.from_numpy(v.astype(np.float32)).to(st[hk].dtype).reshape(st[hk].shape)
        m.state = st
        m.install()

    res = h.metrics(m.model, h.round_data("S01", 3, "vocalized", r, 42, m.base.per_class))   # exp21h's own metrics
    hj = os.path.join(E21H, "results", a.method, "seed42", "e140", "S01.json")
    host = json.load(open(hj))["per_round"][str(r)] if os.path.exists(hj) else {}
    print(f"{a.arm} round {r}: [WDUMP] s={last}")
    for k, v in res.items():
        hv = host.get(k)
        print(f"  {k:9s} board {v:6.2f}   host seed42 e140 {hv if hv is None else round(hv, 2)}")
    out = os.path.join(ZF, "results", a.arm)
    os.makedirs(out, exist_ok=True)
    json.dump({"arm": a.arm, "method": a.method, "round": r, "wdump_step": last, "board": res,
               "host_seed42_e140": host}, open(os.path.join(out, f"round{r}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
