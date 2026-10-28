# SPDX-License-Identifier: MIT
"""Divergence-bisection probe builder for the quantized inference graph (exp9 route (b)).

Truncates qinfer_smoke/network.onnx at a named node's output (keep it fp32 — layout-safe),
prunes unreachable nodes, generates the run_onnx_graph reference for window 0, and writes a
1-sample untiled fixture to Tests/Models/<tag>. Run in agitated_hugle from DeeployTest/.

  python3 make_probe.py --node "QCDQ_/blocks_2_conv_output_dequant/Mul_Dequant" --tag qinfer_probe2
"""
import argparse
import os
import shutil
import sys

import numpy as np
import onnx
from onnx import helper

sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph  # noqa: E402

SRC = "experiments/deliverable/exp9_QZO_round1/qinfer_smoke/network.onnx"
INP = "experiments/deliverable/exp9_QZO_round1/qinfer_smoke/inputs.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True, help="node NAME whose output[0] becomes the probe")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--window", type=int, default=0)
    a = ap.parse_args()

    m = onnx.load(SRC)
    g = m.graph
    probe_out = next(n.output[0] for n in g.node if n.name == a.node)
    need, keep = {probe_out}, []
    for n in reversed(list(g.node)):
        if set(n.output) & need:
            keep.append(n)
            need.update(n.input)
    keep = list(reversed(keep))
    del g.node[:]
    g.node.extend(keep)
    x = np.load(INP)["input"][a.window:a.window + 1]
    del g.output[:]
    g.output.append(helper.make_tensor_value_info(probe_out, onnx.TensorProto.FLOAT, None))
    tmp = f"/tmp/{a.tag}.onnx"
    onnx.save(m, tmp)
    r = run_onnx_graph(tmp, {"input": x})
    ref = np.asarray(r[0] if isinstance(r, (list, tuple)) else r, np.float32)
    del g.output[:]
    g.output.append(helper.make_tensor_value_info(probe_out, onnx.TensorProto.FLOAT,
                                                  list(ref.shape)))
    onnx.save(m, tmp)
    T = f"Tests/Models/{a.tag}"
    os.makedirs(T, exist_ok=True)
    shutil.copy(tmp, f"{T}/network.onnx")
    np.savez(f"{T}/inputs.npz", input=x)
    np.savez(f"{T}/outputs.npz", output=ref)
    print(f"ready: {T}  probe={probe_out}  shape={ref.shape}  nodes={len(keep)}  "
          f"mean|x|={float(np.abs(ref).mean()):.4f}")


if __name__ == "__main__":
    main()
