# SPDX-License-Identifier: MIT
"""Build the QUANTIZED inference fixture for the untiled accuracy eval (route (b)).

Takes the QZO fixture's base graph, integerizes it with the SAME builder the train graph used
(`build_int8_forward` — so scales/datapath are identical to training), then INJECTS the
device-dumped tensors (int8 codes / int32 bias / fp32 BN+fc from extract_qzo_weights.py)
directly into the graph initializers — no dequantize/requantize round trip. Generates the
per-window host reference logits with the device-faithful `run_onnx_graph` executor and writes
the fixture in the layout `speechnet_accuracy_eval_untiled.py` expects:
    out/network.onnx, out/inputs.npz{input,label}, out/outputs.npz{output}

Runs inside `agitated_hugle`. Without --dump-npz the ORIGINAL (pretrained) weights are kept —
that variant is the zero-shot fixture, used for the datapath smoke test.

  python3 build_qzo_infer_fixture.py --fixture-dir /app/Onnx4Deeploy/QZO_exp/exp9_full \
      [--dump-npz results/dumped_weights.npz] --out-dir <dir> [--windows N] [--verify-brevitas K]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, "/app/Onnx4Deeploy")
EVAL_CACHE = "/app/Onnx4Deeploy/QZO_exp/exp_calibration/data_cache_incr.npz"  # evX1/evY1 = whole batch 2


def integerize(fixture_dir: str) -> onnx.ModelProto:
    from onnx4deeploy.transform.qzo_weight_integerize import build_int8_forward
    base = onnx.load(str(Path(fixture_dir) / "network.onnx"))
    model, _scale_map = build_int8_forward(base)
    return model


def inject(model: onnx.ModelProto, dump_npz: str) -> int:
    from onnx import numpy_helper
    dump = np.load(dump_npz)
    initmap = {i.name: k for k, i in enumerate(model.graph.initializer)}
    n = 0
    for name in dump.files:
        assert name in initmap, f"dump tensor {name!r} not an initializer of the infer graph"
        k = initmap[name]
        old = numpy_helper.to_array(model.graph.initializer[k])
        new = np.asarray(dump[name])
        assert old.dtype == new.dtype, f"{name}: dtype {old.dtype} vs dump {new.dtype}"
        assert old.shape == tuple(new.shape) or old.size == new.size, \
            f"{name}: shape {old.shape} vs dump {new.shape}"
        t = numpy_helper.from_array(new.reshape(old.shape), name=name)
        model.graph.initializer[k].CopyFrom(t)
        n += 1
    return n


def reference_logits(net_path: str, X: np.ndarray, log=print) -> np.ndarray:
    from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph
    outs = []
    for i in range(X.shape[0]):
        r = run_onnx_graph(net_path, {"input": X[i:i + 1]})
        arr = np.asarray(r[0] if isinstance(r, (list, tuple)) else r, np.float32).reshape(-1)
        outs.append(arr)
        if (i + 1) % 20 == 0:
            log(f"  reference {i+1}/{X.shape[0]}")
    return np.stack(outs)


def brevitas_logits(X: np.ndarray) -> np.ndarray:
    """fc-float device-faithful Brevitas sim (pretrained weights, pooled@99.99) — the
    zero-shot cross-check target."""
    sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp_calibration")
    sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp3_lr1e-5_stability")
    sys.path.insert(0, "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1/pytorch_ref")
    import torch
    import run_study as S
    import stability_lib as L
    import run_fc_float_ref as RF
    r = S.load_results()
    S.SEED = 42
    _, qm = L.build_qmodel(3)
    S.freeze_config(qm, r, "pooled@99.99", None)
    params = S.build_params(qm)
    RF.to_float_fc(qm, params)
    with torch.no_grad():
        return qm(torch.from_numpy(X)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-dir", required=True)
    ap.add_argument("--dump-npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--windows", type=int, default=0, help="limit (0 = all 180)")
    ap.add_argument("--verify-brevitas", type=int, default=0,
                    help="cross-check first K windows against the fc-float Brevitas sim "
                         "(only meaningful WITHOUT --dump-npz, i.e. pretrained weights)")
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    model = integerize(a.fixture_dir)
    if a.dump_npz:
        n = inject(model, a.dump_npz)
        print(f"[fixture] injected {n} dumped tensors (device representation, no round trip)")
    net = out / "network.onnx"
    onnx.save(model, str(net))

    d = np.load(EVAL_CACHE)
    X, Y = d["evX1"].astype(np.float32), d["evY1"].astype(np.int64)
    if a.windows:
        X, Y = X[:a.windows], Y[:a.windows]
    print(f"[fixture] batch-2 eval set: {X.shape[0]} windows")

    ref = reference_logits(str(net), X)
    np.savez(out / "inputs.npz", input=X, label=Y)
    np.savez(out / "outputs.npz", output=ref)
    acc = float((ref.argmax(1) == Y).mean())
    # balanced accuracy (equal class counts in batch 2, but compute properly anyway)
    bal = float(np.mean([(ref.argmax(1) == Y)[Y == c].mean() for c in np.unique(Y)]))
    print(f"[fixture] host-reference (run_onnx_graph) accuracy on these windows: "
          f"overall={100*acc:.2f}%  balanced={100*bal:.2f}%")

    if a.verify_brevitas:
        K = min(a.verify_brevitas, X.shape[0])
        bl = brevitas_logits(X[:K])
        cos = float(np.mean(np.sum(ref[:K]*bl, 1) /
                            (np.linalg.norm(ref[:K], axis=1)*np.linalg.norm(bl, axis=1) + 1e-12)))
        agree = int((ref[:K].argmax(1) == bl.argmax(1)).sum())
        print(f"[verify] integerized-graph vs Brevitas fc-float sim on {K} windows: "
              f"logit cos={cos:.4f}  argmax agree={agree}/{K}")


if __name__ == "__main__":
    main()
