# Board_exp: does the host int8 state (exp21h / CleanUp, pooled@99.9) equal the fixture's 22 initial trainable inputs?
import sys, numpy as np, onnx, torch
sys.path.insert(0, "/app/SilentWear/SilentWear/PyTorch_for_On_Device/CleanUp")
from common import quantized
from common.quantized_zeroth_order import initial_state
_, model = quantized.build_quantized("S01", 3, "vocalized")
quantized.calibrate(model, "S01", 3, "vocalized", log=lambda *a: None, percentile=99.9)
params = quantized.collect_parameters(model, float_fc=True)
st = initial_state(params, "direct")
F = "/app/TrainDeeploy/DeeployTest/experiments/Board_exp/ZOfam/fixtures/speechnet_train_QZO_r1/"
names = [i.name for i in onnx.load(F + "network.onnx").graph.input][2:]
d = np.load(F + "inputs.npz")
print("host params:", [(p["name"], p["kind"], tuple(st[p["name"]].shape)) for p in params])
for wi, n in enumerate(names):
    v = d[f"arr_{wi + 2:04d}"]
    cands = [(p["name"], float(np.abs(st[p["name"]].detach().numpy().astype(np.float64).ravel() - v.astype(np.float64).ravel()).max()))
             for p in params if st[p["name"]].numel() == v.size]
    best = min(cands, key=lambda c: c[1]) if cands else None
    print(f"wi={wi:2d} {n:28s} {v.dtype} {v.shape} -> {best}")
# --- bias relation: rqsadd vs host int bias and the per-channel requant multiplier
from onnx import numpy_helper
g = onnx.load(F + "network.onnx").graph
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
for b in range(5):
    rq = d[f"arr_{15 + b + 2:04d}"].astype(np.int64)
    hb = st[f"blocks.{b}.conv.bias"].detach().numpy().astype(np.int64)
    mul = [v for k, v in inits.items() if f"blocks_{b}_conv_output_quant" in k and k.endswith("mul_pc")]
    m = np.asarray(mul[0]).astype(np.int64).ravel() if mul else None
    print(f"block {b}: rqsadd[:3]={rq[:3]} host_bias[:3]={hb[:3]} mul[:3]={None if m is None else m[:3]}",
          "exact rqsadd == hb*mul + 32768:" if m is not None else "", None if m is None else bool(np.all(rq == hb * m + 32768)),
          None if m is None else bool(np.all(rq == hb * m)))
