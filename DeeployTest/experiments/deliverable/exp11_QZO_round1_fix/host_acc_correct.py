import sys, numpy as np
sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph
base = "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp11_QZO_round1_fix"
for tag, d in (("device-traj", base+"/qinfer_round1"), ("ref-traj", base+"/qinfer_ref_traj")):
    npz = np.load(d + "/inputs.npz"); X, Y = npz["input"].astype(np.float32), npz["label"].astype(np.int64)
    net = d + "/network.onnx"            # the OFFSET-STRIPPED graph the device actually runs
    pred = []
    for i in range(X.shape[0]):
        r = run_onnx_graph(net, {"input": X[i:i+1]})
        a = np.asarray(r[0] if isinstance(r,(list,tuple)) else r, np.float32).reshape(-1)
        pred.append(a.argmax())
    pred = np.array(pred)
    bal = float(np.mean([(pred==Y)[Y==c].mean() for c in np.unique(Y)]))
    print(f"{tag}: host-executor(network.onnx) balanced = {100*bal:.2f}%", flush=True)
