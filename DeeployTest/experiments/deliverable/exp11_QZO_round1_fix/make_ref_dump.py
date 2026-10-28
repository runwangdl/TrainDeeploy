import numpy as np
o = np.load("/app/Onnx4Deeploy/QZO_exp/exp11_round1_fix/outputs.npz")
out = {}
for k in o.files:
    if k.startswith("updated_"):
        name = k[len("updated_"):]
        v = np.asarray(o[k])
        if name.endswith("_int8"):
            v = np.rint(v).astype(np.int8)
        elif name.endswith("_rqsadd"):
            v = np.rint(v).astype(np.int32)
        else:
            v = v.astype(np.float32)
        out[name] = v
np.savez("/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp11_QZO_round1_fix/results/ref_updated_weights.npz", **out)
print(f"ref weights npz: {len(out)} tensors")
