import numpy as np, onnx
E = "/app/Onnx4Deeploy/QZO_exp/exp11_round1_fix"
d = np.load("/app/Onnx4Deeploy/QZO_exp/exp_calibration/data_cache_incr.npz")
fx = np.load(E + "/inputs.npz")
ok = 0
for j in range(54):
    k = "arr_0000" if j == 0 else f"mb{j}_arr_0000"
    kl = "arr_0001" if j == 0 else f"mb{j}_arr_0001"
    same = (np.array_equal(fx[k].astype(np.float32).reshape(-1), d["trX1"][j].reshape(-1))
            and int(np.asarray(fx[kl]).reshape(-1)[0]) == int(d["trY1"][j]))
    ok += int(same)
print(f"windows byte-compare: {ok}/54")
m = onnx.load(E + "/network.onnx")
lead = next(n for n in m.graph.node if "input_quant" in n.name and n.op_type == "Quant")
sc = [onnx.helper.get_attribute_value(a) for a in lead.attribute if a.name == "scale"][0]
print(f"leading Quant scale: {sc} (expect 22.296875)")
o = np.load(E + "/outputs.npz")
u = [k for k in o.files if k.startswith("updated_")]
print(f"outputs.npz: loss_plus={o['loss_plus'].shape} loss_minus={o['loss_minus'].shape} updated_*={len(u)}")
