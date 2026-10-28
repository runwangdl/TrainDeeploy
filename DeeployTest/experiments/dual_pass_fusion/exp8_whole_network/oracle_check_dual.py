# Expand the dual-pass graph into oracle-known ops and run it; compare L+ / L- with the fixture references.
import sys, numpy as np, onnx, onnx_graphsurgeon as gs, tempfile, os
sys.path.insert(0, "/app/TrainDeeploy")
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.ZODualPassFusion import zoDualPassFusion
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph
fx = "/app/TrainDeeploy/SpeechNet_Test/speechnet_train_QZO_single_step"
model = onnx.load(f"{fx}/network.onnx")
g = zoDualPassFusion(gs.import_onnx(model))
k = [0]
def nm(b):
    k[0] += 1; return f"{b}__x{k[0]}"
new, rm = [], []
for n in list(g.nodes):
    if n.op == "PerturbRademacherFixed":
        mode = int(n.attrs["mode"]); outs = []
        for sgn in ((1, -1) if mode == 0 else ((1,) if mode == 1 else (-1,))):
            o = gs.Variable(nm(n.name), dtype = np.float32)
            new.append(gs.Node("PerturbRademacher", nm(n.name), domain = n.domain, inputs = [n.inputs[0]], outputs = [o],
                               attrs = {"seed": n.attrs["seed"], "idx": n.attrs["idx"], "eps": float(n.attrs["eps"]) * sgn}))
            outs.append(o)
        out = n.outputs[0]; n.outputs = []; rm.append(n)
        if len(outs) == 1:
            new.append(gs.Node("Identity", nm("id"), inputs = outs, outputs = [out]))
        else:
            new.append(gs.Node("Concat", nm("cat"), inputs = outs, outputs = [out], attrs = {"axis": 0}))
    elif n.op == "RequantizedDualConv":
        x, W, mul, add, wmul, bmul = n.inputs[:6]
        stacked = int(n.attrs.get("stacked_input", 0))
        Ci = W.shape[1]; outs = []
        for s in (0, 1):
            xs = x
            if stacked:
                xs = gs.Variable(nm("xs"), dtype = np.int8)
                new.append(gs.Node("Slice", nm("sl"), inputs = [x, gs.Constant(nm("st"), np.array([s * Ci])), gs.Constant(nm("en"), np.array([(s + 1) * Ci])),
                                                               gs.Constant(nm("ax"), np.array([1]))], outputs = [xs]))
            wm_s = wmul if s == 0 else gs.Constant(nm("nwm"), -np.asarray(wmul.values))
            bm_s = bmul if s == 0 else gs.Constant(nm("nbm"), -np.asarray(bmul.values))
            wp = gs.Variable(nm("wp"), dtype = np.int8); bp = gs.Variable(nm("bp"), dtype = np.int32)
            new.append(gs.Node("RQSPerturbRademacher", nm("pw"), domain = "mezo", inputs = [W, wm_s], outputs = [wp],
                               attrs = {"seed": n.attrs["seed_w"], "idx": n.attrs["idx_w"], "div": n.attrs["div_w"], "n_levels": 256, "signed": 1}))
            new.append(gs.Node("RQSPerturbRademacher", nm("pb"), domain = "mezo", inputs = [add, bm_s], outputs = [bp],
                               attrs = {"seed": n.attrs["seed_b"], "idx": n.attrs["idx_b"], "div": n.attrs["div_b"], "n_levels": 4294967296, "signed": 1}))
            co = gs.Variable(nm("co"), dtype = np.int32); y = gs.Variable(nm("y"), dtype = np.int8)
            new.append(gs.Node("Conv", nm("conv"), inputs = [xs, wp], outputs = [co],
                               attrs = {a: n.attrs[a] for a in ("kernel_shape", "pads", "strides", "dilations", "group")}))
            new.append(gs.Node("RequantShift", nm("rqs"), domain = "ai.onnx.contrib", inputs = [co, mul, bp], outputs = [y],
                               attrs = {a: n.attrs[a] for a in ("div", "n_levels", "signed")}))
            outs.append(y)
        out = n.outputs[0]; n.outputs = []; rm.append(n)
        new.append(gs.Node("Concat", nm("cat"), inputs = outs, outputs = [out], attrs = {"axis": 1}))
for n in rm:
    g.nodes.remove(n)
g.nodes.extend(new); g.cleanup().toposort()
m2 = gs.export_onnx(g); m2.opset_import.extend([o for o in model.opset_import if o.domain not in [x.domain for x in m2.opset_import]])
tmp = tempfile.mktemp(suffix = ".onnx"); onnx.save(m2, tmp)
npz = np.load(f"{fx}/inputs.npz"); base = [npz[k] for k in sorted(k for k in npz.files if k.startswith("arr_"))]
feed = {i.name: np.asarray(v) for i, v in zip(model.graph.input, base)}
res = run_onnx_graph(tmp, feed, output_names = ["loss", "loss_minus"])
out = np.load(f"{fx}/outputs.npz")
print("res shapes", [np.asarray(r).shape for r in res], [np.asarray(r).ravel()[:4] for r in res])
print("refs: L+", out["loss_plus"].ravel()[:2], "L-", out["loss_minus"].ravel()[:2])
