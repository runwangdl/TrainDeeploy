import onnx, onnx_graphsurgeon as gs
from Deeploy.Targets.NE16.Engine import NE16Engine
for f in ["b1_1x2_ne16", "b1_ref_1x2"]:
    g = gs.import_onnx(onnx.load(f"DeeployTest/Tests/Models/NE16/{f}/network.onnx"))
    for flag in (False, True):
        e = NE16Engine("NE16", enable1xK = flag)
        acc = [n.name for n in g.nodes if n.op in ("Conv", "RequantizedConv") and e.canExecute(n)]
        print(f"  {f:14s} enable1xK={str(flag):5s} -> accepted {len(acc)} {acc}")
