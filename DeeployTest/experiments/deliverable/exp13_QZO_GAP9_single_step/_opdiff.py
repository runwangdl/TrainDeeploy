import onnx, glob, os
from Deeploy.Targets.GAP9.Platform import GAP9Mapping
keys = set(GAP9Mapping.keys())
base = "/app/ETH/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet"
for fx in ["speechnet_qzo12_train","speechnet_qzo12_update"]:
    onnxs = glob.glob(os.path.join(base,fx,"*.onnx"))
    for o in onnxs:
        m = onnx.load(o)
        ops = {}
        for n in m.graph.node:
            ops[n.op_type] = ops.get(n.op_type,0)+1
        missing = sorted(op for op in ops if op not in keys)
        print(f"\n{fx}/{os.path.basename(o)}: {len(m.graph.node)} nodes")
        print("  op types:", dict(sorted(ops.items())))
        print("  MISSING from GAP9Mapping:", missing if missing else "NONE")
