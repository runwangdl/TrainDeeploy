import onnx, onnx_graphsurgeon as gs, numpy as np, sys
D = "/app/Deeploy/DeeployTest/TEST_GAP9_W_NE16/Tests/Models/NE16/speechnet_b1_pw_rq/deeployStates"
for stage in ["middleware_pre_lowering", "middleware_post_lowering"]:
    g = gs.import_onnx(onnx.load(f"{D}/{stage}.onnx"))
    print(f"\n########## {stage}")
    for n in g.nodes:
        eng = n.attrs.get("engine", "-")
        print(f"  {n.op:22s} engine={str(eng):13s} {n.name[:44]}")
        for t in n.inputs:
            kind = "CONST" if isinstance(t, gs.Constant) else "var  "
            shp = list(t.shape) if t.shape is not None else "?"
            print(f"        in  {kind} {str(shp):22s} {t.name[:44]}")
        for t in n.outputs:
            print(f"        out       {str(list(t.shape) if t.shape is not None else '?'):22s} {t.name[:44]}")
        ks = n.attrs.get("kernel_shape"); wo = n.attrs.get("weight_offset")
        if ks is not None or wo is not None:
            print(f"        attrs kernel_shape={ks} weight_offset={wo} group={n.attrs.get('group')}")
