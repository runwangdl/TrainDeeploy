import onnx, onnx_graphsurgeon as gs, sys
path = sys.argv[1] if len(sys.argv) > 1 else \
    "DeeployTest/TEST_GAP9_W_NE16/dbg/deeployStates/middleware_post_lowering.onnx"
g = gs.import_onnx(onnx.load(path))
for n in g.nodes:
    print(f"{n.op:12s} engine={str(n.attrs.get('engine','-')):13s} {n.name[:38]}")
    for t in n.inputs:
        print(f"   in  {str(list(t.shape) if t.shape is not None else '?'):22s} {t.name[:36]}")
    for t in n.outputs:
        print(f"   out {str(list(t.shape) if t.shape is not None else '?'):22s} {t.name[:36]}")
    if n.op == "Transpose":
        print(f"       perm={n.attrs.get('perm')}")
