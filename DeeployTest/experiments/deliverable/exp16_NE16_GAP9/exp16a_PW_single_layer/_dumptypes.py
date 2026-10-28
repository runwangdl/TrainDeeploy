import pickle, glob, sys
d = sys.argv[1] if len(sys.argv) > 1 else "b1_1x2_ne16"
f = glob.glob(f"DeeployTest/TEST_GAP9_W_NE16/Tests/Models/NE16/{d}/deeployStates/backend_post_binding.pkl")[0]
ctxt = pickle.load(open(f, "rb"))
objs = {**getattr(ctxt, "globalObjects", {}), **getattr(ctxt, "localObjects", {})}
print(f"{len(objs)} buffers")
for n, b in sorted(objs.items()):
    t = getattr(b, "_type", None)
    tn = t.referencedType.typeName if (t is not None and hasattr(t, "referencedType")) else str(t)
    print(f"  {n[:46]:46s} shape={str(getattr(b,'shape',None)):24s} type={tn}")
