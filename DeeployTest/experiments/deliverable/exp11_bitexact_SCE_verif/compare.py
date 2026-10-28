import re, struct, numpy as np, onnx
V="/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp11_bitexact_SCE_verif"
m=onnx.load("/app/Onnx4Deeploy/QZO_exp/exp11_round1_fix/network_zo_train.onnx")
ELEM={1:("f4",1),3:("i1",4),6:("i4",1)}
wmap=[(i.name,i.type.tensor_type.elem_type,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input[2:]]
pat=re.compile(r"\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)")
dev={}
for line in open(V+"/logs/device_to891.log",errors="replace"):
    mm=pat.search(line)
    if not mm: continue
    s,wi=int(mm.group(1)),int(mm.group(2)); words=[int(w,16) for w in mm.group(4).split()]
    name,et,shp=wmap[wi]; dt,_=ELEM[et]; raw=b"".join(struct.pack("<I",w) for w in words)
    a=np.frombuffer(raw,np.dtype(dt)); n=int(np.prod(shp)) if shp else a.size
    dev.setdefault(s,{})[name]=a[:n]
for s in (884,888):
    d=np.load(f"{V}/host/P_step{s}.npz"); host={k:np.asarray(d[k]).reshape(-1) for k in d.files}
    print(f"=== step {s}: device vs host, per param ===")
    for name,et,shp in wmap:
        dv=dev[s][name].astype(np.float64); hv=host[name].astype(np.float64)
        nd=int((dv!=hv).sum()); mx=float(np.abs(dv-hv).max()) if dv.size else 0.0
        tag="int8" if name.endswith("_int8") else ("int32" if name.endswith("_rqsadd") else "fp32")
        if nd: print(f"   {name:28s} {tag:5s} differ={nd}/{dv.size} maxabs={mx:.3e}")
    ndc=sum(int((dev[s][n].astype(np.int64)!=host[n].astype(np.int64)).sum()) for n,_,_ in wmap if n.endswith("_int8"))
    print(f"   -> conv int8 total differing: {ndc}")
