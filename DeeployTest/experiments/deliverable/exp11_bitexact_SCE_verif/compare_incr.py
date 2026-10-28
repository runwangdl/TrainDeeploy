import re, gzip, struct, numpy as np, onnx, sys
V="/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp11_bitexact_SCE_verif"
DEVLOG=V+"/logs/device_to891.log"
# wi -> param name + dtype, from the zo_train graph input order (after 2 data inputs)
m=onnx.load("/app/Onnx4Deeploy/QZO_exp/exp11_round1_fix/network_zo_train.onnx")
ELEM={1:("f4",1),3:("i1",4),6:("i4",1)}
wmap=[]
for i in m.graph.input[2:]:
    et=i.type.tensor_type.elem_type; shp=[d.dim_value for d in i.type.tensor_type.shape.dim]
    wmap.append((i.name, et, shp))
# parse device WDUMP: {step: {wi: np.array}}
pat=re.compile(r"\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)")
dev={}
op=gzip.open(DEVLOG,"rt",errors="replace") if DEVLOG.endswith(".gz") else open(DEVLOG,errors="replace")
for line in op:
    mm=pat.search(line)
    if not mm: continue
    s,wi=int(mm.group(1)),int(mm.group(2)); words=[int(w,16) for w in mm.group(4).split()]
    name,et,shp=wmap[wi]; dt,_=ELEM[et]
    raw=b"".join(struct.pack("<I",w) for w in words); arr=np.frombuffer(raw,dtype=np.dtype(dt))
    n=int(np.prod(shp)) if shp else arr.size
    dev.setdefault(s,{})[name]=arr[:n].reshape(shp)
def host(s):
    d=np.load(f"{V}/host/P_step{s}.npz"); return {k:d[k] for k in d.files}
print("step | weights_in(=prev step) identical dev vs host? | this-step increment identical?")
for s in range(885,891):
    if s not in dev or (s-1) not in dev: continue
    hp=host(s); hpm=host(s-1)
    # weights going INTO step s = weights[s-1]
    win_id = all(np.array_equal(np.asarray(dev[s-1][n]).reshape(-1), np.asarray(hpm[n]).reshape(-1).astype(np.asarray(dev[s-1][n]).dtype)) for n,_,_ in wmap)
    # increment[s] = weights[s]-weights[s-1]; compare device incr vs host incr (per conv-weight)
    def incr(D,A,B):
        return {n: (np.asarray(A[n]).astype(np.int64)-np.asarray(B[n]).astype(np.int64)) if n.endswith(("_int8","_rqsadd")) else (np.asarray(A[n]).astype(np.float64)-np.asarray(B[n]).astype(np.float64)) for n,_,_ in wmap}
    di=incr(dev,dev[s],dev[s-1]); hi=incr(None,hp,hpm)
    incr_id = all(np.array_equal(di[n].reshape(-1), hi[n].reshape(-1)) for n,_,_ in wmap if n.endswith("_int8"))
    # count conv-int8 weights whose device increment != host increment
    ndiff=sum(int((di[n].reshape(-1)!=hi[n].reshape(-1)).sum()) for n,_,_ in wmap if n.endswith("_int8"))
    print(f"  {s}  | weights_in identical: {win_id}  | conv-int8 increment identical: {incr_id}  (mismatched weights: {ndiff})")
