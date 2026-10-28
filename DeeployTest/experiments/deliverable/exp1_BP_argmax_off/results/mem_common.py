import re
def parse(fn):
    html=open(fn).read(); out={}
    for seg in html.split("var fig =")[1:]:
        ml=re.search(r'"text":"(L[0-9]) Memory Size","x":\[[-\d.,]+\],"y":\[(\d+)',seg)
        if not ml: continue
        lvl=ml.group(1); cap=int(ml.group(2)); blocks=[]
        for m in re.finditer(r'"text":"([^"]+)","x":\[([-\d.,]+)\],"y":\[([-\d.,]+)\]',seg):
            n=m.group(1)
            if n.endswith("Memory Size"): continue
            xs=[float(v) for v in m.group(2).split(",")]; ys=[float(v) for v in m.group(3).split(",")]
            blocks.append((n,min(ys),max(ys),max(ys)-min(ys),min(xs),max(xs)))
        out[lvl]=(cap,blocks)
    return out
def cat(n):
    ln=n.lower()
    if "running_mean" in n or "running_var" in n: return "Frozen BN stats"
    if re.search(r'(weight|bias|_gamma|_beta)_tensor$',n) and "grad" not in ln: return "Weights (params)"
    if "grad" in ln: return "Gradients (backward)"
    if any(k in ln for k in ["_transposed","pre_transpose","transpose_in"]): return "Layout-transpose copies"
    if any(k in ln for k in ["buffer","im2col","_split"]): return "Workspace/scratch"
    if any(k in n for k in ["input","label","Loss","SoftmaxCrossEntropy","ReduceSum"]): return "I/O + loss"
    return "Fwd activations"
def highwater(blocks): return max(b[2] for b in blocks)
def peaksum(blocks):
    tmin=int(min(b[4] for b in blocks)); tmax=int(max(b[5] for b in blocks)); best=(0,None,[])
    for t in range(tmin,tmax+1):
        live=[b for b in blocks if b[4]<=t<=b[5]]; s=sum(b[3] for b in live)
        if s>best[0]: best=(s,t,live)
    return best
