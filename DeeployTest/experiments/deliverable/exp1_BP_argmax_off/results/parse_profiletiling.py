import re
from collections import defaultdict
L=open("profiletiling.log").read().splitlines()
# parse per-node (tag) sums of Pre/Kernel/Post over tiles
node=defaultdict(lambda:[0,0,0,0])  # pre,ker,post,ntiles
rx=re.compile(r'^\[([^\]]+)\]\[[SD]B\]\[(\d+) ops\]\[Tile (\d+)\] (Pre-Kernel|Kernel|Post-Kernel)\s*:\s*(\d+) cycles')
for ln in L:
    m=rx.match(ln)
    if not m: continue
    tag=m.group(1); phase=m.group(4); c=int(m.group(5))
    idx={"Pre-Kernel":0,"Kernel":1,"Post-Kernel":2}[phase]
    node[tag][idx]+=c
    if phase=="Pre-Kernel": node[tag][3]+=1
def cat(tag):
    t=tag
    if t.startswith("sgd_"): return "optimizer"
    if "Accumulator" in t: return "grad-accum"
    if "grad" in t.lower() or "Grad" in t or "ReduceSum" in t or "SoftmaxCrossEntropyLossGrad" in t: return "backward"
    return "forward"
def op(tag):
    if tag.startswith("sgd_"): return "SGD"
    if "Accumulator" in tag: return "GradAccum"
    for k in ["ConvGrad","GradConv"]:
        if k in tag: return "ConvGrad"
    if "MaxPoolGrad" in tag or "GradMaxPool" in tag: return "MaxPoolGrad"
    if "ReluGrad" in tag or "GradRelu" in tag: return "ReluGrad"
    if "BatchNormalizationGrad" in tag or "GradBatchNorm" in tag: return "BNGrad"
    if "GemmGrad" in tag or "GradGemm" in tag: return "GemmGrad"
    if "SoftmaxCrossEntropyLossGrad" in tag or ("Softmax" in tag and "grad" in tag.lower()): return "SoftmaxGrad"
    if "ReduceSum" in tag: return "ReduceSum(grad)"
    if "transpose" in tag.lower(): return "transpose"
    if "_Conv_Conv" in tag or tag.endswith("_Conv_L2"): return "Conv"
    if "BatchNormInternal" in tag: return "BatchNorm"
    if "Relu_Relu" in tag: return "ReLU"
    if "MaxPool_MaxPool" in tag: return "MaxPool"
    if "Gemm" in tag: return "Gemm"
    if "Softmax" in tag: return "Softmax"
    return "other"
CAT=defaultdict(lambda:[0,0,0]); OP=defaultdict(lambda:[0,0,0])
tot=[0,0,0]
for tag,(pre,ker,post,nt) in node.items():
    c=cat(tag); o=op(tag)
    CAT[c][0]+=pre; CAT[c][1]+=ker; CAT[c][2]+=post
    OP[(c,o)][0]+=pre; OP[(c,o)][1]+=ker; OP[(c,o)][2]+=post
    tot[0]+=pre; tot[1]+=ker; tot[2]+=post
T=sum(tot)
print(f"NODES parsed: {len(node)}   PROFILED TOTAL = {T:,} cycles")
print(f"  DMA-in (Pre)  : {tot[0]:>12,}  {100*tot[0]/T:5.1f}%")
print(f"  Kernel(compute): {tot[1]:>11,}  {100*tot[1]/T:5.1f}%")
print(f"  DMA-out(Post) : {tot[2]:>12,}  {100*tot[2]/T:5.1f}%")
print(f"  => COMPUTE {100*tot[1]/T:.1f}%   TRANSFER {100*(tot[0]+tot[2])/T:.1f}%")
print("\n=== by phase category (Pre+Kernel+Post) ===")
for c in ["forward","backward","grad-accum","optimizer"]:
    p,k,po=CAT[c]; s=p+k+po
    print(f"  {c:11s}: total {s:>11,} ({100*s/T:5.1f}%)   [DMAin {p:,} | kernel {k:,} | DMAout {po:,}]  compute {100*k/s:.0f}%")
fwd=sum(CAT['forward']); bwd=sum(CAT['backward'])+sum(CAT['grad-accum'])
print(f"\n  BACKWARD/FORWARD ratio = {bwd/fwd:.2f}x   (fwd {fwd:,}  vs  bwd+accum {bwd:,})")
bwd_noacc=sum(CAT['backward'])
print(f"  BACKWARD(excl accum)/FORWARD = {bwd_noacc/fwd:.2f}x")
print("\n=== top ops by total cycles ===")
rows=sorted(OP.items(),key=lambda kv:-sum(kv[1]))
for (c,o),(p,k,po) in rows[:16]:
    s=p+k+po
    print(f"  {c:9s} {o:16s} {s:>10,} ({100*s/T:4.1f}%)  k={100*k/s:3.0f}% pre={100*p/s:3.0f}% post={100*po/s:3.0f}%")
