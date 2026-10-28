# ZO analog of exp1 BP lat_fig2_per_operator.py — per-operator latency for one MeZO step.
# ZO has NO backward/optimizer. Categories: forward (2 perturbed forwards) / perturbation (theta +/- eps*z
# generated before each forward) / update (zo_update in-place theta update). Horizontal bars, kernel+DMA.
import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
from collections import defaultdict
from matplotlib.patches import Patch
L=open("../logs/profiletiling.log").read().splitlines()
node=defaultdict(lambda:[0,0,0])
rx=re.compile(r'^\[([^\]]+)\]\[[SD]B\]\[(\d+) ops\]\[Tile (\d+)\] (Pre-Kernel|Kernel|Post-Kernel)\s*:\s*(\d+) cycles')
for ln in L:
    m=rx.match(ln)
    if m: node[m.group(1)][{"Pre-Kernel":0,"Kernel":1,"Post-Kernel":2}[m.group(4)]]+=int(m.group(5))
def cat(t):
    tl=t.lower()
    if "perturb" in tl and "updated" in tl: return "update"
    if "perturb" in tl: return "perturbation"
    return "forward"
def op(t):
    tl=t.lower()
    if "perturb" in tl and "updated" in tl: return "Perturb (update θ)"
    if "perturb" in tl: return "Perturb (θ±εz)"
    if "transpose" in tl: return "transpose (layout)"
    if "BatchNormalization" in t or "BatchNormInternal" in t: return "BatchNorm"
    if "Relu" in t: return "ReLU"
    if "MaxPool" in t: return "MaxPool"
    if "Gemm" in t: return "Gemm"
    if "CrossEntropyLoss" in t or "Softmax" in t: return "SoftmaxCE loss"
    if "_Conv_L2" in t: return "Conv"
    return "other"
CAT=defaultdict(lambda:[0,0,0]); OP=defaultdict(lambda:[0,0,0]); tot=[0,0,0]
for t,(p,k,po) in node.items():
    c=cat(t); CAT[c][0]+=p;CAT[c][1]+=k;CAT[c][2]+=po
    o=op(t);  OP[(c,o)][0]+=p;OP[(c,o)][1]+=k;OP[(c,o)][2]+=po
    tot[0]+=p;tot[1]+=k;tot[2]+=po
T=sum(tot); M=1e6
COL={"forward":"#4C72B0","perturbation":"#DD8452","update":"#55A868"}
rows=sorted(OP.items(),key=lambda kv:-sum(kv[1]))
rows=[r for r in rows if sum(r[1])>2000][:12]
labels=[o for (c,o),_ in rows]; colcat=[COL[c] for (c,o),_ in rows]
ker=[v[1]/M for _,v in rows]; dma=[(v[0]+v[2])/M for _,v in rows]
y=np.arange(len(rows))[::-1]
fig,ax=plt.subplots(figsize=(11,5.4))
ax.barh(y,ker,color=colcat,height=.66)
ax.barh(y,dma,left=ker,color="#cccccc",height=.66)
for i,((c,o),v) in zip(y,rows):
    s=sum(v); ax.text(s/M+0.2,i,f"{s/M:.3f}M ({100*s/T:.2f}%) · {100*v[1]/s:.0f}% compute",va="center",fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("Mcycles  (solid = compute kernel, grey = DMA transfer)")
ax.set_xlim(0, max(sum(v) for _,v in rows)/M*1.35)
ax.set_title("Per-operator latency — one ZO (MeZO) step")
ax.legend(handles=[Patch(color=COL["forward"],label="forward (×2: +ε, −ε)"),
                   Patch(color=COL["perturbation"],label="perturbation (θ±εz)"),
                   Patch(color=COL["update"],label="update (zo_update)"),
                   Patch(color="#cccccc",label="DMA (transfer)")],fontsize=9,loc="lower right")
plt.tight_layout(); plt.savefig("lat_fig2_per_operator.png",dpi=150); plt.close()
print(f"saved lat_fig2_per_operator.png  | TOTAL {T/M:.2f}M")
for c in ["forward","perturbation","update"]:
    s=sum(CAT[c]); print(f"  {c:12s} {s/M:7.3f}M ({100*s/T:5.2f}%)")
