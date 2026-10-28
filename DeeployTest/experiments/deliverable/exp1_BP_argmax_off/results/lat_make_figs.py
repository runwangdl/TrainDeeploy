import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
from collections import defaultdict
from matplotlib.patches import Patch
L=open("../logs/profiletiling.log").read().splitlines()
node=defaultdict(lambda:[0,0,0])
rx=re.compile(r'^\[([^\]]+)\]\[[SD]B\]\[(\d+) ops\]\[Tile (\d+)\] (Pre-Kernel|Kernel|Post-Kernel)\s*:\s*(\d+) cycles')
for ln in L:
    m=rx.match(ln)
    if not m: continue
    idx={"Pre-Kernel":0,"Kernel":1,"Post-Kernel":2}[m.group(4)]
    node[m.group(1)][idx]+=int(m.group(5))
def cat(t):
    if t.startswith("sgd_"): return "optimizer"
    if "Accumulator" in t: return "backward"
    if "grad" in t.lower() or "Grad" in t or "ReduceSum" in t: return "backward"
    return "forward"
def op(tag):
    if tag.startswith("sgd_"): return "SGD (optimizer)"
    if "Accumulator" in tag: return "GradAccum"
    if "ConvGrad" in tag or "GradConv" in tag: return "ConvGrad"
    if "MaxPoolGrad" in tag or "GradMaxPool" in tag: return "MaxPoolGrad"
    if "ReluGrad" in tag or "GradRelu" in tag: return "ReluGrad"
    if "BatchNormalizationGrad" in tag or "GradBatchNorm" in tag: return "BNGrad"
    if "GemmGrad" in tag or "GradGemm" in tag: return "GemmGrad"
    if "grad" in tag.lower() and "transpose" in tag.lower(): return "grad-transpose"
    if "transpose" in tag.lower(): return "transpose (fwd)"
    if "BatchNormInternal" in tag: return "BatchNorm"
    if "Relu_Relu" in tag: return "ReLU"
    if "MaxPool_MaxPool" in tag: return "MaxPool"
    if "Gemm" in tag: return "Gemm"
    if "Softmax" in tag: return "Softmax"
    if "_Conv_Conv" in tag or tag.endswith("_Conv_L2"): return "Conv"
    return "other"
CAT=defaultdict(lambda:[0,0,0]); OP=defaultdict(lambda:[0,0,0]); tot=[0,0,0]
for tag,(p,k,po) in node.items():
    c=cat(tag); CAT[c][0]+=p;CAT[c][1]+=k;CAT[c][2]+=po
    o=op(tag);  OP[(c,o)][0]+=p;OP[(c,o)][1]+=k;OP[(c,o)][2]+=po
    tot[0]+=p;tot[1]+=k;tot[2]+=po
T=sum(tot); M=1e6
plt.rcParams.update({"font.size":11})

# ---- Fig 1: two panels: (a) compute vs transfer, (b) fwd/bwd/opt stacked ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1,1.4]})
# (a) compute vs transfer
ax1.bar(0, tot[1]/M, width=.6, color="#4C72B0", label="Compute (kernel)")
ax1.bar(0, tot[0]/M, bottom=tot[1]/M, width=.6, color="#DD8452", label="DMA-in")
ax1.bar(0, tot[2]/M, bottom=(tot[1]+tot[0])/M, width=.6, color="#C44E52", label="DMA-out")
ax1.text(0, tot[1]/2/M, f"Compute\n{100*tot[1]/T:.1f}%", ha="center",va="center",color="white",fontweight="bold")
ax1.text(0, (tot[1]+ (tot[0]+tot[2])/2)/M, f"Transfer {100*(tot[0]+tot[2])/T:.1f}%", ha="center",va="bottom",fontsize=9)
ax1.set_xticks([]); ax1.set_ylabel("Mcycles"); ax1.set_title("Compute vs transfer\n(one training step)")
ax1.legend(fontsize=8,loc="upper right")
# (b) fwd/bwd/opt stacked compute+transfer
cats=["forward","backward","optimizer"]; x=np.arange(3)
ker=[CAT[c][1]/M for c in cats]; dma=[(CAT[c][0]+CAT[c][2])/M for c in cats]
ax2.bar(x,ker,width=.6,color="#4C72B0",label="Compute")
ax2.bar(x,dma,bottom=ker,width=.6,color="#DD8452",label="Transfer (DMA)")
for i,c in enumerate(cats):
    s=sum(CAT[c])
    ax2.text(i,(ker[i]+dma[i])+0.4,f"{s/M:.1f}M\n{100*s/T:.1f}%",ha="center",fontsize=10,fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(["Forward","Backward","Optimizer"]); ax2.set_ylabel("Mcycles")
ax2.set_title(f"Forward vs Backward vs Optimizer\nbackward / forward = {sum(CAT['backward'])/sum(CAT['forward']):.2f}×  (not 2×)")
ax2.legend(fontsize=9); ax2.set_ylim(0,20)
plt.tight_layout(); plt.savefig("lat_fig1_compute_transfer_fwd_bwd.png",dpi=150); plt.close()

# ---- Fig 2: per-op horizontal, kernel vs dma ----
rows=sorted(OP.items(),key=lambda kv:-sum(kv[1]))
rows=[r for r in rows if sum(r[1])>3000][:12]
labels=[f"{o}" for (c,o),_ in rows]; colcat=['#C44E52' if c=='backward' else '#8172B3' if c=='optimizer' else '#4C72B0' for (c,o),_ in rows]
ker=[v[1]/M for _,v in rows]; dma=[(v[0]+v[2])/M for _,v in rows]
y=np.arange(len(rows))[::-1]
fig,ax=plt.subplots(figsize=(11,5.2))
ax.barh(y,ker,color=colcat,height=.66)
ax.barh(y,dma,left=ker,color="#cccccc",height=.66)
for i,((c,o),v) in zip(y,rows):
    s=sum(v); ax.text(s/M+0.15,i,f"{s/M:.2f}M ({100*s/T:.1f}%)  ·  {100*v[1]/s:.0f}% compute",va="center",fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlabel("Mcycles (solid=compute, grey=DMA)")
ax.set_xlim(0,17); ax.set_title("Per-operator latency — one training step")
ax.legend(handles=[Patch(color="#4C72B0",label="forward"),Patch(color="#C44E52",label="backward"),Patch(color="#8172B3",label="optimizer"),Patch(color="#cccccc",label="DMA (transfer)")],fontsize=9,loc="lower right")
plt.tight_layout(); plt.savefig("lat_fig2_per_operator.png",dpi=150); plt.close()
print("figs written")
print(f"TOTAL {T/M:.2f}M | compute {100*tot[1]/T:.1f}% transfer {100*(tot[0]+tot[2])/T:.1f}%")
print(f"fwd {sum(CAT['forward'])/M:.2f}M  bwd {sum(CAT['backward'])/M:.2f}M  opt {sum(CAT['optimizer'])/M:.3f}M  bwd/fwd={sum(CAT['backward'])/sum(CAT['forward']):.2f}")
