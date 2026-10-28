# Fused per-operator latency: BP (1 fwd + 1 bwd + opt) vs ZO/MeZO (2 perturbed fwd + update), one step, n_accum=1.
# Vertical grouped bars. Reads both profiletiling.log traces. Regions: shared FORWARD (ZO runs it 2x) |
# BP-only BACKWARD+optimizer | ZO-only PERTURBATION. BP: exp1_BP_argmax_off, ZO: exp6_ZO_single_step_latency.
import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
from collections import defaultdict
from matplotlib.patches import Patch
rx=re.compile(r'^\[([^\]]+)\]\[[SD]B\]\[(\d+) ops\]\[Tile (\d+)\] (Pre-Kernel|Kernel|Post-Kernel)\s*:\s*(\d+) cycles')
def op(t):
    tl=t.lower()
    if "perturb" in tl and "updated" in tl: return "Perturb(update)"
    if "perturb" in tl: return "Perturb(theta)"
    if "convgrad" in tl or "gradconv" in tl: return "ConvGrad"
    if "maxpoolgrad" in tl or "gradmaxpool" in tl: return "MaxPoolGrad"
    if "batchnorm" in tl and "grad" in tl: return "BNGrad"
    if "relu" in tl and "grad" in tl: return "ReluGrad"
    if "gemm" in tl and "grad" in tl: return "GemmGrad"
    if "softmax" in tl and "grad" in tl: return "SoftmaxGrad"
    if "reducesum" in tl: return "Optim"
    if "accumulator" in tl: return "Optim"
    if t.startswith("sgd_"): return "Optim"
    if "transpose" in tl: return "transpose"
    if "batchnorm" in tl: return "BatchNorm"
    if "relu" in tl: return "ReLU"
    if "maxpool" in tl: return "MaxPool"
    if "gemm" in tl: return "Gemm+Loss"
    if "crossentropy" in tl or "softmax" in tl: return "Gemm+Loss"
    if "_conv_l2" in tl: return "Conv"
    return "Gemm+Loss"
def per_op(path):
    node=defaultdict(lambda:0)
    for ln in open(path):
        m=rx.match(ln)
        if m: node[op(m.group(1))]+=int(m.group(5))
    return node
BP=per_op("../../exp1_BP_argmax_off/logs/profiletiling.log")
ZO=per_op("../logs/profiletiling.log")
M=1e6
# x layout: forward (shared) | BP backward+opt | ZO perturbation
fwd =["Conv","transpose","MaxPool","BatchNorm","ReLU","Gemm+Loss"]
bwd =["ConvGrad","MaxPoolGrad","BNGrad","ReluGrad","Optim"]
pert=["Perturb(theta)","Perturb(update)"]
ops=fwd+bwd+pert
labels=[{"Perturb(theta)":"Perturb\n(θ±εz)","Perturb(update)":"Perturb\n(update)",
         "Optim":"SGD\n(update)"}.get(o,o) for o in ops]
bp=[BP.get(o,0)/M for o in ops]; zo=[ZO.get(o,0)/M for o in ops]
x=np.arange(len(ops)); w=0.40
cbp,czo="#4C72B0","#DD8452"
fig,ax=plt.subplots(figsize=(15,6.2))
# region shading
def span(i0,i1,color,txt,fs=11):
    ax.axvspan(i0-0.5,i1+0.5,color=color,alpha=0.09,zorder=0)
    ax.text((i0+i1)/2, 30.9, txt, ha="center", va="center", fontsize=fs, fontweight="bold", color="#333")
span(0,len(fwd)-1,"#4C72B0","FORWARD  (shared — ZO runs it ×2)")
span(len(fwd),len(fwd)+len(bwd)-1,"#C44E52","BP only: BACKWARD + optimizer")
span(len(fwd)+len(bwd),len(ops)-1,"#DD8452","ZO only:\nPERTURBATION",fs=9.5)
bbp=ax.bar(x-w/2,bp,w,color=cbp,label="BP  (1 fwd + 1 bwd + opt)")
bzo=ax.bar(x+w/2,zo,w,color=czo,label="ZO  (2 fwd + update)")
for xi,v in zip(x-w/2,bp):
    if v>0: ax.text(xi,v+0.25,f"{v:.2f}",ha="center",fontsize=7.5,color=cbp,fontweight="bold")
for xi,v in zip(x+w/2,zo):
    if v>0: ax.text(xi,v+0.25,f"{v:.2f}",ha="center",fontsize=7.5,color="#B4652A",fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9.5)
ax.set_ylabel("Mcycles  (one training step, n_accum=1)"); ax.set_ylim(0,32)
ax.yaxis.grid(True,color="#e9e9e9"); ax.set_axisbelow(True)
tbp=sum(BP.values())/M; tzo=sum(ZO.values())/M
ax.set_title(f"Per-operator latency — BP vs ZO (one step)   |   BP total {tbp:.1f}M   ·   ZO total {tzo:.1f}M   "
             f"({tzo/tbp:.2f}×)\nZO doubles the forward but drops the entire backward "
             f"(BP {sum(BP.get(o,0) for o in bwd)/M:.1f}M) for a tiny perturbation (ZO {sum(ZO.get(o,0) for o in pert)/M:.2f}M)",
             fontsize=12)
ax.legend(loc="upper right",bbox_to_anchor=(0.995,0.80),fontsize=11,framealpha=.95)
plt.tight_layout(); plt.savefig("lat_fig2b_per_operator_BP_vs_ZO.png",dpi=150); plt.close()
print("saved lat_fig2b_per_operator_BP_vs_ZO.png")
print(f"BP {tbp:.2f}M  ZO {tzo:.2f}M  ({tzo/tbp:.3f}x)")
print("forward: BP", round(sum(BP.get(o,0) for o in fwd)/M,2),"ZO",round(sum(ZO.get(o,0) for o in fwd)/M,2))
print("bwd/opt: BP", round(sum(BP.get(o,0) for o in bwd)/M,2),"| perturb: ZO",round(sum(ZO.get(o,0) for o in pert)/M,3))
