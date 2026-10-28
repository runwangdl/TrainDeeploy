# ZO analog of exp1 BP lat_fig6_conv_vs_convgrad.py.
# BP compared Conv (forward) vs ConvGrad (dX+dW) per block — the extra cost of the BACKWARD pass.
# ZO is forward-only (no backward), so the ZO analog is Conv (forward) vs PERTURB per block: the perturbation
# kernels that generate θ±εz for that block's params (conv W+b, BN γ+β) — the "extra" cost beyond the forward.
# Both are summed over the 2 perturbed forwards of one ZO step (ratio is pass-invariant). Parsed from the trace.
import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
from collections import defaultdict
L=open("../logs/profiletiling.log").read().splitlines()
node=defaultdict(lambda:[0,0,0])
rx=re.compile(r'^\[([^\]]+)\]\[[SD]B\]\[(\d+) ops\]\[Tile (\d+)\] (Pre-Kernel|Kernel|Post-Kernel)\s*:\s*(\d+) cycles')
for ln in L:
    m=rx.match(ln)
    if m: node[m.group(1)][{"Pre-Kernel":0,"Kernel":1,"Post-Kernel":2}[m.group(4)]]+=int(m.group(5))
conv=[0]*5; pert=[0]*5
for t,(p,k,po) in node.items():
    m=re.search(r'blocks_(\d)',t)
    if not m: continue
    b=int(m.group(1)); s=p+k+po
    tl=t.lower()
    if "_conv_l2" in tl and "transpose" not in tl: conv[b]+=s
    elif "perturb" in tl and "updated" not in tl: pert[b]+=s
M=1e6; x=np.arange(5); w=0.38
fig,ax=plt.subplots(figsize=(10.5,5.2))
ax.bar(x-w/2,[v/M for v in conv],w,color="#4C72B0",label="Conv (forward, ×2 ε)")
ax.bar(x+w/2,[v/M for v in pert],w,color="#DD8452",label="Perturb (θ±εz: W+b, γ+β)")
for i in range(5):
    ax.text(i-w/2,conv[i]/M+0.12,f"{conv[i]/M:.3f}",ha="center",fontsize=8)
    r=100*pert[i]/conv[i] if conv[i] else 0
    ax.text(i+w/2,pert[i]/M+0.12,f"{pert[i]/M:.3f}\n{r:.1f}%",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"B{b}" for b in range(5)])
ax.set_ylabel("Mcycles per block (one ZO step)"); ax.set_ylim(0,16.5)
ax.set_title("Conv (forward) vs Perturb per block — ZO (MeZO)\n"
             "perturbation tracks WEIGHT size, conv tracks ACTIVATION size → perturb negligible early, "
             "comparable at B4")
ax.legend()
plt.tight_layout(); plt.savefig("lat_fig6_conv_vs_perturb.png",dpi=150); plt.close()
print("saved lat_fig6_conv_vs_perturb.png")
for b in range(5):
    r=100*pert[b]/conv[b] if conv[b] else 0
    print(f"  B{b}: conv {conv[b]/M:8.4f}M  perturb {pert[b]/M:8.4f}M  ({r:5.2f}%)")
