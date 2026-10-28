import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
from collections import defaultdict
L=open("../logs/profiletiling.log").read().splitlines()
node=defaultdict(lambda:[0,0,0])
rx=re.compile(r'^\[([^\]]+)\]\[[SD]B\]\[\d+ ops\]\[Tile \d+\] (Pre-Kernel|Kernel|Post-Kernel)\s*:\s*(\d+) cycles')
for ln in L:
    m=rx.match(ln)
    if not m: continue
    idx={"Pre-Kernel":0,"Kernel":1,"Post-Kernel":2}[m.group(2)]; node[m.group(1)][idx]+=int(m.group(3))
def blk(t):
    m=re.search(r'blocks_(\d+)_',t); return int(m.group(1)) if m else -1
fwd=defaultdict(int); bwd=defaultdict(int)
for tag,(p,k,po) in node.items():
    if "transpose" in tag.lower(): continue
    b=blk(tag); tot=p+k+po; g=("grad" in tag.lower() or "Grad" in tag)
    if "_Conv_Conv" in tag and not g: fwd[b]+=tot
    if ("ConvGrad" in tag or "GradConv" in tag): bwd[b]+=tot
M=1e6; blocks=list(range(5))
f=[fwd[b]/M for b in blocks]; g=[bwd[b]/M for b in blocks]
x=np.arange(5); w=0.38
fig,ax=plt.subplots(figsize=(9.5,5))
ax.bar(x-w/2,f,w,color="#4C72B0",label="Forward Conv")
ax.bar(x+w/2,g,w,color="#C44E52",label="ConvGrad (backward)")
for i in blocks:
    r=g[i]/f[i] if f[i] else 0
    ax.text(i, max(f[i],g[i])+0.25, f"{r:.2f}×", ha="center", fontsize=11, fontweight="bold")
ax.axhline(0)
notes=["B0: first layer\ndW only (no dX)","B1","B2\n≈2× (textbook)","B3","B4"]
ax.set_xticks(x); ax.set_xticklabels([f"Block {i}" for i in blocks])
ax.set_ylabel("Mcycles"); ax.set_ylim(0,8.6)
ax.set_title("Per-block: forward Conv vs ConvGrad (backward)\nratio on top — 2× holds for deep blocks, but the dominant B0/B1 run 0.20× / 1.0×")
ax.legend(loc="upper right")
ax.annotate("first-layer dX skipped\n→ biggest fwd, tiny bwd", xy=(0.19,1.46),xytext=(0.9,4.2),
            arrowprops=dict(arrowstyle="->"),fontsize=9)
plt.tight_layout(); plt.savefig("lat_fig3_perblock_conv.png",dpi=150); plt.close()
print("fig3 written")
