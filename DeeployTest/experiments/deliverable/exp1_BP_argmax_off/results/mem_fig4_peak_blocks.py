import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
from matplotlib.patches import Patch
exec(open("/tmp/mem_common.py").read())
def block_of(n):
    m=re.search(r'blocks_(\d+)_',n)
    if m: return f"Block {m.group(1)}"
    if any(k in n for k in ["Gemm","fc","Softmax","SoftmaxCrossEntropy","ReduceSum","Reshape","Flatten"]): return "Head/FC"
    if n.startswith("input") or n.startswith("output") or "_def_" in n or "_token_" in n or "Constant" in n: return "Data/graph I-O"
    return "Data/graph I-O"
d=parse("memory_alloc_deeployStates.html"); cap,bL2=d["L2"]
psum,tp,live=peaksum(bL2)
order=["Block 0","Block 1","Block 2","Block 3","Block 4","Head/FC","Data/graph I-O"]
roles_order=["Fwd activations","Layout-transpose copies","Gradients (backward)","I/O + loss","Frozen BN stats"]
C={"Fwd activations":"#4C72B0","Layout-transpose copies":"#DD8452","Gradients (backward)":"#C44E52",
   "I/O + loss":"#55A868","Frozen BN stats":"#8172B3"}
M={b:{r:0 for r in roles_order} for b in order}
for b in live:
    M[block_of(b[0])][cat(b[0])]+=b[3]
tot={b:sum(M[b].values()) for b in order}

fig,ax=plt.subplots(figsize=(8.5,5))
x=np.arange(len(order)); bottom=np.zeros(len(order))
for r in roles_order:
    vals=np.array([M[b][r]/1024 for b in order])
    ax.bar(x,vals,bottom=bottom,color=C[r],edgecolor="white",width=0.7)
    bottom+=vals
for i,b in enumerate(order):
    if tot[b]/1024>3:
        ax.text(i,tot[b]/1024+8,f"{tot[b]/1024:,.0f} KB\n{100*tot[b]/psum:.0f}%",ha="center",va="bottom",fontsize=10,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(order,rotation=18,ha="right")
ax.set_ylabel("KB live at peak step"); ax.set_ylim(0,1120)
ax.set_title(f"What's resident at the L2 peak (step {tp}, {psum/1024:,.0f} KB live)\nblocks present × role — the full forward stack is co-resident (activation stashing)")
ax.legend(handles=[Patch(facecolor=C[r],label=r) for r in roles_order],loc="upper right",fontsize=9,frameon=True)
plt.tight_layout(); plt.savefig("mem_fig4_peak_blocks.png",dpi=150); plt.close()
print("=== per-block totals at peak ===")
for b in order: print(f"  {b:16s} {tot[b]/1024:7.1f} KB  {100*tot[b]/psum:5.1f}%")
print("fig4 written")
