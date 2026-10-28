import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
from matplotlib.patches import Patch
exec(open("/tmp/mem_common.py").read())
d=parse("memory_alloc_deeployStates.html"); cap,bL2=d["L2"]
def block_of(n):
    m=re.search(r'blocks_(\d+)_',n)
    if m: return f"Block {m.group(1)}"
    return "Input / data"
def op_of(n):
    m=re.search(r'blocks_\d+_blocks_\d+_\d+_([A-Za-z]+)',n)
    op=m.group(1) if m else "-"
    return {"Relu":"ReLU","BatchNormalization":"BatchNorm"}.get(op,op)
def role3(n):
    ln=n.lower()
    if re.search(r"__\d+_grad", n): return "backward-gradient"
    if any(k in ln for k in ["_transposed","pre_transpose","transpose_in","_split"]): return "fwd-transpose"
    if n.startswith(("input","output")) or "_def_" in n or "_token_" in n or "running_" in n or "Constant" in n: return "input/weights"
    return "fwd-activation"
col={"fwd-activation":"#4C72B0","fwd-transpose":"#DD8452","backward-gradient":"#C44E52","input/weights":"#9a9a9a"}
rord=["fwd-activation","fwd-transpose","backward-gradient","input/weights"]
def live_at(t): return [b for b in bL2 if b[4]<=t<=b[5]]

peaks=[("Forward peak  (step 24)",24),("Backward peak  (step 94)",94)]
rows=["Block 0","Block 1","Block 2","Block 3","Block 4","Input / data"]
fig,axes=plt.subplots(2,1,figsize=(11,8.2),sharex=True)
for ax,(title,t) in zip(axes,peaks):
    live=live_at(t); ptot=sum(b[3] for b in live)
    # build per (block,op,role)
    agg={}
    for b in live:
        agg.setdefault(block_of(b[0]),[]).append((op_of(b[0]),role3(b[0]),b[3]))
    y=np.arange(len(rows))[::-1]
    for i,bl in zip(y,rows):
        segs=agg.get(bl,[])
        # merge same (op,role)
        mm={}
        for op,r,s in segs: mm[(op,r)]=mm.get((op,r),0)+s
        items=sorted(mm.items(),key=lambda z:(rord.index(z[0][1]),-z[1]))
        left=0
        for (op,r),s in items:
            v=s/1024
            if v<=0.3: left+=v; continue
            ax.barh(i,v,left=left,color=col[r],edgecolor="white",height=0.6)
            if v>22:
                lab=op if bl!="Input / data" else "input+weights"
                ax.text(left+v/2,i,f"{lab}\n{v:,.0f}KB",ha="center",va="center",color="white",fontsize=8,fontweight="bold")
            left+=v
        if left>1: ax.text(left+8,i,f"{left:,.0f} KB",va="center",fontsize=9,fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(rows)
    ax.set_title(f"{title} — {ptot/1024:,.0f} KB live",fontsize=12,fontweight="bold")
    ax.set_xlim(0,1080)
axes[1].set_xlabel("KB live in L2")
fig.suptitle("L2 peak memory per block × operator, split forward vs backward\nSpeechNet BP baseline, single step (S01/fold3)  ·  colour = data role, label = operator",fontsize=12.5)
leg=[Patch(facecolor=col[r],label=r) for r in rord]
fig.legend(handles=leg,loc="lower center",fontsize=10,frameon=True,ncol=4,bbox_to_anchor=(0.5,-0.01))
plt.tight_layout(rect=[0,0.04,1,0.95]); plt.savefig("mem_fig6_peak_fwd_bwd_by_operator.png",dpi=150); plt.close()
print("fig6 written")
