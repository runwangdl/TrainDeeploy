import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
from matplotlib.patches import Patch
exec(open("/tmp/mem_common.py").read())
d=parse("memory_alloc_deeployStates.html"); cap,bL2=d["L2"]
def block_of(n):
    m=re.search(r'blocks_(\d+)_',n)
    return f"Block {m.group(1)}" if m else "Input / data"
def role3(n):
    ln=n.lower()
    if re.search(r"__\d+_grad", n): return "backward-gradient"
    if any(k in ln for k in ["_transposed","pre_transpose","transpose_in","_split"]): return "fwd-transpose"
    if n.startswith(("input","output")) or "_def_" in n or "_token_" in n or "running_" in n or "Constant" in n: return "input/weights"
    return "fwd-activation"
OPSEQ={"Conv":0,"BN":1,"ReLU":2,"MaxPool":3}
def identity(n):
    if re.search(r'input_\d+_transposed', n): return "weight_T","Conv"     # conv weight, transposed
    m=re.search(r'node_0_blocks_\d+_blocks_\d+_\d+_([A-Za-z]+)__0', n)
    if not m: return None,None
    src={"Relu":"ReLU","BatchNormalization":"BN"}.get(m.group(1),m.group(1))
    ln=n.lower()
    if re.search(r"__\d+_grad",n): suf="_grad"
    elif any(k in ln for k in ["_transposed","pre_transpose","transpose_in"]): suf="_T"
    elif "_split" in ln: suf="_split"
    else: suf=""
    return f"{src}{suf}", src
col={"fwd-activation":"#4C72B0","fwd-transpose":"#DD8452","backward-gradient":"#C44E52","input/weights":"#9a9a9a"}
rord=["fwd-activation","fwd-transpose","backward-gradient","input/weights"]
t=24
live=[b for b in bL2 if b[4]<=t<=b[5]]; ptot=sum(b[3] for b in live)
rows=["Block 0","Block 1","Block 2","Block 3","Block 4","Input / data"]
# aggregate by (block, label); track role + order
agg={}
for b in live:
    bl=block_of(b[0])
    if bl=="Input / data":
        lab="inputs+weights"; src="Z"; r="input/weights"
    else:
        lab,src=identity(b[0]); r=role3(b[0])
        if lab is None: lab="misc"; src="Z"
    agg.setdefault(bl,{}).setdefault(lab,[0,r,src])
    agg[bl][lab][0]+=b[3]

fig,ax=plt.subplots(figsize=(11.5,4.9))
XMAX=1080.0
kb_per_in=XMAX/(fig.get_size_inches()[0]*0.82)
def fits(lab,width_kb):
    return width_kb/kb_per_in >= (len(lab)+0.5)*0.083
y=np.arange(len(rows))[::-1]
for i,bl in zip(y,rows):
    segs=agg.get(bl,{})
    order=sorted(segs.items(),key=lambda kv:(OPSEQ.get(kv[1][2],9), "_T" in kv[0], "_grad" in kv[0], -kv[1][0]))
    left=0
    for lab,(s,r,src) in order:
        v=s/1024
        if v<=0.3: left+=v; continue
        ax.barh(i,v,left=left,color=col[r],edgecolor="white",height=0.62)
        if fits(lab,v):
            ax.text(left+v/2,i,f"{lab}\n{v:,.0f} KB",ha="center",va="center",color="white",fontsize=9,fontweight="bold")
        left+=v
    if left>1: ax.text(left+8,i,f"{left:,.0f} KB",va="center",fontsize=10,fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(rows,fontsize=11)
ax.set_xlabel("KB live in L2 at forward peak (step 24)   ·   _T = transposed (layout copy)",fontsize=10.5)
ax.set_xlim(0,XMAX)
present=[r for r in rord if any(v[1]==r for bl in agg.values() for v in bl.values())]
ax.legend(handles=[Patch(facecolor=col[r],label=r) for r in present],
          loc="lower center",bbox_to_anchor=(0.5,-0.30),ncol=4,fontsize=10,frameon=True)
plt.tight_layout(rect=[0,0.05,1,1]); plt.savefig("mem_fig8_forward_peak_by_operator.png",dpi=150,bbox_inches="tight"); plt.close()
print("fig8 written; kb_per_in≈%.0f"%kb_per_in)
for bl in rows:
    print(bl, {k:round(v[0]/1024) for k,v in agg.get(bl,{}).items()})
