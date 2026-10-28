import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
from matplotlib.patches import Patch
exec(open("/tmp/mem_common.py").read())
d=parse("memory_alloc_deeployStates.html"); cap,bL2=d["L2"]
psum,tp,live=peaksum(bL2)
def block_of(n):
    m=re.search(r'blocks_(\d+)_',n)
    if m: return f"Block {m.group(1)}"
    if any(k in n for k in ["Gemm","fc","Softmax","ReduceSum","Reshape","Flatten"]) and not n.startswith(("input","output")): return "Head / FC"
    return "Input / data"
def op_of(n):
    m=re.search(r'blocks_\d+_blocks_\d+_\d+_([A-Za-z]+)',n)
    op=m.group(1) if m else ("Gemm" if ("Gemm" in n or "fc" in n) else "Softmax" if "Softmax" in n else "ReduceSum" if "ReduceSum" in n else "Reshape" if "Reshape" in n else "Input/data")
    return {"Relu":"ReLU","BatchNormalization":"BatchNorm","GlobalAveragePool":"GlobalAvgPool","AveragePool":"AvgPool"}.get(op,op)
agg={}
for b in live: agg[(block_of(b[0]),op_of(b[0]))]=agg.get((block_of(b[0]),op_of(b[0])),0)+b[3]
rows=["Block 0","Block 1","Block 2","Block 3","Block 4","Input / data"]
ops=["Conv","BatchNorm","ReLU","MaxPool","Gemm","Softmax","Input/data"]
col={"Conv":"#4C72B0","BatchNorm":"#DD8452","ReLU":"#55A868","MaxPool":"#C44E52",
     "Gemm":"#8172B3","Softmax":"#937860","Input/data":"#8c8c8c"}
lbl={"Input/data":"Input / weights / data"}
fig,ax=plt.subplots(figsize=(10,4.6))
y=np.arange(len(rows))[::-1]
for i,bl in zip(y,rows):
    left=0
    for o in ops:
        v=agg.get((bl,o),0)/1024
        if v<=0: continue
        ax.barh(i,v,left=left,color=col[o],edgecolor="white",height=0.62)
        if v>25: ax.text(left+v/2,i,f"{o}\n{v:,.0f}KB",ha="center",va="center",color="white",fontsize=8.5,fontweight="bold")
        left+=v
    tot=sum(agg.get((bl,o),0) for o in ops)/1024
    ax.text(left+8,i,f"{tot:,.0f} KB · {100*tot*1024/psum:.0f}%",va="center",fontsize=9.5,fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(rows)
ax.set_xlabel("KB live in L2 at peak step"); ax.set_xlim(0,1080)
ax.set_title(f"L2 peak memory — per block, by operator  (peak {psum/1024:,.0f} KB live @ step {tp})\nSpeechNet BP baseline, single training step (S01/fold3)")
seen=[o for o in ops if any(agg.get((bl,o),0)>0 for bl in rows)]
ax.legend(handles=[Patch(facecolor=col[o],label=lbl.get(o,o)) for o in seen],loc="lower right",fontsize=9,frameon=True,ncol=2)
plt.tight_layout(); plt.savefig("mem_fig5_peak_per_block_by_operator.png",dpi=150); plt.close()
print("fig5 written")
