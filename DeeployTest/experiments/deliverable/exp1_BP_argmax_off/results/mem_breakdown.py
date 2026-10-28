import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def parse(fn):
    html=open(fn).read(); out={}
    for seg in html.split("var fig =")[1:]:
        ml=re.search(r'"text":"(L[0-9]) Memory Size","x":\[[-\d.,]+\],"y":\[(\d+)',seg)
        if not ml: continue
        lvl=ml.group(1); cap=int(ml.group(2)); blocks=[]
        for m in re.finditer(r'"text":"([^"]+)","x":\[([-\d.,]+)\],"y":\[([-\d.,]+)\]',seg):
            n=m.group(1)
            if n.endswith("Memory Size"): continue
            xs=[float(v) for v in m.group(2).split(",")]; ys=[float(v) for v in m.group(3).split(",")]
            blocks.append((n,min(ys),max(ys),max(ys)-min(ys),min(xs),max(xs)))
        out[lvl]=(cap,blocks)
    return out

def cat(n):
    ln=n.lower()
    if "running_mean" in n or "running_var" in n: return "Frozen BN stats"
    if re.search(r'(weight|bias|_gamma|_beta)_tensor$',n) and "grad" not in ln: return "Weights (params)"
    if "grad" in ln: return "Gradients (backward)"
    if any(k in ln for k in ["_transposed","pre_transpose","transpose_in"]): return "Layout-transpose copies"
    if any(k in ln for k in ["buffer","im2col","_split"]): return "Workspace/scratch"
    if any(k in n for k in ["input","label","Loss","SoftmaxCrossEntropy","ReduceSum"]): return "I/O + loss"
    return "Fwd activations"

def block_of(n):
    m=re.search(r'blocks_(\d+)_',n)
    if m: return f"Block {m.group(1)}"
    if any(k in n for k in ["Gemm","fc","Softmax","ReduceSum","Reshape","Flatten"]): return "Head (FC/loss)"
    return "Other"

def highwater(blocks): return max(b[2] for b in blocks)
def peaksum(blocks):
    tmin=int(min(b[4] for b in blocks)); tmax=int(max(b[5] for b in blocks)); best=(0,None,[])
    for t in range(tmin,tmax+1):
        live=[b for b in blocks if b[4]<=t<=b[5]]; s=sum(b[3] for b in live)
        if s>best[0]: best=(s,t,live)
    return best

d=parse("memory_alloc_deeployStates.html")
capL1,bL1=d["L1"]; capL2,bL2=d["L2"]
hwL1=highwater(bL1); hwL2=highwater(bL2)
psumL2,tpeak,liveL2=peaksum(bL2)
roles={}
for b in liveL2: roles[cat(b[0])]=roles.get(cat(b[0]),0)+b[3]
blkfwd={}
for b in bL2:
    if cat(b[0]) in ("Fwd activations","Layout-transpose copies"):
        blkfwd[block_of(b[0])]=blkfwd.get(block_of(b[0]),0)+b[3]

print("=== HIGH-WATER (arena that must fit) ===")
print(f"L1: {hwL1:,.0f} / {capL1:,} B = {100*hwL1/capL1:.1f}%   (BINDING)")
print(f"L2: {hwL2:,.0f} / {capL2:,} B = {100*hwL2/capL2:.1f}%")
print(f"\n=== L2 peak-simultaneous live = {psumL2:,.0f} B @ t={tpeak} (arena {hwL2:,.0f}, frag {100*(hwL2-psumL2)/hwL2:.1f}%) ===")
for k,v in sorted(roles.items(),key=lambda z:-z[1]):
    print(f"   {k:24s} {v:>10,.0f} B  {100*v/psumL2:5.1f}%")
print("\n=== Per-block activation+layout footprint (L2) ===")
tot=sum(blkfwd.values())
for k,v in sorted(blkfwd.items(),key=lambda z:-z[1]):
    print(f"   {k:14s} {v:>10,.0f} B  {100*v/tot:5.1f}%")

C={"Fwd activations":"#4C72B0","Layout-transpose copies":"#DD8452","Workspace/scratch":"#CCB974",
   "I/O + loss":"#55A868","Gradients (backward)":"#C44E52","Frozen BN stats":"#8172B3","Weights (params)":"#937860"}
plt.rcParams.update({"font.size":12})

fig,ax=plt.subplots(figsize=(6,4.4))
levels=["L1\n(128 KB)","L2\n(2 MB)"]; used=[hwL1/1024,hwL2/1024]; capk=[capL1/1024,capL2/1024]
x=np.arange(2)
ax.bar(x,capk,width=0.55,color="#ececec",edgecolor="#999",label="Capacity")
ax.bar(x,used,width=0.55,color=["#C44E52","#4C72B0"],label="Peak used")
for i,(u,c) in enumerate(zip(used,capk)):
    ax.text(i,u*1.05,f"{u:,.0f} KB\n{100*u/c:.1f}%",ha="center",va="bottom",fontsize=11,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(levels); ax.set_ylabel("KB (log)"); ax.set_yscale("log"); ax.set_ylim(1,3000)
ax.set_title("Peak memory vs capacity per level\nBP baseline, single training step (S01/fold3)")
ax.legend(loc="upper left"); plt.tight_layout(); plt.savefig("mem_fig1_peak_vs_capacity.png",dpi=150); plt.close()

fig,ax=plt.subplots(figsize=(9,2.8)); left=0
for k,v in sorted(roles.items(),key=lambda z:-z[1]):
    ax.barh(0,v/1024,left=left/1024,color=C.get(k,"#888"),edgecolor="white")
    if v/psumL2>0.028:
        ax.text((left+v/2)/1024,0,f"{k}\n{v/1024:,.0f} KB ({100*v/psumL2:.0f}%)",ha="center",va="center",fontsize=8.5,color="white",fontweight="bold")
    left+=v
ax.set_xlim(0,psumL2/1024); ax.set_yticks([]); ax.set_xlabel("KB")
ax.set_title(f"L2 peak composition by training role  (peak live {psumL2/1024:,.0f} KB @ step {tpeak}; arena {hwL2/1024:,.0f} KB)")
plt.tight_layout(); plt.savefig("mem_fig2_L2_role_composition.png",dpi=150); plt.close()

fig,ax=plt.subplots(figsize=(7,4.4))
order=["Block 0","Block 1","Block 2","Block 3","Block 4","Head (FC/loss)","Other"]
items=[(k,blkfwd[k]) for k in order if k in blkfwd]
names=[k for k,_ in items]; vals=[v/1024 for _,v in items]
ax.bar(names,vals,color="#4C72B0",edgecolor="#2f4a70")
for i,v in enumerate(vals): ax.text(i,v+max(vals)*0.01,f"{v:,.0f}",ha="center",va="bottom",fontsize=10)
ax.set_ylabel("KB (activation + layout buffers, L2)")
ax.set_title("Activation memory by network block (L2)\nearly blocks dominate: large spatial dims, few channels")
plt.xticks(rotation=18,ha="right"); plt.tight_layout(); plt.savefig("mem_fig3_perblock_activations.png",dpi=150); plt.close()
print("\nWROTE 3 figures.")
