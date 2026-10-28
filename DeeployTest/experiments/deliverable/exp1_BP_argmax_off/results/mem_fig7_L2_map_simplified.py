import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
exec(open("/tmp/mem_common.py").read())
d=parse("memory_alloc_deeployStates.html"); cap,bL2=d["L2"]
KB=1024.0
def is_weight(n): return (bool(re.match(r'input_\d+$',n)) and n!="input_0") or any(k in n for k in ["running_","saved_","_def_","_token_","Constant"])
def is_input(n): return n=="input_0"
def is_outptr(n): return bool(re.match(r'output_\d+$',n))
def role(n):
    ln=n.lower()
    if "grad" in ln: return "grad"
    if "transpos" in ln or "_split" in ln: return "layout"
    return "act"
def short(n):
    m=re.search(r'blocks_(\d+)_blocks_\d+_\d+_([A-Za-z]+)',n)
    if not m: return None
    blk=m.group(1); op={"Relu":"ReLU","BatchNormalization":"BN"}.get(m.group(2),m.group(2))
    ln=n.lower(); k=""
    if "grad" in ln: k=" grad"
    elif "transpos" in ln: k=" layout"
    elif "_split" in ln: k=" split"
    return f"Blk{blk} {op}{k}"
col={"act":"#4C72B0","layout":"#DD8452","grad":"#C44E52"}
LABEL_MIN=60000            # only label tensors >= 60 KB
fig,ax=plt.subplots(figsize=(15,8.5))
skip=lambda n: is_weight(n) or is_input(n) or is_outptr(n)
for n,y0,y1,sz,x0,x1 in bL2:
    if skip(n): continue
    if sz<LABEL_MIN:        # everything below the label threshold -> grey, no label
        ax.add_patch(Rectangle((x0,y0/KB),x1-x0,(y1-y0)/KB,facecolor="#dcdcdc",edgecolor="#bfbfbf",lw=0.3)); continue
    r=role(n); ax.add_patch(Rectangle((x0,y0/KB),x1-x0,(y1-y0)/KB,facecolor=col[r],edgecolor="white",lw=0.6))
    ax.text((x0+x1)/2,(y0+y1)/2/KB,short(n),ha="center",va="center",fontsize=8.5,color="white",fontweight="bold")
X0=min(b[4] for b in bL2); X1=max(b[5] for b in bL2)
wsum=sum(b[3] for b in bL2 if is_weight(b[0])); isum=sum(b[3] for b in bL2 if is_input(b[0]))
ax.add_patch(Rectangle((X0,0),X1-X0,wsum/KB,facecolor="#937860",edgecolor="k",lw=0.8))
ax.text((X0+X1)/2,wsum/2/KB,f"WEIGHTS / PARAMS  ({wsum/KB:.0f} KB, 57 tensors)",ha="center",va="center",fontsize=10,color="white",fontweight="bold")
ax.add_patch(Rectangle((X0,wsum/KB),X1-X0,isum/KB,facecolor="#55A868",edgecolor="k",lw=0.8))
ax.text((X0+X1)/2,(wsum+isum/2)/KB,f"INPUT: EMG window ({isum/KB:.0f} KB)",ha="center",va="center",fontsize=9,color="white",fontweight="bold")
hw=max(b[2] for b in bL2)
ax.axhline(cap/KB,color="red",ls="--",lw=1.5); ax.text(X1,cap/KB+15,f"L2 capacity {cap/KB:.0f} KB",ha="right",color="red",fontsize=9)
ax.axhline(hw/KB,color="k",ls=":",lw=1.2); ax.text(X1,hw/KB+15,f"arena high-water {hw/KB:.0f} KB",ha="right",color="k",fontsize=9)
ax.set_xlim(X0,X1); ax.set_ylim(0,cap/KB*1.03)
ax.set_xlabel("execution step  (forward → backward → optimizer)"); ax.set_ylabel("L2 memory offset (KB)")
ax.set_title("L2 memory allocation map — simplified  (exp1 BP, argmax-off, single step)\nbottom merged to Weights+Input; only tensors ≥60 KB labelled, all smaller buffers greyed",fontsize=12.5)
ax.legend(handles=[Patch(facecolor=col["act"],label="fwd activation"),Patch(facecolor=col["layout"],label="fwd layout-copy"),
                   Patch(facecolor=col["grad"],label="backward gradient"),Patch(facecolor="#937860",label="weights/params"),
                   Patch(facecolor="#55A868",label="input (EMG)"),Patch(facecolor="#dcdcdc",label="smaller buffers (<60 KB)")],
          loc="upper center",ncol=6,fontsize=9,frameon=True,bbox_to_anchor=(0.5,-0.07))
plt.tight_layout(rect=[0,0.03,1,1]); plt.savefig("mem_fig7_L2_map_simplified.png",dpi=140); plt.close()
print("fig7 re-rendered; labelled tensors >=60KB:",sum(1 for b in bL2 if not skip(b[0]) and b[3]>=LABEL_MIN))
