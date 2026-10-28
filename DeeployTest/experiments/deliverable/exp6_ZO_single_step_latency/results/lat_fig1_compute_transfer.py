# ZO analog of exp1 BP lat_fig1_compute_transfer.py — one MeZO step (2 perturbed forwards + zo_update).
# Single stacked bar: Compute (Kernel) / DMA-in (Pre-Kernel) / DMA-out (Post-Kernel), summed over all tiles.
import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
L=open("../logs/profiletiling.log").read().splitlines()
tot=[0,0,0]
rx=re.compile(r'^\[[^\]]+\]\[[SD]B\]\[\d+ ops\]\[Tile \d+\] (Pre-Kernel|Kernel|Post-Kernel)\s*:\s*(\d+) cycles')
for ln in L:
    m=rx.match(ln)
    if m: tot[{"Pre-Kernel":0,"Kernel":1,"Post-Kernel":2}[m.group(1)]]+=int(m.group(2))
M=1e6; T=sum(tot)
plt.rcParams.update({"font.size":13,"axes.edgecolor":"#444","axes.linewidth":0.9})
fig,ax=plt.subplots(figsize=(7.6,5.4))
c_comp,c_in,c_out="#3B6EA5","#E0913A","#B44A4A"
ax.bar(0, tot[1]/M, width=.42, color=c_comp, edgecolor="white", linewidth=.8)
ax.bar(0, tot[0]/M, bottom=tot[1]/M, width=.42, color=c_in, edgecolor="white", linewidth=.8)
ax.bar(0, tot[2]/M, bottom=(tot[1]+tot[0])/M, width=.42, color=c_out, edgecolor="white", linewidth=.8)
ax.text(0, tot[1]/2/M, f"Compute\n{100*tot[1]/T:.1f}%", ha="center",va="center",color="white",fontweight="bold")
ax.text(.28, (tot[1]+(tot[0]+tot[2])/2)/M, f"Transfer {100*(tot[0]+tot[2])/T:.1f}%", ha="left",va="center",fontsize=11)
ax.set_xticks([0]); ax.set_xticklabels(["one ZO step\n(2 forwards + update)"]); ax.set_xlim(-.55,.8)
ax.set_ylabel("Cycles  (millions)"); ax.set_ylim(0,40)
ax.yaxis.grid(True, color="#e6e6e6", linewidth=.8); ax.set_axisbelow(True)
ax.set_title("Compute vs transfer — ZO (MeZO)", fontsize=15, fontweight="bold", pad=12)
ax.legend(handles=[Patch(fc=c_comp,label="Compute (kernel)"),
                   Patch(fc=c_in,label="DMA-in  (L2->L1)"),
                   Patch(fc=c_out,label="DMA-out (L1->L2)")],
          loc="center left", bbox_to_anchor=(1.02,0.5), frameon=False, fontsize=12)
plt.tight_layout(); plt.savefig("lat_fig1_compute_transfer.png",dpi=150,bbox_inches="tight"); plt.close()
print(f"saved lat_fig1_compute_transfer.png  (total {T/M:.2f}M, compute {100*tot[1]/T:.1f}%, transfer {100*(tot[0]+tot[2])/T:.1f}%)")
