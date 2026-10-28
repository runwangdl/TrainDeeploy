import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
L=open("logs/round1_gvsoc_train.log").read().splitlines()
rx=re.compile(r'\[loss (\d+)\] computed=([\d.eE+-]+)\s+ref=([\d.eE+-]+)')
comp=[]; ref=[]
for ln in L:
    m=rx.search(ln)
    if m: comp.append(float(m.group(2))); ref.append(float(m.group(3)))
comp=np.array(comp); ref=np.array(ref); n=len(comp)
PER=54; E=n//PER
C=comp[:E*PER].reshape(E,PER); R=ref[:E*PER].reshape(E,PER)
cm=C.mean(1); rm=R.mean(1)
drift_max=np.abs(C-R).max(1)
epx=np.arange(E)+1
onset=int(np.argmax(drift_max>1e-3))+1

plt.rcParams.update({"font.size":12})
fig,ax=plt.subplots(figsize=(9.6,5.6))
l1,=ax.plot(epx,cm,color="#1f3b63",lw=2.3,marker="o",ms=4,label="on-device loss (epoch mean)")
l2,=ax.plot(epx,rm,color="#E0913A",lw=2.0,ls="--",marker="s",ms=3.6,label="reference / ORT loss (epoch mean)")
ax.set_yscale("log")
ax.set_xlabel("epoch", fontsize=13)
ax.set_ylabel("training loss  (softmax cross-entropy)", fontsize=13)
ax.set_xlim(1,E); ax.set_xticks(np.arange(0,E+1,5)); ax.set_xticks(np.arange(1,E+1,1),minor=True)
# --- fill out the left log y-axis with explicit decimal ticks ---
ax.set_ylim(0.03, 0.6)
yt=[0.03,0.04,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5]
ax.set_yticks(yt); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:g}"))
ax.grid(True,which="major",axis="both",color="#e6e6e6"); ax.set_axisbelow(True)
# right axis: drift envelope
ax2=ax.twinx()
ax2.fill_between(epx,0,drift_max,color="#C44E52",alpha=0.16)
l4,=ax2.plot(epx,drift_max,color="#C44E52",lw=1.8,label="drift: max |device − ref| per epoch")
ax2.set_ylabel("| device − ref |", fontsize=13, color="#8f2f38")
ax2.tick_params(axis="y", colors="#8f2f38")
ax2.set_ylim(0, drift_max.max()*1.15)
# mark drift-onset position (line only, no text/arrow)
ax.axvline(onset, color="#333", ls="--", lw=1.1)
ax.set_title("On-device vs reference training loss, and their drift — S01/fold3, 40-epoch BP round", fontsize=12.5)
ax.legend(handles=[l1,l2,l4], loc="upper right", fontsize=10, framealpha=0.95)
plt.tight_layout(); plt.savefig("loss_device_vs_ref.png",dpi=150); plt.close()
print(f"onset epoch={onset} max drift={drift_max.max():.2e}")
