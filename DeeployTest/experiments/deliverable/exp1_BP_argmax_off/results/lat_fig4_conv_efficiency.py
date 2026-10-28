import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
fcpm={0:23.61,1:2.45,2:1.53,3:0.70,4:0.18}; bcpm={0:4.65,1:1.23,2:1.46,3:1.24,4:0.35}
x=np.arange(5); w=0.38
fig,ax=plt.subplots(figsize=(9.5,5))
ax.bar(x-w/2,[fcpm[b] for b in range(5)],w,color="#4C72B0",label="Forward conv")
ax.bar(x+w/2,[bcpm[b] for b in range(5)],w,color="#C44E52",label="Backward conv (dX+dW)")
ax.set_yscale("log")
for b in range(5):
    ax.text(b-w/2,fcpm[b]*1.1,f"{fcpm[b]:.1f}",ha="center",fontsize=9)
    ax.text(b+w/2,bcpm[b]*1.1,f"{bcpm[b]:.2f}",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"B{b}" for b in range(5)])
ax.set_ylabel("cycles per MAC  (lower = more efficient, log)")
ax.set_title("Per-block conv efficiency — forward swings 130×, backward stays flat\nforward inefficient at B0/B1 (few ch, big spatial) → ratio<2×; efficient at B3/B4 → ratio>2×")
ax.axhspan(0,1,color="green",alpha=0.05); ax.legend()
ax.annotate("B0: 1 in-ch, 14×700\nreduction depth=4 → 4.5% FPU util",xy=(0-w/2,23.6),xytext=(0.7,9),arrowprops=dict(arrowstyle="->"),fontsize=8.5)
ax.annotate("B4: 32×32 ch, 14×5\ndeep reduction → efficient fwd",xy=(4-w/2,0.18),xytext=(2.4,0.09),arrowprops=dict(arrowstyle="->"),fontsize=8.5)
plt.tight_layout(); plt.savefig("lat_fig4_conv_efficiency.png",dpi=150); plt.close(); print("fig4 written")
