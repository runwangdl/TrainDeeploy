import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
# cycles per block (Total) from trace  (same source as lat_fig5_conv_vs_dxdw.py)
fwd=[7405374,6121186,922827,175772,90059]
dX =[0,3402803,954448,434761,203149]
dW =[1093044,2642291,782108,181157,146955]
# ConvGrad = dX + dW  (one combined backward bar per block)
grad=[dX[i]+dW[i] for i in range(5)]
M=1e6; x=np.arange(5); w=0.38
fig,ax=plt.subplots(figsize=(10.5,5.2))
ax.bar(x-w/2,[v/M for v in fwd ],w,color="#4C72B0",label="Conv (forward)")
ax.bar(x+w/2,[v/M for v in grad],w,color="#C44E52",label="ConvGrad (dX + dW)")
for i in range(5):
    ax.text(i-w/2,fwd[i]/M+0.12,f"{fwd[i]/M:.2f}",ha="center",fontsize=8)
    ax.text(i+w/2,grad[i]/M+0.12,f"{grad[i]/M:.2f}\n{grad[i]/fwd[i]:.1f}×",ha="center",fontsize=8)
    if dX[i]==0: ax.text(i+w/2,0.15,"dX\nskipped",ha="center",fontsize=7,color="white")
ax.set_xticks(x); ax.set_xticklabels([f"B{b}" for b in range(5)])
ax.set_ylabel("Mcycles per kernel"); ax.set_ylim(0,8.2)
ax.set_title("Conv vs ConvGrad (dX + dW) per block")
ax.legend()
plt.tight_layout(); plt.savefig("lat_fig6_conv_vs_convgrad.png",dpi=150); plt.close(); print("fig6 written")
