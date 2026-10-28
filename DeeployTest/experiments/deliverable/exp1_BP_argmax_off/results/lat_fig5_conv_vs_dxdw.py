import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
# cycles per block (Total) from trace
fwd=[7405374,6121186,922827,175772,90059]
dX =[0,3402803,954448,434761,203149]
dW =[1093044,2642291,782108,181157,146955]
M=1e6; x=np.arange(5); w=0.26
fig,ax=plt.subplots(figsize=(10.5,5.2))
ax.bar(x-w,[v/M for v in fwd],w,color="#4C72B0",label="Conv (forward)")
ax.bar(x,  [v/M for v in dX ],w,color="#C44E52",label="ConvGradX (dX)")
ax.bar(x+w,[v/M for v in dW ],w,color="#DD8452",label="ConvGradW (dW)")
for i in range(5):
    if dX[i]==0: ax.text(i,0.15,"dX\nskipped",ha="center",fontsize=8,color="#C44E52")
    ax.text(i-w,fwd[i]/M+0.12,f"{fwd[i]/M:.2f}",ha="center",fontsize=8)
    if dX[i]: ax.text(i,dX[i]/M+0.12,f"{dX[i]/M:.2f}\n{dX[i]/fwd[i]:.1f}×",ha="center",fontsize=8)
    ax.text(i+w,dW[i]/M+0.12,f"{dW[i]/M:.2f}\n{dW[i]/fwd[i]:.1f}×",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"B{b}" for b in range(5)])
ax.set_ylabel("Mcycles per kernel"); ax.set_ylim(0,8.2)
ax.set_title("Conv vs its gradient kernels (dX, dW) per block")
ax.legend()
plt.tight_layout(); plt.savefig("lat_fig5_conv_vs_dxdw.png",dpi=150); plt.close(); print("fig5 written")
