#!/usr/bin/env python3
# Validate the BN-fold fix: regenerate head-only (last_layer now auto-folds BN),
# confirm BatchNormInternal is gone, eval each lr's ORT outputs.npz on batch-2.
import subprocess, numpy as np, torch, sys, onnx
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy
B2="/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2"
CKPT="/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt"
DATA="/app/SilentWear/SilentWear_data/data_raw_and_filt"
X=np.load(B2+"/inputs.npz")["input"].astype(np.float32); y=np.load(B2+"/inputs.npz")["label"].astype(np.int64)
sd=torch.load(CKPT,map_location="cpu",weights_only=False); sd=sd.get("model_state_dict",sd)
def bal(fcw,fcb):
    m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9); m.load_state_dict(sd,strict=True)
    if fcw is not None: m.fc.weight.data=torch.from_numpy(fcw.copy()); m.fc.bias.data=torch.from_numpy(fcb.copy())
    m.eval(); p=np.empty(len(y),np.int64)
    with torch.no_grad():
        for i in range(len(y)): p[i]=m(torch.from_numpy(X[i:i+1])).numpy().argmax(1)[0]
    c=sorted(set(y.tolist())); return float(np.mean([(p[y==k]==k).mean() for k in c]))
zs=bal(None,None); print(f"zero-shot batch2={100*zs:.2f}%",flush=True); print("="*72,flush=True)
def gen(lr,ep,ds,outdir):
    cmd=["python3","/app/Onnx4Deeploy/Onnx4Deeploy.py","-model","SpeechNet","-mode","train","-o",outdir,
         "--dataset","silentwear","--data-path",DATA,"--pretrained-weights",CKPT,
         "--subject","S01","--session","3","--batch","1","--condition","vocalized",
         "--stratified","--data-size",str(ds),"--n-epochs",str(ep),"--n-accum","4","--lr",str(lr),
         "--training-strategy","last_layer"]
    subprocess.run(cmd,cwd="/app/Onnx4Deeploy",stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
res=[]
for lr in [0.005,0.01,0.025,0.05,0.1]:
    od=f"/tmp/foldhead_lr_{lr}"; gen(lr,40,54,od)
    try:
        g=onnx.load(od+"/network.onnx"); ops=[n.op_type for n in g.graph.node]
        nbn=sum('BatchNorm' in o for o in ops)
        o=np.load(od+"/outputs.npz"); L=o["loss"]; acc=bal(o["fc_weight"],o["fc_bias"])
        res.append((lr,acc,acc-zs))
        print(f"lr={lr:<6} BN_ops={nbn} -> batch2={100*acc:.2f}% Δ={100*(acc-zs):+.2f}pp | loss ep1={L[:54].mean():.2f} ep40={L[-54:].mean():.2f} min={L.min():.2f}",flush=True)
    except Exception as e:
        print(f"lr={lr}: FAILED {e}",flush=True)
print("="*72,flush=True)
if res:
    b=max(res,key=lambda r:r[2]); print(f"BEST folded head-only: lr={b[0]} -> {100*b[1]:.2f}% (Δ{100*b[2]:+.2f}pp)",flush=True)
