#!/usr/bin/env python3
# Head-only epoch curve: find min epochs to clear +3pp and check >80 convergence.
# Trajectory (mean over seeds) of batch-2 acc vs epoch. Head-only => frozen features
# => non-compounding precision drift (verified separately on-device).
import numpy as np, torch, torch.nn as nn, sys
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy
B1="/app/TrainDeeploy/DeeployTest/Tests/Models/speechnet_infer_original"
B2="/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2"
CKPT="/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt"
def load(d):
    z=np.load(d+"/inputs.npz"); return z["input"].astype(np.float32), z["label"].astype(np.int64)
def newmodel():
    m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9)
    ck=torch.load(CKPT,map_location="cpu",weights_only=False); m.load_state_dict(ck.get("model_state_dict",ck),strict=True); return m
def bal(model,X,y):
    model.eval(); p=np.empty(len(y),np.int64)
    with torch.no_grad():
        for i in range(len(y)): p[i]=model(torch.from_numpy(X[i:i+1])).numpy().argmax(1)[0]
    c=sorted(set(y.tolist())); return float(np.mean([(p[y==k]==k).mean() for k in c]))
def part(y,pcv=6,seed=0):
    rng=np.random.RandomState(seed); v=[];p=[]
    for c in sorted(set(y.tolist())):
        idx=np.where(y==c)[0].copy(); rng.shuffle(idx); v+=idx[:pcv].tolist(); p+=idx[pcv:].tolist()
    return np.array(p),np.array(v)
def samp(y,pool,pc,seed):
    rng=np.random.RandomState(seed); s=[]
    for c in sorted(set(y.tolist())):
        z=pool[y[pool]==c].copy(); rng.shuffle(z); s+=z[:pc].tolist()
    return np.array(s)
def main():
    Xb1,yb1=load(B1); Xb2,yb2=load(B2); pool,_=part(yb1,6,0)
    zs=bal(newmodel(),Xb2,yb2); print(f"zero-shot batch2={100*zs:.2f}%",flush=True)
    MARKS=[10,20,30,40,60,80,120,160]
    CFGS=[("30%",6,0.01,4),("20%",4,0.01,4),("30%",6,0.005,4)]
    for dn,pc,lr,K in CFGS:
        traj={e:[] for e in MARKS}
        for sd in [0,1,2]:
            torch.manual_seed(sd); tr=samp(yb1,pool,pc,sd); Xtr,ytr=Xb1[tr],yb1[tr]
            m=newmodel()
            for p in m.parameters(): p.requires_grad=False
            for p in m.fc.parameters(): p.requires_grad=True
            opt=torch.optim.SGD(m.fc.parameters(),lr=lr); lf=nn.CrossEntropyLoss()
            rng=np.random.RandomState(1000+sd); order=np.arange(len(tr))
            for ep in range(1,max(MARKS)+1):
                m.eval(); rng.shuffle(order)
                for s in range(0,len(order),K):
                    opt.zero_grad(); b=order[s:s+K]
                    loss=sum(lf(m(torch.from_numpy(Xtr[i:i+1])),torch.from_numpy(ytr[i:i+1])) for i in b)/len(b)
                    loss.backward(); opt.step()
                if ep in MARKS: traj[ep].append(bal(m,Xb2,yb2))
        line=" ".join(f"e{e}:{100*np.mean(traj[e]):.1f}(Δ{100*(np.mean(traj[e])-zs):+.1f})" for e in MARKS)
        print(f"head data={dn}({pc*9}w) lr={lr} K={K} | {line}",flush=True)
if __name__=="__main__": main()
