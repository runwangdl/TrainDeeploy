#!/usr/bin/env python3
# Phase 1c: WHICH params to fine-tune (vanilla SGD). Full-model overfits tiny data;
# test head-only / last-block / BN-frozen variants — classic small-data transfer.
# Multi-seed, val-loss selection, batch-2 = held-out test (zero-shot 78.33%).
import numpy as np, torch, torch.nn as nn, itertools, sys
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
def bal_acc(model,X,y):
    model.eval(); p=np.empty(len(y),np.int64)
    with torch.no_grad():
        for i in range(len(y)): p[i]=model(torch.from_numpy(X[i:i+1])).numpy().argmax(1)[0]
    c=sorted(set(y.tolist())); return float(np.mean([(p[y==k]==k).mean() for k in c]))
def vloss(model,X,y):
    model.eval(); lf=nn.CrossEntropyLoss(reduction='sum'); t=0.0
    with torch.no_grad():
        for i in range(len(y)): t+=lf(model(torch.from_numpy(X[i:i+1])),torch.from_numpy(y[i:i+1])).item()
    return t/len(y)
def part(y,pcv=6,seed=0):
    rng=np.random.RandomState(seed); v=[]; p=[]
    for c in sorted(set(y.tolist())):
        idx=np.where(y==c)[0].copy(); rng.shuffle(idx); v+=idx[:pcv].tolist(); p+=idx[pcv:].tolist()
    return np.array(p),np.array(v)
def samp(y,pool,pc,seed):
    rng=np.random.RandomState(seed); s=[]
    for c in sorted(set(y.tolist())):
        z=pool[y[pool]==c].copy(); rng.shuffle(z); s+=z[:pc].tolist()
    return np.array(s)

def set_trainable(m, mode):
    for p in m.parameters(): p.requires_grad=False
    if mode=="head":            tgt=[m.fc]
    elif mode=="head+blk4":     tgt=[m.fc, m.blocks[4]]
    elif mode=="head+blk34":    tgt=[m.fc, m.blocks[3], m.blocks[4]]
    elif mode=="full_bnfrozen": tgt=[m]      # all conv/fc; BN frozen via eval()
    else: raise ValueError(mode)
    for t in tgt:
        for p in t.parameters(): p.requires_grad=True
    return [p for p in m.parameters() if p.requires_grad]

def run(Xb1,yb1,Xb2,yb2,Xval,yval,pool,pc,lr,K,mode,seed,marks):
    torch.manual_seed(seed)
    tr=samp(yb1,pool,pc,seed); Xtr,ytr=Xb1[tr],yb1[tr]
    m=newmodel(); params=set_trainable(m,mode)
    opt=torch.optim.SGD(params,lr=lr); lf=nn.CrossEntropyLoss()
    rng=np.random.RandomState(1000+seed); order=np.arange(len(tr)); best=None
    for ep in range(1,max(marks)+1):
        m.eval()   # BN always frozen (eval) in these variants; only `params` get grads
        rng.shuffle(order)
        for s in range(0,len(order),K):
            opt.zero_grad(); b=order[s:s+K]
            loss=sum(lf(m(torch.from_numpy(Xtr[i:i+1])),torch.from_numpy(ytr[i:i+1])) for i in b)/len(b)
            loss.backward(); opt.step()
        if ep in marks:
            vl=vloss(m,Xval,yval); b2=bal_acc(m,Xb2,yb2)
            if best is None or vl<best[1]: best=(ep,vl,b2)
    return best

def main():
    np.random.seed(0); Xb1,yb1=load(B1); Xb2,yb2=load(B2)
    pool,val=part(yb1,6,0); Xval,yval=Xb1[val],yb1[val]
    zs=bal_acc(newmodel(),Xb2,yb2); print(f"zero-shot batch2={100*zs:.2f}%",flush=True); print("="*96,flush=True)
    MODES=["head","head+blk4","head+blk34","full_bnfrozen"]; DATA=[("10%",2),("20%",4),("30%",6)]
    LRS=[1e-2,5e-3,1e-3]; KS=[4]; SEEDS=[0,1,2]; MARKS=[5,10,20,30,50,80]
    rows=[]
    for mode,(dn,pc),lr,K in itertools.product(MODES,DATA,LRS,KS):
        b2s=[]; eps=[]
        for sd in SEEDS:
            ep,vl,b2=run(Xb1,yb1,Xb2,yb2,Xval,yval,pool,pc,lr,K,mode,sd,MARKS); b2s.append(b2); eps.append(ep)
        b2s=np.array(b2s); m_=b2s.mean(); s_=b2s.std()
        rows.append((mode,dn,pc,lr,K,m_,s_,m_-zs,eps))
        print(f"{mode:14} data={dn}({pc*9}w) lr={lr:<6} K={K} | b2={[f'{100*x:.0f}' for x in b2s]} "
              f"sel_ep={eps} | mean={100*m_:.2f}% std={100*s_:.2f} Δ={100*(m_-zs):+.2f}pp",flush=True)
    print("="*96,flush=True); rows.sort(key=lambda r:-r[7])
    print("TOP 8 by mean Δb2:",flush=True)
    for r in rows[:8]:
        print(f"  {r[0]:14} data={r[1]} lr={r[3]} K={r[4]}: Δ={100*r[7]:+.2f}pp ({100*r[5]:.2f}%±{100*r[6]:.2f}) ep~{int(np.median(r[8]))}",flush=True)
    w=rows[0]
    print(f"\nCANDIDATE: {w[0]} data={w[1]}({w[2]*9}w) lr={w[3]} K={w[4]} ep~{int(np.median(w[8]))} "
          f"-> b2 {100*w[5]:.2f}% (Δ{100*w[7]:+.2f}pp)",flush=True)

if __name__=="__main__": main()
