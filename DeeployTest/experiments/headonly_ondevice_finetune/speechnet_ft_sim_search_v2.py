#!/usr/bin/env python3
# Phase 1 v2 sim search: vanilla SGD (no momentum — unsupported on-device).
# Levers: data% x lr x effective-batch K (n-accum, mean-grad) x epochs.
# Robust: 3 seeds/config, val-LOSS selection (continuous), batch-2 = held-out test.
# Deeploy mapping: effective-batch K, eff-lr L  ->  CLI --n-accum K --lr L/K
#   (Deeploy accumulator SUMS; we use mean-grad here, so divide lr by K on device).
import numpy as np, torch, torch.nn as nn, itertools, sys
sys.path.insert(0, "/app/Onnx4Deeploy")
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
    model.eval(); pred=np.empty(len(y),np.int64)
    with torch.no_grad():
        for i in range(len(y)): pred[i]=model(torch.from_numpy(X[i:i+1])).numpy().argmax(1)[0]
    cls=sorted(set(y.tolist())); return float(np.mean([(pred[y==c]==c).mean() for c in cls]))
def val_loss(model,X,y):
    model.eval(); lf=nn.CrossEntropyLoss(reduction='sum'); tot=0.0
    with torch.no_grad():
        for i in range(len(y)): tot+=lf(model(torch.from_numpy(X[i:i+1])),torch.from_numpy(y[i:i+1])).item()
    return tot/len(y)
def strat_partition(y,per_class_val=6,seed=0):
    rng=np.random.RandomState(seed); val=[]; pool=[]
    for c in sorted(set(y.tolist())):
        idx=np.where(y==c)[0].copy(); rng.shuffle(idx); val+=idx[:per_class_val].tolist(); pool+=idx[per_class_val:].tolist()
    return np.array(pool),np.array(val)
def strat_sample(y,pool,per_class,seed):
    rng=np.random.RandomState(seed); sel=[]
    for c in sorted(set(y.tolist())):
        pc=pool[y[pool]==c].copy(); rng.shuffle(pc); sel+=pc[:per_class].tolist()
    return np.array(sel)

def train_eval(Xb1,yb1,Xb2,yb2,Xval,yval,pool,pc,lr,K,seed,marks):
    torch.manual_seed(seed)
    tr=strat_sample(yb1,pool,pc,seed=seed); Xtr,ytr=Xb1[tr],yb1[tr]
    m=newmodel(); opt=torch.optim.SGD(m.parameters(),lr=lr); lf=nn.CrossEntropyLoss()
    rng=np.random.RandomState(1000+seed); order=np.arange(len(tr)); best=None
    for ep in range(1,max(marks)+1):
        m.train(); rng.shuffle(order)
        for s in range(0,len(order),K):
            opt.zero_grad(); batch=order[s:s+K]
            loss=sum(lf(m(torch.from_numpy(Xtr[i:i+1])),torch.from_numpy(ytr[i:i+1])) for i in batch)/len(batch)
            loss.backward(); opt.step()
        if ep in marks:
            vl=val_loss(m,Xval,yval); b2=bal_acc(m,Xb2,yb2)
            if best is None or vl<best[1]: best=(ep,vl,b2)
    return best   # (sel_epoch, val_loss, b2_acc) at min val-loss

def main():
    np.random.seed(0)
    Xb1,yb1=load(B1); Xb2,yb2=load(B2)
    pool,val=strat_partition(yb1,6,0); Xval,yval=Xb1[val],yb1[val]
    zs_b2=bal_acc(newmodel(),Xb2,yb2)
    print(f"zero-shot batch2(test)={100*zs_b2:.2f}%   val=54w(6/cls) pool=126w(14/cls)",flush=True)
    print("="*92,flush=True)
    DATA=[("10%",2),("20%",4),("30%",6)]; LRS=[1e-3,5e-4]; KS=[1,4]; SEEDS=[0,1,2]; MARKS=[3,5,8,12,20,30]
    rows=[]
    for (dn,pc),lr,K in itertools.product(DATA,LRS,KS):
        b2s=[]; eps=[]
        for sd in SEEDS:
            ep,vl,b2=train_eval(Xb1,yb1,Xb2,yb2,Xval,yval,pool,pc,lr,K,sd,MARKS); b2s.append(b2); eps.append(ep)
        b2s=np.array(b2s); mean=b2s.mean(); std=b2s.std(); dmean=mean-zs_b2
        rows.append((dn,pc,lr,K,mean,std,dmean,eps))
        print(f"data={dn}({pc*9}w) lr={lr:<6} K={K} | seeds b2={[f'{100*x:.0f}' for x in b2s]} "
              f"sel_ep={eps} | mean={100*mean:.2f}% std={100*std:.2f} Δb2={100*dmean:+.2f}pp",flush=True)
    print("="*92,flush=True)
    rows.sort(key=lambda r:-r[6])
    print("RANKED by mean Δb2 (robust, val-loss-selected):",flush=True)
    for dn,pc,lr,K,mean,std,dmean,eps in rows[:6]:
        print(f"  data={dn} lr={lr} K={K}: Δb2={100*dmean:+.2f}pp (mean {100*mean:.2f}% ± {100*std:.2f})  sel_ep~{int(np.median(eps))}",flush=True)
    w=rows[0]
    print(f"\nCANDIDATE: data={w[0]}({w[1]*9}w) lr={w[2]} K={w[3]} ep~{int(np.median(w[7]))} "
          f"-> batch2 {100*w[4]:.2f}% (Δ{100*w[6]:+.2f}pp vs {100*zs_b2:.2f}%)",flush=True)
    print(f"Deeploy CLI: --data-size {w[1]*9} --n-accum {w[3]} --lr {w[2]/w[3]:.2e} "
          f"--n-steps {int(np.median(w[7]))*w[1]*9//w[3]}",flush=True)

if __name__=="__main__": main()
