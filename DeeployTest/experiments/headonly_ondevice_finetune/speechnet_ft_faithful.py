#!/usr/bin/env python3
# Faithfully replicate Deeploy head-only training on the EXACT fixture windows:
# sum-accumulation (PyTorch .grad sums across micro-steps), fixed-order cycling,
# SGD lr, BN/conv frozen. Reproduce the ORT -2.78pp, then lr-sweep for a positive.
import numpy as np, torch, torch.nn as nn, sys
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy
H="/app/Onnx4Deeploy/onnx/model/speechnet_train_head40"
B2="/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2"
CKPT="/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt"

def fixture_windows():
    z=np.load(H+"/inputs.npz")
    Xs=[z["arr_0000"]]; ys=[z["arr_0001"]]
    i=1
    while f"mb{i}_arr_0000" in z.files:
        Xs.append(z[f"mb{i}_arr_0000"]); ys.append(z[f"mb{i}_arr_0001"]); i+=1
    X=np.concatenate(Xs,0).astype(np.float32); y=np.concatenate(ys,0).astype(np.int64)
    return X,y
def load_b2():
    z=np.load(B2+"/inputs.npz"); return z["input"].astype(np.float32), z["label"].astype(np.int64)
def newmodel():
    m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9)
    ck=torch.load(CKPT,map_location="cpu",weights_only=False); m.load_state_dict(ck.get("model_state_dict",ck),strict=True)
    for p in m.parameters(): p.requires_grad=False
    for p in m.fc.parameters(): p.requires_grad=True
    m.eval(); return m
def bal(m,X,y):
    m.eval(); p=np.empty(len(y),np.int64)
    with torch.no_grad():
        for i in range(len(y)): p[i]=m(torch.from_numpy(X[i:i+1])).numpy().argmax(1)[0]
    c=sorted(set(y.tolist())); return float(np.mean([(p[y==k]==k).mean() for k in c]))

def train(Xtr,ytr,Xb2,yb2,lr,K,epochs,shuffle=False,seed=0):
    m=newmodel(); opt=torch.optim.SGD(m.fc.parameters(),lr=lr); lf=nn.CrossEntropyLoss()
    n=len(ytr); rng=np.random.RandomState(seed); opt.zero_grad()
    micro=0; marks={}; total=epochs*n
    order=np.arange(n)
    for ep in range(epochs):
        if shuffle: rng.shuffle(order)
        for idx in order:
            out=m(torch.from_numpy(Xtr[idx:idx+1]))
            lf(out,torch.from_numpy(ytr[idx:idx+1])).backward()   # .grad ACCUMULATES (sum)
            micro+=1
            if micro%K==0: opt.step(); opt.zero_grad()
        if (ep+1) in (5,10,20,40): marks[ep+1]=bal(m,Xb2,yb2)
    return marks

def main():
    Xtr,ytr=fixture_windows(); Xb2,yb2=load_b2()
    print("fixture train windows:",Xtr.shape,"labels per class:",{c:int((ytr==c).sum()) for c in sorted(set(ytr.tolist()))},flush=True)
    zs=bal(newmodel(),Xb2,yb2); print("zero-shot batch2=%.2f%%"%(100*zs),flush=True)
    print("="*70,flush=True)
    # 1) faithful reproduction: sum-accum K=4, lr=0.0025, fixed order
    rep=train(Xtr,ytr,Xb2,yb2,lr=0.0025,K=4,epochs=40,shuffle=False)
    print("FAITHFUL (sum K=4, lr0.0025, fixed-order):",{e:f'{100*a:.1f}' for e,a in rep.items()},
          "(ORT ref was 75.56%)",flush=True)
    print("-"*70,flush=True)
    # 2) lr sweep in the true (sum-accum, fixed-order) on-device space
    for lr in [0.005,0.01,0.02,0.05,0.1]:
        mk=train(Xtr,ytr,Xb2,yb2,lr=lr,K=4,epochs=40,shuffle=False)
        best=max(mk.values())
        print(f"sum K=4 lr={lr:<6} fixed-order: {{e:val}}={ {e:round(100*a,1) for e,a in mk.items()} }  best Δ={100*(best-zs):+.1f}pp",flush=True)
    print("-"*70,flush=True)
    # 3) does shuffling help? (Deeploy cycles fixed-order; this isolates the effect)
    for lr in [0.01,0.02]:
        mk=train(Xtr,ytr,Xb2,yb2,lr=lr,K=4,epochs=40,shuffle=True,seed=1)
        best=max(mk.values()); print(f"sum K=4 lr={lr} SHUFFLED: best Δ={100*(best-zs):+.1f}pp  {{ {', '.join(f'e{e}:{100*a:.0f}' for e,a in mk.items())} }}",flush=True)

if __name__=="__main__": main()
