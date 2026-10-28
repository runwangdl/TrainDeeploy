#!/usr/bin/env python3
# Progressive + independent head-only (BN-folded) fine-tuning across batches 1..5,
# simulated in PyTorch. For the FOLDED head-only case, PyTorch eval-mode BN == folded
# conv, so this is predictive of on-device (calibrated below against the on-device-
# verified batch1->batch2 = 82.78%). Faithful on-device SGD: sum-accumulation (K=4),
# lr 0.01, 40 epochs, fixed-order cycling; 54 stratified windows (6/class) per batch.
import numpy as np, torch, torch.nn as nn, sys
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy
CKPT="/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt"
FIX={1:"/app/TrainDeeploy/DeeployTest/Tests/Models/speechnet_infer_original",
     2:"/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2",
     3:"/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch3",
     4:"/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch4",
     5:"/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch5"}
HEADFIX="/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_head_ep40"
sd0=torch.load(CKPT,map_location="cpu",weights_only=False); sd0=sd0.get("model_state_dict",sd0)

def batch(b):
    z=np.load(FIX[b]+"/inputs.npz"); return z["input"].astype(np.float32), z["label"].astype(np.int64)
def fresh():
    m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9); m.load_state_dict(sd0,strict=True); m.eval(); return m
def bal(m,X,y):
    m.eval(); p=np.empty(len(y),np.int64)
    with torch.no_grad():
        for i in range(len(y)): p[i]=m(torch.from_numpy(X[i:i+1])).numpy().argmax(1)[0]
    c=sorted(set(y.tolist())); return float(np.mean([(p[y==k]==k).mean() for k in c]))
def strat54(X,y,seed):
    rng=np.random.RandomState(seed); sel=[]
    for c in sorted(set(y.tolist())):
        idx=np.where(y==c)[0].copy(); rng.shuffle(idx); sel+=idx[:6].tolist()
    sel=np.array(sel); return X[sel], y[sel]
def ft_head(m,Xtr,ytr,lr=0.01,K=4,epochs=40):
    for p in m.parameters(): p.requires_grad=False
    for p in m.fc.parameters(): p.requires_grad=True
    opt=torch.optim.SGD(m.fc.parameters(),lr=lr); lf=nn.CrossEntropyLoss()
    n=len(ytr); opt.zero_grad(); micro=0
    for ep in range(epochs):
        m.eval()                                    # frozen BN (== folded conv)
        for idx in range(n):                        # fixed-order cycle (matches device)
            lf(m(torch.from_numpy(Xtr[idx:idx+1])),torch.from_numpy(ytr[idx:idx+1])).backward()  # sums into .grad
            micro+=1
            if micro%K==0: opt.step(); opt.zero_grad()
    return m

def main():
    torch.manual_seed(0)
    Xb={b:batch(b) for b in range(1,6)}
    pre_zs={b:bal(fresh(),*Xb[b]) for b in range(2,6)}   # original pretrained zero-shot per eval batch
    print("pretrained zero-shot per eval batch:", {b:round(100*pre_zs[b],2) for b in range(2,6)},flush=True)

    # ---- CALIBRATION: exact on-device batch1 fixture windows -> eval batch2 (expect 82.78) ----
    z=np.load(HEADFIX+"/inputs.npz")
    Xs=[z["arr_0000"]]+[z[f"mb{i}_arr_0000"] for i in range(1,54)]
    ys=[z["arr_0001"]]+[z[f"mb{i}_arr_0001"] for i in range(1,54)]
    Xc=np.concatenate(Xs,0).astype(np.float32); yc=np.concatenate(ys,0).astype(np.int64)
    mc=ft_head(fresh(),Xc,yc); cal=bal(mc,*Xb[2])
    print("CALIBRATION (exact fixture batch1 -> batch2): %.2f%%  (on-device verified 82.78%%)"%(100*cal),flush=True)
    print("="*78,flush=True)

    # ---- INDEPENDENT: FT from pretrained on batch k, eval batch k+1 ----
    print("INDEPENDENT (each round from pretrained):",flush=True)
    ind={}
    for k in [1,2,3,4]:
        Xtr,ytr=strat54(*Xb[k],seed=100+k)
        m=ft_head(fresh(),Xtr,ytr); acc=bal(m,*Xb[k+1]); ind[k]=acc
        print("  FT batch%d -> eval batch%d : zero-shot %.2f%% -> FT %.2f%%  (Δ %+.2f pp)"%(
            k,k+1,100*pre_zs[k+1],100*acc,100*(acc-pre_zs[k+1])),flush=True)

    # ---- PROGRESSIVE: carry the fine-tuned head forward (SilentWear scheme) ----
    print("PROGRESSIVE (carry head forward across batches):",flush=True)
    m=fresh(); prog={}
    for k in [1,2,3,4]:
        before=bal(m,*Xb[k+1])                      # current (carried) model on next batch
        Xtr,ytr=strat54(*Xb[k],seed=100+k)
        ft_head(m,Xtr,ytr); acc=bal(m,*Xb[k+1]); prog[k]=acc
        print("  round%d: FT batch%d -> eval batch%d : before %.2f%% -> after %.2f%%  (vs pretrained-zs %.2f%%, Δ %+.2f pp)"%(
            k,k,k+1,100*before,100*acc,100*pre_zs[k+1],100*(acc-pre_zs[k+1])),flush=True)

    print("="*78,flush=True)
    print("SUMMARY (balanced acc on eval batch; Δ vs that batch's pretrained zero-shot):",flush=True)
    print("  %-14s %-10s %-14s %-14s"%("FT->eval","pretrn-zs","independent","progressive"),flush=True)
    for k in [1,2,3,4]:
        print("  b%d->b%-10d %-10s %-14s %-14s"%(k,k+1,
            "%.2f%%"%(100*pre_zs[k+1]),
            "%.2f%% (%+.2f)"%(100*ind[k],100*(ind[k]-pre_zs[k+1])),
            "%.2f%% (%+.2f)"%(100*prog[k],100*(prog[k]-pre_zs[k+1]))),flush=True)

if __name__=="__main__": main()
