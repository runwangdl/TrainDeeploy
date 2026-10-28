#!/usr/bin/env python3
# Reproduce the SilentWear paper's INCREMENTAL fine-tuning for S01, fold 3
# (base model pretrained on sessions 1&2, incrementally fine-tune across session-3's batches).
# Uses the RAW data loader (data_raw_and_filt, on-the-fly windowing + rest-downsampling) — NOT the
# pre-computed wins_and_features. Paper FT protocol: Adam (lr 1e-3, wd 1e-4), batch 32, up to 50
# epochs w/ early stopping, 70/30 stratified split, standard (training-mode) BatchNorm.
import numpy as np, torch, sys, types
import torch.nn as nn
sys.path.insert(0,'/app/Onnx4Deeploy')
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy
from onnx4deeploy.data.silent_wear_datasource import SilentWearDataSource
DATA='/app/SilentWear/SilentWear_data/data_raw_and_filt'
CKPT='/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt'
sd0=torch.load(CKPT,map_location='cpu',weights_only=False); sd0=sd0.get('model_state_dict',sd0)
def patched(self,x):
    for b in self.blocks: x=b(x)
    x=self.global_pool(x); x=x.reshape(x.shape[0],self._fc_in); return self.fc(x)
def load_batch(k):
    ds=SilentWearDataSource(data_path=DATA,subject='S01',session=3,batch=k,condition='vocalized',downsample_rest=True)
    X,y=ds._load_windows()
    return (torch.from_numpy(np.concatenate(X).astype(np.float32)),
            torch.from_numpy(np.concatenate(y).astype(np.int64).ravel()))
BATCHES={k:load_batch(k) for k in range(1,6)}
for k,(X,y) in BATCHES.items(): print('session-3 batch %d: %d windows, classes %s'%(k,len(X),sorted(set(y.tolist()))),flush=True)
def mk():
    m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9); m.load_state_dict(sd0); m.forward=types.MethodType(patched,m); return m
def bacc(m,X,y):
    m.eval(); cls=sorted(set(y.tolist()))
    with torch.no_grad(): p=m(X).argmax(1).numpy()
    return 100*float(np.mean([(p[y.numpy()==c]==c).mean() for c in cls]))
def strat_split(X,y,frac=0.7,seed=42):
    rng=np.random.default_rng(seed); tr=[]; va=[]
    for c in sorted(set(y.tolist())):
        idx=np.where(y.numpy()==c)[0]; rng.shuffle(idx); n=max(1,int(round(len(idx)*frac)))
        tr+=list(idx[:n]); va+=list(idx[n:])
    tr=np.array(tr); va=np.array(va if len(va) else tr)
    return X[tr],y[tr],X[va],y[va]
def finetune(m,X,y,lr=1e-3,wd=1e-4,bs=32,epochs=50,patience=10):
    Xtr,ytr,Xva,yva=strat_split(X,y); N=len(Xtr)
    o=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd); crit=nn.CrossEntropyLoss()
    best=1e9; best_sd=None; bad=0
    for e in range(epochs):
        m.train(); idx=torch.randperm(N)
        for s in range(0,N,bs):
            b=idx[s:s+bs]; o.zero_grad(); crit(m(Xtr[b]),ytr[b]).backward(); o.step()
        m.eval()
        with torch.no_grad(): vl=float(crit(m(Xva),yva))
        if vl<best-1e-4: best=vl; best_sd={k:v.clone() for k,v in m.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=patience: break
    if best_sd: m.load_state_dict(best_sd)
    return m
# ---- base zero-shot per batch (no FT) ----
base=mk(); print('\n[base model — zero-shot, NO fine-tuning]',flush=True)
zs={k:bacc(base,*BATCHES[k]) for k in range(1,6)}
for k in range(1,6): print('   session-3 batch %d: balanced acc = %.2f%%'%(k,zs[k]),flush=True)
print('   MEAN zero-shot over session-3 = %.2f%%'%np.mean(list(zs.values())),flush=True)
# ---- incremental FT: adapt progressively across batches ----
print('\n[INCREMENTAL fine-tuning — progressive across session-3 batches]',flush=True)
print('  batch | model-before-FT (adapted on prev) | after FT on this batch | base(no-FT)',flush=True)
m=mk()   # starts from base
for k in range(1,6):
    X,y=BATCHES[k]
    before=bacc(m,X,y)                    # current model (FT'd on batches 1..k-1) on this NEW batch
    m=finetune(m,X,y)                     # FT on this batch (carried forward)
    after=bacc(m,X,y)
    print('    %d   |          %6.2f%%              |        %6.2f%%         |   %6.2f%%   (Δ incr vs base = %+.2f)'%(
        k, before, after, zs[k], before-zs[k]),flush=True)
print('DONE',flush=True)
