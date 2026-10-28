#!/usr/bin/env python3
# Device-faithful (SUMMING) validation to pick the FINAL on-device config. frozen-stat BN, batch-1,
# ep40, gradients SUMMED over n_accum (matches TrainDeeploy: accum += grad; SGD w -= lr*acc; no
# /n_accum => eff_lr = lr*n_accum). 8 stratified 30% draws x 2 seeds; report mean±std per config.
# Compares head-only vs frozen-BN full-FT at their device (lr,n_accum) settings.
import numpy as np, torch, sys, types
import torch.nn as nn
sys.path.insert(0,'/app/Onnx4Deeploy')
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy
ROOT='/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet'
B2='/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2'
CKPT='/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt'
sd0=torch.load(CKPT,map_location='cpu',weights_only=False); sd0=sd0.get('model_state_dict',sd0)
def patched(self,x):
    for b in self.blocks: x=b(x)
    x=self.global_pool(x); x=x.reshape(x.shape[0],self._fc_in); return self.fc(x)
z=np.load(ROOT+'/speechnet_ftpool_ds180/inputs.npz'); Xs=[z['arr_0000']]; ys=[z['arr_0001']]; i=1
while f'mb{i}_arr_0000' in z.files: Xs.append(z[f'mb{i}_arr_0000']); ys.append(z[f'mb{i}_arr_0001']); i+=1
Xp=np.concatenate(Xs).astype(np.float32); yp=np.concatenate(ys).astype(np.int64).ravel()
Xe=np.load(B2+'/inputs.npz')['input'].astype(np.float32); ye=np.load(B2+'/inputs.npz')['label'].astype(np.int64); cls=sorted(set(ye.tolist()))
def draw(seed,per=6):
    rng=np.random.default_rng(seed); idx=[]
    for c in sorted(set(yp.tolist())): idx+=list(rng.choice(np.where(yp==c)[0],per,replace=False))
    idx=np.array(idx); return torch.from_numpy(Xp[idx]), torch.from_numpy(yp[idx])
def fresh():
    m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9); m.load_state_dict(sd0); m.forward=types.MethodType(patched,m); return m
def bacc(m):
    m.eval()
    with torch.no_grad(): p=np.array([m(torch.from_numpy(Xe[k:k+1])).argmax(1).item() for k in range(len(ye))])
    return 100*float(np.mean([(p[ye==k]==k).mean() for k in cls]))
def bnf(m):
    for md in m.modules():
        if isinstance(md,nn.BatchNorm2d): md.eval()
ZS=bacc(fresh()); print('zero-shot=%.2f%% (SUMMING convention = device)'%ZS,flush=True)
def train(X,y,scope,seed,lr,nacc,ep=40):
    torch.manual_seed(seed); m=fresh(); m.train(); bnf(m)
    if scope=='head':
        for n,p in m.named_parameters(): p.requires_grad=n.startswith('fc')
    o=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=lr); crit=nn.CrossEntropyLoss(); N=len(X)
    for e in range(ep):
        idx=torch.randperm(N); o.zero_grad(); c=0
        for j in idx:
            crit(m(X[j:j+1]),y[j:j+1]).backward(); c+=1   # SUM
            if c%nacc==0: o.step(); o.zero_grad()
        if c%nacc: o.step(); o.zero_grad()
    return bacc(m)
DR=[(1000+d,s) for d in range(8) for s in [0,1]]; DATA={d:draw(d) for d,_ in DR}
CFG=[('head',5e-3,1),('head',1e-2,1),('full',1e-3,1),('full',2e-3,1),('full',2.5e-4,4)]
out={}
for scope,lr,na in CFG:
    a=np.array([train(DATA[d][0],DATA[d][1],scope,s,lr,na) for (d,s) in DR]); out[(scope,lr,na)]=a
    print('  %-5s lr=%-8s n_accum=%d (eff=%.1e): mean Δ=%+.2f std=%.2f (min Δ=%+.2f)'%(scope,lr,na,lr*na,a.mean()-ZS,a.std(),a.min()-ZS),flush=True)
print('\nBEST head:',end=''); bh=max([c for c in CFG if c[0]=='head'],key=lambda c:out[c].mean()); print(' lr=%s n%d -> %+.2f±%.2f'%(bh[1],bh[2],out[bh].mean()-ZS,out[bh].std()),flush=True)
print('BEST full:',end=''); bf=max([c for c in CFG if c[0]=='full'],key=lambda c:out[c].mean()); print(' lr=%s n%d -> %+.2f±%.2f'%(bf[1],bf[2],out[bf].mean()-ZS,out[bf].std()),flush=True)
print('DONE',flush=True)
