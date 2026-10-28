#!/usr/bin/env python3
# Resolve the key uncertainty from the grid: is frozen-stat BN full-FT's +5.56pp a robust win over
# head-only (+4.44pp), or within sampling noise? Draw K independent stratified 30% subsamples
# (6/class) from the full 180-window pool, and for each train BOTH recipes (frozen-stat BN, batch-1,
# lr 2e-3, n_accum 1, ep40): (A) full training (conv+BN.affine+fc) vs (B) head-only (fc only).
# Report mean +/- std and the PAIRED difference. Plus a finer lr sweep at 30%. Eval = batch-2.
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
# full 180 pool
z=np.load(ROOT+'/speechnet_ftpool_ds180/inputs.npz'); Xs=[z['arr_0000']]; ys=[z['arr_0001']]; i=1
while f'mb{i}_arr_0000' in z.files: Xs.append(z[f'mb{i}_arr_0000']); ys.append(z[f'mb{i}_arr_0001']); i+=1
Xpool=np.concatenate(Xs).astype(np.float32); ypool=np.concatenate(ys).astype(np.int64).ravel()
Xe=np.load(B2+'/inputs.npz')['input'].astype(np.float32); ye=np.load(B2+'/inputs.npz')['label'].astype(np.int64); cls=sorted(set(ye.tolist()))
def draw(seed,per=6):  # stratified: `per` windows/class
    rng=np.random.default_rng(seed); idx=[]
    for c in sorted(set(ypool.tolist())):
        ci=np.where(ypool==c)[0]; idx += list(rng.choice(ci,per,replace=False))
    idx=np.array(idx)
    return torch.from_numpy(Xpool[idx]), torch.from_numpy(ypool[idx])
def fresh():
    m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9); m.load_state_dict(sd0,strict=True); m.forward=types.MethodType(patched,m); return m
def bacc(m):
    m.eval()
    with torch.no_grad(): p=np.array([m(torch.from_numpy(Xe[k:k+1])).argmax(1).item() for k in range(len(ye))])
    return 100*float(np.mean([(p[ye==k]==k).mean() for k in cls]))
def bn_frozen(m):
    for mod in m.modules():
        if isinstance(mod,nn.BatchNorm2d): mod.eval()
ZS=bacc(fresh()); print('zero-shot=%.2f%%'%ZS,flush=True)
def train(X,y,scope,seed,lr=2e-3,nacc=1,ep=40):
    torch.manual_seed(seed); m=fresh(); m.train(); bn_frozen(m)
    if scope=='head':
        for n,p in m.named_parameters(): p.requires_grad = n.startswith('fc')
    o=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=lr); crit=nn.CrossEntropyLoss(); N=len(X)
    for e in range(ep):
        idx=torch.randperm(N); o.zero_grad(); c=0
        for j in idx:
            (crit(m(X[j:j+1]),y[j:j+1])/nacc).backward(); c+=1
            if c%nacc==0: o.step(); o.zero_grad()
        if c%nacc: o.step(); o.zero_grad()
    return bacc(m)
# ---- variance study: K draws x S seeds, full vs head ----
K=8; SEEDS=[0,1]
full=[]; head=[]; paired=[]
print('\n[variance: %d draws x %d seeds, 30%% (6/class), frozen-BN lr2e-3 n_accum1 ep40]'%(K,len(SEEDS)),flush=True)
for d in range(K):
    X,y=draw(1000+d)
    for s in SEEDS:
        af=train(X,y,'all',s); ah=train(X,y,'head',s)
        full.append(af); head.append(ah); paired.append(af-ah)
        print('  draw %d seed %d: full=%.2f (Δ%+.2f)  head=%.2f (Δ%+.2f)  full-head=%+.2f'%(d,s,af,af-ZS,ah,ah-ZS,af-ah),flush=True)
full=np.array(full); head=np.array(head); paired=np.array(paired)
print('\nSUMMARY (n=%d):'%len(full),flush=True)
print('  full-FT : mean Δ=%+.2f  std=%.2f  (range %+.2f..%+.2f)'%(full.mean()-ZS,full.std(),full.min()-ZS,full.max()-ZS),flush=True)
print('  head    : mean Δ=%+.2f  std=%.2f  (range %+.2f..%+.2f)'%(head.mean()-ZS,head.std(),head.min()-ZS,head.max()-ZS),flush=True)
print('  paired (full-head): mean=%+.2f  std=%.2f  >0 in %d/%d'%(paired.mean(),paired.std(),int((paired>0).sum()),len(paired)),flush=True)
# ---- finer lr sweep at 30% (draw seed 1000) ----
X,y=draw(1000)
print('\n[finer lr sweep @30%%, full-FT n_accum1 ep40]',flush=True)
for lr in [1e-3,1.5e-3,2e-3,2.5e-3,3e-3,4e-3,5e-3]:
    a=train(X,y,'all',0,lr=lr); print('  lr=%-7s -> %.2f%% (Δ%+.2f)'%(lr,a,a-ZS),flush=True)
print('DONE',flush=True)
