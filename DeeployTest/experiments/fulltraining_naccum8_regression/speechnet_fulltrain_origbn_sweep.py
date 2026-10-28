#!/usr/bin/env python3
# Comprehensive host (CPU, full-precision) sweep: FULL-model fine-tuning with the ORIGINAL BatchNorm
# implementation (no kernel modification) = batch-1 batch-statistics BN, running stats NOT updated
# on-device. Goal: demonstrate that NO configuration learns effectively on-device.
#
# Device-realistic eval = reset running stats to pretrained (frozen) before eval, because the device
# kernel never updates them. We ALSO report the "optimistic" eval (with the running stats PyTorch
# accumulated) to show that even that misleading number is marginal, and the device-realistic one fails.
#
# Swept config space:
#   scope     : FULL model (all 22 params)                 [fixed]
#   batch     : 1 (forced), batch-stat BN (BatchNormInternal semantics), RS not updated on device
#   n_accum   : {1, 4, 8}          (averaging convention -> lr is the per-update step; device baked lr = lr/n_accum)
#   data frac : {10, 30, 50, 70}%  (18/54/90/126 windows = 2/6/10/14 per class, stratified, rest-downsampled)
#   lr        : {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
#   epochs    : 40                 [fixed]
#   draws     : 3 independent stratified draws per config (for mean +/- std)
import numpy as np, torch, sys, types, itertools, json
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
PERCLASS={10:2,30:6,50:10,70:14}
def draw(pct,seed):
    rng=np.random.default_rng(seed); idx=[]
    for c in sorted(set(yp.tolist())): idx+=list(rng.choice(np.where(yp==c)[0],PERCLASS[pct],replace=False))
    idx=np.array(idx); return torch.from_numpy(Xp[idx]), torch.from_numpy(yp[idx])
def fresh():
    m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9); m.load_state_dict(sd0); m.forward=types.MethodType(patched,m); return m
def bacc(m):
    m.eval()
    with torch.no_grad(): p=np.array([m(torch.from_numpy(Xe[k:k+1])).argmax(1).item() for k in range(len(ye))])
    return 100*float(np.mean([(p[ye==k]==k).mean() for k in cls]))
ZS=bacc(fresh()); print('zero-shot batch-2 = %.2f%%'%ZS,flush=True); print('='*92,flush=True)
def run(pct,lr,nacc,ep=40,seed=0):
    torch.manual_seed(seed); m=fresh(); m.train()   # model.train() -> ORIGINAL batch-stat BN (batch-1), updates RS
    X,y=draw(pct,1000+seed); N=len(X)
    o=torch.optim.SGD(m.parameters(),lr=lr); crit=nn.CrossEntropyLoss()
    for e in range(ep):
        idx=torch.randperm(N); o.zero_grad(); c=0
        for j in idx:
            (crit(m(X[j:j+1]),y[j:j+1])/nacc).backward(); c+=1
            if c%nacc==0: o.step(); o.zero_grad()
        if c%nacc: o.step(); o.zero_grad()
    opt=bacc(m)                                      # optimistic: eval with the running stats it accumulated
    sd=m.state_dict()                                # device-realistic: RS never updated -> reset to pretrained
    for k in sd0:
        if 'running' in k or 'num_batches' in k: sd[k]=sd0[k].clone()
    m.load_state_dict(sd); dev=bacc(m)
    return opt,dev
res=[]
for pct,lr,nacc in itertools.product([10,30,50,70],[1e-4,3e-4,1e-3,3e-3,1e-2],[1,4,8]):
    o3=[]; d3=[]
    for s in range(3):
        opt,dev=run(pct,lr,nacc,seed=s); o3.append(opt); d3.append(dev)
    o3=np.array(o3); d3=np.array(d3); res.append({'pct':pct,'lr':lr,'nacc':nacc,'opt':o3.mean(),'dev':d3.mean(),'dev_std':d3.std(),'dev_max':d3.max()})
    print('data=%2d%% lr=%-6s n_accum=%d | optimistic Δ=%+6.2f | DEVICE-REALISTIC Δ=%+6.2f±%.2f (best-draw %+6.2f)'%(
        pct,lr,nacc,o3.mean()-ZS,d3.mean()-ZS,d3.std(),d3.max()-ZS),flush=True)
print('='*92,flush=True)
dev_best=max(res,key=lambda r:r['dev']); dev_bestmax=max(res,key=lambda r:r['dev_max'])
opt_best=max(res,key=lambda r:r['opt'])
npos_dev=sum(1 for r in res if r['dev']-ZS>0); npos_devmax=sum(1 for r in res if r['dev_max']-ZS>0)
print('CONFIGS: %d (4 data x 5 lr x 3 n_accum), 3 draws each'%len(res),flush=True)
print('DEVICE-REALISTIC (frozen RS = what deploys):',flush=True)
print('  best MEAN  Δ=%+.2fpp  (data=%d%% lr=%s n_accum=%d)'%(dev_best['dev']-ZS,dev_best['pct'],dev_best['lr'],dev_best['nacc']),flush=True)
print('  best SINGLE-DRAW Δ=%+.2fpp  (data=%d%% lr=%s n_accum=%d)'%(dev_bestmax['dev_max']-ZS,dev_bestmax['pct'],dev_bestmax['lr'],dev_bestmax['nacc']),flush=True)
print('  configs with mean Δ>0: %d/%d   | configs with ANY draw Δ>0: %d/%d'%(npos_dev,len(res),npos_devmax,len(res)),flush=True)
print('OPTIMISTIC (updated RS = NOT achievable on device): best Δ=%+.2fpp'%(opt_best['opt']-ZS),flush=True)
json.dump({'zs':ZS,'res':res},open('/app/TrainDeeploy/DeeployTest/experiments/fulltraining_naccum8_regression/speechnet_fulltrain_origbn_sweep.json','w'),indent=2)
print('saved -> speechnet_fulltrain_origbn_sweep.json',flush=True); print('DONE',flush=True)
