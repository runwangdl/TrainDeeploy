#!/usr/bin/env python3
# Host PyTorch (CPU) config search for the FROZEN-STAT BN full-fine-tuning recipe:
#   full training (conv+BN.affine+fc), BN in eval mode during training (normalize by frozen
#   pretrained running stats -> no per-window stats, train==inference), batch size 1.
# Grid: lr x n_accum x data-size(30/40/50/70%). AVERAGING accumulation (grad divided by n_accum)
# so lr = true per-update step and n_accum = pure gradient smoothing -> the two axes are clean and
# independent. Device (summing) deployment lr = lr/n_accum. Eval = batch-2 balanced accuracy
# (BN eval mode = same frozen stats, so training and inference are consistent). ep fixed at 40.
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
def load_pool(path):   # unique fine-tune windows = mb 0..N-1 of the fixture
    z=np.load(path); Xs=[z['arr_0000']]; ys=[z['arr_0001']]; i=1
    while f'mb{i}_arr_0000' in z.files: Xs.append(z[f'mb{i}_arr_0000']); ys.append(z[f'mb{i}_arr_0001']); i+=1
    return (torch.from_numpy(np.concatenate(Xs).astype(np.float32)),
            torch.from_numpy(np.concatenate(ys).astype(np.int64).ravel()))
POOLS={30:ROOT+'/speechnet_train_full8_e40/inputs.npz', 40:ROOT+'/speechnet_ftpool_ds72/inputs.npz',
       50:ROOT+'/speechnet_ftpool_ds90/inputs.npz',    70:ROOT+'/speechnet_ftpool_ds126/inputs.npz'}
DATA={pct:load_pool(p) for pct,p in POOLS.items()}
for pct,(X,y) in DATA.items(): print(f'data {pct}%: {len(X)} windows',flush=True)
Xe=np.load(B2+'/inputs.npz')['input'].astype(np.float32); ye=np.load(B2+'/inputs.npz')['label'].astype(np.int64); cls=sorted(set(ye.tolist()))
def fresh():
    m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9); m.load_state_dict(sd0,strict=True); m.forward=types.MethodType(patched,m); return m
def bacc(m):
    m.eval()
    with torch.no_grad(): p=np.array([m(torch.from_numpy(Xe[i:i+1])).argmax(1).item() for i in range(len(ye))])
    return 100*float(np.mean([(p[ye==k]==k).mean() for k in cls]))
def bn_frozen(m):
    for mod in m.modules():
        if isinstance(mod,nn.BatchNorm2d): mod.eval()
ZS=bacc(fresh()); print('zero-shot batch2 = %.2f%%'%ZS,flush=True); print('='*70,flush=True)
def train(pct,lr,nacc,ep=40):
    torch.manual_seed(0); m=fresh(); m.train(); bn_frozen(m)
    X,y=DATA[pct]; N=len(X)
    o=torch.optim.SGD(m.parameters(),lr=lr); crit=nn.CrossEntropyLoss()
    for e in range(ep):
        idx=torch.randperm(N); o.zero_grad(); c=0
        for j in idx:
            (crit(m(X[j:j+1]),y[j:j+1])/nacc).backward(); c+=1   # AVERAGING accumulation
            if c%nacc==0: o.step(); o.zero_grad()
        if c%nacc: o.step(); o.zero_grad()
    return bacc(m)
res=[]
for pct,lr,nacc in itertools.product([30,40,50,70],[1e-3,2e-3,5e-3,1e-2,2e-2],[1,4,8]):
    try:
        a=train(pct,lr,nacc); res.append({'pct':pct,'lr':lr,'nacc':nacc,'acc':a,'d':a-ZS})
        print('data=%2d%% lr=%-6s n_accum=%d -> %6.2f%% (Δ%+.2f)  [device lr=%.2e]'%(pct,lr,nacc,a,a-ZS,lr/nacc),flush=True)
    except Exception as e: print('data=%d lr=%s nacc=%d FAILED %s'%(pct,lr,nacc,e),flush=True)
print('='*70,flush=True)
res.sort(key=lambda r:-r['acc'])
print('TOP 10:',flush=True)
for r in res[:10]: print('  data=%2d%% lr=%-6s n_accum=%d -> %.2f%% (Δ%+.2f)'%(r['pct'],r['lr'],r['nacc'],r['acc'],r['d']),flush=True)
print('BEST per data-size:',flush=True)
for pct in [30,40,50,70]:
    b=max([r for r in res if r['pct']==pct],key=lambda r:r['acc'])
    print('  %2d%%: %.2f%% (Δ%+.2f) at lr=%s n_accum=%d'%(pct,b['acc'],b['d'],b['lr'],b['nacc']),flush=True)
json.dump({'zs':ZS,'res':res},open('/app/TrainDeeploy/DeeployTest/speechnet_frozenbn_gridsearch.json','w'),indent=2)
print('saved -> speechnet_frozenbn_gridsearch.json',flush=True)
