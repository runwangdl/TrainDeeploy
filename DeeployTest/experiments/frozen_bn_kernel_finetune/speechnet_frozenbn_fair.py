#!/usr/bin/env python3
# Fair head-to-head: the variance study compared both recipes at the SAME lr (2e-3), which
# under-tunes head-only (its good lr is higher). Here: same 8 draws x 2 seeds, each recipe swept
# over a small a-priori lr set, report mean±std per (recipe,lr), then compare each recipe AT ITS
# OWN BEST-MEAN lr. frozen-stat BN, batch-1, n_accum 1, ep40, eval batch-2.
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
Xpool=np.concatenate(Xs).astype(np.float32); ypool=np.concatenate(ys).astype(np.int64).ravel()
Xe=np.load(B2+'/inputs.npz')['input'].astype(np.float32); ye=np.load(B2+'/inputs.npz')['label'].astype(np.int64); cls=sorted(set(ye.tolist()))
def draw(seed,per=6):
    rng=np.random.default_rng(seed); idx=[]
    for c in sorted(set(ypool.tolist())):
        ci=np.where(ypool==c)[0]; idx += list(rng.choice(ci,per,replace=False))
    idx=np.array(idx); return torch.from_numpy(Xpool[idx]), torch.from_numpy(ypool[idx])
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
def train(X,y,scope,seed,lr,ep=40):
    torch.manual_seed(seed); m=fresh(); m.train(); bn_frozen(m)
    if scope=='head':
        for n,p in m.named_parameters(): p.requires_grad=n.startswith('fc')
    o=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=lr); crit=nn.CrossEntropyLoss(); N=len(X)
    for e in range(ep):
        idx=torch.randperm(N)
        for j in idx:
            o.zero_grad(); crit(m(X[j:j+1]),y[j:j+1]).backward(); o.step()
    return bacc(m)
DRAWS=[(1000+d,s) for d in range(8) for s in [0,1]]
CONFIGS=[('head',5e-3),('head',1e-2),('head',2e-2),('full',1e-3),('full',2.5e-3)]
agg={}
DATA={ds: draw(ds) for ds,_ in DRAWS}
for scope,lr in CONFIGS:
    accs=np.array([train(DATA[ds][0],DATA[ds][1],scope,s,lr) for (ds,s) in DRAWS])
    agg[(scope,lr)]=accs
    print('  %-5s lr=%-7s : mean Δ=%+.2f  std=%.2f  (range %+.2f..%+.2f)'%(scope,lr,accs.mean()-ZS,accs.std(),accs.min()-ZS,accs.max()-ZS),flush=True)
# best-mean lr per recipe, paired comparison
bh=max([c for c in CONFIGS if c[0]=='head'],key=lambda c:agg[c].mean())
bf=max([c for c in CONFIGS if c[0]=='full'],key=lambda c:agg[c].mean())
ph=agg[bh]; pf=agg[bf]; diff=pf-ph
print('\nFAIR head-to-head (each at own best-mean lr):',flush=True)
print('  head-only @ lr%s : mean Δ=%+.2f  std=%.2f'%(bh[1],ph.mean()-ZS,ph.std()),flush=True)
print('  full-FT   @ lr%s : mean Δ=%+.2f  std=%.2f'%(bf[1],pf.mean()-ZS,pf.std()),flush=True)
print('  paired (full-head): mean=%+.2f  std=%.2f  full wins %d/%d'%(diff.mean(),diff.std(),int((diff>0).sum()),len(diff)),flush=True)
print('DONE',flush=True)
