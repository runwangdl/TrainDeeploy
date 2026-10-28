#!/usr/bin/env python3
# Validate the BN_FROZEN_STATS kernel mod: run the SAME 20-step full-training loop on the host
# with frozen-stat BN (PyTorch, BN in eval mode) and compare the resulting 22 weights to the
# device [WDUMP]. If they match (small drift), the modified forward+backward kernels are correct.
import re, struct, numpy as np, torch, sys, types
import torch.nn as nn
sys.path.insert(0,'/app/Onnx4Deeploy')
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy
TR='/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen'
LOG='/app/TrainDeeploy/DeeployTest/fullfrozen_validate.log'
CKPT='/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt'
NSTEPS=20; LR=1e-3
sd0=torch.load(CKPT,map_location='cpu',weights_only=False); sd0=sd0.get('model_state_dict',sd0)
def patched(self,x):
    for b in self.blocks: x=b(x)
    x=self.global_pool(x); x=x.reshape(x.shape[0],self._fc_in); return self.fc(x)
# device micro-batch order: mb 0..NSTEPS-1 (n_accum 1)
z=np.load(TR+'/inputs.npz')
def win(i): return (z['arr_0000'] if i==0 else z[f'mb{i}_arr_0000']).astype(np.float32), int(z['arr_0001'] if i==0 else z[f'mb{i}_arr_0001'])
# ---- host frozen-BN training (SGD, batch-1, lr, frozen running stats) ----
m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9); m.load_state_dict(sd0); m.forward=types.MethodType(patched,m)
m.train()
for md in m.modules():
    if isinstance(md,nn.BatchNorm2d): md.eval()          # frozen stats, no update
o=torch.optim.SGD(m.parameters(),lr=LR); crit=nn.CrossEntropyLoss()
losses=[]
for i in range(NSTEPS):
    x,lab=win(i); o.zero_grad()
    out=m(torch.from_numpy(x)); loss=crit(out,torch.tensor([lab])); loss.backward(); o.step()
    losses.append(float(loss))
print('host frozen-BN losses[0,1,-1] = %.4f %.4f %.4f'%(losses[0],losses[1],losses[-1]))
NAMES=['blocks_0_0_weight','blocks_0_0_bias','blocks_0_1_weight','blocks_0_1_bias','blocks_1_0_weight','blocks_1_0_bias','blocks_1_1_weight','blocks_1_1_bias','blocks_2_0_weight','blocks_2_0_bias','blocks_2_1_weight','blocks_2_1_bias','blocks_3_0_weight','blocks_3_0_bias','blocks_3_1_weight','blocks_3_1_bias','blocks_4_0_weight','blocks_4_0_bias','blocks_4_1_weight','blocks_4_1_bias','fc_weight','fc_bias']
def mapk(k): return ('fc.'+k[3:]) if k.startswith('fc_') else 'blocks.%s.%s.%s'%(tuple(k.split('_')[1:3])+('_'.join(k.split('_')[3:]),))
hostw={nm: m.state_dict()[mapk(nm)].detach().numpy().ravel() for nm in NAMES}
# ---- parse device WDUMP (final step) ----
pat=re.compile(r'\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)'); dump={}
for line in open(LOG):
    mo=pat.search(line)
    if mo: dump[(int(mo[1]),int(mo[2]))]=np.array([struct.unpack('<f',struct.pack('<I',int(w,16)))[0] for w in mo[4].split()],dtype=np.float32)
if not dump:
    print('NO WDUMP yet in device log.'); sys.exit(0)
last=max(s for s,_ in dump); dev={wi:dump[(last,wi)] for (s,wi) in dump if s==last}
print('device final step=%d, %d tensors'%(last,len(dev)))
print('\nper-weight  max|dev-host|   host|max|   relerr')
worst=0
for wi,nm in enumerate(NAMES):
    d=dev.get(wi); h=hostw[nm]
    if d is None: print('  %-22s (missing wi=%d)'%(nm,wi)); continue
    md=float(np.abs(d-h).max()); hm=float(np.abs(h).max()); rel=md/(hm+1e-12); worst=max(worst,rel)
    print('  %-22s %11.3e   %9.3e   %7.2f%%%s'%(nm,md,hm,100*rel,'  <<<' if rel>0.02 else ''))
print('\nworst relative dev-vs-host = %.3f%%  -> %s'%(100*worst,'KERNEL OK (matches host frozen-BN)' if worst<0.02 else 'MISMATCH — check kernel'))
