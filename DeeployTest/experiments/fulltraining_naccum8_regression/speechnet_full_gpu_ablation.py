#!/usr/bin/env python3
# Independent host/GPU (full-precision PyTorch) verification of the device-faithful full-training
# result, plus a paper-gap decomposition. Re-runs the device-faithful config (SGD, batch-1 forward,
# n_accum 8, frozen BN running-stats at eval) in clean PyTorch -- confirming the on-device
# regression is NOT an ORT/Deeploy pipeline artifact -- then relaxes constraints one axis at a time
# toward the SilentWear paper (running-stat update -> true batch -> Adam -> 50 ep). Same 54 training
# windows as the ep40 fixture; eval = batch-2 balanced accuracy. Run in the agitated_hugle container.
import numpy as np, torch, sys, types
import torch.nn as nn
sys.path.insert(0,'/app/Onnx4Deeploy')
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy
torch.manual_seed(0)
TR='/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_full8_e40'
B2='/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2'
CKPT='/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt'
sd0=torch.load(CKPT,map_location='cpu',weights_only=False); sd0=sd0.get('model_state_dict',sd0)

def patched(self,x):  # batch-capable forward (deploy model hardcodes reshape(1, fc_in))
    for b in self.blocks: x=b(x)
    x=self.global_pool(x); x=x.reshape(x.shape[0],self._fc_in); return self.fc(x)

# 54 unique training windows = micro-batches 0..53 of the ep40 fixture
z=np.load(TR+'/inputs.npz'); Xtr=[z['arr_0000']]; ytr=[z['arr_0001']]
for i in range(1,54): Xtr.append(z[f'mb{i}_arr_0000']); ytr.append(z[f'mb{i}_arr_0001'])
Xtr=torch.from_numpy(np.concatenate(Xtr).astype(np.float32)); ytr=torch.from_numpy(np.concatenate(ytr).astype(np.int64).ravel())
Xe=np.load(B2+'/inputs.npz')['input'].astype(np.float32); ye=np.load(B2+'/inputs.npz')['label'].astype(np.int64); cls=sorted(set(ye.tolist()))

def fresh():
    m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9); m.load_state_dict(sd0,strict=True)
    m.forward=types.MethodType(patched,m); return m
def bacc(m):
    m.eval()
    with torch.no_grad(): p=np.array([m(torch.from_numpy(Xe[i:i+1])).argmax(1).item() for i in range(len(ye))])
    return 100*float(np.mean([(p[ye==k]==k).mean() for k in cls]))
def reset_rs(m):  # device never updates BN running stats -> restore pretrained for eval
    sd=m.state_dict()
    for k in sd0:
        if 'running' in k or 'num_batches' in k: sd[k]=sd0[k].clone()
    m.load_state_dict(sd)
ZS=bacc(fresh()); print('zero-shot batch2 = %.2f%%'%ZS); print('-'*64)
def run(tag,opt,batch,nacc,ep,lr,frozen):
    torch.manual_seed(0); m=fresh(); m.train()
    o=torch.optim.SGD(m.parameters(),lr=lr) if opt=='sgd' else torch.optim.Adam(m.parameters(),lr=lr)
    crit=nn.CrossEntropyLoss(); N=len(Xtr)
    for e in range(ep):
        idx=torch.randperm(N)
        if batch==1 and nacc>1:                 # device-faithful: batch-1 fwd, SUM grads over n_accum
            o.zero_grad(); c=0
            for j in idx:
                crit(m(Xtr[j:j+1]),ytr[j:j+1]).backward(); c+=1
                if c%nacc==0: o.step(); o.zero_grad()
            if c%nacc: o.step()
        else:                                   # true mini-batch (joint BN over `batch` windows)
            for s in range(0,N,batch):
                b=idx[s:s+batch]; o.zero_grad(); crit(m(Xtr[b]),ytr[b]).backward(); o.step()
    if frozen: reset_rs(m)
    a=bacc(m); print('%-50s -> %6.2f%%  (Δ%+.2f)'%(tag,a,a-ZS)); return a

print('[device-faithful — independent PyTorch check]')
run('A SGD batch1 n_accum8, FROZEN-RS  (=on-device)','sgd',1,8,40,1e-3,True)
run('B SGD batch1 n_accum8, UPDATED-RS','sgd',1,8,40,1e-3,False)
print('[relax constraints toward the paper (true batch enabled)]')
run('C SGD  true batch-8,  updated-RS','sgd',8,1,40,1e-3,False)
run('D SGD  true batch-32, updated-RS','sgd',32,1,40,1e-3,False)
run('E ADAM true batch-32, updated-RS  (paper optimizer)','adam',32,1,40,1e-3,False)
run('F ADAM true batch-32, 50ep         (paper epochs)','adam',32,1,50,1e-3,False)
# Result: A 55.00(-23.33) B 68.89(-9.44) C/D 85.56(+7.22) E 83.33(+5.00) F 83.89(+5.56)
# => dominant recoverable axis is the TRUE BATCH (B->C, +16.7pp); n_accum cannot replicate it.
