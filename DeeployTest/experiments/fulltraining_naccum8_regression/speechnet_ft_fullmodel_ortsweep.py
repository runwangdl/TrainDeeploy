#!/usr/bin/env python3
# Full-model FT in the predictive ORT space (training-strategy=full, the default), to test
# whether ANY full-training config beats batch-2 zero-shot -- the comparison missing from
# the report (the prior full-model sweep was the non-predictive PyTorch sim). Generate the
# real Deeploy/ORT training graph per config, eval its outputs.npz (ALL weights) on batch-2.
import subprocess, numpy as np, torch, sys, itertools
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy
B2="/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2"
CKPT="/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt"
DATA="/app/SilentWear/SilentWear_data/data_raw_and_filt"
X=np.load(B2+"/inputs.npz")["input"].astype(np.float32); y=np.load(B2+"/inputs.npz")["label"].astype(np.int64)
sd0=torch.load(CKPT,map_location="cpu",weights_only=False); sd0=sd0.get("model_state_dict",sd0)
def mapk(k): return ('fc.'+k[3:]) if k.startswith('fc_') else 'blocks.%s.%s.%s'%(tuple(k.split('_')[1:3])+('_'.join(k.split('_')[3:]),))
def bal(weights):
    m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9); m.load_state_dict(sd0,strict=True)
    sd={k:v.clone() for k,v in m.state_dict().items()}
    if weights is not None:
        for k in weights.files:
            if k=='loss': continue
            sd[mapk(k)]=torch.from_numpy(weights[k].copy())
    m.load_state_dict(sd,strict=True); m.eval()
    p=np.array([m(torch.from_numpy(X[i:i+1])).argmax(1).item() for i in range(len(y))])
    c=sorted(set(y.tolist())); return float(np.mean([(p[y==k]==k).mean() for k in c]))
zs=bal(None); print("zero-shot batch2=%.2f%%"%(100*zs),flush=True); print("="*70,flush=True)
def gen(lr,ep,ds,od):
    subprocess.run(["python3","/app/Onnx4Deeploy/Onnx4Deeploy.py","-model","SpeechNet","-mode","train","-o",od,
      "--dataset","silentwear","--data-path",DATA,"--pretrained-weights",CKPT,
      "--subject","S01","--session","3","--batch","1","--condition","vocalized",
      "--stratified","--data-size",str(ds),"--n-epochs",str(ep),"--n-accum","4","--lr",str(lr)],
      cwd="/app/Onnx4Deeploy",stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)   # NOTE: no --training-strategy => full
res=[]
for ds,lr,ep in itertools.product([54,18],[1e-3,5e-3,1e-2],[10,40]):
    od=f"/tmp/full_d{ds}_lr{lr}_e{ep}"; gen(lr,ep,ds,od)
    try:
        o=np.load(od+"/outputs.npz"); a=bal(o); res.append((ds,lr,ep,a))
        print("FULL data=%2d lr=%-6s ep=%2d -> batch2=%.2f%%  Δ=%+.2fpp  (loss ep1=%.2f epN=%.2f)"%(
            ds,lr,ep,100*a,100*(a-zs),o['loss'][:ds].mean(),o['loss'][-ds:].mean()),flush=True)
    except Exception as e: print("FULL data=%d lr=%s ep=%d FAILED %s"%(ds,lr,ep,e),flush=True)
print("="*70,flush=True)
if res:
    b=max(res,key=lambda r:r[3]); pos=[r for r in res if r[3]>zs]
    print("BEST full-model: data=%d lr=%s ep=%d -> %.2f%% (Δ%+.2fpp)"%(b[0],b[1],b[2],100*b[3],100*(b[3]-zs)),flush=True)
    print("configs beating zero-shot: %d / %d"%(len(pos),len(res)),flush=True)
