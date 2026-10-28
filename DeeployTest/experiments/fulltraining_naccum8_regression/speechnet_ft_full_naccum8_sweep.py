#!/usr/bin/env python3
# Phase-1 host/ORT sweep for the FULL-TRAINING on-device-constrained config the user asked for:
#   SGD (no momentum), full training (all 22 params, BN trainable in batch-1 BatchNormInternal),
#   n_accum=8, data-size=54 (30%), eval whole batch-2.  This is the PREDICTIVE method: gen() emits
#   the real Deeploy/ORT training graph + runs the ORT training loop -> outputs.npz (final weights),
#   then bal() evaluates those weights on batch-2 in eval mode (frozen running stats + trained gamma/
#   beta = exactly what the deployed infer graph does). Goal: ceiling of the constrained config vs
#   SilentWear (+8.33pp) and vs zero-shot, and which (epochs,lr) is worth the on-device GVSoC run.
import subprocess, numpy as np, torch, sys, itertools, json
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
    n_over=0
    if weights is not None:
        for k in weights.files:
            if k=='loss': continue
            tk=mapk(k)
            if tk in sd: sd[tk]=torch.from_numpy(weights[k].copy()); n_over+=1
    m.load_state_dict(sd,strict=True); m.eval()
    p=np.array([m(torch.from_numpy(X[i:i+1])).argmax(1).item() for i in range(len(y))])
    c=sorted(set(y.tolist())); return float(np.mean([(p[y==k]==k).mean() for k in c])), n_over
zs,_=bal(None); print("zero-shot batch2=%.2f%%  (SilentWear reference ceiling=+8.33pp)"%(100*zs),flush=True)
print("="*78,flush=True)
def gen(lr,ep,ds,na,od):
    subprocess.run(["python3","/app/Onnx4Deeploy/Onnx4Deeploy.py","-model","SpeechNet","-mode","train","-o",od,
      "--dataset","silentwear","--data-path",DATA,"--pretrained-weights",CKPT,
      "--subject","S01","--session","3","--batch","1","--condition","vocalized",
      "--stratified","--data-size",str(ds),"--n-epochs",str(ep),"--n-accum",str(na),"--lr",str(lr)],
      cwd="/app/Onnx4Deeploy",stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)   # no --training-strategy => FULL
res=[]; NA=8; DS=54
for ep,lr in itertools.product([10,20,40,50],[1e-3,5e-3,1e-2]):
    od=f"/tmp/full8_e{ep}_lr{lr}"; gen(lr,ep,DS,NA,od)
    try:
        o=np.load(od+"/outputs.npz"); a,nov=bal(o)
        L=o['loss']; res.append({"ep":ep,"lr":lr,"acc":a,"d":a-zs,"l0":float(L[:DS].mean()),"lN":float(L[-DS:].mean()),"nov":nov,"od":od})
        print("FULL n_accum=8 data=54 ep=%2d lr=%-6s -> batch2=%.2f%%  Δ=%+.2fpp  (loss %.2f->%.2f, %d params updated)"%(
            ep,lr,100*a,100*(a-zs),float(L[:DS].mean()),float(L[-DS:].mean()),nov),flush=True)
    except Exception as e: print("FULL ep=%d lr=%s FAILED %s"%(ep,lr,e),flush=True)
print("="*78,flush=True)
if res:
    b=max(res,key=lambda r:r["acc"]); pos=[r for r in res if r["d"]>1e-9]
    print("BEST: ep=%d lr=%s -> %.2f%% (Δ%+.2fpp)"%(b["ep"],b["lr"],100*b["acc"],100*b["d"]),flush=True)
    print("configs beating zero-shot: %d/%d"%(len(pos),len(res)),flush=True)
    print("vs SilentWear +8.33pp: best reaches %.0f%% of that ceiling"%(100*b["d"]/0.0833) if b["d"]>0 else "best does NOT beat zero-shot",flush=True)
    json.dump({"zs":zs,"res":[{k:v for k,v in r.items() if k!='od'} for r in res],"best":{k:v for k,v in b.items() if k!='od'}},
              open("/app/TrainDeeploy/DeeployTest/speechnet_ft_full_naccum8_sweep.json","w"),indent=2)
    print("saved -> speechnet_ft_full_naccum8_sweep.json",flush=True)
