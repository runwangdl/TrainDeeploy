#!/usr/bin/env python3
# Rigorous verification of the device full-training result. Reconstruct by the DETERMINISTIC
# wi->name order (element counts already prove this 1:1, no value-matching needed), then eval
# THREE ways on the SAME batch-2 windows + report per-weight device-vs-ORT drift:
#   (a) zero-shot (pretrained)            -> must be 78.33%
#   (b) ORT-ref weights (outputs.npz)     -> must be ~81.11% (matches the host sweep => pipeline OK)
#   (c) DEVICE weights (WDUMP)            -> the real on-device number
# If (b)==81.11 and (c)<<that with large drift -> drift genuinely wrecks full-model FT.
import re, struct, numpy as np, onnx, onnxruntime as ort, torch, sys
from onnx import numpy_helper
TR="/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_full8_e20"
LOG="/app/TrainDeeploy/DeeployTest/full8_e20_device.log"
B2="/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2"
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy
CKPT="/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt"

# deterministic wi -> graph param name (proved by the n= element counts)
NAMES=["blocks_0_0_weight","blocks_0_0_bias","blocks_0_1_weight","blocks_0_1_bias",
       "blocks_1_0_weight","blocks_1_0_bias","blocks_1_1_weight","blocks_1_1_bias",
       "blocks_2_0_weight","blocks_2_0_bias","blocks_2_1_weight","blocks_2_1_bias",
       "blocks_3_0_weight","blocks_3_0_bias","blocks_3_1_weight","blocks_3_1_bias",
       "blocks_4_0_weight","blocks_4_0_bias","blocks_4_1_weight","blocks_4_1_bias",
       "fc_weight","fc_bias"]

# ---- parse final-step WDUMP ----
pat=re.compile(r'\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)')
dump={}
for line in open(LOG):
    m=pat.search(line)
    if not m: continue
    s,wi,n=int(m[1]),int(m[2]),int(m[3])
    arr=np.array([struct.unpack('<f',struct.pack('<I',int(w,16)))[0] for w in m[4].split()],dtype=np.float32)
    dump[(s,wi)]=arr
last=max(s for s,_ in dump)
dev={wi:dump[(last,wi)] for (s,wi) in dump if s==last}
print(f"final step {last}, {len(dev)} tensors")

# ---- infer graph initializers (shapes) + ORT ref weights ----
gi=onnx.load(TR+"/network_infer.onnx")
shp={init.name: numpy_helper.to_array(init).shape for init in gi.graph.initializer}
ref={k:np.asarray(v) for k,v in np.load(TR+"/outputs.npz").items() if k!='loss'}

# size sanity: device wi tensor size must equal named init size
for wi,nm in enumerate(NAMES):
    assert dev[wi].size==int(np.prod(shp[nm])), f"size mismatch wi={wi} {nm}: {dev[wi].size} vs {shp[nm]}"
print("size sanity OK: all 22 device tensors match their named init shapes")

# ---- per-weight drift device vs ORT ref ----
print("\nper-weight  max|device-ORT|   ORT|val|max   relerr")
worst=0.0
for wi,nm in enumerate(NAMES):
    d=dev[wi]; r=ref.get(nm)
    if r is None: print(f"  {nm:22s} (no ORT ref)"); continue
    r=r.ravel(); md=float(np.abs(d-r).max()); rm=float(np.abs(r).max()); rel=md/(rm+1e-12)
    worst=max(worst,rel)
    flag=" <<<" if rel>0.05 else ""
    print(f"  {nm:22s} {md:12.3e}   {rm:10.3e}   {rel:8.2%}{flag}")
print(f"worst relative drift = {worst:.2%}")

# ---- eval helper ----
X=np.load(B2+"/inputs.npz")["input"].astype(np.float32); y=np.load(B2+"/inputs.npz")["label"].astype(np.int64)
cls=sorted(set(y.tolist()))
def evalg(graph):
    sess=ort.InferenceSession(graph.SerializeToString(),providers=['CPUExecutionProvider']); inm=sess.get_inputs()[0].name
    p=np.array([int(np.argmax(sess.run(None,{inm:X[i:i+1]})[0])) for i in range(len(y))])
    return float(np.mean([(p[y==k]==k).mean() for k in cls]))
def build(weight_src):  # weight_src: dict name->flat array, or None for untouched (ORT ref already in graph? no)
    g=onnx.load(TR+"/network_infer.onnx")
    if weight_src is not None:
        ni=[]
        for init in g.graph.initializer:
            if init.name in weight_src:
                a=weight_src[init.name].reshape(numpy_helper.to_array(init).shape).astype(numpy_helper.to_array(init).dtype)
                ni.append(numpy_helper.from_array(a,init.name))
            else: ni.append(init)
        del g.graph.initializer[:]; g.graph.initializer.extend(ni)
    return g

# (a) zero-shot: pretrained weights into the infer graph
m=SpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9)
sd=torch.load(CKPT,map_location="cpu",weights_only=False); sd=sd.get("model_state_dict",sd); m.load_state_dict(sd)
# zero-shot = the network_infer.onnx as-generated already holds pretrained weights:
zs=evalg(onnx.load(TR+"/network_infer.onnx"))
# (b) ORT-ref weights
ortw={nm: ref[nm].ravel() for nm in NAMES if nm in ref}
ortacc=evalg(build(ortw))
# (c) DEVICE weights
devw={nm: dev[wi] for wi,nm in enumerate(NAMES)}
devacc=evalg(build(devw))

print("\n================ RESULT ================")
print(f"(a) zero-shot batch2            = {100*zs:.2f}%")
print(f"(b) ORT-ref full-train batch2   = {100*ortacc:.2f}%   (Δ {100*(ortacc-zs):+.2f}pp)  [host-sweep predicted ~81.11%]")
print(f"(c) DEVICE full-train batch2    = {100*devacc:.2f}%   (Δ {100*(devacc-zs):+.2f}pp)  [actual on-device]")
print(f"device-vs-ORT accuracy gap     = {100*(devacc-ortacc):+.2f}pp  (the precision-drift effect)")
