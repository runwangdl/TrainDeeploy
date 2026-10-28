#!/usr/bin/env python3
# Host ORT reference for the argmax-flip evidence. Replicates the exact ORT training
# loop that produces the device's reference losses, and computes the SAME MaxPoolGrad
# argmax checksum the on-device kernel computes (core-0 re-scan over all channels:
# for each MaxPool layer, for h_out,w_out,c -> argmax position in_idx (NHWC encoding),
# rolling hash sig = sig*1000003 + (in_idx+1), uint32). Compared step-for-step against
# the device [AMSIG] dumps to prove device-vs-ORT argmax first diverges at the drift onset.
import onnx, numpy as np, onnxruntime as ort
TR="/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_maxpool_90"
LR=0.001; NSTEPS=90
m=onnx.load(TR+"/network_train.onnx"); g=m.graph
in_names=[i.name for i in g.input]
# MaxPool nodes: (input_tensor, Q=kernel_w, SQ=stride_w) in graph (forward) order
mp=[]
for n in g.node:
    if n.op_type=='MaxPool':
        at={a.name:list(a.ints) for a in n.attribute if a.ints}
        mp.append((n.input[0], at['kernel_shape'][1], at['strides'][1]))
# grad tensors from InPlaceAccumulator nodes (param -> grad name), exactly like the exporter
grad_map={}
for n in g.node:
    if 'InPlaceAccumulator' in n.op_type and len(n.input)>=2 and n.input[1].endswith('_grad'):
        grad_map[n.input[1][:-5]]=n.input[1]
# expose MaxPool inputs AND grad tensors as graph outputs
existing_out={o.name for o in g.output}
for t in [t for t,_,_ in mp]+list(grad_map.values()):
    if t not in existing_out:
        g.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
        existing_out.add(t)
sess=ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])
out_names=[o.name for o in sess.get_outputs()]

z=np.load(TR+"/inputs.npz")
# data windows + labels (arr_0000/0001 = mb0, mb{i}_arr_0000/0001 = mb i)
Xs=[z['arr_0000']]; ys=[z['arr_0001']]; i=1
while f"mb{i}_arr_0000" in z.files: Xs.append(z[f"mb{i}_arr_0000"]); ys.append(z[f"mb{i}_arr_0001"]); i+=1
DS=len(Xs); print(f"data_size={DS}")
# initial weights by NAME from network_infer.onnx initializers (as the exporter does)
wnames=[n for n in in_names if n not in ('input','labels','lazy_reset_grad') and 'grad.accumulation' not in n]
infer_init={i.name: onnx.numpy_helper.to_array(i).copy()
            for i in onnx.load(TR+"/network_infer.onnx").graph.initializer}
missing=[n for n in wnames if n not in infer_init]
assert not missing, f"weights missing from network_infer.onnx: {missing}"
W={n: infer_init[n].astype(np.float32) for n in wnames}
gradbufs=[n for n in in_names if 'grad.accumulation' in n]

def argmax_layer(X, Q, SQ):
    # Sum and SumSq of within-window argmax offset (off=argmax_q+1), exactly as the kernel.
    # SpeechNet MaxPools are non-overlapping (stride==kernel, no pad) -> reshape into windows.
    assert SQ == Q
    x = X[0]; C, H, W = x.shape; Wout = W // Q
    xw = x[:, :, :Wout*Q].reshape(C, H, Wout, Q)
    off = xw.argmax(-1).astype(np.int64) + 1      # numpy argmax = first-occurrence max (== kernel strict >)
    return int(off.sum() & 0xFFFFFFFF), int((off*off).sum() & 0xFFFFFFFF)

def feed_for(mb):
    f={'input':Xs[mb%DS].astype(np.float32),'labels':ys[mb%DS].astype(np.int64),
       'lazy_reset_grad':np.array([1],dtype=np.bool_)}
    for n in wnames: f[n]=W[n].astype(np.float32)
    for n in gradbufs:
        shp=[d.dim_value for d in next(i for i in g.input if i.name==n).type.tensor_type.shape.dim]
        f[n]=np.zeros(shp,dtype=np.float32)
    return f

losses=[]; sig=[]; sig2=[]
for mb in range(NSTEPS):
    res=dict(zip(out_names, sess.run(None, feed_for(mb))))
    loss=[v for k,v in res.items() if 'loss' in k.lower() and 'grad' not in k.lower()][0]
    losses.append(float(np.array(loss).flatten()[0]))
    s=0; s2=0
    for t,Q,SQ in mp:                                  # order-independent (sum) -> any order
        a,b=argmax_layer(res[t],Q,SQ); s=(s+a)&0xFFFFFFFF; s2=(s2+b)&0xFFFFFFFF
    sig.append(s); sig2.append(s2)
    # SGD (eff-batch 1): w -= lr*grad  (grad names from InPlaceAccumulator, like the exporter)
    for pname,gname in grad_map.items():
        if pname in W and gname in res: W[pname]=W[pname]-LR*res[gname]

ref=np.load(TR+"/outputs.npz")['loss']
err=np.abs(np.array(losses)-ref[:NSTEPS]).max()
print(f"loss-calibration: max|host_loss - ORT_ref_loss| = {err:.2e}  (should be ~0 -> faithful replica)")
np.savez("/app/TrainDeeploy/DeeployTest/speechnet_argmax_ort_ref.npz",
         loss=np.array(losses), sig=np.array(sig,dtype=np.uint32), sig2=np.array(sig2,dtype=np.uint32))
print("saved ORT argmax sig/sig2. first 5 sig:", [hex(s) for s in sig[:5]])
