import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper
TR='/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_head_ep40_isft_fw'
B2='/app/TrainDeeploy/DeeployTest/Tests/Models/speechnet_infer_b2_isft_fw'
zin=np.load(B2+'/inputs.npz')
print('infer inputs.npz keys:', zin.files)
X=zin['input'].astype(np.float32); y=zin['label'].astype(np.int64).ravel(); cls=sorted(set(y.tolist()))
zt=np.load(TR+'/outputs.npz'); fcw=zt['fc_weight']; fcb=zt['fc_bias']

def run(fc_w,fc_b,tag):
    m=onnx.load(TR+'/network_infer.onnx')
    for i in m.graph.initializer:
        a=numpy_helper.to_array(i)
        if a.shape==(9,32): i.CopyFrom(numpy_helper.from_array(fc_w.astype(a.dtype),i.name))
        elif a.shape==(9,): i.CopyFrom(numpy_helper.from_array(fc_b.astype(a.dtype),i.name))
    s=ort.InferenceSession(m.SerializeToString(),providers=['CPUExecutionProvider'])
    inp=s.get_inputs()[0].name
    p=np.array([int(np.argmax(s.run(None,{inp:X[k:k+1]})[0])) for k in range(len(y))])
    ba=100*float(np.mean([(p[y==c]==c).mean() for c in cls]))
    print('%-30s balanced acc = %.2f%%'%(tag,ba))

m0=onnx.load(TR+'/network_infer.onnx')
pw=pb=None
for i in m0.graph.initializer:
    a=numpy_helper.to_array(i)
    if a.shape==(9,32): pw=a.copy()
    elif a.shape==(9,): pb=a.copy()
run(pw,pb,'ORT zero-shot fc (sanity)')
run(fcw,fcb,'ORT-trained fc (FT target)')
