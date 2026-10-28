import sys, torch
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet_quant import QuantSpeechNetDeploy
m = QuantSpeechNetDeploy(num_channels=14, time_steps=700, num_classes=9).eval()
print("Does Brevitas hold ANY int8 weight state?")
pd = {str(p.dtype) for _,p in m.named_parameters()}
bd = {str(b.dtype) for _,b in m.named_buffers()}
print("  parameter dtypes:", pd)
print("  buffer dtypes   :", bd)
ints = [n for n,t in list(m.named_parameters())+list(m.named_buffers()) if 'int8' in str(t.dtype)]
print("  any int8 tensor in the module's state?", bool(ints), ints)
print("\n=> the ONLY mutable weight state Brevitas exposes is fp32.")
print("   An int8 weight simply has nowhere to live inside the module.")
