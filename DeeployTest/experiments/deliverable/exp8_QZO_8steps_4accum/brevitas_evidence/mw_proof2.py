import sys, torch
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet_quant import QuantSpeechNetDeploy
import brevitas.nn as qnn
m = QuantSpeechNetDeploy(num_channels=14, time_steps=700, num_classes=9).eval()
conv = [x for x in m.modules() if isinstance(x, qnn.QuantConv2d)][1]
elem=(0,0,0,0)
s0=float(conv.quant_weight().scale.flatten()[0])
cont0=conv.weight.data[elem].item()/s0
print(f"start: W/s_w = {cont0:.4f}  -> int8 = {round(cont0)}   (fractional pos in bin = {cont0-round(cont0):+.3f})")
print("nudging the fp32 master by +0.06 LSB per step (the size of our real ZO update):\n")
step=0.06*s0; prev=round(cont0)
for i in range(1,16):
    with torch.no_grad(): conv.weight.data[elem]+=step
    s=float(conv.quant_weight().scale.flatten()[0])          # scale RE-derived each call
    cont=conv.weight.data[elem].item()/s
    q=round(cont)
    flag="   <-- int8 FLIPPED" if q!=prev else ""
    if i%3==0 or flag: print(f"  step {i:2d}: master W/s_w = {cont:.4f}  int8 = {q}{flag}")
    prev=q
print(f"\n=> {round(cont)-round(cont0)} LSB of real int8 movement, produced entirely by accumulating")
print("   sub-LSB (0.06 LSB) updates in the fp32 parameter. No single step could move the int8 alone.")
print(f"\nBONUS (relevant to freezing): s_w drifted {s0:.8f} -> {s:.8f} as the weight changed,")
print("   i.e. vanilla Brevitas re-derives the WEIGHT scale from the parameter every call.")
