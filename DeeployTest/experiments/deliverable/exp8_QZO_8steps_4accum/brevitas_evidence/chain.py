import sys, torch
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet_quant import QuantSpeechNetDeploy
import brevitas.nn as qnn
m=QuantSpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9).eval()
c=[x for x in m.modules() if isinstance(x,qnn.QuantConv2d)][1]
print("live module chain (what our m.weight.data hand-off flows through):")
print("  conv                     :", type(c).__name__)
print("  conv.weight_quant        :", type(c.weight_quant).__name__)
print("  .tensor_quant            :", type(c.weight_quant.tensor_quant).__name__)
print("  .tensor_quant.scaling_impl:", type(c.weight_quant.tensor_quant.scaling_impl).__name__, " <- WE REPLACE THIS (freeze)")
print("  .tensor_quant.int_quant  :", type(c.weight_quant.tensor_quant.int_quant).__name__)
print("  .int_quant.float_to_int_impl:", type(c.weight_quant.tensor_quant.int_quant.float_to_int_impl).__name__)
