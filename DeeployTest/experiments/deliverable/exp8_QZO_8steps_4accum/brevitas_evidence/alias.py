import sys, torch
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet_quant import QuantSpeechNetDeploy
import brevitas.nn as qnn
m=QuantSpeechNetDeploy(num_channels=14,time_steps=700,num_classes=9).eval()
c=[x for x in m.modules() if isinstance(x,qnn.QuantConv2d)][1]

A_latent = torch.zeros_like(c.weight.data) + 0.5      # pretend this is our master weight
c.weight.data = A_latent                              # exactly what loss_with_weights :186 does
print("after `m.weight.data = A_latent`:")
print("  is m.weight.data the SAME OBJECT as A_latent? ", c.weight.data is A_latent)
print("  -> assignment does NOT copy; they alias the same storage")
# demonstrate the hazard: an in-place write through the model corrupts our master
c.weight.data.mul_(0.0)
print("  after an in-place op on m.weight.data, A_latent[0,0,0,0] =", float(A_latent.flatten()[0]),
      " <-- our master got clobbered")
