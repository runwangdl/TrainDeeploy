import sys, torch
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet_quant import QuantSpeechNetDeploy
import brevitas.nn as qnn
m = QuantSpeechNetDeploy(num_channels=14, time_steps=700, num_classes=9).eval()
conv=[x for x in m.modules() if isinstance(x,qnn.QuantConv2d)][1]
s_w = conv.quant_weight().scale.detach().clone()          # FROZEN scale, captured once
def fake_quant(W,s): return torch.clamp(torch.round(W/s),-127,127)*s

W_latent = conv.weight.data.clone()
# simulate a master-weight drift then install like loss_with_weights does
W_latent = W_latent + 0.6*s_w*torch.sign(torch.randn_like(W_latent))
Wq = fake_quant(W_latent, s_w)
conv.weight.data = Wq
s_after = conv.quant_weight().scale.detach()
q_intended = torch.round(Wq/s_w)
q_actual   = torch.round(conv.quant_weight().value.detach()/s_after)
print("frozen s_w[0]      = %.10f" % s_w.flatten()[0])
print("brevitas s_w after = %.10f" % s_after.flatten()[0])
print("scale changed?     ", not torch.allclose(s_w, s_after))
print("max |intended_int - actual_int| =", float((q_intended-q_actual).abs().max()))
print("effective weight matches intended?",
      torch.allclose(conv.quant_weight().value.detach(), Wq, atol=1e-9))
