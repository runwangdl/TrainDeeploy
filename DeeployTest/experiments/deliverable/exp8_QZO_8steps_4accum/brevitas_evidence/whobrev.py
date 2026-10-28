import sys, torch
sys.path.insert(0,"/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet_quant import QuantSpeechNetDeploy
import brevitas.nn as qnn
torch.manual_seed(0)
m = QuantSpeechNetDeploy(num_channels=14, time_steps=700, num_classes=9).train()
conv = [x for x in m.modules() if isinstance(x, qnn.QuantConv2d)][1]

print("### A. What an optimizer captures when you 'fine-tune in Brevitas' ###")
opt = torch.optim.SGD(m.parameters(), lr=1e-3)
captured = opt.param_groups[0]['params']
print("  optimizer holds", len(captured), "tensors; all fp32?",
      all(p.dtype==torch.float32 for p in captured))
print("  is conv.weight literally one of them? ", any(p is conv.weight for p in captured))
print("  -> the optimizer's update target IS the fp32 nn.Parameter conv.weight")

print("\n### B. One real fine-tune step: backward + step ###")
w_before = conv.weight.detach().clone()
q_before = torch.round(conv.weight.detach()/conv.quant_weight().scale)
x = torch.randn(2,1,14,700); y = torch.tensor([0,1])
loss = torch.nn.functional.cross_entropy(m(x), y)
loss.backward()
print("  conv.weight.grad exists?", conv.weight.grad is not None,
      "| grad dtype:", conv.weight.grad.dtype)
opt.step()
print("  conv.weight changed?", not torch.equal(w_before, conv.weight.detach()),
      "| still dtype:", conv.weight.dtype)
print("  max |Δ fp32 weight| = %.3e" % (conv.weight.detach()-w_before).abs().max())
print("  -> accumulation landed in the fp32 parameter (via torch/optim/sgd.py: param.add_)")

print("\n### C. Can we instead make Brevitas STORE int8? ###")
s = conv.quant_weight().scale.detach()
int8_tensor = torch.clamp(torch.round(conv.weight.detach()/s), -127, 127).to(torch.int8)
try:
    conv.weight.data = int8_tensor          # try to keep int8 as the state
    _ = m(x)
    print("  forward with int8 stored: SUCCEEDED (unexpected)")
except Exception as e:
    print("  forward with int8 stored FAILED ->", type(e).__name__+":", str(e)[:110])
print("  -> an int8 weight cannot live in m.weight; the module's math is fp32")
