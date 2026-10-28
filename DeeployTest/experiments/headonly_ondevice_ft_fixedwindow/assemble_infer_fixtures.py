"""Assemble on-device infer fixtures for the fixed-window / inter_session_ft re-run.

Each fixture = the training fixture's FOLDED head-only forward graph (network_infer.onnx,
frozen conv bit-exact to the on-device feature extractor) + that batch's eval windows +
ORT reference logits (key 'output').

Optionally overrides the fc initializers (for the fine-tuned fixture, with device-dumped fc).
"""
import sys, os, shutil
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper

TESTS = '/app/TrainDeeploy/DeeployTest/Tests/Models'
TR = TESTS + '/Training/SpeechNet/speechnet_train_head_ep40_isft_fw'
INFER_NET = TR + '/network_infer.onnx'   # folded, head-only, pretrained fc


def build(name, windows_dir, fc_w=None, fc_b=None):
    out = os.path.join(TESTS, name)
    os.makedirs(out, exist_ok=True)
    m = onnx.load(INFER_NET)
    if fc_w is not None:
        for i in m.graph.initializer:
            a = numpy_helper.to_array(i)
            if a.shape == (9, 32):
                i.CopyFrom(numpy_helper.from_array(fc_w.astype(a.dtype), i.name))
            elif a.shape == (9,):
                i.CopyFrom(numpy_helper.from_array(fc_b.astype(a.dtype), i.name))
    onnx.save(m, os.path.join(out, 'network.onnx'))
    # windows
    zin = np.load(os.path.join(windows_dir, 'inputs.npz'))
    X = zin['input'].astype(np.float32); y = zin['label'].astype(np.int64)
    np.savez(os.path.join(out, 'inputs.npz'), input=X, label=y)
    # ORT reference logits
    s = ort.InferenceSession(os.path.join(out, 'network.onnx'), providers=['CPUExecutionProvider'])
    inp = s.get_inputs()[0].name
    logits = np.concatenate([s.run(None, {inp: X[k:k+1]})[0] for k in range(len(y))], axis=0).astype(np.float32)
    np.savez(os.path.join(out, 'outputs.npz'), output=logits)
    # report host accuracy from these ORT logits
    p = logits.argmax(1); cls = sorted(set(y.tolist()))
    ba = 100*float(np.mean([(p[y==c]==c).mean() for c in cls]))
    print('built %-32s : %d windows, ORT balanced acc = %.2f%%' % (name, len(y), ba))


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'zs'
    if which == 'zs':
        build('speechnet_infer_b1_zs_isft_fw', TESTS + '/speechnet_infer_b1_isft_fw')
        build('speechnet_infer_b2_zs_isft_fw', TESTS + '/speechnet_infer_b2_isft_fw')
    elif which == 'ft':
        # device-dumped fc passed via env FCW/FCB paths (npy)
        fc_w = np.load(os.environ['FCW']); fc_b = np.load(os.environ['FCB'])
        build('speechnet_infer_b2_ft_isft_fw', TESTS + '/speechnet_infer_b2_isft_fw', fc_w, fc_b)
