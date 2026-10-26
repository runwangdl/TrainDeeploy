"""exp21e shared helpers: rebuild the exact exp17/exp18 PyTorch references (agitated_hugle)."""
import sys, numpy as np, torch, onnx
from onnx import numpy_helper
sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy  # noqa: E402
from onnx4deeploy.utils.onnx_node_implementations import _perturb_rademacher       # noqa: E402

FIX = "/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet"
ZO_SRC = "/app/Onnx4Deeploy/onnx/model/speechnet_zo_b1_fold3_ref"
B = "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift"


def load_windows(fixture):
    i = np.load(f"{FIX}/{fixture}/inputs.npz")
    n = int(i["meta_data_size"][0])
    X = [i["arr_0000"]] + [i[f"mb{k}_arr_0000"] for k in range(1, n)]
    Y = [i["arr_0001"]] + [i[f"mb{k}_arr_0001"] for k in range(1, n)]
    return [np.asarray(x, np.float32) for x in X], [np.asarray(y).reshape(-1).astype(np.int64) for y in Y]


def init_map_from(onnx_path):
    return {t.name: numpy_helper.to_array(t) for t in onnx.load(onnx_path).graph.initializer}


def build_model(init_map, dtype=torch.float32):
    m = SpeechNetDeploy(num_channels=14, time_steps=700, num_classes=9)
    sd = m.state_dict()
    for t in list(sd.keys()):
        o = t.replace(".", "_")
        if o in init_map:
            sd[t] = torch.from_numpy(np.asarray(init_map[o])).float().reshape(sd[t].shape)
    m.load_state_dict(sd)
    m.eval()
    return m.to(dtype)


def bp_init_map():
    """BP fixture initial weights: the 22 trainables in inputs.npz arr_0002..arr_0023 + frozen BN stats from
    the infer graph of the ZO export (same base checkpoint)."""
    im = init_map_from(f"{ZO_SRC}/network_infer.onnx")
    i = np.load(f"{FIX}/speechnet_train_fullfrozen_b1_fold3_ref/inputs.npz")
    names = [k for k in im if "running" not in k]
    for j, k in enumerate(names):
        a = i[f"arr_{j+2:04d}"]
        assert a.shape == im[k].shape, (k, a.shape, im[k].shape)
        im[k] = a
    return im


def zo_node_ids():
    g = onnx.load(f"{ZO_SRC}/network_zo_train.onnx").graph
    return {n.input[0]: next(a.i for a in n.attribute if a.name == "idx") for n in g.node if "Perturb" in n.op_type}


def f32bits(x):
    return np.asarray(x, np.float32).view(np.uint32)
