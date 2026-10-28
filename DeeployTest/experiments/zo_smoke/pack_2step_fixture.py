# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
# -- QW: Split an Onnx4Deeploy ZO export dir into the TrainDeeploy two-dir test-fixture layout.
#
# Onnx4Deeploy `-mode zo-train` now emits both graphs with the trainable weights as graph INPUTS
# (byte-structurally like a BP training graph), so this is a pure SPLIT — no promotion, no re-bake:
#   network_zo_train.onnx  (weights as inputs + BatchNormInternal + canonical SCE)  -> <name>_zo_train/network.onnx
#   network_zo_update.onnx (weights as inputs, `*_updated` outputs)                 -> <name>_zo_update/network.onnx
#   inputs.npz  (input, label, the 22 initial weights arr_0002.., meta, mb{k})      -> BOTH dirs (identical)
#   outputs.npz (22 final weights + log_prob + per-step loss_plus/minus)            -> BOTH dirs (identical)
# TrainDeeploy's ZO runner then consumes each dir directly (no deploy-time promotion). -- QW
import sys, shutil
from pathlib import Path

EXPORT = Path(sys.argv[1])          # .../onnx/model/<export>
DEST = Path(sys.argv[2])            # .../Tests/Models/Training/SpeechNet
TRAIN = DEST / sys.argv[3]          # e.g. speechnet_zo_train_2step
UPDATE = DEST / sys.argv[4]         # e.g. speechnet_zo_update_2step

for d, onnx_name in ((TRAIN, "network_zo_train.onnx"), (UPDATE, "network_zo_update.onnx")):
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(EXPORT / onnx_name, d / "network.onnx")
    shutil.copy(EXPORT / "inputs.npz", d / "inputs.npz")
    shutil.copy(EXPORT / "outputs.npz", d / "outputs.npz")

import onnx
zt = onnx.load(str(TRAIN / "network.onnx")).graph
zu = onnx.load(str(UPDATE / "network.onnx")).graph
print(f"train  : {TRAIN}  (network.onnx {len(zt.input)} inputs / {len(zt.initializer)} inits)")
print(f"update : {UPDATE} (network.onnx {len(zu.input)} inputs / {len(zu.output)} outputs)")
