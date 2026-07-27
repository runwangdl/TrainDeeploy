# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
from typing import Callable, Dict, List, Tuple, Type

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.CommonExtensions.OptimizationPasses.BindingsOptimizationPasses.AutoTranspose import AutoTransposeMergePass
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    PULPNCHWtoNHWCPass, RemoveGlobalOutputReshapePass, TransposeMatmulInputsPass
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentPlatform, NodeTemplate, TopologyOptimizer, VariableBuffer
from Deeploy.Targets.GAP9.Platform import GAP9ClusterEngine
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import ReshapeConstOptPass, TransposeConstOptPass, \
    TransposeMergePass, TransposeNoPermOptPass, TransposeSplitPass
from Deeploy.Targets.PULPOpen.Platform import PULPClusterEngine
from Deeploy.Targets.PULPOpen.TopologyOptimizationPasses.Passes import RQAddTransposeSquashPass

_L3AllocTemplate = NodeTemplate("""
${locPtr} = cl_ram_malloc(${size});
""")

_L3InitTemplate = NodeTemplate("""
load_file_to_ram(${locPtr}, "${extName}.hex");
""")

_L3AliasTemplate = NodeTemplate("""
${locPtr} = ${srcPtr};
""")


class PULPDeployer(SignPropDeployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first = False,
                 deeployStateDir: str = "DeeployStateDir",
                 inputOffsets = {}):
        super().__init__(graph,
                         deploymentPlatform,
                         inputTypes,
                         loweringOptimizer,
                         scheduler,
                         name,
                         default_channels_first = default_channels_first,
                         deeployStateDir = deeployStateDir,
                         inputOffsets = inputOffsets)

        self.loweringOptimizer.passes += [
            TransposeMatmulInputsPass(),
            PULPNCHWtoNHWCPass(self.default_channels_first),
            TransposeSplitPass(),
            RQAddTransposeSquashPass(),
            TransposeSplitPass(),
            TransposeMergePass(),
            TransposeConstOptPass(),
            ReshapeConstOptPass(),
            TransposeNoPermOptPass(),
            RemoveGlobalOutputReshapePass(),
        ]

        self.extNameCount = 0

    def annotateNCores(self) -> None:
        for layer in self.layerBinding.values():
            node = layer.node
            engine = self._selectEngine(node)
            opRepr = layer.mapper.parser.operatorRepresentation
            if isinstance(engine, (PULPClusterEngine, GAP9ClusterEngine)):
                opRepr["n_cores"] = engine.n_cores

    def bind(self) -> bool:
        # SCHEREMO: THIS IS A STOP GAP SOLUTION. DONT REUSE. I MEAN IT. I WILL FIND YOU.
        # SCHEREMO: The BindingOptimizationPass system is fairly fragile;
        # it was designed this way because implementing further topology optimizations after
        # parsing is very involved. If there are further use-cases, we should consider making this effort,
        # but if there is only very few cases, this solution is okay.
        autoTransposePass = AutoTransposeMergePass()
        #self.ctxt, self.layerBinding = autoTransposePass.apply(self.ctxt, self.graph, self.layerBinding)

        # LMACAN: THIS IS A STOP GAP SOLUTION. DONT REUSE. I MEAN IT. I WILL FIND YOU.
        self.annotateNCores()

        # SCHEREMO: THIS IS A STOP GAP SOLUTION. DONT REUSE. I MEAN IT. I WILL FIND YOU.
        if not super().bind():
            return False

        self.ctxt.hoistGlobalDefinition("cluster_dev", "extern struct pi_device cluster_dev;")
        return True

    def _l3ConstBuffer(self) -> List[VariableBuffer]:
        return [
            buf for buf in self.ctxt.globalObjects.values() if all([
                isinstance(buf, VariableBuffer) and buf._deploy,
                hasattr(buf, "_users") and len(buf._users) > 0,
                hasattr(buf, "_memoryLevel") and buf._memoryLevel == "L3",
            ])
        ]

    def _newExtName(self) -> str:
        name = str(self.extNameCount)
        self.extNameCount += 1
        return name

    def generateBufferAllocationCode(self) -> str:
        retStr = super().generateBufferAllocationCode()

        L3FileStr = ""
        globalConstBuffers = [
            buf for key, buf in self.ctxt.globalObjects.items() if isinstance(buf, VariableBuffer) and buf._deploy
        ]
        nonArenaBuffers = [buf for buf in globalConstBuffers if buf._users != []]
        outputBuffNames = [outputBuffer.name for outputBuffer in self.graph.outputs]

        l3ConstBuffer = []
        for buf in nonArenaBuffers:
            if hasattr(buf, "_memoryLevel") and buf._memoryLevel == "L3" and not buf.name in outputBuffNames:
                l3ConstBuffer.append(buf)

        # Constants with byte-identical contents get one L3 allocation, not one each.
        #
        # `_duplicateConstants` gives every consumer of a multi-consumer constant its own
        # copy, which the tiler needs -- it builds one geometry per buffer, and a single
        # buffer consumed by two ops with different tile shapes is infeasible. But only
        # the tiling descriptor has to be duplicated; the bytes are read-only and can be
        # shared. Without that, every frozen weight is stored once per consumer, and a
        # frozen weight always has at least two (the forward op and its gradient, which
        # needs W to produce dL/dx).
        #
        # Measured on GAP9, L3 peak, address-deduplicated: duplicates are 58% of the
        # persistent region of a loralib-style conv-LoRA ResNet8 (607KB of 1046KB) and
        # 94% of the same model in PEFT form (1276KB of 1356KB). Transformers are barely
        # affected (CCT 5.5%, CCT-LoRA 1%) because their weight transposes are constant-
        # folded, so forward and backward read different tensors.
        #
        # Aliases also skip load_file_to_ram, so boot-time transfers drop by the same
        # amount, and they reuse the source's extName so no duplicate .hex is written.
        seenConst: Dict[str, Tuple[str, str]] = {}

        for idx, buf in enumerate(l3ConstBuffer):

            locPtr = str(buf._instance)
            size = np.prod(buf.shape) * (buf._type.referencedType.typeWidth // 8)

            if isinstance(buf, ConstantBuffer):
                values = np.ascontiguousarray(np.asarray(buf.values))
                key = f"{values.dtype.str}|{values.shape}|{hashlib.md5(values.tobytes()).hexdigest()}"
                previous = seenConst.get(key)
                if previous is not None:
                    srcPtr, srcExtName = previous
                    buf.extName = srcExtName
                    L3FileStr += _L3AliasTemplate.generate({"locPtr": locPtr, "srcPtr": srcPtr})
                    continue
                extName = str(idx)
                buf.extName = extName
                seenConst[key] = (locPtr, extName)
                L3FileStr += _L3AllocTemplate.generate({"locPtr": locPtr, "extName": extName, "size": size})
            else:
                extName = str(idx)
                buf.extName = extName

            L3FileStr += _L3InitTemplate.generate({"locPtr": locPtr, "extName": extName, "size": size})

        retStr = retStr + L3FileStr

        return retStr
