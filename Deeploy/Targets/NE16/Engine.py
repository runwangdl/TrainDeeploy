# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import DeploymentEngine, NodeMapper
from Deeploy.Targets.Generic.Layers import ConvLayer
from Deeploy.Targets.NE16.Parsers import NE161xKConv2DParser, NE163x3ChunkConv2DParser  # -- QW
from Deeploy.Targets.NE16.Parsers import NE16DenseConv2DParser, NE16DWConv2DParser, NE16PWConv2DParser, \
    NE16RQSDenseConv2DParser, NE16RQSDWConv2DParser, NE16RQSPWConv2DParser
from Deeploy.Targets.NE16.Tiler import NE161xKConv2DTilingReadyBindings, \
    NE163x3ChunkConv2DTilingReadyBindings  # -- QW
from Deeploy.Targets.NE16.Tiler import NE16DenseConv2DTilingReadyBindings, NE16DWConv2DTilingReadyBindings, \
    NE16PWConv2DTilingReadyBindings, NE16RQSDenseConv2DTilingReadyBindings, NE16RQSDWConv2DTilingReadyBindings, \
    NE16RQSPWConv2DTilingReadyBindings
from Deeploy.Targets.PULPOpen.Layers import PULPRQSConvLayer

NE16RqntPWConv2DMapper = NodeMapper(NE16RQSPWConv2DParser(), NE16RQSPWConv2DTilingReadyBindings)
NE16PWConv2DMapper = NodeMapper(NE16PWConv2DParser(), NE16PWConv2DTilingReadyBindings)

NE16RqntDWConv2DMapper = NodeMapper(NE16RQSDWConv2DParser(), NE16RQSDWConv2DTilingReadyBindings)
NE16DWConv2DMapper = NodeMapper(NE16DWConv2DParser(), NE16DWConv2DTilingReadyBindings)

NE16RqntDenseConv2DMapper = NodeMapper(NE16RQSDenseConv2DParser(), NE16RQSDenseConv2DTilingReadyBindings)
NE16DenseConv2DMapper = NodeMapper(NE16DenseConv2DParser(), NE16DenseConv2DTilingReadyBindings)

NE161xKConv2DMapper = NodeMapper(NE161xKConv2DParser(), NE161xKConv2DTilingReadyBindings)  # -- QW
NE163x3ChunkConv2DMapper = NodeMapper(NE163x3ChunkConv2DParser(),  # -- QW (exp16b)
                                      NE163x3ChunkConv2DTilingReadyBindings)  # -- QW

# -- QW (dual_pass_fusion exp7 step 7): both ZO passes of a 1xK / Kx1 conv in one NE16 node (pipelined). -- QW
from Deeploy.Targets.NE16.DualConv2in import NE16DualConv2inLayer, NE16DualConv2inParser  # -- QW
from Deeploy.Targets.NE16.Tiler import NE16DualConv2inTilingReadyBindings  # -- QW
NE16DualConv2inMapper = NodeMapper(NE16DualConv2inParser(), NE16DualConv2inTilingReadyBindings)  # -- QW
from Deeploy.Targets.NE16.DualConv2in import NE16DualConv2inStackedParser  # -- QW exp8
from Deeploy.Targets.NE16.Tiler import NE16DualConv2inStackedTilingReadyBindings  # -- QW exp8
NE16DualConv2inStackedMapper = NodeMapper(NE16DualConv2inStackedParser(), NE16DualConv2inStackedTilingReadyBindings)  # -- QW exp8

NE16Mapping = {
    'RequantizedConv': PULPRQSConvLayer([NE16RqntPWConv2DMapper, NE16RqntDWConv2DMapper, NE16RqntDenseConv2DMapper]),
    'NE16DualConv2in': NE16DualConv2inLayer([NE16DualConv2inMapper, NE16DualConv2inStackedMapper]),  # -- QW exp7 + exp8
    'Conv': ConvLayer([NE163x3ChunkConv2DMapper, NE161xKConv2DMapper,  # -- QW
                       NE16PWConv2DMapper, NE16DWConv2DMapper,
                       NE16DenseConv2DMapper]),
}

_includeList = ["pulp_nnx_ne16.h", "pulp_nnx_util.h", "ne16_pulp_bsp.h", "ne16.h", "ne16_task.h"]

_ne16InitCode = r"""
ne16_pulp_conf_t conf = {.max_stall = 8};
ne16_nnx_init(ne16_pulp_get_dev(), &conf);
"""


class NE16Engine(DeploymentEngine):

    def __init__(self,
                 name: str,
                 Mapping = NE16Mapping,
                 initCode: str = _ne16InitCode,
                 includeList: List[str] = _includeList,
                 enable3x3: bool = False,
                 enableStrides: bool = False,
                 enable1xK: bool = False) -> None:  # -- QW
        super().__init__(name, Mapping, initCode, includeList)

        self.enable3x3 = enable3x3
        self.enableStrides = enableStrides
        self.enable1xK = enable1xK  # -- QW: 1xK -> K pointwise dispatches (STEP 2b)

    # QW (exp16a): NE16 must also accept a conv whose weight is a RUNTIME tensor -- our QZO
    #     training graphs feed weights as graph inputs / RQSPerturbRademacher outputs, never as
    #     gs.Constant. Such a node opts in with `ne16_weight_preencoded=1`, which asserts that
    #     the weight ALREADY carries NE16's bit-serial layout (produced host-side by the same
    #     `_weightEncode`) and that the node supplies its own static `weight_offset`.
    #     `_ne16_adjust_weight_memory_layout_fun` already returns early for non-constant weights
    #     (Passes.py:84), so nothing tries to re-encode them at compile time.
    #     This changes only WHICH nodes are offered to NE16 -- no register, no ISA change. -- QW
    @staticmethod
    def _weightAcceptable(node) -> bool:  # -- QW
        return isinstance(node.inputs[1], gs.Constant) or \
            int(node.attrs.get("ne16_weight_preencoded", 0)) == 1

    def isDenseConv(self, node) -> bool:
        return node.op in ["Conv", "RequantizedConv"] and \
            self._weightAcceptable(node) and \
            node.attrs['kernel_shape'] == [3, 3] and \
            node.attrs['dilations'] == [1, 1] and \
            node.attrs['group'] == 1 and \
            (node.attrs['strides'] == [1, 1] or self.enableStrides)

    def isPWConv(self, node) -> bool:
        return node.op in ["Conv", "RequantizedConv"] and \
            self._weightAcceptable(node) and \
            node.attrs['kernel_shape'] == [1, 1] and \
            node.attrs['dilations'] == [1, 1] and \
            (node.attrs['strides'] == [1, 1] or self.enableStrides)

    def isDWConv(self, node) -> bool:
        return node.op in ["Conv", "RequantizedConv"] and \
            self._weightAcceptable(node) and \
            node.attrs['kernel_shape'] == [3, 3] and \
            node.attrs['dilations'] == [1, 1] and \
            node.attrs['group'] != 1 and \
            (node.attrs['strides'] == [1, 1] or self.enableStrides)

    # QW (exp16a / STEP 2b): a 1xK (or Kx1) DENSE convolution is not an NE16 filter mode --
    #     CONFIG0[6:5] offers only 3x3, 3x3-depthwise and 1x1. But a 1xK conv is exactly K
    #     pointwise convolutions over K input windows shifted along W:
    #         conv_{1xK}(W,X)[co,h,w] = sum_j conv_{1x1}(W[:,:,0,j], X)[co,h,w+j]
    #     and the shift is an ADDRESS offset (ne16_task_t.infeat_addr), not a Slice node. The
    #     K partial sums accumulate inside NE16 via CONFIG0[14] streamin, which forces
    #     32-bit output (gvsoc fsm.cpp:50) -- so the node must be a plain `Conv` emitting
    #     int32, with its RequantShift left on the cluster. Claiming the ORIGINAL 1xK node here
    #     (rather than a pre-decomposed graph) is deliberate: engine coloring runs BEFORE the
    #     NE16 passes, so intent has to be expressed in canExecute or the rewrite never sees
    #     the node. Off by default so PR #183's own behaviour is untouched. -- QW
    def is1xKConv(self, node) -> bool:  # -- QW
        if node.op not in ["Conv", "RequantizedConv"]:
            return False
        ks = node.attrs.get("kernel_shape", None)
        if ks is None or len(ks) != 2:
            return False
        kh, kw = int(ks[0]), int(ks[1])
        # QW (exp16c_SDK_port phase 2): unlike the 3x3/PW paths, the 1xK path does NOT require a
        #     gs.Constant weight. NE16Prepare1xKPass re-encodes it either way -- on the host for a
        #     constant, or by inserting NE16WeightEncode for a runtime tensor (the QZO case, where
        #     the weight is an RQSPerturbRademacher output). Engine coloring runs BEFORE that pass,
        #     so if canExecute insisted on a constant here the rewrite would never see the node.
        #     Gated behind enable1xK, so PR #183's behaviour is untouched. -- QW
        # QW (exp16c_SDK_port phase 3): admissibility lives in ONE place, shared with the rewrite
        #     pass. Coloring must not claim a node the pass cannot rewrite -- kernel shape, group,
        #     dilations, strides, PADDING and input signedness are all decided there. Local import
        #     to avoid an import cycle (the pass module imports the NE16 topology passes). -- QW
        from Deeploy.Targets.NE16.TopologyOptimizationPasses.Prepare1xKPass import ne16_1xkAdmissible
        del kh, kw
        return ne16_1xkAdmissible(node)

    def canExecute(self, node: gs.Node) -> bool:
        if node.op == "NE16DualConv2in":  # -- QW exp7: dispatches NE16 jobs itself
            return True
        if self.enable1xK and self.is1xKConv(node):  # -- QW
            return True
        if self.enable3x3:
            return self.isPWConv(node) or self.isDWConv(node) or self.isDenseConv(node)
        else:
            return self.isPWConv(node)
