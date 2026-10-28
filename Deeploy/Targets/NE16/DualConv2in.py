# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp7, step 7): `NE16DualConv2in` -- both ZO passes of a 1xK / Kx1 quantized conv on NE16 in ONE
node, with the GAP9 SDK's pipeline (word-wise HWC column, double-buffered against NE16's 2-deep job queue).

    NE16DualConv2in(x+ [1,H,W,Ci] i8 HWC, x- [1,H,W,Ci] i8 HWC, wenc2 [2Co,cinMajor,16] u8, mul [Co] i32, add2 [2Co] i32)
        -> y [1,Ho,Wo,2Co] i8 HWC   (A+ in channels [0,Co), A- in [Co,2Co))

wenc2 / add2 come from exp6's prep nodes with ne16_sign_sel = 0 (stacked, one Rademacher draw) and, for wenc2,
ne16_hwc = 1 (rows in (ky, kx, ci) order = the column order). Attributes: kernel_shape, pads, strides, shift (log2 of
the requant div), weight_offset (-128), ne16_pipeline (1 = double buffered, 0 = sequential, for measurement).
Claimed by the NE16 engine, runs on the cluster master core (GAP9ClusterTransformer); the kernel forks the workers to
build columns. Kernel: TargetLibraries/GAP9/src/NE16DualConv2in.c. -- QW
"""

from typing import Dict, List, Tuple, Union

import onnx_graphsurgeon as gs
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import NetworkContext, NodeBinding, NodeParser, NodeTemplate, ONNXLayer, \
    OperatorRepresentation
from Deeploy.Targets.GAP9.Bindings import GAP9ClusterTransformer
from Deeploy.Targets.Generic.TypeCheckers import SignPropTypeChecker
from Deeploy.Targets.NE16.Templates.ConvTemplate import NE162DPWConvTemplate, getNormQuantConf0


_COL_BUDGET = 11264  # bytes per column buffer (two are allocated); 12288 pushed block 1 to two L1 tiles (283 k vs 275 k)


class NE16DualConv2inParser(NodeParser):
    _n_inputs = 5  # QW exp8: 4 for the stacked form

    def parseNode(self, node: gs.Node) -> bool:
        if int(node.attrs.get('stacked_input', 0)) != (self._n_inputs == 4):
            return False
        if not all([
                node.op == "NE16DualConv2in",
                len(node.inputs) == self._n_inputs,
                len(node.outputs) == 1,
                all(a in node.attrs for a in ('kernel_shape', 'pads', 'strides', 'shift')),
        ]):
            return False
        r = self.operatorRepresentation
        kh, kw = (int(v) for v in node.attrs['kernel_shape'])
        pads = [int(v) for v in node.attrs['pads']]
        strides = [int(v) for v in node.attrs['strides']]
        if len(pads) != 4 or len(strides) != 2:
            return False
        r['kernel_shape'], r['pads'], r['strides'] = [kh, kw], pads, strides
        r['dim_kernel_x'], r['dim_kernel_y'] = kh, kw  # Deeploy convention: _x = H, _y = W
        r['padding_y_top'], r['padding_x_left'], r['padding_y_bottom'], r['padding_x_right'] = pads
        r['stride_x'], r['stride_y'] = strides
        r['log2D'] = int(node.attrs['shift'])
        r['weight_offset'] = int(node.attrs.get('weight_offset', -128))
        r['pipelined'] = int(node.attrs.get('ne16_pipeline', 1))
        r['input_offset'] = int(node.attrs.get('ne16_input_offset', 128))  # RW 2026-10-15: 0 = input known >= 0
        import os as _os  # RW 2026-10-15: board A/B -- DEEPLOY_NE16_PIPELINE=0 forces the sequential form
        if _os.environ.get('DEEPLOY_NE16_PIPELINE') is not None:
            r['pipelined'] = int(_os.environ['DEEPLOY_NE16_PIPELINE'])
        return True

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        names = ['data_in', 'data_in_m', 'weight', 'mul', 'add']
        bufs = [ctxt.lookup(i.name) for i in node.inputs]
        out = ctxt.lookup(node.outputs[0].name)
        xs, xms, ws = (tuple(int(s) for s in b.shape) for b in bufs[:3])
        ys = tuple(int(s) for s in out.shape)
        if len(xs) != 4 or xs != xms or len(ys) != 4 or xs[0] != 1 or ys[0] != 1 or len(ws) != 3:
            return ctxt, False
        _, H, W, Ci = xs
        _, Ho, Wo, C2 = ys
        r = self.operatorRepresentation
        K = Ci * r['dim_kernel_x'] * r['dim_kernel_y']
        if C2 % 2 or ws[0] != C2 or ws[1] != (K + 15) // 16 or ws[2] != 16 or Ci % 4:
            return ctxt, False
        for k, b in zip(names, bufs):
            r[k] = b.name
        r['data_out'] = out.name
        r['batch'] = 1
        r['dim_im_in_x'], r['dim_im_in_y'], r['ch_im_in'] = H, W, Ci
        r['dim_im_out_x'], r['dim_im_out_y'], r['ch_im_out'] = Ho, Wo, C2
        r['ch_out_pass'], r['col_len'], r['wo_full'] = C2 // 2, K, Wo
        # job = 3 x job_w output pixels; two column buffers of 3 * job_w * K bytes within _COL_BUDGET each
        # r['job_w'] = max(1, min(_COL_BUDGET // K, Ho * Wo) // 3)
        # RW 2026-10-15: a multiple of 3 -- with job_w = 29 the GAP9 silicon NE16 put +Δacc into column 2 of the first
        # 3x3 output subtile of every job after the first (block 1: 137 of 152 wrong outputs, all +1 LSB), exact in GVSoC.
        r['job_w'] = max(1, min(_COL_BUDGET // K, Ho * Wo) // 3)
        if r['job_w'] >= 3:
            r['job_w'] -= r['job_w'] % 3
        r['nodeName'] = node.name
        # NE16 job configuration, exactly as NE162DPWConvTemplate + getNormQuantConf0 for this node
        # (8-bit weights, 8-bit signed output -> no ReLU, unsigned column input, 32-bit scale, bias, no per-channel shift)
        conf0 = NE162DPWConvTemplate.getConf0(output_bits = 8, weight_bits = 8, input_signed = False, use_wmem = False)
        conf0 |= getNormQuantConf0(False, r['log2D'], 32, True, False)
        r['conf0'] = conf0
        return ctxt, True


class NE16DualConv2inStackedParser(NE16DualConv2inParser):
    """QW (exp8): stacked form -- FOUR inputs [x, wenc2, mul, add2], attr stacked_input = 1, x = [1, H, W, 2*Ci] HWC with
    x+ then x- per pixel. ch_im_in is the per-pass Ci (the kernel's in_stride is 2*Ci)."""

    _n_inputs = 4

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        names = ['data_in', 'weight', 'mul', 'add']
        bufs = [ctxt.lookup(i.name) for i in node.inputs]
        out = ctxt.lookup(node.outputs[0].name)
        xs, ws = (tuple(int(s) for s in b.shape) for b in bufs[:2])
        ys = tuple(int(s) for s in out.shape)
        if len(xs) != 4 or len(ys) != 4 or xs[0] != 1 or ys[0] != 1 or len(ws) != 3 or xs[3] % 2:
            return ctxt, False
        _, H, W, C2in = xs
        _, Ho, Wo, C2 = ys
        Ci = C2in // 2
        r = self.operatorRepresentation
        K = Ci * r['dim_kernel_x'] * r['dim_kernel_y']
        if C2 % 2 or ws[0] != C2 or ws[1] != (K + 15) // 16 or ws[2] != 16 or Ci % 4:
            return ctxt, False
        for k, b in zip(names, bufs):
            r[k] = b.name
        r['data_out'] = out.name
        r['batch'] = 1
        r['dim_im_in_x'], r['dim_im_in_y'], r['ch_im_in'] = H, W, Ci
        r['dim_im_out_x'], r['dim_im_out_y'], r['ch_im_out'] = Ho, Wo, C2
        r['ch_out_pass'], r['col_len'], r['wo_full'] = C2 // 2, K, Wo
        # r['job_w'] = max(1, min(_COL_BUDGET // K, Ho * Wo) // 3)
        # RW 2026-10-15: a multiple of 3 -- with job_w = 29 the GAP9 silicon NE16 put +Δacc into column 2 of the first
        # 3x3 output subtile of every job after the first (block 1: 137 of 152 wrong outputs, all +1 LSB), exact in GVSoC.
        r['job_w'] = max(1, min(_COL_BUDGET // K, Ho * Wo) // 3)
        if r['job_w'] >= 3:
            r['job_w'] -= r['job_w'] % 3
        r['nodeName'] = node.name
        conf0 = NE162DPWConvTemplate.getConf0(output_bits = 8, weight_bits = 8, input_signed = False, use_wmem = False)
        conf0 |= getNormQuantConf0(False, r['log2D'], 32, True, False)
        r['conf0'] = conf0
        return ctxt, True


class _NE16DualConv2inTemplate(NodeTemplate):
    """Hoists the two column buffers (2 x 3 x job_w x K bytes)."""

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        size = 2 * 3 * operatorRepresentation['job_w'] * operatorRepresentation['col_len']
        return [(operatorRepresentation['nodeName'] + "_colbuf", size)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        name, size = _NE16DualConv2inTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(name, size)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size
        return ctxt, operatorRepresentation, [name]


referenceTemplate = _NE16DualConv2inTemplate("""
// NE16DualConv2in (Name: ${nodeName}, Op: ${nodeOp}) -- both passes on NE16, word-wise column, pipeline=${pipelined} -- QW exp7
PULP_NE16DualConv2in((const int8_t *) ${data_in}, (const int8_t *) ${data_in_m}, (const uint8_t *) ${weight},
    (const int32_t *) ${mul}, (const int32_t *) ${add}, (int8_t *) ${data_out}, (uint8_t *) ${ctxtBuffer}, NUM_CORES,
    ${conf0}u, ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in}, ${dim_im_out_x}, ${dim_im_out_y}, ${ch_out_pass},
    ${dim_kernel_x}, ${dim_kernel_y}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
    ${stride_x}, ${stride_y}, ${weight_offset}, ${pipelined}, ${job_w}, ${ch_im_in}, ${input_offset});
""")

# QW (exp8): the stacked-input form (dual-pass graph): ONE input [1, H, W, 2*Ci] = [x+ | x-] per pixel.
stackedTemplate = _NE16DualConv2inTemplate("""
// NE16DualConv2in, stacked [x+ | x-] input (Name: ${nodeName}, Op: ${nodeOp}), pipeline=${pipelined} -- QW exp8
PULP_NE16DualConv2in((const int8_t *) ${data_in}, (const int8_t *) ${data_in} + ${ch_im_in}, (const uint8_t *) ${weight},
    (const int32_t *) ${mul}, (const int32_t *) ${add}, (int8_t *) ${data_out}, (uint8_t *) ${ctxtBuffer}, NUM_CORES,
    ${conf0}u, ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in}, ${dim_im_out_x}, ${dim_im_out_y}, ${ch_out_pass},
    ${dim_kernel_x}, ${dim_kernel_y}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
    ${stride_x}, ${stride_y}, ${weight_offset}, ${pipelined}, ${job_w}, 2 * ${ch_im_in}, ${input_offset});
""")


class NE16DualConv2inChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs, operatorRepresentation):
        return [256]

    def _inferSignedness(self, inputs, operatorRepresentation):
        return [True]


NE16DualConv2inStackedBindings = [  # -- QW exp8
    NodeBinding(
        NE16DualConv2inChecker([PointerClass(int8_t), PointerClass(uint8_t), PointerClass(int32_t), PointerClass(int32_t)],
                               [PointerClass(int8_t)]), stackedTemplate, GAP9ClusterTransformer)
]

NE16DualConv2inBindings = [
    NodeBinding(
        NE16DualConv2inChecker([PointerClass(int8_t), PointerClass(int8_t), PointerClass(uint8_t),
                                PointerClass(int32_t), PointerClass(int32_t)], [PointerClass(int8_t)]),
        referenceTemplate, GAP9ClusterTransformer)
]


class NE16DualConv2inLayer(ONNXLayer):

    def __init__(self, maps):
        super().__init__(maps)

    def computeOps(self):
        r = self.mapper.parser.operatorRepresentation
        return 2 * r['dim_im_out_x'] * r['dim_im_out_y'] * r['ch_out_pass'] * r['col_len']
