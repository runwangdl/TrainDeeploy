# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeParser
from Deeploy.Targets.Generic.Parsers import Conv2DGradWParser, Conv2DGradXParser, Conv2DGradXWBParser, \
    Conv2DGradXWParser, Conv2DParser, GEMMParser, ReduceMeanParser, RQSConv1DParser, RQSConv2DParser, \
    RQSParserInterface


class PULPConv2DParser(RQSConv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['group'] == 1,
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                # QW: mirror shipped ETH/Deeploy — the square-padding guard is intentionally disabled there so
                # non-square padding (e.g. SpeechNet's [0,K/2,0,K/2]) maps; the int8 pulp_nn_conv kernel takes
                # all four pads separately. Our vendored copy had re-enabled it (a regression). -- QW
                # self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                len(node.inputs) == 4,
                'shift' in node.attrs,
            ])

            self.operatorRepresentation['dim_kernel_x'] = int(self.operatorRepresentation['kernel_shape'][0])
            self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][1])
            self.operatorRepresentation['dilation_x'] = int(self.operatorRepresentation['dilations'][0])
            self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][1])
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])
            self.operatorRepresentation['stride_x'] = int(self.operatorRepresentation['strides'][0])
            self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][1])

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class PULPFPConv2DParser(Conv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Current PULP kernel only supports grouping of 1
                self.operatorRepresentation['group'] == 1,

                # Make sure padding is symmetric (left==right, top==bottom)
                # but top/bottom can differ from left/right
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],  # top == bottom
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],  # left == right

                # Check number of inputs
                # 2 inputs if no bias, 3 if layer has bias
                len(node.inputs) in [2, 3],
            ])

            # Extract additional attributes
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            # Set inputs names
            inputs = ['data_in', 'weight']

            # Handle bias, if present
            if len(node.inputs) == 2:
                self.operatorRepresentation["has_bias"] = "false"
                self.operatorRepresentation["bias"] = "NULL"
            else:
                inputs.append("bias")
                self.operatorRepresentation["has_bias"] = "true"

            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class PULPFPDWConv2DParser(Conv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        # Parse root conv 2D information
        wellFormed = super().parseNode(node)

        if wellFormed:
            # Check if the node is a depthwise convolution
            ret = all([
                # Make sure padding is symmetric (left==right, top==bottom)
                # but top/bottom can differ from left/right
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],  # top == bottom
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],  # left == right

                # Check number of inputs
                # 2 inputs if no bias, 3 if layer has bias
                len(node.inputs) in [2, 3],
            ])

            # Extract additional attributes
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        # Parse node context for 2D conv
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            # Define input names
            inputs = ['data_in', 'weight']

            # Handle bias, if present
            if len(node.inputs) == 2:
                self.operatorRepresentation["has_bias"] = "false"
                self.operatorRepresentation["bias"] = "NULL"
            else:
                inputs.append("bias")
                self.operatorRepresentation["has_bias"] = "true"

            # Map input nodes to operator representation
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            # Check if DW
            if self.operatorRepresentation['group'] == self.operatorRepresentation['ch_im_in']:
                return newCtxt, True

        return ctxt, False


# RW: channels-first (NCHW) forward conv parsers. Used by the GAP9 CHW conv
# mappers, which sit ahead of the HWC mappers in the 'Conv' mapping list. These
# parsers only accept a node tagged with attr "keep_channels_first" (set by
# PULPConvKeepCHWPass when the deployer runs with conv_channels_first=True);
# every other conv node makes parseNode return False and falls through to the
# HWC mapper, so non-CHW networks are completely unaffected.
#
# The transpose-skipping pass keeps the conv activations physically NCHW, so the
# dim indexing must be channels-first regardless of the deployer's
# default_channels_first (which is NHWC/False for GAP9). We therefore force
# channels_first=True into the base parseNodeCtxt, matching the kept NCHW tensor
# shapes and the OIHW weight layout consumed by the *_CHW kernels.
class PULPFPConv2DCHWParser(PULPFPConv2DParser):

    def parseNode(self, node: gs.Node) -> (bool):
        if not bool(node.attrs.get("keep_channels_first", False)):
            return False
        return super().parseNode(node)

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        return super().parseNodeCtxt(ctxt, node, channels_first = True)


class PULPFPDWConv2DCHWParser(PULPFPDWConv2DParser):

    def parseNode(self, node: gs.Node) -> (bool):
        if not bool(node.attrs.get("keep_channels_first", False)):
            return False
        return super().parseNode(node)

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        return super().parseNodeCtxt(ctxt, node, channels_first = True)


class PULPDWConv1DParser(RQSConv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                #self.operatorRepresentation['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
                len(node.inputs) == 4,
            ])

            if ret:

                self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][0])
                self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][0])
                self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
                self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][1])
                self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][0])

                if 'n_levels' in node.attrs:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'].values)
                else:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels_out'].values)

                self.operatorRepresentation['signed'] = int(node.attrs['signed'].values)
                self.operatorRepresentation['log2D'] = int(math.log2(node.attrs['div'].values))
            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:

            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            if not self.operatorRepresentation['group'] == newCtxt.lookup(
                    self.operatorRepresentation['weight']).shape[0]:
                return ctxt, False

            # if not newCtxt.is_global(self.operatorRepresentation['weight']):
            #     return ctxt, False

            # SCHEREMO: Transpose weights to be num filters last
            # newCtxt.globalObjects[self.operatorRepresentation['weight']].values = np.transpose(weight.values, list(range(len(weight.shape)))[1:] + [0])

            return newCtxt, True

        return ctxt, False


class PULPDWConv2DParser(RQSConv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                node.op == 'RequantizedConv',
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                #self.operatorRepresentation['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
                len(node.inputs) == 4,
                'shift' in node.attrs,
                any(['n_levels' in node.attrs, 'n_levels_out' in node.attrs]),
                'signed' in node.attrs
            ])

            if ret:
                self.operatorRepresentation['dim_kernel_x'] = int(self.operatorRepresentation['kernel_shape'][0])
                self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][1])
                self.operatorRepresentation['dilation_x'] = int(self.operatorRepresentation['dilations'][0])
                self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][1])
                self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
                self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
                self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
                self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])
                self.operatorRepresentation['stride_x'] = int(self.operatorRepresentation['strides'][0])
                self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][1])

                if 'n_levels' in node.attrs:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'].values)
                else:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels_out'].values)
                self.operatorRepresentation['signed'] = int(node.attrs['signed'].values)
                self.operatorRepresentation['log2D'] = int(math.log2(node.attrs['div'].values))

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:

            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            if not self.operatorRepresentation['group'] == newCtxt.lookup(
                    self.operatorRepresentation['weight']).shape[0]:
                return ctxt, False

            data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
            data_out = newCtxt.lookup(self.operatorRepresentation['data_out'])
            _ = newCtxt.lookup(self.operatorRepresentation['weight'])

            # if not newCtxt.is_global(self.operatorRepresentation['weight']):
            #     return ctxt, False

            # SCHEREMO: Transpose weights to be num filters last
            # newCtxt.globalObjects[self.operatorRepresentation['weight']].values = np.transpose(weight.values, list(range(len(weight.shape)))[1:] + [0])

            if channels_first:
                self.operatorRepresentation['ch_im_in'] = data_in.shape[1]
                self.operatorRepresentation['dim_im_in_x'] = data_in.shape[2]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[3]
                self.operatorRepresentation['ch_im_out'] = data_out.shape[1]
                self.operatorRepresentation['dim_im_out_x'] = data_out.shape[2]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[3]
            else:
                self.operatorRepresentation['ch_im_in'] = data_in.shape[1]
                self.operatorRepresentation['dim_im_in_x'] = data_in.shape[2]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[3]
                self.operatorRepresentation['ch_im_out'] = data_out.shape[3]
                self.operatorRepresentation['dim_im_out_x'] = data_out.shape[1]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]

            return newCtxt, True

        return ctxt, False


class PULPConv1DParser(RQSConv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['group'] == 1,
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                #self.operatorRepresentation['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
                len(node.inputs) == 4,
            ])

            self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][0])
            self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][0])
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][0])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class PULPGEMMParser(GEMMParser, RQSParserInterface):

    def __init__(self):
        super().__init__(noBiasHoisting = True)

    def parseNode(self, node: gs.Node) -> (bool):

        ret_rqs = RQSParserInterface.parseNode(self, node)
        ret_matmul = GEMMParser.parseNode(self, node)

        ret = all([
            ret_rqs == True,
            ret_matmul == True,
            'shift' in node.attrs,
            len(node.inputs) == 4,
        ])

        if ret:
            self.operatorRepresentation['shift'] = int(node.attrs['shift'].values)

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['A', 'B', 'C', 'mul']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            return newCtxt, True

        else:
            return ctxt, False


class PULPMatrixVecParser(PULPGEMMParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        if not (self.operatorRepresentation['M'] == 1 and self.operatorRepresentation['batch'] >= 8):
            return ctxt, False

        return newCtxt, True


class PULPTallGEMMParser(PULPGEMMParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        ret = all([
            self.operatorRepresentation['batch'] < 8,
            self.operatorRepresentation['M'] >= 8,
            self.operatorRepresentation['M'] % 8 < self.operatorRepresentation['O'] % 8,
        ])

        if not ret:
            return ctxt, False

        return newCtxt, True


class PULPReduceMeanParser(ReduceMeanParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        # Inherit the generic ReduceMean parsing
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            # Add to operator representation the non-reduced dimensions for tiling purposes
            originalInputShape = newCtxt.lookup(self.operatorRepresentation['data_in']).shape
            reducedAxes = self.operatorRepresentation['axes']

            for ax in range(len(originalInputShape)):
                if ax not in reducedAxes:
                    self.operatorRepresentation['dim_in_' + str(ax)] = originalInputShape[ax]

            return newCtxt, True
        else:
            return ctxt, False


class PULPConvGradX2DParser(Conv2DGradXParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])

            return newCtxt, True

        return ctxt, False


class PULPDWConvGradX2DParser(PULPConvGradX2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> bool:

        wellFormed = super().parseNode(node)

        ret = all([
            self.operatorRepresentation['group'] > 1,
        ])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if self.operatorRepresentation['group'] == ctxt.lookup(self.operatorRepresentation['weight']).shape[0]:
            return newCtxt, True

        return ctxt, False


class PULPPWConvGradX2DParser(PULPConvGradX2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> bool:

        wellFormed = super().parseNode(node)

        kernel_shape = self.operatorRepresentation['kernel_shape']
        if kernel_shape != [1, 1]:
            return False

        # The PW ConvGradX kernel (direct dX[ci, hw] = sum_co W[co, ci] *
        # dY[co, hw]) implicitly assumes dX and dY share the same spatial
        # extent — i.e. stride == 1. Stride>1 1x1 convolutions (e.g.
        # ResNet8 downsample shortcuts) need the sparse-write semantics
        # `dX[ci, stride*y, stride*x] = ...` which only the im2col-tiled
        # general kernel implements correctly, so reject them here and let
        # them fall through to ConvGradX2DIm2ColHWTileConstraint.
        strides = self.operatorRepresentation.get('strides', [1, 1])
        if list(strides) != [1, 1]:
            return False

        return wellFormed and True


class PULPConvGradW2DParser(Conv2DGradWParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])

            return newCtxt, True

        return ctxt, False


class PULPDWConvGradW2DParser(PULPConvGradW2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> bool:

        wellFormed = super().parseNode(node)

        ret = all([
            self.operatorRepresentation['group'] > 1,
        ])

        return wellFormed and ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if self.operatorRepresentation['group'] == ctxt.lookup(self.operatorRepresentation['grad_weight']).shape[0]:
            return newCtxt, True

        return ctxt, False


class PULPPWConvGradW2DParser(PULPConvGradW2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> bool:

        wellFormed = super().parseNode(node)

        kernel_shape = self.operatorRepresentation['kernel_shape']

        if kernel_shape != [1, 1]:
            return False

        return wellFormed and True


class PULPConvGradXW2DParser(Conv2DGradXWParser):
    """PULP-specific combined ConvGrad no-bias (3 inputs, 2 outputs)."""

    def __init__(self):
        super().__init__()

    def parseNodeCtxt(self, ctxt, node, channels_first = True):
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if ret:
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])
            return newCtxt, True
        return ctxt, False


class PULPConvGradXWB2DParser(Conv2DGradXWBParser):
    """PULP-specific combined ConvGrad with bias (4 inputs, 3 outputs)."""

    def __init__(self):
        super().__init__()

    def parseNodeCtxt(self, ctxt, node, channels_first = True):
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if ret:
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])
            return newCtxt, True
        return ctxt, False


class PULPDualConv2DParser(RQSConv2DParser):  # -- QW (dual_pass_fusion exp5)
    """`RequantizedDualConv`: the fused dual-pass QZO conv.

    Inputs: data_in, weight (UNPERTURBED int8), mul, add (UNPERTURBED int32), wmul, bmul (the two
    RQSPerturbRademacher multipliers). Output: int8 with 2*ch_out channels (A+ then A-).
    Attributes: the Conv/RequantShift set (`shift`, `div`, `n_levels`, `signed`) plus the two perturb
    nodes' `seed`/`idx`/`div` as seed_w/idx_w/div_w and seed_b/idx_b/div_b.
    """

    _n_inputs = 6  # -- QW exp7: data_in, weight, mul, add, wmul, bmul
    _weight_chw = 0  # -- QW exp8

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        if not super().parseNode(node):
            return False
        r = self.operatorRepresentation
        pads = r['pads']
        ok = all([
            r['group'] == 1,
            pads[0] == pads[2],
            pads[1] == pads[3],
            # len(node.inputs) == 6,  # -- QW exp5 original
            len(node.inputs) == self._n_inputs,  # -- QW exp7: 6 (block 0) or 7 (PULPDualConv2D2InParser)
            # -- QW exp8: a raw-CHW-weight node is never block 0's (6-input stacked-input nodes of blocks 1-4 exist now)
            int(node.attrs.get('weight_chw', 0)) == self._weight_chw,
            'shift' in node.attrs,
            all(a in node.attrs for a in ('seed_w', 'idx_w', 'div_w', 'seed_b', 'idx_b', 'div_b')),
        ])
        if not ok:
            return False
        r['dim_kernel_x'] = int(r['kernel_shape'][0])
        r['dim_kernel_y'] = int(r['kernel_shape'][1])
        r['dilation_x'] = int(r['dilations'][0])
        r['dilation_y'] = int(r['dilations'][1])
        r['padding_y_top'] = int(pads[0])
        r['padding_x_left'] = int(pads[1])
        r['padding_y_bottom'] = int(pads[2])
        r['padding_x_right'] = int(pads[3])
        r['stride_x'] = int(r['strides'][0])
        r['stride_y'] = int(r['strides'][1])
        r['seed_w'] = int(node.attrs['seed_w'])
        r['idx_w'] = int(node.attrs['idx_w'])
        r['log2D_w'] = int(math.log2(int(node.attrs['div_w'])))
        r['seed_b'] = int(node.attrs['seed_b'])
        r['idx_b'] = int(node.attrs['idx_b'])
        r['log2D_b'] = int(math.log2(int(node.attrs['div_b'])))
        return True

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if ret:
            for idx, name in enumerate(['data_in', 'weight', 'mul', 'add', 'wmul', 'bmul']):
                self.operatorRepresentation[name] = ctxt.lookup(node.inputs[idx].name).name
            # per-pass output channels == weight rows; the tensor's ch_im_out is twice that
            self.operatorRepresentation['ch_out_pass'] = newCtxt.lookup(self.operatorRepresentation['weight']).shape[0]
            return newCtxt, True
        return ctxt, False


class DualHalfParser(NodeParser):  # -- QW (dual_pass_fusion exp8)
    """`DualHalf`: [rows, 2*C] -> [rows, C], half 0 or 1 (attr `half`)."""

    def parseNode(self, node: gs.Node) -> bool:
        return node.op == "DualHalf" and len(node.inputs) == 1 and len(node.outputs) == 1 and \
            int(node.attrs.get('half', -1)) in (0, 1)

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        si, so = tuple(int(s) for s in data_in.shape), tuple(int(s) for s in data_out.shape)
        if len(si) != 2 or len(so) != 2 or si[0] != so[0] or si[1] != 2 * so[1]:
            return ctxt, False
        r = self.operatorRepresentation
        r['data_in'], r['data_out'] = data_in.name, data_out.name
        r['rows'], r['cols'], r['half'] = so[0], so[1], int(node.attrs['half'])
        r['size'] = so[0] * so[1]
        return ctxt, True


class PerturbRademacherFixedParser(NodeParser):  # -- QW (dual_pass_fusion exp8)
    """`PerturbRademacherFixed`: fp32 ZO perturbation with a codegen-time sign (attr `mode`: 0 stacked [+;-] -> output
    2*size, 1 +, 2 -). Same seed / idx / eps attributes as PerturbRademacher. Emitted by the dual-pass graph pass."""

    def parseNode(self, node: gs.Node) -> bool:
        return all([
            node.op == "PerturbRademacherFixed",
            len(node.inputs) == 1,
            len(node.outputs) == 1,
            all(a in node.attrs for a in ('seed', 'eps', 'idx', 'mode')),
            int(node.attrs.get('mode', -1)) in (0, 1, 2),
        ])

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        size = int(np.prod(data_in.shape))
        mode = int(node.attrs['mode'])
        r = self.operatorRepresentation
        if int(np.prod(data_out.shape)) != (2 * size if mode == 0 else size):
            return ctxt, False
        r['data_in'], r['data_out'] = data_in.name, data_out.name
        r['seed'], r['eps'], r['mode'] = node.attrs['seed'], node.attrs['eps'], mode
        r['size'] = size
        r['nodeIdx'] = node.attrs['idx']
        return ctxt, True


class PULPDualConv2D2InParser(PULPDualConv2DParser):  # -- QW (dual_pass_fusion exp7, step 3)
    """`RequantizedDualConv` with TWO input activations (SpeechNet blocks 1-4), contract in
    DeeployTest/experiments/dual_pass_fusion/exp7_remaining_blocks/Plan.md §4.1.

    Inputs: data_in (x+), weight (UNPERTURBED int8, RAW CHW [co, ci, kh, kw] -- attr `weight_chw` = 1 makes the
    layout pass leave it untransposed), mul, add (UNPERTURBED int32), wmul, bmul, data_in_m (x-). x- is appended
    LAST so everything that assumes inputs[0] = activation and inputs[1] = weight keeps working; the input count
    alone separates this node from block 0's 6-input one. Output: int8 with 2*ch_out channels (A+ then A-).
    """

    _n_inputs = 7
    _weight_chw = 1

    # QW (dual_pass_fusion exp8): a second form for the dual-pass graph -- attr `stacked_input` = 1, SIX inputs
    #     [x, weight, mul, add, wmul, bmul], x the channel-stacked [x+ | x-] tensor (2*Ci channels; device HWC pixel
    #     = Ci bytes of x+ then Ci of x-). The kernel then reads x+ at x, x- at x + Ci, pixel stride 2*Ci.
    _stacked = 0

    def parseNode(self, node: gs.Node) -> (bool):
        if not int(node.attrs.get('weight_chw', 0)) or int(node.attrs.get('stacked_input', 0)) != self._stacked:
            return False
        return super().parseNode(node)

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if ret:
            r = self.operatorRepresentation
            r['stacked_input'] = self._stacked
            if r['stacked_input']:
                r['in_ch'] = r['ch_im_in'] // 2  # per pass
                r['in_stride'] = r['ch_im_in']
                if newCtxt.lookup(r['weight']).shape[1] != r['in_ch']:
                    return ctxt, False
            else:
                r['data_in_m'] = ctxt.lookup(node.inputs[6].name).name
                r['in_ch'] = r['ch_im_in']
                r['in_stride'] = r['ch_im_in']
            return newCtxt, True
        return ctxt, False


class PULPPerturbConv2DParser(PULPDualConv2DParser):  # -- QW (dual_pass_fusion exp11)
    """`RequantizedDualConv` with attr `single_pass` = 1: ONE pass of the QZO conv with the perturbation in the kernel
    (PULP_PerturbConv4x2_i8_i8_i8_HWC) -- the unfused reference of the fused node without the dual-pass fusion.
    Inputs as the 6-input form [x, weight (raw CHW, `weight_chw` = 1), mul, add, wmul, bmul]; output ch_out channels
    (one pass); the sign is the runtime perturbation_sign, as for RQSPerturbRademacher."""

    _n_inputs = 6
    _weight_chw = 1

    def parseNode(self, node: gs.Node) -> (bool):
        if int(node.attrs.get('single_pass', 0)) != 1 or int(node.attrs.get('stacked_input', 0)) != 0:
            return False
        return super().parseNode(node)

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if ret:
            r = self.operatorRepresentation
            r['in_ch'] = r['ch_im_in']
            r['in_stride'] = r['ch_im_in']
            # test hook: attr force_sign (1 / 0) fixes the pass at codegen; default = the runtime perturbation_sign
            fs = int(node.attrs.get('force_sign', -1))
            r['sign_expr'] = 'perturbation_sign' if fs < 0 else f'{fs}u'
            return newCtxt, True
        return ctxt, False


class PULPDualConv2DStackedParser(PULPDualConv2D2InParser):  # -- QW (dual_pass_fusion exp8)
    """The stacked-input form of the blocks 1-4 node (dual-pass graph): SIX inputs [x, W, mul, add, wmul, bmul], attr
    stacked_input = 1, x the channel-stacked [x+ | x-] tensor. A separate parser / template so no template references a
    tensor the node does not have."""
    _n_inputs = 6
    _stacked = 1
