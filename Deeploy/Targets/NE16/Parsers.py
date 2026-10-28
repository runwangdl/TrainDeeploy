# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.Parsers import Conv2DParser, ConvParser, RQSParserInterface


class NE16Conv2DBaseParser(Conv2DParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not all([
                # No dilation support
                self.operatorRepresentation['dilations'] == [1, 1],
                # Channels have to be last
                'channels_first' in self.operatorRepresentation and not self.operatorRepresentation['channels_first'],
                # Expect "weight_offset" attribute in the node
                "weight_offset" in node.attrs,
        ]):
            return False

        self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
        self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
        self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
        self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])
        self.operatorRepresentation['weight_offset'] = int(node.attrs["weight_offset"])

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        # LMACAN: Cannot reuse the Conv2DParser's parserNodeCtxt because it requires the weight shape
        #         to be of length 4 whereas ne16 does a specific weight encoding so the shape
        #         ends up being equal to 3.
        newCtxt, ret = ConvParser.parseNodeCtxt(self, ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        # LMACAN: c/p of Conv2DParser's parserNodeCtxt but with a different weight shape check
        #         and enforcing that the channels_first is false
        data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
        data_out = newCtxt.lookup(self.operatorRepresentation['data_out'])
        weight = newCtxt.lookup(self.operatorRepresentation['weight'])

        if not all([
                channels_first == False,
                len(data_in.shape) == 4,
                # LMACAN: weight shape should be equal to 3 because we have to do the ne16's
                #         special weight encoding. Dense 3x3 uses rank 4,
                #         PW/DW use rank 3.
                len(weight.shape) in (3, 4),
        ]):
            return newCtxt, False

        self.operatorRepresentation['batch'] = data_in.shape[0]
        self.operatorRepresentation['dim_im_in_x'] = data_in.shape[1]
        self.operatorRepresentation['dim_im_in_y'] = data_in.shape[2]
        self.operatorRepresentation['ch_im_in'] = data_in.shape[3]
        self.operatorRepresentation['dim_im_out_x'] = data_out.shape[1]
        self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]
        self.operatorRepresentation['ch_im_out'] = data_out.shape[3]

        # No requantization
        self.operatorRepresentation['mul'] = 'NULL'
        self.operatorRepresentation['add'] = 'NULL'
        self.operatorRepresentation['shift'] = 'NULL'

        return newCtxt, True


class NE16DWConv2DParser(NE16Conv2DBaseParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        # After NE16 weight encoding for DW, the encoded weight shape no longer
        # carries cout==group (all channels are packed into the cinMinor
        # dimension). Trust the ONNX `group` attribute alone: for DW,
        # group > 1 AND group == channel_out AND kernel_shape == [3,3].
        if not all([
                self.operatorRepresentation['kernel_shape'] == [3, 3],
                self.operatorRepresentation['group'] > 1,
        ]):
            return False

        return True


class NE16RQSDWConv2DParser(NE16DWConv2DParser, RQSParserInterface):

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            RQSParserInterface.parseNode(self, node),
            NE16DWConv2DParser.parseNode(self, node),
        ])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        inputs = ['data_in', 'weight', 'mul', 'add']
        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

        return newCtxt, True


class NE16PWConv2DParser(NE16Conv2DBaseParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not all([
                self.operatorRepresentation['kernel_shape'] == [1, 1],
                self.operatorRepresentation['group'] == 1,
        ]):
            return False

        return True


class NE16RQSPWConv2DParser(NE16PWConv2DParser, RQSParserInterface):

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            RQSParserInterface.parseNode(self, node),
            NE16PWConv2DParser.parseNode(self, node),
        ])
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        inputs = ['data_in', 'weight', 'mul', 'add']
        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

        return newCtxt, True


class NE16DenseConv2DParser(NE16Conv2DBaseParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not all([
                self.operatorRepresentation['kernel_shape'] == [3, 3],
                self.operatorRepresentation['group'] == 1,
        ]):
            return False

        return True


class NE16RQSDenseConv2DParser(NE16DenseConv2DParser, RQSParserInterface):

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            RQSParserInterface.parseNode(self, node),
            NE16DenseConv2DParser.parseNode(self, node),
        ])
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        inputs = ['data_in', 'weight', 'mul', 'add']
        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

        return newCtxt, True


class NE161xKConv2DParser(NE16Conv2DBaseParser):
    """QW (exp16a / STEP 2b): a 1xK / Kx1 dense conv executed as K pointwise NE16 dispatches.

    Differs from NE16PWConv2DParser only in the accepted kernel_shape and in recording the tap
    count. The weight is PRE-ENCODED per tap and stacked (taps, cout, cinMajor, encBytes), so
    the base parser's rank-3-or-4 check already passes. -- QW
    """

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False
        ks = self.operatorRepresentation['kernel_shape']
        if not (len(ks) == 2 and self.operatorRepresentation['group'] == 1 and
                ((ks[0] == 1 and ks[1] > 1) or (ks[1] == 1 and ks[0] > 1))):
            return False
        if 'ne16_taps' not in node.attrs:
            return False
        self.operatorRepresentation['ne16_taps'] = int(node.attrs['ne16_taps'])
        # halo: tap j reads input offset j, so the widest window is taps-1 past the output
        self.operatorRepresentation['ne16_halo'] = int(node.attrs['ne16_taps']) - 1  # -- QW
        return True


class NE163x3ChunkConv2DParser(NE16Conv2DBaseParser):
    """QW (exp16b): a 1xK dense conv executed as ceil(K/3) DENSE 3x3 NE16 dispatches.

    Same admission as NE161xKConv2DParser but keyed on `ne16_chunks`, and the weight is
    pre-encoded per CHUNK as 3x3 (rank 4: chunks*cout, cinMajor, bits, HW*cinMinorBytes). -- QW
    """

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False
        ks = self.operatorRepresentation['kernel_shape']
        if not (len(ks) == 2 and self.operatorRepresentation['group'] == 1 and
                ((ks[0] == 1 and ks[1] > 1) or (ks[1] == 1 and ks[0] > 1))):
            return False
        if 'ne16_chunks' not in node.attrs:
            return False
        self.operatorRepresentation['ne16_chunks'] = int(node.attrs['ne16_chunks'])
        self.operatorRepresentation['ne16_taps'] = int(node.attrs.get('ne16_taps', 0))
        # halo: chunk c starts at input offset 3c and a 3x3 kernel reads 2 past its output
        self.operatorRepresentation['ne16_halo'] = 3 * (self.operatorRepresentation['ne16_chunks'] - 1) + 2
        return True
