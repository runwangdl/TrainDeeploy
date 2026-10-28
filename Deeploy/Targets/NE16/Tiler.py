# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


from Deeploy.Targets.NE16.Bindings import NE161xKConv2DBindings, NE163x3ChunkConv2DBindings  # -- QW
from Deeploy.Targets.NE16.Bindings import NE16DenseConv2DBindings, NE16DWConv2DBindings, NE16PWConv2DBindings, \
    NE16RQSDenseConv2DBindings, NE16RQSDWConv2DBindings, NE16RQSPWConv2DBindings
from Deeploy.Targets.NE16.TileConstraints.NE16DenseConstraint import NE16DenseConv2DTileConstraint, \
    NE16RQSDenseConv2DTileConstraint
from Deeploy.Targets.NE16.TileConstraints.NE16DepthwiseConstraint import NE16DWConv2DTileConstraint, \
    NE16RQSDWConv2DTileConstraint
from Deeploy.Targets.NE16.TileConstraints.NE16PointwiseConstraint import NE16PWConv2DTileConstraint, \
    NE16RQSPWConv2DTileConstraint
from Deeploy.Targets.NE16.TileConstraints.NE161xKConstraint import NE161xKConv2DTileConstraint  # -- QW
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

NE16RQSPWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16RQSPWConv2DBindings,
                                                             tileConstraint = NE16RQSPWConv2DTileConstraint())
NE16PWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16PWConv2DBindings,
                                                          tileConstraint = NE16PWConv2DTileConstraint())

NE16RQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16RQSDWConv2DBindings,
                                                             tileConstraint = NE16RQSDWConv2DTileConstraint())
NE16DWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16DWConv2DBindings,
                                                          tileConstraint = NE16DWConv2DTileConstraint())

NE16RQSDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16RQSDenseConv2DBindings,
                                                                tileConstraint = NE16RQSDenseConv2DTileConstraint())
NE16DenseConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16DenseConv2DBindings,
                                                             tileConstraint = NE16DenseConv2DTileConstraint())

NE161xKConv2DTilingReadyBindings = TilingReadyNodeBindings(  # -- QW
    nodeBindings = NE161xKConv2DBindings, tileConstraint = NE161xKConv2DTileConstraint())

NE163x3ChunkConv2DTilingReadyBindings = TilingReadyNodeBindings(  # -- QW (exp16b)
    nodeBindings = NE163x3ChunkConv2DBindings, tileConstraint = NE161xKConv2DTileConstraint())


# -- QW (exp16c / blocker 1b): device-side weight encoder. Untiled -- see the constraint's docstring.
from Deeploy.Targets.NE16.TileConstraints.NE16WeightEncodeConstraint import NE16WeightEncodeTileConstraint  # -- QW
from Deeploy.Targets.NE16.WeightEncode import NE16WeightEncodeBindings as _NE16WEB  # -- QW
NE16WeightEncodeTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _NE16WEB,
                                                              tileConstraint = NE16WeightEncodeTileConstraint())  # -- QW

# -- QW (exp16c phase 4 / BLOCKER 3): signed-activation bias correction. Untiled -- a per-cout
#    reduction over the whole weight, and the tensors are tiny.
from Deeploy.Targets.NE16.TileConstraints.NE16WeightEncodeConstraint import NE16SignedInputBiasTileConstraint  # -- QW
from Deeploy.Targets.NE16.WeightEncode import NE16SignedInputBiasBindings as _NE16SIBB  # -- QW
NE16SignedInputBiasTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _NE16SIBB,
                                                                 tileConstraint = NE16SignedInputBiasTileConstraint())  # -- QW

# -- QW (dual_pass_fusion exp6): column buffer for the SDK im2col route. Untiled (see the constraint).
from Deeploy.Targets.NE16.TileConstraints.NE16Im2ColConstraint import NE16Im2ColU8TileConstraint  # -- QW
from Deeploy.Targets.NE16.Im2Col import NE16Im2ColU8Bindings as _NE16I2CB  # -- QW
NE16Im2ColU8TilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _NE16I2CB,
                                                          tileConstraint = NE16Im2ColU8TileConstraint())  # -- QW

# -- QW (dual_pass_fusion exp6, task 2): fused-pass preparation nodes, untiled (tiny per-step tensors).
from Deeploy.Targets.NE16.TileConstraints.NE16Im2ColConstraint import NE16DualBiasTileConstraint, \
    NE16DualPerturbEncodeTileConstraint  # -- QW
from Deeploy.Targets.NE16.DualPrep import NE16DualBiasBindings as _NE16DBB, \
    NE16DualPerturbEncodeBindings as _NE16DPEB  # -- QW
NE16DualPerturbEncodeTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = _NE16DPEB, tileConstraint = NE16DualPerturbEncodeTileConstraint())  # -- QW
NE16DualBiasTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = _NE16DBB, tileConstraint = NE16DualBiasTileConstraint())  # -- QW
# -- QW (dual_pass_fusion exp7 step 7): both passes on NE16 in one node, SDK-style pipeline. -- QW
from Deeploy.Targets.NE16.DualConv2in import NE16DualConv2inBindings as _NE16DC2B  # -- QW
from Deeploy.Targets.NE16.TileConstraints.NE16DualConv2inConstraint import NE16DualConv2inTileConstraint  # -- QW
NE16DualConv2inTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = _NE16DC2B, tileConstraint = NE16DualConv2inTileConstraint())  # -- QW
from Deeploy.Targets.NE16.DualConv2in import NE16DualConv2inStackedBindings as _NE16DC2SB  # -- QW exp8
NE16DualConv2inStackedTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = _NE16DC2SB, tileConstraint = NE16DualConv2inTileConstraint())  # -- QW exp8
