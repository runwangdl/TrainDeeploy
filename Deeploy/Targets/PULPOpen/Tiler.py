# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import copy

from Deeploy.Targets.Generic.TileConstraints.AddTileConstraint import AddTileConstraint
from Deeploy.Targets.Generic.TileConstraints.ConcatTileConstraint import ConcatTileConstraint
from Deeploy.Targets.Generic.TileConstraints.iHardswishTileConstraint import iHardswishTileConstraint
from Deeploy.Targets.Generic.TileConstraints.iRMSNormTileConstraint import iRMSNormTileConstraint
from Deeploy.Targets.Generic.TileConstraints.MulTileConstraint import MulTileConstraint
from Deeploy.Targets.Generic.TileConstraints.NOPTileConstraint import NOPTileConstraint
from Deeploy.Targets.Generic.TileConstraints.RQSiGELUTileConstraint import RQSiGELUTileConstraint
from Deeploy.Targets.Generic.TileConstraints.RQSiHardswishTileConstraint import RQSiHardswishTileConstraint
from Deeploy.Targets.Generic.TileConstraints.TransposeTileConstraint import TransposeTileConstraint
from Deeploy.Targets.Generic.TileConstraints.UnaryTileConstraint import UnaryRowTileConstraint, UnaryTileConstraint
from Deeploy.Targets.PULPOpen.Bindings import BasicDequantBindings, BasicQuantBindings, PULPAddBindings, \
    PULPAveragePool2DBindings, PULPAveragePoolGrad2DBindings, PULPBatchNormalizationGradBindings, \
    PULPBatchNormInternalBindings, PULPConcatBindings, PULPFloatConv2DBindings, PULPFloatConvGradBBindings, \
    PULPFloatConvGradW2DBindings, PULPFloatConvGradX2DBindings, PULPFloatDWConv2DBindings, \
    PULPFloatDWConvGradW2DBindings, PULPFloatDWConvGradX2DBindings, PULPFloatGELUBinding, PULPFloatGELUGradBinding, \
    PULPFloatGEMMBindings, PULPFloatPWConvGradW2DBindings, PULPFloatPWConvGradX2DBindings, PULPGatherBindings, \
    PULPGlobalAveragePool2DBindings, PULPGlobalAveragePoolGrad2DBindings, PULPiHardswishBindings, \
    PULPInPlaceAccumulatorV2Bindings, PULPiRMSNormBindings, PULPiRQSGELUBindings, PULPLayernormBinding, \
    PULPLayernormGradBinding, PULPMatMulBindings, PULPMaxPool1DBindings, PULPMaxPool2DBindings, \
    PULPMaxPoolArgmaxBindings, PULPMaxPoolGradMaskBindings, \
    PULPMaxPoolGrad2DBindings, PULPMSELossBindings, PULPMSELossGradBindings, PULPMulBindings, PULPReduceMeanBindings, \
    PULPReduceSumBindings, PULPReluBinding, PULPReluGradBinding, PULPReshapeBindings, PULPRQAddBindings, \
    PULPRQSBindings, PULPRQSConv1DBindings, PULPRQSConv2DBindings, PULPRQSDWConv2DBindings, PULPRQSGEMMBindings, \
    PULPRQSiHardswishBindings, PULPRQSMatrixVecBindings, PULPRQSTallGEMMBindings, PULPSGDBindings, PULPSliceBindings, \
    PULPSoftmaxBindings, PULPSoftmaxCrossEntropyLossBindings, PULPSoftmaxCrossEntropyLossGradBindings, \
    PULPSoftmaxGradBindings, PULPTransposeBindings, PULPUniformRQSBindings, \
    PULPPerturbRademacherBindings  # -- QW (FP32 PerturbRademacher ZO op)
from Deeploy.Targets.PULPOpen.TileConstraints.PerturbTileConstraint import PerturbTileConstraint  # -- QW
from Deeploy.Targets.PULPOpen.TileConstraints.AveragePoolTileConstraint import AveragePoolCTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.BatchNormTileConstraint import BatchNormalizationGradTileConstraint, \
    BatchNormInternalTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.BroadcastAddTileConstraint import PULPBroadcastAddTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.ConvGradConstraint import ConvGradBTileConstraint, \
    ConvGradW2DTileConstraint, ConvGradX2DIm2ColHWTileConstraint, DWConvGradW2DTileConstraint, \
    DWConvGradX2DTileConstraint, PWConvGradWTileConstraint, PWConvGradXTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint, RQConv1DTileConstraint, \
    RQConv2DTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.DWConvTileConstraint import DWConv2DTileConstraint, \
    RQDWConv2DTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GatherTileConstraint import GatherTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GeluTileConstraint import GeluGradTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GEMMTileConstraint import FloatGEMMTileConstraint, GEMMTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GlobalAveragePoolTileConstraint import \
    GlobalAveragePoolGradTileConstraint, GlobalAveragePoolTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.InPlaceAccumulatorV2TileConstraint import \
    InPlaceAccumulatorV2TileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.iSoftmaxTileConstraint import SoftmaxGradTileConstraint, \
    iSoftmaxTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.LayernormTileConstraint import LayernormGradTileConstraint, \
    LayernormTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MatMulTileConstraint import MatMulTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MaxPoolGradTileConstraint import MaxPoolGradCTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MaxPoolTileConstraint import MaxPoolCTileConstraint, \
    MaxPoolHCTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MSELossTileConstraint import MSELossGradTileConstraint, \
    MSELossTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.ReduceMeanConstraint import ReduceMeanTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.ReduceSumTileConstraint import ReduceSumTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.RequantShiftTileConstraint import RequantShiftTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SGDTileConstraint import ReluGradTileConstraint, SGDTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SliceConstraint import SliceTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SoftmaxCrossEntropyTileConstraint import \
    SoftmaxCrossEntropyGradTileConstraint, SoftmaxCrossEntropyTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

PULPRQSConv1DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSConv1DBindings,
                                                           tileConstraint = RQConv1DTileConstraint())

PULPRQSConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSConv2DBindings,
                                                           tileConstraint = RQConv2DTileConstraint())

PULPRQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSDWConv2DBindings,
                                                             tileConstraint = RQDWConv2DTileConstraint())

PULPConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatConv2DBindings,
                                                        tileConstraint = Conv2DTileConstraint())

PULPDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatDWConv2DBindings,
                                                          tileConstraint = DWConv2DTileConstraint())

PULPRQSGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSGEMMBindings,
                                                         tileConstraint = GEMMTileConstraint())

PULPFPGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatGEMMBindings,
                                                        tileConstraint = FloatGEMMTileConstraint())

PULPRQSMatrixVecTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSMatrixVecBindings,
                                                              tileConstraint = GEMMTileConstraint())

PULPRQSTallGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSTallGEMMBindings,
                                                             tileConstraint = GEMMTileConstraint())

PULPMatMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMatMulBindings,
                                                        tileConstraint = MatMulTileConstraint())

PULPRQAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQAddBindings,
                                                       tileConstraint = AddTileConstraint())

PULPiHardswishTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPiHardswishBindings,
                                                            tileConstraint = iHardswishTileConstraint())

PULPRQSiHardswishTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSiHardswishBindings,
                                                               tileConstraint = RQSiHardswishTileConstraint())

_BasicFlattenBindings = copy.deepcopy(PULPReshapeBindings)

PULPFlattenTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _BasicFlattenBindings,
                                                         tileConstraint = NOPTileConstraint())

PULPMaxPool1DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMaxPool1DBindings,
                                                           tileConstraint = MaxPoolCTileConstraint())

PULPMaxPool2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMaxPool2DBindings,
                                                           tileConstraint = MaxPoolHCTileConstraint())  # RW 2026-10-13: row tiles, contiguous DMA

# QW: Part-4 MaxPoolArgmax reuses the single-output channel-tiling constraint (like MaxPool). -- QW
PULPMaxPoolArgmaxTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMaxPoolArgmaxBindings,
                                                               tileConstraint = MaxPoolCTileConstraint())

PULPAveragePool2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPAveragePool2DBindings,
                                                               tileConstraint = AveragePoolCTileConstraint())

PULPAveragePoolGrad2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPAveragePoolGrad2DBindings,
                                                                   tileConstraint = AveragePoolCTileConstraint())

PULPMaxPoolGrad2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMaxPoolGrad2DBindings,
                                                               tileConstraint = MaxPoolGradCTileConstraint())

# QW: Part-4 mask-grad reuses the channel-tiling grad constraint (reads x_in actual shape). -- QW
PULPMaxPoolGradMaskTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMaxPoolGradMaskBindings,
                                                                 tileConstraint = MaxPoolGradCTileConstraint())

PULPGlobalAveragePool2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPGlobalAveragePool2DBindings,
                                                                     tileConstraint = GlobalAveragePoolTileConstraint())

PULPGlobalAveragePoolGrad2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = PULPGlobalAveragePoolGrad2DBindings, tileConstraint = GlobalAveragePoolGradTileConstraint())

PULPBatchNormInternalTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPBatchNormInternalBindings,
                                                                   tileConstraint = BatchNormInternalTileConstraint())

PULPBatchNormalizationGradTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = PULPBatchNormalizationGradBindings, tileConstraint = BatchNormalizationGradTileConstraint())

PULPMSELossTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMSELossBindings,
                                                         tileConstraint = MSELossTileConstraint())

PULPMSELossGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMSELossGradBindings,
                                                             tileConstraint = MSELossGradTileConstraint())

PULPRQSTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSBindings,
                                                     tileConstraint = RequantShiftTileConstraint())

PULPUniformRQSTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPUniformRQSBindings,
                                                            tileConstraint = UnaryTileConstraint())

PULPTransposeTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPTransposeBindings,
                                                           tileConstraint = TransposeTileConstraint())

# Dequant is elementwise, single-input/single-output
# (`out[i] = (in[i] - zero_point) * scale`, Targets/Generic/Templates/DequantTemplate.py),
# which is exactly what UnaryTileConstraint expresses. Without this wrapper the raw
# NodeBindings carry no tileConstraint on their template and the tiler aborts with
# `AttributeError: '_DequantTemplate' object has no attribute 'tileConstraint'`.
PULPDequantTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = BasicDequantBindings,
                                                         tileConstraint = UnaryRowTileConstraint())  # RW 2026-10-13

PULPQuantTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = BasicQuantBindings,
                                                       tileConstraint = UnaryTileConstraint())

PULPAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPAddBindings,
                                                     tileConstraint = PULPBroadcastAddTileConstraint())

PULPSoftmaxTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPSoftmaxBindings,
                                                         tileConstraint = iSoftmaxTileConstraint())

PULPConcatTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPConcatBindings,
                                                        tileConstraint = ConcatTileConstraint())

PULPiRMSNormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPiRMSNormBindings,
                                                          tileConstraint = iRMSNormTileConstraint())

PULPiRQSGELUTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPiRQSGELUBindings,
                                                          tileConstraint = RQSiGELUTileConstraint())

PULPMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMulBindings,
                                                     tileConstraint = MulTileConstraint())

PULPReluTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPReluBinding],
                                                      tileConstraint = UnaryTileConstraint())

PULPReluGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPReluGradBinding],
                                                          tileConstraint = ReluGradTileConstraint())

PULPLayernormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPLayernormBinding],
                                                           tileConstraint = LayernormTileConstraint())

PULPLayernormGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPLayernormGradBinding],
                                                               tileConstraint = LayernormGradTileConstraint())

PULPFPGELUTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPFloatGELUBinding],
                                                        tileConstraint = UnaryTileConstraint())

PULPFPGELUGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPFloatGELUGradBinding],
                                                            tileConstraint = GeluGradTileConstraint())

PULPGatherTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPGatherBindings,
                                                        tileConstraint = GatherTileConstraint())

PULPSoftmaxCrossEntropyTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = PULPSoftmaxCrossEntropyLossBindings, tileConstraint = SoftmaxCrossEntropyTileConstraint())

PULPSoftmaxCrossEntropyGradTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = PULPSoftmaxCrossEntropyLossGradBindings, tileConstraint = SoftmaxCrossEntropyGradTileConstraint())

PULPSoftmaxGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPSoftmaxGradBindings,
                                                             tileConstraint = SoftmaxGradTileConstraint())

PULPReduceSumTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPReduceSumBindings,
                                                           tileConstraint = ReduceSumTileConstraint())

PULPSGDTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPSGDBindings,
                                                     tileConstraint = SGDTileConstraint())

PULPInPlaceAccumulatorV2TilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = PULPInPlaceAccumulatorV2Bindings, tileConstraint = InPlaceAccumulatorV2TileConstraint())

# Ported from Deeploy zo-support (FP32 PerturbRademacher ZO op): per-tile seed via PerturbTileConstraint. -- QW
PULPPerturbRademacherTilingReadyBindings = TilingReadyNodeBindings(  # -- QW
    nodeBindings = PULPPerturbRademacherBindings, tileConstraint = PerturbTileConstraint())  # -- QW
# -- QW (dual_pass_fusion exp8): DualHalf (dual-pass tail), untiled. -- QW
from Deeploy.Targets.PULPOpen.Bindings import PULPDualHalfBindings as _DHB  # -- QW
from Deeploy.Targets.PULPOpen.TileConstraints.UntiledTileConstraint import DualHalfTileConstraint  # -- QW
PULPDualHalfTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _DHB, tileConstraint = DualHalfTileConstraint())  # -- QW
# -- QW (dual_pass_fusion exp8): fixed-sign fp32 perturbation (dual-pass graph), untiled. -- QW
from Deeploy.Targets.PULPOpen.Bindings import PULPPerturbRademacherFixedBindings as _PRFB  # -- QW
from Deeploy.Targets.PULPOpen.TileConstraints.UntiledTileConstraint import PerturbRademacherFixedTileConstraint  # -- QW
PULPPerturbRademacherFixedTilingReadyBindings = TilingReadyNodeBindings(  # -- QW
    nodeBindings = _PRFB, tileConstraint = PerturbRademacherFixedTileConstraint())  # -- QW

PULPSliceTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPSliceBindings,
                                                       tileConstraint = SliceTileConstraint())

PULPReduceMeanTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPReduceMeanBindings,
                                                            tileConstraint = ReduceMeanTileConstraint())

PULPConvGradX2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatConvGradX2DBindings,
                                                             tileConstraint = ConvGradX2DIm2ColHWTileConstraint())

PULPConvGradW2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatConvGradW2DBindings,
                                                             tileConstraint = ConvGradW2DTileConstraint())

PULPDWConvGradX2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatDWConvGradX2DBindings,
                                                               tileConstraint = DWConvGradX2DTileConstraint())

PULPDWConvGradW2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatDWConvGradW2DBindings,
                                                               tileConstraint = DWConvGradW2DTileConstraint())

PULPPWConvGradW2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatPWConvGradW2DBindings,
                                                               tileConstraint = PWConvGradWTileConstraint())

PULPPWConvGradX2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatPWConvGradX2DBindings,
                                                               tileConstraint = PWConvGradXTileConstraint())

PULPConvGradBTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatConvGradBBindings,
                                                           tileConstraint = ConvGradBTileConstraint())


# -- QW: RQSPerturbRademacher tiling-ready binding (quantized ZO), ported from shipped Deeploy
from Deeploy.Targets.PULPOpen.TileConstraints.RQSPerturbTileConstraint import RQSPerturbTileConstraint as _RQSPerturbTC  # -- QW
from Deeploy.Targets.PULPOpen.Bindings import PULPRQSPerturbRademacherBindings as _PULPRQSPRB  # -- QW
PULPRQSPerturbRademacherTilingReadyBindings = TilingReadyNodeBindings(nodeBindings=_PULPRQSPRB, tileConstraint=_RQSPerturbTC())  # -- QW


# -- QW: Quant/Dequant tiling-ready bindings (UnaryTileConstraint) for the unfolded-BN mixed quant graph
from Deeploy.Targets.PULPOpen.Bindings import BasicQuantBindings as _BQB2, BasicDequantBindings as _BDB2  # -- QW
PULPQuantTilingReadyBindings = TilingReadyNodeBindings(nodeBindings=_BQB2, tileConstraint=UnaryTileConstraint())  # -- QW
# PULPDequantTilingReadyBindings = TilingReadyNodeBindings(nodeBindings=_BDB2, tileConstraint=UnaryTileConstraint())  # -- QW
PULPDequantTilingReadyBindings = TilingReadyNodeBindings(nodeBindings=_BDB2, tileConstraint=UnaryRowTileConstraint())  # RW 2026-10-13: row tiles (contiguous DMA); this later definition is the one in effect

# -- QW (dual_pass_fusion exp5): tiling-ready binding for the fused dual-pass QZO conv -- QW
from Deeploy.Targets.PULPOpen.Bindings import PULPDualConv2DBindings as _PULPDualConv2DBindings  # -- QW
from Deeploy.Targets.PULPOpen.TileConstraints.DualConvTileConstraint import DualConv2DTileConstraint as _DualConv2DTC  # -- QW
PULPDualConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _PULPDualConv2DBindings,  # -- QW
                                                            tileConstraint = _DualConv2DTC())  # -- QW
# -- QW (dual_pass_fusion exp7 step 3): 7-input RequantizedDualConv (blocks 1-4). -- QW
from Deeploy.Targets.PULPOpen.Bindings import PULPDualConv2D2InBindings as _PULPDualConv2D2InBindings  # -- QW
from Deeploy.Targets.PULPOpen.TileConstraints.DualConv2InTileConstraint import \
    DualConv2D2InTileConstraint as _DualConv2D2InTC  # -- QW
PULPDualConv2D2InTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _PULPDualConv2D2InBindings,  # -- QW
                                                               tileConstraint = _DualConv2D2InTC())  # -- QW
from Deeploy.Targets.PULPOpen.Bindings import PULPDualConv2DStackedBindings as _PULPDualConv2DStackedBindings  # -- QW exp8
PULPDualConv2DStackedTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _PULPDualConv2DStackedBindings,  # -- QW
                                                                   tileConstraint = _DualConv2D2InTC())  # -- QW exp8
# -- QW (dual_pass_fusion exp11): single-pass in-kernel-perturbation conv. -- QW
from Deeploy.Targets.PULPOpen.Bindings import PULPPerturbConv2DBindings as _PULPPerturbConv2DBindings  # -- QW
from Deeploy.Targets.PULPOpen.TileConstraints.DualConv2InTileConstraint import \
    PerturbConv2DTileConstraint as _PerturbConv2DTC  # -- QW
PULPPerturbConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _PULPPerturbConv2DBindings,  # -- QW
                                                               tileConstraint = _PerturbConv2DTC())  # -- QW

# -- QW (dual_pass_fusion exp5a): single-pass control kernel on the ordinary RequantizedConv; same tile constraint
#    as pulp-nn's binding (RQConv2DTileConstraint), so the tiling is identical -- only the kernel call differs. -- QW
from Deeploy.Targets.PULPOpen.Bindings import PULPRQSConv2DFlatBindings as _PULPRQSConv2DFlatBindings  # -- QW
PULPRQSConv2DFlatTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _PULPRQSConv2DFlatBindings,  # -- QW
                                                               tileConstraint = RQConv2DTileConstraint())  # -- QW
# -- QW (dual_pass_fusion exp7a step 2a): Conv4x2 control kernel, same tile constraint as pulp-nn's binding. -- QW
from Deeploy.Targets.PULPOpen.Bindings import PULPRQSConv2D4x2Bindings as _PULPRQSConv2D4x2Bindings  # -- QW
PULPRQSConv2D4x2TilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _PULPRQSConv2D4x2Bindings,  # -- QW
                                                              tileConstraint = RQConv2DTileConstraint())  # -- QW
