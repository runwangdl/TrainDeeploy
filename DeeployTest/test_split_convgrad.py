# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SplitConvGradPass output classification.

A fused ConvGrad emits dX, dW and dB, but ORT emits dX only when something downstream
needs it. A trainable layer whose input needs no gradient -- the first trainable layer
of a network -- emits dW alone. These build the graphs by hand so they run in a second
and need no ONNX asset or backend.

Reading the outputs positionally takes that lone dW to be dX, and nothing downstream
recovers: the parser binds it as the input gradient, the tiler slices it along its
output channels, and ConvGradXTileConstraintBase reads those cube offsets as batch
offsets -- producing a dY offset of (k, 0, 0, 0) on a tensor whose batch extent is 1,
which minimizeRectangle asserts on.
"""

import os
import sys

import numpy as np
import onnx_graphsurgeon as gs
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Deeploy.Targets.PULPOpen.TopologyOptimizationPasses.SplitConvGradPass import SplitConvGradPass

# Collected by ci-deeploy.yml, which runs `pytest -m deeploy_internal`.
pytestmark = pytest.mark.deeploy_internal

_N, _CIN, _COUT, _H, _W, _K = 1, 8, 16, 24, 24, 1

_X_SHAPE = [_N, _CIN, _H, _W]
_W_SHAPE = [_COUT, _CIN, _K, _K]
_DY_SHAPE = [_N, _COUT, _H, _W]


def _graph(outputs):
    """A single ConvGrad with the given outputs, in the order given."""
    dy = gs.Variable("dy", dtype = np.float32, shape = _DY_SHAPE)
    x = gs.Variable("x", dtype = np.float32, shape = _X_SHAPE)
    w = gs.Variable("w", dtype = np.float32, shape = _W_SHAPE)
    node = gs.Node(op = "ConvGrad", name = "cg", inputs = [dy, x, w], outputs = list(outputs),
                   attrs = {"kernel_shape": [_K, _K]})
    g = gs.Graph(nodes = [node], inputs = [dy, x, w], outputs = list(outputs))
    return g


def _dx():
    return gs.Variable("dx", dtype = np.float32, shape = list(_X_SHAPE))


def _dw():
    return gs.Variable("dw", dtype = np.float32, shape = list(_W_SHAPE))


def _db():
    return gs.Variable("db", dtype = np.float32, shape = [_COUT])


def _split(outputs):
    g = SplitConvGradPass().apply(_graph(outputs))
    g = g[0] if isinstance(g, tuple) else g
    return {n.op: n for n in g.nodes}


def test_a_lone_weight_gradient_becomes_ConvGradW():
    # The case that motivated this: a trainable layer whose input needs no gradient.
    ops = _split([_dw()])
    assert "ConvGradW" in ops, "a ConvGrad emitting only dW must not be split into ConvGradX"
    assert "ConvGradX" not in ops
    assert ops["ConvGradW"].outputs[0].shape == _W_SHAPE


def test_a_lone_input_gradient_becomes_ConvGradX():
    ops = _split([_dx()])
    assert "ConvGradX" in ops and "ConvGradW" not in ops
    assert ops["ConvGradX"].outputs[0].shape == _X_SHAPE


def test_both_gradients_are_routed_by_shape_not_position():
    for outs in ([_dx(), _dw()], [_dw(), _dx()]):
        ops = _split(outs)
        assert set(ops) == {"ConvGradX", "ConvGradW"}
        assert ops["ConvGradX"].outputs[0].shape == _X_SHAPE
        assert ops["ConvGradW"].outputs[0].shape == _W_SHAPE


def test_bias_gradient_is_recognised_by_its_rank():
    ops = _split([_dx(), _dw(), _db()])
    assert set(ops) == {"ConvGradX", "ConvGradW", "ConvGradB"}
    assert ops["ConvGradB"].outputs[0].shape == [_COUT]


def test_the_split_nodes_read_the_right_operands():
    ops = _split([_dx(), _dw()])
    # dX comes from dY and W; dW comes from dY and X. Swapping them silently computes
    # the wrong gradient, so pin it.
    assert [t.name for t in ops["ConvGradX"].inputs] == ["dy", "w"]
    assert [t.name for t in ops["ConvGradW"].inputs] == ["dy", "x"]
