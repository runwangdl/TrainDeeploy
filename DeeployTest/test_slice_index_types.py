# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the index operand types of the Slice bindings.

A Slice carries four index operands -- starts, ends, axes, steps -- whose values are
positions in the tensor being sliced, so their width follows that tensor's extent and
not its element type. These check that the binding tables offer an index type wide
enough for realistic bounds, on the tables themselves rather than on a deployed model,
so they run in a second and need no ONNX asset or backend.

The bound that motivated them is 256: a channel-split training graph slices at the
channel count, and MobileNetV1's widest pointwise layer has 256 channels. With only
uint8_t on offer the node has no viable binding at all and the whole graph is rejected
with "Did not find adequate mapping".
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Deeploy.CommonExtensions.DataTypes import FloatDataTypes
from Deeploy.Targets.GAP9.Bindings import GAP9SliceBindings
from Deeploy.Targets.PULPOpen.Bindings import PULPSliceBindings

# Collected by ci-deeploy.yml, which runs `pytest -m deeploy_internal`.
pytestmark = pytest.mark.deeploy_internal

# The four index operands follow the data operand.
_INDEX_OPERANDS = slice(1, 5)


def _indexTypes(binding):
    return [t.referencedType for t in binding.typeChecker.input_types[_INDEX_OPERANDS]]


def _fits(dataType, value):
    return dataType.checkPromotion(value)


@pytest.mark.parametrize("table, name", [(GAP9SliceBindings, "GAP9SliceBindings"),
                                         (PULPSliceBindings, "PULPSliceBindings")])
def test_some_binding_admits_a_bound_of_256(table, name):
    # 256 is one past uint8_t, and it is exactly the bound a 256-channel split produces.
    viable = [b for b in table if all(_fits(t, 256) for t in _indexTypes(b))]
    assert viable, (f"{name} offers no binding whose index operands can hold 256; every "
                    f"Slice with a bound above 255 is unmappable")


@pytest.mark.parametrize("table, name", [(GAP9SliceBindings, "GAP9SliceBindings"),
                                         (PULPSliceBindings, "PULPSliceBindings")])
def test_every_float_data_type_keeps_a_wide_index_option(table, name):
    # Widening must not be available for only one element type: the data type is chosen
    # by the tensor being sliced, so each one needs its own wide-index binding.
    floats = {t.typeName for t in FloatDataTypes}
    covered = {
        b.typeChecker.input_types[0].referencedType.typeName
        for b in table
        if all(_fits(t, 256) for t in _indexTypes(b))
    }
    missing = floats - covered
    assert not missing, f"{name} has no wide-index binding for {sorted(missing)}"


def test_index_operands_are_uniform_within_a_binding():
    # starts/ends/axes/steps are hoisted from the same node and compared against one
    # rule; a binding that mixed widths would type-check inconsistently.
    for b in GAP9SliceBindings:
        names = {t.typeName for t in _indexTypes(b)}
        assert len(names) == 1, f"binding mixes index types: {sorted(names)}"
