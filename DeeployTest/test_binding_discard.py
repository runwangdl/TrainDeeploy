# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for binder selection in NodeMapper.typeCheck.

A NodeMapper instance is shared by every node of the same operator in the graph, so
whether a binder suits one node says nothing about whether it suits the next. These
build the mapper by hand, with stub binders that accept or reject per node, so they
run in a second and do not depend on any ONNX asset or backend:

* a binder rejected for an earlier node is still offered to a later one;
* a binder is not permanently lost just because some node in the graph rejected it;
* within a single node, a rejected binder is not retried, so selection stays linear.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Deeploy.DeeployTypes import NodeMapper

# Collected by ci-deeploy.yml, which runs `pytest -m deeploy_internal`.
pytestmark = pytest.mark.deeploy_internal


class _Ctxt:
    """Stand-in for NetworkContext: binder selection only ever copies it."""

    def copy(self):
        return self


class _Parser:
    operatorRepresentation = {}


class _Binder:
    """Accepts a node iff its name is in `accepts`; counts every check."""

    def __init__(self, name, accepts):
        self.name = name
        self.accepts = accepts
        self.checks = 0

    def typeCheck(self, ctxt, node, operatorRepresentation):
        self.checks += 1
        return ctxt, node in self.accepts


def _mapper(*binders):
    return NodeMapper(_Parser(), list(binders))


def test_binder_rejected_for_an_earlier_node_is_offered_to_a_later_one():
    # The Slice case. Binders are tried in order and selection stops at the first that
    # fits, so a binder is only ever discarded when an earlier node had to walk past
    # it. The bug bites when a later node then needs exactly that one: the widest
    # Slice (ends=[256]) came first and walked past the uint8_t rule, and the narrow
    # node behind it was the only one that rule fit.
    narrow = _Binder("uint8_t", accepts = {"slice_128"})
    wide = _Binder("int64_t", accepts = {"slice_256"})
    mapper = _mapper(narrow, wide)

    _, ret = mapper.typeCheck(_Ctxt(), "slice_256")
    assert ret and mapper.binder is wide

    _, ret = mapper.typeCheck(_Ctxt(), "slice_128")
    assert ret, "the second node has a viable binder and must not be reported unmappable"
    assert mapper.binder is narrow


def test_a_binder_survives_rejection_by_an_intervening_node():
    # The same, with an unrelated node in between: whether a binder suits any other
    # node in the graph must not enter into the decision for this one.
    narrow = _Binder("uint8_t", accepts = {"a"})
    wide = _Binder("int64_t", accepts = {"b", "c"})
    mapper = _mapper(narrow, wide)

    for node, expected in (("b", wide), ("c", wide), ("a", narrow)):
        _, ret = mapper.typeCheck(_Ctxt(), node)
        assert ret, f"node {node} has a viable binder"
        assert mapper.binder is expected


def test_a_node_with_no_viable_binder_still_fails():
    mapper = _mapper(_Binder("uint8_t", accepts = set()), _Binder("int64_t", accepts = set()))
    _, ret = mapper.typeCheck(_Ctxt(), "unmappable")
    assert not ret
    assert mapper.bindingsExhausted()


def test_selection_stays_linear_in_the_number_of_bindings():
    # Retrying discarded binders per node is correct; retrying them within a node is
    # not. Each binder must be checked at most once per node, and selection must stop
    # at the first that fits rather than walking the rest.
    binders = [_Binder(f"b{i}", accepts = {"n"}) for i in range(8)]
    binders[0].accepts = set()  # first one rejects, second one takes it
    mapper = _mapper(*binders)

    _, ret = mapper.typeCheck(_Ctxt(), "n")
    assert ret and mapper.binder is binders[1]
    assert [b.checks for b in binders] == [1, 1] + [0] * 6
