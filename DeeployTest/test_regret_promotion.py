# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the regret-aware promotion selection in scripts/regret_promotion.py.

These check the properties the algorithm is supposed to have, on hand-built tensor
sets rather than on an exported model, so they run in a second and do not depend on
any ONNX asset:

* it never exceeds the budget it is given;
* it promotes everything when everything fits, which is the trivially optimal answer;
* it prefers the tensor that saves more off-chip traffic when only one can fit;
* it beats a size-ordered greedy on a case built so that size ordering is wrong --
  the failure mode regret is meant to fix, where one large high-value tensor crowds
  out several smaller ones that together save more.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

regret_promotion = pytest.importorskip("regret_promotion")

# Collected by ci-deeploy.yml, which runs `pytest -m deeploy_internal`.
pytestmark = pytest.mark.deeploy_internal

Tensor = regret_promotion.Tensor
regret_greedy = regret_promotion.regret_greedy


class _FlatMinimalloc:
    """Stand-in for the allocator: a set fits if the bytes live at once fit.

    The real one packs by lifetime; this sums the sizes of tensors whose intervals
    overlap, which is the same test for the single-interval cases used here and
    keeps the tests independent of the packer.
    """

    def fits(self, blocks, budget):
        # block() yields (name, lo, hi, size).
        if not blocks:
            return True
        lo = min(b[1] for b in blocks)
        hi = max(b[2] for b in blocks)
        for tau in range(lo, hi + 1):
            live = sum(b[3] for b in blocks if b[1] <= tau <= b[2])
            if live > budget:
                return False
        return True


def _tensor(name, size, n_acc, lo, hi):
    # Tensor uses __slots__ and positional arguments.
    return Tensor(name, size, n_acc, lo, hi, True)


def test_promotes_everything_when_everything_fits():
    tensors = [_tensor("a", 100, 2, 0, 3), _tensor("b", 100, 2, 0, 3)]
    promoted = regret_greedy(tensors, budget = 10_000, mm = _FlatMinimalloc())
    assert {t.name for t in promoted} == {"a", "b"}


def test_never_exceeds_the_budget():
    tensors = [_tensor(f"t{i}", 400, 2, 0, 3) for i in range(10)]
    budget = 1000
    promoted = regret_greedy(tensors, budget = budget, mm = _FlatMinimalloc())
    assert _FlatMinimalloc().fits([t.block() for t in promoted], budget)


def test_prefers_the_tensor_that_saves_more_traffic():
    # Same size and lifetime, so the only difference is how often each is read.
    tensors = [_tensor("read_once", 600, 1, 0, 3), _tensor("read_often", 600, 8, 0, 3)]
    promoted = regret_greedy(tensors, budget = 700, mm = _FlatMinimalloc())
    assert [t.name for t in promoted] == ["read_often"]


def test_beats_size_ordering_when_size_ordering_is_wrong():
    # One large tensor read twice against three small ones each read many times:
    # taking the large one first fills the budget and leaves less traffic saved.
    big = _tensor("big", 900, 2, 0, 3)
    smalls = [_tensor(f"small{i}", 300, 9, 0, 3) for i in range(3)]
    budget = 950

    promoted = regret_greedy([big] + smalls, budget = budget, mm = _FlatMinimalloc())
    savedByRegret = sum(t.Q for t in promoted)
    savedByLargestFirst = big.Q

    assert _FlatMinimalloc().fits([t.block() for t in promoted], budget)
    assert savedByRegret > savedByLargestFirst
