"""Canonical glyph-factor resolution stays centralized and copy-safe."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tnfr.constants.canonical import EN_MIX_FACTOR
from tnfr.operators import _op_EN, get_glyph_factors


class _Node:
    def __init__(self, epi: float, neighbors=()):
        self.EPI = epi
        self.epi_kind = "seed"
        self.graph = {}
        self._neighbors = tuple(neighbors)

    def neighbors(self):
        return self._neighbors


def test_partial_graph_override_is_merged_with_canonical_defaults():
    override = {"AL_boost": 0.2}
    node = SimpleNamespace(graph={"GLYPH_FACTORS": override})

    factors = get_glyph_factors(node)

    assert factors["AL_boost"] == 0.2
    assert factors["EN_mix"] == pytest.approx(EN_MIX_FACTOR)
    factors["AL_boost"] = 0.7
    assert override == {"AL_boost": 0.2}


def test_reception_missing_factor_uses_the_canonical_mix():
    neighbor = _Node(1.0)
    node = _Node(0.0, (neighbor,))

    _op_EN(node, {})

    assert node.EPI == pytest.approx(EN_MIX_FACTOR)
