"""Validation utilities."""

from __future__ import annotations

import math
import sys

from .constants import ALIAS_EPI, ALIAS_VF, get_param
from .alias import get_attr
from .glyph_history import last_glyph
from .sense import sigma_vector_from_graph
from .constants_glyphs import GLYPHS_CANONICAL_SET

__all__ = ["run_validators"]


def _require_attr(data, alias, node, name):
    """Return attribute value or raise if missing."""
    val = get_attr(data, alias, None)
    if val is None:
        raise ValueError(f"Missing {name} attribute in node {node}")
    return val

def _validate_sigma(G) -> None:
    sv = sigma_vector_from_graph(G)
    if sv.get("mag", 0.0) > 1.0 + sys.float_info.epsilon:
        raise ValueError("σ norm exceeds 1")


def _check_epi_vf(epi, vf, epi_min, epi_max, vf_min, vf_max, n):
    if epi < epi_min and not math.isclose(epi, epi_min):
        raise ValueError(f"EPI out of range in node {n}: {epi}")
    if epi > epi_max and not math.isclose(epi, epi_max):
        raise ValueError(f"EPI out of range in node {n}: {epi}")
    if vf < vf_min and not math.isclose(vf, vf_min):
        raise ValueError(f"VF out of range in node {n}: {vf}")
    if vf > vf_max and not math.isclose(vf, vf_max):
        raise ValueError(f"VF out of range in node {n}: {vf}")


def _check_glyph(g, n):
    if g and g not in GLYPHS_CANONICAL_SET:
        raise KeyError(f"Invalid glyph {g} in node {n}")


def run_validators(G) -> None:
    """Run all invariant validators on ``G`` with a single node pass."""
    epi_min = float(get_param(G, "EPI_MIN"))
    epi_max = float(get_param(G, "EPI_MAX"))
    vf_min = float(get_param(G, "VF_MIN"))
    vf_max = float(get_param(G, "VF_MAX"))

    for n, data in G.nodes(data=True):
        epi = _require_attr(data, ALIAS_EPI, n, "EPI")
        vf = _require_attr(data, ALIAS_VF, n, "VF")
        _check_epi_vf(epi, vf, epi_min, epi_max, vf_min, vf_max, n)
        _check_glyph(last_glyph(data), n)

    _validate_sigma(G)
