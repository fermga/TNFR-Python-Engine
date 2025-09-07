"""Validation utilities."""

from __future__ import annotations

from .constants import ALIAS_EPI, ALIAS_VF, get_param
from .alias import get_attr
from .glyph_history import last_glyph
from .sense import sigma_vector_from_graph
from .constants_glyphs import GLYPHS_CANONICAL_SET

EPS = 1e-9

__all__ = ["run_validators"]


def _require_attr(data, alias, node, name):
    """Return attribute value or raise if missing."""
    val = get_attr(data, alias, None)
    if val is None:
        raise ValueError(f"Missing {name} attribute in node {node}")
    return val


def _validate_epi_vf(G) -> None:
    epi_min = float(get_param(G, "EPI_MIN"))
    epi_max = float(get_param(G, "EPI_MAX"))
    vf_min = float(get_param(G, "VF_MIN"))
    vf_max = float(get_param(G, "VF_MAX"))
    for n, data in G.nodes(data=True):
        _check_epi_vf(
            _require_attr(data, ALIAS_EPI, n, "EPI"),
            _require_attr(data, ALIAS_VF, n, "VF"),
            epi_min,
            epi_max,
            vf_min,
            vf_max,
            n,
        )


def _validate_sigma(G) -> None:
    sv = sigma_vector_from_graph(G)
    if sv.get("mag", 0.0) > 1.0 + EPS:
        raise ValueError("σ norm exceeds 1")


def _validate_glyphs(G) -> None:
    for n, data in G.nodes(data=True):
        _check_glyph(last_glyph(data), n)


def _check_epi_vf(epi, vf, epi_min, epi_max, vf_min, vf_max, n):
    if not (epi_min - EPS <= epi <= epi_max + EPS):
        raise ValueError(f"EPI out of range in node {n}: {epi}")
    if not (vf_min - EPS <= vf <= vf_max + EPS):
        raise ValueError(f"VF out of range in node {n}: {vf}")


def _check_glyph(g, n):
    if g and g not in GLYPHS_CANONICAL_SET:
        raise KeyError(f"Invalid glyph {g} in node {n}")


def run_validators(G) -> None:
    """Run all invariant validators on ``G``."""
    _validate_epi_vf(G)
    _validate_glyphs(G)
    _validate_sigma(G)
