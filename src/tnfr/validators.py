"""Validation utilities."""

from __future__ import annotations

from .constants import ALIAS_EPI, ALIAS_VF, get_param
from .helpers import get_attr
from .glyph_history import last_glyph
from .sense import sigma_vector_from_graph
from .constants_glyphs import GLYPHS_CANONICAL_SET

EPS = 1e-9


def _validate_epi_vf(G) -> None:
    cfg = {
        k: float(get_param(G, k)) for k in ("EPI_MIN", "EPI_MAX", "VF_MIN", "VF_MAX")
    }
    for n, data in G.nodes(data=True):
        epi = get_attr(data, ALIAS_EPI, 0.0, strict=True)
        if not (cfg["EPI_MIN"] - EPS <= epi <= cfg["EPI_MAX"] + EPS):
            raise ValueError(f"EPI out of range in node {n}: {epi}")
        vf = get_attr(data, ALIAS_VF, 0.0, strict=True)
        if not (cfg["VF_MIN"] - EPS <= vf <= cfg["VF_MAX"] + EPS):
            raise ValueError(f"VF out of range in node {n}: {vf}")


def _validate_sigma(G) -> None:
    sv = sigma_vector_from_graph(G)
    if sv.get("mag", 0.0) > 1.0 + EPS:
        raise ValueError("σ norm exceeds 1")


def _validate_glyphs(G) -> None:
    for n, data in G.nodes(data=True):
        g = last_glyph(data)
        if g and g not in GLYPHS_CANONICAL_SET:
            raise ValueError(f"Invalid glyph {g} in node {n}")


def run_validators(G) -> None:
    """Run all invariant validators on ``G``."""
    _validate_epi_vf(G)
    _validate_sigma(G)
    _validate_glyphs(G)
