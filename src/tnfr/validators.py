"""Validation utilities."""

from __future__ import annotations

from .constants import ALIAS_EPI, ALIAS_VF, get_param
from .alias import get_attr
from .glyph_history import last_glyph
from .sense import sigma_vector_from_graph
from .constants_glyphs import GLYPHS_CANONICAL_SET

EPS = 1e-9


def _validate_epi_vf(G) -> None:
    cfg = {
        k: float(get_param(G, k))
        for k in ("EPI_MIN", "EPI_MAX", "VF_MIN", "VF_MAX")
    }
    get = get_attr
    epi_min, epi_max = cfg["EPI_MIN"], cfg["EPI_MAX"]
    vf_min, vf_max = cfg["VF_MIN"], cfg["VF_MAX"]
    for n, data in G.nodes(data=True):
        try:
            epi = get(data, ALIAS_EPI, None)
            if epi is None:
                raise KeyError
        except KeyError:
            raise ValueError(f"Missing EPI attribute in node {n}")
        if not (epi_min - EPS <= epi <= epi_max + EPS):
            raise ValueError(f"EPI out of range in node {n}: {epi}")
        try:
            vf = get(data, ALIAS_VF, None)
            if vf is None:
                raise KeyError
        except KeyError:
            raise ValueError(f"Missing VF attribute in node {n}")
        if not (vf_min - EPS <= vf <= vf_max + EPS):
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
