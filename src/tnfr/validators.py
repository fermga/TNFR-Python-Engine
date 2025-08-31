"""Validadores de invariantes TNFR."""

from __future__ import annotations
from typing import Iterable

from .constants import ALIAS_EPI, DEFAULTS
from .helpers import _get_attr
from .sense import sigma_vector_global, GLYPHS_CANONICAL
from .helpers import last_glifo


def _validate_epi(G) -> None:
    emin = float(G.graph.get("EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)))
    emax = float(G.graph.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)))
    for n in G.nodes():
        x = float(_get_attr(G.nodes[n], ALIAS_EPI, 0.0))
        if not (emin - 1e-9 <= x <= emax + 1e-9):
            raise ValueError(f"EPI fuera de rango en nodo {n}: {x}")


def _validate_sigma(G) -> None:
    sv = sigma_vector_global(G)
    if sv.get("mag", 0.0) > 1.0 + 1e-9:
        raise ValueError("Norma de σ excede 1")


def _validate_glifos(G) -> None:
    for n in G.nodes():
        g = last_glifo(G.nodes[n])
        if g and g not in GLYPHS_CANONICAL:
            raise ValueError(f"Glifo inválido {g} en nodo {n}")


def run_validators(G) -> None:
    """Ejecuta todos los validadores de invariantes sobre ``G``."""
    _validate_epi(G)
    _validate_sigma(G)
    _validate_glifos(G)
