"""Funciones de validación."""

from __future__ import annotations

from .constants import ALIAS_EPI, ALIAS_VF, get_param
from .helpers import get_attr, last_glifo
from .sense import sigma_vector_from_graph, GLYPHS_CANONICAL_SET


def _validate_epi_vf(G) -> None:
    cfg = {
        k: float(get_param(G, k))
        for k in ("EPI_MIN", "EPI_MAX", "VF_MIN", "VF_MAX")
    }
    for n, data in G.nodes(data=True):
        epi = get_attr(data, ALIAS_EPI, 0.0, strict=True)
        if not (cfg["EPI_MIN"] - 1e-9 <= epi <= cfg["EPI_MAX"] + 1e-9):
            raise ValueError(f"EPI fuera de rango en nodo {n}: {epi}")
        vf = get_attr(data, ALIAS_VF, 0.0, strict=True)
        if not (cfg["VF_MIN"] - 1e-9 <= vf <= cfg["VF_MAX"] + 1e-9):
            raise ValueError(f"VF fuera de rango en nodo {n}: {vf}")


def _validate_sigma(G) -> None:
    sv = sigma_vector_from_graph(G)
    if sv.get("mag", 0.0) > 1.0 + 1e-9:
        raise ValueError("Norma de σ excede 1")


def _validate_glifos(G) -> None:
    for n in G.nodes():
        g = last_glifo(G.nodes[n])
        if g and g not in GLYPHS_CANONICAL_SET:
            raise ValueError(f"Glifo inválido {g} en nodo {n}")


def run_validators(G) -> None:
    """Ejecuta todos los validadores de invariantes sobre ``G``."""
    _validate_epi_vf(G)
    _validate_sigma(G)
    _validate_glifos(G)
