"""ΔNFR generator construction utilities."""
from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

from ..types import TNFRGraph

__all__ = ["build_delta_nfr"]


_VARIANTS = {"laplacian", "adjacency"}


def _ensure_graph(graph: TNFRGraph) -> nx.Graph:
    if not isinstance(graph, nx.Graph):  # pragma: no cover - defensive
        raise TypeError("ΔNFR generators require a networkx graph instance.")
    return graph


def build_delta_nfr(
    graph: TNFRGraph,
    *,
    variant: Literal["laplacian", "adjacency"] = "laplacian",
    weight: str = "weight",
    rng: np.random.Generator | None = None,
    noise_scale: float = 0.0,
    dtype: np.dtype = np.dtype(np.complex128),
) -> np.ndarray:
    """Return Hermitian ΔNFR generator built from ``graph`` topology."""

    graph = _ensure_graph(graph)
    if variant not in _VARIANTS:
        allowed = ", ".join(sorted(_VARIANTS))
        raise ValueError(f"Unknown ΔNFR variant: {variant}. Expected one of: {allowed}.")

    if variant == "laplacian":
        base = nx.laplacian_matrix(graph, weight=weight).astype(float).toarray()
    else:
        base = nx.to_numpy_array(graph, weight=weight, dtype=float)

    hermitian = 0.5 * (base + base.T)
    if rng is not None and noise_scale > 0.0:
        noise = rng.standard_normal(hermitian.shape)
        hermitian = hermitian + noise_scale * 0.5 * (noise + noise.T)

    result = np.asarray(hermitian, dtype=np.complex128)
    return np.asarray(0.5 * (result + result.conj().T), dtype=dtype)
