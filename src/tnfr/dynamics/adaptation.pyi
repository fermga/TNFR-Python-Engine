from __future__ import annotations

from ..types import TNFRGraph

__all__ = [
    "adapt_vf_after_structural_stability",
    "adapt_vf_by_coherence",
]


def adapt_vf_after_structural_stability(
    G: TNFRGraph,
    n_jobs: int | None = None,
) -> None: ...


def adapt_vf_by_coherence(
    G: TNFRGraph,
    n_jobs: int | None = None,
) -> None: ...
