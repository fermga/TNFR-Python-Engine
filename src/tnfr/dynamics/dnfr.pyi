from __future__ import annotations

from typing import Any

from tnfr.types import DeltaNFRHook, TNFRGraph
from tnfr.utils.cache import DnfrCache as DnfrCache

__all__: tuple[str, ...]

def default_compute_delta_nfr(
    G: TNFRGraph,
    *,
    cache_size: int | None = ...,
    n_jobs: int | None = ...,
) -> None: ...
def dnfr_epi_vf_mixed(G: TNFRGraph, *, n_jobs: int | None = ...) -> None: ...
def dnfr_laplacian(G: TNFRGraph, *, n_jobs: int | None = ...) -> None: ...
def dnfr_phase_only(G: TNFRGraph, *, n_jobs: int | None = ...) -> None: ...
def set_delta_nfr_hook(
    G: TNFRGraph,
    func: DeltaNFRHook,
    *,
    name: str | None = ...,
    note: str | None = ...,
) -> None: ...
def __getattr__(name: str) -> Any: ...
