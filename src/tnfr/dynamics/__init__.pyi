from typing import Any, Literal

from tnfr.types import TNFRGraph

__all__: tuple[str, ...]

def __getattr__(name: str) -> Any: ...

_compute_dnfr: Any
_compute_neighbor_means: Any
_init_dnfr_cache: Any
_prepare_dnfr_data: Any
_refresh_dnfr_vectors: Any
adapt_vf_by_coherence: Any
apply_canonical_clamps: Any
coordinate_global_local_phase: Any
default_compute_delta_nfr: Any
default_glyph_selector: Any
dnfr_epi_vf_mixed: Any
dnfr_laplacian: Any
dnfr_phase_only: Any
parametric_glyph_selector: Any
def prepare_integration_params(
    G: TNFRGraph,
    dt: float | None = ...,
    t: float | None = ...,
    method: Literal["euler", "rk4"] | None = ...,
) -> tuple[float, int, float, Literal["euler", "rk4"]]: ...
run: Any
set_delta_nfr_hook: Any
step: Any
def update_epi_via_nodal_equation(
    G: TNFRGraph,
    *,
    dt: float | None = ...,
    t: float | None = ...,
    method: Literal["euler", "rk4"] | None = ...,
    n_jobs: int | None = ...,
) -> None: ...
validate_canon: Any
