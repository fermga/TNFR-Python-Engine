from __future__ import annotations

from typing import Any, Literal, Sequence

from tnfr.types import GlyphCode, TNFRGraph

__all__: tuple[str, ...]

dnfr: Any
integrators: Any
metabolism: Any

ALIAS_D2EPI: Sequence[str]
ALIAS_DNFR: Sequence[str]
ALIAS_DSI: Sequence[str]
ALIAS_EPI: Sequence[str]
ALIAS_SI: Sequence[str]
ALIAS_VF: Sequence[str]

AbstractSelector: Any
DefaultGlyphSelector: Any
ParametricGlyphSelector: Any
StructuralFeedbackLoop: Any
AdaptiveSequenceSelector: Any
StructuralHomeostasis: Any
AdaptiveLearningSystem: Any
_SelectorPreselection: Any
_apply_glyphs: Any
_apply_selector: Any
_choose_glyph: Any
_configure_selector_weights: Any
ProcessPoolExecutor: Any
_maybe_remesh: Any
_normalize_job_overrides: Any
_prepare_dnfr: Any
_prepare_dnfr_data: Any
_prepare_selector_preselection: Any
_resolve_jobs_override: Any
_resolve_preselected_glyph: Any
_run_after_callbacks: Any
_run_before_callbacks: Any
_run_validators: Any
_selector_parallel_jobs: Any
_update_epi_hist: Any
_update_node_sample: Any
_update_nodes: Any
_compute_dnfr: Any
_compute_neighbor_means: Any
_init_dnfr_cache: Any
_refresh_dnfr_vectors: Any
adapt_vf_by_coherence: Any
coordinate_global_local_phase: Any
default_compute_delta_nfr: Any
default_glyph_selector: Any
dnfr_epi_vf_mixed: Any
dnfr_laplacian: Any
dnfr_phase_only: Any
get_numpy: Any
apply_glyph: Any
parametric_glyph_selector: Any

AbstractIntegrator: Any
DefaultIntegrator: Any

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
