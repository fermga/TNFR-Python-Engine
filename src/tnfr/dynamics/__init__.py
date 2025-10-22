"""Facade for TNFR dynamics submodules."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from . import coordination, dnfr, integrators
from .adaptation import adapt_vf_by_coherence
from .aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_DSI,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_VF,
)
from .coordination import coordinate_global_local_phase
from .dnfr import (
    _compute_dnfr,
    _compute_neighbor_means,
    _init_dnfr_cache,
    _prepare_dnfr_data,
    _refresh_dnfr_vectors,
    default_compute_delta_nfr,
    dnfr_epi_vf_mixed,
    dnfr_laplacian,
    dnfr_phase_only,
    set_delta_nfr_hook,
)
from .integrators import (
    AbstractIntegrator,
    DefaultIntegrator,
    prepare_integration_params,
    update_epi_via_nodal_equation,
)
from .runtime import (
    _maybe_remesh,
    _normalize_job_overrides,
    _prepare_dnfr,
    _resolve_jobs_override,
    _run_after_callbacks,
    _run_before_callbacks,
    _run_validators,
    _update_epi_hist,
    _update_nodes,
    apply_canonical_clamps,
    run,
    step,
    validate_canon,
)
from .sampling import update_node_sample as _update_node_sample
from .selectors import (
    AbstractSelector,
    DefaultGlyphSelector,
    GlyphCode,
    ParametricGlyphSelector,
    _SelectorPreselection,
    _apply_glyphs,
    _apply_selector,
    _choose_glyph,
    _collect_selector_metrics,
    _configure_selector_weights,
    _prepare_selector_preselection,
    _resolve_preselected_glyph,
    _selector_parallel_jobs,
    default_glyph_selector,
    parametric_glyph_selector,
)
from ..operators import apply_glyph
from ..metrics.sense_index import compute_Si
from ..utils import get_numpy
from ..validation.grammar import enforce_canonical_grammar, on_applied_glyph

__all__ = (
    "coordination",
    "dnfr",
    "integrators",
    "ALIAS_D2EPI",
    "ALIAS_DNFR",
    "ALIAS_DSI",
    "ALIAS_EPI",
    "ALIAS_SI",
    "ALIAS_VF",
    "AbstractSelector",
    "DefaultGlyphSelector",
    "ParametricGlyphSelector",
    "GlyphCode",
    "_SelectorPreselection",
    "_apply_glyphs",
    "_apply_selector",
    "_choose_glyph",
    "_collect_selector_metrics",
    "_configure_selector_weights",
    "ProcessPoolExecutor",
    "_maybe_remesh",
    "_normalize_job_overrides",
    "_prepare_dnfr",
    "_prepare_dnfr_data",
    "_prepare_selector_preselection",
    "_resolve_jobs_override",
    "_resolve_preselected_glyph",
    "_run_after_callbacks",
    "_run_before_callbacks",
    "_run_validators",
    "_selector_parallel_jobs",
    "_update_epi_hist",
    "_update_node_sample",
    "_update_nodes",
    "_compute_dnfr",
    "_compute_neighbor_means",
    "_init_dnfr_cache",
    "_refresh_dnfr_vectors",
    "adapt_vf_by_coherence",
    "apply_canonical_clamps",
    "coordinate_global_local_phase",
    "compute_Si",
    "default_compute_delta_nfr",
    "default_glyph_selector",
    "dnfr_epi_vf_mixed",
    "dnfr_laplacian",
    "dnfr_phase_only",
    "enforce_canonical_grammar",
    "get_numpy",
    "on_applied_glyph",
    "apply_glyph",
    "parametric_glyph_selector",
    "AbstractIntegrator",
    "DefaultIntegrator",
    "prepare_integration_params",
    "run",
    "set_delta_nfr_hook",
    "step",
    "update_epi_via_nodal_equation",
    "validate_canon",
)

