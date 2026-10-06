"""Facade for configured ΔNFR, νf, phase and operator orchestration.

Attributes
----------
run : callable
    Callable that fully manages the evolution loop, integrating the nodal
    equation while enforcing ΔNFR hooks, νf adaptation and phase coordination
    on every step.
step : callable
    Callable entry point for a single iteration that reuses the
    ΔNFR/νf/phase pipeline while letting callers interleave bespoke telemetry
    or operator injections.
set_delta_nfr_hook : callable
    Callable used to install custom ΔNFR supervision under
    ``G.graph['compute_delta_nfr']`` so each operator reorganization stays
    coupled to νf drift and phase targets.
default_glyph_selector, parametric_glyph_selector : AbstractSelector
    Selector implementations that choose glyphs according to ΔNFR trends,
    νf ranges and phase synchrony. These are configured policies, not derived
    autonomous selection laws or guarantees of monotone coherence.
coordination, dnfr, integrators : module
    Re-exported modules providing explicit control over phase alignment,
    ΔNFR caches and integrator lifecycles to centralize orchestration.
ProcessPoolExecutor, apply_glyph, compute_Si : callable
    Re-exported utilities for parallel selector evaluation, explicit glyph
    execution and Si telemetry so ΔNFR, νf and phase traces remain observable.

Notes
-----
The facade aggregates the default hybrid runtime's helpers:
``dnfr`` manages ΔNFR preparation and caching, ``integrators`` drives the
declared nodal updates, and ``coordination`` applies phase coordination.
Complementary exports such as
:func:`~tnfr.dynamics.adaptation.adapt_vf_after_structural_stability` and
:func:`~tnfr.dynamics.coordination.coordinate_global_local_phase` allow custom
feedback loops under their own admission and provenance requirements.
The explicitly selected joint model in ``dynamics.relational`` has separate
field/step entry points. Its local recovery theorem is conditional on positive
held capacities, positive form dissipation and an acute equilibrium geometry;
it does not change this facade's runtime dispatch.

Examples
--------
>>> import networkx as nx
>>> from tnfr.alias import set_dnfr
>>> from tnfr.dynamics import set_delta_nfr_hook
>>> G = nx.path_graph(2)
>>> def declared_pressure(graph, *, n_jobs=None):
...     for node in graph:
...         set_dnfr(graph, node, 0.08)
>>> set_delta_nfr_hook(G, declared_pressure, note="supplied constant pressure")
>>> # This hook writes pressure only; a declared integrator owns EPI evolution.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from ..metrics.sense_index import compute_Si
from ..operators import apply_glyph
from ..types import GlyphCode
from . import canonical, coordination, dnfr, integrators, metabolism
from .adaptation import adapt_vf_after_structural_stability, adapt_vf_by_coherence
from .adaptive_sequences import AdaptiveSequenceSelector
from .aliases import ALIAS_D2EPI, ALIAS_DNFR, ALIAS_DSI, ALIAS_EPI, ALIAS_SI, ALIAS_VF
from .bifurcation import compute_bifurcation_score, get_bifurcation_paths
from .canonical import (
    NodalEquationResult,
    compute_canonical_nodal_derivative,
    validate_nodal_gradient,
    validate_structural_frequency,
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
from .feedback import StructuralFeedbackLoop
from .homeostasis import StructuralHomeostasis
from .integrators import (
    AbstractIntegrator,
    DefaultIntegrator,
    prepare_integration_params,
    update_epi_via_nodal_equation,
)
from .learning import AdaptiveLearningSystem
from .propagation import (
    compute_network_dissonance_field,
    detect_bifurcation_cascade,
    propagate_dissonance,
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
    run,
    step,
)
from .sampling import update_node_sample as _update_node_sample
from .selectors import (
    AbstractSelector,
    DefaultGlyphSelector,
    ParametricGlyphSelector,
    _apply_glyphs,
    _apply_selector,
    _choose_glyph,
    _collect_selector_metrics,
    _configure_selector_weights,
    _prepare_selector_preselection,
    _resolve_preselected_glyph,
    _selector_parallel_jobs,
    _SelectorPreselection,
    default_glyph_selector,
    parametric_glyph_selector,
)
from .structural_clip import (
    StructuralClipStats,
    get_clip_stats,
    reset_clip_stats,
    structural_clip,
)

__all__ = (
    "canonical",
    "coordination",
    "dnfr",
    "integrators",
    "metabolism",
    # Bifurcation dynamics
    "get_bifurcation_paths",
    "compute_bifurcation_score",
    # Propagation dynamics
    "propagate_dissonance",
    "compute_network_dissonance_field",
    "detect_bifurcation_cascade",
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
    "adapt_vf_after_structural_stability",
    "adapt_vf_by_coherence",
    "coordinate_global_local_phase",
    "compute_Si",
    "compute_canonical_nodal_derivative",
    "NodalEquationResult",
    "validate_nodal_gradient",
    "validate_structural_frequency",
    "default_compute_delta_nfr",
    "default_glyph_selector",
    "dnfr_epi_vf_mixed",
    "dnfr_laplacian",
    "dnfr_phase_only",
    "apply_glyph",
    "parametric_glyph_selector",
    "AbstractIntegrator",
    "DefaultIntegrator",
    "prepare_integration_params",
    "run",
    "set_delta_nfr_hook",
    "step",
    "update_epi_via_nodal_equation",
    "structural_clip",
    "StructuralClipStats",
    "get_clip_stats",
    "reset_clip_stats",
    "AdaptiveLearningSystem",
    "StructuralFeedbackLoop",
    "AdaptiveSequenceSelector",
    "StructuralHomeostasis",
)
