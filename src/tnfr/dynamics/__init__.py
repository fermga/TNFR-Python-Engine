"""Facade that keeps ΔNFR, νf and phase orchestration coherent across TNFR dynamics.

Parameters
----------
run : callable
    Fully managed evolution loop that integrates the nodal equation while
    enforcing ΔNFR hooks, νf adaptation and phase coordination on every step.
step : callable
    Single-iteration entry point that exposes the same ΔNFR/νf/phase pipeline
    but lets callers interleave bespoke telemetry or operator injections.
set_delta_nfr_hook : callable
    Installs custom ΔNFR supervision under ``G.graph['compute_delta_nfr']`` so
    each operator reorganization stays coupled to νf drift and phase targets.
default_glyph_selector, parametric_glyph_selector : AbstractSelector
    Canonical selectors that choose glyphs according to ΔNFR trends, νf ranges
    and phase synchrony, ensuring operator firing reinforces coherence.
coordination, dnfr, integrators : module
    Submodules providing explicit control over phase alignment, ΔNFR caches
    and integrator lifecycles; re-exported here to centralize orchestration.
ProcessPoolExecutor, apply_glyph, compute_Si : callable
    Utilities for parallel selector evaluation, explicit glyph execution and
    Si telemetry so ΔNFR, νf and phase traces remain observable.

Notes
-----
The facade aggregates runtime helpers that preserve canonical TNFR dynamics:
``dnfr`` manages ΔNFR preparation and caching, ``integrators`` drives the
numerical updates of νf and EPI, and ``coordination`` synchronizes global and
local phase. Complementary exports such as
:func:`~tnfr.dynamics.adaptation.adapt_vf_by_coherence` and
:func:`~tnfr.dynamics.coordination.coordinate_global_local_phase` allow custom
feedback loops without breaking operator closure.

Examples
--------
>>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
>>> from tnfr.structural import Coherence, Emission, Resonance, create_nfr, run_sequence
>>> from tnfr.dynamics import parametric_glyph_selector, run, set_delta_nfr_hook, step
>>> G, node = create_nfr("seed", epi=0.22, vf=1.0)
>>> def regulate_delta(graph, *, n_jobs=None):
...     for _, nd in graph.nodes(data=True):
...         delta = nd[VF_PRIMARY] * 0.08
...         nd[DNFR_PRIMARY] = delta
...         nd[EPI_PRIMARY] += delta
...         nd[VF_PRIMARY] += delta * 0.05
...     return None
>>> set_delta_nfr_hook(G, regulate_delta, note="ΔNFR guided by νf")
>>> G.graph["glyph_selector"] = parametric_glyph_selector
>>> run_sequence(G, node, [Emission(), Resonance(), Coherence()])
>>> run(G, steps=2, dt=0.05)
>>> # Automatic integration keeps ΔNFR, νf and phase co-modulated.
>>> step(G, dt=0.05)
>>> # Manual control reuses the selector state to consolidate coherence traces.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from ..metrics.sense_index import compute_Si
from ..operators import apply_glyph
from ..utils import get_numpy
from ..validation.grammar import enforce_canonical_grammar, on_applied_glyph
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
