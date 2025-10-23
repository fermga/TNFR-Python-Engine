"""High-level facade that orchestrates TNFR dynamics workflows.

Parameters
----------
run, step : callable
    Entry points that evolve a :class:`~tnfr.types.TNFRGraph` by integrating
    the canonical nodal equation. Use :func:`run` for fully managed multi-step
    evolution and :func:`step` when a caller needs to interleave bespoke
    telemetry or operator injections between iterations.
set_delta_nfr_hook : callable
    Installs a ΔNFR hook inside ``G.graph['compute_delta_nfr']`` so every
    structural operator keeps ``EPI`` and ``νf`` coupled to coherence changes.
default_glyph_selector, parametric_glyph_selector : AbstractSelector
    Canonical selectors that decide which structural operators (glyph codes)
    should fire on each node based on Sense Index, ΔNFR and acceleration
    metrics. Assign one of them to ``G.graph['glyph_selector']`` to tailor the
    evolution loop.
coordination, dnfr, integrators : module
    Submodules that expose specialised primitives for phase coordination,
    ΔNFR computation and integrator lifecycles. They are re-exported here to
    keep structural control available from a single namespace.
ProcessPoolExecutor, apply_glyph, compute_Si : callable
    Utilities surfaced for selector parallelism, explicit operator execution
    and coherence metrics so callers can extend the default orchestration.

Notes
-----
The facade concentrates every moving part required to keep the TNFR dynamics
loop canonical. ``dnfr`` provides the ΔNFR cache machinery and the
``set_delta_nfr_hook`` helper, ``integrators`` wraps numerical integration of
the nodal equation, and ``coordination`` keeps global and local phase aligned.
Both :func:`run` and :func:`step` trigger selectors, hooks and validators in
the same order; the difference is whether the caller owns the step loop.
Auxiliary exports such as :func:`compute_Si`, :func:`adapt_vf_by_coherence` and
:func:`coordinate_global_local_phase` make it possible to stitch custom
feedback while respecting operator closure and ΔNFR semantics.

Examples
--------
>>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
>>> from tnfr.structural import Coherence, Emission, Resonance, create_nfr, run_sequence
>>> from tnfr.dynamics import parametric_glyph_selector, run, set_delta_nfr_hook
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
>>> # doctest: +SKIP
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
