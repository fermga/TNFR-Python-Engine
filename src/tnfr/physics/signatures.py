"""
Declared graph-field response signatures for TNFR physics analysis.

The module measures the Structural Field Tetrad before and after one explicit
probe word on a detached graph. The resulting labels describe that finite
response only; they do not certify attractors, chemical elements, or stability
outside the stated probe.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

try:
    import networkx as nx
except ImportError:
    nx = None


from ..constants.canonical import (
    AU_CURVATURE_PERMISSIVE_THRESHOLD,
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
    PI,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
)

from .fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
)


def _copy_deepcopyable_attributes(attributes: Any) -> dict[Any, Any]:
    """Copy logical graph data while ignoring opaque runtime attachments."""

    copied: dict[Any, Any] = {}
    for key, value in attributes.items():
        try:
            copied[key] = deepcopy(value)
        except Exception:
            continue
    return copied


def _detached_probe_graph(graph: "nx.Graph") -> "nx.Graph":
    """Build a logical operator probe without monitors, caches, or locks."""

    if graph.is_multigraph():
        probe = nx.MultiDiGraph() if graph.is_directed() else nx.MultiGraph()
    else:
        probe = nx.DiGraph() if graph.is_directed() else nx.Graph()

    graph_attributes = {
        key: value
        for key, value in graph.graph.items()
        if key != "integrity_monitor" and "cache" not in str(key).lower()
    }
    probe.graph.update(_copy_deepcopyable_attributes(graph_attributes))
    for node, data in graph.nodes(data=True):
        probe.add_node(node, **_copy_deepcopyable_attributes(data))
    if graph.is_multigraph():
        for left, right, key, data in graph.edges(keys=True, data=True):
            probe.add_edge(
                left,
                right,
                key=key,
                **_copy_deepcopyable_attributes(data),
            )
    else:
        for left, right, data in graph.edges(data=True):
            probe.add_edge(left, right, **_copy_deepcopyable_attributes(data))
    return probe


def compute_element_signature(
    G: "nx.Graph", apply_synthetic_step: bool = True
) -> dict[str, Any]:
    """Measure a finite graph-field response under the legacy signature API.

    Parameters
    ----------
    G : nx.Graph
        Graph with expected node attributes:
        - phase/theta: float in [0, 2π)
        - delta_nfr/dnfr: float (structural pressure)
        - Optional: coherence (defaults to 1/(1+|ΔNFR|))
    apply_synthetic_step : bool
        If True, evolve a detached graph copy through the valid probe word
        [Emission, Coherence, Silence] and report the resulting Φ_s drift.

    Returns
    -------
    dict
        Element signature with keys:
        - xi_c: coherence length
        - mean_phase_gradient: mean |∇φ| across nodes
        - mean_phase_curvature_abs: mean |K_φ| across nodes
        - max_phase_curvature_abs: max |K_φ| for hotspot detection

        - phi_s_before: structural potential before synthetic step
        - phi_s_after: structural potential after synthetic step (if applied)
        - phi_s_drift: max_i |Δ Φ_s(i)| between before/after
        - phi_s_mean_abs_drift: mean_i |Δ Φ_s(i)| between snapshots
        - phase_gradient_ok: bool, |∇φ| < 0.196 (π/16 threshold)
        - curvature_hotspots_ok: bool, max |K_φ| < 0.9×π ≈ 2.8274 (canonical threshold)
        - coherence_length_category: str in {localized, medium, extended}
        - signature_class: str, one of {stable, marginal, unstable}
        - signature_scope: finite probe response or unperturbed snapshot
        - potential_drift_assessed: whether the declared probe tested U6 drift

    Notes
    -----
    ``signature_class`` summarizes only this snapshot or the finite response to
    ``synthetic_probe_word``. It does not establish asymptotic stability,
    chemical identity, or behavior under a different word.
    """
    if nx is None:
        raise RuntimeError("NetworkX is required for signature computation")

    # Compute base tetrad metrics. The canonical estimate_coherence_length
    # computes per-node coherence C = 1/(1+|ΔNFR|) internally from ΔNFR (the
    # structural_coherence kernel), so no coherence pre-seeding is required.
    xi_c = float(estimate_coherence_length(G))

    grad_dict = compute_phase_gradient(G)
    grad_values = list(grad_dict.values())
    mean_grad = float(sum(grad_values) / len(grad_values)) if grad_values else 0.0

    curv_dict = compute_phase_curvature(G)
    curv_abs_values = [abs(v) for v in curv_dict.values()]
    mean_curv_abs = (
        float(sum(curv_abs_values) / len(curv_abs_values)) if curv_abs_values else 0.0
    )
    max_curv_abs = float(max(curv_abs_values)) if curv_abs_values else 0.0

    # Structural potential before and after synthetic step (for drift)
    phi_s_before = compute_structural_potential(G)
    phi_s_before_mean = (
        sum(phi_s_before.values()) / len(phi_s_before) if phi_s_before else 0.0
    )

    phi_s_after_mean = phi_s_before_mean  # default: no change
    phi_s_drift = 0.0
    phi_s_mean_abs_drift = 0.0

    synthetic_step_applied = False
    if apply_synthetic_step:
        from ..operators.word_execution import run_network_sequence

        probe = _detached_probe_graph(G)
        run_network_sequence(
            probe,
            ["emission", "coherence", "silence"],
            cycles=1,
            suppress_birth_warnings=True,
        )
        phi_s_after = compute_structural_potential(probe)
        phi_s_after_mean = (
            sum(phi_s_after.values()) / len(phi_s_after) if phi_s_after else 0.0
        )
        drift_values = tuple(
            abs(float(phi_s_after.get(node, 0.0)) - float(value))
            for node, value in phi_s_before.items()
        )
        phi_s_drift = max(drift_values, default=0.0)
        phi_s_mean_abs_drift = (
            sum(drift_values) / len(drift_values) if drift_values else 0.0
        )
        synthetic_step_applied = True

    # Threshold checks (audit 2026: the |∇φ| early-warning level is a heuristic,
    # not derived; the genuine bound is the π phase-wrap shared by |∇φ| and K_φ)
    phase_grad_ok = (
        mean_grad < GRAD_PHI_CANONICAL_THRESHOLD
    )  # heuristic ≈ 0.196 (π/16; kinematic bound is π)
    curv_hotspots_ok = (
        max_curv_abs < K_PHI_CANONICAL_THRESHOLD
    )  # selected 0.9×π margin; the exact wrapped-angle bound is π
    potential_drift_ok: bool | None = None
    if synthetic_step_applied:
        potential_drift_ok = phi_s_drift < U6_STRUCTURAL_POTENTIAL_LIMIT

    # Selected size-relative coherence-length categories (heuristic).
    n_nodes = len(G.nodes())
    typical_diameter = math.sqrt(n_nodes) if n_nodes > 0 else 1.0

    if xi_c < typical_diameter * 0.3:
        xi_c_category = "localized"
    elif xi_c > typical_diameter * 1.2:
        xi_c_category = "extended"
    else:
        xi_c_category = "medium"

    # A failed U6 probe is a hard failure of this finite response. The other
    # labels remain heuristic summaries of the declared snapshot/probe only.
    if potential_drift_ok is False:
        signature_class = "unstable"
    elif phase_grad_ok and curv_hotspots_ok:
        signature_class = "stable"
    elif phase_grad_ok or curv_hotspots_ok or xi_c > 0:
        signature_class = "marginal"
    else:
        signature_class = "unstable"

    return {
        "xi_c": xi_c,
        "mean_phase_gradient": mean_grad,
        "mean_phase_curvature_abs": mean_curv_abs,
        "max_phase_curvature_abs": max_curv_abs,
        "phi_s_before": phi_s_before_mean,
        "phi_s_after": phi_s_after_mean,
        "phi_s_drift": phi_s_drift,
        "phi_s_mean_abs_drift": phi_s_mean_abs_drift,
        "synthetic_step_applied": synthetic_step_applied,
        "potential_drift_assessed": synthetic_step_applied,
        "signature_scope": (
            "finite_declared_probe_response"
            if synthetic_step_applied
            else "unperturbed_snapshot"
        ),
        "synthetic_probe_word": (
            ("emission", "coherence", "silence")
            if synthetic_step_applied
            else ()
        ),
        "phase_gradient_ok": phase_grad_ok,
        "curvature_hotspots_ok": curv_hotspots_ok,
        "potential_drift_ok": potential_drift_ok,
        "coherence_length_category": xi_c_category,
        "signature_class": signature_class,
    }


def compute_au_like_signature(G: "nx.Graph") -> dict[str, Any]:
    """Compute the legacy ``Au-like`` graph-field signature.

    This compatibility API applies a selected set of graph-field thresholds:
    - Extended coherence length (ξ_C >> typical diameter)
    - Low phase gradients (synchronized phases)
    - Stable under synthetic evolution (low ΔΦ_s drift)
    - Moderate curvature without hotspots

    Returns the standard signature with the historical boolean field
    ``is_au_like``. The label does not certify atomic number, metallic behavior,
    chemical composition or optimality.
    """
    signature = compute_element_signature(G, apply_synthetic_step=True)

    # Au-specific criteria (heuristic) - more permissive for current implementation
    is_extended_or_complex = (
        signature["coherence_length_category"] in ["medium", "extended"]
        or len(G.nodes()) > 50  # Legacy size branch of the selected policy
    )
    is_phase_synchronized = (
        signature["mean_phase_gradient"] < PI / 2
    )  # π/2 - permissive for current patterns
    is_probe_response_bounded = bool(signature["potential_drift_ok"])
    is_curvature_mild = (
        signature["max_phase_curvature_abs"] < AU_CURVATURE_PERMISSIVE_THRESHOLD
    )  # 0.95·π ≈ 2.985 (permissive |K_φ|)

    signature["is_au_like"] = (
        is_extended_or_complex
        and is_phase_synchronized
        and is_probe_response_bounded
        and is_curvature_mild
    )
    signature["au_like_scope"] = (
        "legacy heuristic graph-field label; no chemical or metallic inference"
    )

    return signature


__all__ = [
    "compute_element_signature",
    "compute_au_like_signature",
]
