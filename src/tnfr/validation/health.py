"""Structural health assessment utilities (Phase 3).

Provides a concise health summary built on the enhanced validation
aggregator. Read-only; never mutates graph state. Intended for CLI
reporting, telemetry dashboards, CI guards.

Key Concepts
------------
Uses canonical field tetrad (Φ_s, |∇φ|, K_φ, ξ_C) plus grammar status
to derive a risk classification and recommended actions.

Public API
----------
compute_structural_health(G, sequence=None, baseline_phi_s=None,
                          **threshold_overrides) -> dict

Return dict keys:
  risk_level            : low | elevated | critical
  status                : valid | invalid (grammar)
  thresholds_exceeded   : mapping of threshold flags
  recommended_actions   : list of action mnemonics
  notes                 : explanatory strings
  field_metrics_subset  : selected scalar metrics for quick display

Recommended Actions Heuristics
------------------------------
If grammar invalid          -> ['review_sequence','add_stabilizer',
                                 'phase_verify']
If ΔΦ_s exceeded            -> ['apply_coherence','reduce_destabilizers']
If |∇φ| exceeded            -> ['phase_resync','apply_coherence']
If |K_φ| flagged            -> ['local_inspection','coherence_cluster']
If ξ_C critical             -> ['checkpoint_state','controlled_silence']
If ξ_C watch only           -> ['monitor_scaling']

Physics Traceability
--------------------
All heuristics escalate stabilizing operators (IL, THOL) or coupling
phase checks (UM prerequisites) aligned with U2/U3.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, List

from .aggregator import run_structural_validation

__all__ = ["compute_structural_health"]


def compute_structural_health(
    G: Any,
    *,
    sequence: Sequence[str] | None = None,
    baseline_phi_s: Dict[Any, float] | None = None,
    **threshold_overrides: Any,
) -> Dict[str, Any]:
    """Compute structural health summary.

    Parameters
    ----------
    G : Graph
        TNFR network.
    sequence : Sequence[str] | None
        Operator glyphs applied (optional for grammar validation).
    baseline_phi_s : dict | None
        Prior structural potential snapshot for drift assessment.
    **threshold_overrides : Any
        Override default thresholds (keys match aggregator params).

    Returns
    -------
    dict
        Health summary payload.
    """

    report = run_structural_validation(
        G,
        sequence=sequence,
        baseline_structural_potential=baseline_phi_s,
        **threshold_overrides,
    )

    recs: List[str] = []
    th = report.thresholds_exceeded

    if report.status == "invalid":
        recs.extend(["review_sequence", "add_stabilizer", "phase_verify"])
    if th.get("delta_phi_s"):
        recs.extend(["apply_coherence", "reduce_destabilizers"])
    if th.get("phase_gradient_max"):
        recs.extend(["phase_resync", "apply_coherence"])
    if th.get("k_phi_flag"):
        recs.extend(["local_inspection", "coherence_cluster"])
    if th.get("xi_c_critical"):
        recs.extend(["checkpoint_state", "controlled_silence"])
    elif th.get("xi_c_watch"):
        recs.append("monitor_scaling")

    # Deduplicate while preserving order
    dedup_recs: List[str] = []
    seen = set()
    for r in recs:
        if r not in seen:
            seen.add(r)
            dedup_recs.append(r)

    # Select scalar metrics for quick display
    fm = report.field_metrics
    field_subset = {
        "mean_phi_s": fm.get("mean_structural_potential"),
        "max_phase_gradient": fm.get("max_phase_gradient"),
        "max_k_phi": fm.get("max_k_phi"),
        "xi_c": fm.get("xi_c"),
        "delta_phi_s": fm.get("delta_phi_s"),
    }

    return {
        "risk_level": report.risk_level,
        "status": report.status,
        "thresholds_exceeded": th,
        "recommended_actions": dedup_recs,
        "notes": report.notes,
        "field_metrics_subset": field_subset,
        "sequence": report.sequence,
        "grammar_errors": [e.to_payload() for e in report.grammar_errors],
    }
