"""Enhanced structural validation aggregator (Phase 3).

Combines grammar validation (U1-U4 primary + optional U6 confinement
telemetry) with canonical structural field thresholds (Φ_s, |∇φ|, K_φ,
ξ_C). Produces a unified report object for downstream tooling (health
checks, telemetry enrichment, CI guards).

Design Principles
-----------------
1. Read-only: Never mutates graph state; all computations are telemetry.
2. Non-invasive: Wraps existing grammar error factory without altering
   its behaviour or the validator core.
3. Extensible: Thresholds overrideable; adding new canonical fields or
   rules only requires updating constants / mapping.
4. Bounded Overhead: Single-pass field computations; avoids recompute.

Threshold Defaults (Canonical / Safety)
--------------------------------------
ΔΦ_s_max      : 2.0    (escape threshold, U6 confinement guidance)
|∇φ|_max      : 0.38   (stable operation upper bound)
|K_φ|_flag    : 3.0    (local confinement / fault zone flag)
ξ_C_crit_mult : 1.0    (ξ_C > system_diameter signals critical approach)
ξ_C_watch_mult: 3.0    (ξ_C > 3× mean_node_distance watch condition)

Report Semantics
----------------
status    : "valid" | "invalid" (grammar only)
risk_level: "low" | "elevated" | "critical" (fields + grammar)
grammar_errors: list[ExtendedGrammarError]
field_metrics : raw field snapshots + aggregates
thresholds_exceeded: dict[name, bool]

Usage
-----
>>> from tnfr.validation.aggregator import run_structural_validation
>>> report = run_structural_validation(G, sequence=["AL","UM","IL"])
>>> if report.status == "invalid":
...     for err in report.grammar_errors: print(err.message)
>>> if report.risk_level != "low":
...     print("Structural risk detected", report.thresholds_exceeded)

Physics Traceability
--------------------
Grammar rules reference nodal equation boundedness and coupling
conditions (U1-U4). Field thresholds derive from empirical validation
summarised in AGENTS.md and docs/XI_C_CANONICAL_PROMOTION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

try:  # Graph dependency (NetworkX-like interface)
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None  # type: ignore

from ..operators.grammar_error_factory import (
    collect_grammar_errors,
    ExtendedGrammarError,
)
from ..physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)
from ..performance.guardrails import PerformanceRegistry

__all__ = [
    "ValidationReport",
    "run_structural_validation",
]


@dataclass(slots=True)
class ValidationReport:
    """Unified structural validation result.

    Attributes
    ----------
    status : str
        "valid" if no grammar errors else "invalid".
    risk_level : str
        "low", "elevated", or "critical" based on field thresholds & grammar.
    grammar_errors : list[ExtendedGrammarError]
        Enriched grammar error payloads (possibly empty).
    field_metrics : dict[str, Any]
        Raw & aggregate field telemetry (per-node maps + summary stats).
    thresholds_exceeded : dict[str, bool]
        Boolean flags per monitored threshold.
    sequence : tuple[str, ...]
        Operator glyph sequence validated.
    notes : list[str]
        Informational annotations (e.g. which conditions set risk level).
    """

    status: str
    risk_level: str
    grammar_errors: List[ExtendedGrammarError]
    field_metrics: Dict[str, Any]
    thresholds_exceeded: Dict[str, bool]
    sequence: tuple[str, ...]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:  # noqa: D401
        return {
            "status": self.status,
            "risk_level": self.risk_level,
            "grammar_errors": [e.to_payload() for e in self.grammar_errors],
            "field_metrics": self.field_metrics,
            "thresholds_exceeded": self.thresholds_exceeded,
            "sequence": self.sequence,
            "notes": self.notes,
        }


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / max(len(vals), 1)


def run_structural_validation(
    G: Any,
    *,
    sequence: Sequence[str] | None = None,
    # Threshold overrides
    max_delta_phi_s: float = 2.0,
    max_phase_gradient: float = 0.38,
    k_phi_flag_threshold: float = 3.0,
    xi_c_critical_multiplier: float = 1.0,
    xi_c_watch_multiplier: float = 3.0,
    # Optional baselines for drift calculations
    baseline_structural_potential: Dict[Any, float] | None = None,
    # Performance instrumentation (opt-in)
    perf_registry: PerformanceRegistry | None = None,
) -> ValidationReport:
    """Run enhanced structural validation aggregating grammar + field safety.

    Parameters
    ----------
    G : Graph
        TNFR network (NetworkX-like) with required node attributes
        for ΔNFR & phase where available.
        where available.
    sequence : Sequence[str] | None
        Operator glyphs applied. If provided, grammar errors collected.
        If None, grammar validation skipped (status remains 'valid'
        unless field risk escalates).
    max_delta_phi_s : float
        Confinement escape threshold (ΔΦ_s). Evaluated against mean
        absolute drift if baseline provided; otherwise potential
        reported without drift flagging.
    max_phase_gradient : float
        Stable operation threshold for |∇φ|.
    k_phi_flag_threshold : float
        Local confinement / fault zone threshold for |K_φ| magnitudes.
    xi_c_critical_multiplier : float
        Critical approach when ξ_C > system_diameter * multiplier.
    xi_c_watch_multiplier : float
        Watch condition when ξ_C > mean_node_distance * multiplier.
    baseline_structural_potential : dict | None
        Optional prior Φ_s snapshot to compute drift; if omitted
        ΔΦ_s not computed.
    perf_registry : PerformanceRegistry | None
        Optional registry for timing measurements (opt-in overhead).

    Returns
    -------
    ValidationReport
        Unified structural validation result.
    """

    notes: List[str] = []

    # Performance start (if instrumentation active)
    start_time = None
    if perf_registry is not None:
        try:
            import time as _t
            start_time = _t.perf_counter()
        except Exception:  # pragma: no cover
            start_time = None

    # Grammar errors (read-only enrichment)
    grammar_errors: List[ExtendedGrammarError] = []
    if sequence is not None:
        grammar_errors = collect_grammar_errors(sequence)
    status = "valid" if not grammar_errors else "invalid"

    # Field computations (canonical tetrad)
    phi_s_map = compute_structural_potential(G)
    grad_map = compute_phase_gradient(G)
    curvature_map = compute_phase_curvature(G)
    xi_c = estimate_coherence_length(G)

    # Aggregates
    mean_phi_s = _mean(phi_s_map.values())
    mean_grad = _mean(grad_map.values())
    max_grad = max(grad_map.values()) if grad_map else 0.0
    max_k_phi = (
        max(abs(v) for v in curvature_map.values())
        if curvature_map
        else 0.0
    )

    # Drift (optional baseline)
    delta_phi_s = None
    if baseline_structural_potential is not None:
        # Mean absolute difference
        diffs = []
        for n, val in phi_s_map.items():
            prev = baseline_structural_potential.get(n)
            if prev is not None:
                diffs.append(abs(val - prev))
        delta_phi_s = _mean(diffs) if diffs else 0.0

    # System geometry approximation (unweighted)
    if nx is not None:
        try:
            # Use fast diameter approximation (46-111× speedup)
            try:
                from ..utils.fast_diameter import (
                    approximate_diameter_2sweep,
                    compute_eccentricity_cached,
                )
                system_diameter = approximate_diameter_2sweep(G)
            except (ImportError, Exception):
                # Fallback to exact (slow) diameter
                system_diameter = nx.diameter(G)  # type: ignore
                compute_eccentricity_cached = None  # type: ignore
        except Exception:  # pragma: no cover - fallback path
            system_diameter = 0
            compute_eccentricity_cached = None  # type: ignore
        # Mean node distance (cached eccentricity, ~2.3s → 0.000s)
        try:
            if compute_eccentricity_cached is not None:
                ecc = compute_eccentricity_cached(G)
            else:
                ecc = nx.eccentricity(G)  # type: ignore
            mean_node_distance = _mean(ecc.values())
        except Exception:  # pragma: no cover
            mean_node_distance = 0.0
    else:  # pragma: no cover
        system_diameter = 0
        mean_node_distance = 0.0

    # Threshold evaluations
    thresholds_exceeded: Dict[str, bool] = {}

    if delta_phi_s is not None:
        exceeded = delta_phi_s >= max_delta_phi_s
        thresholds_exceeded["delta_phi_s"] = exceeded
        if exceeded:
            notes.append(
                (
                    f"ΔΦ_s drift {delta_phi_s:.3f} ≥ "
                    f"{max_delta_phi_s:.3f} (escape threshold)"
                )
            )

    # Phase gradient (mean & max considered; max is more sensitive to spikes)
    grad_exceeded = max_grad >= max_phase_gradient
    thresholds_exceeded["phase_gradient_max"] = grad_exceeded
    if grad_exceeded:
        notes.append(
            (
                f"max |∇φ| {max_grad:.3f} ≥ "
                f"{max_phase_gradient:.3f} (stress threshold)"
            )
        )

    # Curvature confinement pockets
    k_phi_flag = max_k_phi >= k_phi_flag_threshold
    thresholds_exceeded["k_phi_flag"] = k_phi_flag
    if k_phi_flag:
        notes.append(
            (
                f"|K_φ| max {max_k_phi:.3f} ≥ "
                f"{k_phi_flag_threshold:.3f} (fault zone flag)"
            )
        )

    # Coherence length critical / watch thresholds
    xi_c_critical = (
        system_diameter > 0
        and xi_c > system_diameter * xi_c_critical_multiplier
    )
    xi_c_watch = (
        mean_node_distance > 0
        and xi_c > mean_node_distance * xi_c_watch_multiplier
    )
    thresholds_exceeded["xi_c_critical"] = bool(xi_c_critical)
    thresholds_exceeded["xi_c_watch"] = bool(xi_c_watch)
    if xi_c_critical:
        notes.append(
            (
                f"ξ_C {xi_c:.1f} > diameter {system_diameter} * "
                f"{xi_c_critical_multiplier} (critical approach)"
            )
        )
    elif xi_c_watch:
        notes.append(
            (
                f"ξ_C {xi_c:.1f} > mean_dist {mean_node_distance:.1f} * "
                f"{xi_c_watch_multiplier} (watch)"
            )
        )

    # Risk level derivation
    if status == "invalid":
        risk_level = "critical"
        notes.append("Grammar invalid (U1-U4).")
    else:
        if (
            thresholds_exceeded.get("xi_c_critical")
            or thresholds_exceeded.get("delta_phi_s")
        ):
            risk_level = "critical"
        elif (
            thresholds_exceeded.get("phase_gradient_max")
            or thresholds_exceeded.get("k_phi_flag")
            or thresholds_exceeded.get("xi_c_watch")
        ):
            risk_level = "elevated"
        else:
            risk_level = "low"

    field_metrics: Dict[str, Any] = {
        "phi_s": phi_s_map,
        "phase_gradient": grad_map,
        "phase_curvature": curvature_map,
        "xi_c": xi_c,
        "mean_structural_potential": mean_phi_s,
        "mean_phase_gradient": mean_grad,
        "max_phase_gradient": max_grad,
        "max_k_phi": max_k_phi,
        "delta_phi_s": delta_phi_s,
        "system_diameter": system_diameter,
        "mean_node_distance": mean_node_distance,
    }

    report = ValidationReport(
        status=status,
        risk_level=risk_level,
        grammar_errors=grammar_errors,
        field_metrics=field_metrics,
        thresholds_exceeded=thresholds_exceeded,
        sequence=tuple(sequence or []),
        notes=notes,
    )

    if perf_registry is not None and start_time is not None:
        try:
            import time as _t
            perf_registry.record(
                "validation",
                _t.perf_counter() - start_time,
                meta={
                    "nodes": (
                        G.number_of_nodes()
                        if hasattr(G, "number_of_nodes")
                        else None
                    ),
                    "edges": (
                        G.number_of_edges()
                        if hasattr(G, "number_of_edges")
                        else None
                    ),
                    "sequence_len": (
                        len(sequence) if sequence is not None else 0
                    ),
                    "status": status,
                },
            )
        except Exception:  # pragma: no cover
            pass

    return report
