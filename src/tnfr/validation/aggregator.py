"""Shared word validation and structural field observations.

Combines syntax and U1-U5 word policies with separate optional U6 confinement
telemetry and structural field thresholds (Φ_s, |∇φ|, K_φ,
ξ_C). Produces a unified report object for downstream tooling (health
checks, telemetry enrichment, CI guards).

Design Principles
-----------------
1. Observational: Never advances nodal state; shared field/geometry owners may
   maintain rebuildable graph caches. All decisions here are telemetry.
2. Non-invasive: Wraps existing grammar error factory without altering
   its behaviour or the validator core.
3. Extensible: Thresholds overrideable; adding new canonical fields or
   rules only requires updating constants / mapping.
4. Bounded Overhead: Single-pass field computations; avoids recompute.

Threshold Defaults (Selected Monitoring Policies)
-------------------------------------------------
ΔΦ_s_mean     : π/2    (U6 drift policy; evaluated only with a baseline)
|∇φ|_max      : π/16  (selected early-warning policy; exact bound π)
|K_φ|_flag    : 0.9π  (selected margin inside the exact bound π)
ξ_C_crit_mult : 1.0   (finite-size comparison with graph diameter)
ξ_C_watch_mult: π     (comparison with mean node eccentricity)

Report Semantics
----------------
status    : "valid" | "invalid" (grammar only)
risk_level: "low" | "elevated" | "critical" (fields + grammar)
grammar_errors: list[ExtendedGrammarError]
field_metrics : raw field snapshots + aggregates
thresholds_exceeded: dict[name, bool] for evaluated checks only; availability
and failure reasons are retained in field_metrics. Missing observations or
invalid thresholds do not establish a low-risk result.

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
Grammar admission and the configured field cuts are monitoring policies;
they do not derive boundedness, convergence or a universal synchronization
threshold. The exact wrapped-angle bound π is distinct from these selected
warning cuts. Potential drift implements the selected reference-relative U6
policy, not a graph-independent bound on potential.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Sequence

from .._exact_time import finite_represented_real
from ..constants.canonical import (
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
    XI_C_CRITICAL_RATIO,
    XI_C_WATCH_RATIO,
)

try:  # Graph dependency (NetworkX-like interface)
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None  # type: ignore

from ..metrics.common import finite_mean
from ..operators.grammar_error_factory import (
    ExtendedGrammarError,
    collect_grammar_errors,
)
from ..operators.grammar_u6 import validate_structural_potential_confinement
from ..performance.guardrails import PerformanceRegistry
from ..physics.fields import (
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
    observe_phase_curvature,
)

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
    grammar_errors: list[ExtendedGrammarError]
    field_metrics: dict[str, Any]
    thresholds_exceeded: dict[str, bool]
    sequence: tuple[str, ...]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:  # noqa: D401
        return {
            "status": self.status,
            "risk_level": self.risk_level,
            "grammar_errors": [e.to_payload() for e in self.grammar_errors],
            "field_metrics": self.field_metrics,
            "thresholds_exceeded": self.thresholds_exceeded,
            "sequence": self.sequence,
            "notes": self.notes,
        }


def run_structural_validation(
    G: Any,
    *,
    sequence: Sequence[str] | None = None,
    # Threshold overrides
    max_delta_phi_s: float = U6_STRUCTURAL_POTENTIAL_LIMIT,
    max_phase_gradient: float = GRAD_PHI_CANONICAL_THRESHOLD,
    k_phi_flag_threshold: float = K_PHI_CANONICAL_THRESHOLD,
    xi_c_critical_multiplier: float = XI_C_CRITICAL_RATIO,
    xi_c_watch_multiplier: float = XI_C_WATCH_RATIO,
    # Optional baselines for drift calculations
    baseline_structural_potential: dict[Any, float] | None = None,
    # Performance instrumentation (opt-in)
    perf_registry: PerformanceRegistry | None = None,
) -> ValidationReport:
    """Run enhanced structural validation aggregating grammar + field safety.

    Parameters
    ----------
    G : Graph
        TNFR network (NetworkX-like) with required node attributes
        for ΔNFR & phase where available. Field owners may update rebuildable
        graph caches; structural node/edge state is not advanced.
    sequence : Sequence[str] | None
        Operator glyphs applied. If provided, grammar errors collected.
        If None, grammar validation is skipped (status remains 'valid').
        Field risk and evidence availability are reported separately.
    max_delta_phi_s : float
        Selected U6 monitoring threshold for mean absolute ΔΦ_s drift.
        Evaluated only when a baseline is provided.
    max_phase_gradient : float
        Selected early-warning threshold for the maximum local |∇φ|. The
        exact wrapped-angle bound is π; this value is not that bound or the
        measured synchronization onset.
    k_phi_flag_threshold : float
        Selected local warning margin for |K_φ| magnitudes inside the exact
        wrapped-angle bound π.
    xi_c_critical_multiplier : float
        Finite-size diameter flag when ξ_C > system_diameter * multiplier. It
        does not by itself establish a critical transition.
    xi_c_watch_multiplier : float
        Watch condition when ξ_C > mean_node_distance * multiplier.
    baseline_structural_potential : dict | None
        Optional prior Φ_s snapshot to compute drift; if omitted
        ΔΦ_s is not computed. Supplied snapshots must cover exactly the current
        graph nodes and contain finite values. An invalid comparison is reported
        as unavailable, with no drift value or passing threshold flag.
    perf_registry : PerformanceRegistry | None
        Optional registry for timing measurements (opt-in overhead).

    Returns
    -------
    ValidationReport
        Unified structural validation result.
    """

    notes: list[str] = []

    # Performance start (if instrumentation active)
    start_time = None
    if perf_registry is not None:
        try:
            import time as _t

            start_time = _t.perf_counter()
        except Exception:  # pragma: no cover
            start_time = None

    # Grammar errors (read-only enrichment)
    grammar_errors: list[ExtendedGrammarError] = []
    if sequence is not None:
        grammar_errors = collect_grammar_errors(sequence)
    status = "valid" if not grammar_errors else "invalid"

    # Independent shared observations: undefined curvature must not erase a
    # defined phase gradient, and empty support supplies no measured aggregate.
    field_availability = dict.fromkeys(
        ("phi_s", "phase_gradient", "phase_curvature", "xi_c"), False
    )
    field_errors: dict[str, str] = {}
    phi_s_map: dict[Any, float] = {}
    grad_map: dict[Any, float] = {}
    curvature_map: dict[Any, float | None] = {}
    mean_phi_s = mean_grad = max_grad = max_k_phi = xi_c = None
    xi_c_provenance = None
    if G.number_of_nodes():
        try:
            phi_s_map = compute_structural_potential(G)
            mean_phi_s = finite_mean(phi_s_map.values(), name="structural potential")
            field_availability["phi_s"] = bool(phi_s_map)
        except Exception as exc:
            field_errors["phi_s"] = str(exc)
        try:
            phase = observe_phase_curvature(G)
            grad_map = {row.node: row.gradient for row in phase.rows}
            curvature_map = {row.node: row.curvature for row in phase.rows}
            mean_grad = finite_mean(grad_map.values(), name="phase gradient")
            max_grad = max(grad_map.values())
            field_availability["phase_gradient"] = True
            if all(value is not None for value in curvature_map.values()):
                max_k_phi = max(abs(value) for value in curvature_map.values())
                field_availability["phase_curvature"] = True
            else:
                field_errors["phase_curvature"] = (
                    "undefined represented neighbor resultant"
                )
        except Exception as exc:
            field_errors["phase_gradient"] = str(exc)
            field_errors["phase_curvature"] = str(exc)
        try:
            estimate = estimate_coherence_length_with_provenance(G)
            xi_c_provenance = asdict(estimate)
            value = finite_represented_real(xi_c_provenance.pop("value"), "xi_c")[0]
            if value <= 0.0:
                raise ValueError("coherence length requires a finite positive scale")
            xi_c = value
            field_availability["xi_c"] = True
        except Exception as exc:
            field_errors["xi_c"] = str(exc)
    else:
        field_errors = dict.fromkeys(
            field_availability, "empty graph has no field samples"
        )

    # Drift (optional baseline)
    delta_phi_s = None
    u6_status = "not_requested"
    u6_reason = None
    u6_valid = None
    if baseline_structural_potential is not None:
        try:
            u6_valid, delta_phi_s, _ = validate_structural_potential_confinement(
                G,
                baseline_structural_potential,
                phi_s_map,
                threshold=max_delta_phi_s,
                strict=False,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            u6_status = "unavailable"
            u6_reason = str(exc)
            notes.append(f"U6 drift unavailable: {u6_reason}")
        else:
            u6_status = "evaluated"

    # Global finite-size comparisons require connected undirected support;
    # a component-local diameter cannot stand in for disconnected geometry.
    system_diameter = mean_node_distance = None
    if nx is not None:
        try:
            if G.number_of_nodes() < 2 or G.is_directed() or not nx.is_connected(G):
                raise ValueError(
                    "finite-size comparisons require connected undirected support with >= 2 nodes"
                )
            # Use the linear-traversal heuristic; runtime is workload-specific.
            try:
                from ..utils.fast_diameter import (
                    approximate_diameter_2sweep,
                    compute_eccentricity_cached,
                )

                system_diameter = approximate_diameter_2sweep(G)
            except Exception:
                # Fallback to exact (slow) diameter
                system_diameter = nx.diameter(G)  # type: ignore
                compute_eccentricity_cached = None  # type: ignore
        except Exception as exc:
            field_errors["system_geometry"] = str(exc)
            compute_eccentricity_cached = None  # type: ignore
        # Mean node eccentricity (cached when dependencies are unchanged)
        try:
            if system_diameter is None:
                raise ValueError(field_errors["system_geometry"])
            if compute_eccentricity_cached is not None:
                ecc = compute_eccentricity_cached(G)
            else:
                ecc = nx.eccentricity(G)  # type: ignore
            mean_node_distance = finite_mean(ecc.values(), name="node eccentricity")
        except Exception as exc:
            field_errors["mean_node_distance"] = str(exc)
    else:  # pragma: no cover
        field_errors["system_geometry"] = "networkx unavailable"

    # Threshold evaluations
    thresholds_exceeded: dict[str, bool] = {}
    threshold_status = {"delta_phi_s": u6_status}
    threshold_reasons = {"delta_phi_s": u6_reason}
    normalized_thresholds = {}

    def assess(name, value, raw_threshold, *, scale=1.0, inclusive=True):
        """Admit a configured comparison; unavailable is not a passing flag."""
        try:
            threshold = finite_represented_real(raw_threshold, name)[0]
            if raw_threshold < 0:
                raise ValueError("monitoring threshold must be nonnegative")
            if value is None or scale is None or scale <= 0:
                raise ValueError("required field or comparison geometry is unavailable")
            limit = Fraction(threshold) * Fraction(scale)
        except (TypeError, ValueError) as exc:
            threshold_status[name] = "unavailable"
            threshold_reasons[name] = str(exc)
            notes.append(f"{name} unavailable: {exc}")
            return False
        observed = Fraction(value)
        exceeded = observed >= limit if inclusive else observed > limit
        normalized_thresholds[name] = threshold
        threshold_status[name] = "evaluated"
        threshold_reasons[name] = None
        thresholds_exceeded[name] = bool(exceeded)
        return exceeded

    if delta_phi_s is not None:
        exceeded = not u6_valid
        thresholds_exceeded["delta_phi_s"] = exceeded
        if exceeded:
            notes.append(
                (
                    f"ΔΦ_s drift {delta_phi_s:.3f} ≥ "
                    f"{float(max_delta_phi_s):.3f} (selected drift policy)"
                )
            )

    # Phase gradient (mean & max considered; max is more sensitive to spikes)
    grad_exceeded = assess("phase_gradient_max", max_grad, max_phase_gradient)
    if grad_exceeded:
        notes.append(
            (
                f"max |∇φ| {max_grad:.3f} ≥ "
                f"{normalized_thresholds['phase_gradient_max']:.3f} (stress threshold)"
            )
        )

    # Curvature confinement pockets
    k_phi_flag = assess("k_phi_flag", max_k_phi, k_phi_flag_threshold)
    if k_phi_flag:
        notes.append(
            (
                f"|K_φ| max {max_k_phi:.3f} ≥ "
                f"{normalized_thresholds['k_phi_flag']:.3f} (fault zone flag)"
            )
        )

    # Coherence length critical / watch thresholds
    xi_c_critical = assess(
        "xi_c_critical",
        xi_c,
        xi_c_critical_multiplier,
        scale=system_diameter,
        inclusive=False,
    )
    xi_c_watch = assess(
        "xi_c_watch",
        xi_c,
        xi_c_watch_multiplier,
        scale=mean_node_distance,
        inclusive=False,
    )
    if xi_c_critical:
        notes.append(
            (
                f"ξ_C {xi_c:.1f} > diameter {system_diameter} * "
                f"{xi_c_critical_multiplier} (finite-size diameter flag)"
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
        notes.append("Word grammar invalid (syntax or U1-U5).")
    else:
        if thresholds_exceeded.get("xi_c_critical") or thresholds_exceeded.get(
            "delta_phi_s"
        ):
            risk_level = "critical"
        elif (
            not all(field_availability.values())
            or "unavailable" in threshold_status.values()
            or thresholds_exceeded.get("phase_gradient_max")
            or thresholds_exceeded.get("k_phi_flag")
            or thresholds_exceeded.get("xi_c_watch")
        ):
            risk_level = "elevated"
        else:
            risk_level = "low"

    field_metrics: dict[str, Any] = {
        "phi_s": phi_s_map,
        "phase_gradient": grad_map,
        "phase_curvature": curvature_map,
        "xi_c": xi_c,
        "mean_structural_potential": mean_phi_s,
        "mean_phase_gradient": mean_grad,
        "max_phase_gradient": max_grad,
        "max_k_phi": max_k_phi,
        "delta_phi_s": delta_phi_s,
        "u6_status": u6_status,
        "u6_reason": u6_reason,
        "system_diameter": system_diameter,
        "mean_node_distance": mean_node_distance,
        "field_availability": field_availability,
        "field_errors": field_errors,
        "xi_c_provenance": xi_c_provenance,
        "threshold_status": threshold_status,
        "threshold_reasons": threshold_reasons,
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
                        G.number_of_nodes() if hasattr(G, "number_of_nodes") else None
                    ),
                    "edges": (
                        G.number_of_edges() if hasattr(G, "number_of_edges") else None
                    ),
                    "sequence_len": (len(sequence) if sequence is not None else 0),
                    "status": status,
                },
            )
        except Exception:  # pragma: no cover
            pass

    return report
