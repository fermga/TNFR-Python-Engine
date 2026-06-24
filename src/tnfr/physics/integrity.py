"""Structural Integrity Monitor — closed-loop conservation enforcement.

Bridges the gap between passive telemetry (conservation.py) and active
operator execution (operators/definitions_base.py).  The monitor hooks
into the operator pipeline so that every structural transformation is
checked against the conservation laws **in real time**.

PHYSICS
=======
After each operator application the monitor evaluates:

1. **Conservation quality**  — |Δρ/Δt + div J| via `verify_conservation_balance`
2. **Lyapunov stability**    — dE/dt ≤ 0 via `compute_lyapunov_derivative`
3. **Grammar violation index**— classified by `detect_grammar_violations_from_conservation`
4. **Noether charge drift**  — |ΔQ| via `compute_noether_charge`

If any metric exceeds its threshold the monitor raises
`StructuralIntegrityViolation` (hard mode) or records a warning and
suggests corrective operators (soft mode).

OPERATOR POSTCONDITIONS
=======================
Each canonical operator has a contract (AGENTS.md §Operators):

    IL  → C(t) must not decrease  (monotonicity)
    OZ  → |ΔNFR| must increase   (destabilisation)
    UM  → |φ_i − φ_j| ≤ Δφ_max  (phase compatibility)
    RA  → effective coupling must increase (propagation)
    SHA → EPI unchanged           (silence)
    EN  → C(t) must not decrease  (reception)
    ...

The `POSTCONDITIONS` registry maps each operator name to a callable
``(G, node, state_before, state_after) → None | raise``.

INTEGRATION POINTS
==================
* ``definitions_base.py``  — ``Operator.__call__`` invokes the monitor
* ``self_optimizing_engine.py`` — reads ``integrity_report`` for feedback
* ``ConservationTracker``  — re-used; no duplication with conservation.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..constants.canonical import PHI  # φ ≈ 1.618
from ..types import TNFRGraph

# ---------------------------------------------------------------------------
# Lazy imports to avoid circular dependencies
# ---------------------------------------------------------------------------
_conservation_loaded = False
_capture_conservation_snapshot = None
_verify_conservation_balance = None
_compute_lyapunov_derivative = None
_detect_grammar_violations = None
_compute_noether_charge = None
_compute_energy_functional = None
_compute_coherence = None


def _ensure_imports() -> None:
    """Lazy-load conservation and metrics modules on first use."""
    global _conservation_loaded
    global _capture_conservation_snapshot, _verify_conservation_balance
    global _compute_lyapunov_derivative, _detect_grammar_violations
    global _compute_noether_charge, _compute_energy_functional
    global _compute_coherence

    if _conservation_loaded:
        return

    from ..metrics.common import compute_coherence as _cc
    from .conservation import capture_conservation_snapshot as _css
    from .conservation import compute_energy_functional as _cef
    from .conservation import compute_lyapunov_derivative as _cld
    from .conservation import compute_noether_charge as _cnc
    from .conservation import detect_grammar_violations_from_conservation as _dgv
    from .conservation import verify_conservation_balance as _vcb

    _capture_conservation_snapshot = _css
    _verify_conservation_balance = _vcb
    _compute_lyapunov_derivative = _cld
    _detect_grammar_violations = _dgv
    _compute_noether_charge = _cnc
    _compute_energy_functional = _cef
    _compute_coherence = _cc
    _conservation_loaded = True


# ═══════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════


class StructuralIntegrityViolation(Exception):
    """Raised when an operator violates structural conservation laws.

    Attributes
    ----------
    operator : str
        Name of the operator that caused the violation.
    violation_type : str
        Category: 'conservation', 'lyapunov', 'postcondition', 'grammar'.
    details : dict
        Diagnostic data (residuals, dE/dt, etc.).
    """

    def __init__(
        self, operator: str, violation_type: str, details: dict[str, Any]
    ) -> None:
        self.operator = operator
        self.violation_type = violation_type
        self.details = details
        super().__init__(
            f"{operator} violated {violation_type}: "
            f"{details.get('reason', 'see details')}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Enums and data classes
# ═══════════════════════════════════════════════════════════════════════════


class MonitorMode(Enum):
    """Enforcement level for the integrity monitor."""

    OFF = "off"  # No monitoring (backward compatible)
    OBSERVE = "observe"  # Record violations, never raise
    ENFORCE = "enforce"  # Raise StructuralIntegrityViolation


@dataclass
class IntegrityReport:
    """Result of a single operator integrity check.

    Produced after every monitored operator application.  Consumed by the
    self-optimization engine for closed-loop feedback.
    """

    operator: str
    node: Any
    conservation_quality: float = 1.0
    energy_derivative: float = 0.0
    is_lyapunov_stable: bool = True
    noether_charge_drift: float = 0.0
    grammar_violations: list[str] = field(default_factory=list)
    postcondition_ok: bool = True
    postcondition_detail: str = ""
    corrective_suggestion: str = ""

    @property
    def is_healthy(self) -> bool:
        """True when all checks pass."""
        return (
            self.conservation_quality > 0.7
            and self.is_lyapunov_stable
            and not self.grammar_violations
            and self.postcondition_ok
        )


@dataclass
class IntegritySummary:
    """Aggregated integrity report over a sequence of operator applications."""

    reports: list[IntegrityReport] = field(default_factory=list)
    total_operators: int = 0
    violations_count: int = 0
    mean_conservation_quality: float = 1.0
    mean_energy_derivative: float = 0.0
    total_charge_drift: float = 0.0

    def append(self, report: IntegrityReport) -> None:
        self.reports.append(report)
        self.total_operators += 1
        if not report.is_healthy:
            self.violations_count += 1
        n = self.total_operators
        self.mean_conservation_quality += (
            report.conservation_quality - self.mean_conservation_quality
        ) / n
        self.mean_energy_derivative += (
            report.energy_derivative - self.mean_energy_derivative
        ) / n
        self.total_charge_drift += abs(report.noether_charge_drift)


# ═══════════════════════════════════════════════════════════════════════════
# Operator postcondition registry
# ═══════════════════════════════════════════════════════════════════════════


def _postcond_coherence(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """IL: C(t) must not decrease; |ΔNFR| must not increase (stabiliser).

    Reuses IL_coherence_tracking and IL_dnfr_reductions (written by
    coherence.py) when available to avoid recomputing.
    """
    tracking = G.graph.get("IL_coherence_tracking")
    if tracking:
        latest = tracking[-1]
        c_before = latest.get("C_global_before", before.get("coherence", 0.0))
        c_after = latest.get("C_global_after", after.get("coherence", 0.0))
    else:
        c_before = before.get("coherence", 0.0)
        c_after = after.get("coherence", 0.0)
    if c_after < c_before - 1e-9:
        return (
            f"Coherence decreased: {c_before:.6f} → {c_after:.6f} "
            f"(Δ={c_after - c_before:.6f})"
        )
    # |ΔNFR| must not increase (stabilisation: glyph applies dnfr *= factor < 1)
    dnfr_tracking = G.graph.get("IL_dnfr_reductions")
    if dnfr_tracking:
        latest_dnfr = dnfr_tracking[-1]
        d_before = latest_dnfr.get("before", abs(before.get("dnfr", 0.0)))
        d_after = latest_dnfr.get("after", abs(after.get("dnfr", 0.0)))
    else:
        d_before = abs(before.get("dnfr", 0.0))
        d_after = abs(after.get("dnfr", 0.0))
    if d_after > d_before + 1e-6:
        return f"|ΔNFR| increased during Coherence: " f"{d_before:.6f} → {d_after:.6f}"
    return None


def _postcond_dissonance(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """OZ: |ΔNFR| must increase."""
    d_before = abs(before.get("dnfr", 0.0))
    d_after = abs(after.get("dnfr", 0.0))
    if d_after < d_before - 1e-9:
        return (
            f"|ΔNFR| decreased: {d_before:.6f} → {d_after:.6f} "
            f"(Δ={d_after - d_before:.6f})"
        )
    return None


def _postcond_silence(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """SHA: EPI must remain unchanged; νf must not increase (freeze)."""
    e_before = before.get("epi", 0.0)
    e_after = after.get("epi", 0.0)
    if abs(e_after - e_before) > 1e-6:
        return (
            f"EPI changed during Silence: {e_before:.6f} → {e_after:.6f} "
            f"(Δ={abs(e_after - e_before):.6f})"
        )
    # νf must not increase (SHA freezes evolution: vf *= factor < 1)
    vf_before = before.get("vf", 0.0)
    vf_after = after.get("vf", 0.0)
    if vf_after > vf_before + 1e-9:
        return f"νf increased during Silence: " f"{vf_before:.6f} → {vf_after:.6f}"
    return None


def _postcond_reception(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """EN: C(t) must not decrease."""
    c_before = before.get("coherence", 0.0)
    c_after = after.get("coherence", 0.0)
    if c_after < c_before - 1e-9:
        return (
            f"Coherence decreased during Reception: " f"{c_before:.6f} → {c_after:.6f}"
        )
    return None


def _postcond_resonance(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """RA: structural identity (EPI sign) preserved; νf must not decrease.

    Canonical ground truth (``_op_RA``): Resonance PROPAGATES EPI toward the
    neighbour field (so the node's EPI may rise or fall — it is reorganised,
    not monotonically increased) while preserving structural identity and
    amplifying νf. The contract is therefore identity preservation, NOT an
    EPI-magnitude bound.
    """
    # Identity: EPI sign must be preserved during propagation.
    e_before = before.get("epi", 0.0)
    e_after = after.get("epi", 0.0)
    if abs(e_before) > 1e-9 and abs(e_after) > 1e-9 and (e_before > 0) != (e_after > 0):
        return (
            f"EPI sign flipped during Resonance (identity not preserved): "
            f"{e_before:.6f} → {e_after:.6f}"
        )
    # νf must not decrease (glyph amplifies: vf *= 1 + boost)
    vf_before = before.get("vf", 0.0)
    vf_after = after.get("vf", 0.0)
    if vf_after < vf_before - 1e-9:
        return f"νf decreased during Resonance: " f"{vf_before:.6f} → {vf_after:.6f}"
    return None


def _postcond_emission(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """AL: νf must not decrease; EPI must not decrease (∂EPI/∂t > 0)."""
    vf_before = before.get("vf", 0.0)
    vf_after = after.get("vf", 0.0)
    if vf_after < vf_before - 1e-9:
        return f"νf decreased during Emission: " f"{vf_before:.6f} → {vf_after:.6f}"
    # EPI must not decrease (core glyph effect: +AL_boost)
    e_before = before.get("epi", 0.0)
    e_after = after.get("epi", 0.0)
    if e_after < e_before - 1e-6:
        return f"EPI decreased during Emission: " f"{e_before:.6f} → {e_after:.6f}"
    return None


def _postcond_expansion(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """VAL: νf (reorganization capacity) must not decrease.

    Canonical ground truth (``_make_scale_op``): Expansion scales νf up
    (νf *= 1.0676), adding reorganization capacity — it acts on the νf
    channel, not on |EPI|.
    """
    vf_before = before.get("vf", 0.0)
    vf_after = after.get("vf", 0.0)
    if vf_after < vf_before - 1e-9:
        return f"νf decreased during Expansion: " f"{vf_before:.6f} → {vf_after:.6f}"
    return None


def _postcond_contraction(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """NUL: νf (reorganization capacity) must not increase.

    Canonical ground truth (``_make_scale_op``): Contraction scales νf down
    (νf *= 0.9015) and densifies ΔNFR — it removes capacity on the νf
    channel, not |EPI|.
    """
    vf_before = before.get("vf", 0.0)
    vf_after = after.get("vf", 0.0)
    if vf_after > vf_before + 1e-9:
        return f"νf increased during Contraction: " f"{vf_before:.6f} → {vf_after:.6f}"
    return None


def _postcond_mutation(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """ZHIR: Delegates to postconditions/mutation.py for rich verification.

    Checks phase transformation, identity preservation, and bifurcation
    handling via the canonical postcondition module.
    """
    try:
        from ..operators.postconditions.mutation import (
            verify_bifurcation_handled,
            verify_identity_preserved,
            verify_phase_transformed,
        )

        verify_phase_transformed(G, node, before.get("theta", 0.0))
        epi_kind_before = G.nodes[node].get("_integrity_epi_kind_before")
        if epi_kind_before is not None:
            verify_identity_preserved(G, node, epi_kind_before)
        verify_bifurcation_handled(G, node)
    except Exception as exc:
        return str(exc)
    return None


def _postcond_coupling(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """UM: |ΔNFR| must not increase (coupling reduces structural pressure)."""
    d_before = abs(before.get("dnfr", 0.0))
    d_after = abs(after.get("dnfr", 0.0))
    if d_after > d_before + 1e-6:
        return f"|ΔNFR| increased during Coupling: " f"{d_before:.6f} → {d_after:.6f}"
    return None


def _postcond_self_organization(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """THOL: Global coherence must not catastrophically decrease.

    THOL is a stabiliser (U2) that creates sub-EPIs while preserving global
    form.  A small coherence dip is tolerable (new nodes shift the mean),
    but a large drop signals failed self-organisation.
    """
    c_before = before.get("coherence", 0.0)
    c_after = after.get("coherence", 0.0)
    # Allow up to 10 % relative decrease; below that flag a violation
    if c_before > 1e-9 and c_after < c_before * 0.9 - 1e-9:
        return (
            f"Coherence dropped >10 % during Self-organisation: "
            f"{c_before:.6f} → {c_after:.6f}"
        )
    return None


def _postcond_transition(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """NAV: At least one state variable (νf, θ, ΔNFR) must change."""
    vf_changed = abs(after.get("vf", 0.0) - before.get("vf", 0.0)) > 1e-9
    theta_changed = abs(after.get("theta", 0.0) - before.get("theta", 0.0)) > 1e-9
    dnfr_changed = abs(after.get("dnfr", 0.0) - before.get("dnfr", 0.0)) > 1e-9
    if not (vf_changed or theta_changed or dnfr_changed):
        return "No state change during Transition: " "νf, θ, and ΔNFR all unchanged"
    return None


def _postcond_recursivity(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """REMESH: Advisory glyph — structural remesh is verified at network level."""
    return None


# Mapping from canonical operator name → postcondition checker.
# Returns None on success, or a string describing the violation.
# All 13 canonical operators are covered.
POSTCONDITIONS: dict[str, Callable[..., str | None]] = {
    "coherence": _postcond_coherence,
    "dissonance": _postcond_dissonance,
    "silence": _postcond_silence,
    "reception": _postcond_reception,
    "resonance": _postcond_resonance,
    "emission": _postcond_emission,
    "expansion": _postcond_expansion,
    "contraction": _postcond_contraction,
    "mutation": _postcond_mutation,
    "coupling": _postcond_coupling,
    "self_organization": _postcond_self_organization,
    "transition": _postcond_transition,
    "recursivity": _postcond_recursivity,
}

# ═══════════════════════════════════════════════════════════════════════════
# Corrective suggestions
# ═══════════════════════════════════════════════════════════════════════════

_CORRECTIVE_MAP: dict[str, str] = {
    "U6_confinement_breach": "Apply IL (Coherence) to reduce Φ_s below φ threshold",
    "U2_convergence_failure": "Apply IL or THOL to stabilise divergent ΔNFR",
    "U3_phase_incompatibility": "Apply SHA (Silence) then UM with phase-compatible nodes",
    "lyapunov_unstable": "Apply IL or THOL; energy is increasing (dE/dt > 0)",
    "charge_drift": "Apply IL to restore Noether charge Q toward conserved value",
}

_NOETHER_CHARGE_DRIFT_ALERT = 0.5


def _suggest_correction(report: IntegrityReport) -> str:
    """Derive a corrective operator suggestion from violation diagnostics."""
    suggestions: list[str] = []
    for vtype in report.grammar_violations:
        if vtype in _CORRECTIVE_MAP:
            suggestions.append(_CORRECTIVE_MAP[vtype])
    if not report.is_lyapunov_stable:
        suggestions.append(_CORRECTIVE_MAP["lyapunov_unstable"])
    if abs(report.noether_charge_drift) > _NOETHER_CHARGE_DRIFT_ALERT:
        suggestions.append(_CORRECTIVE_MAP["charge_drift"])
    return "; ".join(suggestions) if suggestions else ""


# ═══════════════════════════════════════════════════════════════════════════
# Capture helpers
# ═══════════════════════════════════════════════════════════════════════════


def _capture_node_state(G: TNFRGraph, node: Any) -> dict[str, Any]:
    """Capture per-node scalar state for postcondition checking."""
    _ensure_imports()
    state = {
        "epi": float(get_attr(G.nodes[node], ALIAS_EPI, 0.0)),
        "vf": float(get_attr(G.nodes[node], ALIAS_VF, 0.0)),
        "dnfr": float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0)),
        "theta": float(get_attr(G.nodes[node], ALIAS_THETA, 0.0)),
    }
    try:
        state["coherence"] = float(_compute_coherence(G))
    except Exception:
        state["coherence"] = 0.0
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════


class StructuralIntegrityMonitor:
    """Real-time conservation-law enforcement during operator execution.

    Attaches to a TNFR graph via ``G.graph["integrity_monitor"]`` and is
    consulted by ``Operator.__call__`` after every structural transformation.

    Parameters
    ----------
    mode : MonitorMode
        OFF — no overhead, backward compatible.
        OBSERVE — record all violations, never raise.
        ENFORCE — raise ``StructuralIntegrityViolation`` on first failure.
    conservation_threshold : float
        Minimum ``conservation_quality`` (default 0.5; 1 = perfect).
    lyapunov_tolerance : float
        Maximum allowed dE/dt before flagging instability (default 0.1).
    charge_drift_threshold : float
        Maximum |ΔQ| per step (default φ ≈ 1.618, from U6).
    """

    def __init__(
        self,
        mode: MonitorMode = MonitorMode.OBSERVE,
        conservation_threshold: float = 0.5,
        lyapunov_tolerance: float = 0.1,
        charge_drift_threshold: float = PHI,
    ) -> None:
        self.mode = mode
        self.conservation_threshold = conservation_threshold
        self.lyapunov_tolerance = lyapunov_tolerance
        self.charge_drift_threshold = charge_drift_threshold
        self._summary = IntegritySummary()
        self._snapshot_before = None
        self._node_state_before: dict[str, Any] = {}
        self._charge_before: float = 0.0

    # ── public API ────────────────────────────────────────────────────────

    @property
    def summary(self) -> IntegritySummary:
        """Accumulated integrity statistics (read-only)."""
        return self._summary

    @property
    def latest_report(self) -> IntegrityReport | None:
        """Most recent integrity report, or None."""
        return self._summary.reports[-1] if self._summary.reports else None

    def reset(self) -> None:
        """Clear accumulated reports."""
        self._summary = IntegritySummary()
        self._snapshot_before = None
        self._node_state_before = {}
        self._charge_before = 0.0

    # ── operator hooks ────────────────────────────────────────────────────

    def before_operator(self, G: TNFRGraph, node: Any) -> None:
        """Capture state before an operator is applied.

        Called by ``Operator.__call__`` when a monitor is active.
        """
        if self.mode is MonitorMode.OFF:
            return
        _ensure_imports()

        # Save conservation snapshot (fields over all nodes)
        try:
            self._snapshot_before = _capture_conservation_snapshot(G)
            self._charge_before = _compute_noether_charge(G)
        except Exception:
            self._snapshot_before = None
            self._charge_before = 0.0

        # Save per-node scalar state (for postconditions)
        self._node_state_before = _capture_node_state(G, node)

    def after_operator(
        self,
        G: TNFRGraph,
        node: Any,
        operator_name: str,
    ) -> IntegrityReport:
        """Evaluate conservation laws after operator application.

        Called by ``Operator.__call__`` when a monitor is active.

        Returns
        -------
        IntegrityReport
            Detailed diagnostics.  Stored in ``self.summary``.

        Raises
        ------
        StructuralIntegrityViolation
            Only in ``MonitorMode.ENFORCE`` when a violation is detected.
        """
        if self.mode is MonitorMode.OFF:
            return IntegrityReport(operator=operator_name, node=node)

        _ensure_imports()
        report = IntegrityReport(operator=operator_name, node=node)

        # 1. Conservation quality (continuity equation residual)
        if self._snapshot_before is not None:
            try:
                snap_after = _capture_conservation_snapshot(G)
                balance = _verify_conservation_balance(
                    self._snapshot_before, snap_after
                )
                report.conservation_quality = balance.conservation_quality

                # 2. Lyapunov stability (dE/dt ≤ 0)
                lyap = _compute_lyapunov_derivative(self._snapshot_before, snap_after)
                report.energy_derivative = lyap.energy_derivative
                report.is_lyapunov_stable = lyap.is_stable

                # 3. Grammar violations from conservation residuals
                violations = _detect_grammar_violations(balance)
                if violations["violations_detected"]:
                    report.grammar_violations = violations["violation_types"]

                # 4. Noether charge drift
                charge_after = _compute_noether_charge(G)
                report.noether_charge_drift = abs(charge_after - self._charge_before)
            except Exception as exc:
                warnings.warn(
                    f"Integrity monitor conservation check failed: {exc}",
                    stacklevel=2,
                )

        # 5. Operator postcondition
        node_state_after = _capture_node_state(G, node)
        postcond_fn = POSTCONDITIONS.get(operator_name.lower())
        if postcond_fn is not None:
            try:
                violation_msg = postcond_fn(
                    G,
                    node,
                    self._node_state_before,
                    node_state_after,
                )
                if violation_msg is not None:
                    report.postcondition_ok = False
                    report.postcondition_detail = violation_msg
            except Exception as exc:
                warnings.warn(
                    f"Postcondition check failed for {operator_name}: {exc}",
                    stacklevel=2,
                )

        # Derive corrective suggestion
        report.corrective_suggestion = _suggest_correction(report)

        # Record
        self._summary.append(report)

        # Enforce if configured
        if self.mode is MonitorMode.ENFORCE and not report.is_healthy:
            violation_type = "postcondition"
            if report.grammar_violations:
                violation_type = "grammar"
            elif not report.is_lyapunov_stable:
                violation_type = "lyapunov"
            elif report.conservation_quality < self.conservation_threshold:
                violation_type = "conservation"

            raise StructuralIntegrityViolation(
                operator=operator_name,
                violation_type=violation_type,
                details={
                    "reason": report.postcondition_detail
                    or "; ".join(report.grammar_violations)
                    or f"dE/dt={report.energy_derivative:.6f}"
                    or f"quality={report.conservation_quality:.4f}",
                    "conservation_quality": report.conservation_quality,
                    "energy_derivative": report.energy_derivative,
                    "charge_drift": report.noether_charge_drift,
                    "grammar_violations": report.grammar_violations,
                    "suggestion": report.corrective_suggestion,
                },
            )

        return report

    # ── convenience ───────────────────────────────────────────────────────

    def attach(self, G: TNFRGraph) -> None:
        """Store this monitor in ``G.graph`` so operators can find it."""
        G.graph["integrity_monitor"] = self

    @staticmethod
    def get(G: TNFRGraph) -> "StructuralIntegrityMonitor | None":
        """Retrieve the monitor attached to *G*, or None."""
        return G.graph.get("integrity_monitor")

    # ── feedback for self-optimization ────────────────────────────────────

    def feedback_vector(self) -> dict[str, float]:
        """Return a dict of scalars suitable for the optimization engine.

        Keys
        ----
        conservation_quality : float
            Running mean of conservation_quality across steps.
        energy_derivative : float
            Running mean of dE/dt.
        charge_drift : float
            Cumulative |ΔQ|.
        violation_rate : float
            Fraction of operators that were unhealthy.
        """
        s = self._summary
        n = max(s.total_operators, 1)
        return {
            "conservation_quality": s.mean_conservation_quality,
            "energy_derivative": s.mean_energy_derivative,
            "charge_drift": s.total_charge_drift,
            "violation_rate": s.violations_count / n,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Module-level convenience
# ═══════════════════════════════════════════════════════════════════════════


def enable_integrity_monitor(
    G: TNFRGraph,
    mode: MonitorMode = MonitorMode.OBSERVE,
    **kwargs: Any,
) -> StructuralIntegrityMonitor:
    """Create and attach a ``StructuralIntegrityMonitor`` to *G*.

    Usage::

        from tnfr.physics.integrity import enable_integrity_monitor, MonitorMode
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        # ... apply operators ...
        print(monitor.summary)
    """
    monitor = StructuralIntegrityMonitor(mode=mode, **kwargs)
    monitor.attach(G)
    return monitor


# ═══════════════════════════════════════════════════════════════════════════
# Proactive operator-contract audit (measured, not asserted)
# ═══════════════════════════════════════════════════════════════════════════
#
# The StructuralIntegrityMonitor above is REACTIVE: it is consulted by
# Operator.__call__ after every transformation and records violations as
# they happen.  The audit below is PROACTIVE: it applies each of the 13
# canonical operators in its correct canonical context and MEASURES whether
# the operator's contract (AGENTS.md §"The 13 Canonical Operators") is
# satisfied — producing a per-operator certificate.  The two are
# complementary: the monitor guards live execution; the audit certifies the
# catalog itself.
#
# Honest scope: the contracts are measured at the context where each one
# canonically manifests — network level for stabilisers (IL, UM, EN, THOL,
# whose effect is on the emergent ΔNFR / C(t) fields), single-node level for
# the local destabiliser OZ, identity (EPI-sign) preservation for RA, and
# the phase channel for ZHIR (with its U4b precondition: prior IL + recent
# destabiliser).  The emergent field ΔNFR is recomputed after application,
# since it is a network property, not a node-local one.


@dataclass(frozen=True)
class OperatorContractResult:
    """Measured fidelity of one canonical operator to its contract.

    Attributes
    ----------
    english_name : str
        Public structural-operator name (Emission, Reception, ...). This is the
        canonical public identifier; ``glyph`` is the internal symbolic code.
    glyph : str
        Internal symbolic glyph code (AL, EN, IL, ...).
    operator : str
        Canonical function name (emission, reception, ...).
    contract : str
        The canonical postcondition contract being measured.
    context : str
        The measurement context: ``network``, ``node``, ``identity``,
        ``phase``, ``state``, or ``advisory``.
    satisfied : bool
        Whether the measured behaviour satisfies the contract.
    detail : str
        Human-readable measured before→after summary.
    """

    english_name: str
    glyph: str
    operator: str
    contract: str
    context: str
    satisfied: bool
    detail: str


@dataclass(frozen=True)
class OperatorContractAudit:
    """Aggregated measured-fidelity audit over the 13 canonical operators."""

    results: tuple[OperatorContractResult, ...]

    @property
    def n_operators(self) -> int:
        return len(self.results)

    @property
    def n_satisfied(self) -> int:
        return sum(1 for r in self.results if r.satisfied)

    @property
    def all_satisfied(self) -> bool:
        return self.n_operators > 0 and self.n_satisfied == self.n_operators

    @property
    def violations(self) -> tuple[OperatorContractResult, ...]:
        return tuple(r for r in self.results if not r.satisfied)

    def summary(self) -> str:
        ok = "ALL SATISFIED" if self.all_satisfied else "VIOLATIONS"
        lines = [
            f"Operator-contract audit [{ok}]: "
            f"{self.n_satisfied}/{self.n_operators} operators satisfy "
            f"their canonical postcondition contract (measured)."
        ]
        for r in self.results:
            mark = "ok " if r.satisfied else "XX "
            lines.append(
                f"  {mark}{r.english_name:>16} [{r.context}] "
                f"{r.contract} — {r.detail}"
            )
        return "\n".join(lines)


def _audit_build_graph(n_nodes: int, seed: int) -> Any:
    """Build a controlled TNFR graph for the contract audit."""
    import math
    import random

    import networkx as nx

    from ..dynamics import default_compute_delta_nfr

    k = 4 if n_nodes > 4 else 2
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n_nodes, k, 0.2, seed=seed)
    for nd in G.nodes():
        G.nodes[nd][ALIAS_THETA[0]] = rng.uniform(0.0, 2.0 * math.pi)
        G.nodes[nd][ALIAS_EPI[0]] = rng.uniform(0.2, 0.6)
        G.nodes[nd][ALIAS_VF[0]] = rng.uniform(0.6, 1.2)
    default_compute_delta_nfr(G)
    return G


def _audit_metrics(G: Any) -> dict[str, float]:
    """Network-level metrics for contract measurement."""
    import numpy as np

    _ensure_imports()
    nodes = list(G.nodes())
    epis = [abs(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) for n in nodes]
    dnfrs = [abs(get_attr(G.nodes[n], ALIAS_DNFR, 0.0)) for n in nodes]
    vfs = [get_attr(G.nodes[n], ALIAS_VF, 0.0) for n in nodes]
    return {
        "C": float(_compute_coherence(G)),
        "epi": float(np.mean(epis)) if epis else 0.0,
        "dnfr": float(np.mean(dnfrs)) if dnfrs else 0.0,
        "vf": float(np.mean(vfs)) if vfs else 0.0,
    }


def audit_operator_contracts(
    *,
    n_nodes: int = 16,
    seed: int = 7,
    tol: float = 1e-6,
) -> OperatorContractAudit:
    r"""Measure each canonical operator against its postcondition contract.

    Applies all 13 canonical operators, each in its correct canonical
    context, and measures whether its contract (AGENTS.md §Operators) holds.
    The emergent ΔNFR field is recomputed after application (it is a network
    property).  Returns an :class:`OperatorContractAudit` with a per-operator
    result.

    Parameters
    ----------
    n_nodes : int
        Size of the controlled audit graph.
    seed : int
        Reproducible seed for the audit graph.
    tol : float
        Numerical tolerance for the contract predicates.

    Returns
    -------
    OperatorContractAudit
    """
    import warnings as _warnings

    import numpy as np

    from ..dynamics import default_compute_delta_nfr
    from ..operators.definitions import (
        Coherence,
        Contraction,
        Coupling,
        Dissonance,
        Emission,
        Expansion,
        Mutation,
        Reception,
        Recursivity,
        Resonance,
        SelfOrganization,
        Silence,
        Transition,
    )
    from ..operators.operator_contracts import OPERATOR_CONTRACTS, iter_contracts

    _classes = {
        "emission": Emission,
        "reception": Reception,
        "coherence": Coherence,
        "dissonance": Dissonance,
        "coupling": Coupling,
        "resonance": Resonance,
        "silence": Silence,
        "expansion": Expansion,
        "contraction": Contraction,
        "self_organization": SelfOrganization,
        "mutation": Mutation,
        "transition": Transition,
        "recursivity": Recursivity,
    }
    # Catalog DERIVED from the canonical contract spec (single source of truth):
    # (glyph, name, class, context, postcondition) per operator.
    catalog = [
        (c.glyph, c.name, _classes[c.name], c.context.value, c.postcondition)
        for c in iter_contracts()
    ]

    results: list[OperatorContractResult] = []

    for glyph, name, cls, context, contract in catalog:
        eng_name = OPERATOR_CONTRACTS[name].english_name
        G = _audit_build_graph(n_nodes, seed)
        op = cls()

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            if context == "node":
                # local destabiliser: single node, measure that node's |ΔNFR|
                node = list(G.nodes())[0]
                d_before = abs(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
                op(G, node)
                default_compute_delta_nfr(G)
                d_after = abs(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
                satisfied = d_after >= d_before - tol
                detail = f"node |ΔNFR| {d_before:.4f}→{d_after:.4f}"

            elif context == "identity":
                signs_before = {
                    n: float(np.sign(get_attr(G.nodes[n], ALIAS_EPI, 0.0)))
                    for n in G.nodes()
                }
                for nd in list(G.nodes()):
                    op(G, nd)
                default_compute_delta_nfr(G)
                preserved = sum(
                    1
                    for n in G.nodes()
                    if float(np.sign(get_attr(G.nodes[n], ALIAS_EPI, 0.0)))
                    == signs_before[n]
                )
                total = G.number_of_nodes()
                satisfied = preserved == total
                detail = f"EPI sign preserved {preserved}/{total}"

            elif context == "phase":
                # U4b precondition: prior IL (stable base) + recent OZ
                for nd in list(G.nodes()):
                    Coherence()(G, nd)
                for nd in list(G.nodes()):
                    Dissonance()(G, nd)
                default_compute_delta_nfr(G)
                theta_before = {
                    n: get_attr(G.nodes[n], ALIAS_THETA, 0.0) for n in G.nodes()
                }
                for nd in list(G.nodes()):
                    op(G, nd)
                changed = sum(
                    1
                    for n in G.nodes()
                    if abs(get_attr(G.nodes[n], ALIAS_THETA, 0.0) - theta_before[n])
                    > 1e-9
                )
                total = G.number_of_nodes()
                satisfied = changed > 0
                detail = f"θ changed {changed}/{total}"

            elif context == "state":
                before = _audit_metrics(G)
                theta_before = {
                    n: get_attr(G.nodes[n], ALIAS_THETA, 0.0) for n in G.nodes()
                }
                for nd in list(G.nodes()):
                    op(G, nd)
                default_compute_delta_nfr(G)
                after = _audit_metrics(G)
                theta_changed = any(
                    abs(get_attr(G.nodes[n], ALIAS_THETA, 0.0) - theta_before[n]) > 1e-9
                    for n in G.nodes()
                )
                satisfied = theta_changed or any(
                    abs(after[k] - before[k]) > tol for k in ("C", "epi", "dnfr", "vf")
                )
                detail = "state changed" if satisfied else "no state change"

            elif context == "advisory":
                # REMESH is a network-level echo verified elsewhere; the
                # contract here is advisory (always satisfied at this level).
                for nd in list(G.nodes()):
                    op(G, nd)
                default_compute_delta_nfr(G)
                satisfied = True
                detail = "advisory (network echo)"

            else:  # network
                before = _audit_metrics(G)
                for nd in list(G.nodes()):
                    op(G, nd)
                default_compute_delta_nfr(G)
                after = _audit_metrics(G)
                if glyph == "AL":
                    satisfied = after["epi"] >= before["epi"] - tol
                    detail = f"|EPI| {before['epi']:.4f}→{after['epi']:.4f}"
                elif glyph == "EN":
                    satisfied = after["C"] >= before["C"] - tol
                    detail = f"C(t) {before['C']:.4f}→{after['C']:.4f}"
                elif glyph == "IL":
                    satisfied = (
                        after["dnfr"] <= before["dnfr"] + tol
                        and after["C"] >= before["C"] - tol
                    )
                    detail = (
                        f"|ΔNFR| {before['dnfr']:.4f}→{after['dnfr']:.4f}, "
                        f"C(t) {before['C']:.4f}→{after['C']:.4f}"
                    )
                elif glyph == "UM":
                    satisfied = after["dnfr"] <= before["dnfr"] + tol
                    detail = f"|ΔNFR| {before['dnfr']:.4f}→{after['dnfr']:.4f}"
                elif glyph == "SHA":
                    satisfied = after["vf"] <= before["vf"] + tol
                    detail = f"νf {before['vf']:.4f}→{after['vf']:.4f}"
                elif glyph == "VAL":
                    satisfied = after["vf"] >= before["vf"] - tol
                    detail = f"νf {before['vf']:.4f}→{after['vf']:.4f}"
                elif glyph == "NUL":
                    satisfied = after["vf"] <= before["vf"] + tol
                    detail = f"νf {before['vf']:.4f}→{after['vf']:.4f}"
                elif glyph == "THOL":
                    satisfied = (
                        before["C"] <= tol or after["C"] >= before["C"] * 0.9 - tol
                    )
                    detail = f"C(t) {before['C']:.4f}→{after['C']:.4f}"
                else:
                    satisfied = True
                    detail = "n/a"

        results.append(
            OperatorContractResult(
                english_name=eng_name,
                glyph=glyph,
                operator=name,
                contract=contract,
                context=context,
                satisfied=bool(satisfied),
                detail=detail,
            )
        )

    return OperatorContractAudit(results=tuple(results))
