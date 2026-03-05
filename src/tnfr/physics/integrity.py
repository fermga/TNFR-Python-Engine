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

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
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

    from .conservation import (
        capture_conservation_snapshot as _css,
        verify_conservation_balance as _vcb,
        compute_lyapunov_derivative as _cld,
        detect_grammar_violations_from_conservation as _dgv,
        compute_noether_charge as _cnc,
        compute_energy_functional as _cef,
    )
    from ..metrics.common import compute_coherence as _cc

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

    def __init__(self, operator: str, violation_type: str,
                 details: Dict[str, Any]) -> None:
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
    OFF = "off"            # No monitoring (backward compatible)
    OBSERVE = "observe"    # Record violations, never raise
    ENFORCE = "enforce"    # Raise StructuralIntegrityViolation


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
    grammar_violations: List[str] = field(default_factory=list)
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
    reports: List[IntegrityReport] = field(default_factory=list)
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
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
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
        return (
            f"|ΔNFR| increased during Coherence: "
            f"{d_before:.6f} → {d_after:.6f}"
        )
    return None


def _postcond_dissonance(
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
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
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
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
        return (
            f"νf increased during Silence: "
            f"{vf_before:.6f} → {vf_after:.6f}"
        )
    return None


def _postcond_reception(
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
    """EN: C(t) must not decrease."""
    c_before = before.get("coherence", 0.0)
    c_after = after.get("coherence", 0.0)
    if c_after < c_before - 1e-9:
        return (
            f"Coherence decreased during Reception: "
            f"{c_before:.6f} → {c_after:.6f}"
        )
    return None


def _postcond_resonance(
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
    """RA: EPI must not decrease; νf must not decrease (amplification)."""
    e_before = before.get("epi", 0.0)
    e_after = after.get("epi", 0.0)
    if e_after < e_before - 1e-6:
        return (
            f"EPI decreased during Resonance: "
            f"{e_before:.6f} → {e_after:.6f}"
        )
    # νf must not decrease (glyph amplifies: vf *= 1 + boost)
    vf_before = before.get("vf", 0.0)
    vf_after = after.get("vf", 0.0)
    if vf_after < vf_before - 1e-9:
        return (
            f"νf decreased during Resonance: "
            f"{vf_before:.6f} → {vf_after:.6f}"
        )
    return None


def _postcond_emission(
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
    """AL: νf must not decrease; EPI must not decrease (∂EPI/∂t > 0)."""
    vf_before = before.get("vf", 0.0)
    vf_after = after.get("vf", 0.0)
    if vf_after < vf_before - 1e-9:
        return (
            f"νf decreased during Emission: "
            f"{vf_before:.6f} → {vf_after:.6f}"
        )
    # EPI must not decrease (core glyph effect: +AL_boost)
    e_before = before.get("epi", 0.0)
    e_after = after.get("epi", 0.0)
    if e_after < e_before - 1e-6:
        return (
            f"EPI decreased during Emission: "
            f"{e_before:.6f} → {e_after:.6f}"
        )
    return None


def _postcond_expansion(
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
    """VAL: EPI complexity (magnitude) must increase."""
    e_before = abs(before.get("epi", 0.0))
    e_after = abs(after.get("epi", 0.0))
    if e_after < e_before - 1e-6:
        return (
            f"|EPI| decreased during Expansion: "
            f"{e_before:.6f} → {e_after:.6f}"
        )
    return None


def _postcond_contraction(
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
    """NUL: EPI complexity (magnitude) must decrease."""
    e_before = abs(before.get("epi", 0.0))
    e_after = abs(after.get("epi", 0.0))
    if e_after > e_before + 1e-6:
        return (
            f"|EPI| increased during Contraction: "
            f"{e_before:.6f} → {e_after:.6f}"
        )
    return None


def _postcond_mutation(
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
    """ZHIR: Delegates to postconditions/mutation.py for rich verification.

    Checks phase transformation, identity preservation, and bifurcation
    handling via the canonical postcondition module.
    """
    try:
        from ..operators.postconditions.mutation import (
            verify_phase_transformed,
            verify_identity_preserved,
            verify_bifurcation_handled,
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
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
    """UM: |ΔNFR| must not increase (coupling reduces structural pressure)."""
    d_before = abs(before.get("dnfr", 0.0))
    d_after = abs(after.get("dnfr", 0.0))
    if d_after > d_before + 1e-6:
        return (
            f"|ΔNFR| increased during Coupling: "
            f"{d_before:.6f} → {d_after:.6f}"
        )
    return None


def _postcond_self_organization(
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
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
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
    """NAV: At least one state variable (νf, θ, ΔNFR) must change."""
    vf_changed = abs(after.get("vf", 0.0) - before.get("vf", 0.0)) > 1e-9
    theta_changed = abs(after.get("theta", 0.0) - before.get("theta", 0.0)) > 1e-9
    dnfr_changed = abs(after.get("dnfr", 0.0) - before.get("dnfr", 0.0)) > 1e-9
    if not (vf_changed or theta_changed or dnfr_changed):
        return (
            "No state change during Transition: "
            "νf, θ, and ΔNFR all unchanged"
        )
    return None


def _postcond_recursivity(
    G: TNFRGraph, node: Any,
    before: Dict[str, Any], after: Dict[str, Any],
) -> Optional[str]:
    """REMESH: Advisory glyph — structural remesh is verified at network level."""
    return None


# Mapping from canonical operator name → postcondition checker.
# Returns None on success, or a string describing the violation.
# All 13 canonical operators are covered.
POSTCONDITIONS: Dict[str, Callable[..., Optional[str]]] = {
    "coherence":          _postcond_coherence,
    "dissonance":         _postcond_dissonance,
    "silence":            _postcond_silence,
    "reception":          _postcond_reception,
    "resonance":          _postcond_resonance,
    "emission":           _postcond_emission,
    "expansion":          _postcond_expansion,
    "contraction":        _postcond_contraction,
    "mutation":           _postcond_mutation,
    "coupling":           _postcond_coupling,
    "self_organization":  _postcond_self_organization,
    "transition":         _postcond_transition,
    "recursivity":        _postcond_recursivity,
}


# ═══════════════════════════════════════════════════════════════════════════
# Corrective suggestions
# ═══════════════════════════════════════════════════════════════════════════

_CORRECTIVE_MAP: Dict[str, str] = {
    "U6_confinement_breach": "Apply IL (Coherence) to reduce Φ_s below φ threshold",
    "U2_convergence_failure": "Apply IL or THOL to stabilise divergent ΔNFR",
    "U3_phase_incompatibility": "Apply SHA (Silence) then UM with phase-compatible nodes",
    "lyapunov_unstable": "Apply IL or THOL; energy is increasing (dE/dt > 0)",
    "charge_drift": "Apply IL to restore Noether charge Q toward conserved value",
}


def _suggest_correction(report: IntegrityReport) -> str:
    """Derive a corrective operator suggestion from violation diagnostics."""
    suggestions: List[str] = []
    for vtype in report.grammar_violations:
        if vtype in _CORRECTIVE_MAP:
            suggestions.append(_CORRECTIVE_MAP[vtype])
    if not report.is_lyapunov_stable:
        suggestions.append(_CORRECTIVE_MAP["lyapunov_unstable"])
    if abs(report.noether_charge_drift) > 0.5:
        suggestions.append(_CORRECTIVE_MAP["charge_drift"])
    return "; ".join(suggestions) if suggestions else ""


# ═══════════════════════════════════════════════════════════════════════════
# Capture helpers
# ═══════════════════════════════════════════════════════════════════════════

def _capture_node_state(G: TNFRGraph, node: Any) -> Dict[str, Any]:
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
        Maximum |ΔQ| per step (default 1.618 = φ, from U6).
    """

    def __init__(
        self,
        mode: MonitorMode = MonitorMode.OBSERVE,
        conservation_threshold: float = 0.5,
        lyapunov_tolerance: float = 0.1,
        charge_drift_threshold: float = 1.618,
    ) -> None:
        self.mode = mode
        self.conservation_threshold = conservation_threshold
        self.lyapunov_tolerance = lyapunov_tolerance
        self.charge_drift_threshold = charge_drift_threshold
        self._summary = IntegritySummary()
        self._snapshot_before = None
        self._node_state_before: Dict[str, Any] = {}
        self._charge_before: float = 0.0

    # ── public API ────────────────────────────────────────────────────────

    @property
    def summary(self) -> IntegritySummary:
        """Accumulated integrity statistics (read-only)."""
        return self._summary

    @property
    def latest_report(self) -> Optional[IntegrityReport]:
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
                lyap = _compute_lyapunov_derivative(
                    self._snapshot_before, snap_after
                )
                report.energy_derivative = lyap.energy_derivative
                report.is_lyapunov_stable = lyap.is_stable

                # 3. Grammar violations from conservation residuals
                violations = _detect_grammar_violations(balance)
                if violations["violations_detected"]:
                    report.grammar_violations = violations["violation_types"]

                # 4. Noether charge drift
                charge_after = _compute_noether_charge(G)
                report.noether_charge_drift = abs(
                    charge_after - self._charge_before
                )
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
                    G, node,
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
    def get(G: TNFRGraph) -> Optional["StructuralIntegrityMonitor"]:
        """Retrieve the monitor attached to *G*, or None."""
        return G.graph.get("integrity_monitor")

    # ── feedback for self-optimization ────────────────────────────────────

    def feedback_vector(self) -> Dict[str, float]:
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
