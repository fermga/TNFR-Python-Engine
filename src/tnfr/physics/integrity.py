"""Structural integrity monitor for operator contracts and finite-step alerts.

Bridges passive telemetry (:mod:`conservation`) and active operator execution.
The monitor hooks into the operator pipeline so that every structural
transformation can be checked against its operator postcondition while the
finite structural balance and candidate energy are recorded as diagnostics.

PHYSICS
=======
After each operator application the monitor evaluates:

1. **Balance quality** — ``|Δρ/Δt + div J|`` via
   :func:`verify_conservation_balance`
2. **Candidate-energy change** — sampled ``dE/dt`` via
   :func:`compute_lyapunov_derivative`
3. **Balance alerts** — legacy-scaled residual alerts via
   :func:`detect_grammar_violations_from_conservation`
4. **Structural-charge drift** — sampled ``|ΔQ|`` via
   :func:`compute_noether_charge`

These quantities do not validate U1--U6 and a positive candidate-energy step
does not establish universal Lyapunov instability.  Grammar validity belongs
to the history/state-aware grammar validators.  In ``ENFORCE`` mode this class
enforces its configured *monitor alert policy* and operator postconditions;
the resulting exception is not a grammar verdict.

OPERATOR POSTCONDITIONS
=======================
Each canonical operator has a contract (AGENTS.md §Operators):

    IL  → C(t) must not decrease  (monotonicity)
    OZ  → |ΔNFR| must increase   (destabilisation)
    UM  → |wrap(φ_i − φ_j)| ≤ Δφ_max  (phase compatibility)
    RA  → effective coupling must increase (propagation)
    SHA → EPI unchanged           (silence)
    EN  → immediate operator-local C(t) is unchanged  (reception)
    AL  → EPI nondecrease; νf, ΔNFR and phase unchanged
    ...

The `POSTCONDITIONS` registry maps each lowercase public executable identifier
to a callable
``(G, node, state_before, state_after) → None | raise``.

INTEGRATION POINTS
==================
* ``definitions_base.py``  — ``Operator.__call__`` invokes the monitor
* ``self_optimizing_engine.py`` — reads ``integrity_report`` for feedback
* conservation helpers — re-used; no duplicate field computation
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..alias import get_attr
from ..constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..constants.canonical import (
    U6_STRUCTURAL_POTENTIAL_LIMIT,
    ZHIR_THRESHOLD_XI_CANONICAL,
)
from ..types import TNFRGraph
from ..utils import angle_diff

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
    """Raised when an operator fails a configured monitor policy.

    Attributes
    ----------
    operator : str
        Name of the operator associated with the alert or contract failure.
    violation_type : str
        Category: ``postcondition``, ``balance_alert``,
        ``candidate_energy_alert`` or ``charge_drift_alert``.
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
            f"{operator} triggered {violation_type}: "
            f"{details.get('reason', 'see details')}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Enums and data classes
# ═══════════════════════════════════════════════════════════════════════════


class MonitorMode(Enum):
    """Enforcement level for the integrity monitor."""

    OFF = "off"  # No monitoring (backward compatible)
    OBSERVE = "observe"  # Record postconditions and alerts, never raise
    ENFORCE = "enforce"  # Raise StructuralIntegrityViolation


@dataclass
class IntegrityReport:
    """Operator postcondition result plus finite-step diagnostic alerts.

    Produced after every monitored operator application.  Consumed by the
    self-optimization engine for closed-loop feedback.
    """

    operator: str
    node: Any
    conservation_quality: float = 1.0
    energy_derivative: float = 0.0
    is_lyapunov_stable: bool = True
    noether_charge_drift: float = 0.0
    balance_alerts: list[str] = field(default_factory=list)
    balance_sample_available: bool = False
    balance_within_alert: bool = True
    residual_alerts_within_policy: bool = True
    candidate_energy_within_alert: bool = True
    charge_drift_within_alert: bool = True
    grammar_validated: bool = False
    grammar_violations: list[str] = field(default_factory=list)
    postcondition_evaluated: bool = False
    postcondition_ok: bool = True
    postcondition_detail: str = ""
    corrective_suggestion: str = ""

    @property
    def balance_quality(self) -> float | None:
        """Sampled balance quality, or ``None`` when capture failed."""
        return self.conservation_quality if self.balance_sample_available else None

    @property
    def candidate_energy_derivative(self) -> float | None:
        """Sampled candidate-energy change, or ``None`` without an interval."""
        return self.energy_derivative if self.balance_sample_available else None

    @property
    def candidate_energy_nonincreasing(self) -> bool | None:
        """Finite-step trend, or ``None`` when no interval was sampled."""
        if not self.balance_sample_available:
            return None
        if not math.isfinite(self.energy_derivative):
            return None
        return self.energy_derivative <= 0.0

    @property
    def candidate_energy_within_numerical_tolerance(self) -> bool | None:
        """Historical Lyapunov classification, including its tolerance."""
        return self.is_lyapunov_stable if self.balance_sample_available else None

    @property
    def candidate_energy_alert(self) -> bool:
        """Whether sampled candidate-energy growth exceeds monitor tolerance."""
        return not self.candidate_energy_within_alert

    @property
    def structural_charge_drift(self) -> float | None:
        """Sampled charge drift, or ``None`` when capture failed."""
        return self.noether_charge_drift if self.balance_sample_available else None

    @property
    def structural_charge_drift_alert(self) -> bool:
        """Whether charge drift exceeds the configured monitor threshold."""
        return not self.charge_drift_within_alert

    @property
    def diagnostic_follow_up(self) -> str:
        """Accurately scoped view of the legacy corrective suggestion."""
        return self.corrective_suggestion

    @property
    def within_monitor_policy(self) -> bool:
        """Whether all configured alerts and the operator contract pass.

        This aggregate does not validate grammar and is not a theorem about
        conservation or asymptotic stability.
        """
        return (
            self.balance_within_alert
            and self.residual_alerts_within_policy
            and self.candidate_energy_within_alert
            and self.charge_drift_within_alert
            and self.postcondition_ok
        )

    @property
    def is_healthy(self) -> bool:
        """Backward-compatible alias for :attr:`within_monitor_policy`."""
        return self.within_monitor_policy


@dataclass
class IntegritySummary:
    """Aggregate monitor-policy results over operator applications."""

    reports: list[IntegrityReport] = field(default_factory=list)
    total_operators: int = 0
    balance_samples: int = 0
    violations_count: int = 0
    mean_conservation_quality: float = 1.0
    mean_energy_derivative: float = 0.0
    total_charge_drift: float = 0.0

    @property
    def alerts_count(self) -> int:
        """Accurately named alias for the legacy ``violations_count`` field."""
        return self.violations_count

    @property
    def mean_balance_quality(self) -> float:
        """Accurately named view of the running balance-quality mean."""
        return self.mean_conservation_quality

    @property
    def mean_candidate_energy_derivative(self) -> float:
        """Running mean of sampled candidate-energy derivatives."""
        return self.mean_energy_derivative

    @property
    def total_structural_charge_drift(self) -> float:
        """Accurately named view of accumulated absolute charge drift."""
        return self.total_charge_drift

    @property
    def mean_structural_charge_drift(self) -> float:
        """Mean absolute structural-charge drift per sampled interval."""
        return self.total_charge_drift / max(self.balance_samples, 1)

    def append(self, report: IntegrityReport) -> None:
        self.reports.append(report)
        self.total_operators += 1
        if not report.is_healthy:
            self.violations_count += 1
        if report.balance_sample_available:
            self.balance_samples += 1
            n = self.balance_samples
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
    """IL: bound snapshots must show nondecreasing C(t) and |ΔNFR| contraction.

    The monitor interval owns both dictionaries. Graph telemetry is deliberately
    ignored because its latest record may belong to another target or stage.
    """

    c_before = before.get("coherence", 0.0)
    c_after = after.get("coherence", 0.0)
    if c_after < c_before - 1e-9:
        return (
            f"Coherence decreased: {c_before:.6f} → {c_after:.6f} "
            f"(Δ={c_after - c_before:.6f})"
        )

    magnitude_before = abs(before.get("dnfr", 0.0))
    magnitude_after = abs(after.get("dnfr", 0.0))
    if magnitude_after > magnitude_before + 1e-6:
        return (
            "|ΔNFR| increased during Coherence: "
            f"{magnitude_before:.6f} → {magnitude_after:.6f}"
        )
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
    """EN: pressure, change rate and C(t) stay fixed before refresh."""
    c_before = before.get("coherence", 0.0)
    c_after = after.get("coherence", 0.0)
    if abs(c_after - c_before) > 1e-9:
        return (
            "Immediate operator-local coherence changed during Reception: "
            f"{c_before:.6f} → {c_after:.6f}"
        )
    for key, label in (("dnfr", "ΔNFR"), ("depi", "dEPI")):
        value_before = before.get(key, 0.0)
        value_after = after.get(key, 0.0)
        if abs(value_after - value_before) > 1e-9:
            return (
                f"{label} changed during Reception: "
                f"{value_before:.6f} → {value_after:.6f}"
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
    """AL: EPI cannot decrease; capacity, pressure and phase stay fixed."""
    e_before = before.get("epi", 0.0)
    e_after = after.get("epi", 0.0)
    if e_after < e_before - 1e-6:
        return f"EPI decreased during Emission: " f"{e_before:.6f} → {e_after:.6f}"

    vf_before = before.get("vf", 0.0)
    vf_after = after.get("vf", 0.0)
    if abs(vf_after - vf_before) > 1e-9:
        return f"νf changed during Emission: {vf_before:.6f} → {vf_after:.6f}"

    dnfr_before = before.get("dnfr", 0.0)
    dnfr_after = after.get("dnfr", 0.0)
    if abs(dnfr_after - dnfr_before) > 1e-9:
        return (
            "ΔNFR changed during Emission: "
            f"{dnfr_before:.6f} → {dnfr_after:.6f}"
        )

    theta_before = before.get("theta", 0.0)
    theta_after = after.get("theta", 0.0)
    if abs(angle_diff(theta_after, theta_before)) > 1e-9:
        return (
            "Phase changed during Emission: "
            f"{theta_before:.6f} → {theta_after:.6f}"
        )
    return None


def _postcond_expansion(
    G: TNFRGraph,
    node: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str | None:
    """VAL: νf (reorganization capacity) must not decrease.

    Canonical ground truth (``_make_scale_op``): Expansion scales νf up
    (νf *= 1.05), adding reorganization capacity — it acts on the νf
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
    (νf *= 0.9) and densifies ΔNFR — it removes capacity on the νf
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
        verify_identity_preserved(G, node, before.get("epi_kind"))
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
    theta_changed = (
        abs(angle_diff(after.get("theta", 0.0), before.get("theta", 0.0))) > 1e-9
    )
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
    "balance_residual_above_legacy_alert": (
        "Inspect the sampled balance, source terms, topology, and time step"
    ),
    "balance_rms_above_legacy_alert": (
        "Inspect the sampled balance, source terms, topology, and time step"
    ),
    "balance_peak_above_legacy_alert": (
        "Inspect nodes with large residuals before choosing an operator"
    ),
    "charge_drift_above_legacy_pi_alert": (
        "Inspect the structural-charge definition and trajectory; validate U6 "
        "from Phi_s reference drift separately"
    ),
    "balance_quality_below_monitor_threshold": (
        "Review the finite-step balance; this alert does not identify a grammar rule"
    ),
    "candidate_energy_increase_above_monitor_tolerance": (
        "Review the sampled candidate-energy change and operator postcondition"
    ),
    "structural_charge_drift_above_monitor_threshold": (
        "Review the sampled structural-charge drift and balance source"
    ),
}


def _suggest_correction(report: IntegrityReport) -> str:
    """Describe follow-up checks for measured alerts without inferring grammar."""
    suggestions: list[str] = []
    for alert_type in report.balance_alerts:
        suggestion = _CORRECTIVE_MAP.get(alert_type)
        if suggestion is not None and suggestion not in suggestions:
            suggestions.append(suggestion)
    if not report.candidate_energy_within_alert:
        suggestions.append(
            _CORRECTIVE_MAP["candidate_energy_increase_above_monitor_tolerance"]
        )
    if not report.charge_drift_within_alert:
        suggestions.append(
            _CORRECTIVE_MAP["structural_charge_drift_above_monitor_threshold"]
        )
    if not report.postcondition_ok:
        suggestions.append("Review the operator-specific postcondition failure")
    return "; ".join(dict.fromkeys(suggestions)) if suggestions else ""


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
        "depi": float(get_attr(G.nodes[node], ALIAS_DEPI, 0.0)),
        "theta": float(get_attr(G.nodes[node], ALIAS_THETA, 0.0)),
        "epi_kind": get_attr(
            G.nodes[node],
            ALIAS_EPI_KIND,
            None,
            strict=True,
            conv=lambda value: None if value is None else str(value),
        ),
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
    """Real-time operator-contract checks and finite-step alert monitoring.

    Attaches to a TNFR graph via ``G.graph["integrity_monitor"]`` and is
    consulted by ``Operator.__call__`` after every structural transformation.

    Parameters
    ----------
    mode : MonitorMode
        OFF — no overhead, backward compatible.
        OBSERVE — record postconditions and alerts, never raise.
        ENFORCE — raise ``StructuralIntegrityViolation`` when the configured
        monitor policy or operator postcondition fails.
    conservation_threshold : float
        Minimum ``conservation_quality`` (default 0.5; 1 = perfect).
    lyapunov_tolerance : float
        Maximum sampled candidate-energy increase before an alert (default
        0.1). This is a monitor tolerance, not a stability theorem.
    charge_drift_threshold : float
        Selected alert level for |ΔQ| per step (default π/2). This legacy
        charge diagnostic is separate from the U6 mean |ΔΦ_s| drift policy.
    """

    def __init__(
        self,
        mode: MonitorMode = MonitorMode.OBSERVE,
        conservation_threshold: float = 0.5,
        lyapunov_tolerance: float = 0.1,
        charge_drift_threshold: float = U6_STRUCTURAL_POTENTIAL_LIMIT,
    ) -> None:
        if not math.isfinite(conservation_threshold) or not (
            0.0 <= conservation_threshold <= 1.0
        ):
            raise ValueError("conservation_threshold must be finite and in [0, 1]")
        if not math.isfinite(lyapunov_tolerance) or lyapunov_tolerance < 0.0:
            raise ValueError("lyapunov_tolerance must be finite and non-negative")
        if not math.isfinite(charge_drift_threshold) or charge_drift_threshold < 0.0:
            raise ValueError(
                "charge_drift_threshold must be finite and non-negative"
            )
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
        self.discard_pending_operator()

    def discard_pending_operator(self) -> None:
        """Discard an unfinished before/after interval without a report.

        Operator dispatch calls this when the structural transformation
        raises after :meth:`before_operator`. The next report can therefore
        never compare against a stale pre-failure snapshot.
        """

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
        """Evaluate an operator postcondition and finite-step diagnostics.

        Called by ``Operator.__call__`` when a monitor is active.

        Returns
        -------
        IntegrityReport
            Detailed diagnostics.  Stored in ``self.summary``.

        Raises
        ------
        StructuralIntegrityViolation
            Only in ``MonitorMode.ENFORCE`` when the configured alert policy
            or operator postcondition fails.
        """
        if self.mode is MonitorMode.OFF:
            return IntegrityReport(operator=operator_name, node=node)

        _ensure_imports()
        report = IntegrityReport(operator=operator_name, node=node)

        # 1. Finite-step structural-balance quality and alerts.
        if self._snapshot_before is not None:
            try:
                snap_after = _capture_conservation_snapshot(G)
                balance = _verify_conservation_balance(
                    self._snapshot_before, snap_after
                )
                report.balance_sample_available = True
                report.conservation_quality = balance.conservation_quality
                report.balance_within_alert = (
                    balance.conservation_quality >= self.conservation_threshold
                )

                # 2. Candidate-energy change on this observed step.
                lyap = _compute_lyapunov_derivative(self._snapshot_before, snap_after)
                report.energy_derivative = lyap.energy_derivative
                report.is_lyapunov_stable = lyap.is_stable
                report.candidate_energy_within_alert = (
                    lyap.energy_derivative <= self.lyapunov_tolerance
                )

                # 3. Residual alerts. The legacy helper explicitly performs
                # no grammar validation and therefore cannot populate
                # ``grammar_violations``.
                alert_result = _detect_grammar_violations(balance)
                report.balance_alerts = list(alert_result.get("alert_types", []))
                report.residual_alerts_within_policy = not bool(
                    alert_result.get("alerts_detected", False)
                )
                if not report.balance_within_alert:
                    report.balance_alerts.append(
                        "balance_quality_below_monitor_threshold"
                    )

                # 4. Structural-charge drift on this observed step.
                charge_after = _compute_noether_charge(G)
                report.noether_charge_drift = abs(charge_after - self._charge_before)
                report.charge_drift_within_alert = (
                    report.noether_charge_drift <= self.charge_drift_threshold
                )
                report.balance_alerts = list(dict.fromkeys(report.balance_alerts))
            except Exception as exc:
                warnings.warn(
                    f"Integrity monitor conservation check failed: {exc}",
                    stacklevel=2,
                )

        # 5. Operator postcondition
        node_state_after = _capture_node_state(G, node)
        postcond_fn = POSTCONDITIONS.get(operator_name.lower())
        if postcond_fn is not None and self._node_state_before:
            try:
                report.postcondition_evaluated = True
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

        # Captures are single-use. A direct ``after_operator`` call without a
        # new ``before_operator`` must not reuse an earlier operator interval.
        self._snapshot_before = None
        self._node_state_before = {}
        self._charge_before = 0.0

        # Enforce configured monitor policy. This never raises a grammar
        # violation from residual, charge, or energy data.
        if self.mode is MonitorMode.ENFORCE and not report.is_healthy:
            if not report.postcondition_ok:
                violation_type = "postcondition"
                reason = report.postcondition_detail
            elif not report.residual_alerts_within_policy:
                violation_type = "balance_alert"
                reason = "; ".join(report.balance_alerts)
            elif not report.balance_within_alert:
                violation_type = "balance_alert"
                reason = (
                    f"balance quality {report.conservation_quality:.4f} below "
                    f"monitor threshold {self.conservation_threshold:.4f}"
                )
            elif not report.candidate_energy_within_alert:
                violation_type = "candidate_energy_alert"
                reason = (
                    f"candidate dE/dt={report.energy_derivative:.6f} above "
                    f"monitor tolerance {self.lyapunov_tolerance:.6f}"
                )
            elif not report.charge_drift_within_alert:
                violation_type = "charge_drift_alert"
                reason = (
                    f"structural-charge drift={report.noether_charge_drift:.6f} "
                    f"above monitor threshold {self.charge_drift_threshold:.6f}"
                )

            raise StructuralIntegrityViolation(
                operator=operator_name,
                violation_type=violation_type,
                details={
                    "reason": reason or "operator postcondition failed",
                    "conservation_quality": report.conservation_quality,
                    "balance_quality": report.balance_quality,
                    "energy_derivative": report.energy_derivative,
                    "candidate_energy_derivative": (
                        report.candidate_energy_derivative
                    ),
                    "candidate_energy_nonincreasing": (
                        report.candidate_energy_nonincreasing
                    ),
                    "candidate_energy_within_numerical_tolerance": (
                        report.candidate_energy_within_numerical_tolerance
                    ),
                    "charge_drift": report.noether_charge_drift,
                    "structural_charge_drift": report.structural_charge_drift,
                    "balance_alerts": report.balance_alerts,
                    "grammar_validated": report.grammar_validated,
                    "grammar_violations": report.grammar_violations,
                    "diagnostic_follow_up": report.diagnostic_follow_up,
                    # Backward-compatible key.
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
        """Return finite-diagnostic scalars for the optimization engine.

        Keys
        ----
        Accurate keys are ``balance_sample_count``, ``balance_quality``,
        ``candidate_energy_derivative``, mean/total structural-charge drift,
        and ``monitor_alert_rate``. Historical keys remain as numeric aliases.
        None of these fields reports grammar validity.
        """
        s = self._summary
        n = max(s.total_operators, 1)
        alert_rate = s.violations_count / n
        result = {
            "balance_sample_count": float(s.balance_samples),
            "balance_quality": s.mean_balance_quality,
            "candidate_energy_derivative": s.mean_candidate_energy_derivative,
            "mean_structural_charge_drift": s.mean_structural_charge_drift,
            "total_structural_charge_drift": s.total_structural_charge_drift,
            # Generic canonical view uses the per-sample mean so it does not
            # grow solely because monitoring ran for more intervals.
            "structural_charge_drift": s.mean_structural_charge_drift,
            "monitor_alert_rate": alert_rate,
        }
        result.update(
            {
                # Backward-compatible aliases. The names do not change the
                # finite-diagnostic scope documented above.
                "conservation_quality": result["balance_quality"],
                "energy_derivative": result["candidate_energy_derivative"],
                # Historical behavior accumulated absolute drift.
                "charge_drift": result["total_structural_charge_drift"],
                "violation_rate": result["monitor_alert_rate"],
            }
        )
        return result


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
# canonically manifests — network level for IL, UM and THOL, and for EN's
# immediate aggregate of unchanged node-local C(t) values; single-node level
# for the local destabiliser OZ; identity (EPI-sign) preservation for RA; and
# the phase channel for ZHIR (with its U4b precondition: prior IL + recent
# destabiliser).  The emergent field ΔNFR is normally recomputed after
# application; EN is deliberately measured before that later boundary.


@dataclass(frozen=True)
class OperatorContractResult:
    """Measured fidelity of one canonical operator to its contract.

    Attributes
    ----------
    english_name : str
        Title-case public display/class name (Emission, Reception, ...).
    glyph : str
        Internal symbolic glyph code (AL, EN, IL, ...).
    operator : str
        Canonical lowercase public executable identifier (emission, reception, ...).
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
        ok = "ALL PROBES PASS" if self.all_satisfied else "PROBE FAILURES"
        lines = [
            f"Operator postcondition probes [{ok}]: "
            f"{self.n_satisfied}/{self.n_operators} catalog probes pass "
            f"on the declared deterministic fixtures."
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
        # Phases within a π/4 band so every neighbour pair satisfies the U3
        # gate (|wrap(Δφ)| ≤ Δφ_max = π/2): coupling/resonance are admissible on
        # a phase-coherent network (Invariant #2).
        G.nodes[nd][ALIAS_THETA[0]] = rng.uniform(0.0, math.pi / 4)
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
    r"""Measure each catalog entry on a deterministic postcondition fixture.

    Applies all 13 canonical operators, each in one declared test context,
    and measures whether the corresponding finite probe passes.
    Network readouts normally recompute the emergent ΔNFR field after
    application. The direct OZ pressure postcondition and the immediate EN
    coherence postcondition are measured before recomputation, which would
    overwrite or move the channel under test. Every probe also checks the recorded
    glyph: a grammar fallback cannot count as evidence for the requested
    operator. Passing this finite suite is regression evidence, not a proof
    over all graph states, parameters or operator compositions.
    Returns an :class:`OperatorContractAudit` with a per-operator result.

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
    from ..operators.grammar_types import glyph_function_name

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
        execution_errors = []

        def apply_probe(node):
            op(G, node)
            history = G.nodes[node].get("glyph_history", ())
            actual = history[-1] if history else None
            if glyph_function_name(actual) != name:
                execution_errors.append(f"node {node}: requested {glyph}, executed {actual}")

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            if glyph == "THOL":
                # A transformer needs actual destabilization before its probe.
                for nd in list(G.nodes()):
                    Coherence()(G, nd)
                    Dissonance()(G, nd)
                default_compute_delta_nfr(G)
            if context == "node":
                # local destabiliser: single node, measure that node's |ΔNFR|
                node = list(G.nodes())[0]
                Coherence()(G, node)  # U4a handler in the execution context
                d_before = abs(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
                apply_probe(node)
                d_after = abs(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
                satisfied = d_after >= d_before - tol
                detail = f"direct node |ΔNFR| {d_before:.4f}→{d_after:.4f}"

            elif context == "identity":
                signs_before = {
                    n: float(np.sign(get_attr(G.nodes[n], ALIAS_EPI, 0.0)))
                    for n in G.nodes()
                }
                for nd in list(G.nodes()):
                    apply_probe(nd)
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
                if glyph == "ZHIR":
                    # The deterministic ZHIR probe must carry evidence for its
                    # non-disableable signed-growth trigger. Anchor the newest
                    # sample to the live EPI so the fixture remains coherent.
                    step = ZHIR_THRESHOLD_XI_CANONICAL + 1.0
                    for nd in list(G.nodes()):
                        current = float(get_attr(G.nodes[nd], ALIAS_EPI, 0.0))
                        G.nodes[nd]["epi_history"] = [current - step, current]
                theta_before = {
                    n: get_attr(G.nodes[n], ALIAS_THETA, 0.0) for n in G.nodes()
                }
                for nd in list(G.nodes()):
                    apply_probe(nd)
                changed = sum(
                    1
                    for n in G.nodes()
                    if abs(
                        angle_diff(
                            get_attr(G.nodes[n], ALIAS_THETA, 0.0), theta_before[n]
                        )
                    )
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
                    apply_probe(nd)
                default_compute_delta_nfr(G)
                after = _audit_metrics(G)
                theta_changed = any(
                    abs(
                        angle_diff(
                            get_attr(G.nodes[n], ALIAS_THETA, 0.0), theta_before[n]
                        )
                    )
                    > 1e-9
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
                    apply_probe(nd)
                default_compute_delta_nfr(G)
                satisfied = True
                detail = "advisory (network echo)"

            else:  # network
                before = _audit_metrics(G)
                for nd in list(G.nodes()):
                    apply_probe(nd)
                # EN's direct jump leaves stored ΔNFR and dEPI unchanged, so its
                # canonical C(t) postcondition belongs to this immediate boundary.
                # A refresh observes a later pressure-realisation boundary.
                if glyph != "EN":
                    default_compute_delta_nfr(G)
                after = _audit_metrics(G)
                if glyph == "AL":
                    satisfied = after["epi"] >= before["epi"] - tol
                    detail = f"|EPI| {before['epi']:.4f}→{after['epi']:.4f}"
                elif glyph == "EN":
                    satisfied = abs(after["C"] - before["C"]) <= tol
                    detail = (
                        "immediate operator-local C(t) "
                        f"{before['C']:.4f}→{after['C']:.4f}"
                    )
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

        if execution_errors:
            satisfied = False
            detail = "; ".join(execution_errors[:3])

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
