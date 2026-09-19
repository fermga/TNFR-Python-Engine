"""Nodal rate diagnostics and history-derived acceleration observations.

The nodal equation supplies the instantaneous prediction:

    ∂EPI/∂t = νf · ΔNFR(t)

``validate_nodal_equation`` is an optional declared held-step comparison using
post-state capacity and pressure. It does not authenticate an interval's inputs
or apply as a mandatory flow check to every instantaneous operator event.
History observations and cached integrator telemetry have separate semantics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import set_attr
from ..constants.aliases import ALIAS_D2EPI, ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..errors import TNFRValueError
from ..types import BEPIProtocol, scalarize_epi

__all__ = [
    "NodalEquationViolation",
    "validate_nodal_equation",
    "compute_expected_depi_dt",
    "compute_d2epi_dt2",
    "StructuralAccelerationObservation",
    "observe_structural_acceleration",
]

# Default tolerance for nodal equation validation
DEFAULT_NODAL_EQUATION_TOLERANCE = 1e-3
DEFAULT_NODAL_EQUATION_CLIP_AWARE = True


class NodalEquationViolation(Exception):
    """Raised when the optional declared held-step comparison fails.

    The retained exception name does not classify arbitrary instantaneous
    operator transformations or certify the pressure used during an interval.
    """

    def __init__(
        self,
        operator: str,
        measured_depi_dt: float,
        expected_depi_dt: float,
        tolerance: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize nodal equation violation.

        Parameters
        ----------
        operator : str
            Name of the operator that caused the violation
        measured_depi_dt : float
            Measured ∂EPI/∂t from before/after states
        expected_depi_dt : float
            Expected ∂EPI/∂t from νf · ΔNFR(t)
        tolerance : float
            Tolerance threshold that was exceeded
        details : dict, optional
            Additional diagnostic information
        """
        self.operator = operator
        self.measured_depi_dt = measured_depi_dt
        self.expected_depi_dt = expected_depi_dt
        self.tolerance = tolerance
        self.details = details or {}

        error = abs(measured_depi_dt - expected_depi_dt)
        super().__init__(
            f"Nodal equation violation in {operator}: "
            f"|∂EPI/∂t_measured - νf·ΔNFR| = {error:.3e} > {tolerance:.3e}\n"
            f"  Measured: {measured_depi_dt:.6f}\n"
            f"  Expected: {expected_depi_dt:.6f}"
        )


from .metrics_core import get_node_attr as _get_node_attr


def compute_expected_depi_dt(G: TNFRGraph, node: NodeId) -> float:
    """Compute expected ∂EPI/∂t from current νf and ΔNFR values.

    Implements the canonical TNFR nodal equation:
        ∂EPI/∂t = νf · ΔNFR(t)

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to compute expected rate for

    Returns
    -------
    float
        Expected rate of EPI change (∂EPI/∂t)

    Notes
    -----
    The structural frequency (νf) is in Hz_str (structural hertz) units,
    and ΔNFR is the dimensionless internal reorganization operator.
    Their product gives the rate of structural reorganization.
    """
    vf = _get_node_attr(G, node, ALIAS_VF)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    return vf * dnfr


def validate_nodal_equation(
    G: TNFRGraph,
    node: NodeId,
    epi_before: float,
    epi_after: float,
    dt: float,
    *,
    operator_name: str = "unknown",
    tolerance: float | None = None,
    strict: bool = False,
    clip_aware: bool | None = None,
) -> bool:
    """Validate that EPI change respects the nodal equation.

    Verifies that the change in EPI between before and after states
    matches the prediction from the nodal equation:

        ∂EPI/∂t = νf · ΔNFR(t)

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node that underwent transformation
    epi_before : float
        EPI value before operator application
    epi_after : float
        EPI value after operator application
    dt : float
        Time step (typically 1.0 for discrete operator applications)
    operator_name : str, optional
        Name of the operator for error reporting
    tolerance : float, optional
        Absolute tolerance for equation validation.
        If None, uses graph configuration or default (1e-3).
    strict : bool, default False
        If True, raises NodalEquationViolation on failure.
        If False, returns validation result without raising.
    clip_aware : bool, optional
        If True, validates using structural_clip to account for boundary
        preservation: EPI_expected = structural_clip(EPI_theoretical).
        If False, uses classic mode without clip adjustment.
        If None, uses graph configuration or default (True).

    Returns
    -------
    bool
        True if equation is satisfied within tolerance, False otherwise

    Raises
    ------
    NodalEquationViolation
        If strict=True and validation fails

    Notes
    -----
    The nodal equation is validated using the post-transformation νf and ΔNFR
    values, as these represent the structural state after the operator effect.

    For discrete operator applications, dt is typically 1.0, making the
    validation equivalent to: (epi_after - epi_before) ≈ νf_after · ΔNFR_after

    **Clip-aware mode** (default): When structural_clip intervenes to preserve
    boundaries, the actual EPI differs from the theoretical prediction. This
    mode accounts for boundary preservation by applying structural_clip to the
    theoretical value before comparison:

        EPI_expected = structural_clip(EPI_before + νf · ΔNFR · dt)

    This ensures validation passes when clip interventions are legitimate parts
    of the operator's structural boundary preservation.

    **Classic mode** (clip_aware=False): Validates without clip adjustment,
    useful for detecting when unexpected clipping occurs.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0, dnfr=0.1)
    >>> epi_before = G.nodes[node]["EPI"]
    >>> # Apply some transformation...
    >>> epi_after = G.nodes[node]["EPI"]
    >>> is_valid = validate_nodal_equation(G, node, epi_before, epi_after, dt=1.0)
    """
    if tolerance is None:
        # Try graph configuration first, then use default constant
        tolerance = float(
            G.graph.get("NODAL_EQUATION_TOLERANCE", DEFAULT_NODAL_EQUATION_TOLERANCE)
        )

    if clip_aware is None:
        # Try graph configuration first, then use default
        clip_aware = G.graph.get(
            "NODAL_EQUATION_CLIP_AWARE", DEFAULT_NODAL_EQUATION_CLIP_AWARE
        )

    # Measured rate of EPI change
    measured_depi_dt = (epi_after - epi_before) / dt if dt > 0 else 0.0

    # Expected rate from nodal equation: νf · ΔNFR
    # Use post-transformation values as they represent the new structural state
    expected_depi_dt = compute_expected_depi_dt(G, node)

    if clip_aware:
        # Clip-aware mode: apply structural_clip to theoretical EPI before comparison
        from ..dynamics.structural_clip import structural_clip

        # Get structural boundaries from graph configuration
        epi_min = float(G.graph.get("EPI_MIN", -1.0))
        epi_max = float(G.graph.get("EPI_MAX", 1.0))
        clip_mode = G.graph.get("CLIP_MODE", "hard")

        # Compute theoretical EPI based on nodal equation
        epi_theoretical = epi_before + (expected_depi_dt * dt)

        # Validate and normalize clip_mode
        clip_mode_str = str(clip_mode).lower()
        if clip_mode_str not in ("hard", "soft"):
            clip_mode_str = "hard"  # Default to safe fallback

        # Apply structural_clip to get expected EPI (what the operator should produce)
        epi_expected = structural_clip(
            epi_theoretical, lo=epi_min, hi=epi_max, mode=clip_mode_str  # type: ignore[arg-type]
        )

        # Validate against clipped expected value
        error = abs(epi_after - epi_expected)
        is_valid = error <= tolerance

        if not is_valid and strict:
            vf = _get_node_attr(G, node, ALIAS_VF)
            dnfr = _get_node_attr(G, node, ALIAS_DNFR)

            raise NodalEquationViolation(
                operator=operator_name,
                measured_depi_dt=measured_depi_dt,
                expected_depi_dt=expected_depi_dt,
                tolerance=tolerance,
                details={
                    "epi_before": epi_before,
                    "epi_after": epi_after,
                    "epi_theoretical": epi_theoretical,
                    "epi_expected": epi_expected,
                    "dt": dt,
                    "vf": vf,
                    "dnfr": dnfr,
                    "error": error,
                    "clip_aware": True,
                    "clip_intervened": abs(epi_theoretical - epi_expected) > 1e-10,
                },
            )
    else:
        # Classic mode: validate rate of change directly
        error = abs(measured_depi_dt - expected_depi_dt)
        is_valid = error <= tolerance

        if not is_valid and strict:
            vf = _get_node_attr(G, node, ALIAS_VF)
            dnfr = _get_node_attr(G, node, ALIAS_DNFR)

            raise NodalEquationViolation(
                operator=operator_name,
                measured_depi_dt=measured_depi_dt,
                expected_depi_dt=expected_depi_dt,
                tolerance=tolerance,
                details={
                    "epi_before": epi_before,
                    "epi_after": epi_after,
                    "dt": dt,
                    "vf": vf,
                    "dnfr": dnfr,
                    "error": error,
                    "clip_aware": False,
                },
            )

    return is_valid


@dataclass(frozen=True, slots=True)
class StructuralAccelerationObservation:
    """Detached three-sample finite difference, not causal execution evidence.

    ``value`` is absent when fewer than three active samples are available.
    ``samples`` contains only a successfully validated last-three window:
    physical ``(time, EPI)`` pairs or legacy unit-step EPI scalars. A physical
    window must end at the current EPI; legacy endpoint provenance remains
    unspecified. This observation neither authenticates the history's producer
    nor proves a derivative, bifurcation, or operator execution readiness.

    The integrator's stored RHS-rate difference and Mutation's two-point
    signed secant are separate observations with different contracts.
    """

    source: str | None
    history_length: int
    time_basis: str | None
    available: bool
    value: float | None
    samples: tuple[float | tuple[float, float], ...]
    current_endpoint_matches_state: bool | None
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return detached JSON-compatible observation fields."""
        return {
            "source": self.source,
            "history_length": self.history_length,
            "time_basis": self.time_basis,
            "available": self.available,
            "value": self.value,
            "samples": [
                list(sample) if isinstance(sample, tuple) else sample
                for sample in self.samples
            ],
            "current_endpoint_matches_state": self.current_endpoint_matches_state,
            "reason": self.reason,
        }


def compute_d2epi_dt2(G: "TNFRGraph", node: "NodeId", *, store: bool = True) -> float:
    """Compute ∂²EPI/∂t² (structural acceleration).

    Return the shared three-sample finite difference. Operator-specific gates
    may compare its magnitude with their configured threshold; that comparison
    alone proves neither a birth nor complete execution readiness.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier to compute acceleration for
    store : bool, default=True
        Whether to write the computed acceleration to the node's ``D2_EPI``
        telemetry attribute.  Set to ``False`` for a strictly read-only
        diagnostic evaluation.

    Returns
    -------
    float
        Signed finite-difference acceleration, or compatibility value 0.0
        when history is unavailable. Use ``observe_structural_acceleration``
        to distinguish an unavailable value from an observed zero.

    Notes
    -----
    **Computation method:**

    Timestamped physical histories use the unequal-step three-point estimate

        ∂²EPI/∂t² ≈ 2·(s₂-s₁)/(Δt₁+Δt₂),

    where ``s₁`` and ``s₂`` are the two adjacent secant rates.  For equal
    spacing this reduces to the familiar second-order finite difference:
        ∂²EPI/∂t² ≈ (EPI_t - 2·EPI_{t-1} + EPI_{t-2}) / Δt²

    For discrete operator applications with Δt=1:
        ∂²EPI/∂t² ≈ EPI_t - 2·EPI_{t-1} + EPI_{t-2}

    **History requirements:**

    ``epi_time_history`` is authoritative when present and must contain
    strictly increasing ``(time, EPI)`` pairs whose final EPI matches the
    current nodal state.  ``epi_history`` and ``_epi_history`` retain their
    legacy unit-operator-step interpretation.  At least three samples are
    required; otherwise the function returns 0.0 (acceleration unavailable).

    By default an available value is stored in the node's ``D2_EPI`` attribute
    (using ALIAS_D2EPI aliases) for telemetry and metrics collection.  Passing
    ``store=False`` leaves the graph unchanged. Unavailable history leaves
    existing telemetry untouched even when ``store=True``; that cached value
    is not evidence of a current history-derived acceleration.

    **Physical interpretation:**

    A validated zero is a zero represented finite difference in this window,
    not exact equality of real-valued secants or future stationarity. Rounding
    can erase a subrepresentable difference. The sign describes the change of
    the represented secants, rather than the sign of the EPI rate itself.
    No source causality is inferred.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Emission, Dissonance
    >>> from tnfr.operators.nodal_equation import compute_d2epi_dt2
    >>>
    >>> G, node = create_nfr("test", epi=0.2, vf=1.0)
    >>>
    >>> # Build EPI history through operator applications
    >>> Emission()(G, node)  # EPI increases
    >>> Emission()(G, node)  # EPI increases more
    >>> Dissonance()(G, node)  # Introduce instability
    >>>
    >>> # Compute acceleration
    >>> d2epi = compute_d2epi_dt2(G, node)
    >>>
    >>> # Check if bifurcation threshold exceeded
    >>> tau = G.graph.get("OZ_BIFURCATION_THRESHOLD", 0.5)
    >>> bifurcation_active = abs(d2epi) > tau

    See Also
    --------
    tnfr.dynamics.bifurcation.compute_bifurcation_score : Uses d2epi for scoring
    tnfr.operators.metrics.dissonance_metrics : Reports d2epi in OZ metrics
    tnfr.operators.preconditions.validate_dissonance : Checks d2epi for bifurcation
    """
    if not isinstance(store, bool):
        raise TNFRValueError("store must be a boolean.")

    observation = observe_structural_acceleration(G, node)
    if not observation.available:
        return 0.0
    if store:
        set_attr(G.nodes[node], ALIAS_D2EPI, observation.value)
    return float(observation.value)


def observe_structural_acceleration(
    G: "TNFRGraph",
    node: "NodeId",
) -> StructuralAccelerationObservation:
    """Read one active history without writing telemetry or changing the graph.

    Physical history takes precedence even when short. Otherwise a nonempty
    canonical legacy history precedes its private compatibility counterpart.
    Missing or short history returns explicit unavailability; its samples are
    not validated as a three-point window. Malformed complete windows, stale
    physical endpoints and nonfinite derived intervals/rates raise
    :class:`TNFRValueError`, without a fallback to another source.

    Physical values use ``2*(s2-s1)/(dt1+dt2)``; legacy values use unit operator
    steps. Only the selected final three samples are assessed. The detached
    result is an observation of supplied data, not a sealed runtime record.
    """
    node_data = G.nodes[node]
    source, history = _select_acceleration_history(node_data)
    if history is None:
        return StructuralAccelerationObservation(
            None,
            0,
            None,
            False,
            None,
            (),
            None,
            "missing_history",
        )
    length = _history_length_or_error(history, source)
    physical = source == "epi_time_history"
    time_basis = "physical_time" if physical else "legacy_unit_operator_step"
    if length < 3:
        return StructuralAccelerationObservation(
            source,
            length,
            time_basis,
            False,
            None,
            (),
            None,
            "insufficient_history",
        )

    try:
        samples = history[-3], history[-2], history[-1]
    except (IndexError, KeyError, TypeError) as exc:
        raise TNFRValueError(
            f"{source} must be an indexed, replayable history."
        ) from exc

    if physical:
        timed = tuple(
            _physical_acceleration_sample(value, source, index)
            for index, value in enumerate(samples, start=length - 3)
        )
        (t0, epi0), (t1, epi1), (t2, epi2) = timed
        dt1 = t1 - t0
        dt2 = t2 - t1
        if not math.isfinite(dt1) or not math.isfinite(dt2):
            raise TNFRValueError("epi_time_history intervals must remain finite.")
        if dt1 <= 0.0 or dt2 <= 0.0:
            raise TNFRValueError("epi_time_history timestamps must increase strictly.")
        span = dt1 + dt2
        if not math.isfinite(span):
            raise TNFRValueError("epi_time_history total span must remain finite.")
        current_raw = _first_present(node_data, ALIAS_EPI)
        if current_raw is _MISSING:
            raise TNFRValueError(
                "epi_time_history requires an explicit current EPI endpoint."
            )
        current_epi = _canonical_epi_scalar(current_raw, "current EPI")
        if epi2 != current_epi:
            raise TNFRValueError(
                "epi_time_history final EPI must match the current nodal state.",
                context={"history_endpoint": epi2, "current_epi": current_epi},
            )
        slope1 = (epi1 - epi0) / dt1
        slope2 = (epi2 - epi1) / dt2
        if not math.isfinite(slope1) or not math.isfinite(slope2):
            raise TNFRValueError("epi_time_history secant rates must remain finite.")
        d2epi = 2.0 * (slope2 - slope1) / span
        observed_samples = timed
    else:
        observed_samples = tuple(
            _finite_history_scalar(value, source, index)
            for index, value in enumerate(samples, start=length - 3)
        )
        epi0, epi1, epi2 = observed_samples
        d2epi = epi2 - 2.0 * epi1 + epi0

    if not math.isfinite(d2epi):
        raise TNFRValueError(f"{source} produces non-finite structural acceleration.")

    return StructuralAccelerationObservation(
        source,
        length,
        time_basis,
        True,
        float(d2epi),
        observed_samples,
        True if physical else None,
        None,
    )


_MISSING = object()


def _first_present(data: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if key in data:
            return data[key]
    return _MISSING


def _canonical_epi_scalar(value: Any, label: str) -> float:
    serialized_bepi = isinstance(value, Mapping) and {
        "continuous",
        "discrete",
        "grid",
    }.issubset(value)
    try:
        if isinstance(value, BEPIProtocol) or serialized_bepi:
            result = scalarize_epi(value)
        elif isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError
        else:
            result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(f"{label} must be a finite real scalar.") from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} must be finite.")
    return result


def _finite_real_scalar(value: Any, label: str) -> float:
    try:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(f"{label} must be a finite real scalar.") from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} must be finite.")
    return result


def _history_length_or_error(history: Any, source: str) -> int:
    try:
        return len(history)
    except (OverflowError, TypeError) as exc:
        raise TNFRValueError(f"{source} must be a sized, indexed history.") from exc


def _select_acceleration_history(
    node_data: Mapping[str, Any],
) -> tuple[str, Any | None]:
    physical = node_data.get("epi_time_history")
    if physical is not None:
        return "epi_time_history", physical

    canonical = node_data.get("epi_history")
    if canonical is not None:
        length = _history_length_or_error(canonical, "epi_history")
        if length > 0:
            return "epi_history", canonical

    legacy = node_data.get("_epi_history")
    if legacy is not None:
        return "_epi_history", legacy
    if canonical is not None:
        return "epi_history", canonical
    return "_epi_history", None


def _finite_history_scalar(value: Any, source: str, index: int) -> float:
    return _canonical_epi_scalar(value, f"{source}[{index}]")


def _physical_acceleration_sample(
    value: Any, source: str, index: int
) -> tuple[float, float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TNFRValueError(f"{source}[{index}] must be a (time, EPI) pair.")
    try:
        if len(value) != 2:
            raise TNFRValueError(f"{source}[{index}] must be a (time, EPI) pair.")
        time_raw, epi_raw = value[0], value[1]
    except (IndexError, KeyError, OverflowError, TypeError) as exc:
        raise TNFRValueError(f"{source}[{index}] must be a (time, EPI) pair.") from exc
    time = _finite_real_scalar(time_raw, f"{source}[{index}].time")
    epi = _canonical_epi_scalar(epi_raw, f"{source}[{index}].EPI")
    return time, epi
