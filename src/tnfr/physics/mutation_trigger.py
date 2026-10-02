"""Pure evidence and prediction certificate for the TNFR Mutation trigger.

The nodal equation supplies an instantaneous model prediction,
``dEPI/dt = nu_f * DeltaNFR``.  A two-point EPI history instead supplies a
realized secant rate over its sampling interval.  This module keeps those two
quantities separate and does not assess U4b grammar context or operator
execution readiness.

Physical histories use ``epi_time_history=[(t, epi), ...]``.  Their last
timestamp must increase strictly and their final EPI sample must match the
current EPI state.  Legacy ``epi_history`` and ``_epi_history`` lists retain
their historical unit-operator-step interpretation; they are deliberately
marked as not resolved in physical time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

from ..constants.canonical import ZHIR_THRESHOLD_XI_CANONICAL

__all__ = (
    "MutationTriggerCertificate",
    "MutationTriggerEvidence",
    "MutationTriggerInputError",
    "certify_mutation_trigger",
)


_PHYSICAL_SOURCE = "epi_time_history"
_CANONICAL_LEGACY_SOURCE = "epi_history"
_LEGACY_SOURCE = "_epi_history"
_PHYSICAL_TIME_BASIS = "physical_time"
_LEGACY_TIME_BASIS = "legacy_unit_operator_step"


class MutationTriggerInputError(ValueError):
    """Raised when a scalar certificate input is outside its declared domain."""

    def __init__(self, field: str, value: Any, reason: str) -> None:
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"{field} {reason}; got {value!r}")


@dataclass(frozen=True, slots=True)
class MutationTriggerEvidence:
    """Immutable two-sample observation considered by the threshold gate.

    ``observed_depi_dt`` is a signed secant.  It is unavailable when the
    samples do not define a finite positive interval or finite rate.
    ``current_endpoint_matches_state`` is assessed only for timestamped
    physical histories.  Legacy histories intentionally retain ``None`` for
    that field because their existing contract did not require endpoint
    provenance.
    """

    previous_epi: float
    current_epi: float
    previous_time: float | None
    current_time: float | None
    sample_interval: float | None
    observed_depi_dt: float | None
    source: str
    time_basis: str
    physical_time_resolved: bool
    current_endpoint_matches_state: bool | None
    is_valid: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class MutationTriggerCertificate:
    """Separate instantaneous prediction from observed Mutation evidence.

    The Boolean ``threshold_gate_satisfied`` concerns only the strict observed
    threshold ``observed_depi_dt > xi``.  ``capacity_active`` is reported as a
    separate nodal-equation condition.  The certificate deliberately does not
    assess grammar U4b and therefore exposes no execution-readiness claim.

    ``rate_gap`` is ``observed_depi_dt - predicted_depi_dt`` only when a valid
    timestamped physical observation makes the two rates dimensionally
    comparable.  It is ``None`` for legacy unit-step evidence.
    """

    current_epi: float
    nu_f: float
    delta_nfr: float
    xi: float
    predicted_depi_dt: float
    observed_depi_dt: float | None
    observed_crossed: bool | None
    predicted_crossed: bool
    evidence_available: bool
    evidence_valid: bool
    capacity_active: bool
    threshold_gate_satisfied: bool
    source: str | None
    time_basis: str | None
    physical_time_resolved: bool
    current_endpoint_matches_state: bool | None
    reason: str | None
    rate_gap: float | None
    evidence: MutationTriggerEvidence | None


def _finite_real(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MutationTriggerInputError(field, value, "must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise MutationTriggerInputError(
            field, value, "must be representable as a finite real scalar"
        ) from exc
    if not math.isfinite(result):
        raise MutationTriggerInputError(field, value, "must be finite")
    return result


def _history_length(history: Any) -> int | None:
    try:
        return len(history)
    except (OverflowError, TypeError):
        return None


def _last_two(history: Any) -> tuple[Any, Any] | None:
    try:
        return history[-2], history[-1]
    except (IndexError, KeyError, TypeError):
        return None


def _history_scalar(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _history_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        if len(value) != 2:
            return None
        time_raw, epi_raw = value[0], value[1]
    except (IndexError, KeyError, OverflowError, TypeError):
        return None
    time = _history_scalar(time_raw)
    epi = _history_scalar(epi_raw)
    if time is None or epi is None:
        return None
    return time, epi


def _invalid_certificate(
    *,
    current_epi: float,
    nu_f: float,
    delta_nfr: float,
    xi: float,
    predicted_depi_dt: float,
    predicted_crossed: bool,
    source: str | None,
    time_basis: str | None,
    reason: str,
    evidence_available: bool,
    evidence: MutationTriggerEvidence | None = None,
) -> MutationTriggerCertificate:
    observed = evidence.observed_depi_dt if evidence is not None else None
    endpoint_matches = (
        evidence.current_endpoint_matches_state if evidence is not None else None
    )
    physical_time_resolved = (
        evidence.physical_time_resolved if evidence is not None else False
    )
    return MutationTriggerCertificate(
        current_epi=current_epi,
        nu_f=nu_f,
        delta_nfr=delta_nfr,
        xi=xi,
        predicted_depi_dt=predicted_depi_dt,
        observed_depi_dt=observed,
        observed_crossed=None,
        predicted_crossed=predicted_crossed,
        evidence_available=evidence_available,
        evidence_valid=False,
        capacity_active=nu_f > 0.0,
        threshold_gate_satisfied=False,
        source=source,
        time_basis=time_basis,
        physical_time_resolved=physical_time_resolved,
        current_endpoint_matches_state=endpoint_matches,
        reason=reason,
        rate_gap=None,
        evidence=evidence,
    )


def _select_legacy_history(
    epi_history: Any,
    legacy_epi_history: Any,
) -> tuple[Any, str] | None:
    """Apply the existing canonical-then-legacy history precedence."""

    if epi_history is not None:
        canonical_length = _history_length(epi_history)
        if canonical_length is None or canonical_length > 0:
            return epi_history, _CANONICAL_LEGACY_SOURCE
    if legacy_epi_history is not None:
        return legacy_epi_history, _LEGACY_SOURCE
    if epi_history is not None:
        return epi_history, _CANONICAL_LEGACY_SOURCE
    return None


def _physical_evidence(
    history: Any,
    *,
    current_epi: float,
    endpoint_tolerance: float,
) -> tuple[MutationTriggerEvidence | None, bool, str | None]:
    length = _history_length(history)
    if length is None:
        return None, False, "history_not_replayable"
    if length < 2:
        return None, False, "insufficient_history"
    samples = _last_two(history)
    if samples is None:
        return None, True, "history_not_replayable"
    previous = _history_pair(samples[0])
    current = _history_pair(samples[1])
    if previous is None or current is None:
        return None, True, "invalid_time_history_sample"

    previous_time, previous_epi = previous
    current_time, current_sample_epi = current
    sample_interval = current_time - previous_time
    if not math.isfinite(sample_interval) or sample_interval <= 0.0:
        evidence = MutationTriggerEvidence(
            previous_epi=previous_epi,
            current_epi=current_sample_epi,
            previous_time=previous_time,
            current_time=current_time,
            sample_interval=sample_interval,
            observed_depi_dt=None,
            source=_PHYSICAL_SOURCE,
            time_basis=_PHYSICAL_TIME_BASIS,
            physical_time_resolved=False,
            current_endpoint_matches_state=None,
            is_valid=False,
            reason="non_increasing_time",
        )
        return evidence, True, evidence.reason

    epi_delta = current_sample_epi - previous_epi
    observed_depi_dt = epi_delta / sample_interval
    if not math.isfinite(epi_delta) or not math.isfinite(observed_depi_dt):
        evidence = MutationTriggerEvidence(
            previous_epi=previous_epi,
            current_epi=current_sample_epi,
            previous_time=previous_time,
            current_time=current_time,
            sample_interval=sample_interval,
            observed_depi_dt=None,
            source=_PHYSICAL_SOURCE,
            time_basis=_PHYSICAL_TIME_BASIS,
            physical_time_resolved=True,
            current_endpoint_matches_state=None,
            is_valid=False,
            reason="non_finite_observed_rate",
        )
        return evidence, True, evidence.reason

    endpoint_gap = abs(current_sample_epi - current_epi)
    endpoint_matches = (
        math.isfinite(endpoint_gap) and endpoint_gap <= endpoint_tolerance
    )
    reason = None if endpoint_matches else "stale_physical_endpoint"
    evidence = MutationTriggerEvidence(
        previous_epi=previous_epi,
        current_epi=current_sample_epi,
        previous_time=previous_time,
        current_time=current_time,
        sample_interval=sample_interval,
        observed_depi_dt=observed_depi_dt,
        source=_PHYSICAL_SOURCE,
        time_basis=_PHYSICAL_TIME_BASIS,
        physical_time_resolved=True,
        current_endpoint_matches_state=endpoint_matches,
        is_valid=endpoint_matches,
        reason=reason,
    )
    return evidence, True, reason


def _legacy_evidence(
    history: Any,
    *,
    source: str,
) -> tuple[MutationTriggerEvidence | None, bool, str | None]:
    length = _history_length(history)
    if length is None:
        return None, False, "history_not_replayable"
    if length < 2:
        return None, False, "insufficient_history"
    samples = _last_two(history)
    if samples is None:
        return None, True, "history_not_replayable"
    previous_epi = _history_scalar(samples[0])
    current_epi = _history_scalar(samples[1])
    if previous_epi is None or current_epi is None:
        return None, True, "invalid_legacy_history_sample"

    observed_depi_dt = current_epi - previous_epi
    if not math.isfinite(observed_depi_dt):
        evidence = MutationTriggerEvidence(
            previous_epi=previous_epi,
            current_epi=current_epi,
            previous_time=None,
            current_time=None,
            sample_interval=1.0,
            observed_depi_dt=None,
            source=source,
            time_basis=_LEGACY_TIME_BASIS,
            physical_time_resolved=False,
            current_endpoint_matches_state=None,
            is_valid=False,
            reason="non_finite_observed_rate",
        )
        return evidence, True, evidence.reason

    evidence = MutationTriggerEvidence(
        previous_epi=previous_epi,
        current_epi=current_epi,
        previous_time=None,
        current_time=None,
        sample_interval=1.0,
        observed_depi_dt=observed_depi_dt,
        source=source,
        time_basis=_LEGACY_TIME_BASIS,
        physical_time_resolved=False,
        current_endpoint_matches_state=None,
        is_valid=True,
        reason=None,
    )
    return evidence, True, None


def certify_mutation_trigger(
    *,
    current_epi: Real,
    nu_f: Real,
    delta_nfr: Real,
    xi: Real = ZHIR_THRESHOLD_XI_CANONICAL,
    epi_time_history: Any = None,
    epi_history: Any = None,
    legacy_epi_history: Any = None,
    endpoint_tolerance: Real = 0.0,
) -> MutationTriggerCertificate:
    """Build a pure Mutation threshold certificate from one nodal state.

    Parameters
    ----------
    current_epi, nu_f, delta_nfr : Real
        Finite current nodal channels.  The instantaneous prediction is exactly
        the represented operation ``float(nu_f) * float(delta_nfr)``.
    xi : Real, default=ZHIR_THRESHOLD_XI_CANONICAL
        Finite nonnegative strict Mutation threshold.
    epi_time_history : object, optional
        Preferred timestamped history.  Its last two records must be indexed
        pairs ``(time, epi)`` with a finite strictly positive time interval.
        When supplied, it is authoritative and invalid data do not fall back
        to a legacy history.
    epi_history, legacy_epi_history : object, optional
        Canonical and ``_epi_history``-compatible unit-step histories.  A
        nonempty canonical history takes precedence; an empty canonical list
        may fall back to the legacy list, preserving the existing gate policy.
    endpoint_tolerance : Real, default=0
        Finite nonnegative absolute tolerance used only to verify that the
        physical history endpoint represents ``current_epi``.  It never
        weakens the strict threshold comparison.

    Returns
    -------
    MutationTriggerCertificate
        Immutable diagnostic certificate.  Missing or malformed histories
        yield unavailable/invalid evidence and tri-state ``observed_crossed``
        rather than fabricating a negative observation.

    Raises
    ------
    MutationTriggerInputError
        If a current scalar channel, threshold, tolerance, or predicted rate is
        Boolean, non-real, non-finite, or otherwise outside its stated domain.
    """

    current_epi_value = _finite_real(current_epi, "current_epi")
    nu_f_value = _finite_real(nu_f, "nu_f")
    delta_nfr_value = _finite_real(delta_nfr, "delta_nfr")
    xi_value = _finite_real(xi, "xi")
    tolerance_value = _finite_real(endpoint_tolerance, "endpoint_tolerance")
    if xi_value < 0.0:
        raise MutationTriggerInputError("xi", xi, "must be nonnegative")
    if nu_f_value < 0.0:
        raise MutationTriggerInputError("nu_f", nu_f, "must be nonnegative")
    if tolerance_value < 0.0:
        raise MutationTriggerInputError(
            "endpoint_tolerance", endpoint_tolerance, "must be nonnegative"
        )

    predicted_depi_dt = nu_f_value * delta_nfr_value
    if not math.isfinite(predicted_depi_dt):
        raise MutationTriggerInputError(
            "predicted_depi_dt",
            predicted_depi_dt,
            "must be finite for the represented nu_f * delta_nfr product",
        )
    predicted_crossed = predicted_depi_dt > xi_value

    source: str | None
    time_basis: str | None
    if epi_time_history is not None:
        source = _PHYSICAL_SOURCE
        time_basis = _PHYSICAL_TIME_BASIS
        evidence, available, reason = _physical_evidence(
            epi_time_history,
            current_epi=current_epi_value,
            endpoint_tolerance=tolerance_value,
        )
    else:
        selected = _select_legacy_history(epi_history, legacy_epi_history)
        if selected is None:
            return _invalid_certificate(
                current_epi=current_epi_value,
                nu_f=nu_f_value,
                delta_nfr=delta_nfr_value,
                xi=xi_value,
                predicted_depi_dt=predicted_depi_dt,
                predicted_crossed=predicted_crossed,
                source=None,
                time_basis=None,
                reason="missing_history",
                evidence_available=False,
            )
        history, source = selected
        time_basis = _LEGACY_TIME_BASIS
        evidence, available, reason = _legacy_evidence(history, source=source)

    if evidence is None or not evidence.is_valid:
        return _invalid_certificate(
            current_epi=current_epi_value,
            nu_f=nu_f_value,
            delta_nfr=delta_nfr_value,
            xi=xi_value,
            predicted_depi_dt=predicted_depi_dt,
            predicted_crossed=predicted_crossed,
            source=source,
            time_basis=time_basis,
            reason=reason or "invalid_history",
            evidence_available=available,
            evidence=evidence,
        )

    observed_depi_dt = evidence.observed_depi_dt
    if observed_depi_dt is None:  # Defensive: valid evidence always has a rate.
        return _invalid_certificate(
            current_epi=current_epi_value,
            nu_f=nu_f_value,
            delta_nfr=delta_nfr_value,
            xi=xi_value,
            predicted_depi_dt=predicted_depi_dt,
            predicted_crossed=predicted_crossed,
            source=source,
            time_basis=time_basis,
            reason="observed_rate_unavailable",
            evidence_available=available,
            evidence=evidence,
        )

    observed_crossed = observed_depi_dt > xi_value
    rate_gap: float | None = None
    if evidence.physical_time_resolved:
        candidate_gap = observed_depi_dt - predicted_depi_dt
        if math.isfinite(candidate_gap):
            rate_gap = candidate_gap

    return MutationTriggerCertificate(
        current_epi=current_epi_value,
        nu_f=nu_f_value,
        delta_nfr=delta_nfr_value,
        xi=xi_value,
        predicted_depi_dt=predicted_depi_dt,
        observed_depi_dt=observed_depi_dt,
        observed_crossed=observed_crossed,
        predicted_crossed=predicted_crossed,
        evidence_available=True,
        evidence_valid=True,
        capacity_active=nu_f_value > 0.0,
        threshold_gate_satisfied=observed_crossed,
        source=evidence.source,
        time_basis=evidence.time_basis,
        physical_time_resolved=evidence.physical_time_resolved,
        current_endpoint_matches_state=evidence.current_endpoint_matches_state,
        reason=None if observed_crossed else "threshold_not_crossed",
        rate_gap=rate_gap,
        evidence=evidence,
    )
