"""Prospective reference/source separation for the fixed class comparison.

The complete nonlinear reference is generated independently. This owner builds
its primitive input covers and transfers a supplied reference enclosure to the
original acquired families. The cubic prediction is used only for a separate
consistency check, never to narrow the forward interval.
"""

from fractions import Fraction as Q

from ..mathematics._phase_midpoint import _pi_bounds
from ..physics._sine_class_contrast import _contrast_decision
from .artifact_io import exact_record

__all__ = (
    "canonical_comparison_sources",
    "comparison_producer_inputs",
    "comparison_observation_policy",
    "assess_reference_contrast",
)

_EPSILON = Q(1, 10**32)
_CLASSES = ((1, 1, 1), (1, 2, 1))


def canonical_comparison_sources():
    """Return exact nominal reference covers and separate actual-family covers.

    The reference target uses certified pi endpoints; it is not an acquired
    state substituted for the original preparation. Coordinate covers contain
    the component balls and zero-sum families without encoding correlations.
    """
    pi = _pi_bounds()
    reference_phase, actual_phase = [], []
    for classes in _CLASSES:
        target = []
        for winding in classes:
            for local in range(9):
                coefficient = Q(2 * winding * (local - 4), 9)
                products = tuple(coefficient * endpoint for endpoint in pi)
                target.append((min(products), max(products)))
        reference_phase.append(tuple(target))
        actual_phase.append(tuple((lo - _EPSILON, hi + _EPSILON) for lo, hi in target))
    return dict(
        classes=_CLASSES,
        pi_bounds=pi,
        reference_form_bounds=(((Q(0), Q(0)),) * 27,) * 2,
        reference_phase_bounds=tuple(reference_phase),
        actual_form_bounds=(((-_EPSILON, _EPSILON),) * 27,) * 2,
        actual_phase_bounds=tuple(actual_phase),
    )


def comparison_producer_inputs():
    """Declare the one reference intervention and fixed numerical attempt budget."""
    source = canonical_comparison_sources()
    return dict(
        initial_form_bounds=source["reference_form_bounds"],
        initial_phase_bounds=source["reference_phase_bounds"],
        first_probe_amplitude=Q(7, 10000),
        second_probe_amplitude=Q(7, 10000),
        delay=Q(1),
        total_duration=Q(2),
        time_step=Q(1, 128),
        order=16,
        max_steps=1536,
    )


def comparison_observation_policy():
    """Declare source transfer, sensor errors, prediction and a width postcondition.

    The width budget applies to the transported actual-family interval.
    It is not a promise that the chosen interval solver will attain it.
    """
    return dict(
        endpoint_radius=_EPSILON,
        gamma_upper=Q(1, 3000),
        total_duration=Q(2),
        readout_error_bound=Q(1, 10**30),
        reference_prediction_open_bounds=(Q(-19254, 10**33), Q(-19237, 10**33)),
        prediction_open_bounds=(Q(-19334, 10**33), Q(-19157, 10**33)),
        width_allowance=Q(1, 10**30),
    )


def _pair(value, label):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label} requires two exact endpoints")
    lower, upper = map(exact_record, value)
    if lower > upper:
        raise ValueError(f"{label} endpoints must be ordered")
    return lower, upper


def assess_reference_contrast(
    reference_bounds,
    *,
    endpoint_radius,
    gamma_upper,
    total_duration,
    readout_error_bound,
    reference_prediction_open_bounds,
    prediction_open_bounds,
    width_allowance,
):
    """Apply an independently justified source bound once to a reference band.

    Exact scalar admission is not derivative validation or source acquisition.
    The caller must reconstruct the reference evidence and match the complete
    law, target, original source distance, clock, events and observation first.
    This function never consumes a producer report or an analytic coefficient.
    """
    eps, gamma, total, delta, allowance = map(
        exact_record,
        (
            endpoint_radius,
            gamma_upper,
            total_duration,
            readout_error_bound,
            width_allowance,
        ),
    )
    reference_prediction = _pair(
        reference_prediction_open_bounds, "open reference prediction"
    )
    prediction = _pair(prediction_open_bounds, "open actual-source prediction")
    if not (eps >= 0 and gamma >= 0 and 0 <= total <= 2 and delta >= 0):
        raise ValueError("require nonnegative source, rate, noise and duration<=2")
    if 2 * gamma * total >= 1 or allowance <= 0:
        raise ValueError("require positive source denominator and width allowance")
    if any(pair[0] == pair[1] for pair in (reference_prediction, prediction)):
        raise ValueError("open predictions must have positive width")
    source_error = 8 * eps / (1 - 2 * gamma * total)
    reference = (
        None if reference_bounds is None else _pair(reference_bounds, "reference")
    )
    actual = decision = reference_width = actual_width = within = overlap = contains = (
        None
    )
    reference_overlap = reference_contains = None
    status = "unavailable"
    if reference is not None:
        actual = reference[0] - source_error, reference[1] + source_error
        reference_width = reference[1] - reference[0]
        actual_width = actual[1] - actual[0]
        within = actual_width <= allowance
        reference_overlap = (
            reference[1] > reference_prediction[0]
            and reference[0] < reference_prediction[1]
        )
        reference_contains = (
            reference_prediction[0] < reference[0]
            and reference[1] < reference_prediction[1]
        )
        overlap = actual[1] > prediction[0] and actual[0] < prediction[1]
        contains = prediction[0] < actual[0] and actual[1] < prediction[1]
        decision = _contrast_decision(actual, Q(0), delta)
        if not reference_overlap or not overlap:
            status = "consistency_conflict"
        elif not within:
            status = "resolution_not_certified"
        elif not decision.null_excluded:
            status = "discrimination_not_certified"
        else:
            status = "discrimination_certified"
    return dict(
        reference_bounds=reference,
        actual_bounds=actual,
        reference_width=reference_width,
        actual_width=actual_width,
        source_error_upper_bound=source_error,
        decision=decision,
        width_within_allowance=within,
        reference_theorem_open_interval_overlap=reference_overlap,
        reference_theorem_open_interval_contains_band=reference_contains,
        theorem_open_interval_overlap=overlap,
        theorem_open_interval_contains_actual_band=contains,
        status=status,
    )
