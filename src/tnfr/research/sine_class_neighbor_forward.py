"""Prospective full-law check of distinct-neighbor nonadditivity.

Only primitive sources, events and numerical policy reach the shared producer.
The static prediction is rebuilt for a separate comparison; it never narrows
the forward enclosure. Original acquisition and a common complete source are
conditional premises, not consequences of these arithmetic checks.
"""

from dataclasses import asdict
from fractions import Fraction as Q

from ..physics._sine_class_collective_interface import _bound_collective_interface
from ..physics._sine_class_contrast import _contrast_decision
from ..physics._sine_class_neighbor_nonadditivity import _bound_neighbor_nonadditivity
from ..physics._sine_class_readout_evidence import _reconstruct_class_readout
from ..physics.relational_sine_class_readout import _admit_class_readout_inputs
from .artifact_io import exact_record
from .sine_class_comparison_protocol import canonical_comparison_sources

_EPS, _G, _H, _M = Q(1, 10**32), Q(1, 3000), Q(1, 8), Q(7, 10000)


def neighbor_forward_sources():
    """Select the original (1,2,1) reference and separate acquired-family cover.

    Absolute continuous phases enclose Theta, not its deviation coordinates.
    The Cartesian actual cover retains no claim that every corner is acquired;
    the original component balls, zero sums and preparation remain premises.
    """
    sources = canonical_comparison_sources()
    return dict(
        classes=sources["classes"][1],
        pi_bounds=sources["pi_bounds"],
        **{
            key: sources[key][1]
            for key in (
                "reference_form_bounds",
                "reference_phase_bounds",
                "actual_form_bounds",
                "actual_phase_bounds",
            )
        },
    )


def neighbor_forward_inputs():
    """Fix a single four-history attempt without executing any flow."""
    source = neighbor_forward_sources()
    return dict(
        initial_form_bounds=source["reference_form_bounds"],
        initial_phase_bounds=source["reference_phase_bounds"],
        first_probe_amplitude=_M,
        second_probe_amplitude=_M,
        first_probe_node=4,
        second_probe_node=22,
        readout_node=13,
        delay=Q(0),
        total_duration=_H,
        time_step=Q(1, 128),
        order=16,
        max_steps=64,
    )


def neighbor_forward_policy():
    """Keep full-law source transfer separate from analytic truncation errors.

    All four histories share exp(J*t)*z0 exactly. Only the four remaining
    nonlinear initialization defects survive their signed sum. In particular
    the zero-input defect is retained. No R5 or cubic feedback error is added
    to an independent complete-law forward enclosure.
    """
    source = sum(
        (
            _bound_collective_interface(
                total_input_variation=amplitude,
                horizon=_H,
                endpoint_radius=_EPS,
            ).nonlinear_initialization_error_upper_bound
            for amplitude in (Q(0), _M, _M, 2 * _M)
        ),
        Q(0),
    )
    return dict(
        endpoint_radius=_EPS,
        gamma_upper=_G,
        total_duration=_H,
        source_error_upper_bound=source,
        readout_error_bound=Q(1, 10**30),
        actual_width_allowance=Q(1, 10**30),
        reading_count=4,
        independent_additive_reading_count=4,
        mixed_coefficients=(1, -1, -1, 1),
        prediction_comparison="closed_interval_overlap_without_intersection",
        separation="negative_forward_recorded_upper_below_independent_additive_recorded_lower",
        primary_observation="shared_prefix_cancelled_suffix_increment_mixed_interval",
        source_transfer="four_nonlinear_initialization_defects_after_exact_common_linear_cancellation",
    )


def neighbor_forward_prediction():
    """Rebuild the static theorem and work/identity checks, not a response."""
    return _bound_neighbor_nonadditivity(
        mediator_class=2,
        donor_amplitude=_M,
        receiver_amplitude=_M,
        horizon=_H,
        endpoint_radius=_EPS,
        readout_error_bound=Q(1, 10**30),
        radius=Q(1, 12),
        contact_work_allowance=Q(1, 10**12),
        donor_work_allowance=Q(1, 10**6),
        receiver_work_allowance=Q(1, 10**6),
    )


def _pair(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("reference requires an ordered exact endpoint pair")
    lower, upper = map(exact_record, value)
    if lower > upper:
        raise ValueError("reference endpoints must be ordered")
    return lower, upper


def assess_neighbor_forward_bounds(reference_bounds):
    """Assess an independent scalar enclosure after evidence reconstruction.

    This arithmetic API does not authenticate the supplied reference. A report
    consumer must use the reconstruction entry point below. Prediction bands
    are freshly derived; no stored theoretical or numerical verdict is trusted.
    """
    reference = None if reference_bounds is None else _pair(reference_bounds)
    policy, prediction = neighbor_forward_policy(), neighbor_forward_prediction()
    direct = prediction.static_cone.direct_cubic_heat_bounds
    truncation = (
        prediction.complete_cubic_correction_upper_bound
        + prediction.higher_amplitude_error_upper_bound
    )
    nominal = direct[0] - truncation, direct[1] + truncation
    predicted_actual = prediction.decision.true_bounds
    source = policy["source_error_upper_bound"]
    actual = decision = reference_width = actual_width = None
    narrow = nominal_overlap = actual_overlap = None
    status = "unavailable"
    if reference is not None:
        actual = reference[0] - source, reference[1] + source
        reference_width = reference[1] - reference[0]
        actual_width = actual[1] - actual[0]
        narrow = actual_width <= policy["actual_width_allowance"]
        nominal_overlap = max(reference[0], nominal[0]) <= min(reference[1], nominal[1])
        actual_overlap = max(actual[0], predicted_actual[0]) <= min(
            actual[1], predicted_actual[1]
        )
        decision = _contrast_decision(
            actual, Q(0), policy["readout_error_bound"], reading_count=4
        )
        status = (
            "consistency_conflict"
            if not nominal_overlap or not actual_overlap
            else (
                "numerically_unresolved"
                if not narrow
                else (
                    "all_conditions_met"
                    if decision.orientation == -1 and decision.null_excluded
                    else "discrimination_unresolved"
                )
            )
        )
    return dict(
        reference_bounds=reference,
        actual_bounds=actual,
        reference_width=reference_width,
        actual_width=actual_width,
        source_error_upper_bound=source,
        nominal_prediction_bounds=nominal,
        actual_prediction_bounds=predicted_actual,
        nominal_prediction_overlap=nominal_overlap,
        actual_prediction_overlap=actual_overlap,
        width_within_allowance=narrow,
        decision=None if decision is None else asdict(decision),
        additive_recorded_bounds=(
            -4 * policy["readout_error_bound"],
            4 * policy["readout_error_bound"],
        ),
        all_conditions_met=status == "all_conditions_met",
        status=status,
    )


def assess_neighbor_forward_report(report):
    """Re-admit full-state evidence, including event and observation selectors.

    Replaying retained Taylor arithmetic does not regenerate or authenticate
    its derivative/Picard construction. Complete-law execution and the original
    source theorem remain explicit premises even when every check passes.
    """
    inputs = neighbor_forward_inputs()
    selectors = {
        key: inputs.pop(key)
        for key in ("first_probe_node", "second_probe_node", "readout_node")
    }
    admitted = _admit_class_readout_inputs(**inputs)
    evidence = _reconstruct_class_readout(report, admitted, **selectors)
    reference = evidence.mixed_bounds if evidence.complete else None
    comparison = assess_neighbor_forward_bounds(
        None if reference is None else (reference.lo, reference.hi)
    )
    return dict(
        evidence=evidence,
        comparison=comparison,
        all_conditions_met=evidence.complete and comparison["all_conditions_met"],
        status=comparison["status"],
    )
