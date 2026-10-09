"""Use reconstructed prior coefficients as a conditional amplitude premise.

The shared read-only fixture verifies retained bytes and rebuilds all complete
time-polynomial endpoints, tails and event carry. Derivative generation remains
an explicit execution premise; no coefficient, response or acquisition is run.
The new calculator consumes primitive coefficient endpoints, not old verdicts.
"""

from fractions import Fraction as Q

import pytest

from tests.physics import test_sine_class_cubic_evidence as cubic_audit
from tnfr.physics.relational_sine_class_amplitude_feasibility import (
    bound_sine_class_amplitude_feasibility,
)
from tnfr.research.sine_class_comparison_protocol import (
    comparison_observation_policy,
    comparison_producer_inputs,
)

# Reuse the existing bounded archive and arithmetic owner without collecting
# its test functions or adding another scientific report reader.
retained = cubic_audit.retained
evidence = cubic_audit.evidence
prior_no_execution = cubic_audit.no_coefficient_response_or_worker_execution

DISPLAYED_BASE = Q(-7013764940694, 10**42), Q(-7013764940692, 10**42)
SCALE = Q(4, 3), Q(7, 5)


@pytest.fixture(scope="module")
def reconstructed(evidence):
    inputs, prior_report, gamma, mixed = evidence
    difference = mixed[0] - mixed[1]
    products = tuple(
        g**4 * c for g in (gamma.lo, gamma.hi) for c in (difference.lo, difference.hi)
    )
    coefficient = min(products), max(products)
    result = bound_sine_class_amplitude_feasibility(
        amplitude_scale_lower=SCALE[0],
        amplitude_scale_upper=SCALE[1],
        base_cubic_lower=coefficient[0],
        base_cubic_upper=coefficient[1],
    )
    return inputs, prior_report, gamma, coefficient, result


def test_conditional_coefficient_matches_original_source_law_and_events(reconstructed):
    inputs, prior_report, gamma, coefficient, result = reconstructed
    assert result.gamma_bounds == gamma
    for key, expected in (
        ("base_first_probe_amplitude", inputs["first_probe_amplitude"]),
        ("base_second_probe_amplitude", inputs["second_probe_amplitude"]),
        ("delay", inputs["delay"]),
        ("total_duration", inputs["total_duration"]),
        ("endpoint_radius", inputs["endpoint_radius"]),
        ("readout_error_bound", inputs["readout_error_bound"]),
        ("radius", inputs["radius"]),
        ("contact_work_allowance", inputs["contact_work_allowance"]),
        ("first_probe_work_allowance", inputs["first_probe_work_allowance"]),
        ("second_probe_work_allowance", inputs["second_probe_work_allowance"]),
    ):
        assert getattr(result, key) == expected
    for key in ("compared_classes", "nodes", "edges", "contacts", "clock"):
        assert getattr(result, key) == prior_report[key]
    assert result.base_cubic_lower == coefficient[0]
    assert result.base_cubic_upper == coefficient[1]
    assert DISPLAYED_BASE[0] < coefficient[0] <= coefficient[1] < DISPLAYED_BASE[1]
    assert coefficient[1] - coefficient[0] > 0  # Retain the finite-time error.


def test_whole_scale_interval_uses_corners_and_monotone_complete_error(reconstructed):
    _, _, gamma, coefficient, result = reconstructed
    lower, upper = SCALE
    # A negative coefficient and positive scale attain these extrema on the
    # entire interval, without evaluating selected response amplitudes.
    cubic = upper**3 * coefficient[0], lower**3 * coefficient[1]
    assert result.scaled_cubic_contrast_bounds == cubic
    g, T, eps, delta, impulse = (
        Q(1, 3000),
        Q(2),
        Q(1, 10**32),
        Q(1, 10**30),
        upper / 2000,
    )
    higher = 2 * sum(
        256 * g**6 * A**5 * T / ((1 - 2 * g**2 * T**2) * (1 - 4 * g**2 * A**2))
        for A in (Q(0), impulse, impulse, 2 * impulse)
    )
    source = 8 * eps / (1 - 2 * gamma.hi * T)
    assert result.higher_amplitude_contrast_error_upper_bound == higher
    assert result.source_contrast_error_upper_bound == source
    true = cubic[0] - higher - source, cubic[1] + higher + source
    assert result.decision.true_bounds == true
    assert result.decision.recorded_bounds == (true[0] - 8 * delta, true[1] + 8 * delta)
    assert result.decision.null_separation_margin == -true[1] - 16 * delta > 0
    assert result.feasible_interval_certified and result.status == "feasible_interval"
    assert result.all_identities_certified and result.all_work_within_allowances


def test_displayed_theorem_premise_is_a_conservative_enclosure(reconstructed):
    _, _, _, _, exact = reconstructed
    displayed = bound_sine_class_amplitude_feasibility(
        amplitude_scale_lower=SCALE[0],
        amplitude_scale_upper=SCALE[1],
        base_cubic_lower=DISPLAYED_BASE[0],
        base_cubic_upper=DISPLAYED_BASE[1],
    )
    assert displayed.decision.true_bounds[0] < exact.decision.true_bounds[0]
    assert exact.decision.true_bounds[1] < displayed.decision.true_bounds[1]
    assert Q(-19334, 10**33) < displayed.decision.true_bounds[0]
    assert displayed.decision.true_bounds[1] < Q(-16537, 10**33)
    assert displayed.decision.null_separation_margin > Q(537, 10**33)
    assert displayed.feasible_interval_certified
    assert displayed.upper_scale_history_bounds == exact.upper_scale_history_bounds


def test_fixed_comparison_predictions_enclose_reconstructed_scaled_evidence(
    reconstructed,
):
    inputs, _, gamma, coefficient, _ = reconstructed
    producer = comparison_producer_inputs()
    policy = comparison_observation_policy()
    scale = producer["first_probe_amplitude"] / inputs["first_probe_amplitude"]
    assert scale == Q(7, 5)
    assert (
        producer["second_probe_amplitude"] / inputs["second_probe_amplitude"] == scale
    )
    assert (
        producer["first_probe_amplitude"]
        == producer["second_probe_amplitude"]
        == Q(7, 10000)
    )
    assert producer["delay"] == inputs["delay"]
    assert (
        producer["total_duration"]
        == policy["total_duration"]
        == inputs["total_duration"]
    )
    assert policy["endpoint_radius"] == inputs["endpoint_radius"]
    assert policy["readout_error_bound"] == inputs["readout_error_bound"]

    # Rebuild both bands from the retained complete cubic coefficient, rather
    # than expanding either rounded display or consuming a cached verdict.
    a, b = producer["first_probe_amplitude"], producer["second_probe_amplitude"]
    g, T = policy["gamma_upper"], producer["total_duration"]
    assert gamma.hi < g == Q(1, 3000)
    higher = 2 * sum(
        256 * g**6 * A**5 * T / ((1 - 2 * g**2 * T**2) * (1 - 4 * g**2 * A**2))
        for A in (Q(0), a, b, a + b)
    )
    nominal = scale**3 * coefficient[0] - higher, scale**3 * coefficient[1] + higher
    source = 8 * policy["endpoint_radius"] / (1 - 2 * g * T)
    actual = nominal[0] - source, nominal[1] + source
    for key, rebuilt in (
        ("reference_prediction_open_bounds", nominal),
        ("prediction_open_bounds", actual),
    ):
        lower, upper = policy[key]
        assert lower < rebuilt[0] <= rebuilt[1] < upper
