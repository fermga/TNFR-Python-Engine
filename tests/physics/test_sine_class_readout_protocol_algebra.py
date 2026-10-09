"""Pure pre-response arithmetic of the frozen four-history protocol.

Only three named source/criterion functions are compiled from the archive.
No archived imports, main body, evaluator, source assessment or flow executes.
"""

import ast
import itertools
import zipfile
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import pytest

from tnfr.mathematics._phase_midpoint import _pi_bounds
from tnfr.mathematics._rational_interval import _exact

ROOT = Path(__file__).resolve().parents[2]
HELPER = "build/class-nonlinear-readout-freeze/experiment_support.py"
ARCHIVE = ROOT / "docs/assets/sine_formed_classes/class-nonlinear-readout-v1.source.zip"
FUNCTIONS = {"canonical_source", "producer_inputs", "interval_criteria"}


@pytest.fixture(scope="module")
def arithmetic():
    with zipfile.ZipFile(ARCHIVE) as source:
        body = source.read(HELPER)
    parsed = ast.parse(body, filename=HELPER)
    selected = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    assert {node.name for node in selected} == FUNCTIONS
    assert all(not node.decorator_list for node in selected)
    namespace = {"Q": Q, "_pi_bounds": _pi_bounds, "_exact": _exact}
    exec(compile(ast.Module(body=selected, type_ignores=[]), HELPER, "exec"), namespace)
    return namespace


def test_fixed_complete_branch_schedule_has_exactly384_unique_attempts(arithmetic):
    inputs = arithmetic["producer_inputs"]()
    assert set(inputs) == {
        "initial_form_bounds",
        "initial_phase_bounds",
        "first_probe_amplitude",
        "second_probe_amplitude",
        "delay",
        "total_duration",
        "time_step",
        "order",
        "max_steps",
    }
    assert (
        inputs["first_probe_amplitude"]
        == inputs["second_probe_amplitude"]
        == Q(1, 2000)
    )
    assert inputs["delay"] == 1 and inputs["total_duration"] == 2
    assert inputs["time_step"] == Q(1, 64) and inputs["order"] == 16
    lengths = (inputs["delay"],) * 2 + (inputs["total_duration"] - inputs["delay"],) * 4
    counts = tuple(length / inputs["time_step"] for length in lengths)
    assert all(count.denominator == 1 for count in counts)
    assert sum(counts) == inputs["max_steps"] == 384


def test_canonical_cover_retains_exact_target_signs_and_original_ball_correlations(
    arithmetic,
):
    pi, form, phase = arithmetic["canonical_source"]()
    inputs = arithmetic["producer_inputs"]()
    assert inputs["initial_form_bounds"] == form
    assert inputs["initial_phase_bounds"] == phase
    assert len(form) == len(phase) == 27
    epsilon = Q(1, 10**32)
    context = mpmath.mp.clone()
    context.dps = 100

    def real(value):
        return context.mpf(value.numerator) / value.denominator

    assert real(pi[0]) < context.pi < real(pi[1])
    assert 0 < pi[1] - pi[0] < Q(1, 2**250)
    deviations = tuple(Q(j - 4, 20) * epsilon for j in range(9))
    assert sum(deviations, Q(0)) == 0
    assert sum((value**2 for value in deviations), Q(0)) < epsilon**2
    for component, winding in enumerate((1, 2, 1)):
        for local in range(9):
            index = 9 * component + local
            coefficient = Q(2 * winding * (local - 4), 9)
            products = tuple(coefficient * endpoint for endpoint in pi)
            assert form[index] == (-epsilon, epsilon)
            assert phase[index] == (
                min(products) - epsilon,
                max(products) + epsilon,
            )
            target = 2 * winding * (local - 4) * context.pi / 9
            actual = target + real(deviations[local])
            assert real(phase[index][0]) < actual < real(phase[index][1])
        assert phase[9 * component + 4] == (-epsilon, epsilon)
    # The enclosure covers the correlated family without asserting that every
    # corner belongs to it: this form corner violates both original premises.
    assert all(low <= epsilon <= high for low, high in form)
    assert 9 * epsilon != 0 and 9 * epsilon**2 > epsilon**2


@pytest.mark.parametrize(
    ("upper_multiplier", "expected"),
    (
        (Q(1), (False, False, False)),
        (Q(0), (False, False, False)),
        (Q(-1), (True, False, False)),
        (Q(-4), (True, False, False)),
        (Q(-9, 2), (True, True, False)),
        (Q(-8), (True, True, False)),
        (Q(-17, 2), (True, True, True)),
    ),
)
def test_true_recorded_and_record_set_signs_have_distinct_strict_boundaries(
    arithmetic, upper_multiplier, expected
):
    delta = Q(1, 1024)
    upper = upper_multiplier * delta
    result = arithmetic["interval_criteria"]((upper - delta, upper), delta, (-1, 1))
    keys = (
        "true_mixed_negative",
        "recorded_mixed_negative",
        "four_record_sets_disjoint",
    )
    assert tuple(result[key] for key in keys) == expected


@pytest.mark.parametrize(
    ("bounds", "overlap", "contained"),
    (
        ((Q(-5), Q(-4)), False, False),
        ((Q(-3), Q(-2)), False, False),
        ((Q(-4), Q(-4)), False, False),
        ((Q(-3), Q(-3)), False, False),
        ((Q(-4), Q(-3)), True, False),
        ((Q(-7, 2), Q(-13, 4)), True, True),
        ((Q(-5), Q(-2)), True, False),
    ),
)
def test_closed_forward_band_must_intersect_the_open_prediction(
    arithmetic, bounds, overlap, contained
):
    result = arithmetic["interval_criteria"](bounds, Q(0), (Q(-4), Q(-3)))
    assert result["theorem_open_interval_overlap"] is overlap
    assert result["theorem_open_interval_contains_numerical_band"] is contained


def test_missing_and_invalid_bands_never_supply_passing_observation_flags(arithmetic):
    classify = arithmetic["interval_criteria"]
    assert all(
        value is None for value in classify(None, Q(1, 10**30), (Q(-4), Q(-3))).values()
    )
    for bounds, delta, prediction in (
        ((Q(1), Q(-1)), Q(0), (Q(-4), Q(-3))),
        ((Q(-4), Q(-3)), Q(-1), (Q(-4), Q(-3))),
        ((Q(-4), Q(-3)), Q(0), (Q(-3), Q(-4))),
        ((Q(-4), Q(-3)), Q(0), (Q(-3), Q(-3))),
    ):
        with pytest.raises(ValueError):
            classify(bounds, delta, prediction)


@pytest.mark.parametrize("invalid", (True, False, float("nan"), float("inf"), 0.5))
def test_criterion_primitives_require_exact_finite_nonboolean_scalars(
    arithmetic, invalid
):
    classify = arithmetic["interval_criteria"]
    cases = (
        ((invalid, Q(-3)), Q(0), (Q(-4), Q(-3))),
        ((Q(-4), invalid), Q(0), (Q(-4), Q(-3))),
        ((Q(-4), Q(-3)), invalid, (Q(-4), Q(-3))),
        ((Q(-4), Q(-3)), Q(0), (invalid, Q(-3))),
        ((Q(-4), Q(-3)), Q(0), (Q(-4), invalid)),
        (None, invalid, (Q(-4), Q(-3))),
    )
    for bounds, delta, prediction in cases:
        with pytest.raises((TypeError, ValueError)):
            classify(bounds, delta, prediction)


def test_four_actual_reading_errors_explain_both_distinct_noise_margins(arithmetic):
    delta = Q(1, 64)
    coefficients = (1, -1, -1, 1)
    error_statistics = {
        sum((weight * value for weight, value in zip(coefficients, errors)), Q(0))
        for errors in itertools.product((-delta, delta), repeat=4)
    }
    assert min(error_statistics) == -4 * delta
    assert max(error_statistics) == 4 * delta
    # True negativity alone does not exclude noisy zero, and recorded
    # negativity alone does not exclude the tangent's four-record range.
    true_mixed = -6 * delta
    nonlinear_records = {true_mixed + error for error in error_statistics}
    assert max(nonlinear_records) < 0
    assert nonlinear_records.intersection(error_statistics)
    result = arithmetic["interval_criteria"]((true_mixed, true_mixed), delta, (-1, 0))
    assert result["recorded_mixed_negative"]
    assert not result["four_record_sets_disjoint"]
    # At the exact 8delta boundary the sets can still touch.
    assert -8 * delta + max(error_statistics) == min(error_statistics)
    strictly_separated = -9 * delta
    assert strictly_separated + max(error_statistics) < min(error_statistics)
    result = arithmetic["interval_criteria"](
        (strictly_separated, strictly_separated), delta, (-1, 0)
    )
    assert result["four_record_sets_disjoint"]
