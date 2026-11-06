"""Independent controls for conditional binary64 three-sample arithmetic.

All observations below are synthetic polynomials. No retained research source,
trajectory, operator event or acquisition is consumed or manufactured.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import product

import networkx as nx
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.physics._sample_reader_budget import bound_binary64_acceleration_reader

H = Q(1, 1024)
EPSILON = Q(1, 2**24)
AT = Q(3682305, 524288)


def _arguments(**changes):
    arguments = dict(
        form_bounds=I(Q(251, 1250), Q(53, 250)),
        rate_bounds=I(4, 5),
        acceleration_bounds=I(111, 129),
        sample_step=H,
        observation_time=AT,
        sample_error_bounds=(EPSILON,) * 3,
    )
    arguments.update(changes)
    return arguments


def _read_synthetic(values, times):
    history = [(float(t), float(y)) for t, y in zip(times, values)]
    assert tuple(Q.from_float(t) for t, _ in history) == times
    assert tuple(Q.from_float(y) for _, y in history) == values
    graph = nx.Graph()
    graph.add_node("synthetic", EPI=history[-1][1], epi_time_history=history)
    before = deepcopy(graph.nodes["synthetic"])
    observed = observe_structural_acceleration(graph, "synthetic")
    assert graph.nodes["synthetic"] == before
    assert observed.available
    assert observed.time_basis == "physical_time"
    return Q.from_float(observed.value)


@pytest.mark.parametrize("acceleration", [120, -120])
def test_synthetic_reader_equals_independent_exact_stencil_for_error_corners(
    acceleration,
):
    report = bound_binary64_acceleration_reader(
        **_arguments(acceleration_bounds=I(acceleration))
    )
    assert report.exact
    assert report.arithmetic_error_bound == 0
    assert report.abstention_reason is None
    elapsed = (Q(0), H, 2 * H)
    true_values = tuple(
        Q(13, 64) + Q(17, 4) * t + Q(acceleration, 2) * t**2 for t in elapsed
    )
    # The analytic polynomial satisfies the independent whole-window premises.
    assert all(_arguments()["form_bounds"].contains(y) for y in true_values)
    assert all(4 <= Q(17, 4) + acceleration * t <= 5 for t in elapsed)
    residuals = []
    for perturbations in product((-EPSILON, Q(0), EPSILON), repeat=3):
        values = tuple(y + error for y, error in zip(true_values, perturbations))
        exact_stencil = (values[0] - 2 * values[1] + values[2]) / H**2
        observed = _read_synthetic(values, report.sample_times)
        assert observed == exact_stencil
        assert report.represented_acceleration_bounds.contains(observed)
        residuals.append(observed - acceleration)
    assert min(residuals) == -Q(1, 4)
    assert max(residuals) == Q(1, 4)
    assert report.acceleration_sample_error_bound == Q(1, 4)


def test_prospective_window_preserves_exact_times_and_separate_error_budgets():
    report = bound_binary64_acceleration_reader(**_arguments())
    assert report.sample_times == tuple(
        Q(n, 524288) for n in (3682305, 3682817, 3683329)
    )
    assert report.sample_step == H
    assert report.sample_error_bounds == (EPSILON,) * 3
    assert report.acceleration_sample_error_bound == Q(1, 4)
    assert report.represented_acceleration_bounds == I(Q(443, 4), Q(517, 4))
    assert report.represented_secant_bounds == (I(4 - Q(1, 8192), 5 + Q(1, 8192)),) * 2
    assert report.exact and report.arithmetic_error_bound == 0
    with pytest.raises(FrozenInstanceError):
        report.exact = False


def test_unequal_sample_errors_have_independent_stencil_weights():
    errors = (EPSILON, 2 * EPSILON, 3 * EPSILON)
    report = bound_binary64_acceleration_reader(
        **_arguments(sample_error_bounds=errors)
    )
    assert report.exact
    expected = (errors[0] + 2 * errors[1] + errors[2]) / H**2
    assert report.acceleration_sample_error_bound == expected == Q(1, 2)
    true_values = tuple(Q(13, 64) + Q(17, 4) * t + 60 * t**2 for t in (0, H, 2 * H))
    values = tuple(
        y + sign * error for y, sign, error in zip(true_values, (1, -1, 1), errors)
    )
    assert _read_synthetic(values, report.sample_times) - 120 == expected


def test_exact_cancellation_is_available_without_certifying_small_nonzero_values():
    report = bound_binary64_acceleration_reader(
        **_arguments(acceleration_bounds=I(0), sample_error_bounds=(0, 0, 0))
    )
    assert report.exact
    assert report.represented_acceleration_bounds == I(0)
    values = tuple(Q(13, 64) + Q(17, 4) * t for t in (0, H, 2 * H))
    assert _read_synthetic(values, report.sample_times) == 0

    crossing = bound_binary64_acceleration_reader(
        **_arguments(acceleration_bounds=I(-1, 1), sample_error_bounds=(0, 0, 0))
    )
    assert not crossing.exact
    assert crossing.arithmetic_error_bound is None
    assert "not_separated_from_nonnormal_values" in crossing.abstention_reason


@pytest.mark.parametrize("channel", ["form_bounds", "rate_bounds"])
def test_positive_sterbenz_factor_two_boundary_is_inclusive(channel):
    changes = {channel: I(1, 2), "sample_error_bounds": (0, 0, 0)}
    boundary = bound_binary64_acceleration_reader(**_arguments(**changes))
    assert boundary.exact
    changes[channel] = I(1, 2 + Q(1, 2**100))
    beyond = bound_binary64_acceleration_reader(**_arguments(**changes))
    assert not beyond.exact
    assert beyond.arithmetic_error_bound is None
    assert "sterbenz" in beyond.abstention_reason


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"sample_step": Q(3, 1024)}, "sample_step_is_not_a_power_of_two"),
        ({"sample_step": Q(1, 3)}, "clock_not_exactly_representable_in_binary64"),
        ({"observation_time": Q(1, 3)}, "clock_not_exactly_representable_in_binary64"),
        ({"observation_time": 2**53}, "clock_not_exactly_representable_in_binary64"),
        ({"sample_step": Q(1, 2**1074)}, "clock_outside_normal_finite_range"),
        ({"sample_step": Q(1, 2**1200)}, "clock_outside_normal_finite_range"),
        ({"observation_time": 2**1100}, "clock_not_exactly_representable_in_binary64"),
        (
            {"form_bounds": I(2**1025, 2**1026)},
            "represented_forms_outside_normal_finite_range",
        ),
        (
            {"form_bounds": I(Q(1, 2**1074))},
            "represented_forms_outside_normal_finite_range",
        ),
        ({"form_bounds": I(-2, -1)}, "form_subtractions_lack_positive_sterbenz_bounds"),
        (
            {"rate_bounds": I(-5, -4)},
            "secant_subtraction_lacks_positive_sterbenz_bounds",
        ),
        ({"rate_bounds": I(0)}, "secant_subtraction_lacks_positive_sterbenz_bounds"),
        (
            {"acceleration_bounds": I(Q(1, 2**1074))},
            "acceleration_scaling_not_separated_from_nonnormal_values",
        ),
    ],
)
def test_valid_unsupported_domains_abstain_without_fabricating_zero(changes, reason):
    report = bound_binary64_acceleration_reader(**_arguments(**changes))
    assert not report.exact
    assert report.arithmetic_error_bound is None
    assert reason in report.abstention_reason
    assert report.acceleration_sample_error_bound >= 0


@pytest.mark.parametrize(
    "channel", ["form_bounds", "rate_bounds", "acceleration_bounds"]
)
@pytest.mark.parametrize(
    "value", [True, 0.5, float("nan"), float("inf"), complex(1, 1)]
)
def test_interval_inputs_require_explicit_exact_scalar_admission(channel, value):
    with pytest.raises((TypeError, ValueError)):
        bound_binary64_acceleration_reader(**_arguments(**{channel: value}))


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_step": True},
        {"sample_step": 0},
        {"sample_step": -1},
        {"sample_step": float("nan")},
        {"sample_step": float("inf")},
        {"observation_time": False},
        {"observation_time": -1},
        {"observation_time": float("nan")},
        {"sample_error_bounds": (0, 0)},
        {"sample_error_bounds": (0, 0, 0, 0)},
        {"sample_error_bounds": (0, True, 0)},
        {"sample_error_bounds": (0, -EPSILON, 0)},
        {"sample_error_bounds": (0, float("inf"), 0)},
    ],
)
def test_invalid_clock_or_sample_error_domains_reject(changes):
    with pytest.raises((TypeError, ValueError)):
        bound_binary64_acceleration_reader(**_arguments(**changes))


def test_pure_budget_does_not_invoke_the_observation_reader(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a prospective budget must not observe or invent samples")

    monkeypatch.setattr(
        "tnfr.operators.nodal_equation.observe_structural_acceleration", forbidden
    )
    report = bound_binary64_acceleration_reader(**_arguments())
    assert report.exact
