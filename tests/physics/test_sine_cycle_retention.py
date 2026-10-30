"""Finite full-storage retention with independent matrix and energy controls.

No trajectory is evaluated. The target is a declared rational preparation,
not an exact irrational twist or a demonstrated acquisition endpoint.
"""

import json
from dataclasses import FrozenInstanceError, dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tests.physics.test_sine_cycle_barrier import _state
from tests.physics.test_sine_phase_offset_partition import CYCLE_LEAF_EDGES
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_partition import observe_sine_collective_pulse
from tnfr.physics.relational_sine_regional import assess_sine_cycle_retention
from tnfr.sdk import export_to_json, relational_report_to_dict


@pytest.fixture(scope="module")
def source():
    u, kappa = Q(7, 50), Q(1, 128)
    q = (kappa, -kappa, Q(0), Q(0), Q(0))
    lq = (3 * kappa, -3 * kappa, kappa, Q(0), -kappa)
    receiver = tuple(u / 4 + value for value in q)
    leaves = tuple(-3 * u / 4 + value + gradient for value, gradient in zip(q, lq))
    phase = tuple((i - 2) * Q(1256637, 10**6) for i in range(5)) * 2
    return _state(epi=receiver + leaves, phase=phase)


def _assess(source, **changes):
    args = dict(cycle=tuple(range(5)), scaled_duration=1, source_error_bound=Q(1, 4096))
    return assess_sine_cycle_retention(source, **(args | changes))


@pytest.fixture(scope="module")
def report(source):
    return _assess(source)


def test_positive_width_live_nonrigid_target_retains_for_one_scaled_time(
    source, report
):
    assert report.status == "certified" and not report.reasons
    assert report.source_error_bound == Q(1, 4096)
    assert report.scaled_duration == 1 and report.clock == "tau=t/pi"
    assert report.initial_winding == 1
    assert report.initial_acute_unit_winding_certified
    assert report.whole_window_retention_certified
    assert report.full_storage_bounds.lo > Q(7, 2)
    assert report.taylor_storage_upper_bound < report.direct_storage_bounds.hi
    assert report.full_storage_bounds.hi <= report.direct_storage_bounds.hi
    assert report.edge_cauchy_factors == (Q(8, 9),) * 5
    assert report.retention_margin_lower_bound > 0
    assert all(value > 0 for value in report.edge_retention_margin_lower_bounds)
    pulse = observe_sine_collective_pulse(
        source, cycle=range(5), contact_turn_offsets=(0,) * 5
    )
    assert pulse.receiver_relative_phase_rates == (0,) * 5
    assert any(
        not interval.contains(0)
        for interval in pulse.receiver_relative_phase_jerk_bounds
    )
    with pytest.raises(FrozenInstanceError):
        report.status = "unavailable"


def test_cauchy_factors_are_full_incidence_norms_with_unequal_degrees_and_chords():
    edges = CYCLE_LEAF_EDGES + ((0, 2), (0, 10), (5, 10), (10, 11))
    source = _state(edges=edges, epi=tuple(Q(i * i - 3 * i, 7) for i in range(12)))
    result = _assess(source)
    degree = source.degrees
    for (i, j), factor in zip(
        zip(range(5), (1, 2, 3, 4, 0)), result.edge_cauchy_factors
    ):
        # delta' = b^T L x, b=K(e_j-e_i). Build its edge incidence
        # coefficients from the complete support, independently of kappa's
        # closed formula. Choosing x=b saturates this exact Cauchy bound.
        b = tuple(
            Q(int(k == j), degree[j]) - Q(int(k == i), degree[i]) for k in range(12)
        )
        coefficients = tuple(b[right] - b[left] for left, right in edges)
        norm = sum((value**2 for value in coefficients), Q(0))
        assert norm == factor
        differences = tuple(
            source.epi[right] - source.epi[left] for left, right in edges
        )
        rate = sum((a * x for a, x in zip(coefficients, differences)), Q(0))
        form_twice = sum((value**2 for value in differences), Q(0))
        assert rate**2 <= factor * form_twice
        assert norm**2 == factor * sum((value**2 for value in coefficients), Q(0))
    assert len(set(result.edge_cauchy_factors)) > 1
    assert result.edge_cauchy_factors[0] != Q(8, 9)


def test_direct_and_correlated_energy_bounds_enclose_independent_full_state_corners(
    source, report
):
    def number(value):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator

    with mp.workdps(90):
        for orientation in (-1, 1):
            x = tuple(
                number(
                    value
                    + orientation * (1 if i % 2 else -1) * report.source_error_bound
                )
                for i, value in enumerate(source.epi)
            )
            theta = tuple(
                number(
                    value
                    + orientation * (1 if i % 3 else -1) * report.source_error_bound
                )
                for i, value in enumerate(source.phase)
            )
            energy = mp.fsum(
                (x[j] - x[i]) ** 2 / 2 + 1 - mp.cos(theta[j] - theta[i])
                for i, j in CYCLE_LEAF_EDGES
            )
            assert (
                number(report.direct_storage_bounds.lo)
                <= energy
                <= number(report.direct_storage_bounds.hi)
            )
            assert energy <= number(report.taylor_storage_upper_bound)
            assert (
                number(report.full_storage_bounds.lo)
                <= energy
                <= number(report.full_storage_bounds.hi)
            )
        floor = 5 * (1 - mp.cos(2 * mp.pi / 5))
        assert (
            number(report.phase_storage_floor_bounds.lo)
            <= floor
            <= number(report.phase_storage_floor_bounds.hi)
        )
        for factor, speed in zip(
            report.edge_cauchy_factors, report.edge_phase_speed_upper_bounds
        ):
            expected = mp.sqrt(
                2 * number(factor) * (number(report.full_storage_bounds.hi) - floor)
            )
            assert expected <= number(speed)


def test_taylor_storage_bound_uses_all_full_graph_gradients_at_a_general_state():
    edges = CYCLE_LEAF_EDGES + ((0, 2), (0, 10), (5, 10), (10, 11))
    source = _state(
        edges=edges,
        epi=tuple(Q(i * i - 5 * i + 1, 17) for i in range(12)),
        phase=tuple(Q((-1) ** i * (i + 1), 23) for i in range(12)),
    )
    epsilon = Q(3, 8191)
    report = _assess(source, source_error_bound=epsilon)

    def number(value):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator

    with mp.workdps(90):
        forms, phases = tuple(map(number, source.epi)), tuple(map(number, source.phase))
        form_gradient = [mp.mpf(0)] * len(forms)
        phase_gradient = [mp.mpf(0)] * len(forms)
        for i, j in edges:
            form_gradient[i] += forms[i] - forms[j]
            form_gradient[j] += forms[j] - forms[i]
            phase_gradient[i] += mp.sin(phases[i] - phases[j])
            phase_gradient[j] += mp.sin(phases[j] - phases[i])

        def energy(x, theta):
            return mp.fsum(
                (x[j] - x[i]) ** 2 / 2 + 1 - mp.cos(theta[j] - theta[i])
                for i, j in edges
            )

        nominal = energy(forms, phases)
        linear = number(epsilon) * mp.fsum(map(abs, form_gradient + phase_gradient))
        remainder = 4 * len(edges) * number(epsilon) ** 2
        exact_upper = nominal + linear + remainder
        assert exact_upper <= number(report.taylor_storage_upper_bound)
        assert number(report.taylor_storage_upper_bound) - exact_upper < mp.mpf("1e-32")
        # The gradient-aligned corner independently exercises every signed
        # first derivative. Its nonlinear remainder obeys the edge Hessian
        # bound without any trajectory, stencil fit or cached gradient.
        corner_forms = tuple(
            x + number(epsilon) * mp.sign(g) for x, g in zip(forms, form_gradient)
        )
        corner_phases = tuple(
            theta + number(epsilon) * mp.sign(g)
            for theta, g in zip(phases, phase_gradient)
        )
        corner = energy(corner_forms, corner_phases)
        assert abs(corner - nominal - linear) <= remainder
        assert corner <= number(report.taylor_storage_upper_bound)


def test_reversed_winding_common_origins_and_source_order_preserve_the_claim(
    source, report
):
    shifted = _assess(
        replace(
            source,
            epi=tuple(x + Q(2, 7) for x in source.epi),
            phase=tuple(theta - Q(9, 11) for theta in source.phase),
        )
    )
    reversed_report = _assess(
        replace(source, phase=tuple(-theta for theta in source.phase))
    )
    order = (8, 0, 5, 3, 6, 9, 1, 7, 4, 2)
    mapped_source = _state(
        epi=source.epi, phase=source.phase, order=order, label=lambda i: f"node:{i}"
    )
    mapped = _assess(mapped_source, cycle=tuple(f"node:{i}" for i in range(5)))
    for candidate in (shifted, reversed_report, mapped):
        assert candidate.whole_window_retention_certified
        assert candidate.full_storage_bounds == report.full_storage_bounds
        assert (
            candidate.retention_margin_lower_bound
            == report.retention_margin_lower_bound
        )
    assert reversed_report.initial_winding == -1
    assert mapped.cycle_indices == tuple(order.index(i) for i in range(5))


def test_live_environment_and_all_cached_fields_are_distinguished(source, report):
    poisoned = replace(
        source,
        storage=I(-1000),
        form_gradient=(),
        phase_rates=(),
        form_rates=(),
        relative_resultant=(),
    )
    rebuilt = _assess(poisoned)
    assert rebuilt.full_storage_bounds == report.full_storage_bounds
    assert rebuilt.edge_phase_speed_upper_bounds == report.edge_phase_speed_upper_bounds
    high = _assess(
        replace(poisoned, epi=source.epi[:5] + (source.epi[5] + 5,) + source.epi[6:])
    )
    assert high.initial_acute_unit_winding_certified
    assert high.full_storage_bounds.lo > report.full_storage_bounds.hi
    assert not high.whole_window_retention_certified
    assert high.reasons == ("strict_finite_retention_margin_not_certified",)


def test_zero_winding_nonacute_or_wide_inputs_are_unavailable_not_dynamical_failure(
    source,
):
    zero = _assess(replace(source, phase=(Q(0),) * 10))
    nonacute = _assess(
        replace(source, phase=tuple(map(Q, ("0", "1.7", "2.85", "4", "5.15"))) * 2)
    )
    wide = _assess(source, source_error_bound=2)
    long = _assess(source, scaled_duration=10)
    for candidate in (zero, nonacute, wide, long):
        assert candidate.status == "unavailable"
        assert not candidate.whole_window_retention_certified
    assert zero.initial_winding == 0 and zero.edge_phase_speed_upper_bounds is None
    assert (
        nonacute.initial_winding == 1
        and not nonacute.initial_acute_unit_winding_certified
    )
    assert wide.initial_winding is None
    assert long.initial_acute_unit_winding_certified


@pytest.mark.parametrize(
    "field,value",
    (
        ("scaled_duration", False),
        ("scaled_duration", True),
        ("scaled_duration", 0),
        ("scaled_duration", -1),
        ("scaled_duration", float("nan")),
        ("scaled_duration", "1"),
        ("source_error_bound", False),
        ("source_error_bound", -1),
        ("source_error_bound", float("inf")),
        ("source_error_bound", "1/4096"),
    ),
)
def test_duration_and_uncertainty_are_admitted_before_arithmetic(source, field, value):
    with pytest.raises((TypeError, ValueError)):
        _assess(source, **{field: value})


@pytest.mark.parametrize("field", ("epi", "phase", "capacity"))
def test_invalid_primitive_coordinates_cannot_use_a_cached_success(source, field):
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, **{field: (False,) + getattr(source, field)[1:]}))


@pytest.mark.parametrize(
    "cycle", ((0, 1, 2, 3), (0, 1, 2, 3, 3), (0, 2, 1, 3, 4), {0, 1, 2, 3, 4})
)
def test_cycle_admission_retains_full_ordered_support(source, cycle):
    with pytest.raises((TypeError, ValueError)):
        _assess(source, cycle=cycle)


def test_sdk_export_keeps_exact_budget_and_scoped_retention_fields(report, tmp_path):
    payload = relational_report_to_dict(report)
    assert payload["report_type"] == "SineCycleRetention"
    assert report.to_dict()["schema"] == "tnfr.sine-cycle-retention.v1"
    assert payload["report"]["source_error_bound"] == {
        "numerator": 1,
        "denominator": 4096,
    }
    assert payload["report"]["whole_window_retention_certified"] is True
    path = tmp_path / "retention.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload

    @dataclass(frozen=True)
    class Opaque:
        value: int

    with pytest.raises(TypeError):
        relational_report_to_dict(
            replace(report, cycle=(Opaque(0),) + report.cycle[1:])
        )
