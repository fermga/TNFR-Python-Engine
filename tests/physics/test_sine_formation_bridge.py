"""Independent static controls for the same-orbit formation existence proof.

The algebraic saddle preparation and its instantaneous full rows are checked
at high precision. No nonlinear trajectory, frozen producer or measured source
is evaluated by these tests; numerical source coordinates remain unavailable.
"""

import json
from copy import copy
from dataclasses import dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tests.physics.test_sine_cycle_barrier import _state
from tests.physics.test_sine_phase_offset_partition import CYCLE_LEAF_EDGES
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_corridor import assess_sine_saddle_formation
from tnfr.sdk import export_to_json, relational_report_to_dict


@pytest.fixture(scope="module")
def anchor():
    # Intentionally not in the signed family and not in the narrow energy
    # range. This captured state anchors the support/law, not a preparation.
    return _state(
        epi=tuple(Q(i * i + 2, 3) for i in range(10)),
        phase=tuple(Q(i - 4, 11) for i in range(10)),
    )


def _assess(source, **changes):
    return assess_sine_saddle_formation(source, **(dict(cycle=range(5)) | changes))


@pytest.fixture(scope="module")
def report(anchor):
    return _assess(anchor)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _inside(value, bound):
    assert _mp(bound.lo) <= value <= _mp(bound.hi)


def _reconstruct(values):
    a, b, c, d = values
    return (a, b, mp.mpf(0), -b, -a, c, d, mp.mpf(0), -d, -c)


def _independent_direction():
    lam = mp.findroot(
        lambda z: 324 * z**4 + 1404 * z**3 + 1527 * z**2 + 61 * z - 15,
        (mp.mpf(".0788"), mp.mpf(".0789")),
    )
    sigma = mp.sqrt(lam)
    b = (4 * lam + 1) / (18 * lam**2 + 43 * lam + 5)
    v = (mp.mpf(1), b, (7 - b) / (6 * lam + 8), (10 * b - 1) / (6 * lam + 8))
    # Direct full-edge Hessian action, not the owner's reduced matrix.
    phase = _reconstruct(v)
    target = tuple((i - 2) * mp.pi / 3 for i in range(5)) * 2
    rows = [mp.mpf(0)] * 10
    for i, j in CYCLE_LEAF_EDGES:
        current = mp.cos(target[j] - target[i]) * (phase[j] - phase[i])
        rows[i] += current
        rows[j] -= current
    form = tuple(rows[i] / (3 if i < 5 else 1) / sigma for i in range(10))
    return sigma, form, phase, target


def _energy(form, phase):
    return mp.fsum(
        (form[j] - form[i]) ** 2 / 2 + 1 - mp.cos(phase[j] - phase[i])
        for i, j in CYCLE_LEAF_EDGES
    )


def _flow(form, phase):
    x_rates, phase_rates = [mp.mpf(0)] * 10, [mp.mpf(0)] * 10
    for i, j in CYCLE_LEAF_EDGES:
        current = mp.sin(phase[j] - phase[i])
        x_rates[i] += current
        x_rates[j] -= current
        contrast = form[i] - form[j]
        phase_rates[i] += contrast
        phase_rates[j] -= contrast
    return tuple(x_rates[i] / (3 if i < 5 else 1) for i in range(10)) + tuple(
        phase_rates[i] / (3 if i < 5 else 1) for i in range(10)
    )


def test_correlated_recipe_energy_and_derivative_follow_full_nodal_rows(report):
    with mp.workdps(110):
        sigma, form_direction, phase_direction, target = _independent_direction()
        q = mp.fsum(
            (form_direction[j] - form_direction[i]) ** 2 for i, j in CYCLE_LEAF_EDGES
        )
        phase_q = -mp.fsum(
            mp.cos(target[j] - target[i])
            * (phase_direction[j] - phase_direction[i]) ** 2
            for i, j in CYCLE_LEAF_EDGES
        )
        assert abs(q - phase_q) < mp.mpf("1e-100")
        _inside(q, report.phase_direction_quadratic_bounds)
        assert 1 < q < mp.mpf(6) / 5
        momentum = mp.fsum(
            w * form_direction[i] for w, i in zip((6, 3, 2, 1), (0, 1, 5, 6))
        )
        _inside(momentum, report.momentum_direction_bounds)
        assert momentum > mp.mpf(53) / 10

        epsilon = _mp(report.epsilon)
        forms = tuple(3 * epsilon * value for value in form_direction)
        phases = tuple(
            value - epsilon * direction
            for value, direction in zip(target, phase_direction)
        )
        excess = _energy(forms, phases) - mp.mpf(7) / 2
        assert abs(excess - 4 * q * epsilon**2) <= mp.mpf(40) * epsilon**3 / 3
        assert _mp(report.initial_storage_excess_lower_bound) <= excess
        assert excess <= _mp(report.initial_storage_excess_upper_bound)
        assert 0 < excess < 5 * epsilon**2

        # J(3X,-v)=(-sigma X,3sigma v); the phase row remains exactly
        # linear, while each full form-row remainder is at most2epsilon².
        rows = _flow(forms, phases)
        assert (
            max(abs(rows[i] + epsilon * sigma * form_direction[i]) for i in range(10))
            <= 2 * epsilon**2
        )
        assert max(
            abs(rows[10 + i] - 3 * epsilon * sigma * phase_direction[i])
            for i in range(10)
        ) < mp.mpf("1e-100")


def test_opposite_time_endpoints_use_form_reversal_and_one_correlated_direction(report):
    with mp.workdps(95):
        sigma, form, _, _ = _independent_direction()
        momentum = mp.fsum(w * form[i] for w, i in zip((6, 3, 2, 1), (0, 1, 5, 6)))
        growth = mp.exp(3 * sigma)
        error = _mp(report.normalized_local_remainder_bound)
        outer_phase = growth - 2 / growth - error
        outer_momentum = momentum * (growth + 2 / growth) - 12 * error
        inner_phase = 1 / growth - 2 * growth + error
        # R reverses form only: this negative momentum is not the original
        # positive momentum on the negative-time endpoint.
        inner_momentum = -momentum * (1 / growth + 2 * growth) + 12 * error
        assert (
            outer_phase
            >= _mp(report.outer_phase_displacement_coefficient_lower_bound)
            > 1
        )
        assert (
            outer_momentum
            >= _mp(report.outer_momentum_coefficient_lower_bound)
            > mp.mpf(33) / 2
        )
        assert (
            inner_phase
            <= _mp(report.inner_phase_displacement_coefficient_upper_bound)
            < -1
        )
        assert (
            -inner_momentum
            >= _mp(report.inner_negative_momentum_coefficient_lower_bound)
            > 23
        )
    assert report.normalized_local_remainder_bound < Q(1, 512)
    assert report.momentum_exit_allowance_coefficient_upper_bound <= 264
    assert report.outer_momentum_gate_margin > 0
    assert report.inner_momentum_gate_margin > 0
    assert report.outer_force_lower_bound >= report.epsilon / 3
    assert report.inner_force_lower_bound >= report.epsilon
    assert report.connection_time_upper_bound == 64 / report.epsilon
    assert (
        42 / report.epsilon + 6 + 14 / report.epsilon
        < report.connection_time_upper_bound
    )


def test_uniform_band_controls_full_width_without_assigning_symmetry_to_errors(report):
    band = report.retention_band
    assert band.theorem_certified
    assert band.backward_band_residence_lower_bound >= Q(7, 10)
    assert band.forward_band_residence_lower_bound >= Q(7, 10)
    assert band.scaled_dwell_lower_bound > 1
    assert band.cycle_gap_margin_lower_bound >= Q(1, 12000)
    assert report.full_storage_upper_bound < band.full_storage_upper_bound
    assert report.target_radius == Q(1, 2**18)
    assert report.retained_scaled_duration == 1
    assert 18 * report.target_radius < band.cycle_gap_margin_lower_bound
    assert band.independent_target_ball_implication_certified
    assert report.formation_existence_certified
    assert report.independent_source_ball_existence_certified
    # Store the positive expression r=p*exp(-a). Its enormous exponent is
    # evidence of a formal existence radius, not permission to export zero.
    assert report.source_radius_prefactor == Q(1, 2**19) > 0
    assert report.source_radius_exponent == 128 / report.epsilon > 0
    assert 2 * report.connection_time_upper_bound == report.source_radius_exponent
    assert report.source_radius_prefactor == report.target_radius / 2
    assert not report.numerical_source_center_available


def test_supplied_state_is_only_a_readmitted_law_support_anchor(anchor, report):
    assert not report.captured_source_formation_certified
    assert not report.captured_source_retention_certified
    poisoned = replace(
        anchor, storage=I(-999), form_rates=(), phase_rates=(), form_gradient=()
    )
    rebuilt = _assess(poisoned)
    changed = _assess(replace(poisoned, epi=(Q(1000),) * 10, phase=(Q(0),) * 10))
    for candidate in (rebuilt, changed):
        assert candidate.formation_existence_certified
        assert not candidate.captured_source_formation_certified
        assert (
            candidate.initial_storage_excess_upper_bound
            == report.initial_storage_excess_upper_bound
        )
        assert (
            candidate.connection_time_upper_bound == report.connection_time_upper_bound
        )


def test_tiny_exact_preparation_keeps_a_positive_symbolic_radius_without_underflow(
    anchor,
):
    epsilon = Q(1, 2**512)
    report = _assess(anchor, epsilon=epsilon)
    assert report.formation_existence_certified
    assert report.independent_source_ball_existence_certified
    assert 0 < report.initial_storage_excess_lower_bound
    assert report.full_storage_upper_bound - Q(7, 2) == 5 * epsilon**2
    assert report.source_radius_prefactor > 0
    assert report.source_radius_exponent == 2**519
    payload = report.to_dict()["report"]
    assert payload["source_radius_exponent"] == {
        "numerator": 2**519,
        "denominator": 1,
    }
    assert (
        payload["source_radius_recipe"]
        == "source_radius_prefactor*exp(-source_radius_exponent)"
    )


@pytest.mark.parametrize(
    "epsilon",
    (False, True, Q(0), Q(-1), Q(1, 2**31), float("nan"), float("inf"), "1/4294967296"),
)
def test_invalid_epsilon_is_not_coerced_into_the_certified_family(anchor, epsilon):
    with pytest.raises((TypeError, ValueError)):
        _assess(anchor, epsilon=epsilon)


@pytest.mark.parametrize("field", ("epi", "phase", "capacity"))
def test_invalid_anchor_primitives_cannot_hide_behind_a_hypothetical_target(
    anchor, field
):
    values = list(getattr(anchor, field))
    values[0] = True if field == "capacity" else False
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(anchor, **{field: tuple(values)}))


@pytest.mark.parametrize("field", ("epi_weight", "phase_weight", "storage_scale"))
def test_boolean_model_cannot_become_a_conservative_anchor(anchor, field):
    model = copy(anchor.reference_model)
    object.__setattr__(model, field, False if field == "epi_weight" else True)
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(anchor, reference_model=model))


def test_node_order_and_labels_preserve_theorem_and_atomic_exact_export(
    anchor, report, tmp_path
):
    order = (8, 2, 5, 0, 7, 4, 1, 6, 9, 3)
    source = _state(
        epi=anchor.epi, phase=anchor.phase, order=order, label=lambda i: ("node", i)
    )
    changed = _assess(source, cycle=tuple(("node", i) for i in range(5)))
    assert changed.formation_existence_certified
    assert (
        changed.phase_direction_quadratic_bounds
        == report.phase_direction_quadratic_bounds
    )
    assert changed.connection_time_upper_bound == report.connection_time_upper_bound
    payload = relational_report_to_dict(changed)
    assert payload["report_type"] == "SineSaddleFormation"
    assert payload["report"]["source_radius_exponent"] == {
        "numerator": report.source_radius_exponent.numerator,
        "denominator": report.source_radius_exponent.denominator,
    }
    path = tmp_path / "formation-existence.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert payload["report"]["numerical_source_center_available"] is False

    @dataclass(frozen=True)
    class Opaque:
        value: int

    nested = replace(
        changed.retention_band, cycle=(Opaque(0),) + changed.retention_band.cycle[1:]
    )
    with pytest.raises(TypeError):
        relational_report_to_dict(replace(changed, retention_band=nested))
