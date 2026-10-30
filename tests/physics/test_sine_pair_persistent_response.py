"""Fine-law and full-state controls for a persistent pattern's finite response."""

from fractions import Fraction as Q
from math import factorial
from typing import get_type_hints

import mpmath
import pytest

from tests.physics._sine_pair_oracles import (
    EDGES,
    NEIGHBORS,
    PAIRS,
    _contains,
    _energy,
    _mp,
    _norm,
)
from tnfr.physics import relational_sine_pair as pair
from tnfr.physics import relational_sine_scale as scale
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

DELTA = (Q(3, 100), Q(1, 25))
HORIZON = Q(1, 2)
ERROR = Q(1, 10**6)
RADIUS = Q(1, 8)


def _assess(
    delta=DELTA, horizon=HORIZON, form=ERROR, phase=ERROR, readout=ERROR, radius=RADIUS
):
    return pair.assess_sine_pair_persistent_response(
        delta_bounds=delta,
        horizon_tau=horizon,
        form_error_bound=form,
        phase_error_bound=phase,
        readout_error_bound=readout,
        radius=radius,
    )


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module")
def frozen():
    return _assess()


def _center(mp, delta, orientation):
    alpha = 2 * mp.pi / 5
    delta = _mp(mp, delta)
    forms = [mp.mpf(0) for _ in range(10)]
    phases = [mp.mpf(i // 2) * alpha for i in range(10)]
    if orientation == 0:
        phases[0], phases[1] = delta, -delta
    else:
        amplitude = mp.sqrt(2 * mp.cos(alpha) * (1 - mp.cos(delta)))
        forms[0], forms[1] = amplitude, -amplitude
    return forms, phases


def _phase_row(values):
    return [
        value - sum(values[j] for j in NEIGHBORS[i]) / 4
        for i, value in enumerate(values)
    ]


def _fine_derivatives(mp, forms, phases):
    """Differentiate the ten supplied rows, without a trajectory integrator."""
    velocity = _phase_row(forms)
    first = [
        sum(mp.sin(phases[j] - phases[i]) for j in row) / 4
        for i, row in enumerate(NEIGHBORS)
    ]
    acceleration = _phase_row(first)
    second = [
        sum(mp.cos(phases[j] - phases[i]) * (velocity[j] - velocity[i]) for j in row)
        / 4
        for i, row in enumerate(NEIGHBORS)
    ]
    third = [
        sum(
            -mp.sin(phases[j] - phases[i]) * (velocity[j] - velocity[i]) ** 2
            + mp.cos(phases[j] - phases[i]) * (acceleration[j] - acceleration[i])
            for j in row
        )
        / 4
        for i, row in enumerate(NEIGHBORS)
    ]
    return first, second, third, velocity


def _relative_norm_squared(forms, phases, alpha):
    mean_form = sum(forms) / 10
    errors = [theta - (i // 2) * alpha for i, theta in enumerate(phases)]
    phase_origin = sum(errors) / 10
    return sum((x - mean_form) ** 2 for x in forms) + sum(
        (error - phase_origin) ** 2 for error in errors
    )


@pytest.mark.parametrize("delta", (DELTA[0], sum(DELTA) / 2, DELTA[1]))
def test_nominal_energy_and_means_follow_from_all_twenty_edges(mp, frozen, delta):
    alpha = 2 * mp.pi / 5
    baseline = _energy(
        mp, [mp.mpf(0)] * 10, [mp.mpf(i // 2) * alpha for i in range(10)]
    )
    centers = [_center(mp, delta, orientation) for orientation in range(2)]
    energies = [_energy(mp, *center) - baseline for center in centers]
    assert abs(energies[0] - energies[1]) < mp.mpf("1e-90")
    assert energies[0] > 0
    for forms, phases in centers:
        assert sum(forms) == 0
        for a, (left, right) in enumerate(PAIRS):
            assert forms[left] + forms[right] == 0
            assert abs((phases[left] + phases[right]) / 2 - a * alpha) < mp.mpf("1e-90")
    for energy in energies:
        _contains(mp, frozen.nominal_excess_storage_bounds, energy)
    amplitude = centers[1][0][0]
    _contains(mp, frozen.nominal_internal_form_amplitude_bounds, amplitude)
    _contains(mp, frozen.nominal_internal_form_squared_bounds, amplitude**2)
    _contains(mp, frozen.target_angle_bounds, alpha)
    _contains(mp, frozen.target_cosine_bounds, mp.cos(alpha))
    _contains(mp, frozen.target_sine_bounds, mp.sin(alpha))


@pytest.mark.parametrize("delta", DELTA)
def test_original_rows_distinguish_internal_allocation_at_matched_energy(
    mp, frozen, delta
):
    angle = _mp(mp, delta)
    alpha = 2 * mp.pi / 5
    c, s = mp.cos(alpha), mp.sin(alpha)
    derivatives = []
    for orientation in range(2):
        forms, phases = _center(mp, delta, orientation)
        rows = _fine_derivatives(mp, forms, phases)
        derivatives.append(rows)
        # The four nominal synchronized tips are invariant. This says nothing
        # about arbitrary full-state box perturbations or five active pairs.
        for left, right in PAIRS[1:]:
            assert abs(rows[0][left] - rows[0][right]) < mp.mpf("1e-90")
            assert abs(rows[3][left] - rows[3][right]) < mp.mpf("1e-90")
        assert abs(sum(rows[0])) < mp.mpf("1e-90")
    a, b = derivatives
    assert abs(a[0][0] + c * mp.sin(angle)) < mp.mpf("1e-90")
    assert abs(a[0][1] - c * mp.sin(angle)) < mp.mpf("1e-90")
    assert max(map(abs, a[3])) == 0
    rate = (a[0][2] + a[0][3]) / 2
    assert abs(rate - s * (1 - mp.cos(angle)) / 2) < mp.mpf("1e-90")
    _contains(mp, frozen.nominal_receiver_rate_bounds, rate)
    assert max(map(abs, b[0])) < mp.mpf("1e-90")
    amplitude = _center(mp, delta, 1)[0][0]
    assert b[3][0] == amplitude and b[3][1] == -amplitude
    assert abs(b[1][0] + c * amplitude) < mp.mpf("1e-90")
    assert abs(b[1][1] - c * amplitude) < mp.mpf("1e-90")
    assert abs((b[1][2] + b[1][3]) / 2) < mp.mpf("1e-90")
    assert abs((b[2][2] + b[2][3]) / 2 - s * amplitude**2 / 2) < mp.mpf("1e-90")


def test_zero_amplitude_and_pair_swap_are_fine_law_controls(mp):
    for orientation in range(2):
        zero = _fine_derivatives(mp, *_center(mp, Q(0), orientation))
        assert max(abs(value) for row in zero for value in row) < mp.mpf("1e-90")
        forms, phases = _center(mp, DELTA[1], orientation)
        original = _fine_derivatives(mp, forms, phases)
        forms[0], forms[1] = forms[1], forms[0]
        phases[0], phases[1] = phases[1], phases[0]
        swapped = _fine_derivatives(mp, forms, phases)
        for first, second in zip(original[:3], swapped[:3], strict=True):
            assert abs((first[2] + first[3]) - (second[2] + second[3])) < mp.mpf(
                "1e-90"
            )
            # The target-centered reflection symmetry predicts opposite
            # neighboring mean responses, not disappearance of all currents.
            assert abs(first[2] + first[3] + first[8] + first[9]) < mp.mpf("1e-90")


def test_barrier_uses_the_actual_fine_laplacian_and_strict_margins(mp, frozen):
    laplacian = mp.matrix(10)
    for i, row in enumerate(NEIGHBORS):
        laplacian[i, i] = len(row)
        for j in row:
            laplacian[i, j] = -1
    eigenvalues = mp.eigsy(laplacian, eigvals_only=True)
    gap = eigenvalues[1]
    assert abs(eigenvalues[0]) < mp.mpf("1e-90")
    assert abs(gap - (5 - mp.sqrt(5))) < mp.mpf("1e-90")
    _contains(mp, frozen.spectral_gap_bounds, gap)
    angle = 2 * mp.pi / 5 + mp.sqrt(2) * _mp(mp, RADIUS)
    _contains(mp, frozen.radius_angle_bounds, angle)
    _contains(mp, frozen.acute_radius_margin_bounds, mp.pi / 2 - angle)
    expected = gap * mp.cos(angle) / 2
    assert 0 < expected - _mp(mp, frozen.coercivity_lower_bound) < mp.mpf("1e-30")
    assert (
        0
        <= frozen.coercivity_lower_bound * RADIUS**2 - frozen.barrier_lower_bound
        < Q(1, 2**128)
    )
    for index in range(2):
        assert frozen.initial_radius_margin_bounds[index].contains(
            RADIUS**2 - frozen.initial_norm_squared_upper_bounds[index]
        )
        assert frozen.storage_barrier_margin_bounds[index].contains(
            frozen.barrier_lower_bound
            - frozen.initial_excess_storage_upper_bounds[index]
        )
        assert frozen.initial_radius_margin_bounds[index].lo > 0
        assert frozen.storage_barrier_margin_bounds[index].lo > 0
    assert frozen.persistence_certified_by_preparation == (True, True)


def test_box_bounds_include_all_coordinates_and_exact_gradient_terms(mp, frozen):
    alpha = 2 * mp.pi / 5
    base_forms, base_phases = _center(mp, Q(0), 0)
    baseline = _energy(mp, base_forms, base_phases)
    rho = _mp(mp, ERROR)
    for orientation in range(2):
        forms, phases = _center(mp, DELTA[1], orientation)
        relative_phases = [value - (i // 2) * alpha for i, value in enumerate(phases)]
        norm_bound = (_norm(mp, forms) + mp.sqrt(10) * rho) ** 2 + (
            _norm(mp, relative_phases) + mp.sqrt(10) * rho
        ) ** 2
        assert (
            0
            <= _mp(mp, frozen.initial_norm_squared_upper_bounds[orientation])
            - norm_bound
            < mp.mpf("1e-30")
        )
        form_gradient = [4 * value for value in _phase_row(forms)]
        form_rate = _fine_derivatives(mp, forms, phases)[0]
        phase_gradient = [-4 * value for value in form_rate]
        # Each of twenty edges has |error difference| <=2rho. The original
        # quadratic form and cosine Hessian each contribute (2rho)^2/2.
        quadratic = len(EDGES) * (2 * rho) ** 2
        energy_bound = (
            _energy(mp, forms, phases)
            - baseline
            + rho * sum(abs(value) for value in form_gradient + phase_gradient)
            + quadratic
        )
        assert (
            0
            <= _mp(mp, frozen.initial_excess_storage_upper_bounds[orientation])
            - energy_bound
            < mp.mpf("1e-30")
        )
        perturbations = []
        for coordinate in range(20):
            values = [mp.mpf(0)] * 20
            values[coordinate] = rho if coordinate % 2 else -rho
            perturbations.append(values)
        perturbations.append([rho if i % 3 else -rho for i in range(20)])
        for errors in perturbations:
            changed_forms = [value + errors[i] for i, value in enumerate(forms)]
            changed_phases = [value + errors[i + 10] for i, value in enumerate(phases)]
            assert _relative_norm_squared(changed_forms, changed_phases, alpha) <= _mp(
                mp, frozen.initial_norm_squared_upper_bounds[orientation]
            )
            assert _energy(mp, changed_forms, changed_phases) - baseline <= _mp(
                mp, frozen.initial_excess_storage_upper_bounds[orientation]
            )
        # A permitted perturbation leaves the nominal synchronized-tip
        # manifold: the certificate must use its full-state preparation bound.
        changed_forms = forms.copy()
        changed_forms[2] += rho
        assert changed_forms[2] != changed_forms[3]


def _nominal_majorants(report):
    horizon = report.horizon_tau
    # Invariant nominal tips leave one internal oscillator with |d''|<=|d|.
    # Its Volterra kernel integral is h²/2, for both position and velocity data.
    ratio = horizon**2 / factorial(2)
    finite_series = sum(ratio**k for k in range(7))
    amplification = finite_series / (1 - ratio**7)
    selected_fraction = Q(sum(j in PAIRS[0] for j in NEIGHBORS[2]), len(NEIGHBORS[2]))
    forcing = tuple(
        selected_fraction * size * amplification**2 / 2
        for size in (
            report.delta_bounds[1] ** 2,
            report.nominal_internal_form_squared_bounds.hi,
        )
    )
    # Both normalized phase and mean-pressure gap differences have row norm2.
    phase_norm = Q(1) + sum(Q(1, len(NEIGHBORS[2])) for _ in NEIGHBORS[2])
    feedback = 2 * phase_norm
    growth = []
    for order, source in zip((1, 3), forcing, strict=True):
        # Integral_0^t (t-s)*s^order ds = t^(order+2)/((order+1)(order+2)).
        feedback_ratio = feedback * horizon**2 / ((order + 1) * (order + 2))
        growth.append(source / order / (1 - feedback_ratio))
    remainder_a = (phase_norm * growth[0] + forcing[0]) * horizon**3 / 3
    remainder_b = growth[1] * horizon**3
    return forcing, tuple(growth), (remainder_a, remainder_b)


@pytest.mark.parametrize("horizon", (Q(1, 10), HORIZON, Q(2, 3)))
def test_nominal_bounds_integrate_internal_and_mean_feedback_separately(horizon):
    report = _assess(horizon=horizon)
    forcing, growth, remainder = _nominal_majorants(report)
    assert report.nominal_mean_forcing_upper_bounds == forcing
    assert report.nominal_mean_growth_upper_bounds == growth
    assert report.nominal_response_remainder_upper_bounds == remainder
    assert remainder[0] > remainder[1] > 0


@pytest.mark.parametrize("form,phase", ((0, 0), (ERROR, 0), (0, ERROR), (ERROR, ERROR)))
def test_full_state_preparation_majorant_contains_cosh_sinh_flow(mp, form, phase):
    report = _assess(form=form, phase=phase)
    h = HORIZON
    # Fine-row differences give comparison matrix [[0,2],[2,0]].
    even_terms = [(2 * h) ** (2 * k) / factorial(2 * k) for k in range(1, 8)]
    odd_terms = [(2 * h) ** (2 * k + 1) / factorial(2 * k + 1) for k in range(1, 8)]
    even_ratio = (2 * h) ** 2 / (1 * 2)
    odd_ratio = (2 * h) ** 2 / (2 * 3)
    assert all(term <= even_ratio**k for k, term in enumerate(even_terms, 1))
    assert all(term <= 2 * h * odd_ratio**k for k, term in enumerate(odd_terms, 1))
    expected = Q(form) / (1 - even_ratio) + 2 * h * Q(phase) / (1 - odd_ratio)
    assert report.preparation_error_bound == expected
    assert report.full_dimensional_preparation == (form > 0 and phase > 0)
    exact_comparison = _mp(mp, Q(form)) * mp.cosh(2 * _mp(mp, h)) + _mp(
        mp, Q(phase)
    ) * mp.sinh(2 * _mp(mp, h))
    assert exact_comparison <= _mp(mp, report.preparation_error_bound)


def test_frozen_joint_verdict_requires_response_and_both_persistence_margins(frozen):
    assert frozen.preparation_order == ("phase_split", "form_split")
    assert frozen.full_dimensional_preparation
    assert frozen.response_separation_certified
    assert frozen.recorded_difference_bounds.lo > Q(1, 100000)
    assert frozen.status == "certified_persistent_response"
    assert not frozen.unavailable_reasons
    _, _, remainders = _nominal_majorants(frozen)
    centers = (
        (
            HORIZON * frozen.nominal_receiver_rate_bounds.lo,
            HORIZON * frozen.nominal_receiver_rate_bounds.hi,
        ),
        (Q(0), Q(0)),
    )
    extra = frozen.preparation_error_bound + ERROR
    for center, remainder, nominal, recorded in zip(
        centers,
        remainders,
        frozen.nominal_readout_bounds,
        frozen.recorded_readout_bounds,
        strict=True,
    ):
        assert nominal.contains(center[0] - remainder)
        assert nominal.contains(center[1] + remainder)
        assert recorded.contains(center[0] - remainder - extra)
        assert recorded.contains(center[1] + remainder + extra)
    # The joint contrast uses original endpoints, not twice-rounded marginals.
    radius = sum(remainders) + 2 * extra
    assert frozen.recorded_difference_bounds.contains(centers[0][0] - radius)
    assert frozen.recorded_difference_bounds.contains(centers[0][1] + radius)


def test_missing_response_does_not_erase_valid_geometry():
    report = _assess(readout=Q(1, 100))
    assert report.persistence_certified_by_preparation == (True, True)
    assert not report.response_separation_certified
    assert report.status == "unavailable"
    assert (
        "recorded_response_difference_not_strictly_positive"
        in report.unavailable_reasons
    )


@pytest.mark.parametrize(
    "kwargs,reason",
    (
        ({"radius": Q(1, 2)}, "strict_acute_radius_not_certified"),
        ({"radius": Q(1, 100)}, "strict_initial_radius_not_certified"),
        ({"phase": Q(1, 200)}, "strict_excess_storage_barrier_not_certified"),
    ),
)
def test_each_geometric_obligation_has_an_explicit_failure(kwargs, reason):
    report = _assess(**kwargs)
    assert report.status == "unavailable"
    assert any(
        reason in reasons
        for reasons in report.persistence_unavailable_reasons_by_preparation
    )
    assert not all(report.persistence_certified_by_preparation)


def test_tiny_horizon_and_touching_response_are_not_positive_certificates():
    tiny = _assess(horizon=Q(1, 2**300), form=0, phase=0, readout=0)
    assert tiny.persistence_certified_by_preparation == (True, True)
    assert tiny.recorded_difference_bounds.lo <= 0
    assert not tiny.response_separation_certified
    zero_readout = _assess(readout=0)
    touching = _assess(readout=zero_readout.recorded_difference_bounds.lo / 2)
    assert touching.recorded_difference_bounds.lo <= 0
    assert not touching.response_separation_certified


@pytest.mark.parametrize("field", ("horizon", "form", "phase", "readout", "radius"))
@pytest.mark.parametrize(
    "bad", (True, False, float("nan"), float("inf"), -1, "0", 1j, object())
)
def test_original_scalar_admission(field, bad):
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "bad",
    (
        (),
        (0,),
        (1, 2, 3),
        (0, Q(1, 25)),
        (Q(-1, 10), Q(1, 25)),
        DELTA[::-1],
        (Q(1, 2), Q(3, 2)),
        (True, Q(1, 25)),
        (Q(3, 100), float("nan")),
        "0.03,0.04",
        None,
    ),
)
def test_delta_sequence_domain(bad):
    with pytest.raises((TypeError, ValueError)):
        _assess(delta=bad)


@pytest.mark.parametrize(
    "kwargs", ({"horizon": 0}, {"horizon": Q(3, 4)}, {"radius": 0})
)
def test_positive_time_radius_and_hyperbolic_majorant_domain(kwargs):
    with pytest.raises(ValueError):
        _assess(**kwargs)


def test_one_pass_sequence_and_exact_tiny_errors_are_retained(frozen):
    assert _assess(delta=(value for value in DELTA)) == frozen
    tiny = Q(1, 2**1100)
    report = _assess(form=tiny, phase=tiny, readout=tiny)
    assert (
        report.form_error_bound
        == report.phase_error_bound
        == report.readout_error_bound
        == tiny
    )
    assert report.preparation_error_bound > 0
    assert report.full_dimensional_preparation
    tiny_delta = _assess(delta=(tiny, tiny), form=0, phase=0, readout=0)
    assert tiny_delta.delta_bounds == (tiny, tiny)
    assert tiny_delta.status == "unavailable"


def test_nonzero_nonrational_scalar_lost_in_materialization_is_rejected():
    class LostReal(float):
        def __float__(self):
            return 0.0

    with pytest.raises(ValueError, match="underflows"):
        _assess(form=LostReal(1.0))


def test_public_compatibility_and_exact_sdk_export(tmp_path, frozen):
    for name in ("SinePairPersistentResponse", "assess_sine_pair_persistent_response"):
        assert getattr(pair, name) is getattr(scale, name)
        assert name in pair.__all__ and name in scale.__all__
        assert getattr(pair, name).__module__ == pair.__name__
        get_type_hints(getattr(pair, name))
    direct = frozen.to_dict()
    assert direct["schema"] == "tnfr.sine-pair-persistent-response.v1"
    exported = relational_report_to_dict(frozen)
    assert exported["report_type"] == "SinePairPersistentResponse"
    assert exported["report"] == direct["report"]
    assert exported["report"]["horizon_tau"] == {"numerator": 1, "denominator": 2}
    target = tmp_path / "persistent-response.json"
    export_to_json(exported, target)
    assert json_loads(target.read_text(encoding="utf-8")) == exported
