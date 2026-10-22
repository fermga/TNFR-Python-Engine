"""Independent own-law source transport, storage, and defect controls."""

import importlib
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _constitutive_receiver_source as owner

_ERROR = Q(1, 10**10)
_LIMIT = Q(1, 10000)
_RAMP = Q(2046, 9) * Q(355, 113) ** 2


@pytest.fixture(scope="module", autouse=True)
def no_other_source_verdict_or_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("own-law source imported a sine verdict or trajectory")

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        for module_name, function in (
            ("_organized_receiver_mechanism", "_assess_organized_receiver_mechanism"),
            (
                "_organized_receiver_slow_admission",
                "_assess_organized_receiver_slow_admission",
            ),
            ("_organized_joint_acquisition", "_assess_organized_joint_acquisition"),
        ):
            module = importlib.import_module("tnfr.physics." + module_name)
            patch.setattr(module, function, forbidden)
            if hasattr(owner, function):
                patch.setattr(owner, function, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _assess(**changes):
    inputs = dict(
        deformation_bound=_LIMIT,
        form_error_bound=_ERROR,
        phase_error_bound=_ERROR,
    )
    return owner._assess_constitutive_receiver_source(**(inputs | changes))


@pytest.fixture(scope="module")
def reports():
    return {
        size: _assess(deformation_bound=size) for size in (Q(0), _LIMIT / 2, _LIMIT)
    }


def _edges():
    return tuple(
        (9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in range(3) for j in range(9)
    ) + tuple(
        (9 * a + j, 9 * b + (j + offset) % 9, Q(1, 2))
        for a, b in ((0, 1), (1, 2), (2, 0))
        for j in range(9)
        for offset in (0, 1)
    )


def _potential(mp, angle, deformation):
    return (
        1 - mp.cos(angle) + deformation * mp.cos(angle) * (1 - mp.cos(9 * angle)) ** 2
    )


def _phase_geometry(mp, phases, deformation):
    laplacian, current, jacobian = mp.zeros(27), mp.zeros(27, 1), mp.zeros(27)
    potential = mp.mpf(0)
    for a, b, weight in _edges():
        weight = _mp(mp, weight)
        angle = phases[b] - phases[a]

        def own(value):
            return _potential(mp, value, deformation)

        flow = weight * mp.diff(own, angle) / 4
        derivative = weight * mp.diff(own, angle, 2) / 4
        laplacian[a, a] += weight
        laplacian[b, b] += weight
        laplacian[a, b] -= weight
        laplacian[b, a] -= weight
        current[a] += flow
        current[b] -= flow
        jacobian[a, a] -= derivative
        jacobian[b, b] -= derivative
        jacobian[a, b] += derivative
        jacobian[b, a] += derivative
        potential += weight * own(angle)
    return laplacian / 4, current, jacobian, potential


@pytest.mark.parametrize("size", (Q(0), _LIMIT / 2, _LIMIT))
def test_uniform_source_bounds_match_independent_high_precision_formulas(
    size, reports, mp
):
    report = reports[size]
    error, ramp, e = _mp(mp, _ERROR), _mp(mp, _RAMP), _mp(mp, size)
    gamma, norm = 1 / (1023 * mp.pi), mp.sqrt(27)
    epsilon = gamma**2
    amplitude, stiffness = 1 + mp.mpf(57) * e / 2, 2 * (1 + 328 * e)
    forcing, gap = norm * amplitude, mp.mpf("0.1")
    initial = gamma * (ramp * mp.sqrt(120) + norm * error)
    shift = mp.sqrt(120) * abs(gamma * ramp - 2 * mp.pi / 9) + norm * error * (
        1 + gamma
    )
    layer = mp.exp(-20) * initial + 10 * epsilon * forcing
    release = mp.exp(-40) * initial + 10 * epsilon * forcing
    gradient = ramp * mp.sqrt(mp.mpf(2277) / 2)
    initial_defect = (
        gradient / (4 * gamma) + 2 * norm * error / gamma + stiffness * norm * error
    )
    damping = gap - epsilon * stiffness
    floor = epsilon * stiffness * forcing / damping
    defect = mp.exp(-400 * damping) * initial_defect + floor
    expected = (
        (report.scaled_initial_norm_upper_bound, initial),
        (report.layer_scaled_form_norm_upper_bound, layer),
        (
            report.layer_phase_error_norm_upper_bound,
            shift + layer + 200 * epsilon * forcing,
        ),
        (report.release_scaled_form_norm_upper_bound, release),
        (report.release_form_norm_upper_bound, release / gamma),
        (
            report.release_phase_error_norm_upper_bound,
            shift + release + 400 * epsilon * forcing,
        ),
        (
            report.bridge_phase_error_norm_upper_bound,
            shift + layer + 400 * epsilon * forcing,
        ),
        (report.initial_defect_norm_upper_bound, initial_defect),
        (report.asymptotic_defect_upper_bound, floor),
        (report.uniform_velocity_defect_upper_bound, defect),
        (report.uniform_form_rate_norm_upper_bound, gamma * defect),
        (
            report.early_receiver_edge_upper_bound,
            4 * gamma * ramp
            + 2 * error
            + 4 * gamma * error
            + 1600 * epsilon * amplitude,
        ),
    )
    for actual, value in expected:
        assert value <= _mp(mp, actual) < value + mp.mpf("1e-20")
    assert 0 < _mp(mp, report.damping_lower_bound) <= damping
    assert report.edge_current_absolute_upper_bound == 1 + Q(57, 2) * size
    assert report.global_phase_lipschitz_bound == 2 * (1 + 328 * size)
    for time in (400, 401, 800, 10_000):
        later_defect = initial_defect * mp.exp(-damping * time) + floor * (
            1 - mp.exp(-damping * time)
        )
        assert later_defect < _mp(mp, report.uniform_velocity_defect_upper_bound)


def test_full_interval_retains_every_source_direction_and_fixed_release(reports):
    report = reports[_LIMIT]
    assert report.nodes == tuple(range(27))
    assert report.weighted_edges == _edges()
    assert report.weighted_degrees == (Q(4),) * 27
    assert report.held_capacities == (Q(1),) * 27
    assert report.initial_constituent_error_sums == (Q(0),) * 3
    assert report.relative_preparation_dimension == 48
    assert report.phase_mean_leaf_dimension == 26
    assert report.deformation_interval == (-_LIMIT, _LIMIT)
    assert report.initial_windings == (0, 0, 0)
    assert report.layer_windings == report.release_windings == (1, 1, 0)
    assert report.layer_time == 200 and report.release_time == 400
    assert report.early_receiver_zero_winding_certified
    assert report.donor_release_bridge_certified
    assert (
        report.release_phase_budget_certified and report.release_form_budget_certified
    )
    assert report.uniform_defect_certified
    assert report.release_phase_margin > 0
    assert report.release_form_margin > 0
    assert report.velocity_defect_margin > 0
    assert report.status == "certified_uniform_constitutive_source"
    assert not report.unresolved_conditions
    assert not report.trajectory_evaluated and not report.first_order_exact_closure
    assert report.eventual_receiver_acquisition == "unresolved"
    assert report.eventual_donor_retention == "unresolved"
    assert report.joined_capture == "unresolved"


def test_complete_source_cost_keeps_contacts_and_own_quartic_error(reports, mp):
    report = reports[_LIMIT]
    forms = tuple(_RAMP * (node % 9 - 4) if node < 18 else Q(0) for node in range(27))
    gradient = [Q(0)] * 27
    internal = contact = Q(0)
    for a, b, weight in _edges():
        difference = forms[a] - forms[b]
        gradient[a] += weight * difference
        gradient[b] -= weight * difference
        if a // 9 == b // 9:
            internal += weight * difference**2 / 2
        else:
            contact += weight * difference**2 / 2
    assert report.nominal_initial_forms == forms
    assert report.nominal_initial_form_gradient == tuple(gradient)
    assert report.nominal_initial_internal_form_storage == internal == 72 * _RAMP**2
    assert report.nominal_initial_contact_form_storage == contact == 78 * _RAMP**2
    assert report.nominal_initial_form_storage == internal + contact == 150 * _RAMP**2
    assert sum(map(abs, gradient)) == report.initial_form_gradient_l1 == 117 * _RAMP
    assert sum(value**2 for value in gradient) == _RAMP**2 * Q(2277, 2)
    assert (
        report.initial_potential_deformation_upper_bound
        == 216 * 9**4 * _LIMIT * _ERROR**4
    )
    # A nonsymmetric admissible source retains independently signed form and
    # phase errors, with all initial component means exactly zero.
    phase = mp.matrix([_mp(mp, _ERROR) * (node % 9 - 4) / 4 for node in range(27)])
    form = mp.matrix(
        [
            _mp(
                mp,
                value + _ERROR * Q((node % 9 - 4) * (-1 if node // 9 == 1 else 1), 4),
            )
            for node, value in enumerate(forms)
        ]
    )
    for deformation in (-_LIMIT, Q(0), _LIMIT):
        a, _, _, potential = _phase_geometry(mp, phase, _mp(mp, deformation))
        cost = 2 * (form.T * a * form)[0] + potential
        assert (
            _mp(mp, report.initial_storage_bounds.lo)
            < cost
            < _mp(mp, report.initial_storage_bounds.hi)
        )


@pytest.mark.parametrize("deformation", (-_LIMIT, Q(0), _LIMIT))
def test_changed_complete_rows_rederive_storage_loss_and_actual_defect(
    deformation, reports, mp
):
    report = reports[_LIMIT]
    gamma, epsilon = 1 / (1023 * mp.pi), 1 / (1023 * mp.pi) ** 2
    raw = mp.matrix(
        [mp.sin(node + 1) + mp.mpf(node // 9 - 1) / 3 for node in range(27)]
    )
    form = gamma * (raw - mp.ones(27, 1) * sum(raw) / 27)
    raw_phase = mp.matrix([mp.mpf((7 * node) % 11) / 2 for node in range(27)])
    phase = raw_phase - mp.ones(27, 1) * sum(raw_phase) / 27
    a, field, derivative, _ = _phase_geometry(mp, phase, _mp(mp, deformation))
    form_rate, phase_rate = -a * form + gamma * field, gamma * a * form
    velocity = phase_rate / epsilon
    defect = velocity - field
    direct_derivative = a * form_rate / gamma - derivative * phase_rate
    expected_derivative = (
        -(a + epsilon * derivative) * defect - epsilon * derivative * field
    )
    assert mp.norm(defect + form_rate / gamma) < mp.mpf("1e-90")
    assert mp.norm(direct_derivative - expected_derivative) < mp.mpf("1e-85")
    storage_derivative = ((4 * a * form).T * form_rate)[0] - 4 * (field.T * phase_rate)[
        0
    ]
    assert abs(storage_derivative + 4 * mp.norm(a * form) ** 2) < mp.mpf("1e-85")
    assert abs(sum(field)) < mp.mpf("1e-90")
    assert max(abs(sum(phase_rate[9 * r : 9 * r + 9])) for r in range(3)) > mp.mpf(
        "1e-10"
    )
    assert mp.norm(field) <= _mp(mp, report.forcing_norm_bounds.hi)
    assert max(sum(abs(derivative[i, j]) for j in range(27)) for i in range(27)) <= _mp(
        mp, report.global_phase_lipschitz_bound
    )
    if deformation:
        _, baseline, _, _ = _phase_geometry(mp, phase, mp.mpf(0))
        assert mp.norm(field - baseline) > mp.mpf("0.0001")


@pytest.mark.parametrize(
    "field", ("deformation_bound", "form_error_bound", "phase_error_bound")
)
@pytest.mark.parametrize(
    "invalid", (True, False, float("nan"), float("inf"), complex(0), "0", Q(-1, 100))
)
def test_invalid_primitives_reject_before_geometry(monkeypatch, field, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("geometry rebuilt before invalid primitive rejection")

    monkeypatch.setattr(owner, "_derive_organized_triangle", forbidden)
    with pytest.raises((ValueError, TypeError)):
        _assess(**{field: invalid})


def test_deformation_outside_frozen_proof_range_rejects_before_geometry(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("geometry rebuilt outside the law proof range")

    monkeypatch.setattr(owner, "_derive_organized_triangle", forbidden)
    with pytest.raises(ValueError, match="1/10000"):
        _assess(deformation_bound=_LIMIT + Q(1, 10**50))


@pytest.mark.parametrize(
    "changes", ({"phase_error_bound": Q(1)}, {"form_error_bound": Q(10**20)})
)
def test_larger_source_errors_remain_unavailable_instead_of_reusing_verdicts(changes):
    report = _assess(**changes)
    assert report.status == "unavailable"
    assert report.unresolved_conditions
    assert not report.release_phase_budget_certified
    assert report.release_windings is None
    assert not report.donor_release_bridge_certified
    assert report.eventual_receiver_acquisition == "unresolved"


@pytest.mark.parametrize(
    "form_error,phase_error,dimension",
    ((Q(0), Q(0), 0), (_ERROR, Q(0), 24), (Q(0), _ERROR, 24), (_ERROR, _ERROR, 48)),
)
def test_preparation_dimension_counts_only_available_error_directions(
    form_error, phase_error, dimension
):
    report = _assess(form_error_bound=form_error, phase_error_bound=phase_error)
    assert report.relative_preparation_dimension == dimension
    assert report.status == "certified_uniform_constitutive_source"


def test_exact_zero_error_source_has_no_invented_error_directions():
    report = _assess(form_error_bound=Q(0), phase_error_bound=Q(0))
    assert report.relative_preparation_dimension == 0
    assert (
        report.initial_storage_bounds.lo
        <= 150 * _RAMP**2
        <= report.initial_storage_bounds.hi
    )
    assert report.initial_storage_bounds.width < Q(1, 10**30)
    assert report.initial_storage_correction_upper_bound == 0
    assert report.initial_potential_deformation_upper_bound == 0
    assert report.status == "certified_uniform_constitutive_source"
