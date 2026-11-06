"""Independent fine-graph controls for prepared acquisition and actual form."""

from fractions import Fraction as Q
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
from tnfr.physics import relational_sine_formation_response as formation
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

TIME = Q(100)
ERROR = Q(1, 10**10)
RADIUS = Q(1, 8)
LOSS = Q(1023, 1024)
WEIGHT = Q(1, 1024)


def _assess(time=TIME, form=ERROR, phase=ERROR, readout=ERROR, radius=RADIUS):
    return formation.assess_sine_formation_response(
        scaled_time=time,
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


def _sources():
    values = []
    for selected in (0, 3):
        forms = [Q(4039 * (i // 2 - 2)) for i in range(10)]
        forms[2 * selected] += 96
        forms[2 * selected + 1] -= 96
        values.append(tuple(forms))
    return tuple(values)


def _laplace(values):
    return tuple(
        len(row) * values[i] - sum(values[j] for j in row)
        for i, row in enumerate(NEIGHBORS)
    )


def _laplacian(mp):
    matrix = mp.matrix(10)
    for i, row in enumerate(NEIGHBORS):
        matrix[i, i] = len(row)
        for j in row:
            matrix[i, j] = -1
    return matrix


def _poisson_oracle(mp, phases):
    """Solve the original zero-mean fine Poisson equations independently."""
    currents = tuple(
        sum(mp.sin(phases[j] - phases[i]) for j in row)
        for i, row in enumerate(NEIGHBORS)
    )
    system = mp.matrix(11)
    laplacian = _laplacian(mp)
    for i in range(10):
        for j in range(10):
            system[i, j] = laplacian[i, j]
        system[i, 10] = system[10, i] = 1
    solution = mp.lu_solve(system, mp.matrix((*currents, 0)))
    assert abs(solution[10]) < mp.mpf("1e-90")
    return currents, tuple(solution[i] for i in range(10))


def _upper_close(mp, actual, expected, tolerance="1e-24"):
    difference = _mp(mp, actual) - expected
    assert 0 <= difference < mp.mpf(tolerance)


def test_sources_share_collective_means_and_complete_nominal_storage(frozen):
    sources = _sources()
    assert frozen.initial_forms_by_preparation == sources
    assert frozen.initial_phases == (Q(0),) * 10
    assert frozen.target_phase_turns == tuple(Q(i // 2 - 2, 5) for i in range(10))
    energies = []
    for forms in sources:
        assert sum(forms) == 0
        assert tuple((forms[i] + forms[j]) / 2 for i, j in PAIRS) == tuple(
            Q(4039 * (a - 2)) for a in range(5)
        )
        energies.append(sum((forms[i] - forms[j]) ** 2 / 2 for i, j in EDGES))
    assert energies[0] == energies[1] == 40 * 4039**2 + 4 * 96**2
    assert frozen.initial_form_storage_by_preparation == tuple(energies)
    assert sources[0][2] == sources[0][3] == sources[1][2] == sources[1][3]
    assert frozen.original_time == TIME / LOSS == Q(102400, 1023)


def test_full_original_and_scaled_rows_preserve_means_and_account_for_loss(mp):
    gamma = _mp(mp, WEIGHT / LOSS) / mp.pi
    for source in _sources():
        lx = _laplace(source)
        # Initially every sine current is zero, although both rows evolve.
        form_original = tuple(-LOSS * value / 4 for value in lx)
        form_scaled = tuple(-value / 4 for value in lx)
        phase_original = tuple(_mp(mp, WEIGHT * value / 4) / mp.pi for value in lx)
        phase_scaled = tuple(gamma * _mp(mp, value) / 4 for value in lx)
        assert sum(form_original) == sum(form_scaled) == 0
        assert abs(sum(phase_original)) < mp.mpf("1e-90")
        assert abs(sum(phase_scaled)) < mp.mpf("1e-90")
        for original, scaled in zip(form_original, form_scaled, strict=True):
            assert original == LOSS * scaled
        for original, scaled in zip(phase_original, phase_scaled, strict=True):
            assert abs(original - _mp(mp, LOSS) * scaled) < mp.mpf("1e-90")
        direct_work = sum(
            (source[i] - source[j]) * (form_original[i] - form_original[j])
            for i, j in EDGES
        )
        assert direct_work == -LOSS * sum(value**2 for value in lx) / 4 < 0
        # Away from flat phase, both exchange terms must cancel in the
        # storage balance. Testing the zero-sine source alone misses this.
        phases = [mp.mpf(i * i) / 23 for i in range(10)]
        current = [
            sum(mp.sin(phases[j] - phases[i]) for j in row)
            for i, row in enumerate(NEIGHBORS)
        ]
        nonlinear_form = [
            _mp(mp, drift) + _mp(mp, WEIGHT) * force / (4 * mp.pi)
            for drift, force in zip(form_original, current, strict=True)
        ]
        nonlinear_work = sum(
            _mp(mp, source[i] - source[j]) * (nonlinear_form[i] - nonlinear_form[j])
            + mp.sin(phases[j] - phases[i]) * (phase_original[j] - phase_original[i])
            for i, j in EDGES
        )
        assert abs(nonlinear_work - _mp(mp, direct_work)) < mp.mpf("1e-80")
    # Reallocating internal form does not change either initial receiver mean.
    first = [tuple(-value / 4 for value in _laplace(source)) for source in _sources()]
    assert sum(first[0][i] for i in PAIRS[1]) == sum(first[1][i] for i in PAIRS[1])


def test_declared_diffusion_gap_and_green_matrix_use_the_full_fine_support(mp, frozen):
    laplacian = _laplacian(mp)
    eigenvalues = mp.eigsy(laplacian / 4, eigvals_only=True)
    assert abs(eigenvalues[0]) < mp.mpf("1e-90")
    assert eigenvalues[1] > _mp(mp, Q(2, 3))
    assert 1 < eigenvalues[9] < 2
    assert frozen.lambda_lower_bound == Q(2, 3)
    assert frozen.lambda_upper_bound == 2
    green = frozen.poisson_inverse
    assert len(green) == 10 and all(len(row) == 10 for row in green)
    for i in range(10):
        assert sum(green[i]) == 0
        for j in range(10):
            assert type(green[i][j]) is Q
            assert green[i][j] == green[j][i]
            product = len(NEIGHBORS[i]) * green[i][j] - sum(
                green[k][j] for k in NEIGHBORS[i]
            )
            assert product == Q(i == j) - Q(1, 10)
    # A receiver mean has smaller dual norm on the zero-mean subspace.
    centered_readout = tuple(
        Q(1, 2) - Q(1, 10) if i in PAIRS[1] else Q(-1, 10) for i in range(10)
    )
    dual_squared = sum(value**2 / 4 for value in centered_readout)
    assert dual_squared == Q(1, 10)
    _contains(mp, frozen.receiver_dual_norm_bounds, mp.sqrt(_mp(mp, dual_squared)))


@pytest.fixture(scope="module")
def proxies(mp):
    gamma = _mp(mp, WEIGHT / LOSS) / mp.pi
    values = []
    for source in _sources():
        phases = tuple(gamma * _mp(mp, value) for value in source)
        current, profile = _poisson_oracle(mp, phases)
        values.append((phases, current, profile))
    return tuple(values)


def test_fine_poisson_solution_and_proxy_storage_are_independent_oracles(
    mp, frozen, proxies
):
    gamma = _mp(mp, WEIGHT / LOSS) / mp.pi
    alpha = 2 * mp.pi / 5
    target = tuple((i // 2 - 2) * alpha for i in range(10))
    target_storage = _energy(mp, [mp.mpf(0)] * 10, target)
    _contains(mp, frozen.gamma_bounds, gamma)
    _contains(mp, frozen.eta_bounds, gamma**2)
    centers = []
    for index, (phases, current, profile) in enumerate(proxies):
        for field, expected in zip(
            frozen.scaled_initial_form_bounds[index], phases, strict=True
        ):
            _contains(mp, field, expected)
        for field, expected in zip(
            frozen.phase_current_bounds_by_preparation[index], current, strict=True
        ):
            _contains(mp, field, expected)
        for field, expected in zip(
            frozen.poisson_profile_bounds_by_preparation[index], profile, strict=True
        ):
            _contains(mp, field, expected)
        center = gamma * sum(profile[i] for i in PAIRS[1]) / 2
        centers.append(center)
        _contains(mp, frozen.receiver_center_bounds[index], center)
        storage = _energy(mp, [mp.mpf(0)] * 10, phases)
        _contains(mp, frozen.proxy_phase_storage_bounds[index], storage)
        _contains(
            mp, frozen.proxy_excess_storage_bounds[index], storage - target_storage
        )
        _upper_close(
            mp, frozen.scaled_initial_norm_upper_bounds[index], 2 * _norm(mp, phases)
        )
        _upper_close(
            mp, frozen.poisson_profile_norm_upper_bounds[index], 2 * _norm(mp, profile)
        )
        _upper_close(
            mp, frozen.proxy_phase_gradient_norm_upper_bounds[index], _norm(mp, current)
        )
        _upper_close(
            mp,
            frozen.proxy_target_distance_upper_bounds[index],
            _norm(mp, [p - t for p, t in zip(phases, target, strict=True)]),
        )
    assert centers[0] > centers[1]
    # The Green operator mediates every fine current. A raw receiver current
    # sign or its local rate cannot replace the actual form approximation.
    local_current = gamma * sum(proxies[0][1][i] for i in PAIRS[1]) / 8
    assert abs(local_current - centers[0]) > mp.mpf("1e-12")


def _endpoint_oracle(mp, time, form, phase, readout, proxy):
    phases, current, profile = proxy
    t, rx, rt, eta_readout = map(
        lambda value: _mp(mp, Q(value)), (time, form, phase, readout)
    )
    gamma = _mp(mp, WEIGHT / LOSS) / mp.pi
    eta = gamma**2
    decay_rate = _mp(mp, Q(2, 3))
    # Degree-four averaging gives |KS_i|<=1 and M=4I. The global
    # phase-current Jacobian has M-operator norm bounded by2.
    force_norm = mp.sqrt(sum(mp.mpf(4) for _ in NEIGHBORS))
    lipschitz = max(1 + mp.mpf(len(row)) / 4 for row in NEIGHBORS)
    initial = 2 * _norm(mp, phases)
    initial_error = force_norm * gamma * rx
    phase_error = force_norm * rt
    decay = mp.exp(-decay_rate * t)
    # Integrate the semigroup comparison term by term: the decaying
    # history term convolves to T exp(-lambda*T), not T exp(-2lambda*T).
    remainder = decay * (initial + initial_error + eta * force_norm / decay_rate)
    remainder += (
        eta
        * lipschitz
        * (
            initial * t * decay
            + (initial_error + phase_error) / decay_rate
            + eta * force_norm * (t + 1 / decay_rate) / decay_rate
        )
    )
    phase_endpoint = (
        decay * initial
        + initial_error
        + phase_error
        + eta * force_norm * (t + 1 / decay_rate)
    ) / 2
    form_endpoint = (eta * 2 * _norm(mp, profile) + remainder) / (2 * gamma)
    receiver_error = rx + remainder / (gamma * mp.sqrt(10)) + eta_readout
    return remainder, phase_endpoint, form_endpoint, receiver_error


@pytest.mark.parametrize("time", (Q(0), Q(1), TIME))
def test_duhamel_remainder_retains_full_transient_nonlinear_history_and_mean_error(
    mp, proxies, time
):
    report = _assess(time=time)
    _contains(mp, report.exponential_decay_bounds, mp.exp(-_mp(mp, Q(2, 3) * time)))
    for index, proxy in enumerate(proxies):
        expected = _endpoint_oracle(mp, time, ERROR, ERROR, ERROR, proxy)
        actual = (
            report.scaled_form_remainder_upper_bounds[index],
            report.endpoint_phase_error_norm_upper_bounds[index],
            report.endpoint_form_norm_upper_bounds[index],
            report.receiver_error_upper_bounds[index],
        )
        for retained, oracle in zip(actual, expected, strict=True):
            _upper_close(mp, retained, oracle)


@pytest.mark.parametrize("form,phase", ((ERROR, 0), (0, ERROR), (ERROR, 3 * ERROR)))
def test_preparation_channels_include_mean_and_every_fine_coordinate(
    mp, proxies, form, phase
):
    report = _assess(form=form, phase=phase, readout=0)
    gamma = _mp(mp, WEIGHT / LOSS) / mp.pi
    for index, proxy in enumerate(proxies):
        _upper_close(
            mp,
            report.initial_scaled_form_error_norm_upper_bounds[index],
            mp.sqrt(40) * gamma * _mp(mp, Q(form)),
        )
        expected = _endpoint_oracle(mp, TIME, form, phase, 0, proxy)
        _upper_close(mp, report.receiver_error_upper_bounds[index], expected[-1])
    _upper_close(
        mp, report.initial_phase_error_norm_upper_bound, mp.sqrt(40) * _mp(mp, Q(phase))
    )
    if form:
        # A uniform form perturbation does not change any Laplacian or phase
        # trajectory; it nevertheless shifts the unsubtracted receiver by rho_x.
        for source in _sources():
            assert _laplace(tuple(value + form for value in source)) == _laplace(source)
        assert all(error >= form for error in report.receiver_error_upper_bounds)


def test_endpoint_geometry_uses_full_form_norm_and_phase_gradient(mp, frozen, proxies):
    alpha = 2 * mp.pi / 5
    target = tuple((i // 2 - 2) * alpha for i in range(10))
    baseline = _energy(mp, [mp.mpf(0)] * 10, target)
    for index, (phases, current, _) in enumerate(proxies):
        # Compare with high-precision independent bounds, not a selected
        # numerical trajectory or an independently reset target state.
        _, phase_error, form_error, _ = _endpoint_oracle(
            mp, TIME, ERROR, ERROR, ERROR, proxies[index]
        )
        distance = _norm(mp, [p - t for p, t in zip(phases, target, strict=True)])
        norm_bound = form_error**2 + (distance + phase_error) ** 2
        energy_bound = (
            _energy(mp, [mp.mpf(0)] * 10, phases)
            - baseline
            + _norm(mp, current) * phase_error
            + 4 * (phase_error**2 + form_error**2)
        )
        _upper_close(
            mp, frozen.endpoint_relative_norm_squared_upper_bounds[index], norm_bound
        )
        _upper_close(
            mp, frozen.endpoint_excess_storage_upper_bounds[index], energy_bound
        )
    assert frozen.initial_zero_winding_certified
    assert frozen.initial_zero_winding_margin_bounds.lo > 0
    assert frozen.acute_radius_certified
    assert frozen.formation_certified_by_preparation == (True, True)
    assert all(margin.lo > 0 for margin in frozen.endpoint_radius_margin_bounds)
    assert all(margin.lo > 0 for margin in frozen.storage_barrier_margin_bounds)


def test_frozen_joint_result_requires_actual_receiver_and_both_acquisitions(frozen):
    assert frozen.response_separation_certified
    assert frozen.recorded_difference_bounds.lo > Q(1, 10**8)
    assert frozen.status == "certified_formation_response"
    assert not frozen.unavailable_reasons
    a, b = frozen.receiver_center_bounds
    radius = sum(frozen.receiver_error_upper_bounds)
    assert frozen.recorded_difference_bounds.contains(a.lo - b.hi - radius)
    assert frozen.recorded_difference_bounds.contains(a.hi - b.lo + radius)
    for center, error, recorded in zip(
        frozen.receiver_center_bounds,
        frozen.receiver_error_upper_bounds,
        frozen.recorded_readout_bounds,
        strict=True,
    ):
        assert recorded.contains(center.lo - error)
        assert recorded.contains(center.hi + error)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"time": 0},
        {"form": 10**6},
        {"phase": 2},
        {"radius": Q(1, 100)},
        {"radius": Q(1, 2)},
    ),
)
def test_failed_formation_is_not_rescued_by_proxy_response(kwargs):
    report = _assess(**kwargs)
    assert report.status == "unavailable"
    assert not all(report.formation_certified_by_preparation)
    assert report.unavailable_reasons


def test_response_failure_is_separate_from_successful_formation():
    report = _assess(readout=Q(1, 100))
    assert report.formation_certified_by_preparation == (True, True)
    assert not report.response_separation_certified
    assert report.status == "unavailable"


def test_tiny_rational_errors_remain_exact_and_touching_is_unavailable():
    tiny = Q(1, 2**1100)
    report = _assess(form=tiny, phase=tiny, readout=tiny)
    assert (
        report.form_error_bound
        == report.phase_error_bound
        == report.readout_error_bound
        == tiny
    )
    assert report.initial_phase_error_norm_upper_bound > 0
    baseline = _assess(readout=0)
    touching = _assess(readout=baseline.recorded_difference_bounds.lo / 2)
    assert touching.recorded_difference_bounds.lo <= 0
    assert not touching.response_separation_certified


@pytest.mark.parametrize("field", ("time", "form", "phase", "readout", "radius"))
@pytest.mark.parametrize("bad", (True, float("nan"), float("inf"), -1, "0", 1j))
def test_original_scalar_admission(field, bad):
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize("kwargs", ({"radius": 0}, {"time": 6145}))
def test_positive_radius_and_declared_exponential_work_boundary(kwargs):
    with pytest.raises(ValueError):
        _assess(**kwargs)


def test_nonrational_underflow_rejects_before_materialization():
    class LostReal(float):
        def __float__(self):
            return 0.0

    with pytest.raises(ValueError, match="underflows"):
        _assess(form=LostReal(1.0))


def test_fixed_model_is_not_replaced_by_an_unchecked_report(frozen):
    with pytest.raises(TypeError):
        formation.assess_sine_formation_response(reference_certificate=frozen)


def test_direct_and_sdk_exports_retain_exact_primitives(tmp_path, frozen):
    get_type_hints(formation.SineFormationResponse)
    get_type_hints(formation.assess_sine_formation_response)
    for name, report in (("certificate", frozen), ("unavailable", _assess(time=0))):
        direct = report.to_dict()
        assert direct["schema"] == "tnfr.sine-formation-response.v1"
        projected = relational_report_to_dict(report)
        assert projected["report_type"] == "SineFormationResponse"
        assert projected["report"] == direct["report"]
        assert projected["report"]["form_error_bound"] == {
            "numerator": 1,
            "denominator": 10**10,
        }
        target = tmp_path / (name + ".json")
        export_to_json(projected, target)
        assert json_loads(target.read_text(encoding="utf-8")) == projected
