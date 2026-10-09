"""Independent actual-source recovery, contact and four-branch controls."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_constitutive_handoff as owner

ERROR = Q(1, 10**10)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


def _assess(**changes):
    return owner._assess_constitutive_handoff(
        **(
            dict(deformation=Q(1), form_error_bound=ERROR, phase_error_bound=ERROR)
            | changes
        )
    )


@pytest.fixture(scope="module")
def endpoints():
    return tuple(_assess(deformation=coefficient) for coefficient in (0, 1))


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def test_original_sources_and_actual_return_are_kept(endpoints):
    for report in endpoints:
        assert report.status == "certified_constitutive_handoff"
        assert report.source_certified and report.handoff_certified
        assert not report.unresolved_conditions
        assert report.source.scaled_time == 100
        assert report.source.radius == Q(1, 12)
        assert (
            report.source.form_error_bound == report.source.phase_error_bound == ERROR
        )
        assert report.source.preparation_dimension == 16
        assert report.relaxation_duration == 10**13
        assert report.scaled_handoff_time == 100 + 10**13
        assert (
            report.original_handoff_time * Q(1023, 1024) == report.scaled_handoff_time
        )
        assert report.endpoint_radius == Q(1, 10**70)
        assert report.endpoint_radius != report.form_error_bound
        assert report.exact_decay_upper_bound == Q(1, 2**512)
        assert report.endpoint_form_norm_squared_upper_bound < Q(1, 10**140)
        assert report.endpoint_phase_norm_squared_upper_bound < Q(1, 10**140)


def test_own_hessian_and_strict_nonlinear_decay_have_independent_values(endpoints, mp):
    eta, gap, maximum, epsilon = (
        1 / (1023 * mp.pi) ** 2,
        mp.mpf(1) / 5,
        2,
        mp.mpf(1) / 20,
    )
    for report in endpoints:
        deformation = _mp(mp, report.deformation)
        c = mp.cos(4 * mp.pi / 9 + mp.sqrt(2) / 12) - 2 * deformation / 243
        u = 1 + 2 * deformation / 243
        mu, stiff = eta * c * gap**2, eta * u * maximum**2
        lower = mu / 2 + gap**2 / 16
        upper = stiff / 2 + epsilon * maximum / 2 + epsilon**2
        rate = min(4 * (gap - epsilon) / 3, epsilon * mu / upper)
        assert (
            0
            <= mu - _mp(mp, report.transformed_stiffness_lower_bound)
            < mp.mpf("1e-28")
        )
        assert (
            0
            <= _mp(mp, report.transformed_stiffness_upper_bound) - stiff
            < mp.mpf("1e-28")
        )
        assert (
            0
            <= lower - _mp(mp, report.lyapunov_position_lower_coefficient)
            < mp.mpf("1e-28")
        )
        assert (
            0
            <= _mp(mp, report.lyapunov_position_upper_coefficient) - upper
            < mp.mpf("1e-28")
        )
        assert 0 <= rate - _mp(mp, report.lyapunov_decay_rate) < mp.mpf("1e-28")
        assert 1000 < report.decay_exponent < Q(16000, 7) < 4096
        assert mp.exp(-_mp(mp, report.decay_exponent)) < _mp(
            mp, report.exact_decay_upper_bound
        )
        assert report.transformed_stiffness_upper_bound == (
            report.source.eta_bounds.hi * report.source.phase_hessian_upper_bound * 4
        )
    assert endpoints[1].lyapunov_decay_rate < endpoints[0].lyapunov_decay_rate


def test_return_energy_and_two_coordinate_metrics_are_retained(endpoints):
    for report in endpoints:
        source, energy = report.source, report.returned_lyapunov_upper_bound
        assert energy == report.lyapunov_initial_upper_bound / 2**512
        assert (
            report.endpoint_form_norm_squared_upper_bound * source.eta_bounds.lo / 20
            == energy
        )
        assert (
            report.endpoint_phase_norm_squared_upper_bound
            * report.lyapunov_position_lower_coefficient
            / 2
            == energy
        )
        # These coarse independent design inequalities do not consume the
        # source's tighter component radii or a rounded tiny exponential.
        assert report.lyapunov_initial_upper_bound < Q(1, 500)
        assert Q(4_000_000, 9 * 2**512) < Q(1, 10**148)
        assert Q(8, 5 * 2**512) < Q(1, 10**148)
        assert report.endpoint_form_margin > 0 and report.endpoint_phase_margin > 0


def _edges():
    return tuple(
        (9 * c + j, 9 * c + (j + 1) % 9) for c in range(3) for j in range(9)
    ) + ((4, 13), (13, 22))


def test_actual_contact_and_input_work_enclose_independently_constructed_states(
    endpoints, mp
):
    eps = endpoints[0].endpoint_radius
    # Each component retains exact zero means and a nonzero asymmetric error;
    # the current at either port need not have the nominal reflection symmetry.
    x = tuple(
        eps * Q(c + 1, 8) * ((j == 4) - (j == 0)) for c in range(3) for j in range(9)
    )
    y = tuple(
        eps * Q(3 - c, 10) * ((j == 4) - (j == 1)) for c in range(3) for j in range(9)
    )
    degrees = tuple(sum(i in edge for edge in _edges()) for i in range(27))
    q = [Q(0)] * 27
    for i, j in _edges():
        q[i] += x[i] - x[j]
        q[j] += x[j] - x[i]
    for report in endpoints:
        deformation = _mp(mp, report.deformation)
        work = mp.mpf(0)
        for i, j in report.bridges:
            dx, dy = _mp(mp, x[j] - x[i]), _mp(mp, y[j] - y[i])
            # Stable potential evaluation avoids subtracting almost equal cosines.
            work += (
                dx**2 / 2
                + 2 * mp.sin(dy / 2) ** 2
                + deformation * (2 * mp.sin(9 * dy / 2) ** 2) ** 2 / 39366
            )
        assert (
            _mp(mp, report.attachment_work_bounds[0])
            <= work
            <= _mp(mp, report.attachment_work_bounds[1])
        )
        assert work > 0
        for branch in report.branches:
            a, b = branch.donor_amplitude, branch.receiver_amplitude
            f = tuple(a * (i == 4) + b * (i == 22) for i in range(27))
            actual_work = (
                sum(
                    (
                        (x[j] + f[j] - x[i] - f[i]) ** 2 - (x[j] - x[i]) ** 2
                        for i, j in _edges()
                    ),
                    Q(0),
                )
                / 2
            )
            assert actual_work == a * q[4] + b * q[22] + Q(3, 2) * (a * a + b * b)
            if a + b:
                assert actual_work != Q(3, 2) * (a * a + b * b)
            assert (
                branch.input_work_bounds[0]
                <= actual_work
                <= branch.input_work_bounds[1]
            )
            form_mean = (
                sum((d * (value + jump) for d, value, jump in zip(degrees, x, f)), Q(0))
                / 58
            )
            phase_mean = sum((d * value for d, value in zip(degrees, y)), Q(0)) / 58
            assert branch.form_mean_bounds[0] <= form_mean <= branch.form_mean_bounds[1]
            assert (
                branch.phase_mean_bounds[0] <= phase_mean <= branch.phase_mean_bounds[1]
            )
            assert branch.identity_certified


def test_four_branch_geometric_and_storage_margins_preserve_tiny_bounds(endpoints):
    for report in endpoints:
        eps, u = report.endpoint_radius, report.source.phase_hessian_upper_bound
        assert report.attachment_work_bounds == (Q(0), 4 * (1 + u) * eps**2)
        assert report.joined_excess_storage_upper_bound == 10 * (1 + u) * eps**2
        assert report.joined_form_mean_bounds == (-4 * eps / 58, 4 * eps / 58)
        assert report.joined_phase_mean_bounds == report.joined_form_mean_bounds
        assert report.attachment_work_bounds[1] < Q(1, 2**128)
        assert tuple(branch.name for branch in report.branches) == (
            "neither",
            "donor",
            "receiver",
            "both",
        )
        for branch in report.branches:
            a, b = branch.donor_amplitude, branch.receiver_amplitude
            f = tuple(a * (i == 4) + b * (i == 22) for i in range(27))
            mean = sum(f, Q(0)) / 27
            exact_kick_norm = sum(((v - mean) ** 2 for v in f), Q(0))
            assert (
                branch.relative_norm_squared_upper_bound
                == 6 * eps**2 + 2 * (a + b) * eps + exact_kick_norm
            )
            assert branch.radius_margin > 0 and branch.storage_margin > Q(587, 10**9)
            assert branch.excess_storage_upper_bound < report.joined_storage_barrier


@pytest.mark.parametrize(
    "changes",
    [
        {"form_error_bound": Q(10**6)},
        {"phase_error_bound": Q(1)},
        {"deformation": Q(8)},
    ],
)
def test_missing_source_or_family_premises_remain_unavailable(changes):
    report = _assess(**changes)
    assert report.status == "unavailable"
    assert not report.handoff_certified
    assert all(not branch.identity_certified for branch in report.branches)
    assert report.unresolved_conditions


def test_zero_storage_margin_is_not_a_certificate(monkeypatch, endpoints):
    monkeypatch.setattr(
        owner, "_JOINED_BARRIER", endpoints[1].branches[-1].excess_storage_upper_bound
    )
    report = _assess()
    assert report.handoff_certified
    assert report.branches[-1].storage_margin == 0
    assert not report.branches[-1].identity_certified
    assert report.status == "unavailable"


@pytest.mark.parametrize(
    "field,value",
    [
        ("deformation", True),
        ("deformation", Q(-1)),
        ("form_error_bound", float("nan")),
        ("phase_error_bound", True),
        ("phase_error_bound", Q(-1)),
    ],
)
def test_bad_primitive_values_reject_before_source_construction(
    monkeypatch, field, value
):
    def forbidden(**kwargs):
        pytest.fail("source construction preceded raw scalar admission")

    monkeypatch.setattr(owner, "_assess_constitutive_class_pair", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: value})


def test_incoming_reports_and_altered_dwell_are_not_admitted(endpoints):
    for extra in (
        {"source": endpoints[1].source},
        {"relaxation_duration": Q(0)},
        {"endpoint_radius": Q(0)},
    ):
        with pytest.raises(TypeError):
            _assess(**extra)
