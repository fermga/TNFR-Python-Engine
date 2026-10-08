"""Independent lift, complete-row and retained-error controls for reduced ports.

Static controls need no transfer assessor. The frozen certificate fixture is
used only after the protocol/source archive and first reserved evaluation.
No nonlinear trajectory, ideal-state reset or extra forecast engine is used.
"""

from fractions import Fraction as Q
from pathlib import Path
from typing import get_type_hints

import mpmath
import pytest

from tests.physics.test_sine_formed_class_contact import (
    DEGREES,
    EDGES,
    _contains,
    _laplacian,
    _mp,
    _mv,
)
from tnfr.mathematics._exact_linear_algebra import exact_matrix_product
from tnfr.physics import relational_sine_reduced_class_ports as owner
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

INPUTS = dict(
    formation_time=Q(100),
    relaxation_duration=Q(10**13),
    phase_origin_difference=Q(1, 1000),
    contact_duration=Q(1, 100),
    form_error_bound=Q(1, 10**10),
    phase_error_bound=Q(1, 10**10),
    endpoint_radius=Q(1, 10**32),
    readout_error_bound=Q(1, 10**30),
    radius=Q(1, 12),
    work_allowance=Q(1, 10**6),
    decay_power=512,
    error_fraction=Q(1, 4),
)


def _assess(**changes):
    return owner.assess_sine_reduced_class_ports(**(INPUTS | changes))


def _evaluate(**changes):
    inputs = dict(
        donor_class=1,
        receiver_class=2,
        forms=(Q(0),) * 10,
        phase_deviations=(Q(0),) * 10,
        phase_origin_difference=Q(1, 7),
    )
    return owner.evaluate_sine_reduced_class_ports(**(inputs | changes))


@pytest.fixture(scope="module")
def frozen():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 120
    return context


@pytest.fixture(scope="module", autouse=True)
def no_previous_contact_or_trajectory():
    from tnfr.dynamics import relational
    from tnfr.physics import (
        relational_sine_forecast,
        relational_sine_formed_class_contact,
        relational_sine_formed_classes,
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "reduced transfer must rebuild primitives, not reuse an earlier verdict or trajectory"
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        patch.setattr(relational_sine_forecast, "bound_sine_flow", forbidden)
        patch.setattr(
            relational_sine_formed_class_contact,
            "assess_sine_formed_class_contact",
            forbidden,
        )
        patch.setattr(
            relational_sine_formed_classes,
            "assess_sine_formed_class_response",
            forbidden,
        )
        yield


def _lifts():
    orbit = tuple(abs(i - 4) for i in range(9)) + tuple(
        5 + abs(i - 13) for i in range(9, 18)
    )
    lift = tuple(tuple(Q(j == orbit[i]) for j in range(10)) for i in range(18))
    project = tuple(
        tuple(Q(orbit[i] == j, orbit.count(j)) for i in range(18)) for j in range(10)
    )
    return lift, project


def _full_normalized(edges=EDGES):
    return tuple(
        tuple(value / DEGREES[i] for value in row)
        for i, row in enumerate(_laplacian(edges))
    )


class TestStaticKernel:
    """May run before reserved evaluation: no assessor, fixture or response scan."""

    def test_projection_lift_and_all_three_reduced_matrices(self):
        result = _evaluate()
        lift, project = _lifts()
        identity = tuple(tuple(Q(i == j) for j in range(10)) for i in range(10))
        assert exact_matrix_product(project, lift) == identity
        for full, reduced in (
            (_full_normalized(), result.normalized_form_matrix),
            (_full_normalized(EDGES[:9]), result.normalized_donor_interior_matrix),
            (_full_normalized(EDGES[9:18]), result.normalized_receiver_interior_matrix),
        ):
            assert exact_matrix_product(full, lift) == exact_matrix_product(
                lift, reduced
            )
            assert exact_matrix_product(project, full) == exact_matrix_product(
                reduced, project
            )
        assert result.component_coordinate_count == 10
        assert (
            result.joined_coordinate_count
            == 20
            < result.full_joined_coordinate_count
            == 36
        )
        assert result.orbit_multiplicities == (1, 2, 2, 2, 2)
        weights = (3, 4, 4, 4, 4) * 2
        a = result.normalized_form_matrix
        assert all(
            weights[i] * a[i][j] == weights[j] * a[j][i]
            for i in range(10)
            for j in range(10)
        )
        assert sum(weights) == 38

    @pytest.mark.parametrize("classes", [(1, 1), (1, 2), (2, 1), (2, 2)])
    def test_lifted_rows_equal_linear_interior_with_exact_sine_bridge(
        self, mp, classes
    ):
        x = tuple(Q(i * i - 4 * i + 2, 19) for i in range(10))
        y = tuple(Q((3 * i) % 7 - 3, 17) for i in range(10))
        phi = Q(-1, 5)
        result = _evaluate(
            donor_class=classes[0],
            receiver_class=classes[1],
            forms=x,
            phase_deviations=y,
            phase_origin_difference=phi,
        )
        lift, _ = _lifts()
        xf, yf = _mv(lift, x), _mv(lift, y)
        force = [mp.mpf(0)] * 18
        for i, j in EDGES:
            if (i, j) == (4, 13):
                current = mp.sin(_mp(mp, phi + yf[j] - yf[i]))
            else:
                k = classes[0] if i < 9 else classes[1]
                current = mp.cos(2 * mp.pi * k / 9) * _mp(mp, yf[j] - yf[i])
            force[i] += current / DEGREES[i]
            force[j] -= current / DEGREES[j]
        a_x = _mv(_full_normalized(), xf)
        gamma = 1 / (1023 * mp.pi)
        full_x = tuple(-_mp(mp, a_x[i]) + gamma * force[i] for i in range(18))
        full_y = tuple(gamma * _mp(mp, value) for value in a_x)
        for local, physical in enumerate((4, 3, 2, 1, 0, 13, 12, 11, 10, 9)):
            _contains(mp, result.form_rate_bounds[local], full_x[physical])
            _contains(mp, result.phase_rate_bounds[local], full_y[physical])
        gap = phi + y[5] - y[0]
        assert result.bridge_phase_difference == gap
        _contains(mp, result.bridge_sine_bounds, mp.sin(_mp(mp, gap)))
        assert abs(mp.sin(_mp(mp, gap)) - _mp(mp, gap)) > mp.mpf("1e-8")

    def test_nonlinear_full_interior_has_a_bounded_nonzero_discarded_residual(self, mp):
        # A nonuniform even phase state creates odd fine-node residuals;
        # reflection-even modes are therefore not an exact nonlinear closure.
        y = tuple(Q(i % 3 - 1, 100) for i in range(10))
        result = _evaluate(phase_deviations=y)
        lift, _ = _lifts()
        yf = _mv(lift, y)
        target = [(i - 4) * 2 * mp.pi / 9 for i in range(9)] + [
            (i - 13) * 4 * mp.pi / 9 + mp.mpf(1) / 7 for i in range(9, 18)
        ]
        phase = tuple(target[i] + _mp(mp, yf[i]) for i in range(18))
        nonlinear = [mp.mpf(0)] * 18
        for i, j in EDGES:
            current = mp.sin(phase[j] - phase[i])
            nonlinear[i] += current / DEGREES[i]
            nonlinear[j] -= current / DEGREES[j]
        gamma = 1 / (1023 * mp.pi)
        reduced_x = tuple(
            _mp(mp, interval.midpoint) for interval in result.form_rate_bounds
        )
        lifted = _mv(lift, reduced_x)
        residual = tuple(gamma * nonlinear[i] - lifted[i] for i in range(18))
        ymax = max(abs(_mp(mp, value)) for value in y)
        assert max(map(abs, residual)) <= 2 * gamma * ymax**2
        assert max(abs(residual[i] - residual[8 - i]) for i in range(4)) > mp.mpf(
            "1e-10"
        )

    def test_ordinary_origins_are_retained_not_quotiented(self):
        x, y = tuple(Q(i, 9) for i in range(10)), tuple(Q(i, 11) for i in range(10))
        first = _evaluate(forms=x, phase_deviations=y)
        shifted = _evaluate(
            forms=tuple(v + 17 for v in x), phase_deviations=tuple(v - 23 for v in y)
        )
        assert (
            shifted.forms != first.forms
            and shifted.phase_deviations != first.phase_deviations
        )
        assert shifted.form_rate_bounds == first.form_rate_bounds
        assert shifted.phase_rate_bounds == first.phase_rate_bounds

    def test_receiver_origin_and_deviation_changes_preserve_the_same_phases(self):
        x = tuple(Q(i, 13) for i in range(10))
        y = tuple(Q(i * i - 2 * i, 17) for i in range(10))
        phi, shift = Q(1, 7), Q(19, 3)
        first = _evaluate(forms=x, phase_deviations=y, phase_origin_difference=phi)
        represented = _evaluate(
            forms=x,
            phase_deviations=y[:5] + tuple(value - shift for value in y[5:]),
            phase_origin_difference=phi + shift,
        )
        assert represented.phase_origin_difference != first.phase_origin_difference
        assert represented.phase_deviations != first.phase_deviations
        assert represented.bridge_phase_difference == first.bridge_phase_difference
        assert represented.form_rate_bounds == first.form_rate_bounds
        assert represented.phase_rate_bounds == first.phase_rate_bounds

    @pytest.mark.parametrize("field", ["donor_class", "receiver_class"])
    @pytest.mark.parametrize("bad", [True, 0, 3, Q(1), 1.0, "1"])
    def test_class_labels_are_original_ordinary_integers(self, field, bad):
        with pytest.raises(ValueError):
            _evaluate(**{field: bad})

    @pytest.mark.parametrize("field", ["forms", "phase_deviations"])
    @pytest.mark.parametrize("bad", [True, "1", 1j, float("nan"), float("inf")])
    def test_raw_coordinate_admission(self, monkeypatch, field, bad):
        def forbidden(*args):
            pytest.fail("invalid primitive reached reduced coefficient arithmetic")

        monkeypatch.setattr(owner, "_reduced_parameters", forbidden)
        with pytest.raises((ValueError, TypeError)):
            _evaluate(**{field: (bad,) + (Q(0),) * 9})

    @pytest.mark.parametrize(
        "values", [(), (0,) * 9, (0,) * 11, {0}, {"x": 0}, "0000000000"]
    )
    def test_coordinate_order_and_dimension(self, values):
        with pytest.raises((TypeError, ValueError)):
            _evaluate(forms=values)

    def test_native_exact_tiny_state_is_retained_and_exportable(self, tmp_path):
        tiny = Q(1, 2**600)
        result = _evaluate(forms=(tiny,) + (Q(0),) * 9)
        assert result.forms[0] == tiny
        direct = result.to_dict()
        assert relational_report_to_dict(result)["report"] == direct["report"]
        path = tmp_path / "rows.json"
        export_to_json(result, path)
        assert json_loads(path.read_bytes())["report"] == direct["report"]


def _full_receiver_jets(mp, donor_class, receiver_class, phi):
    gamma = 1 / (1023 * mp.pi)
    a = tuple(tuple(_mp(mp, value) for value in row) for row in _full_normalized())
    phase = [(i - 4) * 2 * mp.pi * donor_class / 9 for i in range(9)] + [
        (i - 13) * 2 * mp.pi * receiver_class / 9 + _mp(mp, phi) for i in range(9, 18)
    ]
    force = [mp.mpf(0)] * 18
    jac = [[mp.mpf(0) for _ in range(18)] for _ in range(18)]
    for i, j in EDGES:
        s, c = mp.sin(phase[j] - phase[i]), mp.cos(phase[j] - phase[i])
        force[i] += s / DEGREES[i]
        force[j] -= s / DEGREES[j]
        jac[i][i] -= c / DEGREES[i]
        jac[i][j] += c / DEGREES[i]
        jac[j][j] -= c / DEGREES[j]
        jac[j][i] += c / DEGREES[j]
    x1 = tuple(gamma * value for value in force)
    x2 = tuple(-value for value in _mv(a, x1))
    y2 = tuple(gamma * value for value in _mv(a, x1))
    x3 = tuple(-av + gamma * jv for av, jv in zip(_mv(a, x2), _mv(jac, y2)))
    y3 = tuple(gamma * value for value in _mv(a, x2))
    x4 = tuple(-av + gamma * jv for av, jv in zip(_mv(a, x3), _mv(jac, y3)))
    return tuple(row[13] for row in (x1, x2, x3, x4))


def test_held_out_derivatives_come_from_full_receiver_two_rows(frozen, mp):
    phi = INPUTS["phase_origin_difference"]
    by_class = []
    for donor_class, intervals in zip(
        (1, 2), frozen.receiver_derivative_bounds_by_donor
    ):
        expected = _full_receiver_jets(mp, donor_class, 2, phi)
        for bounds, value in zip(intervals, expected):
            _contains(mp, bounds, value)
        earlier_receiver = _full_receiver_jets(mp, donor_class, 1, phi)
        assert abs(expected[2] - earlier_receiver[2]) > mp.mpf("1e-16")
        assert all(
            abs(expected[i] - earlier_receiver[i]) < mp.mpf("1e-110") for i in (0, 1)
        )
        by_class.append(expected)
    assert all(
        abs(by_class[1][i] - by_class[0][i]) < mp.mpf("1e-110") for i in range(3)
    )
    _contains(
        mp,
        frozen.ideal_fourth_derivative_contrast_bounds,
        by_class[1][3] - by_class[0][3],
    )
    lift, project = _lifts()
    full_a, full_d = _full_normalized(), _full_normalized(EDGES[:9])
    u = tuple(Q((i == 4) - (i == 13), 3) for i in range(18))
    expected = (
        -_mv(full_a, _mv(full_d, _mv(full_a, u)))[13]
        - _mv(full_d, _mv(full_a, _mv(full_a, u)))[13]
    )
    assert expected == frozen.fourth_derivative_geometry_coefficient == Q(11, 81)
    assert (
        exact_matrix_product(project, exact_matrix_product(full_a, lift))
        == frozen.normalized_form_matrix
    )


def test_prediction_keeps_every_error_channel_and_outward_ratio(frozen, mp):
    h, phi, eps, d = (
        _mp(mp, INPUTS[name])
        for name in (
            "contact_duration",
            "phase_origin_difference",
            "endpoint_radius",
            "readout_error_bound",
        )
    )
    gamma, s = 1 / (1023 * mp.pi), mp.sin(phi)
    dc = mp.cos(2 * mp.pi / 9) - mp.cos(4 * mp.pi / 9)
    lead = mp.mpf(11) / 1944 * gamma**3 * s * dc * h**4
    terms = (
        4 * gamma**3 * dc * s * h**5 / (45 * (1 - h / 2)),
        mp.mpf(8) / 15 * s * gamma**5 * h**5,
        mp.mpf(16) / 45 * gamma**5 * s**2 * h**5 / (1 - 3 * h),
        2 * eps / (1 - 3 * h),
        2 * d,
    )
    reported = tuple(
        getattr(frozen, name)
        for name in (
            "reduced_semigroup_tail_upper_bound",
            "reduced_nonlinear_remainder_upper_bound",
            "surrogate_full_discrepancy_upper_bound",
            "preparation_response_error_upper_bound",
            "readout_contrast_error_upper_bound",
        )
    )
    assert all(_mp(mp, upper) >= value for upper, value in zip(reported, terms))
    assert frozen.total_error_upper_bound == sum(reported)
    assert _mp(mp, frozen.recorded_contrast_bounds.lo) <= lead - sum(terms)
    assert lead + sum(terms) <= _mp(mp, frozen.recorded_contrast_bounds.hi)
    assert (
        frozen.error_ratio_upper_bound
        == frozen.total_error_upper_bound / frozen.recorded_contrast_bounds.lo
    )
    assert frozen.error_fraction_margin_bounds.lo > 0
    assert (
        frozen.total_error_upper_bound
        < frozen.error_fraction * frozen.recorded_contrast_bounds.lo
    )
    assert frozen.recorded_contrast_bounds.lo > Q(1, 10**25)
    assert (
        frozen.status == "certified_reduced_class_ports"
        and frozen.unavailable_reasons == ()
    )
    assert (
        frozen.disconnected_recorded_contrast_bounds.hi
        < frozen.recorded_contrast_bounds.lo
    )
    assert (
        frozen.unprobed_handoff.formation_certificate.form_error_bound
        == INPUTS["form_error_bound"]
    )
    assert frozen.unprobed_handoff.exact_decay_upper_bound == Q(1, 2**512)


def test_fraction_touching_is_not_a_certificate(frozen):
    result = _assess(error_fraction=frozen.error_ratio_upper_bound)
    assert result.response_certified and not result.approximation_certified
    assert (
        result.error_fraction_margin_bounds.lo
        == result.error_fraction_margin_bounds.hi
        == 0
    )
    assert (
        result.status == "unavailable"
        and "relative_error_budget_not_certified" in result.unavailable_reasons
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"formation_time": 0},
        {"relaxation_duration": 0},
        {"endpoint_radius": Q(1, 2**500)},
        {"contact_duration": 0},
        {"phase_origin_difference": 0},
        {"readout_error_bound": Q(1, 10**10)},
        {"work_allowance": 0},
        {"error_fraction": Q(1, 1000)},
        {"phase_origin_difference": 1},
    ],
)
def test_unavailable_premises_do_not_create_success(changes):
    result = _assess(**changes)
    assert result.status == "unavailable" and result.unavailable_reasons
    if not all(result.unprobed_handoff.handoff_certified_by_class):
        for name in (
            "joined_contact_bounds",
            "recorded_contrast_bounds",
            "disconnected_recorded_contrast_bounds",
            "total_error_upper_bound",
            "preparation_response_error_upper_bound",
            "error_ratio_upper_bound",
            "error_fraction_margin_bounds",
        ):
            assert getattr(result, name) is None
        assert not result.response_certified and not result.identity_certified


@pytest.mark.parametrize("field", [name for name in INPUTS if name != "decay_power"])
@pytest.mark.parametrize("bad", [True, "1", 1j, float("inf"), float("nan")])
def test_assessor_original_scalar_admission_precedes_handoff(monkeypatch, field, bad):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached handoff")

    monkeypatch.setattr(owner, "_unprobed_handoff", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "changes",
    [
        {"error_fraction": 0},
        {"error_fraction": 1},
        {"error_fraction": -1},
        {"decay_power": True},
        {"decay_power": Q(512)},
        {"decay_power": -1},
        {"decay_power": 4097},
        {"formation_time": -1},
        {"relaxation_duration": -1},
        {"contact_duration": -1},
        {"contact_duration": Q(1, 3)},
        {"phase_origin_difference": -1},
        {"phase_origin_difference": 2},
        {"endpoint_radius": 0},
        {"radius": 0},
        {"radius": Q(1, 11)},
        {"form_error_bound": -1},
        {"phase_error_bound": -1},
        {"readout_error_bound": -1},
        {"work_allowance": -1},
    ],
)
def test_assessor_domains_precede_handoff(monkeypatch, changes):
    def forbidden(**kwargs):
        pytest.fail("invalid domain reached handoff")

    monkeypatch.setattr(owner, "_unprobed_handoff", forbidden)
    with pytest.raises(ValueError):
        _assess(**changes)


def test_shared_handoff_is_fresh_and_uses_original_normalized_primitives(monkeypatch):
    calls = []
    original = owner._unprobed_handoff

    def observe(**kwargs):
        result = original(**kwargs)
        calls.append((kwargs, result))
        return result

    monkeypatch.setattr(owner, "_unprobed_handoff", observe)
    a, b = _assess(form_error_bound=0), _assess(form_error_bound=Q(1, 2**300))
    assert len(calls) == 2 and a.unprobed_handoff is not b.unprobed_handoff
    assert calls[0][0]["form_error_bound"] == 0
    assert calls[1][0]["form_error_bound"] == Q(1, 2**300)
    with pytest.raises(TypeError):
        owner.assess_sine_reduced_class_ports(unprobed_handoff=a.unprobed_handoff)


def test_separate_exponential_work_caps(frozen):
    assert frozen.relaxation_duration / 5 > 4096
    with pytest.raises(ValueError, match="lyapunov_decay_rate"):
        _assess(
            relaxation_duration=Q(4097) / frozen.unprobed_handoff.lyapunov_decay_rate
        )
    with pytest.raises(ValueError, match="scaled_time"):
        _assess(formation_time=20481)


def test_report_sdk_projection_and_retained_evaluation(frozen, tmp_path):
    direct = frozen.to_dict()
    assert direct["schema"] == "tnfr.sine-reduced-class-ports.v1"
    assert relational_report_to_dict(frozen)["report"] == direct["report"]
    path = tmp_path / "reduced-ports.json"
    export_to_json(frozen, path)
    assert json_loads(path.read_bytes())["report"] == direct["report"]
    artifact = (
        Path(__file__).parents[2]
        / "docs/assets/sine_formed_classes/reduced-ports-v1.json"
    )
    assert json_loads(artifact.read_bytes()) == direct
    assert direct["report"]["receiver_class"] == 2
    assert (
        get_type_hints(owner.assess_sine_reduced_class_ports)["return"]
        is owner.SineReducedClassPorts
    )
    assert (
        get_type_hints(owner.evaluate_sine_reduced_class_ports)["return"]
        is owner.SineReducedClassPortState
    )
    get_type_hints(owner.SineReducedClassPorts)
