"""Prior-only K3 sampling admission and independent static field controls."""

import json
import math
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research.relational_capacity_discriminator import (
    _chart_bounds,
    _field_jets,
    _flow_coefficients,
    certify_relational_capacity_sampling,
)
from tnfr.sdk import export_to_json


@pytest.fixture(scope="module")
def certificate():
    return certify_relational_capacity_sampling()


def _mp(value):
    return mp.mpf(value.numerator) / value.denominator


def _independent_field(state, capacity, beta, mediated):
    """High-precision phasor/Arg definition, independent of the K3 chart formula."""
    form, phase = state[:3], state[3:]
    gradient = tuple(
        sum(form[i] - form[j] for j in range(3) if j != i) for i in range(3)
    )
    resultants = tuple(
        sum(mp.exp(1j * (phase[j] - phase[i])) for j in range(3) if j != i)
        for i in range(3)
    )
    angles = tuple(mp.arg(value) for value in resultants)
    metric = tuple(
        mp.pi * abs(value) * (mp.sin(angle) / angle if angle else 1)
        for value, angle in zip(resultants, angles, strict=True)
    )
    form_rate = tuple(
        capacity[i] * (-gradient[i] / 4 + angles[i] / (2 * mp.pi)) for i in range(3)
    )
    phase_rate = [capacity[i] * gradient[i] / (2 * beta * metric[i]) for i in range(3)]
    if mediated:
        a = tuple(
            sum(mp.sin(phase[i] - phase[j]) for j in range(3) if j != i)
            for i in range(3)
        )
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                mobility = capacity[i] * capacity[j] / (capacity[i] + capacity[j])
                phase_rate[i] += (
                    mobility
                    * ((gradient[i] + gradient[j]) / 2)
                    * mp.sin(phase[j] - phase[i])
                    * a[j]
                    / (8 * beta)
                )
    return form_rate + tuple(phase_rate)


def test_fixed_certificate_covers_both_laws_and_both_arms(certificate):
    assert {(case.law, case.arm) for case in certificate.cases} == {
        (law, arm)
        for law in ("capacity_separable", "capacity_mediated")
        for arm in ("before", "after")
    }
    assert certificate.horizon == 2 * certificate.sample_step == Q(1, 32)
    assert certificate.beta_bounds == (Q(3, 4), Q(5, 4))
    assert certificate.acute_margin > 0
    assert certificate.sinc_argument_upper_bound < 1
    assert min(certificate.phase_metric_lower_bounds) > 0
    for case in certificate.cases:
        assert case.first_exit_margin > Q(3, 1000)
        assert max(case.coordinate_speed_upper_bounds) < Q(2, 5)
        assert (
            case.third_phase_contrast_upper_bound < certificate.third_derivative_bound
        )
        assert case.initial_phase_contrast_rate_bounds[0] > Q(1, 8)
    assert certificate.rate_error_bound == Q(11, 196608)
    assert certificate.rate_error_bound < certificate.rate_error_limit == Q(1, 2048)
    assert certificate.relative_rate_error_upper_bound < Q(1, 256)
    assert (
        "no_source_beta_response_samples_trajectory_fit_or_constitutive_selection"
        in certificate.scope
    )


@pytest.mark.parametrize("neighbor_capacity", (Q(1), Q(1, 2)))
@pytest.mark.parametrize("beta", (Q(3, 4), Q(5, 4)))
def test_separable_static_rates_match_native_owner_without_evolution(
    neighbor_capacity, beta
):
    graph = nx.complete_graph(3)
    form = (Q(1, 3), Q(0), Q(-1, 3))
    angles = (0, math.pi / 6, -math.pi / 6)
    capacity = (Q(1), neighbor_capacity, Q(1))
    for i in graph:
        graph.nodes[i].update(EPI=form[i], theta=angles[i], nu_f=capacity[i])
    native = evaluate_relational_exchange(graph, model=RelationalExchangeModel(beta))
    state = tuple(Jet.constant(I(value), 0) for value in (*form, *map(Q, angles)))
    analytic = _field_jets(state, capacity, I(beta), mediated=False)
    actual = native.form_rate + native.phase_rate
    for value, rate in zip(analytic, actual, strict=True):
        # Native coefficients are represented arithmetic, not an ideal enclosure.
        assert float(value.coeffs[0].midpoint) == pytest.approx(rate, abs=2e-15)


@pytest.mark.parametrize("neighbor_capacity", (Q(1), Q(1, 2)))
@pytest.mark.parametrize("mediated", (False, True))
@pytest.mark.parametrize("beta", (Q(3, 4), Q(5, 4)))
def test_shared_jet_third_derivative_encloses_independent_full_state_chain_rule(
    neighbor_capacity, mediated, beta
):
    pi = pi_interval()
    initial = (I(Q(1, 3)), I(0), I(Q(-1, 3)), I(0), pi / 6, -pi / 6)
    capacity = (Q(1), neighbor_capacity, Q(1))
    coefficients = _flow_coefficients(initial, capacity, I(beta), mediated=mediated)
    third = 6 * (coefficients[3][3] - coefficients[5][3])
    with mp.workdps(90):
        point = (mp.mpf(1) / 3, mp.mpf(0), -mp.mpf(1) / 3, 0, mp.pi / 6, -mp.pi / 6)
        capacities, storage = tuple(map(_mp, capacity)), _mp(beta)

        def field(state):
            return _independent_field(state, capacities, storage, mediated)

        speed = field(point)

        def moved(direction, parameter):
            return tuple(
                x + parameter * v for x, v in zip(point, direction, strict=True)
            )

        acceleration = tuple(
            mp.diff(lambda t: field(moved(speed, t))[i], 0) for i in range(6)
        )

        def relative(state):
            value = field(state)
            return value[3] - value[5]

        # y''' = D²(LF)[F,F] + D(LF)[DF F], using all six state coordinates.
        independent = mp.diff(lambda t: relative(moved(speed, t)), 0, 2)
        independent += mp.diff(lambda t: relative(moved(acceleration, t)), 0)
        assert _mp(third.lo) <= independent <= _mp(third.hi)


def test_whole_box_first_exit_and_third_bounds_retain_interior_state_controls(
    certificate,
):
    with mp.workdps(70):
        point = tuple(
            _mp((lower + 2 * upper) / 3) for lower, upper in certificate.box_bounds
        )
        for case in certificate.cases:
            value = _independent_field(
                point,
                tuple(map(_mp, case.capacity)),
                mp.mpf(7) / 8,
                case.law == "capacity_mediated",
            )
            assert all(
                abs(rate) <= _mp(bound)
                for rate, bound in zip(
                    value, case.coordinate_speed_upper_bounds, strict=True
                )
            )


@pytest.mark.parametrize("angles", ((0, Q(8, 5), 0), (0, Q(6, 5), Q(6, 5))))
def test_unsupported_boxes_cannot_inherit_chart_or_sinc_admission(angles):
    box = (I(0),) * 3 + tuple(map(I, angles))
    with pytest.raises(ValueError, match="acute|sinc"):
        _chart_bounds(box)


def test_certificate_is_immutable_and_uses_shared_exact_json_projection(
    certificate, tmp_path
):
    with pytest.raises(FrozenInstanceError):
        certificate.sample_error_bound = Q(0)
    with pytest.raises(FrozenInstanceError):
        certificate.cases[0].first_exit_margin = Q(0)
    payload = certificate.to_dict()
    path = tmp_path / "sampling-admission.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    encoded = payload["report"]["rate_error_bound"]
    assert (
        Q(encoded["numerator"], encoded["denominator"]) == certificate.rate_error_bound
    )
    payload["report"]["cases"][0]["capacity"][0]["numerator"] = 999
    assert certificate.cases[0].capacity[0] == 1
