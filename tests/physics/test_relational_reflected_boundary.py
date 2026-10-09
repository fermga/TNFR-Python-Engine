"""Static boundary algebra and separate native-field approach controls.

The local-flow theorem, not sampled states, establishes finite-time access.
The floating checks below exercise the existing native field without taking
steps and without choosing a continuation of its undefined boundary value.
"""

import json
import math
import pickle
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics import _validated_taylor
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.physics import relational_reflected_boundary as owner
from tnfr.physics import relational_reflected_transit as transit
from tnfr.physics._relational_reflected_flow import evaluate_reflected_regular_flow

MODEL = RelationalExchangeModel(1, phase_domain="regular")
EDGES = tuple(
    (offset + i, offset + (i + 1) % 5) for offset in (0, 5) for i in range(5)
) + ((0, 5), (1, 6))
REPRESENTATIVES = (0, 4, 5, 9, 10, 14, 15, 19)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("a static boundary certificate must not evolve a state")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        patch.setattr(transit, "certify_relational_reflected_transit", forbidden)
        patch.setattr(transit, "validated_taylor_step", forbidden)
        patch.setattr(_validated_taylor, "validated_taylor_step", forbidden)
        yield


@pytest.fixture(scope="module")
def report(no_trajectory):
    return owner.certify_relational_reflected_boundary_exit(model=MODEL)


def _embed(values):
    p, r, capital_p, capital_r, a, b, capital_a, capital_b = values
    return (
        (p, -p, -r, I(0), r, capital_p, -capital_p, -capital_r, I(0), capital_r),
        (a, -a, -b, I(0), b, capital_a, -capital_a, -capital_b, I(0), capital_b),
    )


def _graph(epsilon, *, central_form_scale=0):
    b = math.pi / 2 - epsilon
    forms = (0, 0, -1, central_form_scale * math.cos(b), 1, 0, 0, 0, 0, 0)
    phases = (0, 0, -b, 0, b, 0, 0, 0, 0, 0)
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from(EDGES)
    for node in graph:
        graph.nodes[node].update(EPI=forms[node], theta=phases[node], nu_f=1)
    return graph


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def test_ideal_boundary_resultants_and_storage_from_independent_graph_sums(report):
    forms, phases = _embed(report.coordinates)
    graph = nx.Graph(EDGES)
    for node, expected in zip(report.row_order, report.limiting_resultants):
        gaps = tuple(phases[other] - phases[node] for other in graph[node])
        real = sum((cos(gap) for gap in gaps), I(0))
        imaginary = sum((sin(gap) for gap in gaps), I(0))
        for actual, ideal in zip((real, imaginary), expected):
            difference = actual - ideal
            assert difference.contains(0) and difference.width < Q(1, 10**30)
    storage = sum(
        (
            (forms[left] - forms[right]) ** 2 / 2
            + 1
            - cos(phases[left] - phases[right])
            for left, right in EDGES
        ),
        I(0),
    )
    assert (storage - report.storage).contains(0)
    assert report.storage == I(6)
    assert report.seven_beta_storage_gap == 1
    assert report.below_seven_beta
    assert report.limiting_resultants[2] == (I(0), I(0))


def test_resultant_derivatives_and_loss_from_independent_full_graph_work(report):
    forms, phases = _embed(report.coordinates)
    form_rates, phase_rates = _embed(report.limiting_rates)
    graph = nx.Graph(EDGES)
    for node, expected in zip(report.row_order, report.limiting_resultant_rates):
        derivative = (I(0), I(0))
        for other in graph[node]:
            gap = phases[other] - phases[node]
            rate = phase_rates[other] - phase_rates[node]
            derivative = (
                derivative[0] - sin(gap) * rate,
                derivative[1] + cos(gap) * rate,
            )
        for actual, ideal in zip(derivative, expected):
            difference = actual - ideal
            assert difference.contains(0) and difference.width < Q(1, 10**30)
    work = sum(
        (
            (forms[i] - forms[j]) * (form_rates[i] - form_rates[j])
            + sin(phases[i] - phases[j]) * (phase_rates[i] - phase_rates[j])
            for i, j in EDGES
        ),
        I(0),
    )
    assert (work + report.continuous_loss).contains(0)
    assert report.continuous_loss.contains(Q(7, 3))
    assert report.limiting_rates[5] == I(Q(1, 4))
    assert report.limiting_resultant_rates[2][0] == I(Q(-1, 2))
    assert report.limiting_rates[4].hi < 0
    # Backward time puts both critical edge gaps below pi/2.
    assert (report.limiting_rates[5] - report.limiting_rates[4]).lo > 0
    assert report.limiting_resultant_rates[1][0].hi < 0


def test_static_certificate_keeps_full_boundary_admission_closed(report):
    with pytest.raises(ValueError, match="all six full resultant"):
        evaluate_reflected_regular_flow(
            report.coordinates,
            epi_weight=Q(1, 2),
            phase_weight=Q(1, 2),
            storage_scale=1,
        )
    outside = (
        report.coordinates[:5]
        + (pi_interval() / 2 + Q(1, 100),)
        + (
            I(0),
            I(0),
        )
    )
    with pytest.raises(ValueError, match="all six full resultant"):
        evaluate_reflected_regular_flow(
            outside, epi_weight=Q(1, 2), phase_weight=Q(1, 2), storage_scale=1
        )


def test_materialized_native_approach_matches_limits_without_advancing(report):
    limiting = np.array([float(value.midpoint) for value in report.limiting_rates])
    errors = []
    for epsilon in (1e-3, 1e-5, 1e-7):
        graph = _graph(epsilon)
        before = _snapshot(graph)
        field = relational.evaluate_relational_exchange(graph, model=MODEL)
        observed = np.array(field.form_rate + field.phase_rate)[list(REPRESENTATIVES)]
        errors.append(float(np.max(np.abs(observed - limiting))))
        assert _snapshot(graph) == before
        assert field.phase_rate[3] == 0
        assert min(field.resultant_regular_margin_lower_bounds) > 0
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 1e-7


@pytest.mark.parametrize("scale", (-2, 0, 3))
def test_native_positive_resultant_approaches_have_distinct_phase_rate_limits(
    report, scale
):
    model = RelationalExchangeModel(1, phase_domain="positive_resultant")
    slope = float(report.full_state_phase_rate_slope.midpoint)
    for epsilon in (1e-3, 1e-5, 1e-7):
        graph = _graph(epsilon, central_form_scale=scale)
        field = relational.evaluate_relational_exchange(graph, model=model)
        assert field.phase_rate[3] == pytest.approx(scale * slope, abs=2e-9)
        assert abs(graph.nodes[3]["EPI"]) <= abs(scale) * epsilon * 1.01
        assert abs(field.form_rate[3]) <= abs(scale) * epsilon


@pytest.mark.parametrize(
    "amplitude,gap,below", ((Q(1, 2), Q(5, 2), True), (2, Q(-5), False))
)
def test_storage_gap_does_not_promote_all_amplitudes_to_low_energy(
    amplitude, gap, below
):
    report = owner.certify_relational_reflected_boundary_exit(
        model=MODEL, form_amplitude=amplitude
    )
    assert report.seven_beta_storage_gap == gap
    assert report.below_seven_beta is below
    assert report.limiting_resultant_rates[2][0].hi < 0


def test_zero_loss_model_still_has_transverse_boundary_access():
    model = RelationalExchangeModel(
        2, epi_weight=0, phase_weight=3, phase_domain="regular"
    )
    report = owner.certify_relational_reflected_boundary_exit(model=model)
    assert report.continuous_loss == I(0)
    assert report.limiting_rates[5] == I(Q(1, 4))
    assert report.storage == I(10)
    assert report.seven_beta_storage_gap == 4


def test_exact_threshold_is_not_reported_as_a_strict_low_energy_gap():
    report = owner.certify_relational_reflected_boundary_exit(
        model=RelationalExchangeModel(6, phase_domain="regular"), form_amplitude=3
    )
    assert report.seven_beta_storage_gap == 0
    assert not report.below_seven_beta


def test_unresolved_positive_transversality_is_not_rounded_into_a_certificate():
    with pytest.raises(ArithmeticError, match="transversality enclosure is unresolved"):
        owner.certify_relational_reflected_boundary_exit(
            model=MODEL, form_amplitude=Q(1, 2**256)
        )


@pytest.mark.parametrize("value", (True, False, 1.0, float("nan"), "1", None, I(1)))
def test_form_amplitude_requires_an_exact_nonboolean_scalar(value):
    with pytest.raises(TypeError, match="exact positive"):
        owner.certify_relational_reflected_boundary_exit(
            model=MODEL, form_amplitude=value
        )


@pytest.mark.parametrize("value", (0, -1, Q(-1, 2)))
def test_form_amplitude_must_be_positive(value):
    with pytest.raises(ValueError, match="must be positive"):
        owner.certify_relational_reflected_boundary_exit(
            model=MODEL, form_amplitude=value
        )


@pytest.mark.parametrize(
    "model",
    (
        None,
        {},
        RelationalExchangeModel(1),
        RelationalExchangeModel(1, phase_domain="positive_resultant"),
    ),
)
def test_model_must_explicitly_select_regular_domain(model):
    with pytest.raises(ValueError, match="explicit regular"):
        owner.certify_relational_reflected_boundary_exit(model=model)


def test_exact_report_projection_retains_boundary_scope_and_provenance(report):
    projected = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert projected["schema"] == "tnfr.relational-reflected-boundary-exit.v1"
    values = projected["report"]
    assert values["boundary_node"] == 3
    assert values["seven_beta_storage_gap"] == {"numerator": 1, "denominator": 1}
    assert values["below_seven_beta"] is True
    assert (
        "no_continuous_full_state_extension_even_from_positive_resultants"
        in values["scope"]
    )
    assert (
        "no_explicit_initial_state_hitting_time_or_frozen_response_prediction"
        in values["scope"]
    )
