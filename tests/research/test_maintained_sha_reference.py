"""Fixed static controls for the prospectively frozen reference audit.

Only the two registered three-node preparations are scientific fixtures.
Other inputs exercise numerical/domain availability or report wiring. No
retained source, new trajectory, event or pressure refresh is evaluated.
"""

import pickle
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.research import maintained_sha_motion as motion
from tnfr.research import maintained_sha_reference as owner

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
PHASES = ((Q(0), Q(0), Q(0)), (Q(0), Q(1, 4), Q(1, 2)))


def _graph(phases=PHASES[0]):
    graph = nx.Graph()
    for node, form, phase, capacity in zip(
        ("parent", "child", "rest"), (0, 1, 2), phases, (2, 3, 5), strict=True
    ):
        graph.add_node(node, EPI=form, theta=phase, nu_f=capacity, delta_nfr=99)
    graph.add_weighted_edges_from((("parent", "child", 2), ("parent", "rest", 3)))
    graph.graph.update(_t=0, GAMMA={"type": "none"})
    return graph


def _bound(graph):
    return owner._bound_reference(
        graph, parent="parent", child="child", reference_model=MODEL
    )


def _persistent(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _overlap(actual, expected):
    assert (actual - expected).contains(0)


@pytest.fixture(autouse=True)
def no_source_or_trajectory(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("manufactured reference controls cannot read or evolve the source")

    for name in (
        "validate_live_sha_successor",
        "validated_box_taylor_step",
        "certify_retention",
        "prepare_sha_motion",
    ):
        monkeypatch.setattr(motion, name, forbidden)
    monkeypatch.setattr(owner, "prepare_sha_pair_budget", forbidden)


@pytest.fixture(scope="module", params=PHASES)
def phase_case(request):
    # Cache only these manufactured evaluations, never a retained-source report.
    graph = _graph(request.param)
    before = _persistent(graph)
    report = _bound(graph)
    assert _persistent(graph) == before
    return request.param, report


def test_fixed_chart_exact_coefficients_and_shared_source_association(phase_case):
    phases, report = phase_case
    chart = report.chart
    assert chart.status == "available"
    assert report.unavailable_reasons == ()
    assert chart.gamma == Q(1, 4)
    assert chart.kappa == Q(35, 16)
    assert chart.form_center == Q(1, 4)
    assert chart.phase_center == phases[1] / 4
    assert chart.external_form_coefficient == Q(-21, 4)
    assert chart.form_reference == Q(-3, 5)
    assert chart.endpoint_force_margin == sin(I(1)).lo - Q(1, 4)
    assert chart.curvature_lower == cos(I(1)).lo - Q(1, 16)
    assert chart.curvature_upper == Q(17, 16)
    assert chart.relative_curvature_upper == Q(35, 32)
    assert chart.relative_curvature_lower == chart.curvature_lower
    alpha = phases[2] - phases[1] / 4
    _overlap(chart.cosine_coefficient, cos(I(alpha)))
    _overlap(chart.sine_coefficient, sin(I(alpha)))
    root = chart.root
    assert root.status == "available"
    assert root.bracket.width <= Q(1, 2**70)
    assert root.iterations <= 80
    assert root.original_endpoint_residuals[0].hi < 0
    assert root.original_endpoint_residuals[1].lo > 0
    # Independent one-neighbor equation, without C/S factorization.
    residual = sin(root.bracket) + sin(I(alpha) + root.bracket / 4) / 4
    assert residual.contains(0)
    pair = report.pair_budget
    assert (
        pair.connected.source.nodes
        == pair.cut.source.nodes
        == ("parent", "child", "rest")
    )
    assert pair.connected.source.degrees == (2, 1, 1)
    assert pair.cut.source.degrees == (1, 0, 1)
    assert pair.cut.source.form_rates[1] == pair.cut.source.phase_rates[1] == I(0)
    with pytest.raises(FrozenInstanceError):
        chart.gamma = Q(1, 2)


def test_relative_storage_has_independent_phase_difference_formula(phase_case):
    phases, report = phase_case
    chart, state = report.chart, report.state
    root = chart.root.bracket
    alpha = phases[2] - phases[1] / 4
    expected = (
        Q(14, 5)
        + cos(root)
        - cos(I(phases[1]))
        + cos(I(alpha) + root / 4)
        - cos(I(phases[2]))
    )
    _overlap(state.relative_storage, expected)
    assert state.relative_storage.subset_of(state.raw_relative_storage)
    assert state.relative_storage.subset_of(state.coercive_storage_enclosure)
    assert state.relative_storage.lo > 0
    _overlap(state.radial_excitation**2, 2 * state.relative_storage)
    _overlap(state.reference_radius_squared, root**2 + Q(18, 25))
    _overlap(state.error_radius_squared, (I(phases[1]) - root) ** 2 + Q(128, 25))
    assert state.geometric_offset_gate.status == "pass"
    assert state.coarse_reference_clearance.hi < state.actual_reference_clearance.lo
    assert state.zero_work_original_energy.contains(
        report.pair_budget.connected.pair_energy
    )
    if phases == PHASES[0]:
        assert root.contains(0)
        assert state.relative_storage.contains(Q(14, 5))
        _overlap(state.radial_excitation, sqrt(I(Q(28, 5))))


def test_complete_coefficient_and_reference_rates_use_independent_nodal_rows(
    phase_case,
):
    phases, report = phase_case
    rates, chart = report.rates, report.chart
    pi = pi_interval()
    alpha = phases[2] - phases[1] / 4
    alpha_rate = 69 / (2 * pi)
    # Direct differentiation of the three full nodal rows gives these values.
    _overlap(rates.center_form_rate, 3 * sin(I(phases[2])) / (4 * pi))
    _overlap(rates.center_phase_rate, -9 / (2 * pi))
    _overlap(rates.external_form_coefficient_rate, 69 * sin(I(phases[2])) / (4 * pi))
    _overlap(rates.cosine_coefficient_rate, -sin(I(alpha)) * alpha_rate)
    _overlap(rates.sine_coefficient_rate, cos(I(alpha)) * alpha_rate)
    expected_work = (
        sqrt(
            (69 * sin(I(phases[2])) / (4 * pi)) ** 2 / Q(35, 16)
            + alpha_rate**2 / chart.curvature_lower
        )
        / 4
    )
    # Overlap respects inherited 64-bit source-row enclosures; it is not a
    # claim that those rows acquire 128-bit accuracy from later arithmetic.
    _overlap(rates.work_norm, expected_work)
    root = chart.root.bracket
    expected_q_rate = (
        -alpha_rate
        * cos(I(alpha) + root / 4)
        / (4 * cos(root) + cos(I(alpha) + root / 4) / 4)
    )
    _overlap(rates.phase_reference_rate, expected_q_rate)
    if phases == PHASES[0]:
        assert rates.external_form_coefficient_rate.contains(0)
        assert rates.cosine_coefficient_rate.contains(0)
        _overlap(rates.sine_coefficient_rate, 69 / (2 * pi))
        _overlap(rates.phase_reference_rate, -138 / (17 * pi))
        _overlap(rates.work_norm, 69 / (8 * pi * sqrt(I(chart.curvature_lower))))


def test_registered_global_envelopes_retain_upper_bound_scope(phase_case):
    phases, report = phase_case
    envelope = report.global_envelope
    pi = pi_interval()
    # Forms contribute 1+6=7 and two phase edges add 2-cos(delta)-cos(theta).
    H = I(9) - cos(I(phases[1])) - cos(I(phases[2]))
    expected_PV = 69 / (4 * pi)
    expected_Pphi = sqrt(2 * H) * (5 * sqrt(I(3)) / pi + 3 * sqrt(I(3)) / (4 * pi))
    _overlap(envelope.form_coefficient_rate_bound, expected_PV)
    _overlap(envelope.phase_coefficient_rate_bound, expected_Pphi)
    _overlap(
        envelope.reference_radius_squared_bound, (1 / (4 * sin(I(1)))) ** 2 + 6 * H / 35
    )
    assert envelope.work_norm_bound.hi >= report.rates.work_norm.lo
    assert envelope.reference_radius_bound.hi >= report.state.reference_radius.lo
    assert (
        "chosen_upper_envelopes_are_not_actual_rate_or_offset_lower_bounds"
        in report.scope
    )


@pytest.mark.parametrize("root", (Q(0), Q(1, 3)))
def test_fixed_exact_force_roots_are_enclosed_without_midpoint_equality_inference(root):
    result = owner._refine_root(lambda q: I(q - root), Q(1))
    assert result.status == "available"
    assert result.bracket.contains(root)
    assert result.bracket.width <= Q(1, 2**70)


def test_unresolved_midpoint_retains_a_valid_box_and_abstains_on_resolution():
    result = owner._refine_root(lambda q: I(q) + I(Q(-1, 4), Q(1, 4)), Q(1))
    assert result.status == "unavailable"
    assert result.reason == "root_refinement_stalled"
    assert result.bracket == I(Q(-1, 4), Q(1, 4))
    assert result.bracket.contains(0)
    assert result.iterations == 2


def test_unresolved_endpoint_signs_do_not_claim_a_reference():
    result = owner._refine_root(lambda q: I(q) + I(-2, 2), Q(1))
    assert result.bracket is None
    assert result.reason == "endpoint_signs_unresolved"
    with pytest.raises(ValueError):
        owner._refine_root(lambda q: I(q), Q(0))


@pytest.mark.parametrize(
    "excitation,signal,energy",
    ((Q(1, 2), "excluded", "pass"), (Q(3, 2), "pass", "excluded")),
)
def test_necessary_signal_and_energy_work_gates_are_independent(
    excitation, signal, energy
):
    result = owner._work_budget(
        excitation=I(excitation),
        offset=I(0),
        work=I(Q(1, 4)),
        lower=Q(1),
        upper=Q(1),
        rotation=I(2),
    )
    assert result.offset_gate.status == "pass"
    _overlap(result.minimum_turn_time, pi_interval())
    _overlap(result.signal_work_gate.margin, excitation - pi_interval() / 4)
    _overlap(result.energy_work_gate.margin, 2 - excitation - pi_interval() / 4)
    assert result.signal_work_gate.status == signal
    assert result.energy_work_gate.status == energy


def test_zero_excitation_and_unresolved_gate_do_not_create_a_turn_time():
    result = owner._work_budget(
        excitation=I(0),
        offset=I(0),
        work=I(1),
        lower=Q(1),
        upper=Q(1),
        rotation=I(2),
    )
    assert result.offset_gate.status == "excluded"
    assert result.minimum_turn_time is None
    assert result.signal_work_gate.status == "unavailable"
    assert owner._gate(I(-1, 1)).status == "unavailable"


def test_outside_fixed_chart_and_failed_universal_package_remain_explicit():
    outside = _graph((Q(0), Q(2), Q(0)))
    report = _bound(outside)
    assert "actual_initial_phase_outside_fixed_chart" in report.unavailable_reasons
    assert report.state is report.rates is None
    bad_curvature = _graph()
    bad_curvature.nodes["parent"]["nu_f"] = 100
    report = _bound(bad_curvature)
    assert "universal_endpoint_margin_not_positive" in report.unavailable_reasons
    assert "universal_chart_curvature_not_positive" in report.unavailable_reasons
    assert report.chart.root is None
    assert report.state is None


@pytest.mark.parametrize(
    "attribute,value", (("EPI", True), ("nu_f", 0), ("theta", float("nan")))
)
def test_consumed_coordinate_admission_is_owned_by_shared_source(attribute, value):
    graph = _graph()
    graph.nodes["rest"][attribute] = value
    with pytest.raises((TypeError, ValueError, TNFRUserError)):
        _bound(graph)


def test_source_entry_always_rebuilds_budget_instead_of_consuming_caller_fields(
    monkeypatch,
):
    admitted, rebuilt, expected = object(), object(), object()
    calls = []

    def prepare(value):
        assert value is admitted
        calls.append("fresh_budget")
        return rebuilt

    def evaluate(value):
        assert value is rebuilt
        calls.append("evaluate_fresh_primitives")
        return expected

    monkeypatch.setattr(owner, "prepare_sha_pair_budget", prepare)
    monkeypatch.setattr(owner, "_reference_from_fresh_budget", evaluate)
    assert owner.prepare_sha_reference(admitted) is expected
    assert calls == ["fresh_budget", "evaluate_fresh_primitives"]
