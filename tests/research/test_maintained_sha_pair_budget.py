"""Static pair-budget controls under the frozen boundary-budget protocol.

Only the two declared three-node preparations are scientific fixtures.
Other altered inputs exercise admission or arithmetic availability. No
retained source, event, pressure refresh or trajectory is evaluated here.
"""

import pickle
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from math import factorial
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.mathematics import _phase_resultant_chamber as phasor
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.research import maintained_sha_motion as owner

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
PHASES = ((Q(0), Q(0), Q(0)), (Q(0), Q(1, 4), Q(1, 2)))
ERRORS = (TypeError, ValueError, TNFRUserError)


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
    return owner._bound_pair_budget(
        graph, parent="parent", child="child", reference_model=MODEL
    )


def _agrees(actual, expected):
    assert (actual - expected).contains(0)
    assert (actual - expected).width < Q(1, 2**90)


def _shared_rate_agrees(actual, expected):
    # The captured support rows use the historical phasor owner's default
    # Taylor terms/grid, not the separate 128-bit sine evaluation above.
    # For these two fixtures every nonzero reduced cosine argument is <2.
    # Its omitted-term width is <=2^(2N)/(2N)!; two outward projections
    # add at most four grid units, plus the mathematical-pi range allowance.
    terms, bits = phasor._COSINE_TERMS, phasor._OUTPUT_BITS
    sine_width = (
        Q(2 ** (2 * terms), factorial(2 * terms))
        + Q(4, 2**bits)
        + 4 * pi_interval().width
    )
    # The largest sine-width coefficient in the relative form rows is
    # (4+1)/pi<2. The cut factor is 2/pi; the angle coefficient is smaller,
    # since |delta|<=1/4, sqrt(w)<2 and R^2>=2. Retain the separate small
    # 128-bit arithmetic allowance for the remaining fixed operations.
    assert (actual - expected).contains(0)
    assert (actual - expected).width < 2 * sine_width + Q(1, 2**90)


def _persistent_snapshot(graph):
    # NetworkX materializes read-only views lazily. Preserve the actual
    # graph/node/edge payloads, not those implementation view-cache objects.
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


@pytest.fixture(autouse=True)
def no_source_or_trajectory(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("fixed static controls cannot load retained sources or evolve")

    for name in (
        "validate_live_sha_successor",
        "validated_box_taylor_step",
        "certify_retention",
        "prepare_sha_motion",
    ):
        monkeypatch.setattr(owner, name, forbidden)


@pytest.mark.parametrize("phases", PHASES)
def test_fixed_ports_rates_storage_and_means_have_independent_expectations(phases):
    graph = _graph(phases)
    before = _persistent_snapshot(graph)
    report = _bound(graph)
    assert _persistent_snapshot(graph) == before
    connected, cut = report.connected, report.cut
    quarter, half = map(I, phases[1:])
    sine_child, sine_rest = sin(quarter), sin(half)
    pi, root_two = pi_interval(), sqrt(I(2))
    energy = 2 - cos(quarter)
    cut_storage = 7 - cos(half)
    norm = sqrt(36 + 2 * sine_rest**2)
    assert connected.source.nodes == cut.source.nodes == ("parent", "child", "rest")
    assert connected.source.degrees == (2, 1, 1)
    assert cut.source.degrees == (1, 0, 1)
    assert connected.source.epi == cut.source.epi == (Q(0), Q(1), Q(2))
    assert connected.source.phase == cut.source.phase == phases
    assert connected.source.capacity == cut.source.capacity == (Q(2), Q(3), Q(5))
    assert connected.reference_conductance == cut.reference_conductance == 2
    assert connected.form_difference == cut.form_difference == 1
    assert connected.phase_difference == cut.phase_difference == phases[1]
    assert connected.external_indices == cut.external_indices == (2,)
    assert connected.external_form_gradient == cut.external_form_gradient == -6
    assert connected.external_sine_sum == cut.external_sine_sum
    assert connected.pair_energy == cut.pair_energy
    assert connected.port_norm == cut.port_norm
    _agrees(connected.external_sine_sum, sine_rest)
    _agrees(connected.pair_energy, energy)
    _agrees(connected.radial_signal, sqrt(2 * energy))
    _agrees(connected.port_norm, norm)
    _agrees(connected.parent_mobility, 1 / pi)
    _agrees(connected.child_mobility, 3 / pi)
    _agrees(connected.internal_rotation_coefficient, 4 * root_two / pi)
    _agrees(cut.parent_mobility, 2 / pi)
    assert cut.child_mobility == cut.internal_rotation_coefficient == I(0)
    _shared_rate_agrees(
        connected.form_difference_rate, (-4 * sine_child - sine_rest) / pi
    )
    _agrees(connected.phase_difference_rate, 14 / pi)
    _shared_rate_agrees(cut.form_difference_rate, -2 * sine_rest / pi)
    _agrees(cut.phase_difference_rate, 12 / pi)
    _agrees(connected.pair_energy_rate, (6 * sine_child - 2 * sine_rest) / pi)
    _agrees(cut.pair_energy_rate, (12 * sine_child - 4 * sine_rest) / pi)
    for snapshot in (connected, cut):
        assert all(value.contains(0) for value in snapshot.pair_row_residuals)
        assert snapshot.power_residual.contains(0)
        assert snapshot.angle_unavailable_reason is None
    radius = phases[1] ** 2 + 2
    _shared_rate_agrees(
        connected.phase_portrait_rate,
        root_two * (phases[1] * (-4 * sine_child - sine_rest) - 14) / (pi * radius),
    )
    _shared_rate_agrees(
        cut.phase_portrait_rate,
        root_two * (-2 * phases[1] * sine_rest - 12) / (pi * radius),
    )
    assert connected.phase_portrait_rate.hi < 0
    assert cut.phase_portrait_rate.hi < 0
    _agrees(connected.source.storage, energy + cut_storage)
    _agrees(cut.source.storage, cut_storage)
    assert connected.mean_weights == (Q(1), Q(1, 3), Q(1, 5))
    assert cut.mean_weights == (Q(1, 2), Q(0), Q(1, 5))
    assert connected.form_mean == Q(11, 23)
    assert cut.form_mean == Q(4, 7)
    assert connected.phase_lift_mean == (Q(0) if phases == PHASES[0] else Q(11, 92))
    assert cut.phase_lift_mean == (Q(0) if phases == PHASES[0] else Q(1, 7))
    assert cut.positive_degree_indices == (0, 2)
    assert cut.source.inverse_phase_metric[1] is None
    assert cut.source.form_rates[1] == cut.source.phase_rates[1] == I(0)
    assert report.structural_time == 0


@pytest.mark.parametrize("phases", PHASES)
def test_joint_energy_lower_limit_and_necessary_margins_keep_their_scope(phases):
    report = _bound(_graph(phases))
    connected = report.connected
    rest_sine = sin(I(phases[2]))
    assert report.external_degree == 1
    assert report.external_conductance_sum == report.joint_energy_coefficient == 3
    assert report.initial_environment_storage == report.cut.source.storage
    assert report.storage_decomposition_residual.contains(0)
    _agrees(report.joint_ellipse_left, 12 + rest_sine**2)
    assert report.joint_ellipse_residual.lo >= -Q(1, 2**100)
    energy_norm = sqrt(6 * (7 - cos(I(phases[2]))))
    _agrees(report.energy_port_norm_lower_limit, energy_norm)
    assert report.cut_global_port_norm_bound == report.energy_port_norm_lower_limit
    # In particular the cut does not subtract its positive marked E again.
    assert (
        report.cut_global_port_norm_bound.lo
        > sqrt(6 * (report.cut.source.storage - connected.pair_energy)).hi
    )
    alpha0 = sqrt(36 + 2 * rest_sine**2) / (4 * sqrt(I(2)))
    alpha_energy = energy_norm / (4 * sqrt(I(2)))
    threshold = sqrt(2 * (2 - cos(I(phases[1])))) / (1 + sqrt(2 * pi_interval())) ** 2
    _agrees(report.initial_norm_alpha, alpha0)
    _agrees(report.energy_bound_alpha, alpha_energy)
    _agrees(report.necessary_alpha_threshold, threshold)
    _agrees(report.initial_norm_necessary_margin, threshold - alpha0)
    _agrees(report.energy_bound_necessary_margin, threshold - alpha_energy)
    _agrees(report.initial_norm_rotation_margin, connected.radial_signal - alpha0)
    _agrees(report.energy_bound_rotation_margin, connected.radial_signal - alpha_energy)
    assert report.uniform_norm_unavailable_reason is None
    assert report.initial_annulus_status == "eligible"
    assert report.initial_annulus_reasons == ()
    assert report.initial_principal_branch_certified
    assert report.initial_positive_subseparatrix_energy_certified
    with pytest.raises(FrozenInstanceError):
        report.structural_time = 1


@pytest.mark.parametrize("field", ("EPI", "theta", "nu_f"))
@pytest.mark.parametrize("bad", (None, False, float("nan"), float("inf"), 1j))
def test_invalid_consumed_node_primitives_reject(field, bad):
    graph = _graph()
    graph.nodes["parent"][field] = bad
    with pytest.raises(ERRORS):
        _bound(graph)


@pytest.mark.parametrize("field", ("EPI", "theta", "nu_f"))
def test_missing_consumed_node_primitives_reject(field):
    graph = _graph()
    del graph.nodes["parent"][field]
    with pytest.raises(ERRORS):
        _bound(graph)


@pytest.mark.parametrize("bad", (None, False, -1, 0, float("nan"), float("inf"), 1j))
def test_invalid_or_nonpositive_consumed_conductance_rejects(bad):
    graph = _graph()
    graph.edges["parent", "child"]["weight"] = bad
    with pytest.raises(ERRORS):
        _bound(graph)


def test_missing_weight_clock_and_nonpositive_capacity_are_not_invented():
    graph = _graph()
    del graph.edges["parent", "child"]["weight"]
    with pytest.raises(ValueError, match="stored conductance"):
        _bound(graph)
    graph = _graph()
    del graph.graph["_t"]
    with pytest.raises(ERRORS):
        _bound(graph)
    for capacity in (0, -1):
        graph = _graph()
        graph.nodes["child"]["nu_f"] = capacity
        with pytest.raises(ERRORS):
            _bound(graph)


@pytest.mark.parametrize(
    "model",
    (
        RelationalExchangeModel(2, epi_weight=0, phase_domain="regular"),
        RelationalExchangeModel(1, epi_weight=1, phase_domain="regular"),
        RelationalExchangeModel(1, epi_weight=0, phase_domain="acute"),
    ),
)
def test_other_complete_model_is_not_silently_normalized(model):
    with pytest.raises(ValueError):
        owner._bound_pair_budget(
            _graph(), parent="parent", child="child", reference_model=model
        )


def test_shape_and_marked_incidence_require_an_external_pendant_pair():
    graph = _graph()
    graph.add_edge("child", "rest", weight=1)
    with pytest.raises(ValueError, match="pendant"):
        _bound(graph)
    graph = _graph()
    with pytest.raises(ValueError, match="pendant"):
        owner._bound_pair_budget(
            graph, parent="rest", child="child", reference_model=MODEL
        )
    with pytest.raises(ValueError, match="distinct"):
        owner._bound_pair_budget(
            graph, parent="parent", child="parent", reference_model=MODEL
        )
    graph.remove_node("rest")
    with pytest.raises(ValueError, match="external"):
        _bound(graph)


def test_zero_and_unresolved_radius_are_unavailable_not_zero_angular_rates():
    graph = _graph()
    graph.nodes["child"]["EPI"] = 0
    zero = _bound(graph)
    for snapshot in (zero.connected, zero.cut):
        assert snapshot.phase_portrait_rate is None
        assert snapshot.angle_unavailable_reason == "zero_relative_state_radius"
    assert zero.initial_annulus_status == "excluded"
    assert not zero.initial_positive_subseparatrix_energy_certified
    graph.nodes["child"]["EPI"] = 1e-300
    unresolved = _bound(graph)
    assert unresolved.connected.phase_portrait_rate is None
    assert (
        unresolved.connected.angle_unavailable_reason
        == "relative_state_radius_not_separated_from_zero"
    )
    assert unresolved.initial_annulus_status == "unavailable"


def test_subgrid_rotation_coefficient_abstains_from_ratio_division():
    graph = _graph()
    graph.nodes["parent"]["nu_f"] = 5e-324
    graph.nodes["child"]["nu_f"] = 5e-324
    report = _bound(graph)
    assert (
        report.uniform_norm_unavailable_reason
        == "internal_rotation_coefficient_not_separated_from_zero"
    )
    assert report.initial_norm_alpha is report.energy_bound_alpha is None
    assert (
        report.initial_norm_necessary_margin
        is report.energy_bound_necessary_margin
        is None
    )


def test_energy_and_zero_branch_applicability_are_separate_from_motion():
    graph = _graph()
    graph.nodes["child"]["EPI"] = 2
    report = _bound(graph)
    assert report.initial_annulus_status == "excluded"
    assert report.connected.phase_portrait_rate is not None
    graph = _graph()
    graph.nodes["child"]["theta"] = 4
    report = _bound(graph)
    assert report.initial_annulus_status == "excluded"
    assert not report.initial_principal_branch_certified


def test_wrapper_revalidates_before_using_any_caller_cache(monkeypatch):
    passed = SimpleNamespace(graph=object(), prediction="forged", assessment="forged")
    calls = []

    def revalidate(value):
        calls.append(value)
        return SimpleNamespace(graph=_graph())

    monkeypatch.setattr(owner, "validate_live_sha_successor", revalidate)
    with pytest.raises(ValueError, match="21-node 53-edge"):
        owner.prepare_sha_pair_budget(passed)
    assert calls == [passed]
