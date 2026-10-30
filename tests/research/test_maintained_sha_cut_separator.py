"""Constructed controls for static matched-cut invariant separation.

All graphs are manufactured preparations. No retained source, future field,
trajectory or observation-derived identity threshold is used by these tests.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics import _validated_taylor as taylor
from tnfr.mathematics._rational_interval import I, pi_interval, sqrt
from tnfr.research import maintained_sha_cut_separator as owner
from tnfr.research import maintained_sha_motion as motion
from tnfr.research import maintained_sha_motion_source as admission

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")


@pytest.fixture(autouse=True)
def no_retained_admission_or_evolution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("constructed separator controls cannot read or evolve a source")

    monkeypatch.setattr(owner, "validate_live_sha_successor", forbidden)
    monkeypatch.setattr(taylor, "picard_tube", forbidden)
    for name in (
        "validate_live_sha_successor",
        "validated_box_taylor_step",
        "certify_retention",
        "prepare_sha_motion",
    ):
        monkeypatch.setattr(motion, name, forbidden)


def _overlap(actual, expected):
    assert (actual - expected).contains(0)


def _snapshot(graph):
    # NetworkX may populate lazy views. Compare persistent graph state only.
    return deepcopy(
        (
            dict(graph.graph),
            tuple(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
        )
    )


def _path(
    *,
    forms=(0, 3, Q(1, 4), Q(-1, 2)),
    phases=(0, 0, 0, 0),
    capacities=(2, 3, 2, 4),
    form_shift=Q(0),
    phase_shift=Q(0),
):
    graph = nx.Graph()
    for node, form, phase, capacity in zip(
        ("parent", "child", "rest", "remote"),
        forms,
        phases,
        capacities,
        strict=True,
    ):
        graph.add_node(
            node,
            EPI=form + form_shift,
            theta=phase + phase_shift,
            nu_f=capacity,
            delta_nfr=99,
        )
    graph.add_weighted_edges_from(
        (("parent", "child", 2), ("parent", "rest", 3), ("rest", "remote", 5))
    )
    graph.graph.update(_t=0, GAMMA={"type": "none"})
    return graph


def _two_node_host(*, forms=(0, 0, 0), phases=(0, 0, 0)):
    graph = nx.Graph()
    for node, form, phase, capacity in zip(
        ("parent", "child", "rest"), forms, phases, (1, 3, 1), strict=True
    ):
        graph.add_node(node, EPI=form, theta=phase, nu_f=capacity, delta_nfr=99)
    graph.add_weighted_edges_from((("parent", "child", 2), ("parent", "rest", 1)))
    graph.graph.update(_t=0, GAMMA={"type": "none"})
    return graph


def _bound(graph):
    return owner._bound_cut_separator(
        graph, parent="parent", child="child", reference_model=MODEL
    )


@pytest.fixture(scope="module")
def unequal_host_case():
    graph = _path()
    before = _snapshot(graph)
    result = _bound(graph)
    assert _snapshot(graph) == before
    return graph, result


def test_cut_rebuilds_unequal_mobilities_means_and_preserves_the_frozen_child(
    unequal_host_case,
):
    graph, result = unequal_host_case
    source, cut = result.source, result.cut
    assert source.nodes == ("parent", "child", "rest", "remote")
    assert source.connected_degrees == (2, 1, 2, 1)
    assert source.cut_degrees == (1, 0, 2, 1)
    assert source.host_indices == (0, 2, 3)
    assert source.host_mean_weights == (Q(1, 2), Q(1), Q(1, 4))
    assert source.host_alpha == (Q(2, 7), Q(4, 7), Q(1, 7))
    assert source.form_mean == Q(1, 14)
    assert source.form_offset == Q(41, 14)
    assert source.reference_conductance == 2
    assert cut.degrees == source.cut_degrees
    assert cut.isolate_indices == (1,)
    assert cut.epi == source.epi
    assert cut.phase == source.phase
    assert cut.capacity == source.capacity
    assert cut.form_rates[1] == I(0)
    assert cut.phase_rates[1] == I(0)
    assert cut.pressure[1] == I(0)
    assert cut.inverse_phase_metric[1] is None
    # Parent capacity2 now divides degree1: its complete phase rate is
    # 2*3*(0-1/4)/pi, rather than the connected degree2 normalization.
    _overlap(cut.phase_rates[0], -I(Q(3, 2)) / pi_interval())
    assert result.storage.contains(Q(3, 2))
    assert result.form_gate.status == "pass"
    expected_margin = I(Q(41, 14) ** 2 - Q(384, 735))
    _overlap(result.form_gate.margin, expected_margin)
    assert result.form_radius_lower.lo > 0
    assert graph.has_edge("parent", "child")
    assert graph.nodes["child"]["EPI"] == 3
    with pytest.raises(FrozenInstanceError):
        result.storage = I(0)


def test_grounded_coefficients_have_independent_exact_residuals_and_gauge(
    unequal_host_case,
):
    _, result = unequal_host_case
    weighted, unweighted = result.weighted_response, result.unweighted_response
    laplacian = ((Q(3), Q(-3), Q(0)), (Q(-3), Q(8), Q(-5)), (Q(0), Q(-5), Q(5)))
    ell = (Q(5, 7), Q(-4, 7), Q(-1, 7))
    solution = (Q(4, 15), Q(1, 35), Q(0))
    assert weighted.laplacian == laplacian
    assert weighted.ell == ell
    assert weighted.ground_index == 2
    assert weighted.solution == solution
    assert weighted.grounded_inverse == ((Q(8, 15), Q(1, 5)), (Q(1, 5), Q(1, 5)))
    assert weighted.inverse_verified is True
    assert unweighted.inverse_verified is True
    assert weighted.residual == (Q(0), Q(0), Q(0))
    assert weighted.resistance == Q(128, 735)
    assert unweighted.solution == (Q(6, 7), Q(1, 7), Q(0))
    assert unweighted.resistance == Q(26, 49)
    # This independently multiplies the full singular matrix, including the
    # dropped row, and changes the gauge without changing its functional.
    for shift in (Q(0), Q(3)):
        shifted = tuple(value + shift for value in solution)
        assert (
            tuple(
                sum(a * b for a, b in zip(row, shifted, strict=True))
                for row in laplacian
            )
            == ell
        )
        assert sum(a * b for a, b in zip(ell, shifted, strict=True)) == Q(128, 735)


def test_global_common_shifts_preserve_both_separators_and_registered_offsets(
    unequal_host_case,
):
    _, original = unequal_host_case
    shifted = _bound(_path(form_shift=Q(2), phase_shift=Q(1, 2)))
    assert shifted.source.form_mean == original.source.form_mean + 2
    assert shifted.source.phase_lift_mean == original.source.phase_lift_mean + Q(1, 2)
    assert shifted.source.form_offset == original.source.form_offset
    assert shifted.source.phase_offset == original.source.phase_offset
    assert shifted.weighted_response.resistance == original.weighted_response.resistance
    assert (
        shifted.unweighted_response.resistance
        == original.unweighted_response.resistance
    )
    _overlap(shifted.storage, original.storage)
    _overlap(shifted.form_gate.margin, original.form_gate.margin)
    _overlap(shifted.joint_margin, original.joint_margin)


def test_a_raw_edge_gap_seven_blocks_the_phase_route_without_blocking_form():
    # The physical phase has a small circular representative, but this test
    # retains the actual real lift7. No wrapping may admit the phase chart.
    result = _bound(_two_node_host(forms=(0, 2, 0), phases=(0, 0, 7)))
    assert result.form_gate.status == "pass"
    assert result.form_radius_lower.lo > 0
    assert result.chart_gate.status == "unavailable"
    assert result.energy_gate.status == "pass"
    assert result.joint_gate.status == "unavailable"
    assert result.joint_radius_lower is None


def test_zero_host_storage_can_pass_only_the_joint_separator():
    result = _bound(_two_node_host(phases=(0, Q(1, 4), 0)))
    assert result.storage == I(0)
    assert result.form_gate.status == "unavailable"
    assert result.form_radius_lower is None
    assert result.chart_gate.status == "pass"
    assert result.energy_gate.status == "pass"
    assert result.joint_gate.status == "pass"
    _overlap(result.joint_margin, 1 / pi_interval() ** 2)
    assert result.joint_radius_lower.lo > 0
    assert result.joint_radius_lower.lo <= Q(1, 4)
    assert result.joint_radius_lower.hi <= Q(1, 4) + Q(1, 2**60)


def test_child_at_host_means_has_no_separator_even_with_nonzero_marked_radius():
    result = _bound(_two_node_host(forms=(Q(-1, 4), 0, Q(1, 4))))
    assert result.source.form_offset == 0
    assert result.source.phase_offset == 0
    assert result.storage.contains(Q(1, 8))
    assert result.source.epi[1] - result.source.epi[0] == Q(1, 4)
    assert result.form_gate.margin.hi < 0
    assert result.joint_margin.hi < 0
    assert result.form_gate.status == result.joint_gate.status == "unavailable"
    assert result.form_radius_lower is None
    assert result.joint_radius_lower is None


def test_strict_form_equality_does_not_supply_a_positive_radius():
    result = _bound(_two_node_host(forms=(0, 0, 1)))
    assert result.weighted_response.resistance == Q(1, 4)
    assert result.storage.contains(Q(1, 2))
    assert result.source.form_offset == Q(-1, 2)
    assert result.form_gate.margin.contains(0)
    assert result.form_gate.status == "unavailable"
    assert result.form_radius_lower is None
    assert result.source.epi[0] == result.source.epi[1]
    assert result.source.phase[0] == result.source.phase[1]


def test_a_small_positive_form_margin_retains_a_positive_outward_radius():
    epsilon = Q(1, 2**120)
    result = _bound(_two_node_host(forms=(0, -epsilon, 1)))
    assert result.source.form_offset == -Q(1, 2) - epsilon
    assert result.form_gate.status == "pass"
    assert result.form_gate.margin.lo > 0
    assert result.form_radius_lower.lo > 0
    # The exact form-only distance is epsilon and the observation scale is
    # sqrt(2). Outward enclosures must retain this small positive result.
    _overlap(result.form_radius_lower, sqrt(I(2)) * epsilon)


def test_high_host_energy_blocks_only_the_joint_chart_certificate():
    result = _bound(_two_node_host(forms=(-2, 10, 2)))
    assert result.storage.contains(8)
    assert result.chart_gate.status == "pass"
    assert result.energy_gate.status == "unavailable"
    assert result.form_gate.status == "pass"
    assert result.form_radius_lower.lo > 0
    assert result.joint_gate.status == "unavailable"
    assert result.joint_radius_lower is None


@pytest.mark.parametrize("bad_weight", (None, 0, -1))
def test_stored_positive_conductance_is_required_before_cut_arithmetic(bad_weight):
    graph = _two_node_host()
    if bad_weight is None:
        del graph.edges["parent", "rest"]["weight"]
    else:
        graph.edges["parent", "rest"]["weight"] = bad_weight
    before = _snapshot(graph)
    with pytest.raises(ValueError):
        _bound(graph)
    assert _snapshot(graph) == before


def test_a_zero_host_capacity_is_not_given_an_inverse_mean_weight():
    graph = _two_node_host()
    graph.nodes["parent"]["nu_f"] = 0
    with pytest.raises(ValueError):
        _bound(graph)


def test_the_selected_child_must_be_pendant_before_the_matched_cut():
    graph = _two_node_host()
    graph.add_edge("child", "rest", weight=1)
    with pytest.raises(ValueError):
        _bound(graph)


@pytest.mark.parametrize(
    "margin,status",
    (
        (I(1, 2), "pass"),
        (I(0), "unavailable"),
        (I(0, 1), "unavailable"),
        (I(-2, -1), "unavailable"),
        (I(-1, 1), "unavailable"),
        (None, "unavailable"),
    ),
)
def test_strict_recipe_gates_do_not_relabel_failure_as_observed_cut_motion(
    margin, status
):
    gate = owner._strict_gate(margin, reason="constructed_recipe")
    assert gate.status == status


def _constructed_wrapper_graph():
    # This graph supplies only the public boundary's fixed shape/clock for a
    # mocked consumer. Its fields are not an imitation of a retained source.
    graph = nx.Graph()
    graph.add_nodes_from(range(20))
    graph.add_node("2_sub_0")
    pairs = tuple((i, j) for i in range(20) for j in range(i + 1, 20))
    graph.add_edges_from(pairs[:52], weight=1)
    graph.add_edge(2, "2_sub_0", weight=1)
    graph.graph.update(_t=owner.TIMES[-1], GAMMA={"type": "none"})
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1, delta_nfr=99)
    return graph


def test_public_wrapper_re_admits_before_deriving_and_copies_fresh_receipts(
    monkeypatch, unequal_host_case
):
    _, base = unequal_host_case
    graph = _constructed_wrapper_graph()
    fresh_receipts = ({"path": "constructed", "nested": {"version": 1}},)
    stale = SimpleNamespace(graph=object(), receipts=({"cached": True},))
    fresh = SimpleNamespace(graph=graph, receipts=fresh_receipts)
    expected = replace(
        base, source=replace(base.source, structural_time=owner.TIMES[-1])
    )
    calls = []

    def readmit(value):
        assert value is stale
        calls.append("fresh_admission")
        return fresh

    def bound(value, *, parent, child, reference_model):
        assert value is graph
        assert (parent, child) == (2, "2_sub_0")
        assert reference_model is owner.MODEL
        calls.append("derive_from_fresh_primitives")
        return expected

    monkeypatch.setattr(owner, "validate_live_sha_successor", readmit)
    monkeypatch.setattr(owner, "_bound_cut_separator", bound)
    result = owner.prepare_sha_cut_separator(stale)
    assert calls == ["fresh_admission", "derive_from_fresh_primitives"]
    assert result.source is expected.source
    assert result.source_receipts == fresh_receipts
    assert result.source_receipts is not fresh_receipts
    fresh_receipts[0]["nested"]["version"] = 2
    assert result.source_receipts[0]["nested"]["version"] == 1


@pytest.mark.parametrize("invalid", ("shape", "clock"))
def test_public_wrapper_checks_the_fresh_shape_and_clock_before_arithmetic(
    monkeypatch, invalid
):
    graph = _constructed_wrapper_graph()
    if invalid == "shape":
        graph.remove_edge(0, 1)
    else:
        graph.graph["_t"] = 0
    fresh = SimpleNamespace(graph=graph, receipts=())
    monkeypatch.setattr(owner, "validate_live_sha_successor", lambda value: fresh)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid fresh shape/clock must not reach separator arithmetic")

    monkeypatch.setattr(owner, "_bound_cut_separator", forbidden)
    with pytest.raises(ValueError):
        owner.prepare_sha_cut_separator(object())


def test_cached_unadmitted_reports_cannot_cross_the_public_boundary(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("an unadmitted object must be rejected before reading files")

    monkeypatch.setattr(admission, "load_sha_successor", forbidden)
    monkeypatch.setattr(
        owner, "validate_live_sha_successor", admission.validate_live_sha_successor
    )
    with pytest.raises(
        ValueError, match="file-backed committed SHA admission required"
    ):
        owner.prepare_sha_cut_separator(
            SimpleNamespace(graph=_two_node_host(), prediction={})
        )
