"""Exact geometry, detached law admission and provenance without trajectories."""

import json
import pickle
from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.phase_cycle_geometry import (
    classify_c5_sine_critical_set,
    derive_phase_cycle_geometry,
)
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_equilibria import assess_sine_asymptotic_equilibria
from tnfr.sdk import export_to_json

MODEL = RelationalExchangeModel(1, phase_domain="regular")
CYCLES = (tuple(range(5)), tuple(range(5, 10)))


def _graph():
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    for row in CYCLES:
        graph.add_edges_from(zip(row, row[1:] + row[:1]))
    graph.add_edges_from(((0, 10), (5, 10)))
    for i in graph:
        graph.nodes[i].update(
            EPI=Q(i - 4, 8),
            theta=Q((-1) ** i * (i + 1), 2),
            nu_f=Q(3, 2) if i == 6 else Q(1, 2) if i == 9 else 1,
        )
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _comparison(graph=None, *, model=MODEL):
    return bound_relational_sine_exchange(
        _graph() if graph is None else graph, reference_model=model
    )


def test_complete_law_admits_arbitrary_finite_nonacute_initial_state_and_exact_means():
    source = _comparison()
    report = source.asymptotic_equilibria(cycles=CYCLES)
    assert report.source is source
    assert report.admitted
    assert report.single_relative_equilibrium_convergence_certified
    assert report.full_lifted_state_convergence_certified
    assert not report.hypothesis_failures
    assert report.critical_set.relative_state_count == 3600
    assert report.selected_equilibrium_status == "unavailable_no_basin_selection"
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    assert report.conserved_form_mean == sum(
        mu * x for mu, x in zip(weights, source.epi)
    ) / sum(weights)
    assert report.conserved_lifted_phase_mean == sum(
        mu * t for mu, t in zip(weights, source.phase)
    ) / sum(weights)
    assert max(abs(source.phase[j] - source.phase[i]) for i, j in source.edges) > 3
    assert report.critical_set.geometry.nodes == source.nodes
    assert "no_convergence_rate_deadline_or_finite_step_certificate" in report.scope


def test_convergence_scope_uses_positive_coefficients_and_every_exact_capacity():
    graph = _graph()
    graph.nodes[10]["nu_f"] = Q(1, 2**200)
    model = RelationalExchangeModel(
        2, epi_weight=3, phase_weight=1, phase_domain="regular"
    )
    source = _comparison(graph, model=model)
    report = source.asymptotic_equilibria(cycles=CYCLES)
    assert report.admitted
    assert source.capacity[10] == Q(1, 2**200)
    assert report.conserved_form_mean is not None
    assert report.conserved_lifted_phase_mean is not None
    assert model.effective_weights != MODEL.effective_weights


@pytest.mark.parametrize(
    "zero_loss,zero_capacity", ((True, False), (False, True), (True, True))
)
def test_zero_boundaries_retain_geometry_but_withhold_asymptotic_theorem(
    zero_loss, zero_capacity
):
    graph = _graph()
    if zero_capacity:
        graph.nodes[10]["nu_f"] = 0
    model = (
        RelationalExchangeModel(1, epi_weight=0, phase_weight=1, phase_domain="regular")
        if zero_loss
        else MODEL
    )
    source = _comparison(graph, model=model)
    report = source.asymptotic_equilibria(cycles=CYCLES)
    assert not report.admitted
    assert report.status == "unavailable"
    assert not report.single_relative_equilibrium_convergence_certified
    assert not report.full_lifted_state_convergence_certified
    assert report.critical_set.relative_state_count == 3600
    assert ("positive_epi_weight_required" in report.hypothesis_failures) == zero_loss
    assert (
        "strictly_positive_held_capacity_required" in report.hypothesis_failures
    ) == zero_capacity
    assert (report.conserved_form_mean is None) == zero_capacity
    assert (report.conserved_lifted_phase_mean is None) == zero_capacity


def test_reader_does_not_recapture_modify_or_evaluate_a_trajectory(monkeypatch):
    graph = _graph()
    source = _comparison(graph)
    before = pickle.dumps(graph)
    source_before = pickle.dumps(source)

    def no_capture(*args, **kwargs):
        raise AssertionError("asymptotic reader must reuse the original capture")

    monkeypatch.setattr(
        "tnfr.physics.relational_sine_comparison._capture_sine_state", no_capture
    )
    report = assess_sine_asymptotic_equilibria(source, cycles=CYCLES)
    assert report.admitted
    assert pickle.dumps(graph) == before
    assert pickle.dumps(source) == source_before


@pytest.mark.parametrize("change", ("extra_edge", "missing_bridge", "extra_node"))
def test_finite_catalog_does_not_transfer_to_a_different_support(change):
    graph = _graph()
    if change == "extra_edge":
        graph.add_edge(1, 6)
    elif change == "missing_bridge":
        # Retain a connected supplied graph while removing the live intermediary topology.
        graph.remove_edge(5, 10)
        graph.add_edge(0, 5)
    else:
        graph.add_node(11, EPI=0, theta=0, nu_f=1)
        graph.add_edge(10, 11)
    source = _comparison(graph)
    with pytest.raises(ValueError):
        source.asymptotic_equilibria(cycles=CYCLES)


@pytest.mark.parametrize(
    "cycles",
    (
        CYCLES[:1],
        (CYCLES[0], CYCLES[0]),
        ((0, 1, 2, 3, 11), CYCLES[1]),
        ((0, 2, 1, 3, 4), CYCLES[1]),
    ),
)
def test_declared_cycles_must_match_all_actual_labeled_support_edges(cycles):
    with pytest.raises(ValueError):
        _comparison().asymptotic_equilibria(cycles=cycles)


def test_model_specific_reader_rejects_alternative_mobility_and_wrong_source_type():
    source = _comparison(
        model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        )
    )
    with pytest.raises(TypeError):
        assess_sine_asymptotic_equilibria(
            source.with_current_squared_mobility(epsilon=1), cycles=CYCLES
        )
    with pytest.raises(TypeError):
        assess_sine_asymptotic_equilibria(_graph(), cycles=CYCLES)


@pytest.mark.parametrize(
    "field,value",
    (
        ("epi", (Q(99),)),
        ("phase", (Q(0),)),
        ("capacity", (Q(1),)),
        ("epi", (float("nan"),) * 11),
        ("phase", (float("inf"),) * 11),
        ("capacity", (float("nan"),) * 11),
        ("epi", (True,) * 11),
        ("phase", (True,) * 11),
        ("capacity", (True,) * 11),
        ("capacity", (Q(-1),) * 11),
        ("degrees", (1,) * 11),
        ("degrees", (True,) * 11),
    ),
)
def test_malformed_consumed_capture_rejects_before_any_theorem_admission(field, value):
    corrupted = replace(_comparison(), **{field: value})
    with pytest.raises((TypeError, ValueError)):
        corrupted.asymptotic_equilibria(cycles=CYCLES)


@pytest.mark.parametrize("model", (None, RelationalExchangeModel(1)))
def test_reference_model_requires_actual_regular_coefficient_contract(model):
    with pytest.raises(ValueError, match="regular reference"):
        replace(_comparison(), reference_model=model).asymptotic_equilibria(
            cycles=CYCLES
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("epi_weight", -1),
        ("phase_weight", True),
        ("phase_weight", 0),
        ("storage_scale", True),
        ("storage_scale", float("nan")),
    ),
)
def test_stored_model_coefficients_reject_before_convergence_geometry(
    field, value, monkeypatch
):
    from tnfr.physics import relational_sine_equilibria as owner

    source = _comparison()
    model = replace(MODEL)
    object.__setattr__(model, field, value)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid coefficients must reject before convergence geometry")

    monkeypatch.setattr(owner, "classify_c5_sine_critical_set", forbidden)
    with pytest.raises((TypeError, ValueError)):
        replace(source, reference_model=model).asymptotic_equilibria(cycles=CYCLES)


def test_consumed_finite_represented_coordinates_keep_exact_rational_means():
    source = _comparison()
    represented = replace(
        source, epi=(0.1,) * 11, phase=(0.3,) * 11, capacity=(0.2,) * 11
    )
    report = represented.asymptotic_equilibria(cycles=CYCLES)
    assert report.admitted
    assert report.conserved_form_mean == Q(0.1)
    assert report.conserved_lifted_phase_mean == Q(0.3)


def test_node_relabeling_and_declared_ring_order_preserve_finite_geometry():
    graph = _graph()
    mapping = {i: ("node", i) for i in graph}
    changed = nx.relabel_nodes(graph, mapping)
    # Explicit reversed insertion order is a coordinate change, not a quotient.
    ordered = nx.Graph()
    ordered.add_nodes_from(mapping[i] for i in reversed(range(11)))
    ordered.add_edges_from(changed.edges())
    for i in graph:
        ordered.nodes[mapping[i]].update(graph.nodes[i])
    ordered.graph.update(graph.graph)
    cycles = tuple(tuple(mapping[i] for i in row) for row in reversed(CYCLES))
    report = _comparison(ordered).asymptotic_equilibria(cycles=cycles)
    assert report.admitted
    critical = report.critical_set
    assert critical.relative_state_count == 3600
    assert critical.cycles == cycles
    assert critical.mediator == mapping[10]
    state = critical.reconstruct(cycle_choices=(0, 0), bridge_turns=(0, Q(1, 2)))
    assert state.nodal_turns[0] == 0
    assert tuple(state.edge_turns[i] for i in critical.bridge_edge_indices) == (
        0,
        Q(1, 2),
    )
    assert not any(state.symbolic_sine_coefficients)
    assert (
        report.conserved_form_mean
        == _comparison().asymptotic_equilibria(cycles=CYCLES).conserved_form_mean
    )


@pytest.mark.parametrize(
    "choices,bridges",
    (
        ((True, 0), (0, 0)),
        ((30, 0), (0, 0)),
        ((0,), (0, 0)),
        ((0, 0), (0, Q(1, 3))),
        ((0, 0), (True, 0)),
    ),
)
def test_exact_branch_choices_reject_unsupported_or_coerced_values(choices, bridges):
    critical = classify_c5_sine_critical_set(
        derive_phase_cycle_geometry(_graph()), cycles=CYCLES
    )
    with pytest.raises((TypeError, ValueError)):
        critical.reconstruct(cycle_choices=choices, bridge_turns=bridges)


def test_reconstruction_revalidates_derived_catalog_metadata():
    critical = _comparison().asymptotic_equilibria(cycles=CYCLES).critical_set
    corrupted = replace(critical, relative_state_count=1)
    with pytest.raises(ValueError, match="fields"):
        corrupted.reconstruct(cycle_choices=(0, 0), bridge_turns=(0, 0))


def test_public_exact_reports_export_through_shared_atomic_writer(tmp_path):
    report = _comparison().asymptotic_equilibria(cycles=CYCLES)
    critical = report.critical_set
    state = critical.reconstruct(cycle_choices=(0, 0), bridge_turns=(0, Q(1, 2)))
    for name, value in (("law", report), ("catalog", critical), ("state", state)):
        payload = value.to_dict()
        path = tmp_path / (name + ".json")
        export_to_json(payload, path)
        assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert report.to_dict()["report"]["source"] == report.source.to_dict()["report"]
    assert state.to_dict()["report"]["edge_turns"][critical.bridge_edge_indices[1]] == {
        "numerator": 1,
        "denominator": 2,
    }
    assert report.to_dict()["schema"] == "tnfr.relational-sine-asymptotic-equilibria.v1"
    assert len(critical.cycle_edge_turn_options) == 30


def test_nested_label_export_does_not_encode_opaque_labels_as_rational_evidence():
    report = _comparison().asymptotic_equilibria(cycles=CYCLES)
    corrupted = replace(
        report, critical_set=replace(report.critical_set, mediator=Q(1, 3))
    )
    with pytest.raises(TypeError, match="node labels"):
        corrupted.to_dict()


def _choice_with_mask(critical, mask):
    return critical.cycle_supplementary_masks.index(mask)


@pytest.mark.parametrize(
    "masks,bridges,expected",
    (
        ((0, 0), (0, 0), (10, 0, 0)),
        ((1, 0), (0, 0), (9, 1, 0)),
        ((0, 0), (Q(1, 2), 0), (9, 1, 0)),
        ((3, 3), (0, 0), (6, 4, 0)),
        ((15, 31), (Q(1, 2), Q(1, 2)), (1, 9, 0)),
        ((31, 31), (Q(1, 2), Q(1, 2)), (0, 10, 0)),
    ),
)
def test_selected_phase_inertia_remains_separate_from_full_law_modes(
    masks, bridges, expected
):
    asymptotic = _comparison().asymptotic_equilibria(cycles=CYCLES)
    critical = asymptotic.critical_set
    choices = tuple(_choice_with_mask(critical, mask) for mask in masks)
    hessian = critical.phase_hessian_inertia(
        cycle_choices=choices, bridge_turns=bridges
    )
    stability = asymptotic.classify_equilibrium(
        cycle_choices=choices, bridge_turns=bridges
    )
    assert hessian.state == critical.reconstruct(
        cycle_choices=choices, bridge_turns=bridges
    )
    assert hessian.relative_inertia == expected
    assert hessian.common_phase_nullity == 1
    assert hessian.cycle_negative_edge_counts == tuple(
        mask.bit_count() for mask in masks
    )
    assert hessian.bridge_signs == tuple(1 if turn == 0 else -1 for turn in bridges)
    assert stability.phase_hessian == hessian
    assert stability.relative_unstable_modes == expected[1]
    assert stability.relative_stable_modes + stability.relative_unstable_modes == 20
    assert stability.relative_center_modes == 0
    assert stability.local_exponential_attraction_certified == (expected[1] == 0)
    assert stability.nonlinear_instability_certified == (expected[1] > 0)
    assert stability.status == (
        "locally_exponentially_attracting"
        if expected[1] == 0
        else "nonlinearly_unstable"
    )
    assert stability.source.source is asymptotic.source
    assert (
        "no_initial_state_terminal_branch_selection_or_receiver_transfer_verdict"
        in stability.scope
    )


def test_factorized_inertia_counts_do_not_materialize_the_full_catalog(monkeypatch):
    critical = _comparison().asymptotic_equilibria(cycles=CYCLES).critical_set

    def no_product(*args, **kwargs):
        raise AssertionError("the index distribution must not reconstruct 3600 members")

    monkeypatch.setattr(type(critical), "reconstruct", no_product)
    counts = critical.phase_hessian_index_counts
    assert len(counts) == 11
    assert counts[0] == 9
    assert sum(counts[1:]) == 3591
    assert sum(counts) == critical.relative_state_count
    assert all(type(value) is int and value > 0 for value in counts)


@pytest.mark.parametrize("zero_loss,zero_capacity", ((True, False), (False, True)))
def test_selected_stability_rechecks_law_instead_of_trusting_old_report_flags(
    zero_loss,
    zero_capacity,
):
    initial = _comparison().asymptotic_equilibria(cycles=CYCLES)
    graph = _graph()
    if zero_capacity:
        graph.nodes[10]["nu_f"] = 0
    model = (
        RelationalExchangeModel(1, epi_weight=0, phase_weight=1, phase_domain="regular")
        if zero_loss
        else MODEL
    )
    different_source = _comparison(graph, model=model)
    stale_report = replace(initial, source=different_source)
    assert stale_report.admitted
    result = stale_report.classify_equilibrium(
        cycle_choices=(0, 0), bridge_turns=(0, 0)
    )
    assert result.status == "unavailable"
    assert not result.source.admitted
    assert result.relative_stable_modes is None
    assert result.relative_unstable_modes is None
    assert result.relative_center_modes is None
    assert not result.local_exponential_attraction_certified
    assert not result.nonlinear_instability_certified
    assert result.phase_hessian.relative_inertia == (10, 0, 0)
    assert ("positive_epi_weight_required" in result.hypothesis_failures) == zero_loss
    assert (
        "strictly_positive_held_capacity_required" in result.hypothesis_failures
    ) == zero_capacity


def test_selected_stability_revalidates_consumed_source_and_catalog_metadata():
    initial = _comparison().asymptotic_equilibria(cycles=CYCLES)
    with pytest.raises(ValueError, match="vectors"):
        replace(
            initial, source=replace(initial.source, epi=(Q(1),))
        ).classify_equilibrium(cycle_choices=(0, 0), bridge_turns=(0, 0))
    corrupted_catalog = replace(
        initial.critical_set, cycle_supplementary_masks=(0,) * 30
    )
    with pytest.raises(ValueError, match="critical-set"):
        replace(initial, critical_set=corrupted_catalog).classify_equilibrium(
            cycle_choices=(0, 0), bridge_turns=(0, 0)
        )
    with pytest.raises(ValueError, match="critical-set"):
        _ = corrupted_catalog.phase_hessian_index_counts
    with pytest.raises(ValueError, match="critical-set"):
        corrupted_catalog.phase_hessian_inertia(
            cycle_choices=(0, 0), bridge_turns=(0, 0)
        )


def test_geometry_and_stability_share_one_pass_branch_iterables_without_recapture(
    monkeypatch,
):
    source = _comparison()
    report = source.asymptotic_equilibria(cycles=CYCLES)
    before = pickle.dumps(source)

    def no_capture(*args, **kwargs):
        raise AssertionError("a selected stability reader must not recapture the graph")

    monkeypatch.setattr(
        "tnfr.physics.relational_sine_comparison._capture_sine_state", no_capture
    )
    result = report.classify_equilibrium(
        cycle_choices=(i for i in (0, 0)), bridge_turns=(turn for turn in (0, Q(1, 2)))
    )
    assert result.nonlinear_instability_certified
    assert result.phase_hessian.cycle_choices == (0, 0)
    assert result.phase_hessian.bridge_turns == (0, Q(1, 2))
    assert pickle.dumps(source) == before


def test_positive_tiny_capacity_and_changed_coefficients_keep_mode_counts_not_rates():
    graph = _graph()
    graph.nodes[10]["nu_f"] = Q(1, 2**200)
    model = RelationalExchangeModel(
        2, epi_weight=3, phase_weight=1, phase_domain="regular"
    )
    report = _comparison(graph, model=model).asymptotic_equilibria(cycles=CYCLES)
    result = report.classify_equilibrium(cycle_choices=(0, 0), bridge_turns=(0, 0))
    assert result.local_exponential_attraction_certified
    assert result.relative_stable_modes == 20
    assert result.relative_center_modes == result.relative_unstable_modes == 0
    assert (
        "no_uniform_exponent_numeric_radius_or_specific_recovery_basin" in result.scope
    )


@pytest.mark.parametrize(
    "choices,bridges", (((True, 0), (0, 0)), ((30, 0), (0, 0)), ((0, 0), (0, Q(1, 3))))
)
def test_selected_stability_keeps_shared_exact_choice_admission(choices, bridges):
    report = _comparison().asymptotic_equilibria(cycles=CYCLES)
    with pytest.raises((TypeError, ValueError)):
        report.classify_equilibrium(cycle_choices=choices, bridge_turns=bridges)


def test_selected_stability_and_geometry_export_preserve_scopes_and_nested_labels(
    tmp_path,
):
    report = _comparison().asymptotic_equilibria(cycles=CYCLES)
    result = report.classify_equilibrium(
        cycle_choices=(0, 0), bridge_turns=(0, Q(1, 2))
    )
    for name, value in (("stability", result), ("inertia", result.phase_hessian)):
        payload = value.to_dict()
        path = tmp_path / (name + ".json")
        export_to_json(payload, path)
        assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert result.to_dict()["schema"] == "tnfr.relational-sine-equilibrium-stability.v1"
    assert (
        result.phase_hessian.to_dict()["schema"] == "tnfr.c5-phase-hessian-inertia.v1"
    )
    assert result.to_dict()["report"]["relative_unstable_modes"] == 1
    bad_inertia = replace(result.phase_hessian, cycles=((Q(1, 3),), CYCLES[1]))
    with pytest.raises(TypeError, match="node labels"):
        replace(result, phase_hessian=bad_inertia).to_dict()
