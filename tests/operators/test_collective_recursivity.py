"""Synthetic associated-history controls for collective REMESH, without flow.

The supplied lag rows are test inputs, not observed research histories.
Public REMESH execution and shared sine observations own every reset/row.
"""

from collections import deque
from copy import deepcopy
from fractions import Fraction as Q
from math import pi

import networkx as nx
import pytest

from tnfr.alias import set_dnfr
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors import TNFRValueError
from tnfr.operators import apply_network_remesh, plan_network_remesh
from tnfr.operators.definitions import Coherence, Recursivity, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.preconditions import OperatorPreconditionError, validate_recursivity
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.physics.relational_sine_symmetry import assess_sine_support_involution

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
TWINS = (0, 1) + tuple(node ^ 1 for node in range(2, 20))


def _fields(graph):
    return bound_relational_sine_support_exchange(graph, reference_model=MODEL)


def _configure(graph, current, local, global_):
    graph.graph.update(
        REMESH_TAU_LOCAL=1,
        REMESH_TAU_GLOBAL=2,
        REMESH_ALPHA=0.5,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        CLIP_MODE="hard",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        _t=2.0,
    )
    for node, form in zip(graph, current, strict=True):
        graph.nodes[node].update(
            EPI=form,
            nu_f=1.0,
            theta=node / 16,
            delta_nfr=0.0,
            glyph_history=deque(("AL", "IL"), maxlen=8),
            _grammar_u2_debt=0,
            _grammar_prior_coherence=True,
        )
    graph.graph["_epi_hist"] = deque(
        [dict(zip(graph, row, strict=True)) for row in (global_, local, current)],
        maxlen=8,
    )
    return graph


def _path(current=(0.0, 0.0, 0.0), local=(1.0, 0.0, -1.0), global_=None):
    graph = nx.path_graph(3)
    graph[0][1]["weight"] = 1.0
    graph[1][2]["weight"] = 2.0
    return _configure(graph, current, local, local if global_ is None else global_)


def _contact():
    doubled = nx.lexicographic_product(nx.cycle_graph(5), nx.empty_graph(2))
    graph = nx.disjoint_union(doubled, doubled)
    nx.set_edge_attributes(graph, 1.0, "weight")
    graph.add_edge(0, 1, weight=7 / 8)
    for left in (0, 1):
        for right in (10, 11):
            graph.add_edge(left, right, weight=7 / 8)
    current = tuple(19 / 64 + (node // 2 % 5) / 128 for node in graph)
    local = tuple(value + 1 / 64 for value in current)
    global_ = tuple(21 / 64 - (node // 2 % 5) / 128 for node in graph)
    _configure(graph, current, local, global_)
    graph.graph.update(EPI_MIN=-1.0, EPI_MAX=1.0)
    for node in graph:
        graph.nodes[node]["theta"] = 2 * pi * (node % 10 // 2) / 5
        graph.nodes[node]["nu_f"] = 0.5 if node in (0, 1) else 1.0
    for node, pressure in zip(graph, _fields(graph).pressure, strict=True):
        set_dnfr(graph, node, float(pressure.midpoint))
    return graph


def _mean(source, values):
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    return sum(weight * value for weight, value in zip(weights, values)) / sum(weights)


def test_advisory_word_preserves_physical_object_without_mixing_history():
    graph = _contact()
    before = _fields(graph)
    lag_rows = deepcopy(graph.graph["_epi_hist"])
    sequence = ValidatedSequence((Recursivity(),))

    stage = execute_network_operator_stage(
        graph, Recursivity(), (0, 1), sequence_context=sequence.step(0)
    )

    assert stage.glyph == "REMESH"
    assert stage.schedule == "two_phase_jacobi"
    assert _fields(graph) == before
    assert graph.graph["_epi_hist"] == lag_rows
    assert "_REMESH_META" not in graph.graph
    assert len(graph.graph["history"]["events"]) == 1
    assert all(graph.nodes[node]["glyph_history"][-1] == "REMESH" for node in (0, 1))


def test_deeper_advisory_requires_word_stabilization_without_implying_mixing():
    with pytest.raises(TNFRValueError, match="U5"):
        ValidatedSequence((Recursivity(depth=2), Silence()))
    sequence = ValidatedSequence((Recursivity(depth=2), Coherence(), Silence()))
    graph = _contact()
    before = _fields(graph)

    execute_network_operator_stage(
        graph, Recursivity(depth=2), (0, 1), sequence_context=sequence.step(0)
    )

    assert _fields(graph) == before
    assert "_REMESH_META" not in graph.graph


def _invoke_advisory(graph, path):
    if path == "validator":
        validate_recursivity(graph, 0)
    elif path == "public":
        Recursivity()(graph, 0)
    else:
        execute_network_operator_stage(graph, Recursivity(), (0, 1))


@pytest.mark.parametrize("path", ["validator", "public", "stage"])
@pytest.mark.parametrize("minimum", [True, False, 1.5, 2.0, "2", -1])
def test_invalid_advisory_support_minimum_rejects_before_writes(path, minimum):
    graph = _path(current=(0.25, 0.25, 0.25))
    graph.graph["REMESH_MIN_NODES"] = minimum
    before = deepcopy(graph)

    with pytest.raises(OperatorPreconditionError, match="REMESH_MIN_NODES"):
        _invoke_advisory(graph, path)

    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("path", ["validator", "public", "stage"])
@pytest.mark.parametrize("minimum", [None, 0, 3], ids=["default_two", "zero", "three"])
def test_valid_advisory_support_minima_keep_the_existing_count_domain(path, minimum):
    graph = _path(current=(0.25, 0.25, 0.25))
    if minimum is not None:
        graph.graph["REMESH_MIN_NODES"] = minimum
    before = _fields(graph)

    _invoke_advisory(graph, path)

    assert _fields(graph) == before


@pytest.mark.parametrize("path", ["validator", "public", "stage"])
def test_valid_count_still_rejects_an_undersized_advisory_before_writes(path):
    graph = _path(current=(0.25, 0.25, 0.25))
    graph.graph["REMESH_MIN_NODES"] = 4
    before = deepcopy(graph)

    with pytest.raises(OperatorPreconditionError, match="Network too small"):
        _invoke_advisory(graph, path)

    assert nx.utils.graphs_equal(graph, before)


def test_missing_history_is_explicit_noop_not_an_inferred_temporal_echo():
    graph = _contact()
    graph.graph["_epi_hist"].clear()
    before = deepcopy(graph)

    result = apply_network_remesh(graph)

    assert result.status == "insufficient_history"
    assert not result.applied
    assert nx.utils.graphs_equal(graph, before)


def test_contact_delayed_mix_preserves_twin_identity_and_changes_phase_row():
    graph = _contact()
    before = _fields(graph)
    histories = deepcopy(graph.graph["_epi_hist"])
    glyph_histories = {
        node: tuple(graph.nodes[node]["glyph_history"]) for node in graph
    }
    expected = tuple(
        Q(graph.nodes[node]["EPI"]) / 4
        + Q(histories[-2][node]) / 4
        + Q(histories[-3][node]) / 2
        for node in graph
    )

    result = apply_network_remesh(graph)
    after = _fields(graph)

    assert result.applied
    assert after.epi == expected
    assert after.phase == before.phase
    assert after.capacity == before.capacity
    assert after.edges == before.edges
    assert after.conductances == before.conductances
    assert after.pressure == before.pressure
    assert after.form_rates == before.form_rates
    assert after.phase_storage == before.phase_storage
    assert after.phase_rate_numerators() != before.phase_rate_numerators()
    assert graph.graph["_epi_hist"] == histories
    assert glyph_histories == {
        node: tuple(graph.nodes[node]["glyph_history"]) for node in graph
    }
    symmetry = assess_sine_support_involution(after, permutation_indices=TWINS)
    assert symmetry.source_trajectory_reduction_certified
    displacement = tuple(y - x for x, y in zip(before.epi, after.epi))
    laplacian_change = tuple(
        sum(
            Q(graph[node][peer]["weight"]) * (displacement[node] - displacement[peer])
            for peer in graph.neighbors(node)
        )
        for node in graph
    )
    assert tuple(
        new - old
        for old, new in zip(
            before.phase_rate_numerators(), after.phase_rate_numerators()
        )
    ) == tuple(
        before.capacity[node] * laplacian_change[node] / before.degrees[node]
        for node in graph
    )
    assert after.form_storage - before.form_storage == sum(
        gradient * step for gradient, step in zip(before.form_gradient, displacement)
    ) + sum(
        Q(data["weight"]) * (displacement[i] - displacement[j]) ** 2 / 2
        for i, j, data in graph.edges(data=True)
    )
    assert _mean(after, after.epi) == (
        _mean(before, before.epi) / 4
        + _mean(before, tuple(Q(histories[-2][node]) for node in graph)) / 4
        + _mean(before, tuple(Q(histories[-3][node]) for node in graph)) / 2
    )


def test_equal_means_do_not_make_historical_disagreement_passive():
    graph = _path()
    before = _fields(graph)

    result = apply_network_remesh(graph, include_stability_evidence=True)
    after = _fields(graph)

    assert result.applied
    assert before.form_storage == 0
    assert after.form_storage == Q(27, 32)
    assert _mean(before, before.epi) == _mean(after, after.epi) == 0
    assert before.pressure == after.pressure
    assert before.form_rates == after.form_rates
    assert after.phase_rate_numerators() == (Q(3, 4), Q(3, 8), Q(-3, 2))
    assert result.evidence is not None
    assert not result.evidence.observed_bounded_disagreement_nonincreasing
    assert result.evidence.raw_convex_disagreement_bound_certified


def test_lower_storage_lag_fields_supply_a_passive_exact_dyadic_mix():
    graph = _path(
        current=(1.0, 0.0, -1.0),
        local=(0.5, 0.0, -0.5),
        global_=(0.0, 0.0, 0.0),
    )
    current = _fields(graph)
    local_graph, global_graph = deepcopy(graph), deepcopy(graph)
    for node in graph:
        local_graph.nodes[node]["EPI"] = graph.graph["_epi_hist"][-2][node]
        global_graph.nodes[node]["EPI"] = graph.graph["_epi_hist"][-3][node]
    local, global_ = _fields(local_graph), _fields(global_graph)

    apply_network_remesh(graph)
    after = _fields(graph)

    assert current.form_storage == Q(3, 2)
    assert local.form_storage == Q(3, 8)
    assert global_.form_storage == 0
    assert after.form_storage == Q(27, 128)
    convex_bound = current.form_storage / 4 + local.form_storage / 4
    assert convex_bound - after.form_storage == Q(33, 128)
    assert after.form_storage < current.form_storage


def test_same_current_object_can_have_distinct_history_conditioned_successors():
    positive = _path()
    negative = _path(local=(-1.0, 0.0, 1.0))
    assert _fields(positive) == _fields(negative)

    apply_network_remesh(positive)
    apply_network_remesh(negative)
    first, second = _fields(positive), _fields(negative)

    assert first.epi == tuple(-value for value in second.epi)
    assert first.phase_rate_numerators() == tuple(
        -value for value in second.phase_rate_numerators()
    )
    assert first.form_rates == second.form_rates


def test_current_twin_symmetry_is_insufficient_without_associated_history():
    graph = _contact()
    assert assess_sine_support_involution(
        _fields(graph), permutation_indices=TWINS
    ).source_membership_certified
    graph.graph["_epi_hist"][-3][2] += 1 / 8

    apply_network_remesh(graph)
    after = _fields(graph)

    assert after.epi[2] - after.epi[3] == Q(1, 16)
    assert not assess_sine_support_involution(
        after, permutation_indices=TWINS
    ).source_membership_certified


def test_exact_history_combination_can_preserve_twins_by_cancellation():
    graph = _contact()
    graph.graph["_epi_hist"][-2][2] += 1 / 8
    graph.graph["_epi_hist"][-3][2] -= 1 / 16

    apply_network_remesh(graph)

    assert assess_sine_support_involution(
        _fields(graph), permutation_indices=TWINS
    ).source_membership_certified


def test_hard_clipping_preserves_identity_but_changes_affine_mean_and_work():
    graph = _path(current=(0.25, 0.25, 0.25), local=(2.0, 0.0, 0.0))
    graph.graph.update(EPI_MIN=0.0, EPI_MAX=0.5)
    before = _fields(graph)

    result = apply_network_remesh(graph)
    after = _fields(graph)

    assert result.plan.any_clipping_intervention
    assert tuple(row.raw_epi for row in result.proposals) == (1.5625, 0.0625, 0.0625)
    assert after.epi == (Q(1, 2), Q(1, 16), Q(1, 16))
    assert after.form_storage - before.form_storage == Q(49, 512)
    assert _mean(after, after.epi) == Q(11, 64)
    assert after.phase == before.phase
    assert after.form_rates == before.form_rates


@pytest.mark.parametrize("defect", ["missing_member", "underflow", "boolean"])
def test_invalid_associated_history_rejects_before_any_collective_write(defect):
    graph = _contact()
    if defect == "missing_member":
        del graph.graph["_epi_hist"][-2][19]
    elif defect == "underflow":
        graph.graph["_epi_hist"][-2][19] = Q(1, 10**400)
    else:
        graph.graph["_epi_hist"][-2][19] = True
    before = deepcopy(graph)

    with pytest.raises(TNFRValueError):
        apply_network_remesh(graph)

    assert nx.utils.graphs_equal(graph, before)


def test_zero_local_coefficient_does_not_skip_runtime_history_admission():
    graph = _contact()
    graph.graph["REMESH_ALPHA"] = 1.0
    graph.graph["_epi_hist"][-2][19] = True

    with pytest.raises(TNFRValueError):
        plan_network_remesh(graph)


def test_explicit_mix_records_jump_boundary_without_creating_measured_rate():
    graph = _path()
    for node in graph:
        graph.nodes[node]["epi_time_history"] = deque(
            ((1.0, 0.0), (2.0, 0.0)), maxlen=8
        )
    delayed = deepcopy(graph.graph["_epi_hist"])

    result = apply_network_remesh(graph)

    assert result.epi_time_boundary_recorded
    assert tuple(graph.nodes[0]["epi_time_history"]) == ((2.0, 0.75),)
    assert tuple(graph.nodes[2]["epi_time_history"]) == ((2.0, -0.75),)
    assert tuple(graph.nodes[1]["epi_time_history"]) == ((1.0, 0.0), (2.0, 0.0))
    assert graph.graph["_epi_hist"] == delayed
