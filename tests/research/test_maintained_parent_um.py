"""Selection-domain and independent exact-algebra controls; no UM execution."""

from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_VF
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_parent_um import (
    CHILD,
    DECISION_GUARD,
    NUMERICAL_ALLOWANCE,
    _decision,
    _score,
    predict_parent_um,
    principal_difference,
)
from tnfr.research.maintained_parent_um_source import load_thol_successor

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def source():
    return load_thol_successor(
        ROOT / "docs/assets/maintained_parent_thol", repository_root=ROOT
    ).graph


@pytest.fixture(autouse=True)
def no_um_response(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("analytical admission must not propose or execute UM")

    for owner, name in (
        (_coupling_stage_kernel, "propose_coupling_target"),
        (_coupling_stage_kernel, "propose_coupling_stage"),
        (network_stage, "execute_coupling_stage"),
        (word_execution, "execute_network_operator_stage"),
    ):
        monkeypatch.setattr(owner, name, forbidden)


def test_complete_inventory_and_read_only_selection(source):
    before = _signature(source)
    result = predict_parent_um(source)
    expected_inventory = tuple(
        index
        for index, node in enumerate(source)
        if node != 2 and not source.has_edge(2, node)
    )
    assert tuple(row.candidate_index for row in result.candidates) == expected_inventory
    assert len(expected_inventory) == 16
    assert result.new_edges == tuple(
        (2, index) for index in (3, 10, 11, 12, 13, 14, 15, 20)
    )
    assert result.compatible_indices == (0, 1, 4, 5)
    assert result.phase_changed_indices == (0, 1, 2, 4, 5)
    assert result.capacity_changed_indices == (2,)
    assert _signature(source) == before


def test_exact_capacity_and_raw_pressure_are_separate_from_refresh(source):
    result = predict_parent_um(source)
    expected_capacity = Q(128) + result.capacity_factor * (Q(3, 4) - Q(128))
    assert result.capacity_after[2].contains(expected_capacity)
    assert result.capacity_after[2].hi < 128
    assert 0 < result.raw_pressure_after[2].lo < result.raw_pressure_before[2]
    for index in range(len(result.nodes)):
        if index != 2:
            assert result.capacity_after[index].contains(result.capacity_before[index])
            assert result.raw_pressure_after[index].contains(
                result.raw_pressure_before[index]
            )
    assert result.phase_after[10] == I(result.phase_before[10])
    assert result.phase_after[10].hi < 0  # A retained lift is not silently wrapped.
    assert result.phase_after[20] == I(result.phase_before[20])


def test_child_default_sense_and_all_old_nodes_participate(source):
    result = predict_parent_um(source)
    assert not any(key in source.nodes[CHILD] for key in ALIAS_SI)
    assert result.sense_indices == (Q(1, 2),) * 20 + (Q(0),)
    assert len(result.new_edges) == 8
    child = result.candidates[-1]
    assert child.selected and child.candidate_index == 20
    assert child.score_margin.lo > Q(27, 1000)
    assert all(
        row.prefilter_margin.abs_max > DECISION_GUARD for row in result.candidates
    )
    assert all(
        row.score_margin.lo > DECISION_GUARD
        for row in result.candidates
        if row.selected
    )


def test_ideal_intervals_do_not_silently_include_the_runtime_budget(source):
    result = predict_parent_um(source)
    assert result.numerical_allowance == NUMERICAL_ALLOWANCE
    assert all(value.width < NUMERICAL_ALLOWANCE / 1000 for value in result.phase_after)
    assert all(
        value.width < NUMERICAL_ALLOWANCE / 1000 for value in result.new_conductances
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("UM_FUNCTIONAL_LINKS", True),
        ("UM_CANDIDATE_COUNT", 1),
        ("UM_COMPAT_THRESHOLD", 0.1),
        ("UM_SYNC_VF", False),
        ("_node_sample", [CHILD]),
        ("DELTA_PHI_MAX", 1.0),
        ("RANDOM_SEED", True),
        ("RANDOM_SEED", 18),
        ("GRAMMAR_REJECTION_MODE", "substitute"),
        ("VALIDATE_OPERATOR_PRECONDITIONS", False),
        ("EDGE_AWARE_ENABLED", True),
        ("_t", 7.0),
        ("GLYPH_FACTORS", {"VAL_scale": 128.0, "UM_theta_push": 0.2}),
    ],
)
def test_selected_default_configuration_cannot_be_retuned(source, key, value):
    graph = source.copy()
    graph.graph[key] = value
    with pytest.raises(ValueError):
        predict_parent_um(graph)


@pytest.mark.parametrize(
    "node,aliases,value",
    [
        (CHILD, ALIAS_SI, 0.5),
        (0, ALIAS_SI, -0.1),
        (0, ALIAS_SI, True),
        (2, ALIAS_VF, 127.0),
        (CHILD, ALIAS_EPI, 0.0),
        (2, ALIAS_DNFR, float("nan")),
    ],
)
def test_primitive_domains_and_missing_sense_policy_are_not_coerced(
    source, node, aliases, value
):
    graph = source.copy()
    set_attr(graph.nodes[node], aliases, value)
    with pytest.raises((TypeError, ValueError)):
        predict_parent_um(graph)


def test_interval_branch_cut_and_unresolved_gate_abstain():
    with pytest.raises(ValueError, match="branch"):
        principal_difference(pi_interval(), I(0))
    with pytest.raises(ValueError, match="margin"):
        _decision(I(-NUMERICAL_ALLOWANCE, NUMERICAL_ALLOWANCE), "test")
    assert _decision(I(5 * NUMERICAL_ALLOWANCE), "test")
    assert not _decision(I(-5 * NUMERICAL_ALLOWANCE), "test")
    assert principal_difference(I(6), I(0)).subset_of(I(-Q(3, 10), -Q(1, 4)))


def test_form_affinity_cancellation_uses_the_represented_epsilon():
    x, y = Q(5, 16), Q(1, 8)
    epsilon = Q(1e-9)
    scaled = 1 - abs(x / x - y / x) / (abs(x) / x + abs(y) / x + epsilon / x)
    expected = Q(1, 2) + Q(1, 4) * (scaled + Q(1, 2))
    assert _score(x, y, Q(1, 2), Q(0), I(0)).contains(expected)
    assert not _score(x, y, Q(1, 2), Q(0), I(0)).contains(Q(1))
