"""Tests for assumption-explicit U5 coherence assessment."""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DEPI, ALIAS_DNFR
from tnfr.operators.metabolism import (
    compute_subepi_amplitude_alignment,
    compute_subepi_collective_coherence,
)
from tnfr.physics import (
    MembraneFluxResult,
    MembraneNodeFlux,
    assess_u5_parent_child_coherence,
)


def _hierarchy() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(
        "parent",
        **{
            ALIAS_DNFR[0]: 1.0,
            ALIAS_DEPI[0]: 0.0,
            "sub_nodes": ["child-a", "child-b"],
            "sub_epis": [{"epi": 0.0}, {"epi": 1.0}],
        },
    )
    graph.add_node(
        "child-a",
        **{
            ALIAS_DNFR[0]: 0.0,
            ALIAS_DEPI[0]: 0.0,
            "parent_node": "parent",
        },
    )
    graph.add_node(
        "child-b",
        **{
            ALIAS_DNFR[0]: 0.0,
            ALIAS_DEPI[0]: 0.0,
            "parent_node": "parent",
        },
    )
    graph.add_edges_from([("parent", "child-a"), ("parent", "child-b")])
    return graph


def test_u5_target_uses_canonical_parent_and_child_coherence() -> None:
    graph = _hierarchy()

    passing = assess_u5_parent_child_coherence(graph, "parent", alpha=0.2)
    failing = assess_u5_parent_child_coherence(graph, "parent", alpha=0.3)

    assert passing.parent_coherence == pytest.approx(0.5)
    assert passing.tolerance == pytest.approx(0.0)
    assert passing.child_coherences == pytest.approx((1.0, 1.0))
    assert passing.required_parent_coherence == pytest.approx(0.4)
    assert passing.residual == pytest.approx(0.1)
    assert passing.satisfies_target is True
    assert failing.residual == pytest.approx(-0.1)
    assert failing.satisfies_target is False


@pytest.mark.parametrize(
    "alpha",
    [True, "0.2", -0.1, float("nan"), float("inf")],
)
def test_u5_target_rejects_implicit_or_invalid_alpha(alpha: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        assess_u5_parent_child_coherence(
            _hierarchy(), "parent", alpha=alpha  # type: ignore[arg-type]
        )


def test_u5_assessment_records_the_declared_tolerance() -> None:
    assessment = assess_u5_parent_child_coherence(
        _hierarchy(), "parent", alpha=0.2505, tolerance=0.001
    )

    assert assessment.residual == pytest.approx(-0.001)
    assert assessment.tolerance == pytest.approx(0.001)
    assert assessment.satisfies_target is True


@pytest.mark.parametrize("invalid", [True, "0.001", float("nan"), -0.1])
def test_u5_assessment_rejects_invalid_tolerance(invalid: object) -> None:
    with pytest.raises(ValueError):
        assess_u5_parent_child_coherence(
            _hierarchy(),
            "parent",
            alpha=0.2,
            tolerance=invalid,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid", [True, "1.0"])
def test_u5_assessment_rejects_non_numeric_node_channels(invalid: object) -> None:
    graph = _hierarchy()
    graph.nodes["parent"][ALIAS_DNFR[0]] = invalid

    with pytest.raises(ValueError, match="finite real scalar"):
        assess_u5_parent_child_coherence(graph, "parent", alpha=0.2)


@pytest.mark.parametrize("node", ["parent", "child-a", "child-b"])
@pytest.mark.parametrize(
    ("aliases", "label"), [(ALIAS_DNFR, "DeltaNFR"), (ALIAS_DEPI, "dEPI")]
)
def test_missing_u5_observation_rejects_without_fabricating_zero(node, aliases, label):
    graph = _hierarchy()
    for alias in aliases:
        graph.nodes[node].pop(alias, None)
    before = deepcopy(graph)

    with pytest.raises(ValueError, match=f"missing required stored {label}"):
        assess_u5_parent_child_coherence(graph, "parent", alpha=0.2)

    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("aliases", [ALIAS_DNFR, ALIAS_DEPI])
@pytest.mark.parametrize("invalid", [None, True, float("nan"), Fraction(1, 10**400)])
def test_invalid_authoritative_u5_channel_cannot_fall_back(aliases, invalid):
    graph = _hierarchy()
    graph.nodes["child-a"][aliases[0]] = invalid
    assert len(aliases) > 1
    graph.nodes["child-a"][aliases[1]] = 0.0

    with pytest.raises(ValueError, match="finite"):
        assess_u5_parent_child_coherence(graph, "parent", alpha=0.2)


@pytest.mark.parametrize("field", ["alpha", "tolerance"])
def test_nonzero_u5_policy_inputs_cannot_disappear_during_materialization(field):
    kwargs = {"alpha": 0.2, field: Fraction(1, 10**400)}

    with pytest.raises(ValueError, match="underflows"):
        assess_u5_parent_child_coherence(_hierarchy(), "parent", **kwargs)


def test_u5_required_coherence_overflow_rejects_instead_of_returning_infinity():
    with pytest.raises(ValueError, match="required parent coherence"):
        assess_u5_parent_child_coherence(
            _hierarchy(), "parent", alpha=sys.float_info.max
        )


def test_nonzero_required_coherence_cannot_underflow_to_a_passing_zero():
    graph = _hierarchy()
    graph.nodes["parent"]["sub_nodes"] = ["child-a"]
    graph.nodes["child-a"][ALIAS_DNFR[0]] = 3.0
    smallest = float.fromhex("0x0.0000000000001p-1022")

    with pytest.raises(ValueError, match="required parent coherence.*underflows"):
        assess_u5_parent_child_coherence(graph, "parent", alpha=smallest)


def test_explicit_zero_and_signed_stored_inputs_remain_valid():
    graph = _hierarchy()
    graph.nodes["parent"][ALIAS_DNFR[0]] = Fraction(-1)
    graph.nodes["parent"][ALIAS_DEPI[0]] = Fraction(-2)

    assessment = assess_u5_parent_child_coherence(
        graph, "parent", alpha=Fraction(0), tolerance=Fraction(0)
    )

    assert assessment.parent_coherence == 0.25
    assert assessment.child_coherences == (1.0, 1.0)
    assert assessment.required_parent_coherence == 0.0
    assert assessment.residual == 0.25
    assert assessment.satisfies_target


def test_present_secondary_aliases_are_explicit_observations():
    graph = _hierarchy()
    for node in graph:
        for aliases in (ALIAS_DNFR, ALIAS_DEPI):
            graph.nodes[node][aliases[1]] = graph.nodes[node].pop(aliases[0])

    assessment = assess_u5_parent_child_coherence(graph, "parent", alpha=0.2)

    assert assessment.parent_coherence == 0.5
    assert assessment.child_coherences == (1.0, 1.0)
    assert assessment.satisfies_target


def test_u5_uses_stable_shared_coherence_for_large_finite_explicit_inputs():
    graph = _hierarchy()
    graph.nodes["parent"][ALIAS_DNFR[0]] = sys.float_info.max
    graph.nodes["parent"][ALIAS_DEPI[0]] = -sys.float_info.max

    assessment = assess_u5_parent_child_coherence(graph, "parent", alpha=0)

    assert math.isfinite(assessment.parent_coherence)
    assert 0 < assessment.parent_coherence < sys.float_info.min
    assert assessment.required_parent_coherence == 0
    assert assessment.satisfies_target


def test_u5_target_rejects_inconsistent_hierarchy() -> None:
    graph = _hierarchy()
    graph.nodes["child-b"]["parent_node"] = "other"

    with pytest.raises(ValueError, match="declares parent"):
        assess_u5_parent_child_coherence(graph, "parent", alpha=0.2)


def test_amplitude_alignment_is_not_reported_as_u5_coherence() -> None:
    graph = _hierarchy()

    alignment = compute_subepi_amplitude_alignment(graph, "parent")

    assert alignment == pytest.approx(0.8)
    assert compute_subepi_collective_coherence(graph, "parent") == pytest.approx(
        alignment
    )
    assert alignment != pytest.approx(0.5)


def test_membrane_result_types_share_the_physics_namespace() -> None:
    assert MembraneNodeFlux.__module__.endswith(".cell")
    assert MembraneFluxResult.__module__.endswith(".cell")
