"""Tests for assumption-explicit U5 coherence assessment."""

from __future__ import annotations

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
    graph.add_edges_from(
        [("parent", "child-a"), ("parent", "child-b")]
    )
    return graph


def test_u5_target_uses_canonical_parent_and_child_coherence() -> None:
    graph = _hierarchy()

    passing = assess_u5_parent_child_coherence(
        graph, "parent", alpha=0.2
    )
    failing = assess_u5_parent_child_coherence(
        graph, "parent", alpha=0.3
    )

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


def test_u5_target_rejects_inconsistent_hierarchy() -> None:
    graph = _hierarchy()
    graph.nodes["child-b"]["parent_node"] = "other"

    with pytest.raises(ValueError, match="declares parent"):
        assess_u5_parent_child_coherence(
            graph, "parent", alpha=0.2
        )


def test_amplitude_alignment_is_not_reported_as_u5_coherence() -> None:
    graph = _hierarchy()

    alignment = compute_subepi_amplitude_alignment(graph, "parent")

    assert alignment == pytest.approx(0.8)
    assert compute_subepi_collective_coherence(
        graph, "parent"
    ) == pytest.approx(alignment)
    assert alignment != pytest.approx(0.5)


def test_membrane_result_types_share_the_physics_namespace() -> None:
    assert MembraneNodeFlux.__module__.endswith(".cell")
    assert MembraneFluxResult.__module__.endswith(".cell")