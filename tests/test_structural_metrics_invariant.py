"""Structural-metrics invariant checks canonical telemetry rather than keys."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.validation.invariants import (
    Invariant9_StructuralMetrics,
    InvariantSeverity,
)


def _graph(*, exposed: object, si: object = 1.2) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, delta_nfr=1.0, dEPI_dt=1.0, Si=si)
    graph.graph["coherence"] = exposed
    return graph


def test_matching_canonical_coherence_and_si_above_one_are_valid() -> None:
    violations = Invariant9_StructuralMetrics().validate(
        _graph(exposed=1.0 / 3.0, si=1.2)
    )

    assert violations == []


def test_arbitrary_coherence_key_does_not_certify_structural_metrology() -> None:
    violations = Invariant9_StructuralMetrics().validate(_graph(exposed=0.9))

    assert any("does not match" in item.description for item in violations)


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_exposed_coherence_domain_is_an_error(value: object) -> None:
    violations = Invariant9_StructuralMetrics().validate(_graph(exposed=value))

    assert any(
        item.severity is InvariantSeverity.ERROR
        and "invalid scalar domain" in item.description
        for item in violations
    )


@pytest.mark.parametrize("si", [True, -0.1, float("nan"), float("inf")])
def test_invalid_sense_index_is_reported_without_imposing_an_upper_bound(
    si: object,
) -> None:
    violations = Invariant9_StructuralMetrics().validate(
        _graph(exposed=1.0 / 3.0, si=si)
    )

    assert any("Sense index" in item.description for item in violations)