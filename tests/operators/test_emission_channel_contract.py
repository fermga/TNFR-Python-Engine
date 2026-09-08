"""AL remains an EPI-only source across runtime, metrics and advice."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import Emission
from tnfr.operators.operator_contracts import StateChannel, contract_for
from tnfr.operators.preconditions.coherence import diagnose_coherence_readiness
from tnfr.operators.preconditions.mutation import diagnose_mutation_readiness
from tnfr.operators.preconditions.resonance import diagnose_resonance_readiness


def _graph(*, epi: float = 0.0, vf: float = 1.0) -> nx.Graph:
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(
            EPI=epi,
            nu_f=vf,
            DeltaNFR=0.2,
            theta=0.1,
            epi_history=[0.0, 0.2],
            glyph_history=["IL", "OZ"],
        )
    return graph


def test_emission_runtime_and_metrics_are_epi_only() -> None:
    graph = _graph()
    graph.graph["COLLECT_OPERATOR_METRICS"] = True
    before = tuple(
        get_attr(graph.nodes[0], aliases)
        for aliases in (ALIAS_VF, ALIAS_DNFR, ALIAS_THETA)
    )

    Emission()(graph, 0)

    assert get_attr(graph.nodes[0], ALIAS_EPI) > 0.0
    assert tuple(
        get_attr(graph.nodes[0], aliases)
        for aliases in (ALIAS_VF, ALIAS_DNFR, ALIAS_THETA)
    ) == before
    metrics = graph.graph["operator_metrics"][-1]
    assert metrics["emission_quality"] == "valid"
    assert metrics["emission_effective"] is True
    assert metrics["delta_vf"] == pytest.approx(0.0)
    assert metrics["frequency_preserved"] is True
    assert metrics["capacity_active"] is True
    assert metrics["frequency_activation"] is False


def test_saturated_emission_satisfies_non_decrease_without_frequency_write() -> None:
    graph = _graph(epi=1.0)
    graph.graph.update(COLLECT_OPERATOR_METRICS=True, EPI_MAX=1.0)

    Emission()(graph, 0)

    metrics = graph.graph["operator_metrics"][-1]
    assert get_attr(graph.nodes[0], ALIAS_EPI) == pytest.approx(1.0)
    assert get_attr(graph.nodes[0], ALIAS_VF) == pytest.approx(1.0)
    assert metrics["emission_quality"] == "valid"
    assert metrics["emission_effective"] is False
    assert metrics["frequency_preserved"] is True


def test_emission_contract_declares_channel_purity() -> None:
    contract = contract_for("emission")

    assert contract.primary_channel is StateChannel.EPI
    assert contract.postcondition == (
        "EPI not decreased; νf, phase and ΔNFR unchanged"
    )


def test_frequency_advice_never_claims_emission_raises_capacity() -> None:
    graph = _graph(epi=0.5, vf=0.0)
    reports = (
        diagnose_coherence_readiness(graph, 0),
        diagnose_resonance_readiness(graph, 0),
        diagnose_mutation_readiness(graph, 0),
    )

    for report in reports:
        advice = " ".join(report["recommendations"])
        assert "AL (Emission)" not in advice
        assert "NAV (Transition)" in advice
