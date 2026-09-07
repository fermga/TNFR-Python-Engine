"""Regression tests for canonical multiscale EPI evolution."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.multiscale.hierarchical import HierarchicalTNFRNetwork, ScaleDefinition


def _hierarchy(*, parallel: bool) -> HierarchicalTNFRNetwork:
    hierarchy = HierarchicalTNFRNetwork(
        [
            ScaleDefinition("micro", 3, 1.0, 1.0),
            ScaleDefinition("macro", 3, 1.0, 1.0),
        ],
        seed=7,
        parallel=parallel,
        max_workers=2,
    )
    phase_by_scale = {
        "micro": (0.0, 0.2, 0.8),
        "macro": (0.1, 0.5, 1.4),
    }
    for scale_name, graph in hierarchy.networks_by_scale.items():
        for node, phase in enumerate(phase_by_scale[scale_name]):
            graph.nodes[node].update(
                EPI=0.4 + 0.05 * node,
                nu_f=0.7 + 0.2 * node,
                phase=phase,
            )
    hierarchy.set_cross_scale_coupling("micro", "macro", 0.4)
    hierarchy.set_cross_scale_coupling("macro", "micro", 0.25)
    return hierarchy


@pytest.mark.parametrize("parallel", [False, True])
def test_multiscale_evolution_uses_the_stored_total_pressure(parallel: bool) -> None:
    hierarchy = _hierarchy(parallel=parallel)
    dt = 0.05
    before = {
        (scale_name, node): graph.nodes[node]["EPI"]
        for scale_name, graph in hierarchy.networks_by_scale.items()
        for node in graph
    }

    hierarchy.evolve_multiscale(dt=dt, steps=1)

    for scale_name, graph in hierarchy.networks_by_scale.items():
        assert graph.graph["_t"] == pytest.approx(dt)
        for node in graph:
            data = graph.nodes[node]
            rate = data["nu_f"] * data["delta_nfr"]
            assert data["EPI"] == pytest.approx(
                before[(scale_name, node)] + dt * rate
            )
            assert data["dEPI_dt"] == pytest.approx(rate)


def test_cross_scale_pressure_uses_a_simultaneous_source_snapshot() -> None:
    hierarchy = _hierarchy(parallel=False)
    hierarchy.networks_by_scale = {
        "micro": nx.empty_graph(1),
        "macro": nx.empty_graph(1),
    }
    hierarchy.networks_by_scale["micro"].nodes[0]["delta_nfr"] = 1.0
    hierarchy.networks_by_scale["macro"].nodes[0]["delta_nfr"] = 2.0
    hierarchy.cross_scale_couplings = {
        ("micro", "macro"): 0.5,
        ("macro", "micro"): 0.5,
    }

    hierarchy._apply_cross_scale_coupling()

    assert hierarchy.networks_by_scale["micro"].nodes[0]["delta_nfr"] == pytest.approx(
        2.0
    )
    assert hierarchy.networks_by_scale["macro"].nodes[0]["delta_nfr"] == pytest.approx(
        2.5
    )
