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

    result = hierarchy.evolve_multiscale(dt=dt, steps=1)

    for scale_name, graph in hierarchy.networks_by_scale.items():
        assert graph.graph["_t"] == pytest.approx(dt)
        for node in graph:
            data = graph.nodes[node]
            rate = data["nu_f"] * data["delta_nfr"]
            assert data["EPI"] == pytest.approx(
                before[(scale_name, node)] + dt * rate
            )
            assert data["dEPI_dt"] == pytest.approx(rate)
        assert result.scale_results[scale_name]["coherence"] == pytest.approx(
            hierarchy._scale_coherence(graph)
        )


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


def test_multiscale_coherence_includes_stored_structural_change_rate() -> None:
    hierarchy = _hierarchy(parallel=False)
    micro = nx.empty_graph(2)
    macro = nx.empty_graph(1)
    micro.nodes[0].update(delta_nfr=1.0, dEPI_dt=2.0)
    micro.nodes[1].update(delta_nfr=-3.0, dEPI_dt=-4.0)
    macro.nodes[0].update(delta_nfr=0.0, dEPI_dt=1.0)
    hierarchy.networks_by_scale = {"micro": micro, "macro": macro}

    assert hierarchy._scale_coherence(micro) == pytest.approx(1.0 / 6.0)
    assert hierarchy._scale_coherence(macro) == pytest.approx(1.0 / 2.0)
    assert hierarchy.compute_total_coherence() == pytest.approx(3.0 / 14.0)
    assert hierarchy.compute_total_coherence() != pytest.approx(5.0 / 18.0)

def test_public_cross_scale_direction_matches_source_to_target_contract() -> None:
    hierarchy = _hierarchy(parallel=False)
    hierarchy.networks_by_scale = {
        "micro": nx.empty_graph(1),
        "macro": nx.empty_graph(1),
    }
    hierarchy.cross_scale_couplings = {}
    hierarchy.networks_by_scale["micro"].nodes[0]["delta_nfr"] = 1.0
    hierarchy.networks_by_scale["macro"].nodes[0]["delta_nfr"] = 2.0

    hierarchy.set_cross_scale_coupling("micro", "macro", 0.5)

    assert hierarchy.cross_scale_couplings == {("macro", "micro"): 0.5}
    assert hierarchy.compute_multiscale_dnfr(0, "micro") == pytest.approx(1.0)
    assert hierarchy.compute_multiscale_dnfr(0, "macro") == pytest.approx(2.5)
    hierarchy._apply_cross_scale_coupling()
    assert hierarchy.networks_by_scale["micro"].nodes[0][
        "delta_nfr"
    ] == pytest.approx(1.0)
    assert hierarchy.networks_by_scale["macro"].nodes[0][
        "delta_nfr"
    ] == pytest.approx(2.5)


def test_unimplemented_multiscale_operator_request_is_not_silently_ignored() -> None:
    hierarchy = _hierarchy(parallel=False)
    before = {
        (scale, node): data["EPI"]
        for scale, graph in hierarchy.networks_by_scale.items()
        for node, data in graph.nodes(data=True)
    }

    with pytest.raises(NotImplementedError, match="operator execution"):
        hierarchy.evolve_multiscale(steps=1, operators=["THOL"])

    after = {
        (scale, node): data["EPI"]
        for scale, graph in hierarchy.networks_by_scale.items()
        for node, data in graph.nodes(data=True)
    }
    assert after == before


@pytest.mark.parametrize(
    ("steps", "dt", "error"),
    [
        (True, 0.1, TypeError),
        (-1, 0.1, ValueError),
        (1, float("nan"), ValueError),
        (1, 0.0, ValueError),
    ],
)
def test_multiscale_evolution_rejects_invalid_time_domain(
    steps: object, dt: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _hierarchy(parallel=False).evolve_multiscale(steps=steps, dt=dt)


def test_scale_definitions_and_couplings_reject_ambiguous_domains() -> None:
    with pytest.raises(ValueError, match="unique"):
        HierarchicalTNFRNetwork(
            [
                ScaleDefinition("same", 1, 0.5),
                ScaleDefinition("same", 2, 0.5),
            ]
        )
    with pytest.raises(TypeError, match="node_count"):
        ScaleDefinition("invalid", True, 0.5)

    hierarchy = _hierarchy(parallel=False)
    with pytest.raises(ValueError, match="distinct"):
        hierarchy.set_cross_scale_coupling("micro", "micro", 0.5)
    with pytest.raises(ValueError, match="finite"):
        hierarchy.set_cross_scale_coupling("micro", "macro", float("nan"))


def test_multiscale_evolution_uses_each_scales_registered_pressure_hook() -> None:
    hierarchy = _hierarchy(parallel=False)
    calls: list[str] = []

    for scale_name, graph in hierarchy.networks_by_scale.items():
        def pressure_hook(target: nx.Graph, *, value: float = len(calls) + 1.0) -> None:
            calls.append(scale_name)
            for node in target:
                target.nodes[node]["delta_nfr"] = value

        graph.graph["compute_delta_nfr"] = pressure_hook

    hierarchy.cross_scale_couplings = {}
    hierarchy.evolve_multiscale(dt=0.1, steps=1)

    assert len(calls) == 2
    for graph in hierarchy.networks_by_scale.values():
        assert {graph.nodes[node]["delta_nfr"] for node in graph} <= {1.0, 2.0}


def test_cross_scale_commit_preserves_existing_pressure_alias_and_cache() -> None:
    from tnfr.alias import get_attr
    from tnfr.constants.aliases import ALIAS_DNFR

    hierarchy = _hierarchy(parallel=False)
    micro = nx.empty_graph(1)
    macro = nx.empty_graph(1)
    legacy_alias = ALIAS_DNFR[-1]
    micro.nodes[0][legacy_alias] = 1.0
    macro.nodes[0][legacy_alias] = 2.0
    micro.graph["_dnfrmax"] = 99.0
    macro.graph["_dnfrmax"] = 99.0
    hierarchy.networks_by_scale = {"micro": micro, "macro": macro}
    hierarchy.cross_scale_couplings = {
        ("micro", "macro"): 0.5,
        ("macro", "micro"): 0.5,
    }

    hierarchy._apply_cross_scale_coupling()

    assert get_attr(micro.nodes[0], ALIAS_DNFR) == pytest.approx(2.0)
    assert get_attr(macro.nodes[0], ALIAS_DNFR) == pytest.approx(2.5)
    assert micro.graph["_dnfrmax"] == pytest.approx(2.0)
    assert macro.graph["_dnfrmax"] == pytest.approx(2.5)
    assert ALIAS_DNFR[0] not in micro.nodes[0] or ALIAS_DNFR[0] == legacy_alias


def test_cross_scale_reduction_rejects_nonfinite_pressure() -> None:
    hierarchy = _hierarchy(parallel=False)
    hierarchy.networks_by_scale["micro"].nodes[0]["delta_nfr"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        hierarchy._apply_cross_scale_coupling()