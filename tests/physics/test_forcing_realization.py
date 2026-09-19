"""Nonlinear phase forcing, numerical residuals and graph-read boundaries."""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.operators import apply_glyph
from tnfr.physics import forcing_realization as realization
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)

F = Fraction


def _graph():
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    # The actual neighbor order differs from sorted snapshot support order.
    graph.add_edge(0, 2, weight=1.0)
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(0, 3, weight=0.0)
    for node, epi, capacity, phase in zip(
        graph,
        (1.0, 0.5, -0.25, 2.0),
        (1.0, 2.0, 4.0, 8.0),
        (0.0, 0.0, 0.0, math.pi / 2),
        strict=True,
    ):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=capacity,
            theta=phase,
            delta_nfr=0.0,
            glyph_history=["IL"],
        )
    graph.graph.update(
        DNFR_WEIGHTS={"phase": 1.0, "epi": 1.0, "vf": 1.0, "topo": 1.0},
        compute_delta_nfr=default_compute_delta_nfr,
    )
    return graph


def _pressure(graph):
    return tuple(
        F(float(get_attr(graph.nodes[node], ALIAS_DNFR, 0.0))) for node in graph
    )


def test_zero_weight_neighbor_contributes_nonlinear_phase_and_exact_support_forcing():
    graph = _graph()
    result = capture_non_epi_forcing(graph)
    assert result.normalized_weights == tuple(
        (name, F(1, 4)) for name in ("phase", "epi", "vf", "topo")
    )
    # Three phasors sum to 2+i; an arithmetic average of angles gives pi/6.
    expected_phase = math.atan2(1.0, 2.0) / math.pi
    assert float(result.phase_gradient[0]) == pytest.approx(expected_phase, abs=1e-16)
    assert abs(float(result.phase_gradient[0]) - F(1, 6)) > 0.01
    assert result.phase_gradient[3] == F(-1, 2)
    assert result.snapshot.epi_gradient[3] == 0
    assert result.snapshot.capacity_gradient == (F(11, 3), -1, -3, -7)
    assert result.snapshot.topology_gradient == (-2, 2, 2, 2)
    assert result.forcing[0] == result.phase_gradient[0] / 4 + F(5, 12)
    assert result.forcing[3] == F(-11, 8)
    without_zero_support = deepcopy(graph)
    without_zero_support.remove_edge(0, 3)
    assert capture_non_epi_forcing(without_zero_support).phase_gradient[0] == 0


def test_detached_channel_decomposition_reuses_exact_products_without_kernel_calls(
    monkeypatch,
):
    result = capture_non_epi_forcing(_graph())

    def forbidden(**kwargs):
        raise AssertionError("detached arithmetic cannot call a phase kernel")

    monkeypatch.setattr(
        realization.fused_dnfr, "compute_fused_gradients_symmetric", forbidden
    )
    channels = dict(decompose_non_epi_forcing(result))
    assert tuple(channels) == ("phase", "vf", "topo")
    assert channels["phase"] == tuple(value / 4 for value in result.phase_gradient)
    assert channels["vf"] == (F(11, 12), F(-1, 4), F(-3, 4), F(-7, 4))
    assert channels["topo"] == (F(-1, 2), F(1, 2), F(1, 2), F(1, 2))
    assert tuple(sum(row) for row in zip(*channels.values())) == result.forcing


def test_channel_decomposition_rebuilds_support_caches_and_excludes_pressure_defects():
    result = capture_non_epi_forcing(_graph())
    expected = decompose_non_epi_forcing(result)
    forged = replace(
        result,
        snapshot=replace(
            result.snapshot,
            capacity_gradient=(F(999),) * 4,
            topology_gradient=(F(999),) * 4,
            stored_pressure=(F(99),) * 4,
        ),
        kernel_pressure_defect=(F(99),) * 4,
        stored_pressure_residual=(F(99),) * 4,
    )
    assert decompose_non_epi_forcing(forged) == expected


@pytest.mark.parametrize(
    "field,value",
    (
        ("forcing", (F(0),) * 4),
        ("epi_weight", F(1, 2)),
        ("phase_gradient", (F(0),)),
        ("normalized_weights", (("phase", F(1)),)),
        (
            "normalized_weights",
            tuple((key, F(-1, 4)) for key in ("phase", "epi", "vf", "topo")),
        ),
    ),
)
def test_channel_decomposition_rejects_inconsistent_coefficients(field, value):
    result = capture_non_epi_forcing(_graph())
    with pytest.raises(ValueError):
        decompose_non_epi_forcing(replace(result, **{field: value}))


def test_channel_decomposition_requires_its_explicit_detached_type():
    with pytest.raises(TypeError, match="NonEpiForcingObservation"):
        decompose_non_epi_forcing({"forcing": (0, 0)})


def test_fresh_full_kernel_matches_runtime_but_exact_channel_assembly_has_own_defect():
    graph = _graph()
    default_compute_delta_nfr(graph)
    result = capture_non_epi_forcing(graph)
    assert result.full_kernel_pressure == _pressure(graph)
    assert result.stored_pressure_residual == (0,) * 4
    assert any(result.kernel_pressure_defect)
    for index in range(4):
        assert result.full_kernel_pressure[index] == (
            result.epi_weight * result.snapshot.epi_gradient[index]
            + result.forcing[index]
            + result.kernel_pressure_defect[index]
        )


def test_named_pressure_write_does_not_become_an_extra_forcing_channel():
    graph = _graph()
    default_compute_delta_nfr(graph)
    before = capture_non_epi_forcing(graph)
    # One named primitive pressure operation; no complete-word claim follows.
    apply_glyph(graph, 0, "OZ")
    after = capture_non_epi_forcing(graph)
    assert after.forcing == before.forcing
    assert after.full_kernel_pressure == before.full_kernel_pressure
    assert after.kernel_pressure_defect == before.kernel_pressure_defect
    assert after.phase == before.phase
    assert after.stored_pressure_residual[0] != 0
    assert after.stored_pressure_residual[0] == (
        after.snapshot.stored_pressure[0] - before.snapshot.stored_pressure[0]
    )
    assert after.stored_pressure_residual[1:] == (0,) * 3


def test_actual_cached_coefficients_win_over_changed_configuration_and_metadata():
    graph = _graph()
    graph.graph["_dnfr_weights"] = {"phase": 0.5, "epi": 0.25, "vf": 0.25, "topo": 0.0}
    graph.graph["DNFR_WEIGHTS"] = {"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}
    graph.graph["_DNFR_META"] = {"weights_norm": {"phase": 1.0}}
    result = capture_non_epi_forcing(graph)
    assert dict(result.normalized_weights) == {
        "phase": F(1, 2),
        "epi": F(1, 4),
        "vf": F(1, 4),
        "topo": F(0),
    }
    assert result.forcing[0] == result.phase_gradient[0] / 2 + F(11, 12)
    default_compute_delta_nfr(graph)
    assert result.full_kernel_pressure == _pressure(graph)


def test_first_setup_normalization_and_capture_do_not_write_graph_or_caches():
    graph = _graph()
    graph.graph["DNFR_WEIGHTS"] = {"phase": 2.0, "epi": 1.0, "vf": 1.0, "topo": 0.0}
    graph.graph["research_note"] = {"samples": [1, 2]}
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_edges = deepcopy(dict(graph.edges))
    before_metadata = deepcopy(graph.graph)
    result = capture_non_epi_forcing(graph)
    assert result.epi_weight == F(1, 4)
    assert dict(result.normalized_weights)["phase"] == F(1, 2)
    assert graph.graph == before_metadata
    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.edges) == before_edges
    assert "_dnfr_weights" not in graph.graph and "_dnfr_prep_cache" not in graph.graph
    with pytest.raises(FrozenInstanceError):
        result.forcing = (F(0),) * 4
    graph.nodes[0]["theta"] = 0.5
    graph.graph["DNFR_WEIGHTS"]["phase"] = 99.0
    assert result.phase[0] == 0
    assert dict(result.normalized_weights)["phase"] == F(1, 2)


def test_kernel_reads_actual_neighbor_insertion_order(monkeypatch):
    graph = _graph()
    original = realization.fused_dnfr.compute_fused_gradients_symmetric
    seen = []

    def recording_kernel(**kwargs):
        seen.append(
            tuple(zip(map(int, kwargs["edge_src"]), map(int, kwargs["edge_dst"])))
        )
        assert kwargs["accumulate_both_directions"] is False
        assert kwargs["use_jit"] is False
        return original(**kwargs)

    monkeypatch.setattr(
        realization.fused_dnfr, "compute_fused_gradients_symmetric", recording_kernel
    )
    result = capture_non_epi_forcing(graph)
    expected = tuple(
        (node, neighbor) for node in graph for neighbor in graph.neighbors(node)
    )
    assert result.snapshot.support_neighbors[0] == (1, 2, 3)
    assert expected[:3] == ((0, 2), (0, 1), (0, 3))
    assert seen and all(edges == expected for edges in seen)


def test_materialized_fused_cancellation_is_not_the_fallback_small_resultant_rule():
    graph = nx.star_graph(2)
    for node, phase in enumerate((0.0, 0.0, math.pi)):
        graph.nodes[node].update(EPI=0.0, nu_f=1.0, theta=phase, delta_nfr=0.0)
    result = capture_non_epi_forcing(graph)
    # The binary64 phasor sum has positive tiny sine and zero cosine.
    assert result.phase_gradient[0] == F(1, 2)
    graph.graph["vectorized_dnfr"] = False
    with pytest.raises(ValueError, match="fallback"):
        capture_non_epi_forcing(graph)


def test_custom_hook_is_rejected_without_execution():
    graph = _graph()

    def custom_hook(_graph):
        raise AssertionError("a read-out must not execute the callback")

    graph.graph["compute_delta_nfr"] = custom_hook
    with pytest.raises(ValueError, match="default pressure callback"):
        capture_non_epi_forcing(graph)


def test_unavailable_numpy_branch_is_rejected(monkeypatch):
    graph = _graph()
    monkeypatch.setattr(realization.fused_dnfr, "np", None)
    with pytest.raises(ValueError, match="NumPy"):
        capture_non_epi_forcing(graph)


@pytest.mark.parametrize(("count", "accepted"), [(50, True), (51, False)])
def test_directed_support_count_bounds_the_non_jit_domain(count, accepted):
    graph = nx.cycle_graph(count)
    for node in graph:
        graph.nodes[node].update(EPI=0.5, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    if accepted:
        assert capture_non_epi_forcing(graph).forcing == (0,) * count
    else:
        with pytest.raises(ValueError, match="100 directed support entries"):
            capture_non_epi_forcing(graph)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_invalid_live_phase_is_rejected(bad):
    graph = _graph()
    graph.nodes[0]["theta"] = bad
    with pytest.raises((TypeError, ValueError)):
        capture_non_epi_forcing(graph)


@pytest.mark.parametrize("bad", [-0.5, float("nan")])
def test_invalid_cached_channel_coefficient_is_rejected(bad):
    graph = _graph()
    graph.graph["_dnfr_weights"] = {"phase": bad, "epi": 0.5, "vf": 0.0, "topo": 0.0}
    with pytest.raises(ValueError):
        capture_non_epi_forcing(graph)
