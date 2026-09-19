"""Production boundaries of a held-capacity EPI pressure cancellation.

These finite controls exercise the canonical pressure reader and named
all-target kernels. They do not identify a fixed capacity profile as an
autonomous localized entity or certify future full multichannel execution.
"""

import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.metrics.common import merge_and_normalize_weights
from tnfr.operators.definitions import Coupling, Silence
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
    execute_pointwise_stage,
)
from tnfr.physics.winding_certificates import certify_phase_winding


def _weights(graph):
    return merge_and_normalize_weights(
        graph, "DNFR_WEIGHTS", ("phase", "epi", "vf", "topo"), default=0.0
    )


def _field(graph, aliases):
    return np.array([get_attr(graph.nodes[node], aliases, 0.0) for node in graph])


def _initialize(graph, capacities, epi, *, winding=0):
    count = len(graph)
    for index, node in enumerate(graph):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: float(epi[index]),
                ALIAS_VF[0]: float(capacities[index]),
                ALIAS_THETA[0]: math.tau * winding * index / count,
                ALIAS_DNFR[0]: 0.0,
                ALIAS_SI[0]: 1.0,
                "glyph_history": ["AL"],
            }
        )
    return graph


def _pinned_cycle(*, winding=1):
    graph = nx.cycle_graph(8)
    capacities = (1, 1, 1, 2, 1, 1, 1, 1)
    weights = _weights(graph)
    ratio = Fraction(weights["vf"]) / Fraction(weights["epi"])
    epi = [float(1 - ratio * capacity) for capacity in capacities]
    return _initialize(graph, capacities, epi, winding=winding)


def _cycle_laplacian(values):
    """Exact unit-cycle L_rw from the two adjacent scalar coordinates."""
    values = tuple(
        value if isinstance(value, Fraction) else Fraction(float(value))
        for value in values
    )
    count = len(values)
    return tuple(
        values[index] - (values[index - 1] + values[(index + 1) % count]) / 2
        for index in range(count)
    )


def _named_stage(graph, operator):
    # A one-step named-kernel observation, with the retained AL history and
    # the production live grammar authoritative; no whole-word claim follows.
    executor = (
        execute_coupling_stage
        if isinstance(operator, Coupling)
        else execute_pointwise_stage
    )
    result = executor(
        graph, operator, tuple(graph), compute_delta_nfr=default_compute_delta_nfr
    )
    assert result.glyph == operator.glyph.value
    assert result.schedule == TWO_PHASE_JACOBI
    assert all(
        graph.nodes[node]["glyph_history"][-1] == operator.glyph.value for node in graph
    )


@pytest.mark.parametrize("winding", [0, 1])
def test_default_full_pressure_cancels_on_unit_cycle_with_uniform_twist(winding):
    graph = _pinned_cycle(winding=winding)
    before_epi = _field(graph, ALIAS_EPI).copy()
    before_capacity = _field(graph, ALIAS_VF).copy()

    default_compute_delta_nfr(graph)

    assert graph.graph["_dnfr_weights"] == _weights(graph)
    assert graph.graph["_dnfr_weights"]["phase"] > 0.0
    np.testing.assert_allclose(_field(graph, ALIAS_DNFR), 0.0, atol=5e-16)
    np.testing.assert_array_equal(_field(graph, ALIAS_EPI), before_epi)
    np.testing.assert_array_equal(_field(graph, ALIAS_VF), before_capacity)
    assert np.ptp(before_epi) > 0.1
    certificate = certify_phase_winding(graph, range(8))
    assert certificate.winding == winding
    assert certificate.u3_admissible


def test_regular_weighted_degree_does_not_identify_epi_and_capacity_walks():
    graph = nx.cycle_graph(4)
    for node in graph:
        graph.edges[node, (node + 1) % 4]["weight"] = 1.0 if node % 2 == 0 else 3.0
    _initialize(graph, (1, 2, 3, 4), (4, 3, 2, 1))
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0,
        "epi": 1.0,
        "vf": 1.0,
        "topo": 0.0,
    }

    default_compute_delta_nfr(graph)

    assert set(dict(graph.degree(weight="weight")).values()) == {4.0}
    assert _weights(graph)["epi"] == _weights(graph)["vf"] == 0.5
    np.testing.assert_array_equal(_field(graph, ALIAS_DNFR), (-0.25, -0.25, 0.25, 0.25))


def test_zero_weight_edge_remains_in_the_capacity_neighborhood():
    graph = nx.cycle_graph(4)
    graph.edges[0, 1]["weight"] = 0.0
    _initialize(graph, (1, 2, 3, 4), (4, 3, 2, 1))
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0,
        "epi": 1.0,
        "vf": 1.0,
        "topo": 0.0,
    }

    default_compute_delta_nfr(graph)

    assert graph.has_edge(0, 1)
    np.testing.assert_array_equal(_field(graph, ALIAS_DNFR), (-0.5, -0.5, 0.0, 0.0))


@pytest.mark.parametrize("epi", [(0, 0, 0), (0.25, 0.5, 0.75), (1, -1, 2)])
def test_weighted_path_capacity_forcing_has_nonzero_conserved_total_drift(epi):
    graph = nx.path_graph(3)
    graph.edges[0, 1]["weight"] = 1.0
    graph.edges[1, 2]["weight"] = 3.0
    _initialize(graph, (1, 2, 4), epi)
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0,
        "epi": 1.0,
        "vf": 1.0,
        "topo": 0.0,
    }

    default_compute_delta_nfr(graph)

    metric = np.array([1.0, 2.0, 0.75])
    rate = _field(graph, ALIAS_VF) * _field(graph, ALIAS_DNFR)
    # h^T x' = -w_vf d_W^T L_support nu = -3/2, independently of x.
    assert metric @ rate == pytest.approx(-1.5, abs=2e-15)


def test_default_um_capacity_sync_breaks_the_pin_after_full_pressure_refresh():
    graph = _pinned_cycle()
    graph.graph.update(UM_BIDIRECTIONAL=False, UM_FUNCTIONAL_LINKS=False)
    default_compute_delta_nfr(graph)
    before_epi = _field(graph, ALIAS_EPI).copy()
    before_capacity = _field(graph, ALIAS_VF).copy()
    frequency_weight = Fraction(_weights(graph)["vf"])
    sync = Fraction(
        resolve_runtime_operator_factors(None, "UM", graph.graph)["UM_vf_sync"]
    )
    laplacian = _cycle_laplacian(before_capacity)
    expected_capacity = [
        float(Fraction(value) - sync * lap)
        for value, lap in zip(before_capacity, laplacian)
    ]
    expected_pressure = [
        float(frequency_weight * sync * lap) for lap in _cycle_laplacian(laplacian)
    ]

    _named_stage(graph, Coupling())

    np.testing.assert_array_equal(_field(graph, ALIAS_EPI), before_epi)
    np.testing.assert_allclose(
        _field(graph, ALIAS_VF), expected_capacity, rtol=0.0, atol=3e-16
    )
    np.testing.assert_allclose(
        _field(graph, ALIAS_DNFR), expected_pressure, rtol=0.0, atol=5e-16
    )
    assert np.max(np.abs(_field(graph, ALIAS_DNFR))) > 0.001
    assert certify_phase_winding(graph, range(8)).winding == 1


def test_default_global_silence_breaks_the_pin_without_changing_epi():
    graph = _pinned_cycle()
    default_compute_delta_nfr(graph)
    before_epi = _field(graph, ALIAS_EPI).copy()
    before_capacity = _field(graph, ALIAS_VF).copy()
    frequency_weight = Fraction(_weights(graph)["vf"])
    factor = Fraction(
        resolve_runtime_operator_factors(None, "SHA", graph.graph)["SHA_vf_factor"]
    )
    expected_pressure = [
        float(frequency_weight * (1 - factor) * lap)
        for lap in _cycle_laplacian(before_capacity)
    ]

    _named_stage(graph, Silence())

    np.testing.assert_array_equal(_field(graph, ALIAS_EPI), before_epi)
    np.testing.assert_allclose(
        _field(graph, ALIAS_VF),
        float(factor) * before_capacity,
        rtol=0.0,
        atol=3e-16,
    )
    np.testing.assert_allclose(
        _field(graph, ALIAS_DNFR), expected_pressure, rtol=0.0, atol=5e-16
    )
    assert np.max(np.abs(_field(graph, ALIAS_DNFR))) > 0.001


def test_zero_capacity_freezes_epi_with_nonzero_refreshed_pressure():
    graph = _pinned_cycle(winding=0)
    graph.graph.update(
        GLYPH_FACTORS={"SHA_vf_factor": 0.0},
        GAMMA={"type": "none"},
        DT_MIN=0.0,
    )
    before_epi = _field(graph, ALIAS_EPI).copy()

    _named_stage(graph, Silence())
    assert np.max(np.abs(_field(graph, ALIAS_DNFR))) > 0.01
    np.testing.assert_array_equal(_field(graph, ALIAS_VF), np.zeros(8))
    update_epi_via_nodal_equation(graph, dt=0.125, t=0.0, method="euler")

    np.testing.assert_array_equal(_field(graph, ALIAS_EPI), before_epi)
