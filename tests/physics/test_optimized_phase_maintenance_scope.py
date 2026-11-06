"""Prepared P5 controls for separate optimized phase and form updates.

These finite proposals advance phase but use only EPI diffusion for form.
The canonical mixed-pressure control establishes that the supplied phases
are active when that channel is selected. No maintained pattern, complete
runtime execution or asymptotic binary64 convergence is certified here.
"""

import math
from copy import deepcopy
from fractions import Fraction as F

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.fft_engine import FFTDynamicsEngine
from tnfr.dynamics.nodal_optimizer import NodalEquationOptimizer
from tnfr.mathematics.spectral import igft

DT = F(1, 8)
UNIFORM_PHASE = (0.25,) * 5
VARYING_PHASE = (0.25, 0.375, 0.5, 0.625, 0.75)
EPI_CASES = (
    pytest.param((0.25,) * 5, id="uniform-form"),
    pytest.param((0.5, 0.25, 0.0, -0.25, -0.5), id="odd-form"),
)


def _graph(epi, phase):
    graph = nx.path_graph(5)
    graph.graph.update(
        _t=0.0,
        DELTA_PHI_MAX=0.25,
        DNFR_WEIGHTS={"phase": 1.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
    )
    for node in graph:
        graph.nodes[node].update(
            EPI=epi[node], nu_f=1.0, theta=phase[node], delta_nfr=-17.0
        )
    for edge in graph.edges:
        graph.edges[edge].update(weight=1.0, length=1.0)
        left, right = edge
        assert abs(math.remainder(phase[left] - phase[right], math.tau)) < 0.25
    return graph


def _prepared_surface(graph, *, include_pressure=True):
    # Spectral/adjacency caches may be added to graph.graph. Prepared inputs,
    # ordered support and every node attribute must otherwise remain intact.
    nodes = tuple(
        (
            (node, deepcopy(dict(data)))
            if include_pressure
            else (node, (data["EPI"], data["nu_f"], data["theta"]))
        )
        for node, data in graph.nodes(data=True)
    )
    return (
        nodes,
        tuple((i, j, deepcopy(data)) for i, j, data in graph.edges(data=True)),
        {
            key: deepcopy(graph.graph[key])
            for key in ("_t", "DELTA_PHI_MAX", "DNFR_WEIGHTS")
        },
    )


def _pure_pressure(epi):
    # Independent nearest-neighbor rule, with degree one at the two ends.
    values = tuple(F(value) for value in epi)
    return tuple(
        sum(values[j] for j in range(5) if abs(i - j) == 1) / (1 if i in (0, 4) else 2)
        - values[i]
        for i in range(5)
    )


def _expected_epi(epi):
    return tuple(
        float(F(value) + DT * pressure)
        for value, pressure in zip(epi, _pure_pressure(epi), strict=True)
    )


def _assert_phase_is_advanced_and_branches_remain_distinct(endpoints):
    first, second = endpoints
    assert np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    for initial, final in zip((UNIFORM_PHASE, VARYING_PHASE), endpoints, strict=True):
        increments = [
            abs(math.remainder(after - before, math.tau))
            for before, after in zip(initial, final, strict=True)
        ]
        assert max(increments) > float(DT) / 2
    assert (
        max(
            abs(math.remainder(a - b, math.tau))
            for a, b in zip(first, second, strict=True)
        )
        > 0.1
    )


@pytest.mark.parametrize("epi", EPI_CASES)
def test_nodal_optimizer_advances_phase_without_feeding_it_into_form_pressure(epi):
    outputs, phases = [], []
    for phase in (UNIFORM_PHASE, VARYING_PHASE):
        graph = _graph(epi, phase)
        before = _prepared_surface(graph)
        proposal = NodalEquationOptimizer(
            enable_cache=False
        ).compute_vectorized_nodal_evolution(graph, float(DT))
        outputs.append(tuple(proposal[node][0] for node in graph))
        phases.append(tuple(proposal[node][1] for node in graph))
        assert proposal.stability_not_certified
        assert _prepared_surface(graph) == before
    assert outputs[0] == outputs[1] == _expected_epi(epi)
    _assert_phase_is_advanced_and_branches_remain_distinct(phases)


@pytest.mark.parametrize("epi", EPI_CASES)
def test_fft_step_has_the_same_phase_independent_epi_proposal_after_igft(epi):
    outputs, phases = [], []
    for phase in (UNIFORM_PHASE, VARYING_PHASE):
        graph = _graph(epi, phase)
        before = _prepared_surface(graph)
        engine = FFTDynamicsEngine(enable_caching=False)
        state = engine.create_fft_state(graph)
        updated = engine.fft_accelerated_step(graph, state, float(DT))
        outputs.append(igft(updated.spectral_epi, updated.eigenvectors))
        phases.append(igft(updated.spectral_phase, updated.eigenvectors))
        assert updated.stability_not_certified
        assert updated.time == float(DT)
        assert _prepared_surface(graph) == before
    np.testing.assert_array_equal(outputs[0], outputs[1])
    # The exact Euler oracle is separate from the dense spectral round trips.
    np.testing.assert_allclose(outputs[0], _expected_epi(epi), rtol=2e-15, atol=2e-15)
    _assert_phase_is_advanced_and_branches_remain_distinct(phases)


@pytest.mark.parametrize("epi", EPI_CASES)
def test_canonical_mixed_pressure_responds_to_the_same_admissible_phase_change(epi):
    pressures = []
    for phase in (UNIFORM_PHASE, VARYING_PHASE):
        original = _graph(epi, phase)
        graph = deepcopy(original)
        before = _prepared_surface(graph, include_pressure=False)
        default_compute_delta_nfr(graph)
        pressures.append(
            np.array(
                [get_attr(graph.nodes[node], ALIAS_DNFR, strict=True) for node in graph]
            )
        )
        assert _prepared_surface(graph, include_pressure=False) == before
        assert all(original.nodes[node]["delta_nfr"] == -17.0 for node in original)
    expected_epi_channel = np.array([float(value) / 2 for value in _pure_pressure(epi)])
    np.testing.assert_allclose(pressures[0], expected_epi_channel, rtol=0, atol=2e-15)
    # Unit phase and EPI coefficients normalize to one half each. The end
    # nodes have exactly one neighbor at phase displacement +/-1/8.
    expected_phase_difference = np.array([1.0, 0.0, 0.0, 0.0, -1.0]) / (16 * math.pi)
    np.testing.assert_allclose(
        pressures[1] - pressures[0], expected_phase_difference, rtol=2e-14, atol=2e-15
    )
    assert not np.array_equal(pressures[0], pressures[1])
