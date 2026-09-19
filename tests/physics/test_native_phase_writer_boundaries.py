"""Component boundaries of the default native phase-renewal budget.

These detached inputs exercise existing integrator, adaptation and clamp
owners. They are not a runtime trajectory or a selector/history certificate.
Stored pressure is declared, not silently replaced by freshly computed pressure.
"""

import math
from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import _graph
from tnfr.alias import get_attr, get_theta_attr
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import adaptation, integrators, runtime
from tnfr.validation.runtime import apply_canonical_clamps


def _input_graph():
    graph = _graph()
    inject_defaults(graph)
    phases = (-math.pi / 6, math.pi / 6, 0.0) * 2
    for node, phase in zip(graph, phases, strict=True):
        data = graph.nodes[node]
        data.update(EPI=float(data["EPI"]), theta=phase, phase=phase)
    return graph


def _channel(graph, aliases):
    return tuple(Q(get_attr(graph.nodes[node], aliases)) for node in graph)


def _phases(graph):
    return tuple(get_theta_attr(graph.nodes[node]) for node in graph)


@pytest.mark.parametrize("vectorized", [False, True], ids=["scalar", "numpy"])
def test_default_integrator_advances_epi_without_a_free_phase_clock(
    monkeypatch, vectorized
):
    graph = _input_graph()
    capacities = (1.0, 1.5, 0.5, 1.0, 2.0, 1.25)
    pressures = (0.25, -0.125, 0.5, -0.25, 0.125, -0.5)
    for node, capacity, pressure in zip(graph, capacities, pressures, strict=True):
        graph.nodes[node].update(nu_f=capacity, delta_nfr=pressure)
    before = _channel(graph, ALIAS_EPI)
    held_phase = _phases(graph)
    held_edges = tuple(graph.edges)
    if vectorized:
        assert integrators.np is not None
    else:
        monkeypatch.setattr(integrators, "np", None)
    assert graph.graph["INTEGRATOR_METHOD"] == "euler"
    assert graph.graph["GAMMA"]["type"] == "none"

    integrators.DefaultIntegrator().integrate(
        graph, dt=0.25, t=0.0, method=None, n_jobs=1
    )

    # Every substep and endpoint is dyadic and stays inside the hard bounds.
    # The independent nodal integral therefore has no rounding/clipping term.
    expected = tuple(
        epi + Q(1, 4) * Q(capacity) * Q(pressure)
        for epi, capacity, pressure in zip(before, capacities, pressures, strict=True)
    )
    assert _channel(graph, ALIAS_EPI) == expected != before
    assert _phases(graph) == held_phase
    assert tuple(graph.nodes[node]["phase"] for node in graph) == held_phase
    assert _channel(graph, ALIAS_VF) == tuple(map(Q, capacities))
    assert _channel(graph, ALIAS_DNFR) == tuple(map(Q, pressures))
    assert graph.graph["_t"] == 0.25
    assert tuple(graph.edges) == held_edges


def test_extended_wrapper_flag_does_not_select_a_native_extended_integrator():
    graph = _input_graph()
    assert "integrator" not in graph.graph
    first = runtime._resolve_integrator_instance(graph)
    assert type(first) is integrators.DefaultIntegrator

    graph.graph["use_extended_dynamics"] = True
    assert runtime._resolve_integrator_instance(graph) is first
    # The result is not an artifact of the cached pre-flag instance.
    graph.graph.pop("_integrator_cache")
    uncached = runtime._resolve_integrator_instance(graph)
    assert type(uncached) is integrators.DefaultIntegrator
    assert uncached is not first


@pytest.mark.parametrize(
    ("capacity", "eligible_defect"),
    [(1.0, Q(0)), (0.3, Q(1, 2**54))],
    ids=["retained-unit-capacity", "nonunit-rounding-countercontrol"],
)
def test_selective_adaptation_distinguishes_unit_invariance_from_rounding(
    capacity, eligible_defect
):
    graph = _input_graph()
    # Declared stored gate inputs exercise count, pressure and Si exclusions;
    # they are not an assertion about eligibility in an unexecuted native step.
    counts = (4, 5, 4, 3, 9, 0)
    senses = (1.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    pressures = (0.0, 0.0, 0.002, 0.0, 0.0, 0.0)
    for node, count, sense, pressure in zip(
        graph, counts, senses, pressures, strict=True
    ):
        graph.nodes[node].update(
            nu_f=capacity, stable_count=count, Si=sense, delta_nfr=pressure
        )
    held_form = _channel(graph, ALIAS_EPI)
    held_phase = _phases(graph)
    held_edges = tuple(graph.edges)
    assert graph.graph["VF_ADAPT_TAU"] == DEFAULTS["VF_ADAPT_TAU"] == 5
    assert graph.graph["VF_ADAPT_MU"] == DEFAULTS["VF_ADAPT_MU"] == 0.1

    adaptation.adapt_vf_after_structural_stability(graph, n_jobs=1)

    assert tuple(graph.nodes[node]["stable_count"] for node in graph) == (
        5,
        0,
        0,
        4,
        10,
        1,
    )
    expected = tuple(
        Q(capacity) + (eligible_defect if index in (0, 4) else 0)
        for index in range(len(graph))
    )
    assert _channel(graph, ALIAS_VF) == expected
    if capacity == 1.0:
        # All snapshot neighbor means equal one. The default represented
        # convex blend is exactly one, independently of the eligible subset.
        assert set(expected) == {Q(1)}
    else:
        # Exact-real uniformity is not an unrestricted binary64 invariant:
        # only the eligible nodes acquire this one-ulp change at stored 0.3.
        assert len(set(expected)) == 2
        assert float(expected[0]) == math.nextafter(capacity, math.inf)
    assert _channel(graph, ALIAS_EPI) == held_form
    assert _channel(graph, ALIAS_DNFR) == tuple(map(Q, pressures))
    assert _phases(graph) == held_phase
    assert tuple(graph.edges) == held_edges


def test_normalization_requires_a_coherent_lift_and_a_represented_defect():
    graph = _input_graph()
    inputs = (math.pi - 0.1, math.pi + 0.1, math.pi + 0.05) * 2
    for node, phase in zip(graph, inputs, strict=True):
        graph.nodes[node].update(theta=phase, phase=phase)
    before = tuple(map(Q, inputs))
    held_form = _channel(graph, ALIAS_EPI)
    held_capacity = _channel(graph, ALIAS_VF)
    held_pressure = _channel(graph, ALIAS_DNFR)

    for node in graph:
        apply_canonical_clamps(graph.nodes[node], graph, node)

    actual = tuple(map(Q, _phases(graph)))
    pi = Q(math.pi)
    assert all(-pi <= phase < pi for phase in actual)
    assert max(actual) - min(actual) > pi
    # Use the declared represented period; no transcendental exactness is
    # claimed. Inputs near pi determine this common lift unambiguously.
    lifted = tuple(phase + 2 * pi if phase < 0 else phase for phase in actual)
    defect = tuple(after - old for after, old in zip(lifted, before, strict=True))
    before_diameter = max(before) - min(before)
    after_diameter = max(lifted) - min(lifted)
    assert 0 < before_diameter < Q(1, 2)
    assert 0 < after_diameter < Q(1, 2)
    assert 0 < max(map(abs, defect)) < Q(1, 2**48)
    assert abs(after_diameter - before_diameter) <= max(defect) - min(defect)
    assert _channel(graph, ALIAS_EPI) == held_form
    assert _channel(graph, ALIAS_VF) == held_capacity
    assert _channel(graph, ALIAS_DNFR) == held_pressure
