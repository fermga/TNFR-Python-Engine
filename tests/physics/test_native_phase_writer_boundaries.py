"""Component boundaries of the default native phase-renewal budget.

These detached inputs exercise existing integrator, adaptation, coordination
and clamp owners. They are not a runtime trajectory or a history certificate.
Stored pressure is declared, not silently replaced by freshly computed pressure.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q
from functools import wraps

import networkx as nx
import pytest

from tests.physics._internal_mode_fixture import _graph
from tnfr.alias import get_attr, get_theta_attr
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import adaptation, coordination, integrators, runtime
from tnfr.mathematics.phasor_resultant import reduce_phasor_components
from tnfr.metrics.trig_cache import get_trig_cache
from tnfr.physics.winding_certificates import certify_phase_winding
from tnfr.utils import angle_diff
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


@pytest.mark.parametrize("capacity", (1.0, 0.3))
def test_selective_adaptation_preserves_uniform_fixed_point(capacity):
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
    expected = (Q(capacity),) * len(graph)
    assert _channel(graph, ALIAS_VF) == expected
    if capacity == 1.0:
        # All snapshot neighbor means equal one. The default represented
        # convex blend is exactly one, independently of the eligible subset.
        assert set(expected) == {Q(1)}
    else:
        # The old redundant two-product blend introduced one ulp only at
        # eligible nodes. The shared fixed-point branch now prevents it.
        mu = graph.graph["VF_ADAPT_MU"]
        assert (1 - mu) * capacity + mu * capacity == math.nextafter(capacity, math.inf)
    assert _channel(graph, ALIAS_EPI) == held_form
    assert _channel(graph, ALIAS_DNFR) == tuple(map(Q, pressures))
    assert _phases(graph) == held_phase
    assert tuple(graph.edges) == held_edges


@pytest.mark.parametrize("graph_owned", (False, True))
def test_centered_phase_clamp_is_an_exact_identity_inside_its_chart(graph_owned):
    graph = _input_graph()
    phases = (0.1, 1e-16, -1e-16, math.ulp(0.0), -math.pi, -0.0)
    for node, phase in zip(graph, phases, strict=True):
        graph.nodes[node].update(theta=phase, phase=phase)
        if graph_owned:
            apply_canonical_clamps(graph.nodes[node], graph, node)
        else:
            apply_canonical_clamps(graph.nodes[node])
        assert graph.nodes[node]["theta"].hex() == phase.hex()
        assert graph.nodes[node]["phase"].hex() == phase.hex()


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


@pytest.fixture(scope="module")
def static_twist_coordination():
    """Two isolated calls from the same declared C5, not a formation trace."""
    source = nx.cycle_graph(5)
    nx.set_edge_attributes(source, 1.0, "weight")
    inject_defaults(source)
    source.graph["RANDOM_SEED"] = 17
    for node in source:
        source.nodes[node].update(
            EPI=0.5,
            nu_f=1.0,
            theta=0.125 + math.tau * node / 5,
            delta_nfr=0.0,
            glyph_history=[],
        )
    cases = {}
    for mode in ("legacy", "exact_components_v1"):
        graph = deepcopy(source)
        before = _phases(graph)
        trig = get_trig_cache(graph)
        resultant = reduce_phasor_components(
            (trig.cos[node], trig.sin[node]) for node in graph
        )
        before_winding = certify_phase_winding(graph, range(5))
        held = {
            "epi": _channel(graph, ALIAS_EPI),
            "capacity": _channel(graph, ALIAS_VF),
            "pressure": _channel(graph, ALIAS_DNFR),
            "support": deepcopy(tuple(graph.edges(data=True))),
            "time": graph.graph.get("_t"),
        }
        targets = []
        original_mean = coordination.neighbor_phase_mean_list

        @wraps(original_mean)
        def local_mean(*args, **kwargs):
            result = original_mean(*args, **kwargs)
            targets.append(result)
            return result

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(coordination, "neighbor_phase_mean_list", local_mean)
            if mode == "legacy":
                # Exercise the actual default argument, not an override.
                result = coordination.coordinate_global_local_phase(graph)
            else:
                result = coordination.coordinate_global_local_phase(
                    graph, global_reduction=mode
                )
        after = _phases(graph)
        gaps = tuple(
            angle_diff(after[(index + 1) % 5], after[index]) for index in range(5)
        )
        cases[mode] = {
            "graph": graph,
            "before": before,
            "after": after,
            "before_winding": before_winding,
            "after_winding": certify_phase_winding(graph, range(5)),
            "resultant": resultant,
            "local_targets": tuple(targets),
            "gaps": gaps,
            "held": held,
            "result": result,
        }
    return cases


def test_native_global_relaxation_changes_twist_geometry_without_changing_winding(
    static_twist_coordination,
):
    # Ideally each local phasor sum is 2*cos(delta)*exp(i*theta_i), and
    # cos(2*pi/5)>0. Thus the local target is theta_i, not a global direction.
    # The active global term creates a long edge at its branch-cut crossing.
    delta = math.tau / 5
    for case in static_twist_coordination.values():
        graph, held = case["graph"], case["held"]
        history = graph.graph["history"]
        k_global = history["phase_kG"][-1]
        k_local = history["phase_kL"][-1]
        assert 0 < k_global < 0.2 and k_local > 0
        assert history["phase_disr"][-1] == 0.0
        # The ideal zero resultant is not asserted of materialized trig.
        assert 0 <= history["phase_R"][-1] < 1e-14
        assert len(case["local_targets"]) == 5
        assert tuple(
            angle_diff(target, phase)
            for target, phase in zip(case["local_targets"], case["before"], strict=True)
        ) == pytest.approx((0.0,) * 5, rel=0, abs=2e-15)
        short = (1 - k_global) * delta
        long = delta + k_global * (math.tau - delta)
        assert tuple(sorted(case["gaps"])) == pytest.approx(
            (short,) * 4 + (long,), rel=0, abs=3e-15
        )
        assert long - short > 0.1
        assert case["before_winding"].winding == 1
        assert case["after_winding"].winding == 1
        assert case["after_winding"].quantization_residual < 1e-14
        assert case["after_winding"].u3_admissible == (long <= math.pi / 2)
        # A common phase rotation preserves every edge gap; this writer does
        # not preserve that geometric orbit despite preserving this winding.
        assert _channel(graph, ALIAS_EPI) == held["epi"]
        assert _channel(graph, ALIAS_VF) == held["capacity"]
        assert _channel(graph, ALIAS_DNFR) == held["pressure"]
        assert tuple(graph.edges(data=True)) == held["support"]
        assert graph.graph.get("_t") == held["time"]
        assert all(graph.nodes[node]["glyph_history"] == [] for node in graph)


def test_exact_component_reduction_does_not_reconstruct_an_ideal_zero_resultant(
    static_twist_coordination,
):
    assert static_twist_coordination["legacy"]["result"] is None
    case = static_twist_coordination["exact_components_v1"]
    evidence = case["result"]
    assert evidence.resultant == case["resultant"]
    assert not evidence.resultant.joint_zero
    assert 0 < float(evidence.resultant.scale) < 1e-14
    assert evidence.global_term_active and evidence.global_target is not None
    assert evidence.local_targets == case["local_targets"]
    assert evidence.realized_phases == case["after"]
    assert evidence.primitive_phases == case["before"]
    assert evidence.gain_mode == "adaptive"
    # This exactness applies only to the represented components. Their small
    # nonzero sum does not supply a direction for the ideal zero resultant.
