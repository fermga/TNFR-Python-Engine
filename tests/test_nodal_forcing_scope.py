"""The optional additive source is a distinct model, including at zero capacity."""

import math
from copy import deepcopy
from fractions import Fraction
from types import MappingProxyType

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.constants.aliases import ALIAS_DEPI
from tnfr.dynamics import integrators
from tnfr.errors.contextual import NetworkConfigError
from tnfr.gamma import GAMMA_REGISTRY, GammaEntry, eval_gamma, eval_gamma_vectorized


def _graph():
    graph = nx.empty_graph(2)
    inject_defaults(graph)
    graph.graph.update(DT_MIN=0.1, CLIP_MODE="hard", GAMMA={"type": "none"})
    for node in graph:
        graph.nodes[node].update(
            {EPI_PRIMARY: 0.25, VF_PRIMARY: 0.0, DNFR_PRIMARY: 2.0, "theta": 0.0}
        )
    return graph


@pytest.mark.parametrize(
    "vectorized,phase", [(True, True), (False, "0.0"), (True, Fraction(1, 2**2000))]
)
def test_active_gamma_rejects_raw_phase_before_form_writes(
    vectorized, phase, monkeypatch
):
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "kuramoto_linear", "beta": 1.0}
    eval_gamma(graph, 0, 0.0, strict=True)  # Warm the valid phase/order cache.
    graph.nodes[0]["theta"] = phase
    graph.nodes[0]["phase"] = 0.0  # A later valid spelling cannot hide it.
    before = deepcopy(dict(graph.nodes(data=True)))
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    with pytest.raises((TypeError, ValueError, NetworkConfigError)):
        integrators.update_epi_via_nodal_equation(graph, dt=0.1)
    assert dict(graph.nodes(data=True)) == before
    assert "_t" not in graph.graph


@pytest.mark.parametrize("vectorized", [False, True])
def test_unforced_row_does_not_consume_unused_phase(vectorized, monkeypatch):
    graph = _graph()
    graph.nodes[0].update(theta="unused", **{VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.5})
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    integrators.update_epi_via_nodal_equation(graph, dt=0.1)
    assert graph.nodes[0][EPI_PRIMARY] == pytest.approx(0.3)


@pytest.mark.parametrize("phases", [[True, 0.0], [0.0], [[0.0, 0.0]]])
def test_gamma_array_api_rejects_coerced_or_misaligned_phase_samples(phases):
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "kuramoto_linear", "beta": 1.0}
    with pytest.raises(ValueError):
        eval_gamma_vectorized(graph, phases, 0.0, np, strict=True)


def test_gamma_scalar_and_array_owners_reject_textual_time():
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "harmonic", "beta": 1.0}
    with pytest.raises(ValueError, match="Gamma time"):
        eval_gamma(graph, 0, "0.5", strict=True)
    with pytest.raises(ValueError, match="Gamma time"):
        eval_gamma_vectorized(graph, np.zeros(2), "0.5", np, strict=True)


@pytest.mark.parametrize("unused_phases", [None, np.array(["unused"]), [None]])
def test_disabled_gamma_array_uses_graph_support_without_consuming_phase(unused_phases):
    graph = _graph()
    assert np.array_equal(
        eval_gamma_vectorized(graph, unused_phases, 0.0, np, strict=True),
        np.zeros(len(graph)),
    )


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("forced", [False, True])
def test_zero_capacity_freezes_only_the_unforced_nodal_channel(
    monkeypatch, vectorized, forced
):
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    graph = nx.empty_graph(1)
    inject_defaults(graph)
    graph.graph.update(DT_MIN=0.0, CLIP_MODE="hard", use_extended_dynamics=False)
    graph.nodes[0].update({EPI_PRIMARY: 0.25, VF_PRIMARY: 0.0, DNFR_PRIMARY: 2.0})
    graph.graph["GAMMA"] = (
        {"type": "harmonic", "beta": 0.125, "omega": 0.0, "phi": math.pi / 2}
        if forced
        else {"type": "none"}
    )
    integrators.update_epi_via_nodal_equation(graph, dt=0.5, method="euler")
    expected_rate = 0.125 if forced else 0.0
    assert graph.nodes[0][EPI_PRIMARY] == 0.25 + 0.5 * expected_rate
    assert get_attr(graph.nodes[0], ALIAS_DEPI) == expected_rate
    assert graph.nodes[0][VF_PRIMARY] == 0.0
    assert graph.nodes[0][DNFR_PRIMARY] == 2.0


@pytest.mark.parametrize(
    "vectorized,spec",
    [
        (False, {"type": "unknown-source"}),
        (False, {"type": "harmonic", "beta": "malformed"}),
        (False, {"type": "harmonic", "beta": math.nan}),
        (False, {"type": "harmonic", "beta": True}),
        (False, {"type": "harmonic", "beta": "0.125"}),
        (True, {"type": "harmonic", "beta": "malformed"}),
        (True, {"type": "harmonic", "beta": math.nan}),
    ],
)
def test_runtime_does_not_replace_invalid_forcing_with_zero(
    monkeypatch, vectorized, spec
):
    graph = _graph()
    graph.graph["GAMMA"] = spec
    before = deepcopy(dict(graph.nodes(data=True)))
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    with pytest.raises(ValueError):
        integrators.update_epi_via_nodal_equation(graph, dt=0.1, method="euler")
    assert dict(graph.nodes(data=True)) == before
    assert "_t" not in graph.graph


@pytest.mark.parametrize("container", [False, 0, [], "missing-gamma.json"])
def test_runtime_rejects_malformed_source_container(container):
    graph = _graph()
    graph.graph["GAMMA"] = container
    before = deepcopy(graph)
    with pytest.raises(ValueError, match="GAMMA must be a mapping"):
        integrators.update_epi_via_nodal_equation(graph, dt=0.1)
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph


@pytest.mark.parametrize(
    "value", [True, "0.125", math.nan, 0.125 + 0j, Fraction(1, 2**2000)]
)
def test_runtime_validates_callback_result_before_numeric_coercion(monkeypatch, value):
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "invalid-output"}
    before = deepcopy(dict(graph.nodes(data=True)))
    calls = []

    def source(current, node, time, spec):
        calls.append(node)
        return value

    monkeypatch.setitem(GAMMA_REGISTRY, "invalid-output", GammaEntry(source, False))
    with pytest.raises(ValueError, match="Gamma rate"):
        integrators.update_epi_via_nodal_equation(graph, dt=0.1)
    assert dict(graph.nodes(data=True)) == before
    assert "_t" not in graph.graph
    assert calls == [0]


def test_zero_duration_does_not_resolve_or_evaluate_forcing():
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "invalid-unused-source"}
    before = deepcopy(graph)
    integrators.update_epi_via_nodal_equation(graph, dt=0.0)
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph


def test_declared_read_only_source_mapping_is_accepted():
    graph = _graph()
    graph.graph["GAMMA"] = MappingProxyType(
        {"type": "harmonic", "beta": 0.125, "omega": 0.0, "phi": math.pi / 2}
    )
    integrators.update_epi_via_nodal_equation(graph, dt=0.1)
    assert [graph.nodes[node][EPI_PRIMARY] for node in graph] == [0.2625, 0.2625]


@pytest.mark.parametrize(
    "vectorized,name,method,stage_count,n_jobs",
    [
        (False, "custom-source", "euler", 1, 2),
        (False, "custom-source", "rk4", 3, None),
        (True, "custom-source", "euler", 1, None),
        (True, "custom-source", "rk4", 3, None),
        (True, "none", "euler", 1, None),
        (True, "harmonic", "rk4", 3, None),
    ],
)
def test_registered_callback_uses_same_live_substeps_once_per_node_and_stage(
    monkeypatch, vectorized, name, method, stage_count, n_jobs
):
    graph = _graph()
    graph.graph["GAMMA"] = {"type": name}
    calls = []

    # A supplied graph-dependent callback, not a derived TNFR source. Each
    # substep holds its graph state; RK4 still samples just the three times.
    def source(current, node, time, spec):
        value = current.nodes[node][EPI_PRIMARY]
        calls.append((node, time, value))
        return value

    monkeypatch.setitem(GAMMA_REGISTRY, name, GammaEntry(source, False))
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    # One end-to-end process case retains arithmetic worker/pickling coverage;
    # the remaining callback cases need only their distinct stage/dispatch path.
    integrators.update_epi_via_nodal_equation(
        graph, dt=0.2, method=method, n_jobs=n_jobs
    )
    expected = 0.25 * (1.0 + 0.1) ** 2
    assert [graph.nodes[node][EPI_PRIMARY] for node in graph] == pytest.approx(
        [expected] * 2
    )
    assert len(calls) == 2 * 2 * stage_count
    assert [value for _, _, value in calls[: 2 * stage_count]] == [0.25] * (
        2 * stage_count
    )
    assert [value for _, _, value in calls[2 * stage_count :]] == pytest.approx(
        [0.275] * (2 * stage_count)
    )
    assert graph.graph["_t"] == 0.2


def test_custom_callback_stays_live_when_parallel_workers_are_requested(monkeypatch):
    """Check callback scheduling without spawning unrelated arithmetic workers."""
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "custom-source"}
    calls = []

    def source(current, node, time, spec):
        assert current is graph
        calls.append((node, time))
        return 0.125 * (node + 1)

    def unexpected_pool(*args, **kwargs):
        pytest.fail("Custom Gamma must not run on process-local graph copies")

    monkeypatch.setitem(GAMMA_REGISTRY, "custom-source", GammaEntry(source, False))
    monkeypatch.setattr(integrators, "ProcessPoolExecutor", unexpected_pool)
    result = integrators._evaluate_gamma_map(graph, list(graph), 0.5, n_jobs=2)
    assert result == {0: 0.125, 1: 0.25}
    assert calls == [(0, 0.5), (1, 0.5)]


@pytest.mark.parametrize("vectorized", [False, True])
def test_late_callback_failure_restores_owned_solver_outputs(monkeypatch, vectorized):
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "failing-source"}
    graph.graph["_t"] = 0.0
    before = deepcopy(dict(graph.nodes(data=True)))
    calls = []

    def source(current, node, time, spec):
        calls.append((node, time))
        if time >= 0.1:
            raise ValueError("declared source failed")
        return 0.1

    monkeypatch.setitem(GAMMA_REGISTRY, "failing-source", GammaEntry(source, False))
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    with pytest.raises(ValueError, match="declared source failed"):
        integrators.update_epi_via_nodal_equation(graph, dt=0.2, method="euler")
    assert dict(graph.nodes(data=True)) == before
    assert graph.graph["_t"] == 0.0
    # Callback-owned external evidence is intentionally not rolled back.
    assert calls == [(0, 0.0), (1, 0.0), (0, 0.1)]


def test_public_vector_gamma_respects_registry_and_permissive_scope(monkeypatch):
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "custom-source"}
    calls = []

    def source(current, node, time, spec):
        calls.append(node)
        return (node + 1) / 8

    monkeypatch.setitem(GAMMA_REGISTRY, "custom-source", GammaEntry(source, False))
    actual = eval_gamma_vectorized(graph, np.zeros(2), 0.0, np, strict=True)
    assert tuple(actual) == (0.125, 0.25)
    assert calls == [0, 1]

    graph.graph["GAMMA"] = {"type": "missing-source"}
    assert eval_gamma(graph, 0, 0.0) == 0.0
    assert tuple(eval_gamma_vectorized(graph, np.zeros(2), 0.0, np)) == (0.0, 0.0)
