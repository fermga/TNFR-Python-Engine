"""Pressure configuration, scalar admission and optional-model boundaries."""

from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics import dnfr
from tnfr.mathematics.epi import BEPIElement
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.types import serialize_bepi


def _weights(channel):
    return {name: float(name == channel) for name in ("phase", "epi", "vf", "topo")}


def _graph():
    graph = nx.path_graph(2)
    graph.graph["DNFR_WEIGHTS"] = _weights("epi")
    for node in graph:
        graph.nodes[node].update(EPI=float(node), nu_f=1.0, theta=0.0, delta_nfr=7.0)
    return graph


def _pressure(graph):
    return tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in graph)


@pytest.mark.parametrize("replace", [False, True])
def test_public_mix_edit_refreshes_engine_and_detached_observer(replace):
    graph = _graph()
    dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == (1.0, -1.0)
    if replace:
        graph.graph["DNFR_WEIGHTS"] = _weights("phase")
    else:
        graph.graph["DNFR_WEIGHTS"].update(_weights("phase"))
    before = deepcopy(graph.graph["_dnfr_weights"])
    observation = capture_non_epi_forcing(graph)
    assert observation.full_kernel_pressure == (0, 0)
    assert dict(observation.normalized_weights) == _weights("phase")
    assert graph.graph["_dnfr_weights"] == before  # detached observation
    dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == (0.0, 0.0)
    assert graph.graph["_DNFR_META"]["weights_effective"] == _weights("phase")


def test_legacy_explicit_mix_survives_until_public_configuration_changes():
    graph = _graph()
    graph.graph["_dnfr_weights"] = _weights("phase")
    dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == (0.0, 0.0)
    graph.graph["DNFR_WEIGHTS"]["epi"] = 2.0
    dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == (1.0, -1.0)


def test_source_snapshot_detaches_accepted_mutable_numeric_weights():
    graph = _graph()
    graph.graph["DNFR_WEIGHTS"] = {
        name: np.array(value) for name, value in _weights("epi").items()
    }
    dnfr.default_compute_delta_nfr(graph)
    graph.graph["DNFR_WEIGHTS"]["epi"][...] = 0.0
    graph.graph["DNFR_WEIGHTS"]["phase"][...] = 1.0
    dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == (0.0, 0.0)


@pytest.mark.parametrize("hook", ["default", "mixed", "laplacian"])
@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("vectorized", [False, True])
def test_scalar_pressure_rejects_rich_form_before_writing(hook, serialized, vectorized):
    graph = _graph()
    graph.graph["vectorized_dnfr"] = vectorized
    rich = BEPIElement((1.0, 2.0), (1.0, 2.0), (0.0, 1.0))
    graph.nodes[1]["EPI"] = serialize_bepi(rich) if serialized else rich
    compute = {
        "default": dnfr.default_compute_delta_nfr,
        "mixed": dnfr.dnfr_epi_vf_mixed,
        "laplacian": dnfr.dnfr_laplacian,
    }[hook]
    with pytest.raises(ValueError, match="uniform-real EPI"):
        compute(graph)
    assert _pressure(graph) == (7.0, 7.0)


@pytest.mark.parametrize("vectorized", [False, True])
def test_uniform_real_embedding_keeps_negative_sign(vectorized):
    graph = _graph()
    graph.graph["vectorized_dnfr"] = vectorized
    graph.nodes[0]["EPI"] = BEPIElement((-2.0, -2.0), (-2.0, -2.0), (0.0, 1.0))
    dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == (3.0, -3.0)


@pytest.mark.parametrize("python_only", [False, True])
def test_optional_hook_rejects_nonfinite_assembled_pressure_before_writes(
    monkeypatch, python_only
):
    graph = _graph()
    graph.nodes[1]["EPI"] = 2.0
    graph.graph["DNFR_WEIGHTS"]["epi"] = 1e308
    if python_only:
        monkeypatch.setattr(dnfr, "np", None)
    with pytest.raises(ValueError, match="finite"):
        dnfr.dnfr_laplacian(graph)
    assert _pressure(graph) == (7.0, 7.0)


def test_optional_models_expose_unweighted_support_and_effective_coefficients():
    graph = nx.Graph([(0, 1), (0, 2)])
    for node in graph:
        graph.nodes[node].update(EPI=float(node == 2), nu_f=1.0, theta=0.0)
    graph[0][1]["weight"] = 1.0
    graph[0][2]["weight"] = 3.0
    graph.graph["DNFR_WEIGHTS"] = _weights("epi")
    dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == (0.75, 0.0, -1.0)
    dnfr.dnfr_laplacian(graph)
    assert _pressure(graph) == (0.5, 0.0, -1.0)
    dnfr.dnfr_epi_vf_mixed(graph)
    assert _pressure(graph) == (0.25, 0.0, -0.5)
    graph.graph["DNFR_WEIGHTS"]["epi"] = 2.0
    dnfr.dnfr_laplacian(graph)
    assert _pressure(graph) == (1.0, 0.0, -2.0)
    meta = graph.graph["_DNFR_META"]
    assert meta["weights_effective"] == {"epi": 2.0, "vf": 0.0}
    assert meta["weights_norm"] == {"epi": 1.0, "vf": 0.0}
