"""Pressure configuration, scalar admission and optional-model boundaries."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import dnfr
from tnfr.errors.contextual import FrequencyError
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


@pytest.mark.parametrize("execution", ["numpy", "python", "process"])
@pytest.mark.parametrize("hook", ["mixed", "laplacian"])
def test_optional_linear_hooks_apply_coefficient_before_range_check(
    hook, execution, monkeypatch
):
    graph = _graph()
    graph.nodes[0]["EPI"], graph.nodes[1]["EPI"] = 1e308, -1e308
    coefficient = 0.5 if hook == "mixed" else 1e-300
    graph.graph["DNFR_WEIGHTS"].update(epi=coefficient, vf=0.0)
    if execution != "numpy":
        monkeypatch.setattr(dnfr, "np", None)
    compute = dnfr.dnfr_epi_vf_mixed if hook == "mixed" else dnfr.dnfr_laplacian
    compute(graph, n_jobs=2 if execution == "process" else None)
    expected = float(
        Fraction.from_float(coefficient)
        * (Fraction.from_float(-1e308) - Fraction.from_float(1e308))
    )
    assert _pressure(graph) == (expected, -expected)
    assert graph.graph["_DNFR_META"]["weights_effective"]["epi"] == coefficient


@pytest.mark.parametrize(
    ("compute", "field", "raw", "python_only"),
    [
        (dnfr.default_compute_delta_nfr, "theta", True, False),
        (dnfr.default_compute_delta_nfr, "theta", "0.5", True),
        (dnfr.default_compute_delta_nfr, "nu_f", -1.0, False),
        (dnfr.default_compute_delta_nfr, "nu_f", -Fraction(1, 2**2000), True),
        (dnfr.dnfr_phase_only, "theta", "0.5", False),
        (dnfr.dnfr_phase_only, "theta", float("nan"), True),
        (dnfr.dnfr_epi_vf_mixed, "nu_f", True, False),
        (dnfr.dnfr_epi_vf_mixed, "nu_f", -1.0, True),
        (dnfr.dnfr_laplacian, "nu_f", "1.0", False),
        (dnfr.dnfr_laplacian, "nu_f", float("inf"), True),
    ],
)
def test_pressure_reads_reject_invalid_raw_state_before_pressure_writes(
    compute, field, raw, python_only, monkeypatch
):
    graph = _graph()
    if python_only:
        monkeypatch.setattr(dnfr, "np", None)
    # A later valid spelling must not hide a malformed authoritative alias.
    aliases = ALIAS_THETA if field == "theta" else ALIAS_VF
    graph.nodes[1][aliases[1]] = 0.5
    graph.nodes[1][aliases[0]] = raw
    before = deepcopy(dict(graph.nodes(data=True)))
    expected = FrequencyError if field == "nu_f" else (TypeError, ValueError)
    with pytest.raises(expected):
        compute(graph)
    assert _pressure(graph) == (7.0, 7.0)
    assert graph.nodes[0] == before[0]
    assert graph.nodes[1][aliases[1]] == 0.5


@pytest.mark.parametrize("field", ["theta", "nu_f"])
def test_cached_pressure_still_validates_current_isolated_state(field):
    graph = _graph()
    graph.add_node(2, EPI=0.0, theta=0.0, nu_f=0.0, delta_nfr=7.0)
    dnfr.default_compute_delta_nfr(graph)
    before = _pressure(graph)
    graph.nodes[2][field] = "0.0"
    expected = FrequencyError if field == "nu_f" else (TypeError, ValueError)
    with pytest.raises(expected):
        dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == before


def test_optional_phase_model_only_requires_its_consumed_coordinates():
    graph = _graph()
    for node in graph:
        graph.nodes[node].update(EPI="unused", nu_f="unused")
    graph.nodes[1]["theta"] = 0.5
    dnfr.dnfr_phase_only(graph)
    assert _pressure(graph) == pytest.approx((0.5 / np.pi, -0.5 / np.pi))


@pytest.mark.parametrize(
    "invalid", [True, "1", -1.0, float("nan"), Fraction(1, 2**2000)]
)
def test_invalid_public_weight_cannot_select_an_alternate_pressure_mix(invalid):
    graph = _graph()
    graph.graph["DNFR_WEIGHTS"]["epi"] = invalid
    with pytest.raises((TypeError, ValueError)):
        dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == (7.0, 7.0)
    assert "_dnfr_weights" not in graph.graph
    with pytest.raises((TypeError, ValueError)):
        capture_non_epi_forcing(graph)


@pytest.mark.parametrize(
    "key,invalid", [("DNFR_WEIGHTS", None), ("_dnfr_weights", {"epi": -1.0})]
)
def test_malformed_weight_configuration_is_rejected_before_preparation(key, invalid):
    graph = _graph()
    graph.graph[key] = invalid
    with pytest.raises((TypeError, ValueError)):
        dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == (7.0, 7.0)
    assert "_dnfr_prep_cache" not in graph.graph


def test_large_finite_mix_normalizes_without_disabling_pressure():
    graph = _graph()
    graph.graph["DNFR_WEIGHTS"].update(epi=1e308, phase=1e308)
    observation = capture_non_epi_forcing(graph)
    assert dict(observation.normalized_weights) == dict(
        epi=0.5, phase=0.5, vf=0.0, topo=0.0
    )
    dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == (0.5, -0.5)
    assert graph.graph["_DNFR_META"]["weights_effective"] == dict(
        observation.normalized_weights
    )


def test_zero_configured_mix_retains_explicit_legacy_uniform_policy():
    graph = _graph()
    graph.graph["DNFR_WEIGHTS"] = dict.fromkeys(("phase", "epi", "vf", "topo"), 0.0)
    dnfr.default_compute_delta_nfr(graph)
    assert _pressure(graph) == (0.25, -0.25)


@pytest.mark.parametrize("disabled", [0, np.bool_(False)])
def test_pressure_execution_respects_shared_false_vectorization_flag(disabled):
    graph = _graph()
    dnfr.default_compute_delta_nfr(graph)  # Warm the NumPy preparation cache.
    graph.graph["vectorized_dnfr"] = disabled
    profile = {}
    dnfr.default_compute_delta_nfr(graph, profile=profile)
    assert profile["dnfr_path"] == "fallback"
    assert _pressure(graph) == (1.0, -1.0)
