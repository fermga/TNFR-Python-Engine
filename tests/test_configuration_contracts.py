"""Configuration isolation and initialization contracts (fixed seeds)."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.config import (
    DEFAULTS,
    TNFRConfig,
    TNFRConfigError,
    apply_config,
    get_config,
    get_graph_param,
    get_param,
    inject_defaults,
    merge_overrides,
)
from tnfr.config.feature_flags import MathFeatureFlags, context_flags, get_flags
from tnfr.constants.aliases import ALIAS_VF
from tnfr.initialization import init_node_attrs


def test_empty_configuration_does_not_restore_unrelated_defaults(tmp_path):
    graph = nx.Graph(DT=0.125, RANDOM_SEED=37)
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    apply_config(graph, path)
    assert graph.graph["DT"] == 0.125
    assert graph.graph["RANDOM_SEED"] == 37
    assert "VF_MAX" not in graph.graph


def test_explicit_empty_instance_defaults_are_respected():
    graph = nx.Graph()
    TNFRConfig({"DT": 0.125}).inject_defaults(graph, defaults={})
    assert "DT" not in graph.graph


def test_mutable_fallback_is_isolated_from_other_graphs():
    original = deepcopy(DEFAULTS["COHERENCE"])
    try:
        fallback = get_param(nx.Graph(), "COHERENCE")
        fallback["weights"]["phase"] = 123.0
        assert get_param(nx.Graph(), "COHERENCE") == original
        graph = nx.Graph()
        inject_defaults(graph)
        assert graph.graph["COHERENCE"] == original
    finally:
        DEFAULTS["COHERENCE"].clear()
        DEFAULTS["COHERENCE"].update(original)


def test_graph_owned_configuration_remains_mutable():
    graph = nx.Graph(COHERENCE={"enabled": True})
    get_param(graph, "COHERENCE")["enabled"] = False
    assert graph.graph["COHERENCE"] == {"enabled": False}


def test_instance_defaults_are_owned_snapshot():
    supplied = {"COHERENCE": {"enabled": True}}
    config = TNFRConfig(supplied)
    supplied["COHERENCE"]["enabled"] = False
    assert config.get_param_with_fallback({}, "COHERENCE") == {"enabled": True}


def test_global_configuration_exposes_canonical_defaults():
    assert get_config().get_param_with_fallback({}, "DT") == DEFAULTS["DT"]


@pytest.mark.parametrize("apply", [
    lambda graph: inject_defaults(graph, {"VF_MIN": 2.0}, override=True),
    lambda graph: merge_overrides(graph, VF_MIN=2.0),
])
def test_invalid_effective_bounds_fail_before_graph_mutation(apply):
    graph = nx.Graph(VF_MIN=0.0, VF_MAX=1.0)
    before = deepcopy(graph.graph)
    with pytest.raises(TNFRConfigError):
        apply(graph)
    assert graph.graph == before


def test_ignored_defaults_do_not_override_valid_graph_bounds():
    graph = nx.Graph(VF_MIN=0.0, VF_MAX=1.0)
    inject_defaults(graph, {"VF_MIN": 2.0, "VF_MAX": 1.0})
    assert graph.graph["VF_MIN"] == 0.0


def test_unknown_override_is_atomic():
    graph = nx.Graph(DT=0.125)
    with pytest.raises(KeyError):
        merge_overrides(graph, DT=0.25, UNKNOWN_PARAMETER=1)
    assert graph.graph == {"DT": 0.125}


@pytest.mark.parametrize("key", [
    "DT", "VF_MIN", "VF_MAX", "EPI_MIN", "EPI_MAX", "INIT_THETA_MIN",
    "INIT_THETA_MAX",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), "invalid"])
def test_numeric_configuration_rejects_nonfinite_or_nonnumeric(key, value):
    with pytest.raises(TNFRConfigError, match=key):
        TNFRConfig().validate_config({key: value})


@pytest.mark.parametrize("step", [0, 0.0, np.float32(0), np.float32(0.125)])
def test_valid_numeric_steps_include_zero_noop(step):
    graph = nx.Graph()
    inject_defaults(graph, {"DT": step})
    assert graph.graph["DT"] == step


def test_negative_frequency_maximum_is_invalid_without_minimum():
    with pytest.raises(TNFRConfigError, match="VF_MAX"):
        TNFRConfig().validate_config({"VF_MAX": -1})


@pytest.mark.parametrize("random_phase", [True, False])
def test_initialization_preserves_legacy_aliases_including_zero(random_phase):
    graph = nx.Graph(RANDOM_SEED=37, INIT_RANDOM_PHASE=random_phase)
    values = {"phase": 0.7, "nu_f": 0.0, "psi": 0.4, "sense_index": 0.0}
    graph.add_node("legacy", **values)
    init_node_attrs(graph, override=False)
    assert dict(graph.nodes["legacy"]) == values
    assert get_attr(graph.nodes["legacy"], ALIAS_VF) == 0.0


def test_initialization_writes_existing_aliases_on_override():
    graph = nx.Graph(RANDOM_SEED=37, INIT_RANDOM_PHASE=False,
                     INIT_VF_MIN=0.0, INIT_VF_MAX=0.0,
                     INIT_SI_MIN=0.0, INIT_SI_MAX=0.0, INIT_EPI_VALUE=0.0)
    graph.add_node(0, phase=0.7, nu_f=1.0, psi=0.4, sense_index=0.2)
    init_node_attrs(graph)
    assert dict(graph.nodes[0]) == {
        "phase": 0.0, "nu_f": 0.0, "psi": 0.0, "sense_index": 0.0,
    }


def test_seeded_initialization_is_reproducible_and_seed_sensitive():
    graphs = [nx.path_graph(5) for _ in range(3)]
    for graph, seed in zip(graphs, (0, 0, 1)):
        graph.graph["RANDOM_SEED"] = seed
        init_node_attrs(graph)
    assert dict(graphs[0].nodes(data=True)) == dict(graphs[1].nodes(data=True))
    assert dict(graphs[0].nodes(data=True)) != dict(graphs[2].nodes(data=True))


def test_feature_flag_context_is_thread_local(monkeypatch):
    from tnfr.config import feature_flags
    monkeypatch.setattr(feature_flags, "_BASE_FLAGS", MathFeatureFlags())
    with context_flags(enable_math_dynamics=True):
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker_flags = executor.submit(get_flags).result(timeout=5)
        assert get_flags().enable_math_dynamics is True
    assert worker_flags.enable_math_dynamics is False
    assert get_flags().enable_math_dynamics is False


def test_nested_feature_flags_restore_after_exception():
    before = get_flags()
    with context_flags(enable_math_dynamics=True):
        with pytest.raises(RuntimeError):
            with context_flags(enable_math_dynamics=False):
                assert get_flags().enable_math_dynamics is False
                raise RuntimeError("controlled failure")
        assert get_flags().enable_math_dynamics is True
    assert get_flags() == before


def test_feature_flag_contexts_are_isolated_across_async_tasks():
    async def exercise():
        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        first_read = asyncio.Event()

        async def first():
            with context_flags(enable_math_dynamics=True):
                first_entered.set()
                await second_entered.wait()
                value = get_flags().enable_math_dynamics
                first_read.set()
                return value

        async def second():
            await first_entered.wait()
            with context_flags(enable_math_dynamics=False):
                second_entered.set()
                await first_read.wait()
                return get_flags().enable_math_dynamics

        return await asyncio.gather(first(), second())

    assert asyncio.run(exercise()) == [True, False]


def test_preset_callers_cannot_mutate_future_operator_sequences():
    from tnfr.config.presets import get_preset
    preset = get_preset("contained_mutation")
    before = deepcopy(preset)
    try:
        preset[2].repeat = 99
        preset.append("changed")
        assert get_preset("contained_mutation") == before
    finally:
        preset[:] = before


@pytest.mark.parametrize("module_name", ["init", "metric"])
def test_legacy_defaults_reexport_canonical_objects(module_name):
    from importlib import import_module
    legacy = import_module(f"tnfr.constants.{module_name}")
    canonical = import_module(f"tnfr.config.defaults_{module_name}")
    # The legacy module's public names remain available, including helper imports.
    for name in vars(legacy):
        if not name.startswith("_"):
            assert getattr(legacy, name) is getattr(canonical, name)


@pytest.mark.parametrize("key,value", [
    ("INIT_EPI_VALUE", float("nan")), ("INIT_VF_MIN", float("inf")),
    ("INIT_VF_STD", -1), ("INIT_SI_MAX", float("inf")),
    ("RANDOM_SEED", 1.5), ("RANDOM_SEED", float("nan")),
])
def test_invalid_initialization_parameters_fail_before_node_mutation(key, value):
    graph = nx.path_graph(2)
    graph.graph[key] = value
    before = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises(TNFRConfigError, match=key):
        init_node_attrs(graph)
    assert dict(graph.nodes(data=True)) == before


def test_none_initialization_seed_records_realized_integer():
    graph = nx.path_graph(2)
    graph.graph["RANDOM_SEED"] = None
    init_node_attrs(graph)
    assert type(graph.graph["RANDOM_SEED"]) is int
    assert all(np.isfinite(value) for _, attrs in graph.nodes(data=True)
               for value in attrs.values())


@pytest.mark.parametrize("raw", [False, 0, "false", " FALSE ", "off", "0"])
def test_graph_boolean_conversion_preserves_explicit_false(raw):
    graph = nx.Graph(INIT_RANDOM_PHASE=raw)
    assert get_graph_param(graph, "INIT_RANDOM_PHASE", bool) is False
    graph.add_node(0)
    init_node_attrs(graph)
    assert graph.nodes[0]["theta"] == 0.0


def test_flag_override_false_string_is_not_truthy():
    with context_flags(enable_math_dynamics="false"):
        assert get_flags().enable_math_dynamics is False


def test_backend_configuration_rejects_methods_and_is_atomic(monkeypatch):
    from tnfr import backend_config
    from tnfr.errors import TNFRValueError
    config = backend_config.TNFRConfig()
    monkeypatch.setattr(backend_config, "_global_config", config)
    before = config.cuda_enabled
    with pytest.raises(TNFRValueError):
        backend_config.configure(cuda_enabled=not before, get_backend_config=False)
    assert config.cuda_enabled is before
    assert callable(config.get_backend_config)


@pytest.mark.parametrize("raw", [" true ", "enabled", "yes"])
def test_backend_boolean_environment_matches_graph_parser(monkeypatch, raw):
    from tnfr.backend_config import TNFRConfig as BackendConfig
    monkeypatch.setenv("TNFR_CUDA_ENABLED", raw)
    assert BackendConfig(cuda_enabled=False).cuda_enabled is True


def test_invalid_backend_boolean_environment_preserves_default(monkeypatch):
    from tnfr.backend_config import TNFRConfig as BackendConfig
    monkeypatch.setenv("TNFR_CUDA_ENABLED", "unrecognized")
    with pytest.warns(UserWarning, match="TNFR_CUDA_ENABLED"):
        assert BackendConfig(cuda_enabled=True).cuda_enabled is True
