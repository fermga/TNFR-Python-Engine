"""Recorded graph seeds reproduce initialization and runtime random streams."""

from copy import deepcopy
import hashlib
import json
import random
import struct
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.config.tnfr_config import TNFRConfigError
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_VF
from tnfr.dynamics.sampling import update_node_sample
from tnfr.initialization import init_node_attrs
from tnfr.node import NodeNX
from tnfr.operators import _um_select_candidates, apply_glyph
from tnfr.operators.definitions import Dissonance
from tnfr.operators.jitter import JitterCache, get_jitter_manager, random_jitter, reset_jitter_manager
from tnfr.operators.remesh import apply_topological_remesh
from tnfr.rng import base_seed, make_rng, resolve_graph_seed, validate_graph_seed


def _graph(seed, n=60):
    graph = nx.path_graph(n)
    graph.graph.update(RANDOM_SEED=seed, UM_CANDIDATE_COUNT=7,
                       OZ_NOISE_MODE=True, OZ_SIGMA=0.1)
    return graph


def _legacy_rng(seed, key):
    """Independent pre-change stream specification, including signed masking."""
    payload = struct.pack(">QQ", seed & ((1 << 64) - 1), key & ((1 << 64) - 1))
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


@pytest.mark.parametrize("seed", [0, 1, -1, 2**130 + 9, np.int64(17)])
def test_integer_factory_and_sampling_keep_legacy_stream(seed):
    graph = _graph(seed)
    rng = make_rng(seed, -1, graph)
    expected = _legacy_rng(int(seed), -1)
    assert [rng.random() for _ in range(12)] == [expected.random() for _ in range(12)]
    update_node_sample(graph, step=3)
    assert graph.graph["_node_sample"] == _legacy_rng(int(seed), 3).sample(tuple(graph), 7)


@pytest.mark.parametrize("seed", [0, 17, -5])
def test_integer_initialization_keeps_legacy_draw_order(seed):
    graph = _graph(seed, 4)
    graph.graph.update(INIT_THETA_MIN=-1, INIT_THETA_MAX=1,
                       INIT_VF_MODE="uniform", INIT_VF_MIN=0.2, INIT_VF_MAX=0.7,
                       INIT_SI_MIN=0.1, INIT_SI_MAX=0.9)
    init_node_attrs(graph)
    rng = _legacy_rng(seed, -1)
    for _, attrs in graph.nodes(data=True):
        assert attrs["theta"] == rng.uniform(-1, 1)
        assert get_attr(attrs, ALIAS_VF) == rng.uniform(0.2, 0.7)
        assert attrs["Si"] == rng.uniform(0.1, 0.9)


def test_none_draws_entropy_once_even_for_concurrent_consumers(monkeypatch):
    calls = []
    monkeypatch.setattr(random.SystemRandom, "getrandbits",
                        lambda self, bits: calls.append(bits) or 123456789)
    graph = _graph(None)
    assert validate_graph_seed(graph) is None
    assert calls == []
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lambda _: resolve_graph_seed(graph), range(20))) == [123456789] * 20
    assert graph.graph["RANDOM_SEED"] == 123456789
    init_node_attrs(graph)
    update_node_sample(graph, step=0)
    apply_glyph(graph, 0, "dissonance")
    apply_topological_remesh(graph, mode="knn", k=2)
    assert base_seed(graph) == 123456789
    assert calls == [64]


@pytest.mark.parametrize("seed", [None, 0, 7])
def test_recorded_seed_replays_initialization_operators_sampling_and_remesh(seed):
    source = _graph(seed)
    init_node_attrs(source)
    replay = _graph(source.graph["RANDOM_SEED"])
    init_node_attrs(replay)
    assert dict(source.nodes(data=True)) == dict(replay.nodes(data=True))
    for step in range(4):
        update_node_sample(source, step=step)
        update_node_sample(replay, step=step)
        assert source.graph["_node_sample"] == replay.graph["_node_sample"]
        for node in (0, 3, 0):
            apply_glyph(source, node, "dissonance")
            apply_glyph(replay, node, "dissonance")
            assert get_attr(source.nodes[node], ALIAS_DNFR) == get_attr(replay.nodes[node], ALIAS_DNFR)
    apply_topological_remesh(source, mode="knn", k=2, p_rewire=0.8)
    apply_topological_remesh(replay, mode="knn", k=2, p_rewire=0.8)
    assert set(source.edges()) == set(replay.edges())
    for node in (0, 3):
        assert source.nodes[node]["_rng_jitter_progress"] == replay.nodes[node]["_rng_jitter_progress"]


def test_coupling_candidate_stream_accepts_none_and_replays_recorded_seed():
    source = _graph(None, 12)
    source_node = NodeNX(source, 0)
    chosen = _um_select_candidates(source_node, iter(NodeNX(source, n) for n in range(1, 12)),
                                   4, "random", 0)
    replay = _graph(source.graph["RANDOM_SEED"], 12)
    repeated = _um_select_candidates(NodeNX(replay, 0), iter(NodeNX(replay, n) for n in range(1, 12)),
                                     4, "random", 0)
    assert [n.n for n in chosen] == [n.n for n in repeated]


def test_copy_after_resolution_has_independent_recorded_jitter_progress():
    source = _graph(None, 3)
    init_node_attrs(source)
    random_jitter(NodeNX(source, 0), 0.2)
    copied = source.copy()
    recorded = deepcopy(source.nodes[0]["_rng_jitter_progress"])
    source_next = random_jitter(NodeNX(source, 0), 0.2)
    assert copied.nodes[0]["_rng_jitter_progress"] == recorded
    assert random_jitter(NodeNX(copied, 0), 0.2) == source_next
    assert source.nodes[0]["_rng_jitter_progress"] == copied.nodes[0]["_rng_jitter_progress"]
    assert source.nodes[0]["_rng_jitter_progress"] is not copied.nodes[0]["_rng_jitter_progress"]


def test_unresolved_copy_gets_its_own_entropy(monkeypatch):
    seeds = iter([101, 202])
    monkeypatch.setattr(random.SystemRandom, "getrandbits", lambda self, bits: next(seeds))
    source = _graph(None, 2)
    copied = source.copy()
    assert resolve_graph_seed(source) == 101
    assert copied.graph["RANDOM_SEED"] is None
    assert resolve_graph_seed(copied) == 202


def test_json_recorded_progress_continues_stream_without_cache_state():
    source = _graph(7, 2)
    node = NodeNX(source, 0)
    random_jitter(node, 0.2)
    persisted = json.loads(json.dumps({"RANDOM_SEED": source.graph["RANDOM_SEED"],
                                      "progress": source.nodes[0]["_rng_jitter_progress"]}))
    replay = nx.path_graph(2)
    replay.graph["RANDOM_SEED"] = persisted["RANDOM_SEED"]
    replay.nodes[0]["_rng_jitter_progress"] = persisted["progress"]
    assert random_jitter(node, 0.2) == random_jitter(NodeNX(replay, 0), 0.2)


def test_cache_clearing_and_disabled_cache_do_not_change_jitter_stream():
    enabled, disabled = _graph(0, 2), _graph(0, 2)
    enabled.graph["JITTER_CACHE_SIZE"] = 128
    disabled.graph["JITTER_CACHE_SIZE"] = 0
    expected = []
    actual = []
    for _ in range(4):
        expected.append(random_jitter(NodeNX(enabled, 0), 0.1))
        reset_jitter_manager()
        actual.append(random_jitter(NodeNX(disabled, 0), 0.1))
    assert expected == actual
    assert len(set(actual)) == 4


def test_legacy_jitter_cache_settings_remain_available():
    cache = JitterCache(max_entries=5)
    assert cache.settings == {"max_entries": 5}
    manager = get_jitter_manager()
    assert manager.settings["max_entries"] == manager.max_entries


def test_zero_amplitude_does_not_resolve_seed_or_advance_draw():
    graph = _graph(None, 2)
    assert random_jitter(NodeNX(graph, 0), 0) == 0
    assert graph.graph["RANDOM_SEED"] is None
    assert "_rng_jitter_progress" not in graph.nodes[0]


def test_seed_change_restarts_jitter_and_node_streams_do_not_alias():
    graph = _graph(0, 2)
    zero = NodeNX(graph, 0)
    random_jitter(zero, 0.2)
    assert random_jitter(zero, 0.2) != random_jitter(NodeNX(graph, 1), 0.2)
    graph.graph["RANDOM_SEED"] = 17
    replay = _graph(17, 2)
    assert random_jitter(zero, 0.2) == random_jitter(NodeNX(replay, 0), 0.2)


def test_progress_validation_is_local_to_the_selected_node():
    graph = _graph(7, 100)
    for node in range(1, 100):
        graph.nodes[node]["_rng_jitter_progress"] = "unreadable unrelated record"
    assert -0.1 <= random_jitter(NodeNX(graph, 0), 0.1) <= 0.1
    assert graph.nodes[0]["_rng_jitter_progress"] == {"seed": 7, "offset": 0, "draws": 1}
    assert graph.nodes[99]["_rng_jitter_progress"] == "unreadable unrelated record"


def test_offset_change_starts_the_new_offset_stream():
    graph = _graph(7, 3)
    random_jitter(NodeNX(graph, 1), 0.2)
    graph.remove_node(0)
    replay = nx.Graph()
    replay.add_nodes_from([1, 2])
    replay.graph["RANDOM_SEED"] = 7
    assert random_jitter(NodeNX(graph, 1), 0.2) == random_jitter(NodeNX(replay, 1), 0.2)
    assert graph.nodes[1]["_rng_jitter_progress"] == {"seed": 7, "offset": 0, "draws": 1}


@pytest.mark.parametrize("seed", [True, False, "0", 1.0, 1.5, float("nan"), float("inf")])
@pytest.mark.parametrize("operation", ["initialize", "sample", "glyph", "public_operator", "remesh"])
def test_invalid_seed_is_rejected_before_mutation(seed, operation):
    graph = _graph(seed, 2)
    before_graph = dict(graph.graph)
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_edges = list(graph.edges(data=True))
    with pytest.raises((ValueError, TNFRConfigError), match="RANDOM_SEED"):
        if operation == "initialize":
            init_node_attrs(graph)
        elif operation == "sample":
            update_node_sample(graph, step=0)
        elif operation == "glyph":
            apply_glyph(graph, 0, "dissonance")
        elif operation == "public_operator":
            Dissonance()(graph, 0)
        else:
            apply_topological_remesh(graph)
    assert graph.graph == before_graph
    assert dict(graph.nodes(data=True)) == before_nodes
    assert list(graph.edges(data=True)) == before_edges


@pytest.mark.parametrize("seed", [True, "2", 2.5])
def test_invalid_remesh_seed_override_is_rejected_even_on_empty_graph(seed):
    graph = _graph(None, 0)
    with pytest.raises(ValueError, match="RANDOM_SEED"):
        apply_topological_remesh(graph, seed=seed)
    assert graph.graph["RANDOM_SEED"] is None


def test_invalid_progress_is_rejected_before_seed_resolution():
    graph = _graph(None, 2)
    graph.nodes[0]["_rng_jitter_progress"] = {"seed": 1, "offset": 0, "draws": -1}
    node = NodeNX(graph, 0)
    with pytest.raises(ValueError, match="draws must be"):
        random_jitter(node, 0.1)
    assert graph.graph["RANDOM_SEED"] is None
    assert graph.nodes[0]["_rng_jitter_progress"]["draws"] == -1
