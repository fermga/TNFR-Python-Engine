"""Unit tests for sense utilities computing sigma vectors and glyph metrics."""

import math
import time

import pytest

from tnfr.callback_utils import CallbackEvent, callback_manager
from tnfr.glyph_history import ensure_history
from tnfr.sense import (
    _node_weight,
    _sigma_from_iterable,
    glyph_angle,
    glyph_unit,
    push_sigma_snapshot,
    register_sigma_callback,
    sigma_rose,
    sigma_vector_from_graph,
    sigma_vector_node,
)
from tnfr.types import Glyph


def _make_graph(graph_canon):
    G = graph_canon()
    G.add_node(0, glyph_history=[Glyph.AL.value], Si=1.0, EPI=2.0)
    G.add_node(1, Si=0.3, EPI=1.5)
    return G


def test_sigma_vector_node_paths(graph_canon):
    G = _make_graph(graph_canon)
    sv_si = sigma_vector_node(G, 0)
    assert sv_si and sv_si["glyph"] == Glyph.AL.value
    assert sv_si["w"] == 1.0
    assert sigma_vector_node(G, 1) is None
    sv_epi = sigma_vector_node(G, 0, weight_mode="EPI")
    assert sv_epi["w"] == 2.0
    assert sv_epi["mag"] == pytest.approx(2 * sv_si["mag"])


def test_sigma_vector_from_graph_paths(graph_canon):
    G = _make_graph(graph_canon)
    sv_si = sigma_vector_from_graph(G)
    sv_epi = sigma_vector_from_graph(G, weight_mode="EPI")
    assert sv_si["n"] == 1
    assert sv_epi["n"] == 1
    assert sv_epi["mag"] == pytest.approx(2 * sv_si["mag"])


def test_sigma_vector_from_graph_rejects_non_graph(graph_canon):
    with pytest.raises(TypeError, match="requires a networkx.Graph"):
        sigma_vector_from_graph(object())

    # Positive path sanity check to ensure the canonical fixture still works.
    G = _make_graph(graph_canon)
    vec = sigma_vector_from_graph(G)
    assert vec["n"] == 1


def _sigma_vector_from_graph_naive(G, weight_mode: str = "Si"):
    """Reference implementation recomputing ``glyph_unit(g) * w`` at each step."""
    pairs = []
    for _, nd in G.nodes(data=True):
        nw = _node_weight(nd, weight_mode)
        if not nw:
            continue
        g, w, _ = nw
        pairs.append((g, w))
    vectors = (glyph_unit(g) * float(w) for g, w in pairs)
    vec = _sigma_from_iterable(vectors)
    return vec


def test_sigma_vector_from_graph_matches_naive(graph_canon):
    """The optimized version matches the naive computation and is not slower."""
    G_opt = graph_canon()
    glyphs = list(Glyph)
    for i in range(1000):
        g = glyphs[i % len(glyphs)].value
        G_opt.add_node(i, glyph_history=[g], Si=float(i % 10) / 10)
    G_ref = G_opt.copy()

    start = time.perf_counter()
    vec_opt = sigma_vector_from_graph(G_opt)
    t_opt = time.perf_counter() - start

    start = time.perf_counter()
    vec_ref = _sigma_vector_from_graph_naive(G_ref)
    t_ref = time.perf_counter() - start

    for key in ("x", "y", "mag", "angle", "n"):
        assert vec_opt[key] == pytest.approx(vec_ref[key])
    assert t_opt <= t_ref * 2


def test_sigma_from_iterable_rejects_str():
    with pytest.raises(TypeError, match="real or complex"):
        _sigma_from_iterable("abc")


def test_sigma_from_iterable_rejects_bytes():
    with pytest.raises(TypeError, match="real or complex"):
        _sigma_from_iterable(b"\x01\x02")


def test_sigma_from_iterable_accepts_reals():
    vec = _sigma_from_iterable([1.0, 3.0])
    assert vec["n"] == 2
    assert vec["x"] == pytest.approx(2.0)
    assert vec["y"] == pytest.approx(0.0)


def test_sigma_from_iterable_large_generator_efficient():
    N = 100_000
    counter = 0

    def gen():
        nonlocal counter
        for i in range(N):
            counter += 1
            yield float(i)

    start = time.perf_counter()
    vec = _sigma_from_iterable(gen())
    elapsed = time.perf_counter() - start

    assert vec["n"] == N
    assert vec["x"] == pytest.approx((N - 1) / 2)
    assert vec["y"] == pytest.approx(0.0)
    assert counter == N
    assert elapsed < 2.0


def test_unknown_glyph_raises():
    with pytest.raises(KeyError):
        glyph_angle("ZZ")
    with pytest.raises(KeyError):
        glyph_unit("ZZ")


def test_sigma_rose_valid_and_invalid_steps(graph_canon):
    G = graph_canon()
    G.graph["history"] = {
        "sigma_counts": [
            {"t": 0, Glyph.AL.value: 1},
            {"t": 1, Glyph.AL.value: 2, Glyph.EN.value: 1},
            {"t": 2, Glyph.EN.value: 3},
        ]
    }
    res = sigma_rose(G, steps=2.0)
    assert res[Glyph.AL.value] == 2
    assert res[Glyph.EN.value] == 4
    with pytest.raises(ValueError):
        sigma_rose(G, steps=-1)


def test_push_sigma_snapshot_disabled_skips_history(graph_canon):
    G = _make_graph(graph_canon)
    G.graph["SIGMA"] = dict(G.graph["SIGMA"], enabled=False)
    sentinel_list = ["baseline"]
    hist = {"untouched": sentinel_list}
    G.graph["history"] = hist

    push_sigma_snapshot(G)

    assert G.graph["history"] is hist
    assert list(hist) == ["untouched"]
    assert hist["untouched"] is sentinel_list


def test_push_sigma_snapshot_applies_smoothing(graph_canon):
    G = _make_graph(graph_canon)
    alpha = 0.4
    G.graph["SIGMA"] = dict(G.graph["SIGMA"], enabled=True, smooth=alpha)
    hist = ensure_history(G)
    previous = {
        "x": 0.5,
        "y": -0.25,
        "mag": math.hypot(0.5, -0.25),
        "angle": math.atan2(-0.25, 0.5),
        "n": 3,
        "t": 1.0,
    }
    hist.setdefault("sigma_global", []).append(previous.copy())
    original_counts_len = len(hist.get("sigma_counts", []))

    current_vector = sigma_vector_from_graph(G)
    current_time = 2.5
    G.graph["_t"] = current_time

    push_sigma_snapshot(G)

    sigma_history = hist["sigma_global"]
    assert len(sigma_history) == 2
    new_entry = sigma_history[-1]

    expected_x = (1 - alpha) * previous["x"] + alpha * current_vector["x"]
    expected_y = (1 - alpha) * previous["y"] + alpha * current_vector["y"]
    assert new_entry["x"] == pytest.approx(expected_x)
    assert new_entry["y"] == pytest.approx(expected_y)
    assert new_entry["mag"] == pytest.approx(math.hypot(expected_x, expected_y))
    assert new_entry["angle"] == pytest.approx(math.atan2(expected_y, expected_x))
    assert new_entry["n"] == current_vector["n"]
    assert new_entry["t"] == pytest.approx(current_time)

    counts = hist["sigma_counts"]
    assert len(counts) == original_counts_len + 1
    assert counts[-1]["t"] == pytest.approx(current_time)


def test_push_sigma_snapshot_records_per_node_traces(graph_canon):
    G = graph_canon()
    G.graph["SIGMA"] = dict(G.graph["SIGMA"], enabled=True, per_node=True)
    G.graph["_t"] = 7.0
    G.add_node(0, glyph_history=[Glyph.AL.value, Glyph.EN.value], Si=1.0)
    G.add_node(1, glyph_history=[Glyph.IL.value], Si=0.4)
    G.add_node(2, Si=0.2)
    hist = ensure_history(G)
    original_counts_len = len(hist.get("sigma_counts", []))

    push_sigma_snapshot(G)

    per_node = hist["sigma_per_node"]
    assert set(per_node) == {0, 1}

    node0_entry = per_node[0][-1]
    assert node0_entry["t"] == pytest.approx(7.0)
    assert node0_entry["g"] == Glyph.EN.value
    assert node0_entry["angle"] == pytest.approx(glyph_angle(Glyph.EN.value))

    node1_entry = per_node[1][-1]
    assert node1_entry["t"] == pytest.approx(7.0)
    assert node1_entry["g"] == Glyph.IL.value
    assert node1_entry["angle"] == pytest.approx(glyph_angle(Glyph.IL.value))

    counts = hist["sigma_counts"]
    assert len(counts) == original_counts_len + 1
    counts_entry = counts[-1]
    assert counts_entry["t"] == pytest.approx(7.0)
    assert counts_entry[Glyph.EN.value] == 1
    assert counts_entry[Glyph.IL.value] == 1


def test_register_sigma_callback_attaches_after_step(graph_canon):
    G = graph_canon()

    register_sigma_callback(G)

    registry = callback_manager._ensure_callbacks(G)
    after_step = registry[CallbackEvent.AFTER_STEP.value]
    spec = after_step["sigma_snapshot"]

    assert spec.name == "sigma_snapshot"
    assert spec.func is push_sigma_snapshot
