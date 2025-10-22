"""Unit tests for metric evaluation plumbing and configuration."""



import builtins
import pytest
from typing import Any

from tnfr.constants import (
    inject_defaults,
    get_aliases,
)
from tnfr.alias import get_attr, set_attr
from tnfr.metrics.coherence import (
    GLYPH_LOAD_STABILIZERS_KEY,
    _track_stability,
    _aggregate_si,
    _update_sigma,
)
from tnfr.metrics.core import _metrics_step, register_metrics_callbacks
from tnfr.metrics.glyph_timing import (
    LATENT_GLYPH,
    _update_latency_index,
    _update_epi_support,
    _compute_advanced_metrics,
    np as glyph_numpy,
)
from tnfr.metrics.glyph_timing import DEFAULT_EPI_SUPPORT_LIMIT
from tnfr.metrics.reporting import build_metrics_summary

ALIAS_EPI = get_aliases("EPI")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_DEPI = get_aliases("DEPI")
ALIAS_SI = get_aliases("SI")
ALIAS_DSI = get_aliases("DSI")
ALIAS_VF = get_aliases("VF")
ALIAS_DVF = get_aliases("DVF")
ALIAS_D2VF = get_aliases("D2VF")


def test_track_stability_updates_hist(graph_canon):
    """_track_stability aggregates stability and derivatives."""

    G = graph_canon()
    hist = {"stable_frac": [], "delta_Si": [], "B": []}

    G.add_node(0)
    G.add_node(1)

    # Node 0: stable
    set_attr(G.nodes[0], ALIAS_DNFR, 0.0)
    set_attr(G.nodes[0], ALIAS_DEPI, 0.0)
    set_attr(G.nodes[0], ALIAS_SI, 2.0)
    G.nodes[0]["_prev_Si"] = 1.0
    set_attr(G.nodes[0], ALIAS_VF, 1.0)
    G.nodes[0]["_prev_vf"] = 0.5
    G.nodes[0]["_prev_dvf"] = 0.2

    # Node 1: unstable
    set_attr(G.nodes[1], ALIAS_DNFR, 10.0)
    set_attr(G.nodes[1], ALIAS_DEPI, 10.0)
    set_attr(G.nodes[1], ALIAS_SI, 3.0)
    G.nodes[1]["_prev_Si"] = 1.0
    set_attr(G.nodes[1], ALIAS_VF, 1.0)
    G.nodes[1]["_prev_vf"] = 1.0
    G.nodes[1]["_prev_dvf"] = 0.0

    _track_stability(G, hist, dt=1.0, eps_dnfr=1.0, eps_depi=1.0, n_jobs=None)

    assert hist["stable_frac"] == [0.5]
    assert hist["delta_Si"] == [pytest.approx(1.5)]
    assert hist["B"] == [pytest.approx(0.15)]


def test_track_stability_vectorized_updates_graph(monkeypatch, graph_canon):
    """Vectorized tracking updates history and nodal derivatives."""

    np_mod = pytest.importorskip("numpy")
    monkeypatch.setattr("tnfr.metrics.coherence.get_numpy", lambda: np_mod)

    G = graph_canon()
    hist = {"stable_frac": [], "delta_Si": [], "B": []}

    G.add_node(0)
    G.add_node(1)

    set_attr(G.nodes[0], ALIAS_DNFR, 0.0)
    set_attr(G.nodes[0], ALIAS_DEPI, 0.0)
    set_attr(G.nodes[0], ALIAS_SI, 2.0)
    G.nodes[0]["_prev_Si"] = 1.0
    set_attr(G.nodes[0], ALIAS_VF, 1.0)
    G.nodes[0]["_prev_vf"] = 0.5
    G.nodes[0]["_prev_dvf"] = 0.2

    set_attr(G.nodes[1], ALIAS_DNFR, 10.0)
    set_attr(G.nodes[1], ALIAS_DEPI, 10.0)
    set_attr(G.nodes[1], ALIAS_SI, 3.0)
    G.nodes[1]["_prev_Si"] = 1.0
    set_attr(G.nodes[1], ALIAS_VF, 1.0)
    G.nodes[1]["_prev_vf"] = 1.0
    G.nodes[1]["_prev_dvf"] = 0.0

    _track_stability(G, hist, dt=1.0, eps_dnfr=1.0, eps_depi=1.0, n_jobs=None)

    assert hist["stable_frac"][-1] == pytest.approx(0.5)
    assert hist["delta_Si"][-1] == pytest.approx(1.5)
    assert hist["B"][-1] == pytest.approx(0.15)

    assert G.nodes[0]["_prev_Si"] == pytest.approx(2.0)
    assert get_attr(G.nodes[0], ALIAS_DSI) == pytest.approx(1.0)
    assert G.nodes[0]["_prev_vf"] == pytest.approx(1.0)
    assert get_attr(G.nodes[0], ALIAS_DVF) == pytest.approx(0.5)
    assert get_attr(G.nodes[0], ALIAS_D2VF) == pytest.approx(0.3)

    assert G.nodes[1]["_prev_Si"] == pytest.approx(3.0)
    assert get_attr(G.nodes[1], ALIAS_DSI) == pytest.approx(2.0)
    assert G.nodes[1]["_prev_vf"] == pytest.approx(1.0)
    assert get_attr(G.nodes[1], ALIAS_DVF) == pytest.approx(0.0)
    assert get_attr(G.nodes[1], ALIAS_D2VF) == pytest.approx(0.0)


def test_track_stability_parallel_fallback(monkeypatch, graph_canon):
    """Fallback path splits work across processes when NumPy is absent."""

    monkeypatch.setattr("tnfr.metrics.coherence.get_numpy", lambda: None)

    created: list[Any] = []

    class _FakeFuture:
        def __init__(self, value: Any):
            self._value = value

        def result(self) -> Any:
            return self._value

    class _RecorderExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers
            self.submissions: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

        def __enter__(self):
            created.append(self)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):  # noqa: ANN001 - testing stub
            self.submissions.append((fn, args, kwargs))
            return _FakeFuture(fn(*args, **kwargs))

    monkeypatch.setattr("tnfr.metrics.coherence.ProcessPoolExecutor", _RecorderExecutor)

    G = graph_canon()
    hist = {"stable_frac": [], "delta_Si": [], "B": []}

    G.add_node(0)
    G.add_node(1)

    set_attr(G.nodes[0], ALIAS_DNFR, 0.0)
    set_attr(G.nodes[0], ALIAS_DEPI, 0.0)
    set_attr(G.nodes[0], ALIAS_SI, 2.0)
    G.nodes[0]["_prev_Si"] = 1.0
    set_attr(G.nodes[0], ALIAS_VF, 1.0)
    G.nodes[0]["_prev_vf"] = 0.5
    G.nodes[0]["_prev_dvf"] = 0.2

    set_attr(G.nodes[1], ALIAS_DNFR, 10.0)
    set_attr(G.nodes[1], ALIAS_DEPI, 10.0)
    set_attr(G.nodes[1], ALIAS_SI, 3.0)
    G.nodes[1]["_prev_Si"] = 1.0
    set_attr(G.nodes[1], ALIAS_VF, 1.0)
    G.nodes[1]["_prev_vf"] = 1.0
    G.nodes[1]["_prev_dvf"] = 0.0

    _track_stability(G, hist, dt=1.0, eps_dnfr=1.0, eps_depi=1.0, n_jobs=2)

    assert created and created[0].max_workers == 2
    assert created[0].submissions, "fallback should submit work chunks"

    assert hist["stable_frac"][-1] == pytest.approx(0.5)
    assert hist["delta_Si"][-1] == pytest.approx(1.5)
    assert hist["B"][-1] == pytest.approx(0.15)

    assert get_attr(G.nodes[0], ALIAS_D2VF) == pytest.approx(0.3)
    assert get_attr(G.nodes[1], ALIAS_D2VF) == pytest.approx(0.0)

def test_update_sigma_uses_default_window(monkeypatch, graph_canon):
    G = graph_canon()
    captured: dict[str, int | None] = {}

    monkeypatch.setattr("tnfr.metrics.coherence.DEFAULT_GLYPH_LOAD_SPAN", 7)

    def fake_glyph_load(G, window=None):  # noqa: ANN001 - test double
        captured["window"] = window
        return {
            "_stabilizers": 0.25,
            "_disruptors": 0.75,
            "AL": 0.25,
            "RA": 0.75,
        }

    sigma = {"x": 1.0, "y": 2.0, "mag": 3.0, "angle": 4.0}

    monkeypatch.setattr("tnfr.metrics.coherence.glyph_load", fake_glyph_load)
    monkeypatch.setattr("tnfr.metrics.coherence.sigma_vector", lambda dist: sigma)

    hist: dict[str, list] = {}
    _update_sigma(G, hist)

    assert captured["window"] == 7
    assert hist["glyph_load_stabilizers"] == [0.25]
    assert "glyph_load_estab" not in hist
    assert hist["glyph_load_disr"] == [0.75]
    assert hist["sense_sigma_x"] == [sigma["x"]]
    assert hist["sense_sigma_y"] == [sigma["y"]]
    assert hist["sense_sigma_mag"] == [sigma["mag"]]
    assert hist["sense_sigma_angle"] == [sigma["angle"]]


def test_update_sigma_rejects_legacy_history(monkeypatch, graph_canon):
    G = graph_canon()

    monkeypatch.setattr("tnfr.metrics.coherence.DEFAULT_GLYPH_LOAD_SPAN", 5)

    def fake_glyph_load(G, window=None):  # noqa: ANN001 - test double
        return {
            "_stabilizers": 0.25,
            "_disruptors": 0.75,
        }

    monkeypatch.setattr("tnfr.metrics.coherence.glyph_load", fake_glyph_load)
    monkeypatch.setattr(
        "tnfr.metrics.coherence.sigma_vector", lambda dist: {}
    )

    hist: dict[str, list] = {"glyph_load_estab": [0.5]}

    with pytest.raises(ValueError, match="glyph_load_estab"):
        _update_sigma(G, hist)


def test_metrics_basic_verbosity_skips_collectors(monkeypatch, graph_canon):
    G = graph_canon()
    calls: list[str] = []

    def record(name: str):
        def _rec(*_args, **_kwargs):
            calls.append(name)

        return _rec

    monkeypatch.setattr("tnfr.metrics.core._update_phase_sync", record("phase"))
    monkeypatch.setattr("tnfr.metrics.core._update_sigma", record("sigma"))
    monkeypatch.setattr("tnfr.metrics.core._aggregate_si", record("aggregate"))
    monkeypatch.setattr(
        "tnfr.metrics.core._compute_advanced_metrics",
        record("advanced"),
    )

    G.graph["METRICS"]["verbosity"] = "basic"
    _metrics_step(G)

    assert calls == []
    hist = G.graph["history"]
    assert GLYPH_LOAD_STABILIZERS_KEY not in hist
    assert "phase_sync" not in hist
    assert "Si_mean" not in hist


def test_metrics_detailed_verbosity_runs_collectors(monkeypatch, graph_canon):
    G = graph_canon()
    calls: list[str] = []

    def record(name: str):
        def _rec(*_args, **_kwargs):
            calls.append(name)

        return _rec

    monkeypatch.setattr("tnfr.metrics.core._update_phase_sync", record("phase"))
    monkeypatch.setattr("tnfr.metrics.core._update_sigma", record("sigma"))
    monkeypatch.setattr("tnfr.metrics.core._aggregate_si", record("aggregate"))
    monkeypatch.setattr(
        "tnfr.metrics.core._compute_advanced_metrics",
        record("advanced"),
    )

    G.graph["METRICS"]["verbosity"] = "detailed"
    _metrics_step(G)

    assert calls == ["phase", "sigma", "aggregate"]
    hist = G.graph["history"]
    assert GLYPH_LOAD_STABILIZERS_KEY in hist
    assert "phase_sync" in hist
    assert "Si_mean" in hist

    calls_debug: list[str] = []

    def record_debug(name: str):
        def _rec(*_args, **_kwargs):
            calls_debug.append(name)

        return _rec

    monkeypatch.setattr(
        "tnfr.metrics.core._update_phase_sync",
        record_debug("phase"),
        raising=False,
    )
    monkeypatch.setattr(
        "tnfr.metrics.core._update_sigma",
        record_debug("sigma"),
        raising=False,
    )
    monkeypatch.setattr(
        "tnfr.metrics.core._aggregate_si",
        record_debug("aggregate"),
        raising=False,
    )
    monkeypatch.setattr(
        "tnfr.metrics.core._compute_advanced_metrics",
        record_debug("advanced"),
        raising=False,
    )

    G_debug = graph_canon()
    G_debug.graph["METRICS"]["verbosity"] = "debug"
    _metrics_step(G_debug)

    assert calls_debug == ["phase", "sigma", "aggregate", "advanced"]


def test_register_metrics_callbacks_respects_verbosity(monkeypatch, graph_canon):
    recorded: list[str] = []

    def _recorder(tag: str):
        def _inner(_G):
            recorded.append(tag)

        return _inner

    monkeypatch.setattr(
        "tnfr.metrics.core.register_coherence_callbacks",
        _recorder("coherence"),
    )
    monkeypatch.setattr(
        "tnfr.metrics.core.register_diagnosis_callbacks",
        _recorder("diagnosis"),
    )

    G = graph_canon()
    G.graph["METRICS"]["verbosity"] = "basic"
    register_metrics_callbacks(G)
    assert recorded == []

    recorded_detailed: list[str] = []

    def _recorder_detailed(tag: str):
        def _inner(_G):
            recorded_detailed.append(tag)

        return _inner

    monkeypatch.setattr(
        "tnfr.metrics.core.register_coherence_callbacks",
        _recorder_detailed("coherence"),
        raising=False,
    )
    monkeypatch.setattr(
        "tnfr.metrics.core.register_diagnosis_callbacks",
        _recorder_detailed("diagnosis"),
        raising=False,
    )

    G_detailed = graph_canon()
    G_detailed.graph["METRICS"]["verbosity"] = "detailed"
    register_metrics_callbacks(G_detailed)
    assert recorded_detailed == ["coherence"]

    recorded_high: list[str] = []

    def _recorder_high(tag: str):
        def _inner(_G):
            recorded_high.append(tag)

        return _inner

    monkeypatch.setattr(
        "tnfr.metrics.core.register_coherence_callbacks",
        _recorder_high("coherence"),
        raising=False,
    )
    monkeypatch.setattr(
        "tnfr.metrics.core.register_diagnosis_callbacks",
        _recorder_high("diagnosis"),
        raising=False,
    )

    G_high = graph_canon()
    register_metrics_callbacks(G_high)
    assert recorded_high == ["coherence", "diagnosis"]


def _si_graph(graph_canon):
    G = graph_canon()
    inject_defaults(G)
    hist = {"Si_mean": [], "Si_hi_frac": [], "Si_lo_frac": []}
    for idx, value in enumerate((0.2, 0.5, 0.8)):
        G.add_node(idx)
        set_attr(G.nodes[idx], ALIAS_SI, value)
    # Include a node without Si to ensure NaN entries are ignored.
    G.add_node(3)
    return G, hist


def test_aggregate_si_numpy_vectorized(monkeypatch, graph_canon):
    """NumPy path aggregates Si statistics with vector operations."""

    np_mod = pytest.importorskip("numpy")
    monkeypatch.setattr("tnfr.metrics.coherence.get_numpy", lambda: np_mod)

    G, hist = _si_graph(graph_canon)

    _aggregate_si(G, hist, n_jobs=4)

    assert hist["Si_mean"][0] == pytest.approx(0.5)
    assert hist["Si_hi_frac"][0] == pytest.approx(1 / 3)
    assert hist["Si_lo_frac"][0] == pytest.approx(1 / 3)


def test_aggregate_si_python_sequential(monkeypatch, graph_canon):
    """Sequential fallback keeps the same Si statistics."""

    monkeypatch.setattr("tnfr.metrics.coherence.get_numpy", lambda: None)

    G, hist = _si_graph(graph_canon)

    _aggregate_si(G, hist, n_jobs=None)

    assert hist["Si_mean"][0] == pytest.approx(0.5)
    assert hist["Si_hi_frac"][0] == pytest.approx(1 / 3)
    assert hist["Si_lo_frac"][0] == pytest.approx(1 / 3)


def test_aggregate_si_python_parallel(monkeypatch, graph_canon):
    """Parallel fallback distributes the counting across chunks."""

    monkeypatch.setattr("tnfr.metrics.coherence.get_numpy", lambda: None)

    created: list[Any] = []

    class _FakeFuture:
        def __init__(self, value: Any):
            self._value = value

        def result(self) -> Any:
            return self._value

    class _RecorderExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers
            self.submissions: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

        def __enter__(self):
            created.append(self)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):  # noqa: ANN001 - testing stub
            self.submissions.append((fn, args, kwargs))
            return _FakeFuture(fn(*args, **kwargs))

    monkeypatch.setattr("tnfr.metrics.coherence.ProcessPoolExecutor", _RecorderExecutor)

    G, hist = _si_graph(graph_canon)

    _aggregate_si(G, hist, n_jobs=2)

    assert created and created[0].max_workers == 2
    assert created[0].submissions, "parallel path must submit work chunks"

    assert hist["Si_mean"][0] == pytest.approx(0.5)
    assert hist["Si_hi_frac"][0] == pytest.approx(1 / 3)
    assert hist["Si_lo_frac"][0] == pytest.approx(1 / 3)


def test_compute_advanced_metrics_populates_history(graph_canon):
    """_compute_advanced_metrics records glyph-based metrics."""

    G = graph_canon()
    inject_defaults(G)
    hist: dict[str, Any] = {}
    cfg = G.graph["METRICS"]

    G.add_node(0)
    set_attr(G.nodes[0], ALIAS_EPI, 0.1)
    G.nodes[0]["glyph_history"] = ["OZ"]

    G.add_node(1)
    set_attr(G.nodes[1], ALIAS_EPI, 0.2)
    G.nodes[1]["glyph_history"] = [LATENT_GLYPH]

    _compute_advanced_metrics(G, hist, t=0, dt=1.0, cfg=cfg)

    assert hist["glyphogram"][0]["OZ"] == 1
    assert hist["latency_index"][0]["value"] == pytest.approx(0.5)
    rec = hist["EPI_support"][0]
    assert rec["size"] == 2
    assert rec["epi_norm"] == pytest.approx(0.15)
    morph = hist["morph"][0]
    assert morph["ID"] == pytest.approx(0.5)


def test_pp_val_zero_when_no_remesh(graph_canon):
    """PP metric should be 0.0 when no REMESH events occur."""
    G = graph_canon()
    # Node in SHA state, but without any REMESH events
    G.add_node(0, EPI_kind=LATENT_GLYPH)
    inject_defaults(G)

    _metrics_step(G, ctx=None)

    morph = G.graph["history"]["morph"][0]
    assert morph["PP"] == 0.0


def test_pp_val_handles_missing_sha(graph_canon):
    """PP metric handles absence of SHA counts gracefully."""
    G = graph_canon()
    # Node in REMESH state but without SHA nodes
    G.add_node(0, EPI_kind="REMESH")
    inject_defaults(G)

    _metrics_step(G, ctx=None)

    morph = G.graph["history"]["morph"][0]
    assert morph["PP"] == 0.0


def test_metrics_step_rejects_legacy_history(graph_canon):
    """Legacy glyph load history keys abort metrics setup."""

    G = graph_canon()
    inject_defaults(G)
    G.graph["history"] = {"glyph_load_estab": [0.1]}

    with pytest.raises(ValueError, match="glyph_load_estab"):
        _metrics_step(G, ctx=None)


def test_save_by_node_flag_keeps_metrics_equal(graph_canon):
    """Disabling per-node storage should not alter global metrics."""
    G_true = graph_canon()
    G_true.graph["METRICS"] = dict(G_true.graph["METRICS"])
    G_true.graph["METRICS"]["save_by_node"] = True

    G_false = graph_canon()
    G_false.graph["METRICS"] = dict(G_false.graph["METRICS"])
    G_false.graph["METRICS"]["save_by_node"] = False

    for G in (G_true, G_false):
        G.add_node(0, EPI_kind="OZ")
        G.add_node(1, EPI_kind=LATENT_GLYPH)
        inject_defaults(G)
        for n in G.nodes():
            nd = G.nodes[n]
            nd["glyph_history"] = [nd.get("EPI_kind")]
        G.graph["_t"] = 0
        _metrics_step(G, ctx=None)
        G.nodes[0]["EPI_kind"] = "NAV"
        G.nodes[0].setdefault("glyph_history", []).append("NAV")
        G.graph["_t"] = 1
        _metrics_step(G, ctx=None)

    hist_true = G_true.graph["history"]
    hist_false = G_false.graph["history"]

    assert hist_true["Tg_total"] == hist_false["Tg_total"]
    assert hist_true["glyphogram"] == hist_false["glyphogram"]
    assert hist_true["latency_index"] == hist_false["latency_index"]
    assert hist_true["morph"] == hist_false["morph"]
    assert hist_true["Tg_by_node"] != {}
    assert hist_false.get("Tg_by_node", {}) == {}


def test_build_metrics_summary_reuses_metrics_helpers(monkeypatch):
    G = object()
    calls: dict[str, Any] = {}

    def fake_tg(graph, *, normalize=True):  # noqa: ANN001 - test helper
        calls["tg"] = {"graph": graph, "normalize": normalize}
        return {"AL": 0.75}

    def fake_latency(graph):  # noqa: ANN001 - test helper
        calls["latency"] = graph
        return {"value": [1.0, 2.0, 3.0]}

    def fake_glyphogram(graph):  # noqa: ANN001 - test helper
        calls["glyphogram"] = graph
        return {"t": list(range(12)), "AL": [1, 2, 3]}

    def fake_sigma(graph):  # noqa: ANN001 - test helper
        calls["sigma"] = graph
        return {"mag": 0.5}

    monkeypatch.setattr("tnfr.metrics.reporting.Tg_global", fake_tg)
    monkeypatch.setattr("tnfr.metrics.reporting.latency_series", fake_latency)
    monkeypatch.setattr("tnfr.metrics.reporting.glyphogram_series", fake_glyphogram)
    monkeypatch.setattr("tnfr.metrics.reporting.sigma_rose", fake_sigma)

    summary, has_latency = build_metrics_summary(G, series_limit=10)

    assert has_latency is True
    assert calls["tg"]["graph"] is G
    assert calls["tg"]["normalize"] is True
    assert calls["latency"] is G
    assert calls["glyphogram"] is G
    assert calls["sigma"] is G
    assert summary["Tg_global"] == {"AL": 0.75}
    assert summary["latency_mean"] == pytest.approx(2.0)
    assert summary["rose"] == {"mag": 0.5}
    assert summary["glyphogram"]["t"] == list(range(10))
    assert summary["glyphogram"]["AL"] == [1, 2, 3]


def test_build_metrics_summary_handles_empty_latency(monkeypatch):
    G = object()

    monkeypatch.setattr("tnfr.metrics.reporting.Tg_global", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("tnfr.metrics.reporting.latency_series", lambda *_: {"value": []})
    monkeypatch.setattr("tnfr.metrics.reporting.glyphogram_series", lambda *_: {"t": []})
    monkeypatch.setattr("tnfr.metrics.reporting.sigma_rose", lambda *_: {})

    summary, has_latency = build_metrics_summary(G)

    assert has_latency is False
    assert summary["latency_mean"] == 0.0


def test_build_metrics_summary_accepts_unbounded_limit(monkeypatch):
    G = object()

    monkeypatch.setattr("tnfr.metrics.reporting.Tg_global", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        "tnfr.metrics.reporting.latency_series", lambda *_: {"value": [1.0]}
    )
    monkeypatch.setattr(
        "tnfr.metrics.reporting.glyphogram_series",
        lambda *_: {"t": list(range(12)), "AL": list(range(12))},
    )
    monkeypatch.setattr("tnfr.metrics.reporting.sigma_rose", lambda *_: {})

    summary, has_latency = build_metrics_summary(G, series_limit=0)

    assert has_latency is True
    assert summary["glyphogram"]["t"] == list(range(12))
    assert summary["glyphogram"]["AL"] == list(range(12))


def test_latency_index_uses_max_denominator(graph_canon):
    """Latency index uses max(1, n_total) to avoid zero division."""
    G = graph_canon()
    hist = {}
    _update_latency_index(G, hist, n_total=0, n_latent=2, t=0)
    assert hist["latency_index"][0]["value"] == 2.0


def test_update_epi_support_matches_manual(graph_canon):
    """_update_epi_support computes size and norm as expected."""
    G = graph_canon()
    # valores diversos de EPI
    G.add_node(0, EPI=0.06)
    G.add_node(1, EPI=-0.1)
    G.add_node(2, EPI=0.01)
    G.add_node(3, EPI=0.05)
    inject_defaults(G)
    hist = {}
    threshold = DEFAULT_EPI_SUPPORT_LIMIT
    _update_epi_support(G, hist, t=0, threshold=threshold)

    expected_vals = [
        abs(get_attr(G.nodes[n], ALIAS_EPI, 0.0))
        for n in G.nodes()
        if abs(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) >= threshold
    ]
    expected_size = len(expected_vals)
    expected_norm = (
        sum(expected_vals) / expected_size if expected_size else 0.0
    )

    rec = hist["EPI_support"][0]
    assert rec["size"] == expected_size
    assert rec["epi_norm"] == pytest.approx(expected_norm)


def test_advanced_metrics_vectorized_path(monkeypatch, graph_canon):
    """Vectorised accumulation avoids spawning process pools when NumPy is present."""

    if glyph_numpy is None:
        class FakeBoolArray(list):
            def sum(self):  # noqa: D401 - emulate numpy boolean vector
                return builtins.sum(1 for value in self if value)

        class FakeArray(list):
            def __ge__(self, other):
                return FakeBoolArray(value >= other for value in self)

            def sum(self):  # noqa: D401 - emulate numpy vector sum
                return builtins.sum(self)

            def __getitem__(self, item):
                if isinstance(item, FakeBoolArray):
                    return FakeArray(value for value, flag in zip(self, item) if flag)
                result = super().__getitem__(item)
                if isinstance(item, slice):
                    return FakeArray(result)
                return result

        class FakeNumpy:
            int64 = int

            @staticmethod
            def fromiter(iterable, dtype=float, count=-1):  # noqa: D401 - numpy compatible signature
                del dtype, count
                return FakeArray(list(iterable))

            @staticmethod
            def bincount(array, minlength=0):  # noqa: D401 - numpy compatible signature
                freq = [0] * max(minlength, (max(array) + 1 if array else 0))
                for value in array:
                    freq[value] += 1
                return FakeArray(freq)

        monkeypatch.setattr("tnfr.metrics.glyph_timing.np", FakeNumpy())

    class FailExecutor:  # noqa: D401 - helper for test assertions
        """Placeholder that fails if instantiated."""

        def __init__(self, *args, **kwargs):  # noqa: D401 - signature enforced by ProcessPoolExecutor
            raise AssertionError("ProcessPoolExecutor should not be used with NumPy available")

    monkeypatch.setattr("tnfr.metrics.glyph_timing.ProcessPoolExecutor", FailExecutor)

    G = graph_canon()
    inject_defaults(G)
    hist: dict[str, Any] = {}
    cfg = dict(G.graph["METRICS"])
    cfg["n_jobs"] = 4

    samples = [
        ("AL", 0.06),
        ("RA", 0.10),
        (LATENT_GLYPH, 0.02),
        ("AL", 0.20),
        ("OZ", 0.07),
    ]
    for idx, (glyph, epi) in enumerate(samples):
        G.add_node(idx)
        nd = G.nodes[idx]
        nd["glyph_history"] = [glyph]
        set_attr(nd, ALIAS_EPI, epi)

    _compute_advanced_metrics(G, hist, t=1.0, dt=0.5, cfg=cfg, n_jobs=cfg["n_jobs"])

    glyphogram = hist["glyphogram"][0]
    assert glyphogram["AL"] == 2
    assert glyphogram["RA"] == 1
    assert glyphogram[LATENT_GLYPH] == 1

    latency = hist["latency_index"][0]["value"]
    assert latency == pytest.approx(1 / 5)

    epi_support = hist["EPI_support"][0]
    assert epi_support["size"] == 4
    assert epi_support["epi_norm"] == pytest.approx((0.06 + 0.10 + 0.20 + 0.07) / 4)


def test_advanced_metrics_process_pool_fallback(monkeypatch, graph_canon):
    """When NumPy is unavailable the implementation falls back to multiprocessing."""

    monkeypatch.setattr("tnfr.metrics.glyph_timing.np", None)

    submissions: list[list[tuple[Any, tuple[Any, ...], Any]]] = []

    class DummyFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            self._entries: list[tuple[Any, tuple[Any, ...], Any]] = []
            submissions.append(self._entries)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            result = fn(*args)
            self._entries.append((fn, args, result))
            return DummyFuture(result)

    monkeypatch.setattr("tnfr.metrics.glyph_timing.ProcessPoolExecutor", DummyExecutor)

    G = graph_canon()
    inject_defaults(G)
    hist: dict[str, Any] = {}
    cfg = dict(G.graph["METRICS"])
    cfg["n_jobs"] = 3

    samples = [
        ("AL", 0.06),
        ("RA", 0.10),
        (LATENT_GLYPH, 0.02),
        ("AL", 0.20),
        ("OZ", 0.07),
    ]
    for idx, (glyph, epi) in enumerate(samples):
        G.add_node(idx)
        nd = G.nodes[idx]
        nd["glyph_history"] = [glyph]
        set_attr(nd, ALIAS_EPI, epi)

    _compute_advanced_metrics(G, hist, t=2.0, dt=0.5, cfg=cfg, n_jobs=cfg["n_jobs"])

    assert len(submissions) == 2  # glyph counts + EPI support
    assert all(entries for entries in submissions)

    glyphogram = hist["glyphogram"][0]
    assert glyphogram["AL"] == 2
    assert glyphogram["RA"] == 1
    assert glyphogram[LATENT_GLYPH] == 1

    latency = hist["latency_index"][0]["value"]
    assert latency == pytest.approx(1 / 5)

    epi_support = hist["EPI_support"][0]
    assert epi_support["size"] == 4
    assert epi_support["epi_norm"] == pytest.approx((0.06 + 0.10 + 0.20 + 0.07) / 4)
