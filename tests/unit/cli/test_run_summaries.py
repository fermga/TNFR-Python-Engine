from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from typing import Any

import networkx as nx
import pytest

from tnfr.cli import execution


class RecordingLogger:
    """Record log messages while forwarding them to a base logger."""

    def __init__(self, base_logger: logging.Logger) -> None:
        self._base = base_logger
        self.records: list[str] = []

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        formatted = msg % args if args else msg
        self.records.append(formatted)
        self._base.info(msg, *args, **kwargs)


def make_graph(*, coherence_enabled: bool = True, diagnosis_enabled: bool = True) -> nx.Graph:
    graph = nx.Graph()
    graph.graph["HISTORY_MAXLEN"] = 0
    graph.graph["COHERENCE"] = {
        "enabled": coherence_enabled,
        "stats_history_key": "W_stats",
    }
    graph.graph["DIAGNOSIS"] = {
        "enabled": diagnosis_enabled,
        "history_key": "nodal_diag",
    }
    graph.graph["history"] = {
        "W_stats": [{"Tg": 1.0}, {"Tg": 42.0}],
        "nodal_diag": [
            OrderedDict(
                (
                    ("n1", 11),
                    ("n2", 12),
                    ("n3", 13),
                    ("n4", 14),
                )
            )
        ],
    }
    return graph


@pytest.mark.parametrize(
    ("coherence_enabled", "diagnosis_enabled", "expected_messages"),
    [
        (
            True,
            True,
            [
                "[COHERENCE] last step: {'Tg': 42.0}",
                "[DIAGNOSIS] sample: [11, 12, 13]",
            ],
        ),
        (False, False, []),
    ],
)
def test_log_run_summaries_history_logging_toggled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    coherence_enabled: bool,
    diagnosis_enabled: bool,
    expected_messages: list[str],
) -> None:
    graph = make_graph(
        coherence_enabled=coherence_enabled, diagnosis_enabled=diagnosis_enabled
    )
    args = argparse.Namespace(summary=False)

    summary_limits: list[int] = []
    glyph_calls: list[int] = []

    def fake_build_metrics_summary(
        G: nx.Graph, *, series_limit: int | None
    ) -> tuple[dict[str, Any], bool]:
        summary_limits.append(series_limit if series_limit is not None else -999)
        return ({"Tg_global": 1.0, "latency_mean": 2.0}, True)

    def fake_glyph_top(G: nx.Graph, *, k: int) -> list[str]:
        glyph_calls.append(k)
        return ["glyph"]

    base_logger = logging.getLogger(
        f"tnfr.cli.execution.test.history.{coherence_enabled}.{diagnosis_enabled}"
    )
    recorder = RecordingLogger(base_logger)

    monkeypatch.setattr(execution, "build_metrics_summary", fake_build_metrics_summary)
    monkeypatch.setattr(execution, "glyph_top", fake_glyph_top)
    monkeypatch.setattr(execution, "logger", recorder)

    caplog.clear()
    caplog.set_level(logging.INFO, logger=base_logger.name)

    execution._log_run_summaries(graph, args)

    assert summary_limits == []
    assert glyph_calls == []
    assert recorder.records == expected_messages
    assert caplog.messages == expected_messages


@pytest.mark.parametrize("summary_limit", [5, 0, -1])
def test_log_run_summaries_summary_limit_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    summary_limit: int,
) -> None:
    graph = make_graph()
    args = argparse.Namespace(summary=True, summary_limit=summary_limit)

    summary_limits: list[int] = []
    glyph_calls: list[tuple[nx.Graph, int]] = []

    def fake_build_metrics_summary(
        G: nx.Graph, *, series_limit: int | None
    ) -> tuple[dict[str, Any], bool]:
        summary_limits.append(series_limit if series_limit is not None else -999)
        return ({"Tg_global": "tg", "latency_mean": 2.0}, True)

    def fake_glyph_top(G: nx.Graph, *, k: int) -> list[str]:
        glyph_calls.append((G, k))
        return ["g0"]

    base_logger = logging.getLogger(f"tnfr.cli.execution.test.summary.{summary_limit}")
    recorder = RecordingLogger(base_logger)

    monkeypatch.setattr(execution, "build_metrics_summary", fake_build_metrics_summary)
    monkeypatch.setattr(execution, "glyph_top", fake_glyph_top)
    monkeypatch.setattr(execution, "logger", recorder)

    caplog.clear()
    caplog.set_level(logging.INFO, logger=base_logger.name)

    execution._log_run_summaries(graph, args)

    assert summary_limits == [summary_limit]
    assert glyph_calls == [(graph, 5)]
    assert recorder.records == [
        "[COHERENCE] last step: {'Tg': 42.0}",
        "[DIAGNOSIS] sample: [11, 12, 13]",
        "Global Tg: tg",
        "Top operators by Tg: ['g0']",
        "Average latency: 2.0",
    ]
    assert caplog.messages == recorder.records
