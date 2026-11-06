"""SDK and file readers share retained-sample admission without changing state."""

import math
from collections import deque
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.metrics.export import export_metrics
from tnfr.sdk import Network
from tnfr.utils.io import json_loads


@pytest.mark.parametrize(
    "container", (list, tuple, deque, lambda x: np.array(x, dtype=object))
)
def test_history_and_export_retain_missing_samples_and_subnormals(tmp_path, container):
    smallest = float.fromhex("0x0.0000000000001p-1022")
    samples = container((0, None, Fraction(1, 4), Fraction.from_float(smallest)))
    graph = nx.Graph()
    graph.graph["history"] = {"C_steps": samples, "sense_sigma_x": samples}
    network = Network(graph)

    observed = network.history()
    export_metrics(graph, str(tmp_path / "history"), "json")
    exported = json_loads((tmp_path / "history.json").read_text())
    assert observed["C_steps"] == [0.0, None, 0.25, smallest]
    assert observed["C_steps"] == exported["sigma"]["sigma_x"]
    assert observed["kuramoto_R"] == []
    assert graph.graph["history"]["C_steps"] is samples
    observed["C_steps"].append(99.0)
    assert len(samples) == 4


@pytest.mark.parametrize("channel", ("kuramoto_R", "C_steps", "phase_sync", "Si_mean"))
@pytest.mark.parametrize(
    "invalid",
    (True, np.bool_(False), math.nan, math.inf, "0.5", 1j, Fraction(1, 10**400)),
)
def test_each_consumed_series_rejects_invalid_samples_before_float_conversion(
    channel, invalid
):
    graph = nx.Graph()
    samples = [0.0, invalid]
    graph.graph["history"] = {channel: samples}
    with pytest.raises((TypeError, ValueError), match=channel):
        Network(graph).history()
    assert graph.graph["history"][channel] is samples
    assert samples[1] is invalid


@pytest.mark.parametrize(
    "bad", (None, "12", {1, 2}, {0: 1}, np.array(1), np.ones((1, 2)))
)
def test_unordered_or_multidimensional_series_cannot_become_a_temporal_stream(
    tmp_path, bad
):
    graph = nx.Graph()
    graph.graph["history"] = {"C_steps": bad, "sense_sigma_x": bad}
    with pytest.raises(TypeError):
        Network(graph).history()
    with pytest.raises(TypeError):
        export_metrics(graph, str(tmp_path / "bad"), "json")
    assert list(tmp_path.iterdir()) == []


def test_one_shot_history_rejects_without_advancing_either_reader(tmp_path):
    graph = nx.Graph()
    samples = iter((0.25, 0.5))
    graph.graph["history"] = {"C_steps": samples, "sense_sigma_x": samples}
    with pytest.raises(TypeError):
        Network(graph).history()
    with pytest.raises(TypeError):
        export_metrics(graph, str(tmp_path / "bad"), "json")
    assert next(samples) == 0.25


def test_missing_history_is_empty_but_malformed_history_rejects():
    graph = nx.Graph()
    network = Network(graph)
    assert all(series == [] for series in network.history().values())
    assert "history" not in graph.graph
    graph.graph["history"] = None
    with pytest.raises(TypeError, match="history"):
        network.history()
