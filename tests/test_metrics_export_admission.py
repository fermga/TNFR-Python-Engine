"""Stored metric exports preserve absence and pre-admit both output formats."""

from __future__ import annotations

import csv
import math
from collections import deque
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.metrics.export import export_metrics
from tnfr.utils.io import json_loads


def _graph():
    graph = nx.Graph()
    graph.graph["history"] = {
        "glyphogram": [{"t": 17.0, "AL": 0}, {"AL": 2}],
        "sense_sigma_t": deque([2.5]),
        "sense_sigma_x": [0.0, -0.25],
        "sense_sigma_y": [None, 0.5],
        "sense_sigma_mag": [],
        "sense_sigma_angle": [None],
        "morph": [{"t": 2.5, "ID": 0.0}],
        "EPI_support": [{"t": 2.5, "size": 0, "epi_norm": None}],
    }
    return graph


def test_json_and_csv_share_missing_samples_without_inventing_clocks(tmp_path):
    graph = _graph()
    before = deepcopy(graph.graph)
    base = str(tmp_path / "observed")
    export_metrics(graph, base, "json")
    export_metrics(graph, base, "csv")
    result = json_loads((tmp_path / "observed.json").read_text())
    assert result["sigma"] == {
        "t": [2.5, None],
        "sigma_x": [0.0, -0.25],
        "sigma_y": [None, 0.5],
        "mag": [None, None],
        "angle": [None, None],
    }
    assert result["glyphogram"]["t"] == [17.0, None]
    assert result["glyphogram"]["AL"] == [0.0, 2.0]
    assert result["glyphogram"]["IL"] == [None, None]
    with (tmp_path / "observed_sigma.csv").open(newline="") as stream:
        assert list(csv.reader(stream)) == [
            ["t", "x", "y", "mag", "angle"],
            ["2.5", "0.0", "", "", ""],
            ["", "-0.25", "0.5", "", ""],
        ]
    with (tmp_path / "observed_glyphogram.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
        assert rows[1]["t"] == ""
        assert rows[0]["IL"] == ""
    assert graph.graph == before


@pytest.mark.parametrize("fmt", ("csv", "json"))
@pytest.mark.parametrize(
    "invalid",
    (
        math.nan,
        math.inf,
        -math.inf,
        True,
        np.bool_(False),
        "1.0",
        1j,
        Fraction(1, 10**400),
    ),
)
def test_invalid_late_metric_cannot_replace_any_output(tmp_path, fmt, invalid):
    graph = _graph()
    graph.graph["history"]["EPI_support"][0]["epi_norm"] = invalid
    base = str(tmp_path / "retained")
    names = (
        "retained.json",
        "retained_glyphogram.csv",
        "retained_sigma.csv",
        "retained_morph.csv",
        "retained_epi_support.csv",
    )
    for name in names:
        (tmp_path / name).write_text("existing evidence", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        export_metrics(graph, base, fmt)
    assert all(
        (tmp_path / name).read_text(encoding="utf-8") == "existing evidence"
        for name in names
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(names)


@pytest.mark.parametrize("series", ("sense_sigma_t", "sense_sigma_x"))
@pytest.mark.parametrize("invalid", (True, "2.0", Fraction(1, 10**400)))
def test_clock_and_sigma_sources_are_admitted_before_materialization(
    tmp_path, series, invalid
):
    graph = _graph()
    graph.graph["history"][series] = [invalid]
    with pytest.raises((TypeError, ValueError)):
        export_metrics(graph, str(tmp_path / "bad"))
    assert list(tmp_path.iterdir()) == []


def test_empty_history_export_does_not_install_runtime_history(tmp_path):
    graph = nx.Graph()
    export_metrics(graph, str(tmp_path / "empty"), "json")
    assert graph.graph == {}
    result = json_loads((tmp_path / "empty.json").read_text())
    assert result["glyphogram"] == {"t": []}
    assert all(values == [] for values in result["sigma"].values())


def test_present_invalid_history_and_series_are_not_treated_as_empty(tmp_path):
    graph = nx.Graph()
    for history in (None, {"sense_sigma_x": None}, {"sense_sigma_x": {1, 2}}):
        graph.graph["history"] = history
        with pytest.raises(TypeError):
            export_metrics(graph, str(tmp_path / "bad"), "json")
    assert list(tmp_path.iterdir()) == []


def test_nonzero_representable_subnormal_is_not_erased(tmp_path):
    graph = nx.Graph()
    smallest = float.fromhex("0x0.0000000000001p-1022")
    graph.graph["history"] = {"sense_sigma_x": [Fraction.from_float(-smallest)]}
    export_metrics(graph, str(tmp_path / "tiny"), "json")
    result = json_loads((tmp_path / "tiny.json").read_text())
    assert result["sigma"]["sigma_x"] == [-smallest]
    assert result["sigma"]["t"] == [None]
