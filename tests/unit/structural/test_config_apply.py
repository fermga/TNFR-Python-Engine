"""Tests for ``load_config`` and ``apply_config``."""

import json
import pytest
from tnfr.config import load_config, apply_config
from collections import UserDict
from tnfr.constants import DEFAULTS, merge_overrides

try:  # pragma: no cover - dependencia opcional
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - skip if not installed
    yaml = None


@pytest.mark.parametrize(
    "suffix,dump",
    [
        (".json", lambda data: json.dumps(data)),
        pytest.param(
            ".yaml",
            lambda data: yaml.safe_dump(data),
            marks=pytest.mark.skipif(
                yaml is None, reason="pyyaml not installed"
            ),
        ),
    ],
)
def test_apply_config_injects_graph_params(tmp_path, suffix, dump, graph_canon):
    cfg = {"RANDOM_SEED": 123, "INIT_THETA_MIN": -1.23}
    path = tmp_path / f"cfg{suffix}"
    path.write_text(dump(cfg), encoding="utf-8")

    loaded = load_config(path)
    assert loaded == cfg

    G = graph_canon()
    G.add_node(0)
    G.graph["RANDOM_SEED"] = 0
    G.graph["INIT_THETA_MIN"] = 0.0

    apply_config(G, path)
    assert G.graph["RANDOM_SEED"] == 123
    assert G.graph["INIT_THETA_MIN"] == -1.23


def test_load_config_accepts_mapping(monkeypatch, tmp_path):
    data = UserDict({"RANDOM_SEED": 1})

    def fake_reader(path):
        return data

    monkeypatch.setattr("tnfr.config.init.read_structured_file", fake_reader)
    path = tmp_path / "dummy.json"
    path.write_text("{}", encoding="utf-8")
    loaded = load_config(path)
    assert loaded == data


def test_load_config_accepts_str(tmp_path):
    cfg = {"RANDOM_SEED": 7}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = load_config(str(path))
    assert loaded == cfg


def test_apply_config_passes_path_object(monkeypatch, tmp_path, graph_canon):
    path = tmp_path / "cfg.json"
    path.write_text("{}", encoding="utf-8")
    received = {}

    def fake_load(p):
        received["path"] = p
        return {}

    monkeypatch.setattr("tnfr.config.init.load_config", fake_load)
    G = graph_canon()
    apply_config(G, path)
    assert received["path"] is path


def test_merge_overrides_does_not_modify_defaults(graph_canon):
    G = graph_canon()
    orig_enabled = DEFAULTS["TRACE"]["enabled"]
    merge_overrides(G, TRACE=DEFAULTS["TRACE"])
    assert G.graph["TRACE"] is not DEFAULTS["TRACE"]
    G.graph["TRACE"]["enabled"] = not orig_enabled
    assert DEFAULTS["TRACE"]["enabled"] == orig_enabled
