"""Pruebas de load_config y apply_config."""
import json
import networkx as nx
import pytest
from tnfr.config import load_config, apply_config

try:  # pragma: no cover - dependencia opcional
    import yaml  # type: ignore
except Exception:  # pragma: no cover - skip if not installed
    yaml = None


@pytest.mark.parametrize(
    "suffix,dump",
    [
        (".json", lambda data: json.dumps(data)),
        pytest.param(
            ".yaml",
            lambda data: yaml.safe_dump(data),
            marks=pytest.mark.skipif(yaml is None, reason="pyyaml no está instalado"),
        ),
    ],
)
def test_apply_config_injects_graph_params(tmp_path, suffix, dump):
    cfg = {"RANDOM_SEED": 123, "INIT_THETA_MIN": -1.23}
    path = tmp_path / f"cfg{suffix}"
    path.write_text(dump(cfg), encoding="utf-8")

    loaded = load_config(path)
    assert loaded == cfg

    G = nx.Graph()
    G.add_node(0)
    G.graph["RANDOM_SEED"] = 0
    G.graph["INIT_THETA_MIN"] = 0.0

    apply_config(G, path)
    assert G.graph["RANDOM_SEED"] == 123
    assert G.graph["INIT_THETA_MIN"] == -1.23
