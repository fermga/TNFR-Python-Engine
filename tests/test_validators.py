import pytest
from tnfr.scenarios import build_graph
from tnfr.constants import (
    inject_defaults,
    ALIAS_EPI_KIND,
    ALIAS_EPI,
    ALIAS_VF,
)
from tnfr.validators import run_validators
from tnfr.helpers import set_attr_str, set_attr, read_structured_file
from tnfr.config import load_config


def _base_graph():
    G = build_graph(n=4, topology="ring", seed=1)
    inject_defaults(G)
    return G


def test_validator_epi_range():
    G = _base_graph()
    n0 = list(G.nodes())[0]
    set_attr(G.nodes[n0], ALIAS_EPI, 2.0)
    with pytest.raises(ValueError):
        run_validators(G)


def test_validator_vf_range():
    G = _base_graph()
    n0 = list(G.nodes())[0]
    set_attr(G.nodes[n0], ALIAS_VF, 2.0)
    with pytest.raises(ValueError):
        run_validators(G)


def test_validator_sigma_norm(monkeypatch):
    G = _base_graph()

    def fake_sigma(G):
        return {"mag": 1.5}

    monkeypatch.setattr("tnfr.validators.sigma_vector_global", fake_sigma)
    with pytest.raises(ValueError):
        run_validators(G)


def test_validator_glifo_invalido():
    G = _base_graph()
    n0 = list(G.nodes())[0]
    set_attr_str(G.nodes[n0], ALIAS_EPI_KIND, "INVALID")
    with pytest.raises(ValueError):
        run_validators(G)


def test_read_structured_file_json(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("{\"x\": 1}", encoding="utf-8")
    data = read_structured_file(path)
    assert data == {"x": 1}


def test_read_structured_file_invalid_extension(tmp_path):
    path = tmp_path / "cfg.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        read_structured_file(path)


def test_load_config_json(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("{\"a\": 5}", encoding="utf-8")
    data = load_config(path)
    assert data["a"] == 5
