"""Pruebas de validators."""

import pytest
from tnfr.scenarios import build_graph
from tnfr.constants import (
    inject_defaults,
    ALIAS_EPI_KIND,
    ALIAS_EPI,
    ALIAS_VF,
)
from tnfr.validators import run_validators
from tnfr.alias import set_attr_str, set_attr
from tnfr.io import read_structured_file
from tnfr.config import load_config

try:  # pragma: no cover - compatibilidad Python
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore


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

    monkeypatch.setattr("tnfr.validators.sigma_vector_from_graph", fake_sigma)
    with pytest.raises(ValueError):
        run_validators(G)


def test_validator_glyph_invalido():
    G = _base_graph()
    n0 = list(G.nodes())[0]
    set_attr_str(G.nodes[n0], ALIAS_EPI_KIND, "INVALID")
    G.nodes[n0]["glyph_history"] = ["INVALID"]
    with pytest.raises(ValueError):
        run_validators(G)


def test_validator_glyph_valido():
    G = _base_graph()
    run_validators(G)


def test_read_structured_file_json(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text('{"x": 1}', encoding="utf-8")
    data = read_structured_file(path)
    assert data == {"x": 1}


@pytest.mark.skipif(tomllib is None, reason="tomllib/tomli no está instalado")
def test_read_structured_file_toml(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text("x = 1", encoding="utf-8")
    data = read_structured_file(path)
    assert data == {"x": 1}


def test_read_structured_file_invalid_extension(tmp_path):
    path = tmp_path / "cfg.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        read_structured_file(path)


def test_load_config_json(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text('{"a": 5}', encoding="utf-8")
    data = load_config(path)
    assert data["a"] == 5
