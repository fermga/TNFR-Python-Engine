import pytest
from tnfr.scenarios import build_graph
from tnfr.constants import (
    inject_defaults,
    DEFAULTS,
    ALIAS_EPI_KIND,
    ALIAS_EPI,
    ALIAS_VF,
)
from tnfr.validators import run_validators
from tnfr.helpers import _set_attr_str, _set_attr


def _base_graph():
    G = build_graph(n=4, topology="ring", seed=1)
    inject_defaults(G, DEFAULTS)
    return G


def test_validator_epi_range():
    G = _base_graph()
    n0 = list(G.nodes())[0]
    _set_attr(G.nodes[n0], ALIAS_EPI, 2.0)
    with pytest.raises(ValueError):
        run_validators(G)


def test_validator_vf_range():
    G = _base_graph()
    n0 = list(G.nodes())[0]
    _set_attr(G.nodes[n0], ALIAS_VF, 2.0)
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
    _set_attr_str(G.nodes[n0], ALIAS_EPI_KIND, "INVALID")
    with pytest.raises(ValueError):
        run_validators(G)
