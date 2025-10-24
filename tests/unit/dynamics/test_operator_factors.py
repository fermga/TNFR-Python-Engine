"""Tests for glyph factor retrieval helper."""

from types import SimpleNamespace

import pytest

import tnfr.operators as operators
from tnfr.node import NodeNX
from tnfr.types import Glyph


def test_get_factor_returns_float():
    gf = {"a": "1.23"}
    assert operators.get_factor(gf, "a", 0.0) == pytest.approx(1.23)
    assert operators.get_factor(gf, "missing", 4.5) == pytest.approx(4.5)


def test_op_al_uses_factor():
    node = SimpleNamespace(EPI=1.0, graph={})
    gf = {"AL_boost": "0.2"}
    operators._op_AL(node, gf)
    assert node.EPI == pytest.approx(1.2)


def test_op_en_uses_mix():
    neigh = SimpleNamespace(EPI=10.0, epi_kind="n1")
    node = SimpleNamespace(
        EPI=1.0,
        epi_kind="self",
        graph={},
        neighbors=lambda: [neigh],
    )
    gf = {"EN_mix": "0.5"}
    operators._op_EN(node, gf)
    assert node.EPI == pytest.approx(5.5)
    assert node.epi_kind == "n1"


def test_op_il_uses_factor():
    node = SimpleNamespace(dnfr=2.0, graph={})
    gf = {"IL_dnfr_factor": "0.5"}
    operators._op_IL(node, gf)
    assert node.dnfr == pytest.approx(1.0)


def test_op_oz_uses_factor():
    node = SimpleNamespace(dnfr=2.0, graph={})
    gf = {"OZ_dnfr_factor": "2.0"}
    operators._op_OZ(node, gf)
    assert node.dnfr == pytest.approx(4.0)


def test_op_oz_applies_noise_when_enabled(monkeypatch):
    node = SimpleNamespace(dnfr=1.5, graph={"OZ_NOISE_MODE": True, "OZ_SIGMA": 0.25})
    gf = {"OZ_dnfr_factor": "2.0"}
    sentinel = 0.37

    monkeypatch.setattr(operators, "random_jitter", lambda *args, **kwargs: sentinel)

    operators._op_OZ(node, gf)

    assert node.dnfr == pytest.approx(1.5 + sentinel)


def test_op_oz_ignores_noise_with_non_positive_sigma():
    node = SimpleNamespace(dnfr=1.5, graph={"OZ_NOISE_MODE": True, "OZ_SIGMA": 0.0})
    gf = {"OZ_dnfr_factor": "3.0"}

    operators._op_OZ(node, gf)

    assert node.dnfr == pytest.approx(1.5)


def test_op_um_uses_theta_push(graph_canon):
    G = graph_canon()
    G.add_node(0, **{"theta": 0.0, "EPI": 0.0, "Si": 0.0})
    G.add_node(1, **{"theta": 1.0, "EPI": 0.0, "Si": 0.0})
    G.add_edge(0, 1)
    node = NodeNX(G, 0)
    gf = {"UM_theta_push": "0.5"}
    operators._op_UM(node, gf)
    assert node.theta == pytest.approx(0.5)


def test_op_ra_uses_diff():
    neigh = SimpleNamespace(EPI=10.0, epi_kind="n1")
    node = SimpleNamespace(
        EPI=0.0,
        epi_kind="",
        graph={},
        neighbors=lambda: [neigh],
    )
    gf = {"RA_epi_diff": "0.5"}
    operators._op_RA(node, gf)
    assert node.EPI == pytest.approx(5.0)
    assert node.epi_kind == "n1"


def test_op_sha_uses_factor():
    node = SimpleNamespace(vf=2.0, graph={})
    gf = {"SHA_vf_factor": "0.5"}
    operators._op_SHA(node, gf)
    assert node.vf == pytest.approx(1.0)


def test_scale_ops_use_factor_val():
    node = SimpleNamespace(vf=3.0, graph={})
    gf = {"VAL_scale": "2.0"}
    operators.GLYPH_OPERATIONS[Glyph.VAL](node, gf)
    assert node.vf == pytest.approx(6.0)


def test_scale_ops_use_factor_nul():
    node = SimpleNamespace(vf=3.0, graph={})
    gf = {"NUL_scale": "0.5"}
    operators.GLYPH_OPERATIONS[Glyph.NUL](node, gf)
    assert node.vf == pytest.approx(1.5)


def test_op_thol_uses_accel():
    node = SimpleNamespace(dnfr=1.0, d2EPI=2.0, graph={})
    gf = {"THOL_accel": "0.5"}
    operators._op_THOL(node, gf)
    assert node.dnfr == pytest.approx(2.0)


def test_op_zhir_uses_shift():
    node = SimpleNamespace(theta=0.0, graph={})
    gf = {"ZHIR_theta_shift": "1.0"}
    operators._op_ZHIR(node, gf)
    assert node.theta == pytest.approx(1.0)


def test_op_nav_uses_eta_and_jitter():
    node = SimpleNamespace(dnfr=1.0, vf=2.0, graph={"NAV_RANDOM": False})
    gf = {"NAV_eta": "0.25", "NAV_jitter": "0.05"}
    operators._op_NAV(node, gf)
    expected_base = (1 - 0.25) * 1.0 + 0.25 * 2.0
    assert node.dnfr == pytest.approx(expected_base + 0.05)
