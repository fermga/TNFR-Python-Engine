import math

import math
import networkx as nx
import pytest

from tnfr.constants import ALIAS_DNFR, ALIAS_SI, ALIAS_THETA, ALIAS_VF
from tnfr.metrics_utils import (
    compute_Si_node,
    get_Si_weights,
    precompute_trigonometry,
)
from tnfr.alias import get_attr, set_attr


def test_get_si_weights_normalization():
    G = nx.Graph()
    G.graph["SI_WEIGHTS"] = {"alpha": 2, "beta": 1, "gamma": 1}
    alpha, beta, gamma = get_Si_weights(G)
    assert (alpha, beta, gamma) == pytest.approx((0.5, 0.25, 0.25))
    assert G.graph["_Si_weights"] == {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }
    assert G.graph["_Si_sensitivity"] == {
        "dSi_dvf_norm": alpha,
        "dSi_ddisp_fase": -beta,
        "dSi_ddnfr_norm": -gamma,
    }


def test_precompute_trigonometry():
    G = nx.Graph()
    G.add_nodes_from([1, 2])
    set_attr(G.nodes[1], ALIAS_THETA, 0.0)
    set_attr(G.nodes[2], ALIAS_THETA, math.pi / 2)
    trig = precompute_trigonometry(G)
    cos_th, sin_th, thetas = trig.cos, trig.sin, trig.theta
    assert cos_th[1] == pytest.approx(1.0)
    assert sin_th[1] == pytest.approx(0.0)
    assert cos_th[2] == pytest.approx(0.0, abs=1e-8)
    assert sin_th[2] == pytest.approx(1.0)
    assert thetas[2] == pytest.approx(math.pi / 2)


def test_compute_Si_node():
    G = nx.Graph()
    G.add_edge(1, 2)
    set_attr(G.nodes[1], ALIAS_VF, 0.5)
    set_attr(G.nodes[1], ALIAS_DNFR, 0.2)
    set_attr(G.nodes[1], ALIAS_THETA, 0.0)
    set_attr(G.nodes[2], ALIAS_THETA, 0.0)
    trig = precompute_trigonometry(G)
    cos_th, sin_th, thetas = trig.cos, trig.sin, trig.theta
    neighbors = {n: list(G.neighbors(n)) for n in G}
    Si = compute_Si_node(
        1,
        G.nodes[1],
        alpha=0.5,
        beta=0.25,
        gamma=0.25,
        vfmax=1.0,
        dnfrmax=1.0,
        cos_th=cos_th,
        sin_th=sin_th,
        thetas=thetas,
        neighbors=neighbors,
        inplace=True,
    )
    assert Si == pytest.approx(0.7)
    assert get_attr(G.nodes[1], ALIAS_SI, 0.0) == pytest.approx(0.7)
