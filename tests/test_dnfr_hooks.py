from tnfr.constants import ALIAS_THETA, ALIAS_EPI, ALIAS_VF, ALIAS_DNFR
from tnfr.dynamics import dnfr_phase_only, dnfr_epi_vf_mixed, dnfr_laplacian
from tnfr.alias import get_attr


def test_dnfr_phase_only_computes_gradient(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    G.nodes[0][ALIAS_THETA[0]] = 0.0
    G.nodes[1][ALIAS_THETA[0]] = 1.5707963267948966  # pi/2
    dnfr_phase_only(G)
    assert get_attr(G.nodes[0], ALIAS_DNFR, 0.0) == 0.5
    assert get_attr(G.nodes[1], ALIAS_DNFR, 0.0) == -0.5


def test_dnfr_epi_vf_mixed_sets_average(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    G.nodes[0][ALIAS_EPI[0]] = 1.0
    G.nodes[0][ALIAS_VF[0]] = 0.0
    G.nodes[1][ALIAS_EPI[0]] = 0.0
    G.nodes[1][ALIAS_VF[0]] = 1.0
    dnfr_epi_vf_mixed(G)
    assert get_attr(G.nodes[0], ALIAS_DNFR, 1.0) == 0.0
    assert get_attr(G.nodes[1], ALIAS_DNFR, 1.0) == 0.0


def test_dnfr_laplacian_respects_weights(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    G.graph["DNFR_WEIGHTS"] = {"epi": 1.0, "vf": 0.0}
    G.nodes[0][ALIAS_EPI[0]] = 1.0
    G.nodes[1][ALIAS_EPI[0]] = 0.0
    dnfr_laplacian(G)
    assert get_attr(G.nodes[0], ALIAS_DNFR, 0.0) == -1.0
    assert get_attr(G.nodes[1], ALIAS_DNFR, 0.0) == 1.0
