import pytest

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.metrics_utils import _get_vf_dnfr_max

ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def test_get_vf_dnfr_max_updates_graph_on_none(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])
    set_attr(G.nodes[1], ALIAS_VF, 0.5)
    set_attr(G.nodes[2], ALIAS_VF, -1.5)
    set_attr(G.nodes[1], ALIAS_DNFR, 0.2)
    set_attr(G.nodes[2], ALIAS_DNFR, -0.4)
    G.graph["_vfmax"] = None
    G.graph["_dnfrmax"] = None

    vfmax, dnfrmax = _get_vf_dnfr_max(G)

    assert vfmax == pytest.approx(1.5)
    assert dnfrmax == pytest.approx(0.4)
    assert G.graph["_vfmax"] == pytest.approx(1.5)
    assert G.graph["_dnfrmax"] == pytest.approx(0.4)
