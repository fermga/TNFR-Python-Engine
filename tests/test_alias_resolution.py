import pytest

from tnfr.constants import get_param


def test_legacy_remesh_tau_alias(graph_canon):
    G = graph_canon()
    # Remove modern keys to force resolution via legacy alias
    del G.graph["REMESH_TAU_GLOBAL"]
    del G.graph["REMESH_TAU_LOCAL"]
    G.graph["REMESH_TAU"] = 7
    with pytest.warns(DeprecationWarning):
        assert get_param(G, "REMESH_TAU_GLOBAL") == 7
    with pytest.warns(DeprecationWarning):
        assert get_param(G, "REMESH_TAU_LOCAL") == 7
