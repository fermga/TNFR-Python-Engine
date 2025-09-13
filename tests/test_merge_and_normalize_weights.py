import pytest

from tnfr.metrics_utils import merge_and_normalize_weights


def test_merge_and_normalize_weights(graph_canon):
    G = graph_canon()
    G.graph["DNFR_WEIGHTS"] = {"phase": 2, "epi": 1, "vf": 1, "topo": 0}
    weights = merge_and_normalize_weights(
        G, "DNFR_WEIGHTS", ("phase", "epi", "vf", "topo"), default=0.0
    )
    assert weights == pytest.approx(
        {"phase": 0.5, "epi": 0.25, "vf": 0.25, "topo": 0.0}
    )
