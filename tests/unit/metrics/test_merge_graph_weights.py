import pytest

from tnfr.constants import DEFAULTS
from tnfr.metrics.common import merge_and_normalize_weights, merge_graph_weights


@pytest.mark.parametrize(
    "key",
    [
        ("SELECTOR_WEIGHTS", ("w_si", "w_dnfr", "w_accel")),
        ("SI_WEIGHTS", ("alpha", "beta", "gamma")),
    ],
)
def test_merge_graph_weights_fallback_for_none(graph_canon, key):
    key_name, fields = key
    G = graph_canon()
    G.graph[key_name] = None

    merged = merge_graph_weights(G, key_name)
    normalized = merge_and_normalize_weights(G, key_name, fields)

    assert merged == DEFAULTS[key_name]
    assert normalized == pytest.approx(DEFAULTS[key_name])


@pytest.mark.parametrize(
    "override,key",
    [
        (1.0, ("SELECTOR_WEIGHTS", ("w_si", "w_dnfr", "w_accel"))),
        (0, ("SI_WEIGHTS", ("alpha", "beta", "gamma"))),
    ],
)
def test_merge_graph_weights_fallback_for_scalars(graph_canon, override, key):
    key_name, fields = key
    G = graph_canon()
    G.graph[key_name] = override

    merged = merge_graph_weights(G, key_name)
    normalized = merge_and_normalize_weights(G, key_name, fields)

    assert merged == DEFAULTS[key_name]
    assert normalized == pytest.approx(DEFAULTS[key_name])
