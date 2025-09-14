import networkx as nx
from tnfr.constants import inject_defaults
import tnfr.json_utils as json_utils


def build_graph(n, graph_canon=None):
    G = graph_canon() if graph_canon is not None else nx.Graph()
    inject_defaults(G)
    for i in range(n):
        G.add_node(i, **{"θ": 0.0, "EPI": 0.0})
    return G


def clear_orjson_cache() -> None:
    """Clear cached :mod:`orjson` module and warning state."""
    json_utils._warn_orjson_params_once.clear()
    cache_clear = getattr(json_utils.cached_import, "cache_clear", None)
    if cache_clear:
        cache_clear()
