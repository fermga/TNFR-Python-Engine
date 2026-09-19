"""Shared distance and fitting contract for the static coherence read-out.

This fits uncentered products C_i C_j, not connected covariance or a dynamical
correlation theorem. Its pair counts, distance bins and correlation floor are
estimation policies. Neither backend may select a different geometry.

Graph paths use floating accumulation of materialized edge lengths. The
provenance phrase "exact represented distance" means equality of the float
distance keys used for binning, not exact rational path sums or unrounded
physical distances. This estimator does not substitute the exact path model
used by the separate geometry-realization observer.
"""

import math

import networkx as nx

from ..mathematics.unified_numerical import np
from ._edge_semantics import effective_edge_length, structural_path_weight
from ._helpers import finite_real_scalar

DISTANCE_DESCRIPTION = (
    "outgoing shortest paths: length, else weight, else 1; parallel minimum; "
    "positive finite pair distances only"
)
FIT_DESCRIPTION = (
    "uncentered mean(C_i*C_j) by exact represented distance; >=10 pairs, "
    ">=2 pairs/bin, >=3 bins above 1e-9; negative log-linear slope; "
    "no goodness-of-fit acceptance test"
)


def coherence_sources(nodes, precision_mode):
    """Select the same declared source sample independently of backend."""
    nodes = tuple(nodes)
    if len(nodes) < 1000:
        return nodes
    minimum = 30 if precision_mode == "research" else 20
    count = max(minimum, len(nodes) // 20)
    return nodes[:: max(1, len(nodes) // count)]


def coherence_sample_description(graph, nodes, sources):
    pairs = "ordered outgoing" if graph.is_directed() else "unordered"
    if len(sources) == len(nodes):
        return f"all {pairs} node pairs"
    return (
        f"{pairs} pairs incident from {len(sources)} deterministic evenly-spaced "
        "source nodes in graph insertion order; source IDs included in fit cache key"
    )


def _validate_graph(graph, nodes):
    if len(set(nodes)) != len(nodes) or set(nodes) != set(graph):
        raise ValueError(
            "coherence node order must contain every graph node exactly once"
        )
    # Validate every component, including unreachable and unsampled edges.
    for _, _, attributes in graph.edges(data=True):
        effective_edge_length(attributes)


def _distance_array(values, size, directed):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.shape != (size, size):
        raise ValueError(
            "distance_matrix must be a real square array matching node order"
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        distances = np.asarray(raw, dtype=float)
    if np.any(np.isfinite(raw) & ~np.isfinite(distances)) or np.any(
        (raw != 0) & (distances == 0)
    ):
        raise ValueError("distance_matrix contains an unrepresentable finite distance")
    if np.any(np.isnan(distances)) or np.any(distances < 0):
        raise ValueError(
            "distance_matrix permits nonnegative distances and positive infinity only"
        )
    if np.any(np.diag(distances) != 0):
        raise ValueError("distance_matrix must have a zero diagonal")
    if not directed and not np.array_equal(distances, distances.T):
        raise ValueError("undirected distance_matrix must be symmetric")
    return distances


def _graph_distance_rows(graph, sources):
    weight = structural_path_weight(graph)
    for source in sources:
        row = nx.single_source_dijkstra_path_length(graph, source, weight=weight)
        if any(not math.isfinite(distance) for distance in row.values()):
            raise ValueError(
                "reachable path distance exceeds the finite represented range"
            )
        yield source, row


def _fit_products(pairs):
    bins = {}
    count = 0
    lowest = math.inf
    highest = -math.inf
    for distance, product in pairs:
        if not math.isfinite(distance) or distance <= 0:
            continue
        bins.setdefault(float(distance), []).append(float(product))
        lowest = min(lowest, float(product))
        highest = max(highest, float(product))
        count += 1
    if count < 10 or lowest == highest:
        return float("nan")
    fitted = []
    for distance, values in sorted(bins.items()):
        if len(values) >= 2:
            mean = math.fsum(values) / len(values)
            if mean > 1e-9:
                fitted.append((distance, mean))
    if len(fitted) < 3:
        return float("nan")
    distances, means = zip(*fitted)
    # A constant product field has no negative decay slope. Do not let a
    # least-squares rounding residual turn it into a huge finite decay length.
    if min(means) == max(means):
        return float("nan")
    scale = max(distances)
    x = np.asarray(distances, dtype=float) / scale
    y = np.log(np.asarray(means, dtype=float))
    try:
        slope, _ = np.polyfit(x, y, 1)
    except np.linalg.LinAlgError:
        return float("nan")
    if not math.isfinite(slope) or slope >= 0:
        return float("nan")
    value = float(scale / -slope)
    return value if math.isfinite(value) and value > 0 else float("nan")


def fit_coherence_length(
    graph,
    nodes,
    delta_nfr,
    *,
    sources,
    dtype=np.float64,
    materialize=False,
    distance_matrix=None,
):
    """Fit the declared static product profile with one common pair policy.

    Nonnegative effective edge lengths define outgoing distances; zero-distance
    off-diagonal pairs (a pseudometric) and unreachable pairs are omitted.
    Invalid edge metrics raise instead of invoking the spectral fallback.
    Supplied matrices must satisfy the numeric distance domain and symmetry on
    undirected graphs; they are caller-declared, not authenticated shortest paths.
    Both graph-derived and supplied distances are binned by equality of their
    materialized float values. Rational path accumulation is a different model.
    """
    from ..metrics.common import structural_coherence

    nodes, sources = tuple(nodes), tuple(sources)
    _validate_graph(graph, nodes)
    if len(set(sources)) != len(sources) or not set(sources).issubset(nodes):
        raise ValueError("coherence sources must be distinct graph nodes")
    index = {node: i for i, node in enumerate(nodes)}
    pressure = np.asarray(
        [
            finite_real_scalar(delta_nfr.get(node, 0.0), "coherence pressure")
            for node in nodes
        ],
        dtype=dtype,
    )
    coherence = structural_coherence(pressure)
    matrix = None
    row_index = index
    if distance_matrix is not None:
        matrix = _distance_array(distance_matrix, len(nodes), graph.is_directed())
    elif materialize:
        row_index = {source: i for i, source in enumerate(sources)}
        matrix = np.full((len(sources), len(nodes)), np.inf)
        for source, row in _graph_distance_rows(graph, sources):
            for target, distance in row.items():
                matrix[row_index[source], index[target]] = distance

    rows = (
        (
            (
                source,
                {
                    target: matrix[row_index[source], j]
                    for j, target in enumerate(nodes)
                },
            )
            for source in sources
        )
        if matrix is not None
        else _graph_distance_rows(graph, sources)
    )
    source_set = set(sources)

    def pairs():
        for source, row in rows:
            i = index[source]
            for target, distance in row.items():
                j = index[target]
                if i == j:
                    continue
                if not graph.is_directed() and target in source_set and j < i:
                    continue  # Count each sampled unordered pair once.
                yield distance, coherence[i] * coherence[j]

    return _fit_products(pairs())
