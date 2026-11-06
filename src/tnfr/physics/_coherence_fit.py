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
from dataclasses import dataclass
from typing import Any

from ..mathematics.unified_numerical import np
from ._edge_semantics import structural_distance_rows, validate_structural_graph
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


@dataclass(frozen=True)
class CoherenceFitDiagnostics:
    """Descriptive log-fit evidence; no goodness-of-fit threshold is imposed.

    ``slope`` uses the declared path-distance unit. ``normalized_slope`` and
    the singular values refer to the actual fit on distance/distance_scale.
    Rank and residuals describe that numerical regression, not independent
    statistical samples or confidence intervals. Missing quantities are None.
    """

    value: float | None
    available: bool
    status: str
    selected_distances: tuple[float, ...]
    slope: float | None = None
    intercept: float | None = None
    residual_sum_squares: float | None = None
    residual_rms: float | None = None
    r_squared: float | None = None
    design_rank: int | None = None
    singular_values: tuple[float, ...] = ()
    rcond: float | None = None
    distance_scale: float | None = None
    normalized_slope: float | None = None
    log_residuals: tuple[float, ...] = ()


@dataclass(frozen=True)
class CoherenceDistanceBin:
    """One complete sampled distance bin, including bins excluded from the fit.

    Endpoint moments pool both ends. Source/target roles are distinct only on
    directed support; undirected role moments symmetrize each unordered pair.
    ``centered_product`` uses the mean over every graph node. In contrast,
    ``endpoint_covariance`` centers by each bin's own endpoint-role means.
    Neither is automatically a stationary random-field covariance.
    ``fit_included`` records per-bin eligibility (two pairs and a product
    mean above the floor), including when a separate whole-fit condition
    fails. The fit record states whether a length was actually obtained.
    """

    distance: float
    pair_count: int
    product_mean: float
    endpoint_mean: float
    endpoint_second_moment: float
    source_mean: float
    target_mean: float
    source_second_moment: float
    target_second_moment: float
    centered_product: float
    endpoint_covariance: float
    semivariogram: float
    fit_included: bool
    fit_exclusion_reason: str | None


@dataclass(frozen=True)
class CoherenceProfile:
    """Immutable static observations on the scalar estimator's pair sample.

    This is a detached report, with no spectral substitution or dynamics.
    Products/moments use represented floating arithmetic; a represented zero
    need not prove an exact-real zero. The fit retains its historical policy.
    """

    nodes: tuple[Any, ...]
    sources: tuple[Any, ...]
    coherence: tuple[float, ...]
    global_mean: float | None
    global_variance: float | None
    candidate_pair_count: int
    positive_pair_count: int
    zero_distance_pair_count: int
    unreachable_pair_count: int
    bins: tuple[CoherenceDistanceBin, ...]
    fit: CoherenceFitDiagnostics
    distance_weighting: str
    sample_selection: str
    endpoint_convention: str
    precision_mode: str
    arithmetic_dtype: str
    distance_origin: str
    fit_policy: str


def coherence_sources(nodes, precision_mode):
    """Select the declared source sample independently of backend and mode.

    The mode argument is retained for caller compatibility. Once ``n>=1000``,
    ``n//20>=50`` already exceeds both historical mode-dependent minima.
    """
    nodes = tuple(nodes)
    if len(nodes) < 1000:
        return nodes
    count = len(nodes) // 20
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
    validate_structural_graph(graph, nodes)


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
    yield from structural_distance_rows(graph, sources)


def _fit_bins(bins, count, lowest, highest, *, diagnostics):
    """One fit definition serves both the historical scalar and rich report."""
    fitted = [(r, mean) for r, size, mean in bins if size >= 2 and mean > 1e-9]
    selected = tuple(r for r, _ in fitted)

    def unavailable(status):
        if diagnostics:
            return CoherenceFitDiagnostics(None, False, status, selected)
        return float("nan")

    if count < 10:
        return unavailable("insufficient_pairs")
    if lowest == highest:
        return unavailable("constant_products")
    if len(fitted) < 3:
        return unavailable("insufficient_bins")
    distances, means = zip(*fitted)
    # A constant product field has no negative decay slope. Do not let a
    # least-squares rounding residual turn it into a huge finite decay length.
    if min(means) == max(means):
        return unavailable("constant_bin_means")
    scale = max(distances)
    x = np.asarray(distances, dtype=float) / scale
    y = np.log(np.asarray(means, dtype=float))
    try:
        if diagnostics:
            coefficients, _, rank, singular, rcond = np.polyfit(x, y, 1, full=True)
            slope, intercept = coefficients
        else:
            slope, intercept = np.polyfit(x, y, 1)
    except np.linalg.LinAlgError:
        return unavailable("linear_algebra_failure")
    status = "fitted"
    value = None
    if not math.isfinite(slope) or slope >= 0:
        status = "nonnegative_or_nonfinite_slope"
    else:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            candidate = float(scale / -slope)
        if math.isfinite(candidate) and candidate > 0:
            value = candidate
        else:
            status = "unrepresentable_length"
    if not diagnostics:
        return value if value is not None else float("nan")

    residuals = tuple(
        float(observed - (slope * r + intercept)) for r, observed in zip(x, y)
    )
    finite_residuals = all(math.isfinite(value) for value in residuals)
    rss = math.fsum(value * value for value in residuals) if finite_residuals else None
    mean_log = math.fsum(float(value) for value in y) / len(y)
    tss = math.fsum((float(value) - mean_log) ** 2 for value in y)
    # Do not clip descriptive R² or treat its value as an acceptance policy.
    r_squared = 1.0 - rss / tss if rss is not None and tss > 0 else None
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        dimensional_slope = float(slope / scale)
    return CoherenceFitDiagnostics(
        value=value,
        available=value is not None,
        status=status,
        selected_distances=selected,
        slope=dimensional_slope if math.isfinite(dimensional_slope) else None,
        intercept=float(intercept) if math.isfinite(intercept) else None,
        residual_sum_squares=rss,
        residual_rms=math.sqrt(rss / len(y)) if rss is not None else None,
        r_squared=r_squared,
        design_rank=int(rank),
        singular_values=tuple(float(value) for value in singular),
        rcond=float(rcond),
        distance_scale=scale,
        normalized_slope=float(slope) if math.isfinite(slope) else None,
        log_residuals=residuals if finite_residuals else (),
    )


def _observe_bin(distance, entries, product_mean, global_mean, directed):
    """Reduce bounded admitted endpoints without subtractive covariance sums."""
    count = len(entries)
    source_mean = math.fsum(row[0] for row in entries) / count
    target_mean = math.fsum(row[1] for row in entries) / count
    source_second = math.fsum(row[0] ** 2 for row in entries) / count
    target_second = math.fsum(row[1] ** 2 for row in entries) / count
    endpoint_mean = math.fsum(value for row in entries for value in row[:2]) / (
        2 * count
    )
    endpoint_second = math.fsum(
        value * value for row in entries for value in row[:2]
    ) / (2 * count)
    if not directed:
        source_mean = target_mean = endpoint_mean
        source_second = target_second = endpoint_second
    centered = (
        math.fsum(
            (left - global_mean) * (right - global_mean) for left, right, _ in entries
        )
        / count
    )
    covariance = (
        math.fsum(
            (left - source_mean) * (right - target_mean) for left, right, _ in entries
        )
        / count
    )
    semivariogram = math.fsum((left - right) ** 2 for left, right, _ in entries) / (
        2 * count
    )
    exclusion = (
        "insufficient_bin_pairs"
        if count < 2
        else "product_at_or_below_floor" if product_mean <= 1e-9 else None
    )
    return CoherenceDistanceBin(
        distance=distance,
        pair_count=count,
        product_mean=product_mean,
        endpoint_mean=endpoint_mean,
        endpoint_second_moment=endpoint_second,
        source_mean=source_mean,
        target_mean=target_mean,
        source_second_moment=source_second,
        target_second_moment=target_second,
        centered_product=centered,
        endpoint_covariance=covariance,
        semivariogram=semivariogram,
        fit_included=exclusion is None,
        fit_exclusion_reason=exclusion,
    )


def fit_coherence_length(
    graph,
    nodes,
    delta_nfr,
    *,
    sources,
    dtype=np.float64,
    materialize=False,
    distance_matrix=None,
    return_profile=False,
    precision_mode="unspecified",
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
    requested_dtype = np.dtype(dtype)
    if requested_dtype.kind != "f":
        raise ValueError("coherence dtype must be a real floating dtype")
    admitted_pressure = np.asarray(
        [
            finite_real_scalar(delta_nfr.get(node, 0.0), "coherence pressure")
            for node in nodes
        ],
        dtype=float,
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        pressure = np.asarray(admitted_pressure, dtype=requested_dtype)
    if np.any(~np.isfinite(pressure)) or np.any(
        (admitted_pressure != 0) & (pressure == 0)
    ):
        raise ValueError("coherence pressure is unrepresentable in the requested dtype")
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

    bins = {}
    count = zero_count = 0
    lowest, highest = math.inf, -math.inf
    directed = bool(graph.is_directed())
    for source, row in rows:
        i = index[source]
        for target, distance in row.items():
            j = index[target]
            if i == j:
                continue
            if not directed and target in source_set and j < i:
                continue  # Count each sampled unordered pair once.
            if not math.isfinite(distance):
                continue
            if distance == 0:
                zero_count += 1
                continue
            product = float(coherence[i] * coherence[j])
            entry = (
                (float(coherence[i]), float(coherence[j]), product)
                if return_profile
                else product
            )
            bins.setdefault(float(distance), []).append(entry)
            count += 1
            lowest, highest = min(lowest, product), max(highest, product)
    summaries = tuple(
        (
            distance,
            len(values),
            (
                math.fsum(row[2] for row in values) / len(values)
                if return_profile
                else math.fsum(values) / len(values)
            ),
        )
        for distance, values in sorted(bins.items())
    )
    fit = _fit_bins(summaries, count, lowest, highest, diagnostics=return_profile)
    if not return_profile:
        return fit
    coherence_values = tuple(float(value) for value in coherence)
    mean = math.fsum(coherence_values) / len(nodes) if nodes else None
    variance = (
        math.fsum((value - mean) ** 2 for value in coherence_values) / len(nodes)
        if nodes
        else None
    )
    n, s = len(nodes), len(sources)
    candidates = s * (n - 1) if directed else s * (n - s) + s * (s - 1) // 2
    return CoherenceProfile(
        nodes=nodes,
        sources=sources,
        coherence=coherence_values,
        global_mean=mean,
        global_variance=variance,
        candidate_pair_count=candidates,
        positive_pair_count=count,
        zero_distance_pair_count=zero_count,
        unreachable_pair_count=candidates - count - zero_count,
        bins=tuple(
            _observe_bin(distance, bins[distance], product_mean, mean, directed)
            for distance, _, product_mean in summaries
        ),
        fit=fit,
        distance_weighting=DISTANCE_DESCRIPTION + "; fitted value in path-length units",
        sample_selection=coherence_sample_description(graph, nodes, sources),
        endpoint_convention=(
            "ordered outgoing roles" if directed else "symmetrized unordered endpoints"
        ),
        precision_mode=precision_mode,
        arithmetic_dtype=requested_dtype.name
        + "; coherence, products and fit materialized to binary64",
        distance_origin=(
            "caller-declared matrix"
            if distance_matrix is not None
            else "graph shortest paths"
        ),
        fit_policy=FIT_DESCRIPTION,
    )
