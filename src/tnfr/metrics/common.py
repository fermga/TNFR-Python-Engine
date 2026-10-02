"""Shared helpers for TNFR metrics."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Collection, Iterable, Mapping, Sequence

from .._coherence_validation import validate_structural_coherence
from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..constants import DEFAULTS
from ..constants.aliases import ALIAS_D2EPI, ALIAS_DEPI, ALIAS_DNFR, ALIAS_VF
from ..mathematics.unified_numerical import np
from ..types import GraphLike, NodeAttrMap
from ..utils import (
    clamp01,
    edge_version_cache,
    normalize_optional_int,
    normalize_weights,
)

__all__ = (
    "GraphLike",
    "finite_mean_absolute",
    "finite_mean",
    "finite_population_std",
    "finite_pearson_correlation",
    "validate_structural_coherence",
    "compute_coherence",
    "structural_coherence",
    "is_structural_equilibrium",
    "compute_dnfr_accel_max",
    "normalize_dnfr",
    "ensure_neighbors_map",
    "merge_graph_weights",
    "merge_and_normalize_weights",
    "min_max_range",
    "_coerce_jobs",
    "_get_vf_dnfr_max",
)

# Default tolerances for the zero-pressure/zero-rate diagnostic, shared with
# tnfr.metrics.coherence._track_stability. This is narrower than EPI stationarity:
# zero capacity can give dEPI/dt = 0 even when pressure is nonzero.
_EPS_DNFR_STABLE: float = float(DEFAULTS["EPS_DNFR_STABLE"])
_EPS_DEPI_STABLE: float = float(DEFAULTS["EPS_DEPI_STABLE"])


def _finite_scalar(value: float, *, name: str) -> float:
    """Admit one represented real without erasing nonzero source evidence."""
    if isinstance(value, bool) or (np is not None and isinstance(value, np.bool_)):
        raise TypeError(f"{name} must be a finite real scalar, not bool")
    # Most stored observations already are binary64. Avoid constructing an
    # unused rational for this hot path; other real types use the shared
    # materialization contract, including source-to-zero underflow rejection.
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return float(value)
    return finite_represented_real(value, name)[0]


def _stored_metric_scalar(nd: NodeAttrMap, aliases: tuple[str, ...]) -> float:
    """Validate the first present alias; only absent channels default to zero."""
    value = get_attr(nd, aliases, 0.0, strict=True, conv=lambda item: item)
    normalized = _finite_scalar(value, name=aliases[0])
    if aliases == ALIAS_VF and normalized < 0.0:
        raise ValueError("stored structural capacity must be nonnegative")
    return normalized


def _finite_array(value: Any, *, name: str) -> Any:
    """Admit real arrays without NumPy erasing input types or tiny sources."""
    if not isinstance(value, np.ndarray) or value.dtype.kind == "O":
        # Object materialization preserves booleans in otherwise numeric
        # sequences, unlike np.asarray([True, 2.0])'s implicit float coercion.
        raw = np.asarray(value, dtype=object)
        normalized = np.fromiter(
            (_finite_scalar(item, name=name) for item in raw.flat),
            dtype=float,
            count=raw.size,
        )
        return normalized.reshape(raw.shape)
    if value.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain finite real values")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = np.asarray(value, dtype=float)
    if not bool(np.all(np.isfinite(normalized))):
        raise ValueError(f"{name} must contain only finite values")
    if bool(np.any((value != 0) & (normalized == 0))):
        raise ValueError(f"{name} contains nonzero values that underflow to zero")
    return normalized


def _stored_metric_values(
    G: GraphLike, nodes: Iterable[Any], aliases: tuple[str, ...]
) -> Iterable[float]:
    """Read authoritative aliases strictly, preserving missing-value zero.

    Validate original scalar types before materializing the shared coordinate.
    A malformed first alias must not turn into a later value or default zero.
    """
    for node in nodes:
        yield _stored_metric_scalar(G.nodes[node], aliases)


def finite_mean_absolute(values: Iterable[float], *, name: str) -> float:
    """Return a finite mean magnitude without overflowing the intermediate sum.

    Scaling by the largest magnitude keeps the reduction in ``[0, count]``.
    This matters when several valid binary64 inputs are close to the maximum
    finite value: their mathematical mean is representable even though their
    unscaled sum is not.

    Parameters
    ----------
    values : iterable of float
        Real scalar values to reduce. Truth values and non-finite values are
        rejected rather than silently converted.
    name : str
        Channel label included in validation errors.
    """
    magnitudes = tuple(abs(_finite_scalar(value, name=name)) for value in values)
    if not magnitudes:
        return 0.0
    scale = max(magnitudes)
    if scale == 0.0:
        return 0.0
    normalized_mean = math.fsum(value / scale for value in magnitudes) / len(magnitudes)
    result = scale * normalized_mean
    if not math.isfinite(result):
        raise ValueError(f"mean absolute {name} exceeds finite range")
    return result


def finite_mean(values: Iterable[float], *, name: str = "values") -> float:
    """Return a signed finite mean using the shared stable linear reduction.

    Empty samples retain the aggregate convention zero. The neighborhood
    kernel at center zero preserves cancellation and handles representable
    means whose unscaled sum overflows; this is an observation, not pressure.
    """
    from ..mathematics._neighbor_differences import mean_neighbor_difference

    samples = tuple(_finite_scalar(value, name=name) for value in values)
    return mean_neighbor_difference(0.0, samples)


def _normalized_sample_deviations(
    samples: tuple[float, ...],
) -> tuple[float, tuple[float, ...]]:
    """Translate admitted samples before scaling; zero scale means constant."""
    if not samples:
        return 0.0, ()
    low, high = min(samples), max(samples)
    if low == high:
        return 0.0, ()
    midpoint = low / 2.0 + high / 2.0
    deviations = tuple(value - midpoint for value in samples)
    scale = max(map(abs, deviations))
    return scale, tuple(value / scale for value in deviations)


def finite_population_std(values: Iterable[float], *, name: str = "values") -> float:
    """Return population standard deviation without unscaled squares or sums.

    Finite real scalar inputs are validated before arithmetic. Empty and
    constant samples return zero. Translation by the range midpoint precedes
    scaling, preserving the small spread of adjacent large same-sign values.
    The final product retains ordinary binary64 rounding, including subnormal
    results and underflow when the standard deviation is unrepresentable.
    """
    samples = tuple(_finite_scalar(value, name=name) for value in values)
    scale, normalized = _normalized_sample_deviations(samples)
    if scale == 0.0:
        return 0.0
    mean = math.fsum(normalized) / len(normalized)
    variance = math.fsum((value - mean) ** 2 for value in normalized) / len(samples)
    # A variable in [-1, 1] has population variance at most one. Enforce the
    # exact bound against a rounding excursion near the largest finite float.
    return scale * min(1.0, math.sqrt(variance))


def finite_pearson_correlation(
    left: Iterable[float], right: Iterable[float], *, name: str = "values"
) -> float | None:
    """Return Pearson correlation of paired finite represented real samples.

    Both streams are validated before checking availability. Unequal lengths,
    non-finite coordinates and invalid raw scalar types are rejected. Fewer
    than two pairs or an exactly constant represented sample return ``None``;
    there is no fixed amplitude cutoff and missing values are never dropped.

    Midpoint translation and separate scaling preserve variation in adjacent
    large coordinates and subnormal samples without squaring their original
    magnitudes. Centered sums use ordinary binary64 products and stable sums;
    this is a rounded diagnostic, not an exact-arithmetic correlation theorem.
    """
    left_samples = tuple(_finite_scalar(value, name=f"{name}.left") for value in left)
    right_samples = tuple(
        _finite_scalar(value, name=f"{name}.right") for value in right
    )
    if len(left_samples) != len(right_samples):
        raise ValueError(
            f"{name} must have equal sample lengths for paired correlation"
        )
    if len(left_samples) < 2:
        return None
    left_scale, left_normalized = _normalized_sample_deviations(left_samples)
    right_scale, right_normalized = _normalized_sample_deviations(right_samples)
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    count = len(left_samples)
    left_mean = math.fsum(left_normalized) / count
    right_mean = math.fsum(right_normalized) / count
    left_centered = tuple(value - left_mean for value in left_normalized)
    right_centered = tuple(value - right_mean for value in right_normalized)
    covariance = math.fsum(
        a * b for a, b in zip(left_centered, right_centered, strict=True)
    )
    left_norm = math.sqrt(math.fsum(value * value for value in left_centered))
    right_norm = math.sqrt(math.fsum(value * value for value in right_centered))
    result = covariance / (left_norm * right_norm)
    return max(-1.0, min(1.0, result))


def _dispersion_coherence(values: Iterable[float]) -> float:
    """Evaluate the auxiliary ``1 - std(p) / max(abs(p))`` read-out.

    Normalize finite signed pressures before computing the population variance.
    The normalized values are bounded by one, avoiding overflow and underflow
    from squaring the original pressure scale. Network and neighborhood
    wrappers share this arithmetic, independently of the available backend.
    Empty and all-zero inputs retain the public value one.
    """

    pressures = tuple(_finite_scalar(value, name="dnfr") for value in values)
    if not pressures:
        return 1.0
    scale = max(abs(value) for value in pressures)
    if scale == 0.0:
        return 1.0
    normalized = tuple(value / scale for value in pressures)
    mean = math.fsum(normalized) / len(normalized)
    variance = math.fsum((value - mean) ** 2 for value in normalized) / len(normalized)
    return clamp01(1.0 - math.sqrt(variance))


def structural_coherence(dnfr: Any, depi: Any = 0.0) -> Any:
    r"""Structural coherence ``C = 1/(1 + |ΔNFR| + |dEPI|)``.

    The single-node kernel used by the canonical network coherence
    :func:`compute_coherence`. It is the shared local coherence map used by
    several TNFR domain models. It returns ``1`` when both arguments vanish
    and decreases towards ``0`` as either magnitude grows without bound.
    These properties motivate the chosen reciprocal map; the nodal equation
    does not uniquely derive that map.

    Both inputs are represented numerical coordinates in the engine's chosen
    pressure and rate scales. Adding their magnitudes to ``1`` assumes those
    scales; this is not a unit-independent dimensional identity. In particular,
    changing the time coordinate while holding pressure fixed changes ``depi``
    and generally changes this diagnostic.

    Domains differ in their state spaces, definitions of ``ΔNFR`` and available
    dynamics. Reusing this scalar map centralizes a convention; it does not
    prove a cross-domain physical identity.

    Parameters
    ----------
    dnfr : real scalar or array-like
        Structural reorganization pressure ``ΔNFR``. Arrays are evaluated
        elementwise for vectorized field read-outs.
    depi : real scalar or array-like, optional
        Structural change rate ``dEPI`` (default ``0`` for static fields).
        Array inputs follow NumPy broadcasting rules.

    Notes
    -----
    Inputs must be real numerical coordinates, not truth values or parseable
    text. Materialization to binary64 rejects non-finite values and nonzero
    sources that would become zero. Ordinary rounding of the resulting
    diagnostic still applies: a value of one need not mean exact zero input.
    """
    if np is not None and (not np.isscalar(dnfr) or not np.isscalar(depi)):
        arrays = [_finite_array(dnfr, name="dnfr"), _finite_array(depi, name="depi")]
        try:
            pressure, rate = np.broadcast_arrays(np.abs(arrays[0]), np.abs(arrays[1]))
        except ValueError as exc:
            raise ValueError(
                "dnfr and depi arrays must be broadcast-compatible"
            ) from exc

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            denominator = 1.0 + pressure + rate
            direct = np.reciprocal(denominator)
            scale = np.maximum(pressure, rate)
            safe_scale = np.where(scale == 0.0, 1.0, scale)
            inverse_scale = np.reciprocal(safe_scale)
            scaled = inverse_scale / (
                inverse_scale + pressure / safe_scale + rate / safe_scale
            )
        return np.where(np.isfinite(denominator), direct, scaled)

    dnfr_value = _finite_scalar(dnfr, name="dnfr")
    depi_value = _finite_scalar(depi, name="depi")
    pressure = abs(dnfr_value)
    rate = abs(depi_value)
    denominator = 1.0 + pressure + rate
    if math.isfinite(denominator):
        return 1.0 / denominator

    # The exact denominator may exceed binary64 even though its reciprocal is
    # representable. Scale only on that exceptional path so ordinary inputs
    # retain the historical arithmetic and bit pattern.
    scale = max(pressure, rate)
    inverse_scale = 1.0 / scale
    return inverse_scale / (inverse_scale + pressure / scale + rate / scale)


def _equilibrium_tolerances(eps_dnfr: float, eps_depi: float) -> tuple[float, float]:
    """Validate diagnostic cuts independently of the number of observed nodes."""
    dnfr_tolerance = _finite_scalar(eps_dnfr, name="eps_dnfr")
    depi_tolerance = _finite_scalar(eps_depi, name="eps_depi")
    if dnfr_tolerance < 0.0 or depi_tolerance < 0.0:
        raise ValueError("equilibrium tolerances must be non-negative")
    return dnfr_tolerance, depi_tolerance


def is_structural_equilibrium(
    dnfr: float,
    depi: float = 0.0,
    *,
    eps_dnfr: float = _EPS_DNFR_STABLE,
    eps_depi: float = _EPS_DEPI_STABLE,
) -> bool:
    r"""Test the configured zero-pressure and zero-rate tolerance region.

    Tests ``|ΔNFR| <= eps_dnfr`` **and** ``|dEPI| <= eps_depi`` -- the
    per-node diagnostic used by the engine's coherence tracker
    (:func:`tnfr.metrics.coherence._track_stability`). Positive tolerances do
    not certify an exact fixed point. Zero capacity can freeze EPI under
    nonzero pressure, outside this region; phase, capacity and graph evolution
    are not tested. The arguments are read-outs, not a check of ``depi=nu_f*dnfr``.

    Graph, arithmetic and shell models can all apply this same numerical test
    to their own pressure fields. They then share a predicate and tolerance
    convention, not a fixed point in one common phase space. Closed-loop phase
    winding is a different topological observation and is not evaluated here.

    Parameters
    ----------
    dnfr : float
        Structural reorganization pressure ``ΔNFR``.
    depi : float, optional
        Structural change rate ``dEPI`` (default ``0``).
    eps_dnfr, eps_depi : float, optional
        Finite nonnegative real tolerances (default: ``EPS_*_STABLE``).
        Truth values, text and nonzero source-to-zero underflow are rejected.
    """
    dnfr_value = _finite_scalar(dnfr, name="dnfr")
    depi_value = _finite_scalar(depi, name="depi")
    dnfr_tolerance, depi_tolerance = _equilibrium_tolerances(eps_dnfr, eps_depi)
    return abs(dnfr_value) <= dnfr_tolerance and abs(depi_value) <= depi_tolerance


def _coherence_on_nodes(
    G: GraphLike, nodes: Collection[Any]
) -> tuple[float, float, float]:
    """Aggregate stored pressure/rate over a reusable selected node collection.

    Empty support has the public aggregate value zero. Local callers choose
    their support explicitly; admission, stable means and the reciprocal map
    are identical to the global observation. No graph copy is needed.
    """
    if not nodes:
        return 0.0, 0.0, 0.0
    dnfr_mean = finite_mean_absolute(
        _stored_metric_values(G, nodes, ALIAS_DNFR), name="dnfr"
    )
    depi_mean = finite_mean_absolute(
        _stored_metric_values(G, nodes, ALIAS_DEPI), name="depi"
    )
    return structural_coherence(dnfr_mean, depi_mean), dnfr_mean, depi_mean


def compute_coherence(
    G: GraphLike, *, return_means: bool = False
) -> float | tuple[float, float, float]:
    r"""Compute the canonical total coherence ``C(t)`` of the network.

    This is the **primary canonical coherence metric** of the TNFR engine:
    the value recorded in ``history['C_steps']`` on every ``step()`` (see
    :func:`tnfr.metrics.coherence._update_coherence`) and exposed through the
    SDK, telemetry, and the structural-health interface.

    .. math::
        C(t) = \frac{1}{1 + \overline{|\Delta\mathrm{NFR}|} + \overline{|d\mathrm{EPI}|}}

    where the bars denote network means of the stored pressure and rate
    aliases. Missing aliases default to zero; malformed authoritative aliases
    raise rather than fall through to later aliases or fabricate zero pressure.
    This function does not refresh pressure or reconstruct ``dEPI`` from
    capacity, so its inputs need not be contemporaneous samples of the nodal
    equation.

    The reciprocal diagnostic is a convention with chosen numerical pressure
    and rate scales, not a unique consequence of the nodal equation. It is
    **not** scale-invariant. For a nonempty graph, applying it to the mean
    magnitudes is generally different from averaging nodal coherence: the
    former is at most the latter, with equality exactly when the nodal sums
    ``|ΔNFR| + |dEPI|`` are constant (in exact arithmetic). Nor is it coherence
    of a projected parent state, where signed cancellation can occur.

    An empty graph returns ``0`` by API convention, rather than the local
    zero-input value ``1``.
    """

    observation = _coherence_on_nodes(G, G.nodes)
    return observation if return_means else observation[0]


def ensure_neighbors_map(G: GraphLike) -> Mapping[Any, Sequence[Any]]:
    """Return cached neighbors list keyed by node as a read-only mapping."""

    def builder() -> Mapping[Any, Sequence[Any]]:
        return MappingProxyType({n: tuple(G.neighbors(n)) for n in G})

    return edge_version_cache(G, "_neighbors", builder)


def merge_graph_weights(
    G: GraphLike, key: str, *, strict: bool = False
) -> dict[str, float]:
    """Merge default weights for ``key`` with any graph overrides."""

    overrides = G.graph.get(key, {})
    if overrides is None or not isinstance(overrides, Mapping):
        if strict:
            raise TypeError(f"{key} must be a weight mapping")
        overrides = {}
    return {**DEFAULTS[key], **overrides}


def _materialize_weight_mapping(
    weights: Mapping[str, Any],
    fields: Sequence[str],
    *,
    default: float = 0.0,
    name: str = "weights",
) -> dict[str, float]:
    """Admit consumed nonnegative real coefficients before normalization.

    Numeric zero-dimensional NumPy containers retain configuration compatibility;
    arrays, booleans, text and nonzero sources lost to binary64 are rejected.
    The returned floats detach mutable numeric containers from cached policy.
    """
    if not isinstance(weights, Mapping):
        raise TypeError(f"{name} must be a weight mapping")
    materialized = {}
    for key in fields:
        raw = weights.get(key, default)
        if np is not None and isinstance(raw, np.ndarray):
            if raw.ndim != 0 or raw.dtype.kind not in "iuf":
                raise TypeError(f"{name}[{key!r}] must be a real scalar")
            raw = raw.item()
        value = _finite_scalar(raw, name=f"{name}[{key!r}]")
        if value < 0.0:
            raise ValueError(f"{name}[{key!r}] must be nonnegative")
        materialized[key] = value
    return materialized


def merge_and_normalize_weights(
    G: GraphLike,
    key: str,
    fields: Sequence[str],
    *,
    default: float = 0.0,
) -> dict[str, float]:
    """Admit configured coefficients and normalize the consumed fields.

    Missing entries overlay defaults. Zero total retains the declared legacy
    uniform-mixture policy; invalid declarations never select that fallback.
    """

    w = _materialize_weight_mapping(
        merge_graph_weights(G, key, strict=True), fields, default=default, name=key
    )
    return normalize_weights(
        w,
        fields,
        default=default,
        error_on_conversion=True,
        error_on_negative=True,
        warn_once=True,
    )


def compute_dnfr_accel_max(G: GraphLike) -> dict[str, float]:
    """Read absolute maxima of stored pressure and EPI acceleration aliases.

    No derivatives are recomputed. The default integrator records a difference
    of consecutive nodal rates divided by its supplied time step; this is a
    numerical read-out, not an independent second-order evolution law. Missing
    aliases default to zero; malformed authoritative values raise.
    """

    return _stored_abs_maxima(G, {"dnfr_max": ALIAS_DNFR, "accel_max": ALIAS_D2EPI})


def _stored_abs_maxima(
    G: GraphLike, alias_map: Mapping[str, tuple[str, ...]]
) -> dict[str, float]:
    """Reduce validated stored channels without falling through invalid aliases."""
    maxima: dict[str, float] = {}
    for _, nd in G.nodes(data=True):
        for name, aliases in alias_map.items():
            magnitude = abs(_stored_metric_scalar(nd, aliases))
            maxima[name] = max(maxima.get(name, 0.0), magnitude)
    return maxima


def normalize_dnfr(nd: NodeAttrMap, max_val: float) -> float:
    """Normalize admitted pressure by a finite nonnegative supplied maximum.

    A zero maximum retains the zero diagnostic convention. It does not exempt
    the stored pressure from validation or prove that the node is equilibrated.
    """

    max_val = _finite_scalar(max_val, name="max_val")
    if max_val < 0.0:
        raise ValueError("max_val must be nonnegative")
    val = abs(_stored_metric_scalar(nd, ALIAS_DNFR))
    if max_val == 0.0:
        return 0.0
    return clamp01(val / max_val)


def min_max_range(
    values: Iterable[float], *, default: tuple[float, float] = (0.0, 0.0)
) -> tuple[float, float]:
    """Return the minimum and maximum values observed in ``values``."""

    it = iter(values)
    try:
        first = next(it)
    except StopIteration:
        return default
    min_val = max_val = first
    for val in it:
        if val < min_val:
            min_val = val
        elif val > max_val:
            max_val = val
    return min_val, max_val


def _get_vf_dnfr_max(G: GraphLike) -> tuple[float, float]:
    """Refresh current absolute maxima and return nonzero Si divisors.

    Direct node writes can bypass the alias setters' cache maintenance. The
    Python Si path therefore refreshes these read-outs on every call, matching
    the NumPy path. Cached zero remains zero; only its returned divisor is one.
    """

    maxes = _stored_abs_maxima(G, {"_vfmax": ALIAS_VF, "_dnfrmax": ALIAS_DNFR})
    vfmax = maxes.get("_vfmax", 0.0)
    dnfrmax = maxes.get("_dnfrmax", 0.0)
    G.graph["_vfmax"] = vfmax
    G.graph["_dnfrmax"] = dnfrmax
    vfmax = 1.0 if vfmax == 0 else vfmax
    dnfrmax = 1.0 if dnfrmax == 0 else dnfrmax
    return float(vfmax), float(dnfrmax)


def _coerce_jobs(raw_jobs: Any | None) -> int | None:
    """Normalise parallel job hints shared by metrics modules."""

    return normalize_optional_int(
        raw_jobs,
        allow_non_positive=False,
        strict=False,
        sentinels=None,
    )
