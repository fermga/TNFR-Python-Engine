"""Shared helpers for TNFR metrics."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .._coherence_validation import validate_structural_coherence
from ..alias import collect_attr, get_attr, multi_recompute_abs_max
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

# Canonical structural-equilibrium tolerances: the nodal-equation fixed point
# dEPI/dt = vf*dNFR = 0. Sourced from the same DEFAULTS the per-node stability
# tracker uses (tnfr.metrics.coherence._track_stability), so this module is the
# single source of truth for the equilibrium criterion.
_EPS_DNFR_STABLE: float = float(DEFAULTS["EPS_DNFR_STABLE"])
_EPS_DEPI_STABLE: float = float(DEFAULTS["EPS_DEPI_STABLE"])


def _finite_scalar(value: float, *, name: str) -> float:
    """Normalize a finite real scalar while rejecting truth values."""
    if isinstance(value, bool) or (
        np is not None and isinstance(value, np.bool_)
    ):
        raise TypeError(f"{name} must be a finite real scalar, not bool")
    if isinstance(value, (str, bytes)) or (
        np is not None and not bool(np.isscalar(value))
    ):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


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
    normalized_mean = math.fsum(value / scale for value in magnitudes) / len(
        magnitudes
    )
    result = scale * normalized_mean
    if not math.isfinite(result):
        raise ValueError(f"mean absolute {name} exceeds finite range")
    return result



def structural_coherence(dnfr: Any, depi: Any = 0.0) -> Any:
    r"""Structural coherence ``C = 1/(1 + |ΔNFR| + |dEPI|)``.

    The single-node kernel used by the canonical network coherence
    :func:`compute_coherence`. It is the shared local coherence map used by
    several TNFR domain models and is motivated by the nodal equation
    :math:`\partial\mathrm{EPI}/\partial t = \nu_f\,\Delta\mathrm{NFR}`: at the
    equilibrium fixed point (:math:`\Delta\mathrm{NFR}=0\Rightarrow d\mathrm{EPI}=0`)
    it returns ``1`` and decays monotonically towards ``0`` under unbounded
    reorganization pressure.

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
    """
    if np is not None and (not np.isscalar(dnfr) or not np.isscalar(depi)):
        arrays = []
        for value, name in ((dnfr, "dnfr"), (depi, "depi")):
            raw = np.asarray(value)
            if raw.dtype.kind in "bSUOc":
                raise TypeError(f"{name} must contain finite real values")
            try:
                normalized = np.asarray(value, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(f"{name} must contain finite real values") from exc
            if not bool(np.all(np.isfinite(normalized))):
                raise ValueError(f"{name} must contain only finite values")
            arrays.append(normalized)
        try:
            pressure, rate = np.broadcast_arrays(
                np.abs(arrays[0]), np.abs(arrays[1])
            )
        except ValueError as exc:
            raise ValueError("dnfr and depi arrays must be broadcast-compatible") from exc

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


def is_structural_equilibrium(
    dnfr: float,
    depi: float = 0.0,
    *,
    eps_dnfr: float = _EPS_DNFR_STABLE,
    eps_depi: float = _EPS_DEPI_STABLE,
) -> bool:
    r"""Canonical equilibrium fixed-point predicate of the nodal equation.

    Returns ``True`` iff the node sits at the structural-equilibrium fixed
    point of :math:`\partial\mathrm{EPI}/\partial t = \nu_f\,\Delta\mathrm{NFR}`,
    tested as ``|ΔNFR| <= eps_dnfr`` **and** ``|dEPI| <= eps_depi`` -- exactly
    the per-node stability criterion used by the engine's coherence tracker
    (:func:`tnfr.metrics.coherence._track_stability`).

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
        Equilibrium tolerances (default: the canonical ``EPS_*_STABLE``).
    """
    dnfr_value = _finite_scalar(dnfr, name="dnfr")
    depi_value = _finite_scalar(depi, name="depi")
    dnfr_tolerance = _finite_scalar(eps_dnfr, name="eps_dnfr")
    depi_tolerance = _finite_scalar(eps_depi, name="eps_depi")
    if dnfr_tolerance < 0.0 or depi_tolerance < 0.0:
        raise ValueError("equilibrium tolerances must be non-negative")
    return (
        abs(dnfr_value) <= dnfr_tolerance
        and abs(depi_value) <= depi_tolerance
    )


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

    where the bars denote network means. It is derived directly from the nodal
    equation :math:`\partial\mathrm{EPI}/\partial t = \nu_f\,\Delta\mathrm{NFR}`:
    structural equilibrium is :math:`\Delta\mathrm{NFR}\to 0` (no pressure) and
    :math:`d\mathrm{EPI}\to 0` (no change), so :math:`C\to 1` at equilibrium and
    :math:`C\to 0` under unbounded pressure/change. The map
    :math:`[0,\infty)\to(0,1]` is monotone and, unlike the dispersion variant
    :func:`coherence.compute_global_coherence`, is **not** scale-invariant: it
    tracks the absolute magnitude of reorganization pressure.
    """

    count = G.number_of_nodes()
    if count == 0:
        return (0.0, 0.0, 0.0) if return_means else 0.0

    nodes = G.nodes
    dnfr_values = collect_attr(G, nodes, ALIAS_DNFR, 0.0)
    depi_values = collect_attr(G, nodes, ALIAS_DEPI, 0.0)

    dnfr_mean = finite_mean_absolute(dnfr_values, name="dnfr")
    depi_mean = finite_mean_absolute(depi_values, name="depi")

    coherence = structural_coherence(dnfr_mean, depi_mean)
    return (coherence, dnfr_mean, depi_mean) if return_means else coherence


def ensure_neighbors_map(G: GraphLike) -> Mapping[Any, Sequence[Any]]:
    """Return cached neighbors list keyed by node as a read-only mapping."""

    def builder() -> Mapping[Any, Sequence[Any]]:
        return MappingProxyType({n: tuple(G.neighbors(n)) for n in G})

    return edge_version_cache(G, "_neighbors", builder)


def merge_graph_weights(G: GraphLike, key: str) -> dict[str, float]:
    """Merge default weights for ``key`` with any graph overrides."""

    overrides = G.graph.get(key, {})
    if overrides is None or not isinstance(overrides, Mapping):
        overrides = {}
    return {**DEFAULTS[key], **overrides}


def merge_and_normalize_weights(
    G: GraphLike,
    key: str,
    fields: Sequence[str],
    *,
    default: float = 0.0,
) -> dict[str, float]:
    """Merge defaults for ``key`` and normalise ``fields``."""

    w = merge_graph_weights(G, key)
    return normalize_weights(
        w,
        fields,
        default=default,
        error_on_conversion=False,
        error_on_negative=False,
        warn_once=True,
    )


def compute_dnfr_accel_max(G: GraphLike) -> dict[str, float]:
    """Compute absolute maxima of |ΔNFR| and |d²EPI/dt²|."""

    return multi_recompute_abs_max(
        G, {"dnfr_max": ALIAS_DNFR, "accel_max": ALIAS_D2EPI}
    )


def normalize_dnfr(nd: NodeAttrMap, max_val: float) -> float:
    """Normalise ``|ΔNFR|`` using ``max_val``."""

    if max_val <= 0:
        return 0.0
    val = abs(get_attr(nd, ALIAS_DNFR, 0.0))
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
    """Ensure and return absolute maxima for ``νf`` and ``ΔNFR``."""

    vfmax = G.graph.get("_vfmax")
    dnfrmax = G.graph.get("_dnfrmax")
    if vfmax is None or dnfrmax is None:
        maxes = multi_recompute_abs_max(G, {"_vfmax": ALIAS_VF, "_dnfrmax": ALIAS_DNFR})
        if vfmax is None:
            vfmax = maxes["_vfmax"]
        if dnfrmax is None:
            dnfrmax = maxes["_dnfrmax"]
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
