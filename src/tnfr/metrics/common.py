"""Shared helpers for TNFR metrics."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ..alias import collect_attr, get_attr, multi_recompute_abs_max
from ..constants import DEFAULTS
from ..constants.aliases import ALIAS_D2EPI, ALIAS_DEPI, ALIAS_DNFR, ALIAS_VF
from ..utils import clamp01, kahan_sum_nd, normalize_optional_int
from ..types import GraphLike, NodeAttrMap
from ..utils import edge_version_cache, normalize_weights
from ..mathematics.unified_numerical import np

__all__ = (
    "GraphLike",
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

def structural_coherence(dnfr: float, depi: float = 0.0) -> float:
    r"""Per-node structural coherence ``C = 1/(1 + |ΔNFR| + |dEPI|)``.

    The single-node kernel of the canonical network coherence
    :func:`compute_coherence`. It is the **one** local coherence map shared by
    every TNFR domain, derived directly from the nodal equation
    :math:`\partial\mathrm{EPI}/\partial t = \nu_f\,\Delta\mathrm{NFR}`: at the
    equilibrium fixed point (:math:`\Delta\mathrm{NFR}=0\Rightarrow d\mathrm{EPI}=0`)
    it returns ``1`` and decays monotonically towards ``0`` under unbounded
    reorganization pressure.

    Domains differ only in how they *realise* ``ΔNFR`` -- the graph random-walk
    Laplacian for the dynamics, an arithmetic pressure for number theory, a
    valence pressure for chemistry. The coherence map itself is invariant; this
    is what makes the equilibrium structure fractal and resonant across scales.

    Parameters
    ----------
    dnfr : float
        Structural reorganization pressure ``ΔNFR`` at the node.
    depi : float, optional
        Structural change rate ``dEPI`` at the node (default ``0`` for static
        scalar fields that carry no explicit time derivative).
    """
    return 1.0 / (1.0 + abs(dnfr) + abs(depi))

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

    This is the ONE deep structural invariant that recurs fractally across
    TNFR: a relaxed graph node (``ΔNFR → 0``), a structural prime
    (``ΔNFR_arith = 0``) and a noble-gas element (``ΔNFR_chem = 0``) are the
    *same* fixed point read out under a domain-specific ``ΔNFR``. The tolerance
    is a per-domain numerical scale (``1e-3`` for the graph dynamics; ``1e-12``
    for exact integer arithmetic), **not** a different logic.

    Particles read this fixed point *directly* as a topological winding of the
    phase field; the arithmetic and chemical read-outs are *symbolic* -- the
    per-node ΔNFR consumes the domain data -- while number theory additionally
    carries a genuinely emergent *spectral* read-out (the Paley/residue Fiedler
    gap; theory §9.5). One fixed point, a spectrum of emergence.

    Parameters
    ----------
    dnfr : float
        Structural reorganization pressure ``ΔNFR``.
    depi : float, optional
        Structural change rate ``dEPI`` (default ``0``).
    eps_dnfr, eps_depi : float, optional
        Equilibrium tolerances (default: the canonical ``EPS_*_STABLE``).
    """
    return abs(dnfr) <= eps_dnfr and abs(depi) <= eps_depi

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

    if np is not None:
        dnfr_mean = float(np.mean(np.abs(dnfr_values)))
        depi_mean = float(np.mean(np.abs(depi_values)))
    else:
        dnfr_sum, depi_sum = kahan_sum_nd(
            ((abs(d), abs(e)) for d, e in zip(dnfr_values, depi_values)),
            dims=2,
        )
        dnfr_mean = dnfr_sum / count
        depi_mean = depi_sum / count

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

    return multi_recompute_abs_max(G, {"dnfr_max": ALIAS_DNFR, "accel_max": ALIAS_D2EPI})

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
