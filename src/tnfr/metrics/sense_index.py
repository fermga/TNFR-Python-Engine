"""Sense index helpers."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Iterable, Mapping

from ..alias import get_attr, set_attr
from ..constants import get_aliases
from ..helpers.numeric import angle_diff, clamp01
from ..types import GraphLike
from ..utils import (
    edge_version_cache,
    get_numpy,
    normalize_weights,
    stable_json,
)
from .common import (
    _get_vf_dnfr_max,
    ensure_neighbors_map,
    merge_graph_weights,
)
from .trig import neighbor_phase_mean_list
from .trig_cache import get_trig_cache

ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_SI = get_aliases("SI")

PHASE_DISPERSION_KEY = "dSi_dphase_disp"
_VALID_SENSITIVITY_KEYS = frozenset(
    {"dSi_dvf_norm", PHASE_DISPERSION_KEY, "dSi_ddnfr_norm"}
)
__all__ = ("get_Si_weights", "compute_Si_node", "compute_Si")


def _normalise_si_sensitivity_mapping(
    mapping: Mapping[str, float], *, warn: bool
) -> dict[str, float]:
    """Preserve structural sensitivities compatible with the Si operator.

    Parameters
    ----------
    mapping : Mapping[str, float]
        Mapping of raw sensitivity weights keyed by structural derivatives.
    warn : bool
        Compatibility flag kept for trace helpers. It is not used directly but
        retained so upstream logging keeps a consistent signature.

    Returns
    -------
    dict[str, float]
        Sanitised mapping containing only the supported sensitivity keys.

    Raises
    ------
    ValueError
        If the mapping defines keys outside of the supported sensitivity set.

    Examples
    --------
    >>> _normalise_si_sensitivity_mapping({"dSi_dvf_norm": 1.0}, warn=False)
    {'dSi_dvf_norm': 1.0}
    >>> _normalise_si_sensitivity_mapping({"unknown": 1.0}, warn=False)
    Traceback (most recent call last):
        ...
    ValueError: Si sensitivity mappings accept only {dSi_ddnfr_norm, dSi_dphase_disp, dSi_dvf_norm}; unexpected key(s): unknown
    """

    normalised = dict(mapping)
    _ = warn  # kept for API compatibility with trace helpers
    unexpected = sorted(k for k in normalised if k not in _VALID_SENSITIVITY_KEYS)
    if unexpected:
        allowed = ", ".join(sorted(_VALID_SENSITIVITY_KEYS))
        received = ", ".join(unexpected)
        raise ValueError(
            "Si sensitivity mappings accept only {%s}; unexpected key(s): %s"
            % (allowed, received)
        )
    return normalised


def _cache_weights(G: GraphLike) -> tuple[float, float, float]:
    """Normalise and persist Si weights attached to the graph coherence.

    Parameters
    ----------
    G : GraphLike
        Graph structure whose global Si sensitivities must be harmonised.

    Returns
    -------
    tuple[float, float, float]
        Ordered tuple ``(alpha, beta, gamma)`` with normalised Si weights.

    Raises
    ------
    ValueError
        Propagated if the graph stores unsupported sensitivity keys.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.graph["SI_WEIGHTS"] = {"alpha": 0.2, "beta": 0.5, "gamma": 0.3}
    >>> tuple(round(v, 2) for v in _cache_weights(G))
    (0.2, 0.5, 0.3)
    """

    w = merge_graph_weights(G, "SI_WEIGHTS")
    cfg_key = stable_json(w)

    existing = G.graph.get("_Si_sensitivity")
    if isinstance(existing, Mapping):
        migrated = _normalise_si_sensitivity_mapping(existing, warn=True)
        if migrated != existing:
            G.graph["_Si_sensitivity"] = migrated

    def builder() -> tuple[float, float, float]:
        weights = normalize_weights(w, ("alpha", "beta", "gamma"), default=0.0)
        alpha = weights["alpha"]
        beta = weights["beta"]
        gamma = weights["gamma"]
        G.graph["_Si_weights"] = weights
        G.graph["_Si_weights_key"] = cfg_key
        G.graph["_Si_sensitivity"] = {
            "dSi_dvf_norm": alpha,
            PHASE_DISPERSION_KEY: -beta,
            "dSi_ddnfr_norm": -gamma,
        }
        return alpha, beta, gamma

    return edge_version_cache(G, ("_Si_weights", cfg_key), builder)


def get_Si_weights(G: GraphLike) -> tuple[float, float, float]:
    """Expose the normalised Si weights associated with ``G``.

    Parameters
    ----------
    G : GraphLike
        Graph that carries optional ``SI_WEIGHTS`` metadata.

    Returns
    -------
    tuple[float, float, float]
        The ``(alpha, beta, gamma)`` weights after normalisation.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> get_Si_weights(G)
    (0.0, 0.0, 0.0)
    """

    return _cache_weights(G)


def compute_Si_node(
    n: Any,
    nd: dict[str, Any],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    vfmax: float,
    dnfrmax: float,
    phase_dispersion: float | None = None,
    inplace: bool,
    **kwargs: Any,
) -> float:
    """Evaluate the Si contribution of a single node within its resonance.

    Parameters
    ----------
    n : Any
        Node identifier whose structural perception is computed.
    nd : dict[str, Any]
        Mutable node attributes containing cached structural magnitudes.
    alpha : float
        Normalised weight applied to the node's structural frequency.
    beta : float
        Normalised weight applied to the phase alignment term.
    gamma : float
        Normalised weight applied to the ΔNFR attenuation term.
    vfmax : float
        Maximum structural frequency used for normalisation.
    dnfrmax : float
        Maximum |ΔNFR| used for normalisation.
    phase_dispersion : float, optional
        Phase dispersion ratio in ``[0, 1]`` for the node against its
        neighbours. The value must be supplied by the caller.
    inplace : bool
        Whether to write the resulting Si back to ``nd``.
    **kwargs : Any
        Additional keyword arguments are not accepted and will raise.

    Returns
    -------
    float
        The clamped Si value in ``[0, 1]``.

    Raises
    ------
    TypeError
        If ``phase_dispersion`` is missing or unsupported keyword arguments
        are provided.

    Examples
    --------
    >>> nd = {"nu_f": 1.0, "delta_nfr": 0.1}
    >>> compute_Si_node(
    ...     "n0",
    ...     nd,
    ...     alpha=0.4,
    ...     beta=0.3,
    ...     gamma=0.3,
    ...     vfmax=1.0,
    ...     dnfrmax=1.0,
    ...     phase_dispersion=0.2,
    ...     inplace=False,
    ... )
    0.91
    """

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    if phase_dispersion is None:
        raise TypeError("Missing required keyword-only argument: 'phase_dispersion'")

    vf = get_attr(nd, ALIAS_VF, 0.0)
    vf_norm = clamp01(abs(vf) / vfmax)

    dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
    dnfr_norm = clamp01(abs(dnfr) / dnfrmax)

    Si = alpha * vf_norm + beta * (1.0 - phase_dispersion) + gamma * (1.0 - dnfr_norm)
    Si = clamp01(Si)
    if inplace:
        set_attr(nd, ALIAS_SI, Si)
    return Si


def _coerce_jobs(raw_jobs: Any | None) -> int | None:
    """Normalise ``n_jobs`` values coming from user configuration.

    Parameters
    ----------
    raw_jobs : Any or None
        Raw configuration value that specifies how many worker processes
        should participate in the Si computation. Values are accepted even
        when provided as strings so long as they can be coerced to integers.

    Returns
    -------
    int or None
        A strictly positive integer describing the requested level of
        parallelism, or ``None`` when the configuration is absent or
        considered invalid. Returning ``None`` allows the caller to fall back
        to the single-process implementation that preserves ΔNFR sampling
        determinism.

    Examples
    --------
    >>> _coerce_jobs("4")
    4
    >>> _coerce_jobs(-2) is None
    True
    >>> _coerce_jobs("not-an-int") is None
    True
    >>> _coerce_jobs(None) is None
    True
    
    Notes
    -----
    Invalid inputs—such as non-integer values or non-positive integers—are
    mapped to ``None`` so that Si calculations reuse the deterministic code
    path. This guarantees reproducible ΔNFR sampling regardless of user
    configuration quirks.
    """

    try:
        jobs = None if raw_jobs is None else int(raw_jobs)
    except (TypeError, ValueError):
        return None
    if jobs is not None and jobs <= 0:
        return None
    return jobs


def _compute_si_python_chunk(
    chunk: Iterable[tuple[Any, tuple[Any, ...], float, float, float]],
    *,
    cos_th: dict[Any, float],
    sin_th: dict[Any, float],
    alpha: float,
    beta: float,
    gamma: float,
    vfmax: float,
    dnfrmax: float,
) -> dict[Any, float]:
    """Compute Si contributions for a chunk of nodes without NumPy.

    Parameters
    ----------
    chunk : Iterable[tuple[Any, tuple[Any, ...], float, float, float]]
        Iterable of node payloads ``(node, neighbors, theta, vf, dnfr)``.
    cos_th : dict[Any, float]
        Cached cosine values keyed by node identifiers.
    sin_th : dict[Any, float]
        Cached sine values keyed by node identifiers.
    alpha : float
        Normalised weight for structural frequency.
    beta : float
        Normalised weight for phase dispersion.
    gamma : float
        Normalised weight for ΔNFR dispersion.
    vfmax : float
        Maximum |νf| reference for normalisation.
    dnfrmax : float
        Maximum |ΔNFR| reference for normalisation.

    Returns
    -------
    dict[Any, float]
        Mapping of node identifiers to their clamped Si values.

    Examples
    --------
    >>> _compute_si_python_chunk(
    ...     [("n0", ("n1",), 0.0, 0.5, 0.1)],
    ...     cos_th={"n1": 1.0},
    ...     sin_th={"n1": 0.0},
    ...     alpha=0.5,
    ...     beta=0.3,
    ...     gamma=0.2,
    ...     vfmax=1.0,
    ...     dnfrmax=1.0,
    ... )
    {'n0': 0.73}
    """

    results: dict[Any, float] = {}
    for n, neigh, theta, vf, dnfr in chunk:
        th_bar = neighbor_phase_mean_list(
            neigh, cos_th=cos_th, sin_th=sin_th, np=None, fallback=theta
        )
        phase_dispersion = abs(angle_diff(theta, th_bar)) / math.pi
        vf_norm = clamp01(abs(vf) / vfmax)
        dnfr_norm = clamp01(abs(dnfr) / dnfrmax)
        Si = (
            alpha * vf_norm
            + beta * (1.0 - phase_dispersion)
            + gamma * (1.0 - dnfr_norm)
        )
        results[n] = clamp01(Si)
    return results


def compute_Si(
    G: GraphLike,
    *,
    inplace: bool = True,
    n_jobs: int | None = None,
) -> dict[Any, float]:
    """Compute the Si metric for each node and optionally persist it.

    Si (sense index) quantifies how effectively a node sustains coherent
    reorganisation within the TNFR triad. The metric aggregates three
    structural contributions: the node's structural frequency (weighted by
    ``alpha``), its phase alignment with neighbours (weighted by ``beta``),
    and the attenuation of disruptive ΔNFR (weighted by ``gamma``). The
    weights therefore bias Si towards faster reorganisation, tighter phase
    coupling, or reduced dissonance respectively, depending on the scenario.

    Parameters
    ----------
    G : GraphLike
        Graph that exposes ``νf`` (structural frequency), ``ΔNFR`` and phase
        attributes for each node.
    inplace : bool, default: True
        If ``True`` the resulting Si values are written back to ``G``.
    n_jobs : int or None, optional
        Maximum number of worker processes for the pure-Python fallback. Use
        ``None`` to auto-detect the configuration.

    Returns
    -------
    dict[Any, float]
        Mapping from node identifiers to their Si scores.

    Raises
    ------
    ValueError
        Propagated if graph-level sensitivity settings include unsupported
        keys or invalid weights.

    Examples
    --------
    Build a minimal resonance graph with two nodes sharing a phase-locked
    edge. The structural weights bias the result towards phase coherence.

    >>> import networkx as nx
    >>> from tnfr.metrics.sense_index import compute_Si
    >>> G = nx.Graph()
    >>> G.add_edge("a", "b")
    >>> G.nodes["a"].update({"nu_f": 0.8, "delta_nfr": 0.2, "phase": 0.0})
    >>> G.nodes["b"].update({"nu_f": 0.6, "delta_nfr": 0.1, "phase": 0.1})
    >>> G.graph["SI_WEIGHTS"] = {"alpha": 0.3, "beta": 0.5, "gamma": 0.2}
    >>> {k: round(v, 3) for k, v in compute_Si(G, inplace=False).items()}
    {'a': 0.784, 'b': 0.809}
    """

    neighbors = ensure_neighbors_map(G)
    alpha, beta, gamma = get_Si_weights(G)
    vfmax, dnfrmax = _get_vf_dnfr_max(G)

    np = get_numpy()
    trig = get_trig_cache(G, np=np)
    cos_th, sin_th, thetas = trig.cos, trig.sin, trig.theta

    pm_fn = partial(neighbor_phase_mean_list, cos_th=cos_th, sin_th=sin_th, np=np)

    if n_jobs is None:
        n_jobs = _coerce_jobs(G.graph.get("SI_N_JOBS"))
    else:
        n_jobs = _coerce_jobs(n_jobs)

    supports_vector = (
        np is not None
        and hasattr(np, "ndarray")
        and all(hasattr(np, attr) for attr in ("fromiter", "abs", "clip", "remainder"))
    )

    nodes_data = list(G.nodes(data=True))
    if not nodes_data:
        return {}

    if supports_vector:
        node_ids: list[Any] = []
        theta_vals: list[float] = []
        mean_vals: list[float] = []
        vf_vals: list[float] = []
        dnfr_vals: list[float] = []
        for n, nd in nodes_data:
            theta = thetas.get(n, 0.0)
            neigh = neighbors[n]
            node_ids.append(n)
            theta_vals.append(theta)
            mean_vals.append(pm_fn(neigh, fallback=theta))
            vf_vals.append(get_attr(nd, ALIAS_VF, 0.0))
            dnfr_vals.append(get_attr(nd, ALIAS_DNFR, 0.0))

        count = len(node_ids)
        theta_arr = np.fromiter(theta_vals, dtype=float, count=count)
        mean_arr = np.fromiter(mean_vals, dtype=float, count=count)
        diff = np.remainder(theta_arr - mean_arr + math.pi, math.tau) - math.pi
        phase_dispersion_arr = np.abs(diff) / math.pi

        vf_arr = np.fromiter(vf_vals, dtype=float, count=count)
        dnfr_arr = np.fromiter(dnfr_vals, dtype=float, count=count)
        vf_norm = np.clip(np.abs(vf_arr) / vfmax, 0.0, 1.0)
        dnfr_norm = np.clip(np.abs(dnfr_arr) / dnfrmax, 0.0, 1.0)

        si_arr = np.clip(
            alpha * vf_norm
            + beta * (1.0 - phase_dispersion_arr)
            + gamma * (1.0 - dnfr_norm),
            0.0,
            1.0,
        )

        out = {node_ids[i]: float(si_arr[i]) for i in range(count)}
    else:
        out: dict[Any, float] = {}
        if n_jobs is not None and n_jobs > 1:
            node_payload: list[tuple[Any, tuple[Any, ...], float, float, float]] = []
            for n, nd in nodes_data:
                theta = thetas.get(n, 0.0)
                vf = float(get_attr(nd, ALIAS_VF, 0.0))
                dnfr = float(get_attr(nd, ALIAS_DNFR, 0.0))
                neigh = neighbors[n]
                node_payload.append((n, tuple(neigh), theta, vf, dnfr))

            if node_payload:
                chunk_size = math.ceil(len(node_payload) / n_jobs)
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [
                        executor.submit(
                            _compute_si_python_chunk,
                            node_payload[idx : idx + chunk_size],
                            cos_th=cos_th,
                            sin_th=sin_th,
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
                            vfmax=vfmax,
                            dnfrmax=dnfrmax,
                        )
                        for idx in range(0, len(node_payload), chunk_size)
                    ]
                    for future in futures:
                        out.update(future.result())
        else:
            for n, nd in nodes_data:
                theta = thetas.get(n, 0.0)
                neigh = neighbors[n]
                th_bar = pm_fn(neigh, fallback=theta)
                phase_dispersion = abs(angle_diff(theta, th_bar)) / math.pi
                out[n] = compute_Si_node(
                    n,
                    nd,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    vfmax=vfmax,
                    dnfrmax=dnfrmax,
                    phase_dispersion=phase_dispersion,
                    inplace=False,
                )

    if inplace:
        for n, value in out.items():
            set_attr(G.nodes[n], ALIAS_SI, value)
    return out
