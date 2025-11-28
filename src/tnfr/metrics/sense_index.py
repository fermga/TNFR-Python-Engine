r"""Sense Index computation for TNFR networks.

The **Sense Index** (:math:`\text{Si}`) quantifies a node's capacity for stable
structural reorganization. It blends three structural signals: frequency :math:`\nu_f`
(reorganization rate), phase coupling :math:`\theta` (network synchrony), and
reorganization pressure :math:`\Delta\text{NFR}`.

Mathematical Foundation
-----------------------

The Sense Index is defined as a weighted combination:

.. math::
    \text{Si} = \alpha \cdot \nu_{f,\text{norm}}
              + \beta \cdot (1 - \text{disp}_\theta)
              + \gamma \cdot (1 - |\Delta\text{NFR}|_{\text{norm}})

**Component definitions**:

1. **Normalized frequency** :math:`\nu_{f,\text{norm}}`:

   .. math::
       \nu_{f,\text{norm}} = \frac{|\nu_f|}{\nu_{f,\max}}

   Measures how fast a node reorganizes relative to network maximum.
   Range: [0, 1] where 1 = maximum reorganization rate.

2. **Phase dispersion** :math:`\text{disp}_\theta`:

   .. math::
       \text{disp}_\theta = \frac{|\theta - \bar{\theta}|}{\pi}

   where :math:`\bar{\theta}` is the circular mean of neighbor phases:

   .. math::
       \bar{\theta} = \text{atan2}\left(\sum_{j \in N(i)} \sin\theta_j, \sum_{j \in N(i)} \cos\theta_j\right)

   Measures phase misalignment with neighbors.
   Range: [0, 1] where 0 = perfect synchrony, 1 = maximum dispersion.

3. **Normalized reorganization magnitude** :math:`|\Delta\text{NFR}|_{\text{norm}}`:

   .. math::
       |\Delta\text{NFR}|_{\text{norm}} = \frac{|\Delta\text{NFR}|}{\Delta\text{NFR}_{\max}}

   Measures structural pressure relative to network maximum.
   Range: [0, 1] where 0 = equilibrium, 1 = maximum pressure.

**Structural weights**:

- :math:`\alpha`: Frequency weight (default: 0.4) - emphasizes reorganization capacity
- :math:`\beta`: Phase weight (default: 0.3) - emphasizes network synchrony
- :math:`\gamma`: ΔNFR weight (default: 0.3) - emphasizes pressure damping
- Constraint: :math:`\alpha + \beta + \gamma = 1`

**Final clamping**: :math:`\text{Si}_{\text{final}} = \max(0, \min(1, \text{Si}))`

Physical Interpretation
------------------------

**High Si (> 0.7)**:
- Node reorganizes efficiently (:math:`\nu_f` high)
- Stays synchronized with network (:math:`\text{disp}_\theta` low)
- Experiences manageable pressure (:math:`|\Delta\text{NFR}|` low)
- **Implication**: Stable, well-integrated node

**Low Si (< 0.3)**:
- Slow reorganization OR high phase dispersion OR high pressure
- **Implication**: Risk of structural instability or network decoupling

**Moderate Si (0.3-0.7)**:
- Trade-offs between frequency, synchrony, and pressure
- **Implication**: Balanced state, monitor for bifurcation

Implementation Map
------------------

**Core Functions**:

- :func:`compute_Si` : Network-wide Si computation (vectorized when possible)
- :func:`compute_Si_node` : Single-node Si calculation
- :func:`get_Si_weights` : Extract or default Si weights from graph

**Helper Functions**:

- :func:`_compute_si_python_chunk` : Parallel worker for chunked computation
- :func:`_SiStructuralCache` : Cache for aligned :math:`\nu_f` and :math:`\Delta\text{NFR}` arrays

**Performance**:

- Uses NumPy vectorization for networks with >10 nodes
- Parallel computation for networks with >1000 nodes
- Trigonometric caching to avoid redundant phase calculations

Theoretical References
----------------------

See the following for complete derivation:

- **Mathematical Foundations**: `docs/source/theory/mathematical_foundations.md`
- **Worked Example**: `docs/source/examples/worked_examples.md` Example 1 (full walkthrough)
- **Style Guide**: `docs/source/style_guide.md` for notation conventions

Examples
--------

**Basic network-wide computation**:

>>> import networkx as nx
>>> from tnfr.metrics.sense_index import compute_Si
>>> G = nx.Graph()
>>> G.add_edge("sensor", "relay")
>>> G.nodes["sensor"].update({"nu_f": 0.9, "delta_nfr": 0.3, "phase": 0.0})
>>> G.nodes["relay"].update({"nu_f": 0.4, "delta_nfr": 0.05, "phase": 0.1})
>>> G.graph["SI_WEIGHTS"] = {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
>>> result = compute_Si(G, inplace=False)
>>> round(result["sensor"], 3), round(result["relay"], 3)
(0.767, 0.857)

The heavier :math:`\alpha` weight privileges the sensor's fast :math:`\nu_f` even
though it suffers larger :math:`\Delta\text{NFR}`. The relay keeps Si high thanks
to calmer :math:`\Delta\text{NFR}` despite slower frequency.

**Single-node computation**:

>>> from tnfr.metrics.sense_index import compute_Si_node
>>> node_attrs = {
...     "nu_f": 0.8,
...     "delta_nfr": 0.2,
...     "phase": 0.5,
...     "neighbors": [{"phase": 0.4}, {"phase": 0.6}]
... }
>>> Si = compute_Si_node(
...     "node_id",
...     node_attrs,
...     alpha=0.4, beta=0.3, gamma=0.3,
...     vfmax=1.0, dnfrmax=1.0,
...     phase_dispersion=0.0,  # Already computed
...     inplace=False
... )
>>> 0.8 < Si < 0.9  # High stability
True

**In-place update**:

>>> G = nx.Graph()
>>> G.add_node("a", nu_f=0.8, delta_nfr=0.2, phase=0.0)
>>> compute_Si(G, inplace=True)  # Writes to G.nodes[n]['Si']
>>> "Si" in G.nodes["a"]
True

See Also
--------

coherence.compute_coherence : Total network coherence :math:`C(t)`
coherence.coherence_matrix : Coherence operator approximation :math:`W \approx \hat{C}`
observers.kuramoto_order : Kuramoto order parameter for phase synchrony
observers.phase_sync : Phase synchronization metrics

Notes
-----

**Sensitivity analysis**:

The module can compute partial derivatives :math:`\frac{\partial \text{Si}}{\partial x}`
for :math:`x \in \{\nu_{f,\text{norm}}, \text{disp}_\theta, |\Delta\text{NFR}|_{\text{norm}}\}`
when `return_sensitivities=True` is passed to `compute_Si`.

**Edge cases**:

- If a node has no neighbors, :math:`\bar{\theta} = \theta` (zero dispersion)
- If :math:`\nu_{f,\max} = 0`, normalization defaults to 0 (frozen network)
- If :math:`\Delta\text{NFR}_{\max} = 0`, normalization defaults to 0 (equilibrium network)
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, cast

from ..alias import get_attr, set_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_SI, ALIAS_VF
from ..utils import angle_diff, angle_diff_array, clamp01
from ..types import GraphLike, NodeAttrMap
from ..utils import (
    edge_version_cache,
    get_numpy,
    normalize_weights,
    resolve_chunk_size,
    stable_json,
)
from .buffer_cache import ensure_numpy_buffers
from .common import (
    _coerce_jobs,
    _get_vf_dnfr_max,
    ensure_neighbors_map,
    merge_graph_weights,
)
from .trig import neighbor_phase_mean_bulk, neighbor_phase_mean_list
from .trig_cache import get_trig_cache

PHASE_DISPERSION_KEY = "dSi_dphase_disp"
_SI_APPROX_BYTES_PER_NODE = 64
_VALID_SENSITIVITY_KEYS = frozenset({"dSi_dvf_norm", PHASE_DISPERSION_KEY, "dSi_ddnfr_norm"})
__all__ = ("get_Si_weights", "compute_Si_node", "compute_Si")


class _SiStructuralCache:
    """Cache aligned ``νf`` and ``ΔNFR`` arrays for vectorised Si."""

    __slots__ = ("node_ids", "vf_values", "dnfr_values", "vf_snapshot", "dnfr_snapshot")

    def __init__(self, node_ids: tuple[Any, ...]):
        self.node_ids = node_ids
        self.vf_values: Any | None = None
        self.dnfr_values: Any | None = None
        self.vf_snapshot: list[float] = []
        self.dnfr_snapshot: list[float] = []

    def rebuild(
        self,
        node_ids: Iterable[Any],
        node_data: Mapping[Any, NodeAttrMap],
        *,
        np: Any,
    ) -> tuple[Any, Any]:
        node_tuple = tuple(node_ids)
        count = len(node_tuple)
        if count == 0:
            self.node_ids = node_tuple
            self.vf_values = np.zeros(0, dtype=float)
            self.dnfr_values = np.zeros(0, dtype=float)
            self.vf_snapshot = []
            self.dnfr_snapshot = []
            return self.vf_values, self.dnfr_values

        vf_arr = np.fromiter(
            (float(get_attr(node_data[n], ALIAS_VF, 0.0)) for n in node_tuple),
            dtype=float,
            count=count,
        )
        dnfr_arr = np.fromiter(
            (float(get_attr(node_data[n], ALIAS_DNFR, 0.0)) for n in node_tuple),
            dtype=float,
            count=count,
        )

        self.node_ids = node_tuple
        self.vf_values = vf_arr
        self.dnfr_values = dnfr_arr
        self.vf_snapshot = [float(value) for value in vf_arr]
        self.dnfr_snapshot = [float(value) for value in dnfr_arr]
        return self.vf_values, self.dnfr_values

    def ensure_current(
        self,
        node_ids: Iterable[Any],
        node_data: Mapping[Any, NodeAttrMap],
        *,
        np: Any,
    ) -> tuple[Any, Any]:
        node_tuple = tuple(node_ids)
        if node_tuple != self.node_ids:
            return self.rebuild(node_tuple, node_data, np=np)

        for idx, node in enumerate(node_tuple):
            nd = node_data[node]
            vf = float(get_attr(nd, ALIAS_VF, 0.0))
            if vf != self.vf_snapshot[idx]:
                return self.rebuild(node_tuple, node_data, np=np)
            dnfr = float(get_attr(nd, ALIAS_DNFR, 0.0))
            if dnfr != self.dnfr_snapshot[idx]:
                return self.rebuild(node_tuple, node_data, np=np)

        return self.vf_values, self.dnfr_values


def _build_structural_cache(
    node_ids: Iterable[Any],
    node_data: Mapping[Any, NodeAttrMap],
    *,
    np: Any,
) -> _SiStructuralCache:
    cache = _SiStructuralCache(tuple(node_ids))
    cache.rebuild(node_ids, node_data, np=np)
    return cache


def _ensure_structural_arrays(
    G: GraphLike,
    node_ids: Iterable[Any],
    node_data: Mapping[Any, NodeAttrMap],
    *,
    np: Any,
) -> tuple[Any, Any]:
    node_key = tuple(node_ids)

    def builder() -> _SiStructuralCache:
        return _build_structural_cache(node_key, node_data, np=np)

    cache = edge_version_cache(G, ("_si_structural", node_key), builder)
    return cache.ensure_current(node_key, node_data, np=np)


def _ensure_si_buffers(
    G: GraphLike,
    *,
    count: int,
    np: Any,
) -> tuple[Any, Any, Any]:
    """Return reusable NumPy buffers sized for ``count`` nodes.

    Allocates three computation buffers used in Si vectorization:
    1. phase_dispersion: Phase alignment metric per node
    2. raw_si: Intermediate Si values before clamping
    3. si_values: Final Si values after normalization

    These buffers are reused across computation steps to minimize allocation
    overhead in the hot path. Cache key: ``("_si_buffers", count, 3)``
    """
    return ensure_numpy_buffers(G, key_prefix="_si_buffers", count=count, buffer_count=3, np=np)


def _ensure_chunk_workspace(
    G: GraphLike,
    *,
    mask_count: int,
    np: Any,
) -> tuple[Any, Any]:
    """Return reusable scratch buffers sized to the masked neighbours.

    Allocates workspace for chunked phase dispersion computation:
    1. chunk_theta: Theta values for current chunk
    2. chunk_values: Intermediate values for current chunk

    Used when processing large neighbor sets in chunks to manage memory.
    Cache key: ``("_si_chunk_workspace", mask_count, 2)``
    """
    return ensure_numpy_buffers(
        G, key_prefix="_si_chunk_workspace", count=mask_count, buffer_count=2, np=np
    )


def _ensure_neighbor_bulk_buffers(
    G: GraphLike,
    *,
    count: int,
    np: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return reusable buffers for bulk neighbour phase aggregation.

    Allocates five buffers for neighbor accumulation in vectorized Si:
    1. neighbor_cos_sum: Sum of cos(theta) from neighbors
    2. neighbor_sin_sum: Sum of sin(theta) from neighbors
    3. neighbor_counts: Number of neighbors per node
    4. mean_cos_buf: Mean cos(theta) per node
    5. mean_sin_buf: Mean sin(theta) per node

    These enable efficient neighbor phase mean computation without Python loops.
    Cache key: ``("_si_neighbor_buffers", count, 5)``
    """
    return ensure_numpy_buffers(
        G, key_prefix="_si_neighbor_buffers", count=count, buffer_count=5, np=np
    )


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
            "Si sensitivity mappings accept only {%s}; unexpected key(s): %s" % (allowed, received)
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
    """Evaluate how a node's structure tilts Si within its local resonance.

    Parameters
    ----------
    n : Any
        Node identifier whose structural perception is computed.
    nd : dict[str, Any]
        Mutable node attributes containing cached structural magnitudes.
    alpha : float
        Normalised weight applied to the node's structural frequency, boosting
        Si when the node reorganises faster than the network baseline.
    beta : float
        Normalised weight applied to the phase alignment term so that tighter
        synchrony raises the index.
    gamma : float
        Normalised weight applied to the ΔNFR attenuation term, rewarding nodes
        that keep internal turbulence under control.
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
    """Propagate Si contributions for a node chunk using pure Python.

    The fallback keeps the νf/phase/ΔNFR balance explicit so that structural
    effects remain traceable even without vectorised support.

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
        Si = alpha * vf_norm + beta * (1.0 - phase_dispersion) + gamma * (1.0 - dnfr_norm)
        results[n] = clamp01(Si)
    return results


def _iter_python_payload_chunks(
    nodes_data: Iterable[tuple[Any, NodeAttrMap]],
    *,
    neighbors: Mapping[Any, Iterable[Any]],
    thetas: Mapping[Any, float],
    chunk_size: int,
) -> Iterator[tuple[tuple[Any, tuple[Any, ...], float, float, float], ...]]:
    """Yield lazily constructed Si payload chunks for the Python fallback.

    Each batch keeps the structural triad explicit—θ, νf, and ΔNFR—so that the
    downstream worker preserves the coherence balance enforced by the Si
    operator.  Streaming prevents a single monolithic buffer that would skew
    memory pressure on dense graphs while still producing deterministic ΔNFR
    sampling. The iterator is consumed lazily by :func:`compute_Si` so that the
    Python fallback can submit and harvest chunk results incrementally, keeping
    both memory usage and profiling telemetry representative of the streamed
    execution.
    """

    if chunk_size <= 0:
        return

    buffer: list[tuple[Any, tuple[Any, ...], float, float, float]] = []
    for node, data in nodes_data:
        theta = thetas.get(node, 0.0)
        vf = float(get_attr(data, ALIAS_VF, 0.0))
        dnfr = float(get_attr(data, ALIAS_DNFR, 0.0))
        neigh = tuple(neighbors[node])
        buffer.append((node, neigh, theta, vf, dnfr))
        if len(buffer) >= chunk_size:
            yield tuple(buffer)
            buffer.clear()

    if buffer:
        yield tuple(buffer)


def compute_Si(
    G: GraphLike,
    *,
    inplace: bool = True,
    n_jobs: int | None = None,
    chunk_size: int | None = None,
    profile: MutableMapping[str, Any] | None = None,
) -> dict[Any, float] | Any:
    """Compute the Si metric for each node by integrating structural drivers.

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
    chunk_size : int or None, optional
        Maximum number of nodes processed per batch when building the Si
        mapping. ``None`` derives a safe value from the node count, the
        available CPUs, and conservative memory heuristics. Non-positive values
        fall back to the automatic mode. Graphs may also provide a default via
        ``G.graph["SI_CHUNK_SIZE"]``.
    profile : MutableMapping[str, Any] or None, optional
        Mutable mapping that aggregates wall-clock durations for the internal
        stages of the computation. The mapping receives the keys
        ``"cache_rebuild"``, ``"neighbor_phase_mean_bulk"``,
        ``"normalize_clamp"`` and ``"inplace_write"`` accumulating seconds for
        each step, plus ``"path"`` describing whether the vectorised (NumPy)
        or fallback implementation executed the call. When the Python fallback
        streams chunk execution, ``"fallback_chunks"`` records how many payload
        batches completed. Reusing the mapping across invocations accumulates
        the timings and chunk counts.

    Returns
    -------
    dict[Any, float] | numpy.ndarray
        Mapping from node identifiers to their Si scores when ``inplace`` is
        ``False``. When ``inplace`` is ``True`` and the NumPy accelerated path
        is available the function updates the graph in place and returns the
        vector of Si values as a :class:`numpy.ndarray`. The pure-Python
        fallback always returns a mapping for compatibility.

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

    if profile is not None:
        for key in (
            "cache_rebuild",
            "neighbor_phase_mean_bulk",
            "normalize_clamp",
            "inplace_write",
            "fallback_chunks",
        ):
            profile.setdefault(key, 0.0)

        def _profile_start() -> float:
            return perf_counter()

        def _profile_stop(key: str, start: float) -> None:
            profile[key] = float(profile.get(key, 0.0)) + (perf_counter() - start)

        def _profile_mark_path(path: str) -> None:
            profile["path"] = path

    else:

        def _profile_start() -> float:
            return 0.0

        def _profile_stop(key: str, start: float) -> None:
            return None

        def _profile_mark_path(path: str) -> None:
            return None

    neighbors = ensure_neighbors_map(G)
    alpha, beta, gamma = get_Si_weights(G)
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
        and all(
            hasattr(np, attr)
            for attr in (
                "fromiter",
                "abs",
                "clip",
                "remainder",
                "zeros",
                "add",
                "bincount",
                "arctan2",
                "where",
                "divide",
                "errstate",
                "max",
            )
        )
    )

    nodes_view = G.nodes
    nodes_data = list(nodes_view(data=True))
    if not nodes_data:
        return {}

    node_mapping = cast(Mapping[Any, NodeAttrMap], nodes_view)
    node_count = len(nodes_data)

    trig_order = list(getattr(trig, "order", ()))
    node_ids: list[Any]
    node_idx: dict[Any, int]
    using_cache_order = False
    if trig_order and len(trig_order) == node_count:
        node_ids = trig_order
        node_idx = dict(getattr(trig, "index", {}))
        using_cache_order = len(node_idx) == len(node_ids)
        if not using_cache_order:
            node_idx = {n: i for i, n in enumerate(node_ids)}
    else:
        node_ids = [n for n, _ in nodes_data]
        node_idx = {n: i for i, n in enumerate(node_ids)}

    chunk_pref = chunk_size if chunk_size is not None else G.graph.get("SI_CHUNK_SIZE")

    if supports_vector:
        _profile_mark_path("vectorized")
        node_key = tuple(node_ids)
        count = len(node_key)

        cache_theta = getattr(trig, "theta_values", None)
        cache_cos = getattr(trig, "cos_values", None)
        cache_sin = getattr(trig, "sin_values", None)

        trig_index_map = dict(getattr(trig, "index", {}) or {})
        index_arr: Any | None = None
        cached_mask = None
        if trig_index_map and count:
            index_values: list[int] = []
            mask_values: list[bool] = []
            for node in node_ids:
                cached_idx = trig_index_map.get(node)
                if cached_idx is None:
                    index_values.append(-1)
                    mask_values.append(False)
                else:
                    index_values.append(int(cached_idx))
                    mask_values.append(True)
            cached_mask = np.asarray(mask_values, dtype=bool)
            if cached_mask.any():
                index_arr = np.asarray(index_values, dtype=np.intp)
        if cached_mask is None:
            cached_mask = np.zeros(count, dtype=bool)

        def _gather_values(
            cache_values: Any | None, fallback_getter: Callable[[Any], float]
        ) -> Any:
            if (
                index_arr is not None
                and cache_values is not None
                and cached_mask.size
                and cached_mask.any()
            ):
                out = np.empty(count, dtype=float)
                cached_indices = np.nonzero(cached_mask)[0]
                if cached_indices.size:
                    out[cached_indices] = np.take(
                        np.asarray(cache_values, dtype=float), index_arr[cached_indices]
                    )
                missing_indices = np.nonzero(~cached_mask)[0]
                if missing_indices.size:
                    missing_nodes = [node_ids[i] for i in missing_indices]
                    out[missing_indices] = np.fromiter(
                        (fallback_getter(node) for node in missing_nodes),
                        dtype=float,
                        count=missing_indices.size,
                    )
                return out
            return np.fromiter(
                (fallback_getter(node) for node in node_ids),
                dtype=float,
                count=count,
            )

        cache_timer = _profile_start()

        if using_cache_order and cache_theta is not None:
            theta_arr = np.asarray(cache_theta, dtype=float)
        else:
            theta_arr = _gather_values(cache_theta, lambda node: thetas.get(node, 0.0))

        if using_cache_order and cache_cos is not None:
            cos_arr = np.asarray(cache_cos, dtype=float)
        else:
            cos_arr = _gather_values(
                cache_cos,
                lambda node: cos_th.get(node, math.cos(thetas.get(node, 0.0))),
            )

        if using_cache_order and cache_sin is not None:
            sin_arr = np.asarray(cache_sin, dtype=float)
        else:
            sin_arr = _gather_values(
                cache_sin,
                lambda node: sin_th.get(node, math.sin(thetas.get(node, 0.0))),
            )

        cached_edge_src = None
        cached_edge_dst = None
        if using_cache_order:
            cached_edge_src = getattr(trig, "edge_src", None)
            cached_edge_dst = getattr(trig, "edge_dst", None)
            if cached_edge_src is not None and cached_edge_dst is not None:
                cached_edge_src = np.asarray(cached_edge_src, dtype=np.intp)
                cached_edge_dst = np.asarray(cached_edge_dst, dtype=np.intp)
                if cached_edge_src.shape != cached_edge_dst.shape:
                    cached_edge_src = None
                    cached_edge_dst = None

        if cached_edge_src is not None and cached_edge_dst is not None:
            edge_src = cached_edge_src
            edge_dst = cached_edge_dst
        else:

            def _build_edge_arrays() -> tuple[Any, Any]:
                edge_src_list: list[int] = []
                edge_dst_list: list[int] = []
                for node in node_ids:
                    dst_idx = node_idx[node]
                    for neighbor in neighbors[node]:
                        src_idx = node_idx.get(neighbor)
                        if src_idx is None:
                            continue
                        edge_src_list.append(src_idx)
                        edge_dst_list.append(dst_idx)
                src_arr = np.asarray(edge_src_list, dtype=np.intp)
                dst_arr = np.asarray(edge_dst_list, dtype=np.intp)
                return src_arr, dst_arr

            edge_src, edge_dst = edge_version_cache(
                G,
                ("_si_edges", node_key),
                _build_edge_arrays,
            )
            if using_cache_order:
                trig.edge_src = edge_src
                trig.edge_dst = edge_dst

        (
            neighbor_cos_sum,
            neighbor_sin_sum,
            neighbor_counts,
            mean_cos_buf,
            mean_sin_buf,
        ) = _ensure_neighbor_bulk_buffers(
            G,
            count=count,
            np=np,
        )

        vf_arr, dnfr_arr = _ensure_structural_arrays(
            G,
            node_ids,
            node_mapping,
            np=np,
        )
        raw_vfmax = float(np.max(np.abs(vf_arr))) if getattr(vf_arr, "size", 0) else 0.0
        raw_dnfrmax = float(np.max(np.abs(dnfr_arr))) if getattr(dnfr_arr, "size", 0) else 0.0
        G.graph["_vfmax"] = raw_vfmax
        G.graph["_dnfrmax"] = raw_dnfrmax
        vfmax = 1.0 if raw_vfmax == 0.0 else raw_vfmax
        dnfrmax = 1.0 if raw_dnfrmax == 0.0 else raw_dnfrmax

        (
            phase_dispersion,
            raw_si,
            si_values,
        ) = _ensure_si_buffers(
            G,
            count=count,
            np=np,
        )

        _profile_stop("cache_rebuild", cache_timer)

        neighbor_timer = _profile_start()
        mean_theta, has_neighbors = neighbor_phase_mean_bulk(
            edge_src,
            edge_dst,
            cos_values=cos_arr,
            sin_values=sin_arr,
            theta_values=theta_arr,
            node_count=count,
            np=np,
            neighbor_cos_sum=neighbor_cos_sum,
            neighbor_sin_sum=neighbor_sin_sum,
            neighbor_counts=neighbor_counts,
            mean_cos=mean_cos_buf,
            mean_sin=mean_sin_buf,
        )
        _profile_stop("neighbor_phase_mean_bulk", neighbor_timer)
        norm_timer = _profile_start()
        # Reuse the Si buffers as scratch space to avoid transient allocations during
        # the normalization pass and keep the structural buffers coherent with the
        # cached layout.
        np.abs(vf_arr, out=raw_si)
        np.divide(raw_si, vfmax, out=raw_si)
        np.clip(raw_si, 0.0, 1.0, out=raw_si)
        vf_norm = raw_si
        np.abs(dnfr_arr, out=si_values)
        np.divide(si_values, dnfrmax, out=si_values)
        np.clip(si_values, 0.0, 1.0, out=si_values)
        dnfr_norm = si_values
        phase_dispersion.fill(0.0)
        neighbor_mask = np.asarray(has_neighbors, dtype=bool)
        neighbor_count = int(neighbor_mask.sum())
        use_chunked = False
        if neighbor_count:
            effective_chunk = resolve_chunk_size(
                chunk_pref,
                neighbor_count,
                approx_bytes_per_item=_SI_APPROX_BYTES_PER_NODE,
            )
            if effective_chunk <= 0 or effective_chunk >= neighbor_count:
                effective_chunk = neighbor_count
            else:
                use_chunked = True

        if neighbor_count and not use_chunked:
            angle_diff_array(
                theta_arr,
                mean_theta,
                np=np,
                out=phase_dispersion,
                where=neighbor_mask,
            )
            np.abs(phase_dispersion, out=phase_dispersion, where=neighbor_mask)
            np.divide(
                phase_dispersion,
                math.pi,
                out=phase_dispersion,
                where=neighbor_mask,
            )
        elif neighbor_count and use_chunked:
            neighbor_indices = np.nonzero(neighbor_mask)[0]
            chunk_theta, chunk_values = _ensure_chunk_workspace(
                G,
                mask_count=neighbor_count,
                np=np,
            )
            for start in range(0, neighbor_count, effective_chunk):
                end = min(start + effective_chunk, neighbor_count)
                slice_indices = neighbor_indices[start:end]
                chunk_len = end - start
                theta_view = chunk_theta[:chunk_len]
                values_view = chunk_values[:chunk_len]
                np.take(theta_arr, slice_indices, out=theta_view)
                np.take(mean_theta, slice_indices, out=values_view)
                angle_diff_array(theta_view, values_view, np=np, out=values_view)
                np.abs(values_view, out=values_view)
                np.divide(values_view, math.pi, out=values_view)
                phase_dispersion[slice_indices] = values_view
        else:
            np.abs(phase_dispersion, out=phase_dispersion)
            np.divide(
                phase_dispersion,
                math.pi,
                out=phase_dispersion,
                where=neighbor_mask,
            )

        np.multiply(vf_norm, alpha, out=raw_si)
        np.subtract(1.0, phase_dispersion, out=phase_dispersion)
        np.multiply(phase_dispersion, beta, out=phase_dispersion)
        np.add(raw_si, phase_dispersion, out=raw_si)
        np.subtract(1.0, dnfr_norm, out=si_values)
        np.multiply(si_values, gamma, out=si_values)
        np.add(raw_si, si_values, out=raw_si)
        np.clip(raw_si, 0.0, 1.0, out=si_values)

        _profile_stop("normalize_clamp", norm_timer)

        if inplace:
            write_timer = _profile_start()
            for idx, node in enumerate(node_ids):
                set_attr(G.nodes[node], ALIAS_SI, float(si_values[idx]))
            _profile_stop("inplace_write", write_timer)
            return np.copy(si_values)

        return {node: float(value) for node, value in zip(node_ids, si_values)}

    vfmax, dnfrmax = _get_vf_dnfr_max(G)

    out: dict[Any, float] = {}
    _profile_mark_path("fallback")
    if n_jobs is not None and n_jobs > 1:
        node_count = len(nodes_data)
        if node_count:
            effective_chunk = resolve_chunk_size(
                chunk_pref,
                node_count,
                approx_bytes_per_item=_SI_APPROX_BYTES_PER_NODE,
            )
            if effective_chunk <= 0:
                effective_chunk = node_count
            payload_chunks = _iter_python_payload_chunks(
                nodes_data,
                neighbors=neighbors,
                thetas=thetas,
                chunk_size=effective_chunk,
            )
            chunk_count = 0
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                worker = partial(
                    _compute_si_python_chunk,
                    cos_th=cos_th,
                    sin_th=sin_th,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    vfmax=vfmax,
                    dnfrmax=dnfrmax,
                )
                payload_iter = iter(payload_chunks)
                futures: list[Any] = []
                for chunk in payload_iter:
                    futures.append(executor.submit(worker, chunk))
                    if len(futures) >= n_jobs:
                        future = futures.pop(0)
                        chunk_result = future.result()
                        chunk_count += 1
                        out.update(chunk_result)
                for future in futures:
                    chunk_result = future.result()
                    chunk_count += 1
                    out.update(chunk_result)
            if profile is not None:
                profile["fallback_chunks"] = float(profile.get("fallback_chunks", 0.0)) + float(
                    chunk_count
                )
    else:
        for n, nd in nodes_data:
            theta = thetas.get(n, 0.0)
            neigh = neighbors[n]
            th_bar = pm_fn(neigh, fallback=theta)
            phase_dispersion = abs(angle_diff(theta, th_bar)) / math.pi
            norm_timer = _profile_start()
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
            _profile_stop("normalize_clamp", norm_timer)

    if inplace:
        write_timer = _profile_start()
        for n, value in out.items():
            set_attr(G.nodes[n], ALIAS_SI, value)
        _profile_stop("inplace_write", write_timer)
    return out
