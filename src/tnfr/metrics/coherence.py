r"""Coherence metrics for TNFR networks.

This module implements the coherence operator :math:`\hat{C}` and related
metrics for measuring structural stability in resonant fractal networks.

Mathematical Foundation
-----------------------

The **coherence operator** :math:`\hat{C}` is a Hermitian operator on the Hilbert
space :math:`H_{\text{NFR}}` with spectral decomposition:

.. math::
    \hat{C} = \sum_i \lambda_i |\phi_i\rangle\langle\phi_i|

where :math:`\lambda_i \geq 0` are coherence eigenvalues and :math:`|\phi_i\rangle`
are coherence eigenstates (maximally stable configurations).

**Properties**:

1. **Hermiticity**: :math:`\hat{C}^\dagger = \hat{C}` (ensures real eigenvalues)
2. **Positivity**: :math:`\langle\psi|\hat{C}|\psi\rangle \geq 0` (coherence is non-negative)
3. **Boundedness**: :math:`\|\hat{C}\| \leq M` (prevents runaway growth)

In the discrete node basis :math:`\{|i\rangle\}`, matrix elements are approximated:

.. math::
    w_{ij} \approx \langle i | \hat{C} | j \rangle

The **total coherence** is computed as the trace:

.. math::
    C(t) = \text{Tr}(\hat{C}\rho) = \sum_i w_{ii} \rho_i

where :math:`\rho_i` is the density of node :math:`i` (typically uniform: :math:`\rho_i = 1/N`).

Similarity Components
---------------------

Matrix elements :math:`w_{ij}` are computed from four structural similarity components:

.. math::
    w_{ij} = w_{\text{phase}} \cdot s_{\text{phase}}(i,j)
           + w_{\text{EPI}} \cdot s_{\text{EPI}}(i,j)
           + w_{\nu_f} \cdot s_{\nu_f}(i,j)
           + w_{\text{Si}} \cdot s_{\text{Si}}(i,j)

where:

- :math:`s_{\text{phase}}(i,j) = \frac{1}{2}\left(1 + \cos(\theta_i - \theta_j)\right)` : Phase similarity
- :math:`s_{\text{EPI}}(i,j) = 1 - \frac{|\text{EPI}_i - \text{EPI}_j|}{\Delta_{\text{EPI}}}` : Structural form similarity
- :math:`s_{\nu_f}(i,j) = 1 - \frac{|\nu_{f,i} - \nu_{f,j}|}{\Delta_{\nu_f}}` : Frequency similarity
- :math:`s_{\text{Si}}(i,j) = 1 - |\text{Si}_i - \text{Si}_j|` : Stability similarity

and :math:`w_{\text{phase}}, w_{\text{EPI}}, w_{\nu_f}, w_{\text{Si}}` are structural weights
(default: 0.25 each).

Implementation Map
------------------

**Core Functions**:

- :func:`coherence_matrix` : Constructs :math:`W \approx \hat{C}` matrix representation
- :func:`compute_coherence` : Computes :math:`C(t) = \text{Tr}(\hat{C}\rho)` from graph (imported from `.common`)
- :func:`compute_wij_phase_epi_vf_si` : Computes similarity components :math:`(s_{\text{phase}}, s_{\text{EPI}}, s_{\nu_f}, s_{\text{Si}})`

**Helper Functions**:

- :func:`_combine_similarity` : Weighted combination: :math:`w_{ij} = \sum_k w_k s_k`
- :func:`_compute_wij_phase_epi_vf_si_vectorized` : Vectorized computation for all pairs
- :func:`_wij_vectorized` : Builds full matrix with NumPy acceleration
- :func:`_wij_sparse` : Builds sparse matrix for large networks

**Parallel Computation**:

- :func:`_coherence_matrix_parallel` : Multi-process matrix construction
- :func:`_parallel_wij_worker` : Worker function for parallel chunks

Theoretical References
----------------------

See the following for complete mathematical derivation:

- **Mathematical Foundations**: `docs/source/theory/mathematical_foundations.md` §3.1
- **Coherence Operator Theory**: Sections 3.1 (operator definition), 3.1.1 (implementation bridge)
- **Spectral Properties**: Section 3.1 on eigenvalue decomposition
- **Style Guide**: `docs/source/style_guide.md` for notation conventions

Examples
--------

**Basic coherence computation**:

>>> import networkx as nx
>>> from tnfr.metrics.coherence import coherence_matrix
>>> from tnfr.metrics.common import compute_coherence
>>> G = nx.Graph()
>>> G.add_edge("a", "b")
>>> G.nodes["a"].update({"EPI": 0.5, "nu_f": 0.8, "phase": 0.0, "Si": 0.7})
>>> G.nodes["b"].update({"EPI": 0.6, "nu_f": 0.7, "phase": 0.1, "Si": 0.8})
>>> C = compute_coherence(G)
>>> 0 <= C <= 1
True

**Matrix representation**:

>>> nodes, W = coherence_matrix(G)
>>> len(nodes) == 2
True
>>> W.shape == (2, 2)  # Assuming numpy backend
True

**Worked examples** with step-by-step calculations:

See `docs/source/examples/worked_examples.md` Example 2 for detailed coherence
matrix element computation walkthrough.

Notes
-----

- Matrix element computation can use different backends (NumPy, JAX, PyTorch)
- Sparse matrix format is automatically selected for large networks (>1000 nodes)
- Parallel computation is enabled for networks with >500 nodes by default
- Trigonometric values are cached to avoid redundant cos/sin evaluations

See Also
--------

compute_coherence : Total coherence :math:`C(t)` computation
sense_index.compute_Si : Sense Index :math:`\text{Si}` computation
observers.kuramoto_order : Kuramoto order parameter :math:`r`
observers.phase_sync : Phase synchronization metrics
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from types import ModuleType
from typing import Any, MutableMapping, cast

from .._compat import TypeAlias
from ..alias import collect_attr, collect_theta_attr, get_attr, set_attr
from ..utils import CallbackEvent, callback_manager
from ..constants import get_param
from ..constants.aliases import (
    ALIAS_D2VF,
    ALIAS_DNFR,
    ALIAS_DSI,
    ALIAS_DVF,
    ALIAS_DEPI,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_VF,
)
from ..glyph_history import append_metric, ensure_history
from ..utils import clamp01
from ..observers import (
    DEFAULT_GLYPH_LOAD_SPAN,
    DEFAULT_WBAR_SPAN,
    glyph_load,
    kuramoto_order,
    phase_sync,
)
from ..sense import sigma_vector
from ..types import (
    CoherenceMetric,
    FloatArray,
    FloatMatrix,
    GlyphLoadDistribution,
    HistoryState,
    NodeId,
    ParallelWijPayload,
    SigmaVector,
    TNFRGraph,
)
from ..utils import (
    ensure_node_index_map,
    get_logger,
    get_numpy,
    normalize_weights,
    resolve_chunk_size,
)
from .common import compute_coherence, min_max_range
from .trig_cache import compute_theta_trig, get_trig_cache

logger = get_logger(__name__)

GLYPH_LOAD_STABILIZERS_KEY = "glyph_load_stabilizers"


@dataclass
class SimilarityInputs:
    """Similarity inputs and optional trigonometric caches."""

    th_vals: Sequence[float]
    epi_vals: Sequence[float]
    vf_vals: Sequence[float]
    si_vals: Sequence[float]
    cos_vals: Sequence[float] | None = None
    sin_vals: Sequence[float] | None = None


CoherenceMatrixDense = list[list[float]]
CoherenceMatrixSparse = list[tuple[int, int, float]]
CoherenceMatrixPayload = CoherenceMatrixDense | CoherenceMatrixSparse
PhaseSyncWeights: TypeAlias = Sequence[float] | CoherenceMatrixSparse | CoherenceMatrixDense

SimilarityComponents = tuple[float, float, float, float]
VectorizedComponents: TypeAlias = tuple[FloatMatrix, FloatMatrix, FloatMatrix, FloatMatrix]
ScalarOrArray: TypeAlias = float | FloatArray
StabilityChunkArgs = tuple[
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[float | None],
    Sequence[float],
    Sequence[float | None],
    Sequence[float | None],
    float,
    float,
    float,
]
StabilityChunkResult = tuple[
    int,
    int,
    float,
    float,
    list[float],
    list[float],
    list[float],
]

MetricValue: TypeAlias = CoherenceMetric
MetricProvider = Callable[[], MetricValue]
MetricRecord: TypeAlias = tuple[MetricValue | MetricProvider, str]


def _compute_wij_phase_epi_vf_si_vectorized(
    epi: FloatArray,
    vf: FloatArray,
    si: FloatArray,
    cos_th: FloatArray,
    sin_th: FloatArray,
    epi_range: float,
    vf_range: float,
    np: ModuleType,
) -> VectorizedComponents:
    """Vectorized computation of similarity components.

    All parameters are expected to be NumPy arrays already cast to ``float``
    when appropriate. ``epi_range`` and ``vf_range`` are normalized inside the
    function to avoid division by zero.
    """

    epi_range = epi_range if epi_range > 0 else 1.0
    vf_range = vf_range if vf_range > 0 else 1.0
    s_phase = 0.5 * (1.0 + cos_th[:, None] * cos_th[None, :] + sin_th[:, None] * sin_th[None, :])
    s_epi = 1.0 - np.abs(epi[:, None] - epi[None, :]) / epi_range
    s_vf = 1.0 - np.abs(vf[:, None] - vf[None, :]) / vf_range
    s_si = 1.0 - np.abs(si[:, None] - si[None, :])
    return s_phase, s_epi, s_vf, s_si


def compute_wij_phase_epi_vf_si(
    inputs: SimilarityInputs,
    i: int | None = None,
    j: int | None = None,
    *,
    trig: Any | None = None,
    G: TNFRGraph | None = None,
    nodes: Sequence[NodeId] | None = None,
    epi_range: float = 1.0,
    vf_range: float = 1.0,
    np: ModuleType | None = None,
) -> SimilarityComponents | VectorizedComponents:
    r"""Compute structural similarity components for coherence matrix elements.

    Returns four similarity components :math:`(s_{\text{phase}}, s_{\text{EPI}}, s_{\nu_f}, s_{\text{Si}})`
    that approximate coherence operator matrix elements :math:`w_{ij} \approx \langle i | \hat{C} | j \rangle`.

    Mathematical Foundation
    -----------------------

    Each similarity component measures structural resemblance between nodes :math:`i` and :math:`j`
    in a specific dimension:

    **Phase similarity** (synchronization):

    .. math::
        s_{\text{phase}}(i,j) = \frac{1}{2}\left(1 + \cos(\theta_i - \theta_j)\right)

    Range: [0, 1] where 1 = perfect synchrony, 0 = anti-phase.

    **EPI similarity** (structural form):

    .. math::
        s_{\text{EPI}}(i,j) = 1 - \frac{|\text{EPI}_i - \text{EPI}_j|}{\Delta_{\text{EPI}}}

    Range: [0, 1] where 1 = identical structure, 0 = maximally different.

    **Frequency similarity** (reorganization rate):

    .. math::
        s_{\nu_f}(i,j) = 1 - \frac{|\nu_{f,i} - \nu_{f,j}|}{\Delta_{\nu_f}}

    Range: [0, 1] where 1 = matching frequencies.

    **Si similarity** (stability):

    .. math::
        s_{\text{Si}}(i,j) = 1 - |\text{Si}_i - \text{Si}_j|

    Range: [0, 1] where 1 = equal reorganization stability.

    These components are combined via weighted sum to obtain :math:`w_{ij}`:

    .. math::
        w_{ij} = w_{\text{phase}} \cdot s_{\text{phase}} + w_{\text{EPI}} \cdot s_{\text{EPI}}
               + w_{\nu_f} \cdot s_{\nu_f} + w_{\text{Si}} \cdot s_{\text{Si}}

    where :math:`w_{ij} \approx \langle i | \hat{C} | j \rangle` (coherence operator matrix element).

    Parameters
    ----------
    inputs : SimilarityInputs
        Container with structural data:

        - `th_vals` : Sequence[float] - Phase values :math:`\theta` in radians
        - `epi_vals` : Sequence[float] - EPI values
        - `vf_vals` : Sequence[float] - Structural frequencies :math:`\nu_f` in Hz_str
        - `si_vals` : Sequence[float] - Sense Index values
        - `cos_vals` : Sequence[float] | None - Precomputed :math:`\cos\theta` (optional cache)
        - `sin_vals` : Sequence[float] | None - Precomputed :math:`\sin\theta` (optional cache)

    i : int | None, optional
        Index of first node for pairwise computation. If None, vectorized mode is used.
    j : int | None, optional
        Index of second node for pairwise computation. If None, vectorized mode is used.
    trig : Any | None, optional
        Trigonometric cache object with `cos` and `sin` dictionaries. If None, computed on demand.
    G : TNFRGraph | None, optional
        Source graph (used to retrieve cached trigonometric values if available).
    nodes : Sequence[NodeId] | None, optional
        Node identifiers corresponding to indices in `inputs` arrays.
    epi_range : float, default=1.0
        Normalization range :math:`\Delta_{\text{EPI}}` for EPI similarity.
        Should be :math:`\text{EPI}_{\max} - \text{EPI}_{\min}`.
    vf_range : float, default=1.0
        Normalization range :math:`\Delta_{\nu_f}` for frequency similarity.
        Should be :math:`\nu_{f,\max} - \nu_{f,\min}`.
    np : ModuleType | None, optional
        NumPy-like module (numpy, jax.numpy, torch) for vectorized computation.
        If provided with `i=None, j=None`, returns vectorized arrays for all pairs.

    Returns
    -------
    SimilarityComponents or VectorizedComponents
        **Pairwise mode** (i and j provided):
            tuple of (s_phase, s_epi, s_vf, s_si) : tuple[float, float, float, float]
            Normalized similarity scores :math:`\in [0,1]` for the pair (i, j).

        **Vectorized mode** (i=None, j=None, np provided):
            tuple of (S_phase, S_epi, S_vf, S_si) : tuple[FloatMatrix, FloatMatrix, FloatMatrix, FloatMatrix]
            Matrices of shape (N, N) containing all pairwise similarities.

    Raises
    ------
    ValueError
        If pairwise mode is requested (i or j provided) but both are not specified.

    See Also
    --------
    coherence_matrix : Constructs full :math:`W \approx \hat{C}` matrix
    compute_coherence : Computes :math:`C(t) = \text{Tr}(\hat{C}\rho)`
    _combine_similarity : Weighted combination of similarity components

    Notes
    -----

    **Performance**:

    - Vectorized mode (with `np`) is ~10-100x faster for large networks
    - Trigonometric caching avoids redundant cos/sin evaluations
    - Use `get_trig_cache(G)` to populate cache before repeated calls

    **Normalization**:

    - `epi_range` and `vf_range` should reflect actual network ranges for proper scaling
    - If ranges are 0, defaults to 1.0 to avoid division by zero
    - Si similarity uses absolute difference (already bounded to [0,1])

    References
    ----------
    .. [1] Mathematical Foundations, §3.1.1 - Implementation Bridge
    .. [2] docs/source/theory/mathematical_foundations.md#311-implementation-bridge-theory-to-code
    .. [3] docs/source/examples/worked_examples.md - Example 2: Coherence Matrix Elements

    Examples
    --------

    **Pairwise computation**:

    >>> from tnfr.metrics.coherence import compute_wij_phase_epi_vf_si, SimilarityInputs
    >>> inputs = SimilarityInputs(
    ...     th_vals=[0.0, 0.1],
    ...     epi_vals=[0.5, 0.6],
    ...     vf_vals=[0.8, 0.7],
    ...     si_vals=[0.7, 0.8]
    ... )
    >>> s_phase, s_epi, s_vf, s_si = compute_wij_phase_epi_vf_si(
    ...     inputs, i=0, j=1, epi_range=1.0, vf_range=1.0
    ... )
    >>> 0.9 < s_phase < 1.0  # Nearly synchronized (theta_diff = 0.1 rad)
    True
    >>> 0.8 < s_epi < 1.0    # Similar EPI values
    True

    **Vectorized computation**:

    >>> import numpy as np
    >>> S_phase, S_epi, S_vf, S_si = compute_wij_phase_epi_vf_si(
    ...     inputs, epi_range=1.0, vf_range=1.0, np=np
    ... )
    >>> S_phase.shape  # All pairwise similarities
    (2, 2)
    >>> np.allclose(S_phase[0, 1], S_phase[1, 0])  # Symmetric
    True

    **With graph and caching**:

    >>> import networkx as nx
    >>> from tnfr.metrics.trig_cache import get_trig_cache
    >>> G = nx.Graph()
    >>> G.add_edge(0, 1)
    >>> G.nodes[0].update({"phase": 0.0, "EPI": 0.5, "nu_f": 0.8, "Si": 0.7})
    >>> G.nodes[1].update({"phase": 0.1, "EPI": 0.6, "nu_f": 0.7, "Si": 0.8})
    >>> trig = get_trig_cache(G, np=np)  # Precompute cos/sin
    >>> # ... use trig in repeated calls for efficiency
    """

    trig = trig or (get_trig_cache(G, np=np) if G is not None else None)
    cos_vals = inputs.cos_vals
    sin_vals = inputs.sin_vals
    if cos_vals is None or sin_vals is None:
        th_vals = inputs.th_vals
        pairs = zip(nodes or range(len(th_vals)), th_vals)
        trig_local = compute_theta_trig(pairs, np=np)
        index_iter = nodes if nodes is not None else range(len(th_vals))
        if trig is not None and nodes is not None:
            cos_vals = [trig.cos.get(n, trig_local.cos[n]) for n in nodes]
            sin_vals = [trig.sin.get(n, trig_local.sin[n]) for n in nodes]
        else:
            cos_vals = [trig_local.cos[i] for i in index_iter]
            sin_vals = [trig_local.sin[i] for i in index_iter]
        inputs.cos_vals = cos_vals
        inputs.sin_vals = sin_vals

    epi_vals = inputs.epi_vals
    vf_vals = inputs.vf_vals
    si_vals = inputs.si_vals

    if np is not None and i is None and j is None:
        epi = cast(FloatArray, np.asarray(epi_vals, dtype=float))
        vf = cast(FloatArray, np.asarray(vf_vals, dtype=float))
        si = cast(FloatArray, np.asarray(si_vals, dtype=float))
        cos_th = cast(FloatArray, np.asarray(cos_vals, dtype=float))
        sin_th = cast(FloatArray, np.asarray(sin_vals, dtype=float))
        return _compute_wij_phase_epi_vf_si_vectorized(
            epi,
            vf,
            si,
            cos_th,
            sin_th,
            epi_range,
            vf_range,
            np,
        )

    if i is None or j is None:
        raise ValueError("i and j are required for non-vectorized computation")
    epi_range = epi_range if epi_range > 0 else 1.0
    vf_range = vf_range if vf_range > 0 else 1.0
    cos_i = cos_vals[i]
    sin_i = sin_vals[i]
    cos_j = cos_vals[j]
    sin_j = sin_vals[j]
    s_phase = 0.5 * (1.0 + (cos_i * cos_j + sin_i * sin_j))
    s_epi = 1.0 - abs(epi_vals[i] - epi_vals[j]) / epi_range
    s_vf = 1.0 - abs(vf_vals[i] - vf_vals[j]) / vf_range
    s_si = 1.0 - abs(si_vals[i] - si_vals[j])
    return s_phase, s_epi, s_vf, s_si


def _combine_similarity(
    s_phase: ScalarOrArray,
    s_epi: ScalarOrArray,
    s_vf: ScalarOrArray,
    s_si: ScalarOrArray,
    phase_w: float,
    epi_w: float,
    vf_w: float,
    si_w: float,
    np: ModuleType | None = None,
) -> ScalarOrArray:
    """Combine similarity components into coherence weight wᵢⱼ ≈ ⟨i|Ĉ|j⟩.

    Returns wᵢⱼ ∈ [0, 1] clamped to maintain operator boundedness.

    See: Mathematical Foundations §3.1.1 for spectral projection details.
    """
    wij = phase_w * s_phase + epi_w * s_epi + vf_w * s_vf + si_w * s_si
    if np is not None:
        return cast(FloatArray, np.clip(wij, 0.0, 1.0))
    return clamp01(wij)


def _wij_components_weights(
    G: TNFRGraph,
    nodes: Sequence[NodeId] | None,
    inputs: SimilarityInputs,
    wnorm: Mapping[str, float],
    i: int | None = None,
    j: int | None = None,
    epi_range: float = 1.0,
    vf_range: float = 1.0,
    np: ModuleType | None = None,
) -> tuple[
    ScalarOrArray,
    ScalarOrArray,
    ScalarOrArray,
    ScalarOrArray,
    float,
    float,
    float,
    float,
]:
    """Return similarity components together with their weights.

    This consolidates repeated computations ensuring that both the
    similarity components and the corresponding weights are derived once and
    consistently across different implementations.
    """

    s_phase, s_epi, s_vf, s_si = compute_wij_phase_epi_vf_si(
        inputs,
        i,
        j,
        G=G,
        nodes=nodes,
        epi_range=epi_range,
        vf_range=vf_range,
        np=np,
    )
    phase_w = wnorm["phase"]
    epi_w = wnorm["epi"]
    vf_w = wnorm["vf"]
    si_w = wnorm["si"]
    return s_phase, s_epi, s_vf, s_si, phase_w, epi_w, vf_w, si_w


def _wij_vectorized(
    G: TNFRGraph,
    nodes: Sequence[NodeId],
    inputs: SimilarityInputs,
    wnorm: Mapping[str, float],
    epi_min: float,
    epi_max: float,
    vf_min: float,
    vf_max: float,
    self_diag: bool,
    np: ModuleType,
) -> FloatMatrix:
    epi_range = epi_max - epi_min if epi_max > epi_min else 1.0
    vf_range = vf_max - vf_min if vf_max > vf_min else 1.0
    (
        s_phase,
        s_epi,
        s_vf,
        s_si,
        phase_w,
        epi_w,
        vf_w,
        si_w,
    ) = _wij_components_weights(
        G,
        nodes,
        inputs,
        wnorm,
        epi_range=epi_range,
        vf_range=vf_range,
        np=np,
    )
    wij_matrix = cast(
        FloatMatrix,
        _combine_similarity(s_phase, s_epi, s_vf, s_si, phase_w, epi_w, vf_w, si_w, np=np),
    )
    if self_diag:
        np.fill_diagonal(wij_matrix, 1.0)
    else:
        np.fill_diagonal(wij_matrix, 0.0)
    return wij_matrix


def _compute_wij_value_raw(
    i: int,
    j: int,
    epi_vals: Sequence[float],
    vf_vals: Sequence[float],
    si_vals: Sequence[float],
    cos_vals: Sequence[float],
    sin_vals: Sequence[float],
    weights: tuple[float, float, float, float],
    epi_range: float,
    vf_range: float,
) -> float:
    epi_range = epi_range if epi_range > 0 else 1.0
    vf_range = vf_range if vf_range > 0 else 1.0
    phase_w, epi_w, vf_w, si_w = weights
    cos_i = cos_vals[i]
    sin_i = sin_vals[i]
    cos_j = cos_vals[j]
    sin_j = sin_vals[j]
    s_phase = 0.5 * (1.0 + (cos_i * cos_j + sin_i * sin_j))
    s_epi = 1.0 - abs(epi_vals[i] - epi_vals[j]) / epi_range
    s_vf = 1.0 - abs(vf_vals[i] - vf_vals[j]) / vf_range
    s_si = 1.0 - abs(si_vals[i] - si_vals[j])
    wij = phase_w * s_phase + epi_w * s_epi + vf_w * s_vf + si_w * s_si
    return clamp01(wij)


_PARALLEL_WIJ_DATA: ParallelWijPayload | None = None


def _init_parallel_wij(data: ParallelWijPayload) -> None:
    """Store immutable state for parallel ``wij`` computation."""

    global _PARALLEL_WIJ_DATA
    _PARALLEL_WIJ_DATA = data


def _parallel_wij_worker(
    pairs: Sequence[tuple[int, int]],
) -> list[tuple[int, int, float]]:
    """Compute coherence weights for ``pairs`` using shared state."""

    if _PARALLEL_WIJ_DATA is None:
        raise RuntimeError("Parallel coherence data not initialized")

    data = _PARALLEL_WIJ_DATA
    epi_vals: Sequence[float] = data["epi_vals"]
    vf_vals: Sequence[float] = data["vf_vals"]
    si_vals: Sequence[float] = data["si_vals"]
    cos_vals: Sequence[float] = data["cos_vals"]
    sin_vals: Sequence[float] = data["sin_vals"]
    weights: tuple[float, float, float, float] = data["weights"]
    epi_range: float = data["epi_range"]
    vf_range: float = data["vf_range"]

    compute = _compute_wij_value_raw
    return [
        (
            i,
            j,
            compute(
                i,
                j,
                epi_vals,
                vf_vals,
                si_vals,
                cos_vals,
                sin_vals,
                weights,
                epi_range,
                vf_range,
            ),
        )
        for i, j in pairs
    ]


def _wij_loops(
    G: TNFRGraph,
    nodes: Sequence[NodeId],
    node_to_index: Mapping[NodeId, int],
    inputs: SimilarityInputs,
    wnorm: Mapping[str, float],
    epi_min: float,
    epi_max: float,
    vf_min: float,
    vf_max: float,
    neighbors_only: bool,
    self_diag: bool,
    n_jobs: int | None = 1,
) -> CoherenceMatrixDense:
    n = len(nodes)
    cos_vals = inputs.cos_vals
    sin_vals = inputs.sin_vals
    if cos_vals is None or sin_vals is None:
        th_vals = inputs.th_vals
        trig_local = compute_theta_trig(zip(nodes, th_vals))
        cos_vals = [trig_local.cos[n] for n in nodes]
        sin_vals = [trig_local.sin[n] for n in nodes]
        inputs.cos_vals = cos_vals
        inputs.sin_vals = sin_vals
    assert cos_vals is not None
    assert sin_vals is not None
    epi_vals = list(inputs.epi_vals)
    vf_vals = list(inputs.vf_vals)
    si_vals = list(inputs.si_vals)
    cos_vals_list = list(cos_vals)
    sin_vals_list = list(sin_vals)
    inputs.epi_vals = epi_vals
    inputs.vf_vals = vf_vals
    inputs.si_vals = si_vals
    inputs.cos_vals = cos_vals_list
    inputs.sin_vals = sin_vals_list
    wij = [[1.0 if (self_diag and i == j) else 0.0 for j in range(n)] for i in range(n)]
    epi_range = epi_max - epi_min if epi_max > epi_min else 1.0
    vf_range = vf_max - vf_min if vf_max > vf_min else 1.0
    weights = (
        float(wnorm["phase"]),
        float(wnorm["epi"]),
        float(wnorm["vf"]),
        float(wnorm["si"]),
    )
    pair_list: list[tuple[int, int]] = []
    if neighbors_only:
        seen: set[tuple[int, int]] = set()
        for u, v in G.edges():
            i = node_to_index[u]
            j = node_to_index[v]
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            if pair in seen:
                continue
            seen.add(pair)
            pair_list.append(pair)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                pair_list.append((i, j))

    total_pairs = len(pair_list)
    max_workers = 1
    if n_jobs is not None:
        try:
            max_workers = int(n_jobs)
        except (TypeError, ValueError):
            max_workers = 1
    if max_workers <= 1 or total_pairs == 0:
        for i, j in pair_list:
            wij_ij = _compute_wij_value_raw(
                i,
                j,
                epi_vals,
                vf_vals,
                si_vals,
                cos_vals,
                sin_vals,
                weights,
                epi_range,
                vf_range,
            )
            wij[i][j] = wij[j][i] = wij_ij
        return wij

    approx_chunk = math.ceil(total_pairs / max_workers) if max_workers else None
    chunk_size = resolve_chunk_size(
        approx_chunk,
        total_pairs,
        minimum=1,
    )
    payload: ParallelWijPayload = {
        "epi_vals": tuple(epi_vals),
        "vf_vals": tuple(vf_vals),
        "si_vals": tuple(si_vals),
        "cos_vals": tuple(cos_vals),
        "sin_vals": tuple(sin_vals),
        "weights": weights,
        "epi_range": float(epi_range),
        "vf_range": float(vf_range),
    }

    def _init() -> None:
        _init_parallel_wij(payload)

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init) as executor:
        futures = []
        for start in range(0, total_pairs, chunk_size):
            chunk = pair_list[start : start + chunk_size]
            futures.append(executor.submit(_parallel_wij_worker, chunk))
        for future in futures:
            for i, j, value in future.result():
                wij[i][j] = wij[j][i] = value
    return wij


def _compute_stats(
    values: Iterable[float] | Any,
    row_sum: Iterable[float] | Any,
    n: int,
    self_diag: bool,
    np: ModuleType | None = None,
) -> tuple[float, float, float, list[float], int]:
    """Return aggregate statistics for ``values`` and normalized row sums.

    ``values`` and ``row_sum`` can be any iterables. They are normalized to
    either NumPy arrays or Python lists depending on the availability of
    NumPy. The computation then delegates to the appropriate numerical
    functions with minimal branching.
    """

    if np is not None:
        if not isinstance(values, np.ndarray):
            values_arr = np.asarray(list(values), dtype=float)
        else:
            values_arr = cast(Any, values.astype(float))
        if not isinstance(row_sum, np.ndarray):
            row_arr = np.asarray(list(row_sum), dtype=float)
        else:
            row_arr = cast(Any, row_sum.astype(float))
        count_val = int(values_arr.size)
        min_val = float(values_arr.min()) if values_arr.size else 0.0
        max_val = float(values_arr.max()) if values_arr.size else 0.0
        mean_val = float(values_arr.mean()) if values_arr.size else 0.0
    else:
        values_list = list(values)
        row_arr = list(row_sum)
        count_val = len(values_list)
        min_val = min(values_list) if values_list else 0.0
        max_val = max(values_list) if values_list else 0.0
        mean_val = sum(values_list) / len(values_list) if values_list else 0.0

    row_count = n if self_diag else n - 1
    denom = max(1, row_count)
    if np is not None:
        Wi = (row_arr / denom).astype(float).tolist()  # type: ignore[operator]
    else:
        Wi = [float(row_arr[i]) / denom for i in range(n)]
    return min_val, max_val, mean_val, Wi, count_val


def _coherence_numpy(
    wij: Any,
    mode: str,
    thr: float,
    np: ModuleType,
) -> tuple[int, Any, Any, CoherenceMatrixPayload]:
    """Aggregate coherence weights using vectorized operations.

    Produces the structural weight matrix ``W`` along with the list of off
    diagonal values and row sums ready for statistical analysis.
    """

    n = wij.shape[0]
    mask = ~np.eye(n, dtype=bool)
    values = wij[mask]
    row_sum = wij.sum(axis=1)
    if mode == "dense":
        W = wij.tolist()
    else:
        idx = np.where((wij >= thr) & mask)
        W = [(int(i), int(j), float(wij[i, j])) for i, j in zip(idx[0], idx[1])]
    return n, values, row_sum, W


def _coherence_python_worker(
    args: tuple[Sequence[Sequence[float]], int, str, float],
) -> tuple[int, list[float], list[float], CoherenceMatrixSparse]:
    rows, start, mode, thr = args
    values: list[float] = []
    row_sum: list[float] = []
    sparse: list[tuple[int, int, float]] = []
    dense_mode = mode == "dense"

    for offset, row in enumerate(rows):
        i = start + offset
        total = 0.0
        for j, w in enumerate(row):
            total += w
            if i != j:
                values.append(w)
                if not dense_mode and w >= thr:
                    sparse.append((i, j, w))
        row_sum.append(total)

    return start, values, row_sum, sparse


def _coherence_python(
    wij: Sequence[Sequence[float]],
    mode: str,
    thr: float,
    n_jobs: int | None = 1,
) -> tuple[int, list[float], list[float], CoherenceMatrixPayload]:
    """Aggregate coherence weights using pure Python loops."""

    n = len(wij)
    values: list[float] = []
    row_sum = [0.0] * n

    if n_jobs is not None:
        try:
            max_workers = int(n_jobs)
        except (TypeError, ValueError):
            max_workers = 1
    else:
        max_workers = 1

    if max_workers <= 1:
        if mode == "dense":
            W: CoherenceMatrixDense = [list(row) for row in wij]
            for i in range(n):
                for j in range(n):
                    w = W[i][j]
                    if i != j:
                        values.append(w)
                    row_sum[i] += w
        else:
            W_sparse: CoherenceMatrixSparse = []
            for i in range(n):
                row_i = wij[i]
                for j in range(n):
                    w = row_i[j]
                    if i != j:
                        values.append(w)
                        if w >= thr:
                            W_sparse.append((i, j, w))
                    row_sum[i] += w
        return n, values, row_sum, W if mode == "dense" else W_sparse

    approx_chunk = math.ceil(n / max_workers) if max_workers else None
    chunk_size = resolve_chunk_size(
        approx_chunk,
        n,
        minimum=1,
    )
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, n, chunk_size):
            rows = wij[start : start + chunk_size]
            tasks.append(
                executor.submit(
                    _coherence_python_worker,
                    (tuple(tuple(row) for row in rows), start, mode, thr),
                )
            )
        results = [task.result() for task in tasks]

    results.sort(key=lambda item: item[0])
    sparse_entries: list[tuple[int, int, float]] | None = [] if mode != "dense" else None
    for start, chunk_values, chunk_row_sum, chunk_sparse in results:
        values.extend(chunk_values)
        for offset, total in enumerate(chunk_row_sum):
            row_sum[start + offset] = total
        if sparse_entries is not None:
            sparse_entries.extend(chunk_sparse)

    if mode == "dense":
        W_dense: CoherenceMatrixDense = [list(row) for row in wij]
        return n, values, row_sum, W_dense
    sparse_result: CoherenceMatrixSparse = sparse_entries if sparse_entries is not None else []
    return n, values, row_sum, sparse_result


def _finalize_wij(
    G: TNFRGraph,
    nodes: Sequence[NodeId],
    wij: FloatMatrix | Sequence[Sequence[float]],
    mode: str,
    thr: float,
    scope: str,
    self_diag: bool,
    np: ModuleType | None = None,
    *,
    n_jobs: int = 1,
) -> tuple[list[NodeId], CoherenceMatrixPayload]:
    """Finalize the coherence matrix ``wij`` and store results in history.

    When ``np`` is provided and ``wij`` is a NumPy array, the computation is
    performed using vectorized operations. Otherwise a pure Python loop-based
    approach is used.
    """

    use_np = np is not None and isinstance(wij, np.ndarray)
    if use_np:
        assert np is not None
        n, values, row_sum, W = _coherence_numpy(wij, mode, thr, np)
    else:
        n, values, row_sum, W = _coherence_python(wij, mode, thr, n_jobs=n_jobs)

    min_val, max_val, mean_val, Wi, count_val = _compute_stats(
        values, row_sum, n, self_diag, np if use_np else None
    )
    stats = {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "n_edges": count_val,
        "mode": mode,
        "scope": scope,
    }

    hist = ensure_history(G)
    cfg = get_param(G, "COHERENCE")
    append_metric(hist, cfg.get("history_key", "W_sparse"), W)
    append_metric(hist, cfg.get("Wi_history_key", "W_i"), Wi)
    append_metric(hist, cfg.get("stats_history_key", "W_stats"), stats)
    return list(nodes), W


def coherence_matrix(
    G: TNFRGraph,
    use_numpy: bool | None = None,
    *,
    n_jobs: int | None = None,
) -> tuple[list[NodeId] | None, CoherenceMatrixPayload | None]:
    """Compute coherence matrix W approximating operator Ĉ.

    Returns matrix W where wᵢⱼ ≈ ⟨i|Ĉ|j⟩ computed from structural
    similarities: phase, EPI, frequency, and sense index.

    Mathematical Foundation:
        Ĉ ≈ Σᵢⱼ wᵢⱼ |i⟩⟨j|

    Matrix W satisfies Hermiticity (W=W^T), element bounds (wᵢⱼ ∈ [0,1]),
    and provides spectrum σ(Ĉ) via eigenvalues.

    Parameters
    ----------
    G:
        Graph with node attributes: theta, EPI, vf, Si
    use_numpy:
        Force NumPy (True), pure Python (False), or auto-detect (None)
    n_jobs:
        Worker processes for Python fallback (None or ≤1 = serial)

    Returns
    -------
    nodes:
        Ordered node list matching matrix indexing
    W:
        Coherence matrix (dense or sparse per configuration)

    See Also
    --------
    compute_coherence : Computes C(t) = Tr(Ĉρ)
    Mathematical Foundations §3.1: Theory + Implementation Bridge

    Examples
    --------
    >>> nodes, W = coherence_matrix(G)
    >>> # W[i][j] ≈ ⟨i|Ĉ|j⟩ for computational basis
    """

    cfg = get_param(G, "COHERENCE")
    if not cfg.get("enabled", True):
        return None, None

    node_to_index: Mapping[NodeId, int] = ensure_node_index_map(G)
    nodes: list[NodeId] = list(node_to_index.keys())
    n = len(nodes)
    if n == 0:
        return nodes, []

    # NumPy handling for optional vectorized operations
    np = get_numpy()
    use_np = np is not None if use_numpy is None else (use_numpy and np is not None)

    cfg_jobs = cfg.get("n_jobs")
    parallel_jobs = n_jobs if n_jobs is not None else cfg_jobs

    # Precompute indices to avoid repeated list.index calls within loops

    th_vals = collect_theta_attr(G, nodes, 0.0, np=np if use_np else None)
    epi_vals = collect_attr(G, nodes, ALIAS_EPI, 0.0, np=np if use_np else None)
    vf_vals = collect_attr(G, nodes, ALIAS_VF, 0.0, np=np if use_np else None)
    si_vals = collect_attr(G, nodes, ALIAS_SI, 0.0, np=np if use_np else None)
    if use_np:
        assert np is not None
        si_vals = np.clip(si_vals, 0.0, 1.0)
    else:
        si_vals = [clamp01(v) for v in si_vals]
    epi_min, epi_max = min_max_range(epi_vals)
    vf_min, vf_max = min_max_range(vf_vals)

    wdict = dict(cfg.get("weights", {}))
    for k in ("phase", "epi", "vf", "si"):
        wdict.setdefault(k, 0.0)
    wnorm = normalize_weights(wdict, ("phase", "epi", "vf", "si"), default=0.0)

    scope = str(cfg.get("scope", "neighbors")).lower()
    neighbors_only = scope != "all"
    self_diag = bool(cfg.get("self_on_diag", True))
    mode = str(cfg.get("store_mode", "sparse")).lower()
    thr = float(cfg.get("threshold", 0.0))
    if mode not in ("sparse", "dense"):
        mode = "sparse"
    trig = get_trig_cache(G, np=np)
    cos_map, sin_map = trig.cos, trig.sin
    trig_local = compute_theta_trig(zip(nodes, th_vals), np=np)
    cos_vals = [cos_map.get(n, trig_local.cos[n]) for n in nodes]
    sin_vals = [sin_map.get(n, trig_local.sin[n]) for n in nodes]
    inputs = SimilarityInputs(
        th_vals=th_vals,
        epi_vals=epi_vals,
        vf_vals=vf_vals,
        si_vals=si_vals,
        cos_vals=cos_vals,
        sin_vals=sin_vals,
    )
    if use_np:
        assert np is not None
        wij_matrix = _wij_vectorized(
            G,
            nodes,
            inputs,
            wnorm,
            epi_min,
            epi_max,
            vf_min,
            vf_max,
            self_diag,
            np,
        )
        if neighbors_only:
            adj = np.eye(n, dtype=bool)
            for u, v in G.edges():
                i = node_to_index[u]
                j = node_to_index[v]
                adj[i, j] = True
                adj[j, i] = True
            wij_matrix = cast(FloatMatrix, np.where(adj, wij_matrix, 0.0))
        wij: FloatMatrix | CoherenceMatrixDense = wij_matrix
    else:
        wij = _wij_loops(
            G,
            nodes,
            node_to_index,
            inputs,
            wnorm,
            epi_min,
            epi_max,
            vf_min,
            vf_max,
            neighbors_only,
            self_diag,
            n_jobs=parallel_jobs,
        )

    return _finalize_wij(
        G,
        nodes,
        wij,
        mode,
        thr,
        scope,
        self_diag,
        np,
        n_jobs=parallel_jobs if not use_np else 1,
    )


def local_phase_sync_weighted(
    G: TNFRGraph,
    n: NodeId,
    nodes_order: Sequence[NodeId] | None = None,
    W_row: PhaseSyncWeights | None = None,
    node_to_index: Mapping[NodeId, int] | None = None,
) -> float:
    """Compute local phase synchrony using explicit weights.

    ``nodes_order`` is the node ordering used to build the coherence matrix
    and ``W_row`` contains either the dense row corresponding to ``n`` or the
    sparse list of ``(i, j, w)`` tuples for the whole matrix.
    """
    if W_row is None or nodes_order is None:
        raise ValueError("nodes_order and W_row are required for weighted phase synchrony")

    if node_to_index is None:
        node_to_index = ensure_node_index_map(G)
    i = node_to_index.get(n)
    if i is None:
        i = nodes_order.index(n)

    num = 0 + 0j
    den = 0.0

    trig = get_trig_cache(G)
    cos_map, sin_map = trig.cos, trig.sin

    if isinstance(W_row, Sequence) and W_row:
        first = W_row[0]
        if isinstance(first, (int, float)):
            row_vals = cast(Sequence[float], W_row)
            for w, nj in zip(row_vals, nodes_order):
                if nj == n:
                    continue
                den += w
                cos_j = cos_map.get(nj)
                sin_j = sin_map.get(nj)
                if cos_j is None or sin_j is None:
                    trig_j = compute_theta_trig(((nj, G.nodes[nj]),))
                    cos_j = trig_j.cos[nj]
                    sin_j = trig_j.sin[nj]
                num += w * complex(cos_j, sin_j)
            return abs(num / den) if den else 0.0

        if (
            isinstance(first, Sequence)
            and len(first) == 3
            and isinstance(first[0], int)
            and isinstance(first[1], int)
            and isinstance(first[2], (int, float))
        ):
            sparse_entries = cast(CoherenceMatrixSparse, W_row)
            for ii, jj, w in sparse_entries:
                if ii != i:
                    continue
                nj = nodes_order[jj]
                if nj == n:
                    continue
                den += w
                cos_j = cos_map.get(nj)
                sin_j = sin_map.get(nj)
                if cos_j is None or sin_j is None:
                    trig_j = compute_theta_trig(((nj, G.nodes[nj]),))
                    cos_j = trig_j.cos[nj]
                    sin_j = trig_j.sin[nj]
                num += w * complex(cos_j, sin_j)
            return abs(num / den) if den else 0.0

        dense_matrix = cast(CoherenceMatrixDense, W_row)
        if i is None:
            raise ValueError("node index resolution failed for dense weights")
        row_vals = cast(Sequence[float], dense_matrix[i])
        for w, nj in zip(row_vals, nodes_order):
            if nj == n:
                continue
            den += w
            cos_j = cos_map.get(nj)
            sin_j = sin_map.get(nj)
            if cos_j is None or sin_j is None:
                trig_j = compute_theta_trig(((nj, G.nodes[nj]),))
                cos_j = trig_j.cos[nj]
                sin_j = trig_j.sin[nj]
            num += w * complex(cos_j, sin_j)
        return abs(num / den) if den else 0.0

    sparse_entries = cast(CoherenceMatrixSparse, W_row)
    for ii, jj, w in sparse_entries:
        if ii != i:
            continue
        nj = nodes_order[jj]
        if nj == n:
            continue
        den += w
        cos_j = cos_map.get(nj)
        sin_j = sin_map.get(nj)
        if cos_j is None or sin_j is None:
            trig_j = compute_theta_trig(((nj, G.nodes[nj]),))
            cos_j = trig_j.cos[nj]
            sin_j = trig_j.sin[nj]
        num += w * complex(cos_j, sin_j)

    return abs(num / den) if den else 0.0


def local_phase_sync(G: TNFRGraph, n: NodeId) -> float:
    """Compute unweighted local phase synchronization for node ``n``."""
    nodes, W = coherence_matrix(G)
    if nodes is None:
        return 0.0
    return local_phase_sync_weighted(G, n, nodes_order=nodes, W_row=W)


def _coherence_step(G: TNFRGraph, ctx: dict[str, Any] | None = None) -> None:
    del ctx

    if not get_param(G, "COHERENCE").get("enabled", True):
        return
    coherence_matrix(G)


def register_coherence_callbacks(G: TNFRGraph) -> None:
    """Attach coherence matrix maintenance to the ``AFTER_STEP`` event."""

    callback_manager.register_callback(
        G,
        event=CallbackEvent.AFTER_STEP.value,
        func=_coherence_step,
        name="coherence_step",
    )


# ---------------------------------------------------------------------------
# Coherence and observer-related metric updates
# ---------------------------------------------------------------------------


def _record_metrics(
    hist: HistoryState,
    *pairs: MetricRecord,
    evaluate: bool = False,
) -> None:
    """Record metric values for the trace history."""

    metrics = cast(MutableMapping[str, list[Any]], hist)
    for payload, key in pairs:
        if evaluate:
            provider = cast(MetricProvider, payload)
            append_metric(metrics, key, provider())
        else:
            append_metric(metrics, key, payload)


def _update_coherence(G: TNFRGraph, hist: HistoryState) -> None:
    """Update network coherence and related means."""

    coherence_payload = cast(
        tuple[CoherenceMetric, float, float],
        compute_coherence(G, return_means=True),
    )
    C, dnfr_mean, depi_mean = coherence_payload
    _record_metrics(
        hist,
        (C, "C_steps"),
        (dnfr_mean, "dnfr_mean"),
        (depi_mean, "depi_mean"),
    )

    cs = hist["C_steps"]
    if cs:
        window = min(len(cs), DEFAULT_WBAR_SPAN)
        w = max(1, window)
        wbar = sum(cs[-w:]) / w
        _record_metrics(hist, (wbar, "W_bar"))


def _update_phase_sync(G: TNFRGraph, hist: HistoryState) -> None:
    """Capture phase synchrony and Kuramoto order."""

    ps = phase_sync(G)
    ko = kuramoto_order(G)
    _record_metrics(
        hist,
        (ps, "phase_sync"),
        (ko, "kuramoto_R"),
    )


def _update_sigma(G: TNFRGraph, hist: HistoryState) -> None:
    """Record glyph load and associated Σ⃗ vector."""

    metrics = cast(MutableMapping[str, list[Any]], hist)
    if "glyph_load_estab" in metrics:
        raise ValueError(
            "History payloads using 'glyph_load_estab' are no longer supported. "
            "Rename the series to 'glyph_load_stabilizers' before loading the graph."
        )
    if metrics.get(GLYPH_LOAD_STABILIZERS_KEY) is None:
        metrics.setdefault(GLYPH_LOAD_STABILIZERS_KEY, [])

    gl: GlyphLoadDistribution = glyph_load(G, window=DEFAULT_GLYPH_LOAD_SPAN)
    stabilizers = float(gl.get("_stabilizers", 0.0))
    disruptors = float(gl.get("_disruptors", 0.0))
    _record_metrics(
        hist,
        (stabilizers, GLYPH_LOAD_STABILIZERS_KEY),
        (disruptors, "glyph_load_disr"),
    )

    dist: GlyphLoadDistribution = {k: v for k, v in gl.items() if not k.startswith("_")}
    sig: SigmaVector = sigma_vector(dist)
    _record_metrics(
        hist,
        (sig.get("x", 0.0), "sense_sigma_x"),
        (sig.get("y", 0.0), "sense_sigma_y"),
        (sig.get("mag", 0.0), "sense_sigma_mag"),
        (sig.get("angle", 0.0), "sense_sigma_angle"),
    )


def _stability_chunk_worker(args: StabilityChunkArgs) -> StabilityChunkResult:
    """Compute stability aggregates for a chunk of nodes."""

    (
        dnfr_vals,
        depi_vals,
        si_curr_vals,
        si_prev_vals,
        vf_curr_vals,
        vf_prev_vals,
        dvf_prev_vals,
        dt,
        eps_dnfr,
        eps_depi,
    ) = args

    inv_dt = (1.0 / dt) if dt else 0.0
    stable = 0
    delta_sum = 0.0
    B_sum = 0.0
    delta_vals: list[float] = []
    dvf_dt_vals: list[float] = []
    B_vals: list[float] = []

    for idx in range(len(si_curr_vals)):
        curr_si = float(si_curr_vals[idx])
        prev_si_raw = si_prev_vals[idx]
        prev_si = float(prev_si_raw) if prev_si_raw is not None else curr_si
        delta = curr_si - prev_si
        delta_vals.append(delta)
        delta_sum += delta

        curr_vf = float(vf_curr_vals[idx])
        prev_vf_raw = vf_prev_vals[idx]
        prev_vf = float(prev_vf_raw) if prev_vf_raw is not None else curr_vf
        dvf_dt = (curr_vf - prev_vf) * inv_dt if dt else 0.0
        prev_dvf_raw = dvf_prev_vals[idx]
        prev_dvf = float(prev_dvf_raw) if prev_dvf_raw is not None else dvf_dt
        B = (dvf_dt - prev_dvf) * inv_dt if dt else 0.0
        dvf_dt_vals.append(dvf_dt)
        B_vals.append(B)
        B_sum += B

        if abs(float(dnfr_vals[idx])) <= eps_dnfr and abs(float(depi_vals[idx])) <= eps_depi:
            stable += 1

    chunk_len = len(si_curr_vals)
    return (
        stable,
        chunk_len,
        delta_sum,
        B_sum,
        delta_vals,
        dvf_dt_vals,
        B_vals,
    )


def _track_stability(
    G: TNFRGraph,
    hist: MutableMapping[str, Any],
    dt: float,
    eps_dnfr: float,
    eps_depi: float,
    *,
    n_jobs: int | None = None,
) -> None:
    """Track per-node stability and derivative metrics."""

    nodes: tuple[NodeId, ...] = tuple(G.nodes)
    total_nodes = len(nodes)
    if not total_nodes:
        hist.setdefault("stable_frac", []).append(0.0)
        hist.setdefault("delta_Si", []).append(0.0)
        hist.setdefault("B", []).append(0.0)
        return

    np_mod = get_numpy()

    dnfr_vals = collect_attr(G, nodes, ALIAS_DNFR, 0.0, np=np_mod)
    depi_vals = collect_attr(G, nodes, ALIAS_DEPI, 0.0, np=np_mod)
    si_curr_vals = collect_attr(G, nodes, ALIAS_SI, 0.0, np=np_mod)
    vf_curr_vals = collect_attr(G, nodes, ALIAS_VF, 0.0, np=np_mod)

    prev_si_data = [G.nodes[n].get("_prev_Si") for n in nodes]
    prev_vf_data = [G.nodes[n].get("_prev_vf") for n in nodes]
    prev_dvf_data = [G.nodes[n].get("_prev_dvf") for n in nodes]

    inv_dt = (1.0 / dt) if dt else 0.0

    if np_mod is not None:
        np = np_mod
        dnfr_arr = dnfr_vals
        depi_arr = depi_vals
        si_curr_arr = si_curr_vals
        vf_curr_arr = vf_curr_vals

        si_prev_arr = np.asarray(
            [
                (
                    float(prev_si_data[idx])
                    if prev_si_data[idx] is not None
                    else float(si_curr_arr[idx])
                )
                for idx in range(total_nodes)
            ],
            dtype=float,
        )
        vf_prev_arr = np.asarray(
            [
                (
                    float(prev_vf_data[idx])
                    if prev_vf_data[idx] is not None
                    else float(vf_curr_arr[idx])
                )
                for idx in range(total_nodes)
            ],
            dtype=float,
        )

        if dt:
            dvf_dt_arr = (vf_curr_arr - vf_prev_arr) * inv_dt
        else:
            dvf_dt_arr = np.zeros_like(vf_curr_arr, dtype=float)

        dvf_prev_arr = np.asarray(
            [
                (
                    float(prev_dvf_data[idx])
                    if prev_dvf_data[idx] is not None
                    else float(dvf_dt_arr[idx])
                )
                for idx in range(total_nodes)
            ],
            dtype=float,
        )

        if dt:
            B_arr = (dvf_dt_arr - dvf_prev_arr) * inv_dt
        else:
            B_arr = np.zeros_like(dvf_dt_arr, dtype=float)

        stable_mask = (np.abs(dnfr_arr) <= eps_dnfr) & (np.abs(depi_arr) <= eps_depi)
        stable_frac = float(stable_mask.mean()) if total_nodes else 0.0

        delta_si_arr = si_curr_arr - si_prev_arr
        delta_si_mean = float(delta_si_arr.mean()) if total_nodes else 0.0
        B_mean = float(B_arr.mean()) if total_nodes else 0.0

        hist.setdefault("stable_frac", []).append(stable_frac)
        hist.setdefault("delta_Si", []).append(delta_si_mean)
        hist.setdefault("B", []).append(B_mean)

        for idx, node in enumerate(nodes):
            nd = G.nodes[node]
            curr_si = float(si_curr_arr[idx])
            delta_val = float(delta_si_arr[idx])
            nd["_prev_Si"] = curr_si
            set_attr(nd, ALIAS_DSI, delta_val)

            curr_vf = float(vf_curr_arr[idx])
            nd["_prev_vf"] = curr_vf

            dvf_dt_val = float(dvf_dt_arr[idx])
            nd["_prev_dvf"] = dvf_dt_val
            set_attr(nd, ALIAS_DVF, dvf_dt_val)
            set_attr(nd, ALIAS_D2VF, float(B_arr[idx]))

        return

    # NumPy not available: optionally parallel fallback or sequential computation.
    dnfr_list = list(dnfr_vals)
    depi_list = list(depi_vals)
    si_curr_list = list(si_curr_vals)
    vf_curr_list = list(vf_curr_vals)

    if n_jobs and n_jobs > 1:
        approx_chunk = math.ceil(total_nodes / n_jobs) if n_jobs else None
        chunk_size = resolve_chunk_size(
            approx_chunk,
            total_nodes,
            minimum=1,
        )
        chunk_results: list[
            tuple[
                int,
                tuple[int, int, float, float, list[float], list[float], list[float]],
            ]
        ] = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures: list[tuple[int, Any]] = []
            for start in range(0, total_nodes, chunk_size):
                end = min(start + chunk_size, total_nodes)
                chunk_args = (
                    dnfr_list[start:end],
                    depi_list[start:end],
                    si_curr_list[start:end],
                    prev_si_data[start:end],
                    vf_curr_list[start:end],
                    prev_vf_data[start:end],
                    prev_dvf_data[start:end],
                    dt,
                    eps_dnfr,
                    eps_depi,
                )
                futures.append((start, executor.submit(_stability_chunk_worker, chunk_args)))

            for start, fut in futures:
                chunk_results.append((start, fut.result()))

        chunk_results.sort(key=lambda item: item[0])

        stable_total = 0
        delta_sum = 0.0
        B_sum = 0.0
        delta_vals_all: list[float] = []
        dvf_dt_all: list[float] = []
        B_vals_all: list[float] = []

        for _, result in chunk_results:
            (
                stable_count,
                chunk_len,
                chunk_delta_sum,
                chunk_B_sum,
                delta_vals,
                dvf_vals,
                B_vals,
            ) = result
            stable_total += stable_count
            delta_sum += chunk_delta_sum
            B_sum += chunk_B_sum
            delta_vals_all.extend(delta_vals)
            dvf_dt_all.extend(dvf_vals)
            B_vals_all.extend(B_vals)

        total = len(delta_vals_all)
        stable_frac = stable_total / total if total else 0.0
        delta_si_mean = delta_sum / total if total else 0.0
        B_mean = B_sum / total if total else 0.0

    else:
        stable_total = 0
        delta_sum = 0.0
        B_sum = 0.0
        delta_vals_all = []
        dvf_dt_all = []
        B_vals_all = []

        for idx in range(total_nodes):
            curr_si = float(si_curr_list[idx])
            prev_si_raw = prev_si_data[idx]
            prev_si = float(prev_si_raw) if prev_si_raw is not None else curr_si
            delta = curr_si - prev_si
            delta_vals_all.append(delta)
            delta_sum += delta

            curr_vf = float(vf_curr_list[idx])
            prev_vf_raw = prev_vf_data[idx]
            prev_vf = float(prev_vf_raw) if prev_vf_raw is not None else curr_vf
            dvf_dt_val = (curr_vf - prev_vf) * inv_dt if dt else 0.0
            prev_dvf_raw = prev_dvf_data[idx]
            prev_dvf = float(prev_dvf_raw) if prev_dvf_raw is not None else dvf_dt_val
            B_val = (dvf_dt_val - prev_dvf) * inv_dt if dt else 0.0
            dvf_dt_all.append(dvf_dt_val)
            B_vals_all.append(B_val)
            B_sum += B_val

            if abs(float(dnfr_list[idx])) <= eps_dnfr and abs(float(depi_list[idx])) <= eps_depi:
                stable_total += 1

        total = len(delta_vals_all)
        stable_frac = stable_total / total if total else 0.0
        delta_si_mean = delta_sum / total if total else 0.0
        B_mean = B_sum / total if total else 0.0

    hist.setdefault("stable_frac", []).append(stable_frac)
    hist.setdefault("delta_Si", []).append(delta_si_mean)
    hist.setdefault("B", []).append(B_mean)

    for idx, node in enumerate(nodes):
        nd = G.nodes[node]
        curr_si = float(si_curr_list[idx])
        delta_val = float(delta_vals_all[idx])
        nd["_prev_Si"] = curr_si
        set_attr(nd, ALIAS_DSI, delta_val)

        curr_vf = float(vf_curr_list[idx])
        nd["_prev_vf"] = curr_vf

        dvf_dt_val = float(dvf_dt_all[idx])
        nd["_prev_dvf"] = dvf_dt_val
        set_attr(nd, ALIAS_DVF, dvf_dt_val)
        set_attr(nd, ALIAS_D2VF, float(B_vals_all[idx]))


def _si_chunk_stats(
    values: Sequence[float], si_hi: float, si_lo: float
) -> tuple[float, int, int, int]:
    """Compute partial Si aggregates for ``values``.

    The helper keeps the logic shared between the sequential and parallel
    fallbacks when NumPy is unavailable.
    """

    total = 0.0
    count = 0
    hi_count = 0
    lo_count = 0
    for s in values:
        if math.isnan(s):
            continue
        total += s
        count += 1
        if s >= si_hi:
            hi_count += 1
        if s <= si_lo:
            lo_count += 1
    return total, count, hi_count, lo_count


def _aggregate_si(
    G: TNFRGraph,
    hist: MutableMapping[str, list[float]],
    *,
    n_jobs: int | None = None,
) -> None:
    """Aggregate Si statistics across nodes."""

    try:
        thr_sel = get_param(G, "SELECTOR_THRESHOLDS")
        thr_def = get_param(G, "GLYPH_THRESHOLDS")
        si_hi = float(thr_sel.get("si_hi", thr_def.get("hi", 0.66)))
        si_lo = float(thr_sel.get("si_lo", thr_def.get("lo", 0.33)))

        node_ids = list(G.nodes)
        if not node_ids:
            hist["Si_mean"].append(0.0)
            hist["Si_hi_frac"].append(0.0)
            hist["Si_lo_frac"].append(0.0)
            return

        sis = []
        for node in node_ids:
            raw = get_attr(
                G.nodes[node],
                ALIAS_SI,
                None,
                conv=lambda value: value,  # Preserve NaN sentinels
            )
            try:
                sis.append(float(raw) if raw is not None else math.nan)
            except (TypeError, ValueError):
                sis.append(math.nan)

        np_mod = get_numpy()
        if np_mod is not None:
            sis_array = np_mod.asarray(sis, dtype=float)
            valid = sis_array[~np_mod.isnan(sis_array)]
            n = int(valid.size)
            if n:
                hist["Si_mean"].append(float(valid.mean()))
                hi_frac = np_mod.count_nonzero(valid >= si_hi) / n
                lo_frac = np_mod.count_nonzero(valid <= si_lo) / n
                hist["Si_hi_frac"].append(float(hi_frac))
                hist["Si_lo_frac"].append(float(lo_frac))
            else:
                hist["Si_mean"].append(0.0)
                hist["Si_hi_frac"].append(0.0)
                hist["Si_lo_frac"].append(0.0)
            return

        if n_jobs is not None and n_jobs > 1:
            approx_chunk = math.ceil(len(sis) / n_jobs) if n_jobs else None
            chunk_size = resolve_chunk_size(
                approx_chunk,
                len(sis),
                minimum=1,
            )
            futures = []
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for idx in range(0, len(sis), chunk_size):
                    chunk = sis[idx : idx + chunk_size]
                    futures.append(executor.submit(_si_chunk_stats, chunk, si_hi, si_lo))
            totals = [future.result() for future in futures]
            total = sum(part[0] for part in totals)
            count = sum(part[1] for part in totals)
            hi_count = sum(part[2] for part in totals)
            lo_count = sum(part[3] for part in totals)
        else:
            total, count, hi_count, lo_count = _si_chunk_stats(sis, si_hi, si_lo)

        if count:
            hist["Si_mean"].append(total / count)
            hist["Si_hi_frac"].append(hi_count / count)
            hist["Si_lo_frac"].append(lo_count / count)
        else:
            hist["Si_mean"].append(0.0)
            hist["Si_hi_frac"].append(0.0)
            hist["Si_lo_frac"].append(0.0)
    except (KeyError, AttributeError, TypeError) as exc:
        logger.debug("Si aggregation failed: %s", exc)


def compute_global_coherence(G: TNFRGraph) -> float:
    """Compute global coherence C(t) for entire network.

    C(t) = 1 - (σ_ΔNFR / ΔNFR_max)

    This is the canonical TNFR coherence metric that measures global structural
    stability through the dispersion of reorganization pressure (ΔNFR) across
    the network.

    Parameters
    ----------
    G : TNFRGraph
        Network graph with nodes containing ΔNFR attributes

    Returns
    -------
    float
        Global coherence value in [0, 1] where:
        - 1.0 = perfect coherence (no reorganization pressure variance)
        - 0.0 = maximum incoherence (extreme ΔNFR dispersion)

    Notes
    -----
    **Mathematical Foundation:**

    Global coherence quantifies the network's structural stability by measuring
    how uniformly reorganization pressure is distributed across nodes:

    - **σ_ΔNFR**: Standard deviation of ΔNFR values measures dispersion
    - **ΔNFR_max**: Maximum ΔNFR provides normalization scale
    - **C(t)**: Higher values indicate more uniform structural state

    **Special Cases:**

    - Empty network: Returns 1.0 (perfect coherence by definition)
    - All ΔNFR = 0: Returns 1.0 (no reorganization pressure)
    - ΔNFR_max = 0: Returns 1.0 (degenerate case, no pressure)

    **TNFR Context:**

    C(t) is the primary metric for measuring IL (Coherence) operator
    effectiveness. When IL is applied, C(t) should increase as ΔNFR
    becomes more uniformly distributed (ideally all approaching zero).

    See Also
    --------
    compute_local_coherence : Local coherence for node neighborhoods
    compute_coherence : Alternative coherence metric (legacy)

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.metrics.coherence import compute_global_coherence
    >>> from tnfr.constants import DNFR_PRIMARY
    >>> G = nx.Graph()
    >>> G.add_nodes_from([1, 2, 3])
    >>> G.nodes[1][DNFR_PRIMARY] = 0.1
    >>> G.nodes[2][DNFR_PRIMARY] = 0.2
    >>> G.nodes[3][DNFR_PRIMARY] = 0.15
    >>> C_global = compute_global_coherence(G)
    >>> 0.0 <= C_global <= 1.0
    True
    """
    # Collect all ΔNFR values
    dnfr_values = [float(get_attr(G.nodes[n], ALIAS_DNFR, 0.0)) for n in G.nodes()]

    if not dnfr_values or all(v == 0 for v in dnfr_values):
        return 1.0  # Perfect coherence when no reorganization pressure

    np = get_numpy()
    if np is not None:
        dnfr_array = np.array(dnfr_values)
        sigma_dnfr = float(np.std(dnfr_array))
        dnfr_max = float(np.max(dnfr_array))
    else:
        # Pure Python fallback
        mean_dnfr = sum(dnfr_values) / len(dnfr_values)
        variance = sum((v - mean_dnfr) ** 2 for v in dnfr_values) / len(dnfr_values)
        sigma_dnfr = variance**0.5
        dnfr_max = max(dnfr_values)

    if dnfr_max == 0:
        return 1.0

    C_t = 1.0 - (sigma_dnfr / dnfr_max)

    # Clamp to [0, 1] to handle numerical edge cases
    if np is not None:
        return float(np.clip(C_t, 0.0, 1.0))
    return max(0.0, min(1.0, C_t))


def compute_local_coherence(G: TNFRGraph, node: Any, radius: int = 1) -> float:
    """Compute local coherence for node and its neighborhood.

    Local coherence applies the same C(t) formula to a neighborhood subgraph:
    C_local(t) = 1 - (σ_ΔNFR_local / ΔNFR_max_local)

    This measures structural stability within a node's local vicinity, useful
    for identifying coherence gradients and structural weak points in networks.

    Parameters
    ----------
    G : TNFRGraph
        Network graph
    node : Any
        Central node for local coherence computation
    radius : int, default=1
        Neighborhood radius:
        - 1 = immediate neighbors (default)
        - 2 = neighbors + neighbors-of-neighbors
        - etc.

    Returns
    -------
    float
        Local coherence value in [0, 1] where:
        - 1.0 = perfect local coherence
        - 0.0 = maximum local incoherence

    Notes
    -----
    **Use Cases:**

    - **Hotspot Detection**: Identify regions of structural instability
    - **IL Targeting**: Prioritize nodes needing coherence stabilization
    - **Network Health**: Monitor local vs. global coherence balance
    - **Bifurcation Risk**: Low local C(t) may predict structural splits

    **Radius Selection:**

    - **radius=1**: Fast, captures immediate structural environment
    - **radius=2**: Better for mesoscale patterns, slower
    - **radius>2**: Approaches global coherence, expensive

    **Special Cases:**

    - Isolated node (no neighbors): Returns 1.0
    - All neighborhood ΔNFR = 0: Returns 1.0
    - Single-node neighborhood: Returns 1.0 (no variance)

    See Also
    --------
    compute_global_coherence : Global network coherence

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.metrics.coherence import compute_local_coherence
    >>> from tnfr.constants import DNFR_PRIMARY
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4)])
    >>> for n in [1, 2, 3, 4]:
    ...     G.nodes[n][DNFR_PRIMARY] = 0.1 * n
    >>> C_local = compute_local_coherence(G, node=2, radius=1)
    >>> 0.0 <= C_local <= 1.0
    True
    """
    import networkx as nx

    # Get neighborhood
    if radius == 1:
        neighbors = set(G.neighbors(node)) | {node}
    else:
        neighbors = set(nx.single_source_shortest_path_length(G, node, cutoff=radius).keys())

    # Collect ΔNFR for neighborhood
    dnfr_values = [float(get_attr(G.nodes[n], ALIAS_DNFR, 0.0)) for n in neighbors]

    if not dnfr_values or all(v == 0 for v in dnfr_values):
        return 1.0

    np = get_numpy()
    if np is not None:
        dnfr_array = np.array(dnfr_values)
        sigma_dnfr = float(np.std(dnfr_array))
        dnfr_max = float(np.max(dnfr_array))
    else:
        # Pure Python fallback
        mean_dnfr = sum(dnfr_values) / len(dnfr_values)
        variance = sum((v - mean_dnfr) ** 2 for v in dnfr_values) / len(dnfr_values)
        sigma_dnfr = variance**0.5
        dnfr_max = max(dnfr_values)

    if dnfr_max == 0:
        return 1.0

    C_local = 1.0 - (sigma_dnfr / dnfr_max)

    # Clamp to [0, 1]
    if np is not None:
        return float(np.clip(C_local, 0.0, 1.0))
    return max(0.0, min(1.0, C_local))
