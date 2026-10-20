"""Explicit delayed REMESH and auxiliary structural-memory operations.

Execution paths
---------------
The registered Recursivity glyph is an advisory owned by
``_recursivity_stage_kernel``. It does not run this module's delayed map,
create replicas or perform topology replacement. ``apply_network_remesh``
separately applies one supplied-history EPI mixing operation on fixed ordered
node support. Explicit topology and identity-propagation helpers have their
own state changes and admission contracts; they are not implicit glyph effects.

Delayed-state map
----------------
For declared alpha and two retained snapshots, the ideal unclipped row is

    EPI_next = (1-alpha)**2 * EPI_now
             + alpha*(1-alpha) * EPI_local + alpha * EPI_global.

The shared ``_delayed_remesh_kernel`` separates that exact-real expression,
its nested binary64 evaluation and the configured clipping policy. Delayed
mixing does not itself write capacity, phase, stored pressure or topology.
Pressure refresh is a separately declared operation. Integer delays select
retained sample positions; they denote fixed elapsed durations only with an
explicit sampling-clock convention. The map does not advance delayed history.

History, identity and evidence
------------------------------
The event/REMESH bridge owns its pre-map history append and post-map temporal
boundary. Separate finite causal wrappers bind consecutive executions to one
graph. Caller-supplied equal endpoints or histories alone do not establish
causal provenance, complete grammar-state continuity or future stability.

``StructuralIdentity`` is a configured snapshot signature. Similarity,
lineage, latency labels and a successful match are observations or policies,
not invariant NFR identities or autonomous maintenance theorems. The explicitly
requested identity-propagation helper changes form, capacity and circular
phase; its action is distinct from delayed EPI mixing. Pattern grouping,
origin selection and invocation remain supplied policies.

An environmental memory obtained by eliminating hidden dynamical state keeps
its hidden initialization and its own causal law. It cannot be replaced by
this configured finite echo without a separate correspondence proof; the P5
comparison in ``theory/DERIVED_EPI_MEMORY.md`` includes a counterexample.

Grammar and scientific scope
----------------------------
``operator_contracts`` and ``grammar_canon`` own the operator catalog and
shared grammar policies. U3 belongs to the actual coupling/resonance action;
Recursivity supplies neither Mutation's measured trigger nor its U4b context.
The delayed-map and auxiliary APIs do not establish a grammar for all their
compositions. See ``docs/contracts/OPERATOR_EVENTS.md`` and
``theory/STRUCTURAL_OPERATORS.md`` for execution, storage and occurrence
boundaries. No operator name, memory label or finite matching result proves
physical identification, universal catalog completeness or autonomous action.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import random
from collections import defaultdict, deque
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from itertools import combinations, islice
from operator import ge, le
from statistics import StatisticsError, fmean
from types import ModuleType
from typing import Any, TypeAlias, cast

from .._remesh_contract import (
    DelayedRemeshConfiguration,
    materialize_delayed_remesh_configuration,
)
from .._runtime_steps import (
    RUNTIME_STEP_NEXT_KEY,
    RUNTIME_STEP_UNSET,
    resolve_runtime_step_index,
)
from ..alias import get_attr, set_attr
from ..constants import DEFAULTS, REMESH_DEFAULTS, get_param
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..constants.canonical import REMESH_SIMILARITY_THRESHOLD_CANONICAL
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..rng import make_rng, resolve_graph_seed, validate_seed
from ..types import RemeshMeta
from ..utils import angle_diff, cached_import, edge_version_update
from ..utils._structural_signature import (
    StructuralSignatureError,
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._delayed_remesh_kernel import (
    DelayedRemeshNodeProposal,
    DelayedRemeshPlan,
    DelayedRemeshResult,
    DelayedRemeshStabilityEvidence,
    _materialize_indexed_history,
    build_delayed_remesh_plan,
)
from .factor_contracts import (
    canonical_glyph_factor_defaults,
    validate_glyph_factor,
    validate_glyph_factors,
)
from .network_stage import (
    GraphTransactionSnapshot,
    _networkx_runtime_layout,
    _runtime_class_mro,
    _runtime_mapping_items,
    _runtime_stored_attribute,
    _set_runtime_mapping_item,
)

CommunityGraph: TypeAlias = Any
NetworkxModule: TypeAlias = ModuleType
CommunityModule: TypeAlias = ModuleType
RemeshEdge: TypeAlias = tuple[Hashable, Hashable]
NetworkxModules: TypeAlias = tuple[NetworkxModule, CommunityModule]
RemeshConfigValue: TypeAlias = bool | float | int

_NUMPY_INTEGER_TYPE_NAMES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)
_NUMPY_EXACT_INTEGER_SCALAR_TYPES = (
    frozenset(
        scalar_type
        for name in _NUMPY_INTEGER_TYPE_NAMES
        if (scalar_type := getattr(np, name, None)) is not None
    )
    if np is not None
    else frozenset()
)
_NUMPY_EXACT_REAL_SCALAR_TYPES = _NUMPY_EXACT_INTEGER_SCALAR_TYPES | (
    frozenset(
        scalar_type
        for name in ("float16", "float32", "float64", "longdouble")
        if (scalar_type := getattr(np, name, None)) is not None
    )
    if np is not None
    else frozenset()
)

# ---------------------------------------------------------------------------
# Latency detection threshold (νf → 0 for SHA-frozen nodes)
# ---------------------------------------------------------------------------
_VF_FROZEN_THRESHOLD = 0.05

# ==============================================================================
# Phase 1: Structural Memory & Pattern Recognition
# ==============================================================================


def _identity_real(value: Any, label: str, *, nonnegative: bool = False) -> float:
    """Admit one consumed signature scalar before materialization."""
    from .._exact_time import bounded_represented_real

    try:
        return bounded_represented_real(
            value, f"StructuralIdentity {label}", minimum=0 if nonnegative else None
        )[0]
    except (TypeError, ValueError) as exc:
        raise TNFRValueError(str(exc)) from exc


def _identity_channel(
    node_data: Mapping[str, Any], aliases: tuple[str, ...], label: str
) -> float | None:
    """Read an authoritative channel; absence stays distinct from invalid data."""
    from ..types import require_finite_real_scalar_epi

    missing = object()
    value = get_attr(node_data, aliases, missing, strict=True, conv=lambda item: item)
    if value is missing:
        return None
    if aliases == ALIAS_EPI:
        return require_finite_real_scalar_epi(value, f"StructuralIdentity {label}")
    return _identity_real(value, label, nonnegative=aliases == ALIAS_VF)


@dataclass
class StructuralIdentity:
    """Configured nodal snapshot signature for matching reorganized states.

    Stores scalar EPI, a capacity range and an optional circular phase target.
    A match checks these read-outs against a selected tolerance; it does not
    establish an invariant set, orbital stability, autonomous maintenance or
    a complete NFR identity. Dynamic identity and permitted perturbations need
    a separate model contract.

    **REMESH ↔ SHA Relationship**: SHA (Silence) reduces capacity while
    preserving EPI at the event. Later unforced EPI evolution obeys
    ∂EPI/∂t = νf · ΔNFR; exact freezing requires that product to vanish.
    Captured low-capacity states can supply candidate structural anchors for
    REMESH, but their maintenance must be checked on the subsequent evolution.

    **Usage Pattern**:
    1. Apply SHA to reduce capacity while preserving EPI at the event
    2. Capture identity from the resulting state (this class)
    3. Apply REMESH to propagate pattern across scales
    4. Validate identity preservation post-reorganization

    Attributes
    ----------
    epi_signature : float
        Characteristic EPI pattern value (preserved when SHA applied)
    vf_range : tuple[float, float]
        Range of structural frequencies (min, max) in Hz_str
    phase_pattern : float | None
        Optional circular phase target in the shared represented-real domain
    frozen_by_sha : bool
        Low-capacity SHA capture flag; not a future invariance certificate
    lineage : list[str]
        Supplied transformation labels, not proof of identity preservation
    tolerance : float
        Maximum deviation for identity match (default: 0.1)

    Notes
    -----
    The limit νf → 0 suppresses the unforced EPI rate only when
    νf · ΔNFR → 0; bounded pressure is sufficient. For example,
    νf = 1/(1+t) and ΔNFR = 1+t instead give unit EPI rate for t >= 0.
    Even a vanishing rate alone does not prove finite accumulated change or
    identity preservation. At exact zero capacity and finite pressure the
    unforced continuous EPI rate is zero; separate sources and event writes
    require their own checks. This class stores identity evidence, not a
    dynamical memory-maintenance theorem.

    **Do NOT reimplement SHA** - use existing Silence operator from
    tnfr.operators.definitions. This class only captures and validates
    identity, it does NOT apply SHA itself.

    See Also
    --------
    tnfr.operators.definitions.Silence : SHA operator implementation
    SHA_ALGEBRA_PHYSICS.md : SHA as identity operator derivation
    """

    epi_signature: float
    vf_range: tuple[float, float]
    phase_pattern: float | None = None
    frozen_by_sha: bool = False
    lineage: list[str] = field(default_factory=list)
    tolerance: float = 0.1

    def __post_init__(self) -> None:
        self._validated_fields()

    def _validated_fields(
        self, tolerance: float | None = None
    ) -> tuple[float, float, float, float | None, bool, float]:
        """Re-admit mutable configuration without replacing its source values."""
        from .._exact_time import bounded_represented_real
        from ..config.parsing import parse_bool
        from ..types import require_finite_real_scalar_epi

        epi = require_finite_real_scalar_epi(
            self.epi_signature, "StructuralIdentity EPI signature"
        )
        if not isinstance(self.vf_range, (tuple, list)) or len(self.vf_range) != 2:
            raise TNFRValueError("StructuralIdentity capacity range needs two bounds")
        lower_source, upper_source = self.vf_range
        upper = _identity_real(upper_source, "upper capacity", nonnegative=True)
        try:
            lower = bounded_represented_real(
                lower_source,
                "StructuralIdentity lower capacity",
                minimum=0,
                maximum=upper_source,
            )[0]
            frozen = parse_bool(self.frozen_by_sha)
        except (TypeError, ValueError) as exc:
            raise TNFRValueError(str(exc)) from exc
        phase = (
            None
            if self.phase_pattern is None
            else _identity_real(self.phase_pattern, "phase target")
        )
        tol = _identity_real(
            self.tolerance if tolerance is None else tolerance,
            "tolerance",
            nonnegative=True,
        )
        return epi, lower, upper, phase, frozen, tol

    def matches(
        self, node_data: Mapping[str, Any], *, tolerance: float | None = None
    ) -> bool:
        """Check whether a nodal snapshot matches the configured signature.

        Parameters
        ----------
        node_data : Mapping
            Node attributes containing EPI, vf, and optionally phase
        tolerance : float, optional
            Override default tolerance for this check

        Returns
        -------
        bool
            True if node matches this identity within tolerance

        Notes
        -----
        If frozen_by_sha=True, the vf check admits low capacity in addition
        to the captured range. EPI matching remains a separate snapshot check;
        neither predicate proves continued dynamical freezing.

        Missing required observations return False. Invalid present channels
        or signature configuration raise TNFRValueError; an invalid earlier
        alias cannot fall through to a later value. Only a declared phase
        target consumes phase. The signature and node mapping remain unchanged.
        """
        epi, vf_min, vf_max, phase, frozen, tol = self._validated_fields(tolerance)
        node_epi = _identity_channel(node_data, ALIAS_EPI, "EPI")
        node_vf = _identity_channel(node_data, ALIAS_VF, "capacity")
        node_phase = (
            _identity_channel(node_data, ALIAS_THETA, "phase")
            if phase is not None
            else None
        )
        if (
            node_epi is None
            or node_vf is None
            or (phase is not None and node_phase is None)
        ):
            return False

        # Exact represented comparisons avoid overflow in finite differences
        # or in tolerance-expanded capacity endpoints.
        exact_tol = Fraction.from_float(tol)
        if abs(Fraction.from_float(node_epi) - Fraction.from_float(epi)) > exact_tol:
            return False
        low_capacity = frozen and node_vf < _VF_FROZEN_THRESHOLD
        if not low_capacity and not (
            Fraction.from_float(vf_min) - exact_tol
            <= Fraction.from_float(node_vf)
            <= Fraction.from_float(vf_max) + exact_tol
        ):
            return False
        if phase is not None:
            # Use the shared circular reader directly. Separately reducing
            # tiny negative representatives can erase their real difference.
            return abs(angle_diff(node_phase, phase)) <= tol
        return True

    def record_transformation(self, operation: str) -> None:
        """Record a structural transformation in this identity's lineage.

        Parameters
        ----------
        operation : str
            Description of the transformation applied
        """
        self.lineage.append(operation)

    @classmethod
    def capture_from_node(
        cls,
        node_data: Mapping[str, Any],
        *,
        is_sha_frozen: bool = False,
        tolerance: float = 0.1,
    ) -> "StructuralIdentity":
        """Capture structural identity from a node's current state.

        EPI and nonnegative capacity are required. An absent optional phase
        remains None; an invalid present phase rejects. No node state is
        written, and the capture does not establish a maintained NFR identity.

        Parameters
        ----------
        node_data : Mapping
            Node attributes containing EPI, vf, phase
        is_sha_frozen : bool, default=False
            Supplied low-capacity capture label. This does not apply SHA,
            authenticate its occurrence or prove later freezing.
        tolerance : float, default=0.1
            Tolerance for future identity matching

        Returns
        -------
        StructuralIdentity
            Captured structural identity

        Notes
        -----
        **REMESH ↔ SHA Integration Pattern**:

        1. Apply SHA to reduce capacity, if that intervention is intended.
        2. Capture the resulting available channels.
        3. Invoke the selected separate REMESH or propagation operation.
        4. Use identity.matches() to compare the new snapshot.

        SHA remains a separate operator with its own admission and lifecycle.

        Example
        -------
        >>> from tnfr.operators.definitions import Silence
        >>> from tnfr.structural import run_sequence
        >>> # Step 1: Apply SHA to reduce capacity
        >>> run_sequence(G, node, [Silence()])
        >>> # Step 2: Capture the supplied low-capacity label and snapshot
        >>> identity = StructuralIdentity.capture_from_node(
        ...     G.nodes[node],
        ...     is_sha_frozen=True
        ... )
        >>> # Step 3: Invoke the intended separate REMESH operation
        >>> # ... REMESH operations ...
        >>> # Step 4: Compare the available snapshot channels
        >>> assert identity.matches(G.nodes[node])
        """
        from ..config.parsing import parse_bool

        epi = _identity_channel(node_data, ALIAS_EPI, "EPI")
        vf = _identity_channel(node_data, ALIAS_VF, "capacity")
        phase = _identity_channel(node_data, ALIAS_THETA, "phase")
        if epi is None or vf is None:
            raise TNFRValueError("StructuralIdentity capture requires EPI and capacity")
        try:
            frozen = parse_bool(is_sha_frozen)
        except (TypeError, ValueError) as exc:
            raise TNFRValueError(str(exc)) from exc

        # For vf_range, use small window around current value
        # (unless SHA-frozen, in which case we expect νf ≈ 0)
        vf_tolerance = 0.1
        vf_min = max(0.0, vf - vf_tolerance)
        vf_max = vf + vf_tolerance

        identity = cls(
            epi_signature=epi,
            vf_range=(vf_min, vf_max),
            phase_pattern=phase,
            frozen_by_sha=frozen,
            tolerance=tolerance,
        )

        if frozen:
            identity.record_transformation(
                "Captured with supplied low-capacity SHA label"
            )

        return identity


def structural_similarity(
    epi1: float | Sequence[float],
    epi2: float | Sequence[float],
    *,
    metric: str = "euclidean",
) -> float:
    """Compute structural similarity between two EPI patterns.

    Implements pattern matching for structural memory recognition.
    Returns a similarity score in [0, 1] where 1 = identical patterns.

    Parameters
    ----------
    epi1, epi2 : float or array-like
        EPI patterns to compare. Can be scalars or vectors.
    metric : {'euclidean', 'cosine', 'correlation'}
        Distance/similarity metric to use

    Returns
    -------
    float
        Similarity score in [0, 1], where 1 indicates identical patterns

    Notes
    -----
    This function is fundamental to REMESH's structural memory capability,
    enabling pattern recognition across network scales.

    Examples
    --------
    >>> structural_similarity(0.5, 0.52)  # Nearly identical scalars
    0.98
    >>> structural_similarity([0.5, 0.3], [0.52, 0.31])  # Similar vectors
    0.97
    """
    # Convert to numpy arrays for consistent handling
    if np is None:
        # Fallback: scalar comparison only
        if isinstance(epi1, (list, tuple)) or isinstance(epi2, (list, tuple)):
            raise ImportError(
                "NumPy required for vector EPI comparison. "
                "Install numpy: pip install numpy"
            )
        # Simple scalar distance -> similarity
        distance = abs(float(epi1) - float(epi2))
        # Map distance to similarity using exponential decay
        # similarity = exp(-k * distance) where k controls sensitivity
        import math

        k = 5.0  # Sensitivity parameter (higher = stricter matching)
        return math.exp(-k * distance)

    # NumPy available - use vector operations
    arr1 = np.atleast_1d(np.asarray(epi1, dtype=float))
    arr2 = np.atleast_1d(np.asarray(epi2, dtype=float))

    if arr1.shape != arr2.shape:
        raise TNFRValueError(
            f"EPI patterns must have same shape: {arr1.shape} vs {arr2.shape}",
            context={"shape1": arr1.shape, "shape2": arr2.shape},
        )

    if metric == "euclidean":
        distance = np.linalg.norm(arr1 - arr2)
        # Normalize by dimension for fair comparison across scales
        distance /= np.sqrt(len(arr1))
        # Convert to similarity
        import math

        k = 5.0
        return float(math.exp(-k * distance))

    elif metric == "cosine":
        # Cosine similarity: (a · b) / (||a|| ||b||)
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0  # Zero vectors have no meaningful similarity
        similarity = dot_product / (norm1 * norm2)
        # Map [-1, 1] to [0, 1]
        return float((similarity + 1.0) / 2.0)

    elif metric == "correlation":
        # Pearson correlation coefficient
        if len(arr1) < 2:
            # Fall back to euclidean for scalars
            return structural_similarity(epi1, epi2, metric="euclidean")
        corr_matrix = np.corrcoef(arr1, arr2)
        correlation = corr_matrix[0, 1]
        # Handle NaN (constant arrays)
        if not np.isfinite(correlation):
            correlation = 1.0 if np.allclose(arr1, arr2) else 0.0
        # Map [-1, 1] to [0, 1]
        return float((correlation + 1.0) / 2.0)

    else:
        raise TNFRValueError(
            f"Unknown metric '{metric}'. Choose from: euclidean, cosine, correlation",
            context={"metric": metric},
            suggestion="Choose from: euclidean, cosine, correlation",
        )


def structural_memory_match(
    G: CommunityGraph,
    source_node: Hashable,
    target_nodes: Iterable[Hashable] | None = None,
    *,
    threshold: float = REMESH_SIMILARITY_THRESHOLD_CANONICAL,
    metric: str = "euclidean",
) -> list[tuple[Hashable, float]]:
    """Identify nodes with EPI patterns similar to source node.

    Implements REMESH's structural memory: recognizing coherent patterns
    across the network that resonate with the source pattern.

    Parameters
    ----------
    G : TNFRGraph
        Network containing nodes with EPI attributes
    source_node : Hashable
        Node whose pattern to match against
    target_nodes : Iterable, optional
        Nodes to search. If None, searches all nodes except source.
    threshold : float, default=0.75
        Minimum similarity score for a match
    metric : str, default='euclidean'
        Similarity metric (see structural_similarity)

    Returns
    -------
    list of (node, similarity) tuples
        Nodes matching the source pattern, sorted by similarity (highest first)

    Notes
    -----
    This function is critical for REMESH's ability to propagate patterns
    coherently across scales, as specified in TNFR theoretical foundation.
    """
    source_epi = _as_float(get_attr(G.nodes[source_node], ALIAS_EPI, 0.0))

    if target_nodes is None:
        target_nodes = [n for n in G.nodes() if n != source_node]

    matches = []
    for target in target_nodes:
        if target == source_node:
            continue
        target_epi = _as_float(get_attr(G.nodes[target], ALIAS_EPI, 0.0))
        similarity = structural_similarity(source_epi, target_epi, metric=metric)
        if similarity >= threshold:
            matches.append((target, similarity))

    # Sort by similarity descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def compute_structural_signature(
    G: CommunityGraph,
    node: Hashable,
) -> Any:
    """Compute multidimensional structural signature of a node.

    The signature is an operational seven-feature identity proxy containing:
    - EPI: coherence magnitude
    - νf: structural frequency
    - θ: phase
    - ΔNFR: reorganization gradient
    - Topological properties: degree, local clustering

    This lossy signature enables REMESH's structural-memory heuristic for
    recognizing similar patterns across nodes and scales. It does not reconstruct
    the complete node, neighbourhood, or graph state.

    Parameters
    ----------
    G : TNFRGraph
        Network containing the node
    node : Hashable
        Node identifier whose signature to compute

    Returns
    -------
    signature : ndarray, shape (n_features,)
        Normalized structural signature vector. If NumPy unavailable, returns
        a tuple of features.

    Notes
    -----
    The signature is normalized to unit length for consistent similarity
    comparisons across different network scales and configurations.

    Features included:
    1. EPI magnitude (coherence)
    2. νf (structural frequency in Hz_str)
    3. sin(θ), cos(θ) (circular phase representation)
    4. ΔNFR (reorganization gradient)
    5. Normalized degree (relative connectivity)
    6. Local clustering coefficient (triadic closure)

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_node(1, EPI=0.5, vf=1.0, theta=0.2, DNFR=0.1)
    >>> G.add_edge(1, 2)
    >>> sig = compute_structural_signature(G, 1)
    >>> len(sig)  # 7 features
    7
    """
    # Extract TNFR structural attributes
    epi = _as_float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    vf = _as_float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    from ..constants.aliases import ALIAS_THETA

    theta = _as_float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))
    dnfr = _as_float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

    # Compute topological features
    degree = G.degree(node) if G.has_node(node) else 0
    n_nodes = G.number_of_nodes()
    normalized_degree = degree / n_nodes if n_nodes > 0 else 0.0

    # Local clustering coefficient (requires networkx)
    try:
        nx, _ = _get_networkx_modules()
        clustering = nx.clustering(G, node)
    except Exception:
        clustering = 0.0

    # Build feature vector
    import math

    features = [
        epi,
        vf,
        math.sin(theta),  # Circular phase representation
        math.cos(theta),
        dnfr,
        normalized_degree,
        clustering,
    ]

    # Try to use NumPy for normalization
    if np is not None:
        signature = np.array(features, dtype=float)
        norm = np.linalg.norm(signature)
        if norm > 1e-10:
            signature = signature / norm
        return signature
    else:
        # Fallback: manual normalization
        norm = math.sqrt(sum(f * f for f in features))
        if norm > 1e-10:
            features = [f / norm for f in features]
        return tuple(features)


def detect_recursive_patterns(
    G: CommunityGraph,
    threshold: float = REMESH_SIMILARITY_THRESHOLD_CANONICAL,
    metric: str = "cosine",
    min_cluster_size: int = 2,
) -> list[list[Hashable]]:
    """Detect groups of nodes with similar structural patterns.

    These groups represent the same EPI pattern replicated across different
    nodes/scales in the network. This is fundamental to REMESH's capability
    to recognize and propagate structural identity.

    Parameters
    ----------
    G : TNFRGraph
        Network to analyze
    threshold : float, default=0.75
        Minimum similarity score (0-1) to consider patterns as "same identity".
        Higher values require stricter matching.
    metric : str, default='cosine'
        Similarity metric for signature comparison.
        Options: 'cosine', 'euclidean', 'correlation'
    min_cluster_size : int, default=2
        Minimum number of nodes required to consider as a recursive pattern.
        Single nodes are not patterns.

    Returns
    -------
    clusters : list of list
        Each cluster contains nodes sharing a structural pattern.
        Clusters are independent - nodes appear in at most one cluster.

    Notes
    -----
    Algorithm uses greedy clustering:
    1. Compute structural signatures for all nodes
    2. For each unvisited node, find all similar nodes (similarity >= threshold)
    3. Form cluster if size >= min_cluster_size
    4. Mark all cluster members as visited

    This implements TNFR's principle: "A node can recognize itself in other
    nodes" through structural resonance.

    Examples
    --------
    >>> # Network with two groups of similar nodes
    >>> clusters = detect_recursive_patterns(G, threshold=0.8)
    >>> len(clusters)
    2
    >>> all(len(c) >= 2 for c in clusters)
    True
    """
    nodes = list(G.nodes())
    n = len(nodes)

    if n < min_cluster_size:
        return []

    # Compute all signatures
    signatures = {}
    for node in nodes:
        signatures[node] = compute_structural_signature(G, node)

    # Compute similarity matrix (only upper triangle needed)
    if np is not None:
        # Use NumPy for efficient computation
        sig_array = np.array([signatures[node] for node in nodes])

        if metric == "cosine":
            # Cosine similarity matrix
            norms = np.linalg.norm(sig_array, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Avoid division by zero
            normalized = sig_array / norms
            similarities = np.dot(normalized, normalized.T)
        elif metric == "euclidean":
            # Euclidean distance -> similarity
            from scipy.spatial.distance import pdist, squareform

            distances = squareform(pdist(sig_array, metric="euclidean"))
            max_dist = np.sqrt(2)  # Max distance for unit vectors
            similarities = 1.0 - (distances / max_dist)
        elif metric == "correlation":
            # Pearson correlation
            similarities = np.corrcoef(sig_array)
            # Handle NaN (constant signatures)
            similarities = np.nan_to_num(similarities, nan=0.0)
        else:
            raise TNFRValueError(
                f"Unknown metric: {metric}",
                context={"metric": metric},
                suggestion="Choose from: euclidean, cosine, correlation",
            )
    else:
        # Fallback: pairwise computation
        similarities = {}
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i == j:
                    similarities[(i, j)] = 1.0
                elif i < j:
                    # Use existing structural_similarity function
                    sim = structural_similarity(
                        signatures[node1], signatures[node2], metric=metric
                    )
                    similarities[(i, j)] = sim
                    similarities[(j, i)] = sim

    # Greedy clustering
    visited = set()
    clusters = []

    for i, node in enumerate(nodes):
        if node in visited:
            continue

        # Find all similar nodes
        cluster = [node]
        visited.add(node)

        for j, other_node in enumerate(nodes):
            if other_node in visited:
                continue

            # Get similarity
            if np is not None:
                sim = similarities[i, j]
            else:
                sim = similarities.get((i, j), 0.0)

            if sim >= threshold:
                cluster.append(other_node)
                visited.add(other_node)

        # Only keep clusters meeting minimum size
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


def identify_pattern_origin(
    G: CommunityGraph,
    cluster: Sequence[Hashable],
) -> Hashable | None:
    """Identify the origin node of a recursive structural pattern.

    The origin is the node with the strongest structural manifestation,
    determined by combining coherence (EPI) and reorganization capacity (νf).
    This represents the "source" from which the pattern can most coherently
    propagate.

    Parameters
    ----------
    G : TNFRGraph
        Network containing the cluster
    cluster : sequence of node identifiers
        Nodes sharing the same structural pattern

    Returns
    -------
    origin_node : Hashable or None
        Node identified as pattern origin (highest structural strength).
        Returns None if cluster is empty.

    Notes
    -----
    Structural strength score = EPI × νf

    This metric captures:
    - EPI: How coherent the pattern is (magnitude)
    - νf: How actively the pattern reorganizes (frequency)

    High score indicates a node that both maintains strong coherence AND
    has high reorganization capacity, making it ideal for propagation.

    Physical interpretation: The origin node is the "loudest" instance of
    the pattern in the network's structural field.

    Examples
    --------
    >>> cluster = [1, 2, 3]  # Nodes with similar patterns
    >>> origin = identify_pattern_origin(G, cluster)
    >>> # Origin will be node with highest EPI × νf
    """
    if not cluster:
        return None

    # Compute structural strength for each node
    scores = []
    for node in cluster:
        epi = _as_float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        vf = _as_float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
        # Structural strength = coherence × reorganization capacity
        strength = epi * vf
        scores.append((strength, node))

    # Return node with maximum strength
    scores.sort(reverse=True)
    return scores[0][1]


def propagate_structural_identity(
    G: CommunityGraph,
    origin_node: Hashable,
    target_nodes: Sequence[Hashable],
    propagation_strength: float = 0.5,
) -> None:
    """Apply an explicitly requested structural-memory interpolation event.

    This auxiliary writer is separate from delayed-EPI REMESH and from a
    continuous solution of the nodal equation. It holds stored pressure and
    physical time unchanged; it does not preserve an arbitrary capacity law
    ``nu_f = g(EPI)`` or establish identity persistence.

    Strength must be a represented real in [0, 1]. Zero is a true no-op and
    consumes no nodal state or clipping configuration. Otherwise all unique
    non-origin targets use the same origin snapshot, with every consumed input
    and proposal validated before the first write. Repeated targets act once.
    Missing channels are rejected, not initialized. EPI admits raw reals or
    materialized uniform-real BEPI and is written in the scalar chart.

    Form and capacity use the exact convex combination of represented inputs,
    rounded once to binary64; the shared EPI boundary policy then applies.
    Phase follows the shortest circular arc and is stored in [0, 2*pi).
    The shared ``angle_diff`` signed antipodal convention chooses a direction
    at a tie; this is an execution convention, not a uniquely emergent law.
    Strength one copies capacity and circular phase; EPI still obeys clipping.
    ``structural_lineage`` records the actual jump and runtime step ordinal,
    which is not a measured elapsed duration. Custom mapping/setter failures
    are outside the preflight guarantee for invalid supplied inputs.
    """
    from numbers import Integral, Real

    from .._exact_time import finite_represented_real
    from ..alias import set_theta, set_vf
    from ..dynamics.structural_clip import resolve_clip_policy, structural_clip
    from ..glyph_history import current_step_idx
    from ..mathematics import BEPIElement
    from ..types import require_finite_real_scalar_epi
    from ..utils import mark_dnfr_prep_dirty
    from ._argument_validation import require_list_sink

    operator = "REMESH structural memory"

    def finite(value: Any, label: str) -> float:
        try:
            return finite_represented_real(value, f"{operator} {label}")[0]
        except (TypeError, ValueError) as exc:
            raise TNFRValueError(str(exc)) from exc

    strength = finite(propagation_strength, "propagation_strength")
    if not 0.0 <= strength <= 1.0:
        raise TNFRValueError("propagation_strength must lie in [0, 1]")
    if strength == 0.0:
        return
    targets = tuple(dict.fromkeys(n for n in target_nodes if n != origin_node))
    if not targets:
        return

    missing = object()

    def read(node: Hashable, aliases: tuple[str, ...], label: str) -> float:
        value = get_attr(
            G.nodes[node], aliases, missing, strict=True, conv=lambda item: item
        )
        if value is missing:
            raise TNFRValueError(f"{operator} node {node!r} is missing {label}")
        if aliases == ALIAS_EPI:
            if isinstance(value, Real):
                value = finite(value, label)
            elif not isinstance(value, BEPIElement):
                raise TNFRValueError("EPI requires a real scalar or materialized BEPI")
            value = require_finite_real_scalar_epi(value, label)
        result = finite(value, label)
        if aliases == ALIAS_VF and result < 0.0:
            raise TNFRValueError("structural memory requires nonnegative capacity")
        return result

    origin = tuple(
        read(origin_node, aliases, label)
        for aliases, label in (
            (ALIAS_EPI, "EPI"),
            (ALIAS_VF, "nu_f"),
            (ALIAS_THETA, "theta"),
        )
    )
    lo, hi, mode, knee = resolve_clip_policy(G.graph)
    step = current_step_idx(G)
    trig_version = G.graph.get("_trig_version", 0)
    if isinstance(trig_version, bool) or not isinstance(trig_version, Integral):
        raise TNFRValueError("_trig_version must be an integer")
    alpha = Fraction.from_float(strength)

    def blend(left: float, right: float, label: str) -> float:
        return finite(
            (1 - alpha) * Fraction.from_float(left)
            + alpha * Fraction.from_float(right),
            label,
        )

    proposals = []
    for target in targets:
        epi = read(target, ALIAS_EPI, "EPI")
        capacity = read(target, ALIAS_VF, "nu_f")
        theta = read(target, ALIAS_THETA, "theta")
        require_list_sink(G.nodes[target], "structural_lineage", operator=operator)
        raw_epi = blend(epi, origin[0], "EPI proposal")
        next_epi = finite(
            structural_clip(raw_epi, lo, hi, mode, knee), "bounded EPI proposal"
        )
        next_capacity = blend(capacity, origin[1], "capacity proposal")
        target_phase, origin_phase = theta % math.tau, origin[2] % math.tau
        next_phase = (
            origin_phase
            if strength == 1.0
            else (target_phase + strength * angle_diff(origin_phase, target_phase))
            % math.tau
        )
        next_phase = finite(next_phase, "phase proposal")
        proposals.append(
            (
                target,
                next_epi,
                next_capacity,
                next_phase,
                {
                    "origin": origin_node,
                    "step": step,
                    "scope": "auxiliary_structural_memory_event",
                    "propagation_strength": strength,
                    "epi_before": epi,
                    "epi_unclipped": raw_epi,
                    "epi_after": next_epi,
                    "vf_before": capacity,
                    "vf_after": next_capacity,
                    "theta_before": theta,
                    "theta_after": next_phase,
                },
            )
        )

    for target, epi, capacity, theta, event in proposals:
        set_attr(G.nodes[target], ALIAS_EPI, epi)
        set_vf(G, target, capacity, update_max=False)
        set_theta(G, target, theta)
        G.nodes[target].setdefault("structural_lineage", []).append(event)
    G.graph.pop("_vfmax", None)
    G.graph.pop("_vfmax_node", None)
    mark_dnfr_prep_dirty(G)


# ==============================================================================
# Phase 2: Coherence Preservation & Fidelity Validation
# ==============================================================================


class RemeshCoherenceLossError(Exception):
    """Raised when REMESH reorganization loses structural coherence.

    REMESH must preserve coherence during reorganization. This error indicates
    that the structural fidelity dropped below acceptable thresholds, violating
    TNFR's requirement that "coherence propagates structurally, not imposed."
    """

    def __init__(
        self,
        fidelity: float,
        min_fidelity: float,
        details: dict[str, Any] | None = None,
    ):
        """Initialize coherence loss error.

        Parameters
        ----------
        fidelity : float
            Measured structural fidelity (coherence_after / coherence_before)
        min_fidelity : float
            Minimum required fidelity threshold
        details : dict, optional
            Additional diagnostic information
        """
        self.fidelity = fidelity
        self.min_fidelity = min_fidelity
        self.details = details or {}

        super().__init__(
            f"REMESH coherence loss: structural fidelity {fidelity:.2%} "
            f"< minimum {min_fidelity:.2%}\n"
            f"  Details: {details}"
        )


def validate_coherence_preservation(
    G_before: CommunityGraph,
    G_after: CommunityGraph,
    *,
    min_fidelity: float = 0.85,
    rollback_on_failure: bool = False,
) -> float:
    """Validate that reorganization preserved structural coherence.

    Implements TNFR requirement that REMESH reorganization must occur
    "without loss of coherence" - the total structural stability must
    be maintained within acceptable bounds.

    This function uses the canonical TNFR coherence computation from
    tnfr.metrics.common.compute_coherence() which is based on ΔNFR and dEPI.

    Parameters
    ----------
    G_before : TNFRGraph
        Network state before reorganization
    G_after : TNFRGraph
        Network state after reorganization
    min_fidelity : float, default=0.85
        Minimum acceptable fidelity (coherence_after / coherence_before)
    rollback_on_failure : bool, default=False
        If True and fidelity check fails, raise RemeshCoherenceLossError

    Returns
    -------
    float
        Structural fidelity score (coherence_after / coherence_before)

    Raises
    ------
    RemeshCoherenceLossError
        If fidelity < min_fidelity and rollback_on_failure=True

    Notes
    -----
    Structural fidelity ≈ 1.0 indicates perfect coherence preservation.
    Fidelity > 1.0 is possible (reorganization increased coherence).
    Fidelity < min_fidelity indicates unacceptable coherence loss.

    Uses canonical TNFR coherence: C = 1/(1 + |ΔNFR|_mean + |dEPI|_mean)
    from tnfr.metrics.common module.
    """
    # Use canonical TNFR coherence computation
    from ..metrics.common import compute_coherence

    coherence_before = compute_coherence(G_before)
    coherence_after = compute_coherence(G_after)

    if coherence_before < 1e-10:
        # Edge case: network had no coherence to begin with
        return 1.0

    structural_fidelity = coherence_after / coherence_before

    if rollback_on_failure and structural_fidelity < min_fidelity:
        details = {
            "coherence_before": coherence_before,
            "coherence_after": coherence_after,
            "n_nodes_before": G_before.number_of_nodes(),
            "n_nodes_after": G_after.number_of_nodes(),
        }
        raise RemeshCoherenceLossError(structural_fidelity, min_fidelity, details)

    return structural_fidelity


# ==============================================================================
# Original Helper Functions
# ==============================================================================


def _as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort conversion to ``float`` returning ``default`` on failure."""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ordered_edge(u: Hashable, v: Hashable) -> RemeshEdge:
    """Return a deterministic ordering for an undirected edge."""

    return (u, v) if repr(u) <= repr(v) else (v, u)


COOLDOWN_KEY = "REMESH_COOLDOWN_WINDOW"


@cache
def _get_networkx_modules() -> NetworkxModules:
    nx = cached_import("networkx")
    if nx is None:
        raise ImportError(
            "networkx is required for network operators; install 'networkx' "
            "to enable this feature"
        )
    nx_comm = cached_import("networkx.algorithms", "community")
    if nx_comm is None:
        raise ImportError(
            "networkx.algorithms.community is required for community-based "
            "operations; install 'networkx' to enable this feature"
        )
    return cast(NetworkxModule, nx), cast(CommunityModule, nx_comm)


_RAW_MISSING = object()


def _raw_string_entry(
    mapping: MutableMapping[Any, Any],
    key: str,
    default: Any = _RAW_MISSING,
) -> tuple[bool, Any]:
    """Read one known string key without invoking mapping overrides."""

    for candidate, value in _runtime_mapping_items(mapping):
        if type(candidate) is str and candidate == key:
            return True, value
    return False, default


def _raw_graph_parameter(
    graph_mapping: MutableMapping[Any, Any],
    key: str,
) -> Any:
    """Resolve one graph configuration scalar through raw storage."""

    present, value = _raw_string_entry(graph_mapping, key)
    if present:
        return value
    return DEFAULTS[key]


def _raw_alias_entry(
    mapping: MutableMapping[Any, Any],
    aliases: tuple[str, ...],
    default: Any = _RAW_MISSING,
) -> tuple[str | None, Any]:
    """Resolve the first present canonical alias from raw mapping items."""

    items = _runtime_mapping_items(mapping)
    for alias in aliases:
        for key, value in items:
            if type(key) is str and key == alias:
                return alias, value
    return None, default


def _set_raw_alias(
    mapping: MutableMapping[Any, Any],
    aliases: tuple[str, ...],
    value: Any,
) -> None:
    """Write the first stored alias through a known mapping base."""

    alias, _previous = _raw_alias_entry(mapping, aliases)
    selected = aliases[0] if alias is None else alias
    _set_runtime_mapping_item(mapping, selected, value)


def _runtime_sequence_length(value: Any, *, label: str) -> int:
    """Return the length of a safely materialized runtime sequence."""

    return len(_materialize_indexed_history(value, label=label))


def _remesh_alpha_info(
    G: CommunityGraph,
    *,
    _layout: Any | None = None,
) -> tuple[float, str]:
    """Return a validated ``(alpha, source)`` with explicit precedence."""
    layout = _networkx_runtime_layout(G) if _layout is None else _layout
    graph_mapping = layout.graph_mapping
    _present, hard_override = _raw_string_entry(
        graph_mapping,
        "REMESH_ALPHA_HARD",
        REMESH_DEFAULTS["REMESH_ALPHA_HARD"],
    )
    if type(hard_override) is not bool:
        raise TNFRValueError("REMESH_ALPHA_HARD must be a bool")
    if hard_override:
        _present, value = _raw_string_entry(
            graph_mapping,
            "REMESH_ALPHA",
            REMESH_DEFAULTS["REMESH_ALPHA"],
        )
        return (
            validate_glyph_factor("REMESH_alpha", value),
            "REMESH_ALPHA",
        )

    _present, raw_factors = _raw_string_entry(graph_mapping, "GLYPH_FACTORS", None)
    if raw_factors is not None:
        factors = validate_glyph_factors(
            dict(_runtime_mapping_items(raw_factors)),
            glyph="REMESH",
        )
        if "REMESH_alpha" in factors:
            return factors["REMESH_alpha"], "GLYPH_FACTORS.REMESH_alpha"

    alpha_present, alpha_value = _raw_string_entry(graph_mapping, "REMESH_ALPHA")
    if alpha_present:
        return (
            validate_glyph_factor("REMESH_alpha", alpha_value),
            "REMESH_ALPHA",
        )

    default = canonical_glyph_factor_defaults()["REMESH_alpha"]
    return (
        validate_glyph_factor("REMESH_alpha", default),
        "GLYPH_FACTORS.REMESH_alpha(default)",
    )


def _materialize_network_remesh_configuration(
    G: CommunityGraph,
    *,
    _layout: Any | None = None,
) -> DelayedRemeshConfiguration:
    """Resolve and freeze every deterministic delayed-map control."""

    layout = _networkx_runtime_layout(G) if _layout is None else _layout
    graph_mapping = layout.graph_mapping
    alpha, alpha_source = _remesh_alpha_info(G, _layout=layout)
    return materialize_delayed_remesh_configuration(
        tau_local=_raw_graph_parameter(graph_mapping, "REMESH_TAU_LOCAL"),
        tau_global=_raw_graph_parameter(graph_mapping, "REMESH_TAU_GLOBAL"),
        alpha=alpha,
        alpha_source=alpha_source,
        epi_min=_raw_graph_parameter(graph_mapping, "EPI_MIN"),
        epi_max=_raw_graph_parameter(graph_mapping, "EPI_MAX"),
        clip_mode=_raw_graph_parameter(graph_mapping, "CLIP_MODE"),
    )


_REMESH_CONFIGURATION_INPUT_KEYS = (
    "REMESH_TAU_LOCAL",
    "REMESH_TAU_GLOBAL",
    "REMESH_ALPHA_HARD",
    "REMESH_ALPHA",
    "GLYPH_FACTORS",
    "EPI_MIN",
    "EPI_MAX",
    "CLIP_MODE",
)


def _remesh_configuration_input_signature(
    G: CommunityGraph,
    *,
    retained_references: list[Any] | None = None,
) -> tuple[Any, ...]:
    """Seal raw delayed-map controls without re-running numeric coercions."""

    layout = _networkx_runtime_layout(G)
    entries: list[tuple[str, bool, Any]] = []
    for key in _REMESH_CONFIGURATION_INPUT_KEYS:
        present, value = _raw_string_entry(layout.graph_mapping, key, None)
        entries.append(
            (
                key,
                present,
                structural_proof_signature(
                    value,
                    identity_sensitive=True,
                    _retained_references=retained_references,
                ),
            )
        )
    return (
        type(G),
        id(layout.graph_mapping),
        tuple(entries),
    )


def _materialize_remesh_runtime_controls(
    graph_mapping: MutableMapping[Any, Any],
) -> tuple[bool, bool, bool, int]:
    """Freeze strict logging/history controls outside the delayed-map record."""

    from ..validation.window import validate_window

    log_present, log_events = _raw_string_entry(
        graph_mapping,
        "REMESH_LOG_EVENTS",
        REMESH_DEFAULTS["REMESH_LOG_EVENTS"],
    )
    if type(log_events) is not bool:
        raise TNFRValueError("REMESH_LOG_EVENTS must be a bool")
    maxlen_present, raw_maxlen = _raw_string_entry(
        graph_mapping,
        "HISTORY_MAXLEN",
        DEFAULTS["HISTORY_MAXLEN"],
    )
    if type(raw_maxlen) is int:
        maxlen = raw_maxlen
    elif type(raw_maxlen) in _NUMPY_EXACT_INTEGER_SCALAR_TYPES:
        maxlen = int(raw_maxlen)
    else:
        raise TNFRValueError("HISTORY_MAXLEN must be a nonnegative int")
    return (
        log_present,
        log_events,
        maxlen_present,
        validate_window(maxlen),
    )


_REMESH_PROTECTED_GRAPH_MISSING = object()


def _contract_values_equal(left: Any, right: Any) -> bool:
    """Compare protected values by the shared bit-faithful structure."""

    try:
        return bool(
            structural_proof_signature(left) == structural_proof_signature(right)
        )
    except (StructuralSignatureError, TypeError, ValueError):
        return False


def _remesh_configuration_input_signatures_are_identical(
    left: tuple[Any, ...],
    right: tuple[Any, ...],
) -> bool:
    """Compare two raw configuration seals without invoking live values."""

    return bool(
        len(left) == 3
        and len(right) == 3
        and left[0] is right[0]
        and left[1] == right[1]
        and proof_stamps_are_identical(left[2], right[2])
    )


@dataclass(frozen=True, slots=True)
class _RemeshGraphSurfaceState:
    """Immutable proof of one graph-owned surface and optional list content."""

    signature: Any
    sequence_items: tuple[Any, ...] | None
    sequence_signature: Any | None
    opaque_references: tuple[Any, ...] = field(repr=False, compare=False)
    retained_references: tuple[Any, ...] = field(repr=False, compare=False)


def _snapshot_graph_surface(
    G: CommunityGraph,
    key: str,
) -> tuple[bool, Any, Any]:
    """Capture surface identity and structure without virtual mapping reads."""

    layout = _networkx_runtime_layout(G)
    present, value = _raw_string_entry(layout.graph_mapping, key)
    if not present:
        return False, _REMESH_PROTECTED_GRAPH_MISSING, None
    opaque_references = (
        G,
        *(node for node, _data in layout.node_data),
    )
    retained_references: list[Any] = []
    try:
        signature = structural_proof_signature(
            value,
            opaque_references=opaque_references,
            identity_sensitive=True,
            _retained_references=retained_references,
        )
        if list in _runtime_class_mro(type(value)):
            sequence_items = tuple(list.__iter__(value))
            sequence_signature = structural_proof_signature(
                sequence_items,
                opaque_references=opaque_references,
            )
        else:
            sequence_items = None
            sequence_signature = None
    except (StructuralSignatureError, TNFRValueError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"graph surface {key!r} cannot be frozen for REMESH"
        ) from exc
    state = _RemeshGraphSurfaceState(
        signature=signature,
        sequence_items=sequence_items,
        sequence_signature=sequence_signature,
        opaque_references=opaque_references,
        retained_references=tuple(retained_references),
    )
    return True, value, state


def _require_same_graph_surface(
    G: CommunityGraph,
    key: str,
    expected: tuple[bool, Any, Any],
) -> None:
    """Reject callback mutation of one frozen graph-owned surface."""

    observed = _snapshot_graph_surface(G, key)
    changed = observed[0] != expected[0]
    if not changed and expected[0]:
        changed = bool(
            observed[1] is not expected[1]
            or observed[2].signature != expected[2].signature
        )
    if changed:
        raise TNFRValueError(f"REMESH callback changed protected graph surface {key!r}")


def _snapshot_alias_channels(
    G: CommunityGraph,
    nodes: tuple[Hashable, ...],
) -> tuple[Any, ...]:
    """Freeze every stored capacity, pressure and phase alias."""

    aliases = (ALIAS_VF, ALIAS_DNFR, ALIAS_THETA)
    try:
        layout = _networkx_runtime_layout(G)
        observed_nodes = tuple(node for node, _data in layout.node_data)
        if structural_proof_signature(observed_nodes) != structural_proof_signature(
            nodes
        ):
            raise TNFRValueError("REMESH node support changed")
        state = tuple(
            (
                node,
                tuple(
                    tuple(
                        (
                            alias,
                            *(lambda entry: (entry[0], entry[1]))(
                                _raw_string_entry(node_data, alias, None)
                            ),
                        )
                        for alias in channel_aliases
                    )
                    for channel_aliases in aliases
                ),
            )
            for node, node_data in layout.node_data
        )
        return structural_proof_signature(
            state,
            opaque_references=(G, *nodes),
        )
    except (StructuralSignatureError, TNFRValueError, TypeError, ValueError) as exc:
        raise TNFRValueError("REMESH structural channels cannot be frozen") from exc


def _snapshot_edge_state(G: CommunityGraph) -> tuple[Any, ...]:
    """Freeze ordered edge support, keys and attributes."""

    try:
        layout = _networkx_runtime_layout(G)
        return structural_proof_signature(
            (
                layout.directed,
                layout.multigraph,
                tuple(
                    (*edge[:-1], tuple(_runtime_mapping_items(edge[-1])))
                    for edge in layout.edges
                ),
            ),
            opaque_references=(
                G,
                *(node for node, _data in layout.node_data),
            ),
        )
    except (StructuralSignatureError, TNFRValueError, TypeError, ValueError) as exc:
        raise TNFRValueError("REMESH edge state cannot be frozen") from exc


def _snapshot_epi_time_histories(
    G: CommunityGraph,
    nodes: tuple[Hashable, ...],
) -> tuple[tuple[Hashable, Any, tuple[Any, ...]], ...]:
    """Freeze canonical physical EPI histories after the right endpoint."""

    snapshots = []
    layout = _networkx_runtime_layout(G)
    if structural_proof_signature(
        tuple(node for node, _data in layout.node_data)
    ) != structural_proof_signature(nodes):
        raise TNFRValueError("REMESH EPI-time node support changed")
    for node, node_data in layout.node_data:
        _present, history = _raw_string_entry(
            node_data,
            "epi_time_history",
            None,
        )
        if type(history) is not deque:
            raise TNFRValueError(
                "REMESH EPI-time boundary did not materialize a canonical deque"
            )
        snapshots.append((node, history, tuple(deque.__iter__(history))))
    return tuple(snapshots)


def _require_same_epi_time_histories(
    G: CommunityGraph,
    expected: tuple[tuple[Hashable, Any, tuple[Any, ...]], ...],
) -> None:
    """Reject observer mutation of a recorded physical EPI boundary."""

    observed = _snapshot_epi_time_histories(
        G, tuple(node for node, _history, _samples in expected)
    )
    for current, frozen in zip(observed, expected, strict=True):
        if (
            current[0] is not frozen[0]
            or current[1] is not frozen[1]
            or structural_proof_signature(current[2])
            != structural_proof_signature(frozen[2])
        ):
            raise TNFRValueError(
                "REMESH callback changed authoritative epi_time_history"
            )


def _snapshot_runtime_clock(G: CommunityGraph) -> tuple[bool, type[Any], Any]:
    """Freeze the declared simulation-time coordinate for the REMESH jump."""

    graph_mapping = _networkx_runtime_layout(G).graph_mapping
    present, value = _raw_string_entry(graph_mapping, "_t")
    if not present:
        return False, type(None), None
    return True, type(value), structural_proof_signature(value)


def _require_same_runtime_clock(
    G: CommunityGraph,
    expected: tuple[bool, type[Any], Any],
) -> None:
    observed = _snapshot_runtime_clock(G)
    if (
        observed[0] != expected[0]
        or observed[1] is not expected[1]
        or not _contract_values_equal(observed[2], expected[2])
    ):
        raise TNFRValueError("REMESH callback changed the runtime clock")


def _snapshot_topology(G: CommunityGraph, nx: NetworkxModule) -> str | None:
    """Return a safe hash of full ordered topology and multigraph keys."""

    del nx
    try:
        layout = _networkx_runtime_layout(G)
        topology = (
            layout.directed,
            layout.multigraph,
            tuple(node for node, _data in layout.node_data),
            tuple(edge[:-1] for edge in layout.edges),
        )
        signature = structural_proof_signature(
            topology,
            opaque_references=(
                G,
                *(node for node, _data in layout.node_data),
            ),
        )
        return hashlib.blake2b(
            repr(signature).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
    except (StructuralSignatureError, TNFRValueError, TypeError, ValueError):
        return None


def _finite_remesh_mean(
    values: Sequence[float],
    *,
    label: str,
) -> float:
    """Return a finite mean without overflowing an intermediate sum."""

    if not values:
        return 0.0
    exact_mean = sum(
        (Fraction.from_float(value) for value in values),
        Fraction(0),
    ) / len(values)
    try:
        result = float(exact_mean)
    except (OverflowError, ValueError) as exc:
        raise TNFRValueError(
            f"{label} exceeds the finite binary64 aggregate range"
        ) from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} exceeds the finite binary64 aggregate range")
    return result


def _scalar_epi_state(G: CommunityGraph) -> tuple[tuple[Hashable, float], ...]:
    """Return ordered real-scalar EPI state for history-boundary detection."""

    from ._epi_domain import require_real_scalar_epi

    values: list[tuple[Hashable, float]] = []
    for node, data in _networkx_runtime_layout(G).node_data:
        _alias, raw = _raw_alias_entry(data, ALIAS_EPI, 0.0)
        values.append(
            (
                node,
                require_real_scalar_epi(
                    raw,
                    operator="Recursivity",
                    label=f"node {node!r} EPI",
                ),
            )
        )
    return tuple(values)


def _snapshot_epi_values(
    nodes: tuple[Hashable, ...],
    values: tuple[float, ...],
) -> tuple[float, str]:
    """Return the exact checksum of one already materialized EPI vector."""

    if len(nodes) != len(values):
        raise TNFRValueError("EPI snapshot support and values must align")
    records = tuple(
        (
            structural_proof_signature(node, identity_sensitive=True),
            value,
        )
        for node, value in zip(nodes, values, strict=True)
    )
    mean_val = _finite_remesh_mean(values, label="EPI snapshot mean")
    signature = structural_proof_signature(records)
    checksum = hashlib.blake2b(
        repr(signature).encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    return float(mean_val), checksum


def _snapshot_epi(G: CommunityGraph) -> tuple[float, str]:
    """Return mean and an exact binary64, hook-free node/EPI checksum."""

    nodes: list[Hashable] = []
    values: list[float] = []
    for node, data in _networkx_runtime_layout(G).node_data:
        _alias, raw = _raw_alias_entry(data, ALIAS_EPI, 0.0)
        nodes.append(node)
        values.append(_as_float(raw))
    return _snapshot_epi_values(tuple(nodes), tuple(values))


def _latest_remesh_history_scalar(
    history: Mapping[str, Any],
    key: str,
) -> float | None:
    """Read one optional finite scalar without retaining mutable metadata."""

    try:
        present, series = _raw_string_entry(history, key, None)
    except TNFRValueError as exc:
        raise TNFRValueError("graph history must be a supported mapping") from exc
    if not present:
        return None
    if series is None:
        return None
    try:
        items = _materialize_indexed_history(
            series,
            label=f"history {key!r}",
        )
        if not items:
            return None
        raw = items[-1]
    except TNFRValueError as exc:
        raise TNFRValueError(
            f"history {key!r} must be an indexed scalar series"
        ) from exc
    raw_type = type(raw)
    if (
        raw_type not in (int, float, Fraction)
        and raw_type not in _NUMPY_EXACT_REAL_SCALAR_TYPES
    ):
        raise TNFRValueError(f"history {key!r} must contain finite scalars")
    try:
        value = float(raw)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(f"history {key!r} must contain finite scalars") from exc
    if not math.isfinite(value):
        raise TNFRValueError(f"history {key!r} must contain finite scalars")
    return value


def _ensure_raw_remesh_history(
    graph_mapping: MutableMapping[Any, Any],
    *,
    maxlen: int,
) -> MutableMapping[Any, Any]:
    """Materialize canonical telemetry history through raw mapping storage."""

    from ..glyph_history import HistoryDict

    present, history = _raw_string_entry(graph_mapping, "history", None)
    replaced = False
    if maxlen == 0:
        if type(history) is HistoryDict:
            history = {
                key: (
                    list(deque.__iter__(value))
                    if deque in _runtime_class_mro(type(value))
                    else value
                )
                for key, value in _runtime_mapping_items(history)
            }
            replaced = True
        elif not present or history is None:
            history = {}
            replaced = True
        else:
            _runtime_mapping_items(history)
    else:
        history_maxlen = (
            _runtime_stored_attribute(history, "_maxlen")
            if type(history) is HistoryDict
            else None
        )
        if type(history) is not HistoryDict or history_maxlen != maxlen:
            source_items = () if history is None else _runtime_mapping_items(history)
            safe_source: dict[Any, Any] = {}
            for key, value in source_items:
                owners = _runtime_class_mro(type(value))
                if list in owners:
                    safe_source[key] = list(list.__iter__(value))
                elif deque in owners:
                    safe_source[key] = list(deque.__iter__(value))
                else:
                    safe_source[key] = value
            history = HistoryDict(safe_source, maxlen=maxlen)
            replaced = True
        else:
            # Raw writes can insert a new list without HistoryDict's normal
            # access path. Bound each such series, never the number of keys.
            for key, value in _runtime_mapping_items(history):
                owners = _runtime_class_mro(type(value))
                if list in owners:
                    series = deque(list.__iter__(value), maxlen=maxlen)
                elif deque in owners:
                    if deque.maxlen.__get__(value) == maxlen:
                        continue
                    series = deque(deque.__iter__(value), maxlen=maxlen)
                else:
                    continue
                _set_runtime_mapping_item(history, key, series)
    if replaced:
        _set_runtime_mapping_item(graph_mapping, "history", history)
    if history is None:
        raise TNFRValueError("graph history must be a mapping")
    _runtime_mapping_items(history)
    return history


def _raw_runtime_step_marker(
    graph_mapping: MutableMapping[Any, Any],
) -> Any:
    """Read the ordinal marker without dispatching custom scalar conversions."""

    present, raw = _raw_string_entry(
        graph_mapping, RUNTIME_STEP_NEXT_KEY, RUNTIME_STEP_UNSET
    )
    if (
        present
        and type(raw) is not int
        and type(raw) not in _NUMPY_EXACT_INTEGER_SCALAR_TYPES
    ):
        raise TNFRValueError(f"{RUNTIME_STEP_NEXT_KEY} must be a nonnegative integer")
    return raw


def _raw_current_step_idx(
    graph_mapping: MutableMapping[Any, Any], history: MutableMapping[Any, Any]
) -> int:
    """Read execution chronology, with retained samples as a legacy fallback."""

    next_index = _raw_runtime_step_marker(graph_mapping)
    if next_index is not RUNTIME_STEP_UNSET:
        return resolve_runtime_step_index(next_index)

    present, steps = _raw_string_entry(history, "C_steps", None)
    if not present or steps is None:
        return 0
    return _runtime_sequence_length(steps, label="history 'C_steps'")


def _append_raw_history_event(
    history: MutableMapping[Any, Any],
    meta: RemeshMeta,
    *,
    maxlen: int,
) -> Any:
    """Append one event through built-in list/deque primitives."""

    present, events = _raw_string_entry(history, "remesh_events", None)
    if not present:
        events = deque(maxlen=maxlen) if maxlen else []
        _set_runtime_mapping_item(history, "remesh_events", events)
    owners = _runtime_class_mro(type(events))
    record = dict(meta)
    if deque in owners:
        deque.append(events, record)
    elif list in owners:
        list.append(events, record)
    else:
        raise TNFRValueError(
            "canonical remesh_events telemetry must be a list or deque"
        )
    return events


def _log_remesh_event(
    G: CommunityGraph,
    meta: RemeshMeta,
    *,
    log_events: bool,
    history_maxlen: int,
) -> bool:
    """Store canonical metadata and invoke read-only REMESH observers."""
    from ..utils import CallbackEvent, CallbackSpec, callback_manager
    from .event_runtime import (
        _filtered_graph_configuration_signature,
        _networkx_cached_view_items,
        _remove_transient_networkx_cached_views,
    )

    if type(log_events) is not bool:
        raise TNFRValueError("REMESH_LOG_EVENTS must be a bool")
    layout = _networkx_runtime_layout(G)
    graph_mapping = layout.graph_mapping
    _set_runtime_mapping_item(graph_mapping, "_REMESH_META", meta)
    logged_history = None
    logged_events = None
    logged_signature = None
    if log_events:
        logged_history = _ensure_raw_remesh_history(
            graph_mapping,
            maxlen=history_maxlen,
        )
        logged_events = _append_raw_history_event(
            logged_history, meta, maxlen=history_maxlen
        )
        try:
            logged_signature = structural_proof_signature(
                logged_events,
                identity_sensitive=True,
            )
        except StructuralSignatureError as exc:
            raise TNFRValueError("canonical REMESH telemetry cannot be frozen") from exc

    callbacks_present, _callbacks = _raw_string_entry(
        graph_mapping,
        "callbacks",
        None,
    )
    observer_graph_preserved = True
    if callbacks_present:
        if type(_callbacks) not in (dict, defaultdict):
            raise TNFRValueError(
                "REMESH callback registry must use canonical mapping storage"
            )
        normalization_layout = _networkx_runtime_layout(G)
        normalization_nodes = tuple(
            node for node, _data in normalization_layout.node_data
        )
        normalization_state = (
            _filtered_graph_configuration_signature(
                G,
                excluded_keys=frozenset({"callbacks", "_callbacks_dirty"}),
                opaque_references=normalization_nodes,
            ),
            structural_proof_signature(
                (
                    normalization_layout.node_data,
                    normalization_layout.edges,
                ),
                opaque_references=normalization_nodes,
            ),
        )
        callback_registry = callback_manager._ensure_callbacks(G)
        normalized_layout = _networkx_runtime_layout(G)
        normalized_nodes = tuple(node for node, _data in normalized_layout.node_data)
        normalized_state = (
            _filtered_graph_configuration_signature(
                G,
                excluded_keys=frozenset({"callbacks", "_callbacks_dirty"}),
                opaque_references=normalization_nodes,
            ),
            structural_proof_signature(
                (normalized_layout.node_data, normalized_layout.edges),
                opaque_references=normalization_nodes,
            ),
        )
        if not proof_stamps_are_identical(
            normalization_state,
            normalized_state,
        ):
            raise TNFRValueError(
                "REMESH callback normalization changed unrelated graph state"
            )
        normalized_callbacks_present, normalized_callbacks = _raw_string_entry(
            normalized_layout.graph_mapping,
            "callbacks",
            None,
        )
        normalized_dirty_present, _normalized_dirty = _raw_string_entry(
            normalized_layout.graph_mapping,
            "_callbacks_dirty",
            None,
        )
        if (
            not normalized_callbacks_present
            or normalized_callbacks is not callback_registry
            or type(callback_registry) not in (dict, defaultdict)
            or normalized_dirty_present
        ):
            raise TNFRValueError(
                "REMESH callback registry normalization is not canonical"
            )
        opaque_callbacks: list[Any] = []
        for _event, registry in _runtime_mapping_items(callback_registry):
            for _name, spec in _runtime_mapping_items(registry):
                if type(spec) is not CallbackSpec:
                    raise TNFRValueError("REMESH callback registry is not canonical")
                opaque_callbacks.append(tuple.__getitem__(spec, 1))
        transient_views: list[tuple[str, Any]] = []
        callback_references: list[Any] = []
        _networkx_cached_view_items(G, transient_views=transient_views)
        callback_graph_state = structural_proof_signature(
            G,
            opaque_references=tuple(opaque_callbacks),
            identity_sensitive=True,
            _retained_references=callback_references,
        )
        try:
            callback_manager.invoke_callbacks(
                G,
                CallbackEvent.ON_REMESH.value,
                dict(meta),
            )
            observed_callback_graph_state = structural_proof_signature(
                G,
                opaque_references=tuple(opaque_callbacks),
                identity_sensitive=True,
            )
        finally:
            _remove_transient_networkx_cached_views(G, transient_views)
        observer_graph_preserved = proof_stamps_are_identical(
            callback_graph_state,
            observed_callback_graph_state,
        )
    if logged_history is not None:
        observed_mapping = _networkx_runtime_layout(G).graph_mapping
        history_present, observed_history = _raw_string_entry(
            observed_mapping,
            "history",
            None,
        )
        events_present, observed_events = (
            _raw_string_entry(observed_history, "remesh_events", None)
            if history_present and observed_history is logged_history
            else (False, None)
        )
        try:
            observed_signature = structural_proof_signature(
                observed_events,
                identity_sensitive=True,
            )
        except StructuralSignatureError:
            observed_signature = None
        if (
            not history_present
            or observed_history is not logged_history
            or not events_present
            or observed_events is not logged_events
            or observed_signature != logged_signature
        ):
            raise TNFRValueError(
                "REMESH callback changed canonical remesh_events telemetry"
            )
    return observer_graph_preserved


def plan_network_remesh(
    G: CommunityGraph,
    *,
    include_stability_evidence: bool = False,
    metric_weights: Mapping[Hashable, Any] | Sequence[Any] | None = None,
) -> DelayedRemeshPlan:
    """Build the explicit delayed EPI proposal without mutating the graph.

    Empty graph support and insufficient history return distinct immutable
    no-op plans. Once both guards pass, the selected delayed snapshots must be
    node mappings with
    support exactly equal to the current graph support. No absent delayed
    value is replaced with current EPI.
    """

    transaction = GraphTransactionSnapshot(G)
    retained_references: list[Any] = []
    try:
        before = structural_proof_signature(
            G,
            identity_sensitive=True,
            _retained_references=retained_references,
        )
        layout = _networkx_runtime_layout(G)
        configuration = _materialize_network_remesh_configuration(
            G,
            _layout=layout,
        )
        plan = _plan_network_remesh_from_materialized(
            G,
            layout=layout,
            configuration=configuration,
            include_stability_evidence=include_stability_evidence,
            metric_weights=metric_weights,
        )
        after = structural_proof_signature(G, identity_sensitive=True)
        if after != before:
            raise TNFRValueError(
                "REMESH planning input materialization changed graph state"
            )
    except BaseException as failure:
        transaction.restore_after_failure(G, failure)
        raise
    return plan


def _plan_network_remesh_from_materialized(
    G: CommunityGraph,
    *,
    layout: Any,
    configuration: DelayedRemeshConfiguration,
    include_stability_evidence: bool,
    metric_weights: Mapping[Hashable, Any] | Sequence[Any] | None,
) -> DelayedRemeshPlan:
    """Build a proposal from one already materialized graph/layout boundary."""

    from ..dynamics.structural_clip import structural_clip

    nodes = tuple(node for node, _data in layout.node_data)
    current_epi: dict[Hashable, Any] = {}
    for node, node_data in layout.node_data:
        _alias, value = _raw_alias_entry(node_data, ALIAS_EPI, 0.0)
        current_epi[node] = value
    history_present, history = _raw_string_entry(
        layout.graph_mapping,
        "_epi_hist",
        (),
    )
    if not history_present:
        history = ()
    return build_delayed_remesh_plan(
        node_order=nodes,
        current_epi=current_epi,
        history=history,
        tau_local=configuration.tau_local,
        tau_global=configuration.tau_global,
        alpha=configuration.alpha,
        alpha_source=configuration.alpha_source,
        epi_min=configuration.epi_min,
        epi_max=configuration.epi_max,
        clip_mode=configuration.clip_mode,
        clipper=structural_clip,
        include_stability_evidence=include_stability_evidence,
        metric_weights=metric_weights,
    )


def apply_network_remesh(
    G: CommunityGraph,
    *,
    include_stability_evidence: bool = False,
    metric_weights: Mapping[Hashable, Any] | Sequence[Any] | None = None,
) -> DelayedRemeshResult:
    """Atomically apply the separately invoked delayed REMESH EPI map.

    The Recursivity glyph stage remains advisory and never calls this function.
    This runtime operation reads two per-node temporal snapshots. It does not
    compute a network mean or append/shift delayed ``_epi_hist``. A committed
    EPI jump records its same-time right endpoint in ``epi_time_history`` so
    later Mutation evidence cannot absorb that jump into a finite secant.

    The returned immutable result distinguishes empty-support and
    insufficient-history no-ops, raw affine proposals, bounded committed
    values and optional scoped
    stability evidence. Existing callers may continue to ignore the return.
    ON_REMESH callbacks are observers of this EPI-only commit. They may not
    alter capacity, pressure, phase, topology, delayed or physical EPI history,
    the runtime clock, hybrid event log, pressure hook, deterministic map
    configuration, alpha-source provenance or canonical event telemetry.
    Violations restore graph-owned state, including callback state reachable
    from the graph. Effects outside graph and callback reachability cannot be
    retracted by the transaction.
    """

    preflight_layout = _networkx_runtime_layout(G)
    history_present, raw_history = _raw_string_entry(
        preflight_layout.graph_mapping,
        "_epi_hist",
        (),
    )
    if history_present:
        _materialize_indexed_history(raw_history)
    transaction = GraphTransactionSnapshot(G)
    try:
        retained_plan_references: list[Any] = []
        plan_state_before = structural_proof_signature(
            G,
            identity_sensitive=True,
            _retained_references=retained_plan_references,
        )
        layout = _networkx_runtime_layout(G)
        retained_configuration_references: list[Any] = []
        configuration_input_signature = _remesh_configuration_input_signature(
            G,
            retained_references=retained_configuration_references,
        )
        runtime_controls = _materialize_remesh_runtime_controls(layout.graph_mapping)
        log_events = runtime_controls[1]
        history_maxlen = runtime_controls[3]
        configuration = _materialize_network_remesh_configuration(
            G,
            _layout=layout,
        )
        plan = _plan_network_remesh_from_materialized(
            G,
            layout=layout,
            configuration=configuration,
            include_stability_evidence=include_stability_evidence,
            metric_weights=metric_weights,
        )
        if structural_proof_signature(G, identity_sensitive=True) != plan_state_before:
            raise TNFRValueError(
                "REMESH planning input materialization changed graph state"
            )
        if not plan.applied:
            return DelayedRemeshResult(status=plan.status, plan=plan)

        nx, _ = _get_networkx_modules()
        topo_hash = _snapshot_topology(G, nx)
        epi_mean_before, epi_checksum_before = _snapshot_epi_values(
            plan.node_order,
            tuple(proposal.epi_now for proposal in plan.proposals),
        )
        protected_channels = _snapshot_alias_channels(G, plan.node_order)
        protected_edges = _snapshot_edge_state(G)
        protected_epi_history = _snapshot_graph_surface(G, "_epi_hist")
        protected_event_log = _snapshot_graph_surface(G, "hybrid_event_log")
        protected_clock = _snapshot_runtime_clock(G)
        pressure_hook_present, pressure_hook = _raw_string_entry(
            layout.graph_mapping,
            "compute_delta_nfr",
            None,
        )

        for proposal, (node, node_data) in zip(
            plan.proposals,
            layout.node_data,
            strict=True,
        ):
            if proposal.node is not node:
                raise RuntimeError("REMESH proposal lost node identity binding")
            if type(node_data) is dict:
                # Preserve the long-standing patch point for exact built-in
                # mappings; factory subclasses use the raw writer below.
                set_attr(node_data, ALIAS_EPI, proposal.bounded_epi)
            else:
                _set_raw_alias(
                    node_data,
                    ALIAS_EPI,
                    proposal.bounded_epi,
                )

        _set_runtime_mapping_item(
            layout.graph_mapping,
            "_REMESH_ALPHA_SRC",
            plan.alpha_source,
        )
        epi_mean_after, epi_checksum_after = _snapshot_epi_values(
            plan.node_order,
            tuple(proposal.bounded_epi for proposal in plan.proposals),
        )
        history = _ensure_raw_remesh_history(
            layout.graph_mapping,
            maxlen=history_maxlen,
        )
        step_idx = _raw_current_step_idx(layout.graph_mapping, history)
        raw_mean_after = _finite_remesh_mean(
            tuple(proposal.raw_epi for proposal in plan.proposals),
            label="raw REMESH proposal mean",
        )
        meta: RemeshMeta = {
            "alpha": plan.alpha,
            "alpha_source": plan.alpha_source,
            "tau_global": plan.tau_global,
            "tau_local": plan.tau_local,
            "step": step_idx,
            "topo_hash": topo_hash,
            "epi_mean_before": float(epi_mean_before),
            "epi_mean_after": float(epi_mean_after),
            "epi_raw_mean_after": float(raw_mean_after),
            "epi_checksum_before": epi_checksum_before,
            "epi_checksum_after": epi_checksum_after,
            "clip_mode": plan.clip_mode,
            "clipped_node_count": sum(
                proposal.clipping_intervened for proposal in plan.proposals
            ),
        }

        for history_key, meta_key in (
            ("stable_frac", "stable_frac_last"),
            ("phase_sync", "phase_sync_last"),
            ("glyph_load_disr", "glyph_disr_last"),
        ):
            latest = _latest_remesh_history_scalar(history, history_key)
            if latest is not None:
                meta[meta_key] = latest

        from ..dynamics.runtime import _record_mutation_flow_boundary

        _record_mutation_flow_boundary(G)
        epi_time_histories = _snapshot_epi_time_histories(G, plan.node_order)
        observer_graph_preserved = _log_remesh_event(
            G,
            meta,
            log_events=log_events,
            history_maxlen=history_maxlen,
        )

        current_layout = _networkx_runtime_layout(G)
        if len(current_layout.node_data) != len(plan.node_order) or any(
            observed is not expected
            for (observed, _data), expected in zip(
                current_layout.node_data,
                plan.node_order,
                strict=True,
            )
        ):
            raise TNFRValueError(
                "REMESH callback changed graph support during the transaction"
            )
        for proposal, (_node, node_data) in zip(
            plan.proposals,
            current_layout.node_data,
            strict=True,
        ):
            _alias, raw_epi = _raw_alias_entry(node_data, ALIAS_EPI, 0.0)
            if type(raw_epi) is not float or structural_proof_signature(
                raw_epi
            ) != structural_proof_signature(proposal.bounded_epi):
                raise TNFRValueError(
                    "REMESH callback changed a committed EPI value",
                    context={"node": proposal.node},
                )
        meta_present, observed_meta = _raw_string_entry(
            current_layout.graph_mapping,
            "_REMESH_META",
            None,
        )
        if (
            not meta_present
            or observed_meta is not meta
            or not _contract_values_equal(observed_meta, meta)
        ):
            raise TNFRValueError("REMESH callback changed canonical event metadata")
        alpha_present, observed_alpha_source = _raw_string_entry(
            current_layout.graph_mapping,
            "_REMESH_ALPHA_SRC",
            None,
        )
        if not alpha_present or structural_proof_signature(
            observed_alpha_source
        ) != structural_proof_signature(plan.alpha_source):
            raise TNFRValueError("REMESH callback changed the canonical alpha source")
        if not _contract_values_equal(
            _snapshot_alias_channels(G, plan.node_order), protected_channels
        ):
            raise TNFRValueError("REMESH callback changed capacity, pressure or phase")
        if not _contract_values_equal(_snapshot_edge_state(G), protected_edges):
            raise TNFRValueError("REMESH callback changed edge support or attributes")
        _require_same_graph_surface(G, "_epi_hist", protected_epi_history)
        _require_same_graph_surface(G, "hybrid_event_log", protected_event_log)
        _require_same_runtime_clock(G, protected_clock)
        _require_same_epi_time_histories(G, epi_time_histories)
        observed_hook_present, observed_hook = _raw_string_entry(
            _networkx_runtime_layout(G).graph_mapping,
            "compute_delta_nfr",
            None,
        )
        if observed_hook_present != pressure_hook_present or (
            observed_hook_present and observed_hook is not pressure_hook
        ):
            raise TNFRValueError("REMESH callback changed the pressure-refresh hook")
        if not _remesh_configuration_input_signatures_are_identical(
            _remesh_configuration_input_signature(G),
            configuration_input_signature,
        ):
            raise TNFRValueError(
                "REMESH callback changed deterministic delayed-map configuration"
            )
        current_controls = _materialize_remesh_runtime_controls(
            current_layout.graph_mapping
        )
        if current_controls != runtime_controls:
            raise TNFRValueError(
                "REMESH callback changed logging or history configuration"
            )
        if observer_graph_preserved is False:
            raise TNFRValueError("REMESH observer changed graph-owned state")
    except BaseException as failure:
        transaction.restore_after_failure(G, failure)
        raise

    return DelayedRemeshResult(
        status="applied",
        plan=plan,
        metadata_items=tuple(meta.items()),
        epi_time_boundary_recorded=True,
    )


def apply_network_remesh_with_memory(
    G: CommunityGraph,
    *,
    enable_structural_memory: bool = True,
    similarity_threshold: float = REMESH_SIMILARITY_THRESHOLD_CANONICAL,
    similarity_metric: str = "cosine",
    propagation_strength: float = 0.5,
    min_cluster_size: int = 2,
) -> None:
    """Apply delayed REMESH followed by a supplied pattern-propagation policy.

    Similarity-based grouping, origin selection and interpolation are explicit
    operations requested by this call. They do not derive autonomous
    recognition, a maintained collective NFR identity or an environmental
    memory law from the nodal equation.

    The function performs:
    1. Standard REMESH reorganization (temporal EPI mixing)
    2. Pattern detection (find groups of structurally similar nodes)
    3. Interpolation of form, capacity and phase from each selected origin

    When delayed history is insufficient, phase 1 is a declared no-op. The
    independently requested pattern-detection/propagation phase still runs
    when enable_structural_memory is true.

    Parameters
    ----------
    G : TNFRGraph
        Network to reorganize (modified in-place)
    enable_structural_memory : bool, default=True
        Whether to apply the additional grouping and interpolation policy.
        If False, performs only standard REMESH.
    similarity_threshold : float, default=0.75
        Configured similarity threshold [0-1] for grouping snapshots.
        Passing this score does not establish identity equivalence.
    similarity_metric : str, default='cosine'
        Metric for comparing structural signatures.
        Options: 'cosine', 'euclidean', 'correlation'
    propagation_strength : float, default=0.5
        Interpolation weight [0-1] for identity propagation.
        - 0.0: No propagation (structural memory detection only)
        - 0.5: Equal interpolation weights
        - 1.0: Copy the origin's channels, with form still subject to clipping
    min_cluster_size : int, default=2
        Minimum group size used by the configured pattern detector.

    Notes
    -----
    The grouping step is a snapshot policy. Subsequent unforced evolution,
    storage balance, identity retention and complete operator-word admission
    need their own contracts. This wrapper does not acquire those properties
    merely by composing its existing helpers.

    **Workflow**:
    1. `apply_network_remesh(G)` - Standard temporal memory mixing
    2. `detect_recursive_patterns(G, threshold)` - Find similar node groups
    3. For each cluster:
       - `identify_pattern_origin(G, cluster)` - Apply the origin-selection score
       - `propagate_structural_identity(G, origin, targets)` - Apply interpolation

    **Telemetry**: Logs structural memory events under the graph history key
    structural_memory_events.
    including cluster statistics and propagation metadata.

    Examples
    --------
    >>> # Explicitly request delayed mixing and pattern propagation
    >>> apply_network_remesh_with_memory(G)
    >>>
    >>> # Strict pattern matching with gentle propagation
    >>> apply_network_remesh_with_memory(
    ...     G,
    ...     similarity_threshold=0.85,
    ...     propagation_strength=0.3
    ... )
    >>>
    >>> # Disable structural memory (standard REMESH only)
    >>> apply_network_remesh_with_memory(G, enable_structural_memory=False)
    """
    # Phase 1: Apply standard REMESH (temporal memory)
    delayed_result = apply_network_remesh(G)

    if not enable_structural_memory:
        return

    # Phase 2: Structural memory - detect and propagate patterns
    memory_epi_before = _scalar_epi_state(G)
    try:
        # Detect recursive patterns across network
        clusters = detect_recursive_patterns(
            G,
            threshold=similarity_threshold,
            metric=similarity_metric,
            min_cluster_size=min_cluster_size,
        )

        # Propagate identity from origin to similar nodes
        propagation_events = []
        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue

            # Identify strongest instance of pattern
            origin = identify_pattern_origin(G, cluster)
            if origin is None:
                continue

            # Propagate to other cluster members
            targets = [n for n in cluster if n != origin]
            if targets:
                propagate_structural_identity(
                    G,
                    origin,
                    targets,
                    propagation_strength=propagation_strength,
                )

                propagation_events.append(
                    {
                        "origin": origin,
                        "n_targets": len(targets),
                        "targets": targets[:5],  # Sample for telemetry (avoid bloat)
                    }
                )

        # Log structural memory event
        if G.graph.get("REMESH_LOG_EVENTS", REMESH_DEFAULTS["REMESH_LOG_EVENTS"]):
            from ..glyph_history import append_metric

            hist = G.graph.setdefault("history", {})
            append_metric(
                hist,
                "structural_memory_events",
                {
                    "n_clusters": len(clusters),
                    "cluster_sizes": [len(c) for c in clusters],
                    "n_propagations": len(propagation_events),
                    "propagation_events": propagation_events[:10],  # Sample
                    "similarity_threshold": similarity_threshold,
                    "similarity_metric": similarity_metric,
                    "propagation_strength": propagation_strength,
                    "min_cluster_size": min_cluster_size,
                },
            )

    except Exception as e:
        # Preserve the independently completed delayed phase.
        import warnings

        delayed_state = (
            "committed successfully"
            if delayed_result.applied
            else "was a declared insufficient-history no-op"
        )
        warnings.warn(
            f"Structural memory activation failed: {e}. "
            f"The delayed REMESH phase {delayed_state}.",
            RuntimeWarning,
            stacklevel=2,
        )
    finally:
        if _scalar_epi_state(G) != memory_epi_before:
            from ..dynamics.runtime import _record_mutation_flow_boundary

            _record_mutation_flow_boundary(G)


def _mst_edges_from_epi(
    nx: NetworkxModule,
    nodes: Sequence[Hashable],
    epi: Mapping[Hashable, float],
) -> set[RemeshEdge]:
    """Return MST edges based on absolute EPI distance."""
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_weighted_edges_from(
        (u, v, abs(epi[u] - epi[v])) for u, v in combinations(nodes, 2)
    )
    return {_ordered_edge(u, v) for u, v in nx.minimum_spanning_edges(H, data=False)}


def _knn_edges(
    nodes: Sequence[Hashable],
    epi: Mapping[Hashable, float],
    k_val: int,
    p_rewire: float,
    rnd: random.Random,
) -> set[RemeshEdge]:
    """Edges linking each node to its ``k`` nearest neighbours in EPI."""
    new_edges = set()
    node_set = set(nodes)
    for u in nodes:
        epi_u = epi[u]
        neighbours = [
            v
            for _, v in heapq.nsmallest(
                k_val,
                ((abs(epi_u - epi[v]), v) for v in nodes if v != u),
            )
        ]
        for v in neighbours:
            if rnd.random() < p_rewire:
                choices = list(node_set - {u, v})
                if choices:
                    v = rnd.choice(choices)
            new_edges.add(_ordered_edge(u, v))
    return new_edges


def _community_graph(
    comms: Iterable[Iterable[Hashable]],
    epi: Mapping[Hashable, float],
    nx: NetworkxModule,
) -> CommunityGraph:
    """Return community graph ``C`` with mean EPI per community."""
    C = nx.Graph()
    for idx, comm in enumerate(comms):
        members = list(comm)
        try:
            epi_mean = fmean(_as_float(epi.get(n)) for n in members)
        except StatisticsError:
            epi_mean = 0.0
        C.add_node(idx)
        set_attr(C.nodes[idx], ALIAS_EPI, epi_mean)
        C.nodes[idx]["members"] = members
    for i, j in combinations(C.nodes(), 2):
        w = abs(
            _as_float(get_attr(C.nodes[i], ALIAS_EPI, 0.0))
            - _as_float(get_attr(C.nodes[j], ALIAS_EPI, 0.0))
        )
        C.add_edge(i, j, weight=w)
    return cast(CommunityGraph, C)


def _community_k_neighbor_edges(
    C: CommunityGraph,
    k_val: int,
    p_rewire: float,
    rnd: random.Random,
) -> tuple[set[RemeshEdge], dict[int, int], list[tuple[int, int, int]]]:
    """Edges linking each community to its ``k`` nearest neighbours."""
    epi_vals = {n: _as_float(get_attr(C.nodes[n], ALIAS_EPI, 0.0)) for n in C.nodes()}
    ordered = sorted(C.nodes(), key=lambda v: epi_vals[v])
    new_edges = set()
    attempts = {n: 0 for n in C.nodes()}
    rewired = []
    node_set = set(C.nodes())
    for idx, u in enumerate(ordered):
        epi_u = epi_vals[u]
        left = idx - 1
        right = idx + 1
        added = 0
        while added < k_val and (left >= 0 or right < len(ordered)):
            if left < 0:
                v = ordered[right]
                right += 1
            elif right >= len(ordered):
                v = ordered[left]
                left -= 1
            else:
                if abs(epi_u - epi_vals[ordered[left]]) <= abs(
                    epi_vals[ordered[right]] - epi_u
                ):
                    v = ordered[left]
                    left -= 1
                else:
                    v = ordered[right]
                    right += 1
            original_v = v
            rewired_now = False
            if rnd.random() < p_rewire:
                choices = list(node_set - {u, original_v})
                if choices:
                    v = rnd.choice(choices)
                    rewired_now = True
            new_edges.add(_ordered_edge(u, v))
            attempts[u] += 1
            if rewired_now:
                rewired.append((u, original_v, v))
            added += 1
    return new_edges, attempts, rewired


def _community_remesh(
    G: CommunityGraph,
    epi: Mapping[Hashable, float],
    k_val: int,
    p_rewire: float,
    rnd: random.Random,
    nx: NetworkxModule,
    nx_comm: CommunityModule,
    mst_edges: Iterable[RemeshEdge],
    n_before: int,
) -> None:
    """Remesh ``G`` replacing nodes by modular communities."""
    from ..glyph_history import append_metric

    comms = list(nx_comm.greedy_modularity_communities(G))
    if len(comms) <= 1:
        with edge_version_update(G):
            G.clear_edges()
            G.add_edges_from(mst_edges)
        return
    C = _community_graph(comms, epi, nx)
    mst_c = nx.minimum_spanning_tree(C, weight="weight")
    new_edges: set[RemeshEdge] = {_ordered_edge(u, v) for u, v in mst_c.edges()}
    extra_edges, attempts, rewired_edges = _community_k_neighbor_edges(
        C, k_val, p_rewire, rnd
    )
    new_edges |= extra_edges

    extra_degrees = {idx: 0 for idx in C.nodes()}
    for u, v in extra_edges:
        extra_degrees[u] += 1
        extra_degrees[v] += 1

    with edge_version_update(G):
        G.clear_edges()
        G.remove_nodes_from(list(G.nodes()))
        for idx in C.nodes():
            data = dict(C.nodes[idx])
            G.add_node(idx, **data)
        G.add_edges_from(new_edges)

    if G.graph.get("REMESH_LOG_EVENTS", REMESH_DEFAULTS["REMESH_LOG_EVENTS"]):
        hist = G.graph.setdefault("history", {})
        mapping = {idx: C.nodes[idx].get("members", []) for idx in C.nodes()}
        append_metric(
            hist,
            "remesh_events",
            {
                "mode": "community",
                "n_before": n_before,
                "n_after": G.number_of_nodes(),
                "mapping": mapping,
                "k": int(k_val),
                "p_rewire": float(p_rewire),
                "extra_edges_added": len(extra_edges),
                "extra_edge_attempts": attempts,
                "extra_edge_degrees": extra_degrees,
                "rewired_edges": [
                    {"source": int(u), "from": int(v0), "to": int(v1)}
                    for u, v0, v1 in rewired_edges
                ],
            },
        )


def apply_topological_remesh(
    G: CommunityGraph,
    mode: str | None = None,
    *,
    k: int | None = None,
    p_rewire: float = 0.2,
    seed: int | None = None,
) -> None:
    """Approximate topological remeshing.

    When ``seed`` is ``None`` the RNG draws its base seed from
    ``G.graph['RANDOM_SEED']`` to keep runs reproducible.
    """
    if seed is None:
        base_seed = resolve_graph_seed(G)
    else:
        base_seed = validate_seed(seed, allow_none=False)
    nodes = list(G.nodes())
    n_before = len(nodes)
    if n_before <= 1:
        return
    rnd = make_rng(base_seed, -2, G)

    if mode is None:
        mode = str(
            G.graph.get("REMESH_MODE", REMESH_DEFAULTS.get("REMESH_MODE", "knn"))
        )
    mode = str(mode)
    nx, nx_comm = _get_networkx_modules()
    epi = {n: _as_float(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) for n in nodes}
    mst_edges = _mst_edges_from_epi(nx, nodes, epi)
    default_k = int(
        G.graph.get("REMESH_COMMUNITY_K", REMESH_DEFAULTS.get("REMESH_COMMUNITY_K", 2))
    )
    k_val = max(1, int(k) if k is not None else default_k)

    if mode == "community":
        _community_remesh(
            G,
            epi,
            k_val,
            p_rewire,
            rnd,
            nx,
            nx_comm,
            mst_edges,
            n_before,
        )
        return

    new_edges = set(mst_edges)
    if mode == "knn":
        new_edges |= _knn_edges(nodes, epi, k_val, p_rewire, rnd)

    with edge_version_update(G):
        G.clear_edges()
        G.add_edges_from(new_edges)


def _retained_tail(series: Sequence[float], window: int) -> Sequence[float]:
    """Read a chronological sample window from lists or bounded deques."""

    if isinstance(series, deque):
        return tuple(islice(reversed(series), window))[::-1]
    return series[-window:]


def _extra_gating_ok(
    hist: MutableMapping[str, Sequence[float]],
    cfg: Mapping[str, RemeshConfigValue],
    w_estab: int,
) -> bool:
    """Check additional stability gating conditions."""
    checks = [
        ("phase_sync", "REMESH_MIN_PHASE_SYNC", ge),
        ("glyph_load_disr", "REMESH_MAX_GLYPH_DISR", le),
        ("sense_sigma_mag", "REMESH_MIN_SIGMA_MAG", ge),
        ("kuramoto_R", "REMESH_MIN_KURAMOTO_R", ge),
        ("Si_hi_frac", "REMESH_MIN_SI_HI_FRAC", ge),
    ]
    for hist_key, cfg_key, op in checks:
        series = hist.get(hist_key)
        if series is not None and len(series) >= w_estab:
            win = _retained_tail(series, w_estab)
            avg = sum(win) / len(win)
            threshold = _as_float(cfg[cfg_key])
            if not op(avg, threshold):
                return False
    return True


def apply_remesh_if_globally_stable(
    G: CommunityGraph,
    stable_step_window: int | None = None,
    **kwargs: Any,
) -> None:
    """Trigger remeshing when global stability indicators satisfy thresholds.

    Cooldown counts runtime ordinals when the runtime has established an epoch.
    Standalone callers retain the historical stable-sample-count convention.
    Changing between these incomparable bases starts a fresh cooldown on the
    next successful application; physical-time cooldown remains independent.
    """

    from ..glyph_history import ensure_history

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "apply_remesh_if_globally_stable() got unexpected keyword argument(s): "
            f"{unexpected}"
        )

    params = [
        (
            "REMESH_STABILITY_WINDOW",
            int,
            REMESH_DEFAULTS["REMESH_STABILITY_WINDOW"],
        ),
        (
            "REMESH_REQUIRE_STABILITY",
            bool,
            REMESH_DEFAULTS["REMESH_REQUIRE_STABILITY"],
        ),
        (
            "REMESH_MIN_PHASE_SYNC",
            float,
            REMESH_DEFAULTS["REMESH_MIN_PHASE_SYNC"],
        ),
        (
            "REMESH_MAX_GLYPH_DISR",
            float,
            REMESH_DEFAULTS["REMESH_MAX_GLYPH_DISR"],
        ),
        (
            "REMESH_MIN_SIGMA_MAG",
            float,
            REMESH_DEFAULTS["REMESH_MIN_SIGMA_MAG"],
        ),
        (
            "REMESH_MIN_KURAMOTO_R",
            float,
            REMESH_DEFAULTS["REMESH_MIN_KURAMOTO_R"],
        ),
        (
            "REMESH_MIN_SI_HI_FRAC",
            float,
            REMESH_DEFAULTS["REMESH_MIN_SI_HI_FRAC"],
        ),
        (COOLDOWN_KEY, int, REMESH_DEFAULTS[COOLDOWN_KEY]),
        ("REMESH_COOLDOWN_TS", float, REMESH_DEFAULTS["REMESH_COOLDOWN_TS"]),
    ]
    cfg = {}
    for key, conv, _default in params:
        cfg[key] = conv(get_param(G, key))
    frac_req = _as_float(get_param(G, "FRACTION_STABLE_REMESH"))
    w_estab = (
        stable_step_window
        if stable_step_window is not None
        else cfg["REMESH_STABILITY_WINDOW"]
    )

    hist = ensure_history(G)
    sf = hist.setdefault("stable_frac", [])
    if len(sf) < w_estab:
        return
    win_sf = _retained_tail(sf, w_estab)
    if not all(v >= frac_req for v in win_sf):
        return
    if cfg["REMESH_REQUIRE_STABILITY"] and not _extra_gating_ok(hist, cfg, w_estab):
        return

    next_index = _raw_runtime_step_marker(G.graph)
    runtime_basis = next_index is not RUNTIME_STEP_UNSET
    step_idx = resolve_runtime_step_index(next_index, fallback=len(sf))
    step_basis = "runtime" if runtime_basis else "stable_samples"
    last_basis = G.graph.get("_last_remesh_step_basis", "stable_samples")
    if last_basis == step_basis and "_last_remesh_step" in G.graph:
        if step_idx - G.graph["_last_remesh_step"] < cfg[COOLDOWN_KEY]:
            return
    t_now = _as_float(G.graph.get("_t", 0.0))
    if cfg["REMESH_COOLDOWN_TS"] > 0 and "_last_remesh_ts" in G.graph:
        last_ts = _as_float(G.graph["_last_remesh_ts"])
        if (t_now - last_ts) < cfg["REMESH_COOLDOWN_TS"]:
            return

    result = apply_network_remesh(G)
    if not result.applied:
        return
    G.graph["_last_remesh_step"] = step_idx
    G.graph["_last_remesh_step_basis"] = step_basis
    G.graph["_last_remesh_ts"] = t_now


__all__ = [
    # Core remesh functions (existing API)
    "apply_network_remesh",
    "plan_network_remesh",
    "DelayedRemeshNodeProposal",
    "DelayedRemeshPlan",
    "DelayedRemeshResult",
    "DelayedRemeshStabilityEvidence",
    "apply_topological_remesh",
    "apply_remesh_if_globally_stable",
    # Phase 1: Structural memory & pattern recognition
    "StructuralIdentity",
    "structural_similarity",
    "structural_memory_match",
    "compute_structural_signature",
    "detect_recursive_patterns",
    "identify_pattern_origin",
    "propagate_structural_identity",
    "apply_network_remesh_with_memory",
    # Phase 2: Coherence preservation & validation
    "RemeshCoherenceLossError",
    "validate_coherence_preservation",
]
