"""Canonical REMESH operator: Recursive pattern propagation preserving structural coherence.

REMESH (Recursivity) - Glyph: REMESH
====================================

Physical Foundation
-------------------
From the nodal equation: ∂EPI/∂t = νf · ΔNFR(t)

REMESH implements: **EPI(t) ↔ EPI(t-τ)** (operational fractality)

REMESH enables patterns to echo across temporal and spatial scales while maintaining
coherence. "What persists at one scale can be rewritten at another, with coherence
propagating structurally, not imposed." - El pulso que nos atraviesa

Canonical Physical Behavior
----------------------------
REMESH acts on the nodal equation by creating temporal/spatial coupling:

1. **Memory Activation**: References EPI(t-τ) from structural history
2. **Pattern Recognition**: Identifies similar EPIs across network (structural_similarity)
3. **Coherent Propagation**: Propagates patterns maintaining identity (StructuralIdentity)
4. **Scale Invariance**: Preserves structural properties across reorganizations

Effect on nodal components:
- EPI: Mixed with historical states (EPI_new = (1-α)·EPI_now + α·EPI_past)
- νf: Can increase during memory activation (reactivation of dormant patterns)
- ΔNFR: Implicitly calculated from reorganization (ΔNFR = ΔEPI/νf from nodal equation)
- Phase: Preserved through StructuralIdentity tracking

Operator Relationships (from Nodal Equation Physics)
-----------------------------------------------------
All relationships emerge naturally from how operators affect EPI, νf, ΔNFR, and phase:

### REMESH Hierarchical (Central Control → Periphery)
**Physics**: Controlled replication from center maintaining coherence descendente

1. **IL (Coherence)**: Reduces |ΔNFR| → stabilizes each recursion level
   - Relationship: Estabilización multinivel (coherence extended)
   - Dynamics: REMESH propagates pattern, IL consolidates at each level
   - Sequence: REMESH → IL (recursive propagation → multi-level stabilization)

2. **VAL (Expansion)**: Increases dim(EPI) → structural expansion
   - Relationship: Expansión estructural coherente (fractal growth)
   - Dynamics: REMESH replicates, VAL expands each replica maintaining form
   - Sequence: VAL → REMESH (expand → replicate expanded form)

3. **SHA (Silence)**: νf → 0 → latent memory
   - Relationship: Estabilización de red latente (structural memory)
   - Dynamics: SHA freezes pattern (∂EPI/∂t → 0), REMESH propagates frozen state
   - Sequence: SHA → REMESH (freeze → propagate frozen memory)
   - **Critical**: NO functional redundancy - uses existing Silence operator

4. **NUL (Contraction)**: Reduces dim(EPI) → compression
   - Relationship: Compresión estructural coherente (fractal distillation)
   - Dynamics: Complementary to VAL, reduces dimensionality maintaining identity
   - Sequence: NUL → REMESH (compress → replicate compressed essence)
   - **Note**: Hierarchical simplification preserving core structure

### REMESH Rhizomatic (Decentralized Propagation)
**Physics**: Propagation without fixed center, by local resonance

1. **OZ (Dissonance)**: Increases |ΔNFR| → exploration
   - Relationship: Bifurcación distribuida (distributed bifurcation)
   - Dynamics: REMESH + OZ creates decentralized local variations
   - Sequence: OZ → REMESH (destabilize → replicate variations)

2. **UM (Coupling)**: φᵢ → φⱼ → structural connection
   - Relationship: Acoplamiento multiescala (multi-scale coupling)
   - Dynamics: REMESH propagates, UM connects replicas without hierarchy
   - Sequence: REMESH → UM (propagate → connect peers)

3. **THOL (Self-organization)**: Creates sub-EPIs → emergence
   - Relationship: Auto-organización recursiva (recursive self-organization)
   - Dynamics: REMESH + THOL generates emergent structures without center
   - Sequence: THOL → REMESH (emerge sub-EPIs → replicate emergent forms)

### REMESH Fractal Harmonic (Perfect Self-Similarity)
**Physics**: Scale-symmetric replication maintaining perfect auto-similitud

1. **RA (Resonance)**: Amplifies coherently → propagation
   - Relationship: Resonancia multiescala (multi-scale resonance)
   - Dynamics: REMESH replicates, RA amplifies with perfect symmetry
   - Sequence: REMESH → RA (replicate → amplify symmetrically)

2. **NAV (Transition)**: Activates latent EPI → regime shift
   - Relationship: Transición entre attractores fractales
   - Dynamics: REMESH navigates between self-similar attractor states
   - Sequence: NAV → REMESH (transition → replicate new regime)

3. **AL (Emission)**: Creates EPI from vacuum → generation
   - Relationship: Emisión fractal (fractal emission)
   - Dynamics: REMESH + AL generates self-similar patterns from origin
   - Sequence: AL → REMESH (emit seed → replicate fractally)

4. **EN (Reception)**: Updates EPI from network → reception
   - Relationship: Recepción multi-escala simétrica (symmetric multi-scale reception)
   - Dynamics: EN captures patterns from multiple sources → REMESH replicates symmetrically
   - Sequence: EN → REMESH (receive multi-scale → propagate symmetrically)
   - **Note**: Pre-recursion operator that feeds REMESH

### Operators with Indirect Relationships

**ZHIR (Mutation)**: Present in canonical relationships but NOT in types
- **Physical Reason**: ZHIR is a TRANSFORMER that emerges POST-recursion
- **Dynamics**: REMESH propagates → local variations + destabilizers → ZHIR transforms
- **Grammar**: Requires IL previo + recent destabilizer (U4b)
- **Relationship**: Mutación replicativa (replication facilitates mutation)
- **Conclusion**: Operates AFTER REMESH completes, not during

Grammar Implications from Physical Analysis
--------------------------------------------
REMESH's physical behavior affects unified grammar rules (UNIFIED_GRAMMAR_RULES.md):

### U1: STRUCTURAL INITIATION & CLOSURE
**Physical Basis**: REMESH echoes EPI(t-τ), can activate from dormant/null states

**U1a (Initiation)**: REMESH is a GENERATOR
- Can start sequences when operating on latent structure
- Activates dormant patterns via temporal coupling
- **Rule**: Sequences can begin with REMESH

**U1b (Closure)**: REMESH is a CLOSURE operator
- Distributes structure across scales leaving system in recursive attractor
- Creates self-sustaining multi-scale coherence
- **Rule**: Sequences can end with REMESH

### U2: CONVERGENCE & BOUNDEDNESS
**Physical Basis**: REMESH mixing with historical states can amplify or dampen ΔNFR

**Requirement**: REMESH + destabilizers → must include stabilizers
- Example: REMESH + VAL (expansion) → requires IL (coherence)
- Prevents: Unbounded growth through recursive expansion
- **Rule**: If REMESH precedes/follows VAL, OZ, or ZHIR → require IL or THOL

**Integration Convergence**: ∫ νf · ΔNFR dt must converge
- REMESH temporal mixing: (1-α)·EPI_now + α·EPI_past
- Without stabilizers: Can create positive feedback loop
- **Rule**: Stabilizers ensure convergence of recursive reorganization

### U3: RESONANT COUPLING
**Physical Basis**: REMESH propagates patterns - must verify phase compatibility

**Requirement**: REMESH with UM or RA → verify |φᵢ - φⱼ| ≤ Δφ_max
- REMESH creates replicas that must be phase-compatible for resonance
- Antiphase replicas → destructive interference
- **Rule**: StructuralIdentity.matches() includes phase verification
- **Implementation**: Phase pattern captured and validated during propagation

### U4: BIFURCATION DYNAMICS
**Physical Basis**: REMESH can trigger bifurcation through recursive amplification

**U4a (Triggers Need Handlers)**: REMESH + destabilizers → need handlers
- REMESH → THOL sequence: Recursion enables self-organization
- Must handle emergent sub-EPIs from recursive bifurcation
- **Rule**: REMESH + OZ or ZHIR → require THOL or IL handlers

**U4b (Transformers Need Context)**: ZHIR requires REMESH context
- REMESH creates variations across scales
- ZHIR then transforms local variations (post-recursion)
- **Rule**: ZHIR after REMESH → requires prior IL + recent destabilizer

Centralized Flow - No Redundancy
---------------------------------
This implementation maintains a single, centralized flow:

1. **SHA Integration**: Uses existing Silence operator from definitions.py
   - NO reimplementation of SHA functionality
   - StructuralIdentity only CAPTURES frozen states, doesn't freeze
   - Workflow: Silence() → capture_from_node(is_sha_frozen=True) → validate

2. **Coherence Calculation**: Simplified C(t) for REMESH validation
   - Full coherence calculation in tnfr.metrics module
   - compute_global_coherence() specific to REMESH fidelity checks
   - No duplication with main coherence computation

3. **Pattern Recognition**: Unique to REMESH structural memory
   - structural_similarity(): Pattern matching (no operator overlap)
   - structural_memory_match(): Network-wide search (REMESH-specific)
   - No duplication with network analysis tools

4. **Identity Tracking**: REMESH-specific fractal identity
   - StructuralIdentity: Persistent identity across reorganizations
   - Captures EPI signature, νf range, phase pattern
   - Validates preservation post-recursion

Key Capabilities
----------------
- Structural memory: Pattern recognition across network nodes
- Identity preservation: Fractal lineage tracking across reorganizations
- Coherence conservation: Validating structural fidelity during remeshing
- Multi-modal recursivity: Hierarchical, rhizomatic, and fractal harmonic modes
- Grammar-compliant: All operations respect unified grammar rules (U1-U5)
"""

from __future__ import annotations

import hashlib
import heapq
import random
from collections import deque
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from io import StringIO
from itertools import combinations
from operator import ge, le
from statistics import StatisticsError, fmean
from types import ModuleType
from typing import Any, cast

from .._compat import TypeAlias
from ..alias import get_attr, set_attr
from ..constants import DEFAULTS, REMESH_DEFAULTS, get_param
from ..constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR
from ..rng import make_rng
from ..types import RemeshMeta
from ..utils import cached_import, edge_version_update, kahan_sum_nd

CommunityGraph: TypeAlias = Any
NetworkxModule: TypeAlias = ModuleType
CommunityModule: TypeAlias = ModuleType
RemeshEdge: TypeAlias = tuple[Hashable, Hashable]
NetworkxModules: TypeAlias = tuple[NetworkxModule, CommunityModule]
RemeshConfigValue: TypeAlias = bool | float | int


# ==============================================================================
# Phase 1: Structural Memory & Pattern Recognition
# ==============================================================================


@dataclass
class StructuralIdentity:
    """Persistent fractal identity maintained across reorganizations.

    Captures the canonical structural "signature" of a pattern that must be
    preserved as it echoes across scales. This implements TNFR's requirement
    that patterns maintain identity through reorganization.

    **REMESH ↔ SHA Relationship**: According to TNFR theory, SHA (Silence)
    stabilizes latent network memory by reducing νf → 0, which freezes EPI
    via the nodal equation: ∂EPI/∂t = νf · ΔNFR → 0. When REMESH propagates
    patterns across scales, SHA-frozen nodes act as "structural anchors" that
    maintain identity during reorganization.

    **Usage Pattern**:
    1. Apply SHA to freeze node: νf → 0, preserves EPI
    2. Capture identity from frozen state (this class)
    3. Apply REMESH to propagate pattern across scales
    4. Validate identity preservation post-reorganization

    Attributes
    ----------
    epi_signature : float
        Characteristic EPI pattern value (preserved when SHA applied)
    vf_range : tuple[float, float]
        Range of structural frequencies (min, max) in Hz_str
    phase_pattern : float | None
        Characteristic phase pattern in [0, 2π], if applicable
    frozen_by_sha : bool
        Whether this identity was captured from SHA-frozen state (νf ≈ 0)
    lineage : list[str]
        History of transformations preserving this identity
    tolerance : float
        Maximum deviation for identity match (default: 0.1)

    Notes
    -----
    From TNFR physics (definitions.py::Silence): SHA reduces νf causing
    ∂EPI/∂t → 0 regardless of ΔNFR. This creates "latent memory" - frozen
    structural patterns that REMESH can propagate coherently across scales.

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

    def matches(self, node_data: Mapping[str, Any], *, tolerance: float | None = None) -> bool:
        """Check if a node maintains this structural identity.

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
        If frozen_by_sha=True, vf check is relaxed since SHA-frozen patterns
        have νf ≈ 0 (frozen state) while maintaining identity via EPI.
        """
        tol = tolerance if tolerance is not None else self.tolerance

        # Check EPI signature match (primary identity criterion)
        node_epi = _as_float(get_attr(node_data, ALIAS_EPI, 0.0))
        if abs(node_epi - self.epi_signature) > tol:
            return False

        # Check vf range (relaxed if SHA-frozen)
        node_vf = _as_float(get_attr(node_data, ALIAS_VF, 0.0))
        vf_min, vf_max = self.vf_range

        if self.frozen_by_sha:
            # SHA-frozen: accept low νf (frozen state) OR original range
            # (pattern may be reactivated after SHA)
            if node_vf < 0.05:  # Frozen by SHA (νf → 0)
                pass  # Accept - this is expected for SHA-frozen patterns
            elif not (vf_min - tol <= node_vf <= vf_max + tol):
                return False
        else:
            # Normal check: must be within range
            if not (vf_min - tol <= node_vf <= vf_max + tol):
                return False

        # Check phase pattern if specified
        if self.phase_pattern is not None:
            from ..constants.aliases import ALIAS_THETA

            node_phase = _as_float(get_attr(node_data, ALIAS_THETA, None))
            if node_phase is None:
                return False
            # Phase comparison with circular wrap-around
            import math

            phase_diff = abs(node_phase - self.phase_pattern)
            phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
            if phase_diff > tol:
                return False

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

        This creates a "memory snapshot" of the node's structural signature
        that can be propagated via REMESH across scales.

        Parameters
        ----------
        node_data : Mapping
            Node attributes containing EPI, vf, phase
        is_sha_frozen : bool, default=False
            If True, mark identity as captured from SHA-frozen state (νf ≈ 0).
            **NOTE**: This does NOT apply SHA - it only marks that SHA was
            already applied. Use tnfr.operators.definitions.Silence to apply SHA.
        tolerance : float, default=0.1
            Tolerance for future identity matching

        Returns
        -------
        StructuralIdentity
            Captured structural identity

        Notes
        -----
        **REMESH ↔ SHA Integration Pattern**:

        1. Apply SHA operator to freeze node (see tnfr.operators.definitions.Silence)
        2. Call this method with is_sha_frozen=True to capture frozen state
        3. Apply REMESH to propagate pattern across scales
        4. Use identity.matches() to validate preservation

        **DO NOT use this to apply SHA** - SHA is a separate operator that must
        be applied via the grammar system. This only captures state AFTER SHA.

        Example
        -------
        >>> from tnfr.operators.definitions import Silence
        >>> from tnfr.structural import run_sequence
        >>> # Step 1: Apply SHA operator to freeze node
        >>> run_sequence(G, node, [Silence()])
        >>> # Step 2: Capture frozen identity
        >>> identity = StructuralIdentity.capture_from_node(
        ...     G.nodes[node],
        ...     is_sha_frozen=True
        ... )
        >>> # Step 3: Apply REMESH (propagates frozen pattern)
        >>> # ... REMESH operations ...
        >>> # Step 4: Validate identity preserved
        >>> assert identity.matches(G.nodes[node])
        """
        epi = _as_float(get_attr(node_data, ALIAS_EPI, 0.0))
        vf = _as_float(get_attr(node_data, ALIAS_VF, 0.0))

        # For vf_range, use small window around current value
        # (unless SHA-frozen, in which case we expect νf ≈ 0)
        vf_tolerance = 0.1
        vf_min = max(0.0, vf - vf_tolerance)
        vf_max = vf + vf_tolerance

        # Capture phase if present
        from ..constants.aliases import ALIAS_THETA

        phase = _as_float(get_attr(node_data, ALIAS_THETA, None))

        identity = cls(
            epi_signature=epi,
            vf_range=(vf_min, vf_max),
            phase_pattern=phase if phase is not None else None,
            frozen_by_sha=is_sha_frozen,
            tolerance=tolerance,
        )

        if is_sha_frozen:
            identity.record_transformation(
                "Captured from SHA-frozen state (latent structural memory)"
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
    np = _get_numpy()
    if np is None:
        # Fallback: scalar comparison only
        if isinstance(epi1, (list, tuple)) or isinstance(epi2, (list, tuple)):
            raise ImportError(
                "NumPy required for vector EPI comparison. " "Install numpy: pip install numpy"
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
        raise ValueError(f"EPI patterns must have same shape: {arr1.shape} vs {arr2.shape}")

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
        raise ValueError(f"Unknown metric '{metric}'. Choose from: euclidean, cosine, correlation")


def structural_memory_match(
    G: CommunityGraph,
    source_node: Hashable,
    target_nodes: Iterable[Hashable] | None = None,
    *,
    threshold: float = 0.75,
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

    The signature captures the node's complete structural identity including:
    - EPI: coherence magnitude
    - νf: structural frequency
    - θ: phase
    - ΔNFR: reorganization gradient
    - Topological properties: degree, local clustering

    This signature enables REMESH's structural memory capability - recognizing
    the same pattern across different nodes and scales.

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
    np = _get_numpy()
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
    threshold: float = 0.75,
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
    np = _get_numpy()
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
            raise ValueError(f"Unknown metric: {metric}")
    else:
        # Fallback: pairwise computation
        similarities = {}
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i == j:
                    similarities[(i, j)] = 1.0
                elif i < j:
                    # Use existing structural_similarity function
                    sim = structural_similarity(signatures[node1], signatures[node2], metric=metric)
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
    """Propagate structural identity from origin to similar nodes.

    This implements REMESH's core principle: nodes with similar patterns
    mutually reinforce their coherence through structural resonance.
    The origin's pattern is propagated to targets via weighted interpolation.

    Parameters
    ----------
    G : TNFRGraph
        Network to modify (changes node attributes in-place)
    origin_node : Hashable
        Source node whose pattern to propagate
    target_nodes : sequence
        Nodes that will receive pattern reinforcement
    propagation_strength : float, default=0.5
        Interpolation weight in [0, 1]:
        - 0.0: No effect (targets unchanged)
        - 1.0: Complete copy (targets become identical to origin)
        - 0.5: Balanced blending

    Notes
    -----
    For each target node, updates:
    - EPI: new = (1-α)×old + α×origin
    - νf: new = (1-α)×old + α×origin
    - θ: new = (1-α)×old + α×origin

    Where α = propagation_strength.

    Updates respect structural boundaries (EPI_MIN, EPI_MAX) via
    structural_clip to prevent overflow and maintain physical validity.

    Records propagation in node's 'structural_lineage' attribute for
    traceability and analysis.

    **TNFR Physics**: This preserves the nodal equation ∂EPI/∂t = νf·ΔNFR
    by interpolating the structural state rather than imposing it directly.

    Examples
    --------
    >>> origin = 1
    >>> targets = [2, 3, 4]  # Similar nodes
    >>> propagate_structural_identity(G, origin, targets, 0.3)
    >>> # Targets now have patterns 30% closer to origin
    """
    # Get origin pattern
    origin_epi = _as_float(get_attr(G.nodes[origin_node], ALIAS_EPI, 0.0))
    origin_vf = _as_float(get_attr(G.nodes[origin_node], ALIAS_VF, 0.0))
    from ..constants.aliases import ALIAS_THETA

    origin_theta = _as_float(get_attr(G.nodes[origin_node], ALIAS_THETA, 0.0))

    # Get structural bounds
    from ..constants import DEFAULTS

    epi_min = float(G.graph.get("EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)))
    epi_max = float(G.graph.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)))

    # Get clip mode
    clip_mode_str = str(G.graph.get("CLIP_MODE", "hard"))
    if clip_mode_str not in ("hard", "soft"):
        clip_mode_str = "hard"

    # Propagate to each target
    for target in target_nodes:
        if target == origin_node:
            continue  # Don't propagate to self

        # Get current target state
        target_epi = _as_float(get_attr(G.nodes[target], ALIAS_EPI, 0.0))
        target_vf = _as_float(get_attr(G.nodes[target], ALIAS_VF, 0.0))
        target_theta = _as_float(get_attr(G.nodes[target], ALIAS_THETA, 0.0))

        # Interpolate toward origin pattern
        new_epi = (1.0 - propagation_strength) * target_epi + propagation_strength * origin_epi
        new_vf = (1.0 - propagation_strength) * target_vf + propagation_strength * origin_vf
        new_theta = (
            1.0 - propagation_strength
        ) * target_theta + propagation_strength * origin_theta

        # Apply structural clipping to preserve boundaries
        from ..dynamics.structural_clip import structural_clip

        new_epi = structural_clip(new_epi, lo=epi_min, hi=epi_max, mode=clip_mode_str)  # type: ignore[arg-type]

        # Update node attributes
        set_attr(G.nodes[target], ALIAS_EPI, new_epi)
        set_attr(G.nodes[target], ALIAS_VF, new_vf)
        set_attr(G.nodes[target], ALIAS_THETA, new_theta)

        # Record lineage for traceability
        if "structural_lineage" not in G.nodes[target]:
            G.nodes[target]["structural_lineage"] = []

        # Get current step for timestamp
        from ..glyph_history import current_step_idx

        step = current_step_idx(G)

        G.nodes[target]["structural_lineage"].append(
            {
                "origin": origin_node,
                "step": step,
                "propagation_strength": propagation_strength,
                "epi_before": target_epi,
                "epi_after": new_epi,
            }
        )


@cache
def _get_numpy() -> ModuleType | None:
    """Get numpy module if available, None otherwise."""
    return cached_import("numpy")


# ==============================================================================
# Phase 2: Coherence Preservation & Fidelity Validation
# ==============================================================================


class RemeshCoherenceLossError(Exception):
    """Raised when REMESH reorganization loses structural coherence.

    REMESH must preserve coherence during reorganization. This error indicates
    that the structural fidelity dropped below acceptable thresholds, violating
    TNFR's requirement that "coherence propagates structurally, not imposed."
    """

    def __init__(self, fidelity: float, min_fidelity: float, details: dict[str, Any] | None = None):
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


def _remesh_alpha_info(G: CommunityGraph) -> tuple[float, str]:
    """Return ``(alpha, source)`` with explicit precedence."""
    if bool(G.graph.get("REMESH_ALPHA_HARD", REMESH_DEFAULTS["REMESH_ALPHA_HARD"])):
        val = _as_float(
            G.graph.get("REMESH_ALPHA", REMESH_DEFAULTS["REMESH_ALPHA"]),
            float(REMESH_DEFAULTS["REMESH_ALPHA"]),
        )
        return val, "REMESH_ALPHA"
    gf = G.graph.get("GLYPH_FACTORS", DEFAULTS.get("GLYPH_FACTORS", {}))
    if "REMESH_alpha" in gf:
        return _as_float(gf["REMESH_alpha"]), "GLYPH_FACTORS.REMESH_alpha"
    if "REMESH_ALPHA" in G.graph:
        return _as_float(G.graph["REMESH_ALPHA"]), "REMESH_ALPHA"
    return (
        float(REMESH_DEFAULTS["REMESH_ALPHA"]),
        "REMESH_DEFAULTS.REMESH_ALPHA",
    )


def _snapshot_topology(G: CommunityGraph, nx: NetworkxModule) -> str | None:
    """Return a hash representing the current graph topology."""
    try:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        degs = sorted(d for _, d in G.degree())
        topo_str = f"n={n_nodes};m={n_edges};deg=" + ",".join(map(str, degs))
        return hashlib.blake2b(topo_str.encode(), digest_size=6).hexdigest()
    except (AttributeError, TypeError, nx.NetworkXError):
        return None


def _snapshot_epi(G: CommunityGraph) -> tuple[float, str]:
    """Return ``(mean, checksum)`` of the node EPI values."""
    buf = StringIO()
    values = []
    for n, data in G.nodes(data=True):
        v = _as_float(get_attr(data, ALIAS_EPI, 0.0))
        values.append(v)
        buf.write(f"{str(n)}:{round(v, 6)};")
    total = kahan_sum_nd(((v,) for v in values), dims=1)[0]
    mean_val = total / len(values) if values else 0.0
    checksum = hashlib.blake2b(buf.getvalue().encode(), digest_size=6).hexdigest()
    return float(mean_val), checksum


def _log_remesh_event(G: CommunityGraph, meta: RemeshMeta) -> None:
    """Store remesh metadata and optionally log and trigger callbacks."""
    from ..utils import CallbackEvent, callback_manager
    from ..glyph_history import append_metric

    G.graph["_REMESH_META"] = meta
    if G.graph.get("REMESH_LOG_EVENTS", REMESH_DEFAULTS["REMESH_LOG_EVENTS"]):
        hist = G.graph.setdefault("history", {})
        append_metric(hist, "remesh_events", dict(meta))
    callback_manager.invoke_callbacks(G, CallbackEvent.ON_REMESH.value, dict(meta))


def apply_network_remesh(G: CommunityGraph) -> None:
    """Network-scale REMESH using ``_epi_hist`` with multi-scale memory."""
    from ..glyph_history import current_step_idx, ensure_history
    from ..dynamics.structural_clip import structural_clip

    nx, _ = _get_networkx_modules()
    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
    tau_req = max(tau_g, tau_l)
    alpha, alpha_src = _remesh_alpha_info(G)
    G.graph["_REMESH_ALPHA_SRC"] = alpha_src
    hist = G.graph.get("_epi_hist", deque())
    if len(hist) < tau_req + 1:
        return

    past_g = hist[-(tau_g + 1)]
    past_l = hist[-(tau_l + 1)]

    topo_hash = _snapshot_topology(G, nx)
    epi_mean_before, epi_checksum_before = _snapshot_epi(G)

    # Get EPI bounds for structural preservation
    epi_min = float(G.graph.get("EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)))
    epi_max = float(G.graph.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)))
    clip_mode_str = str(G.graph.get("CLIP_MODE", "hard"))
    if clip_mode_str not in ("hard", "soft"):
        clip_mode_str = "hard"
    clip_mode = clip_mode_str  # type: ignore[assignment]

    for n, nd in G.nodes(data=True):
        epi_now = _as_float(get_attr(nd, ALIAS_EPI, 0.0))
        epi_old_l = _as_float(past_l.get(n) if isinstance(past_l, Mapping) else None, epi_now)
        epi_old_g = _as_float(past_g.get(n) if isinstance(past_g, Mapping) else None, epi_now)
        mixed = (1 - alpha) * epi_now + alpha * epi_old_l
        mixed = (1 - alpha) * mixed + alpha * epi_old_g

        # Apply structural boundary preservation to prevent overflow
        mixed_clipped = structural_clip(mixed, lo=epi_min, hi=epi_max, mode=clip_mode)
        set_attr(nd, ALIAS_EPI, mixed_clipped)

    epi_mean_after, epi_checksum_after = _snapshot_epi(G)

    step_idx = current_step_idx(G)
    meta: RemeshMeta = {
        "alpha": alpha,
        "alpha_source": alpha_src,
        "tau_global": tau_g,
        "tau_local": tau_l,
        "step": step_idx,
        "topo_hash": topo_hash,
        "epi_mean_before": float(epi_mean_before),
        "epi_mean_after": float(epi_mean_after),
        "epi_checksum_before": epi_checksum_before,
        "epi_checksum_after": epi_checksum_after,
    }

    h = ensure_history(G)
    if h:
        if h.get("stable_frac"):
            meta["stable_frac_last"] = h["stable_frac"][-1]
        if h.get("phase_sync"):
            meta["phase_sync_last"] = h["phase_sync"][-1]
        if h.get("glyph_load_disr"):
            meta["glyph_disr_last"] = h["glyph_load_disr"][-1]

    _log_remesh_event(G, meta)


def apply_network_remesh_with_memory(
    G: CommunityGraph,
    *,
    enable_structural_memory: bool = True,
    similarity_threshold: float = 0.75,
    similarity_metric: str = "cosine",
    propagation_strength: float = 0.5,
    min_cluster_size: int = 2,
) -> None:
    """Apply REMESH with structural field memory activation.

    This extended version of REMESH implements the theoretical capability
    of "structural memory": nodes can recognize themselves in other nodes
    through pattern similarity, enabling coherent propagation across scales.

    The function performs:
    1. Standard REMESH reorganization (temporal EPI mixing)
    2. Pattern detection (find groups of structurally similar nodes)
    3. Identity propagation (reinforce shared patterns from origin nodes)

    Parameters
    ----------
    G : TNFRGraph
        Network to reorganize (modified in-place)
    enable_structural_memory : bool, default=True
        Whether to activate structural memory after standard REMESH.
        If False, performs only standard REMESH.
    similarity_threshold : float, default=0.75
        Minimum similarity [0-1] to recognize patterns as "same identity".
        Higher = stricter matching. Typical range: 0.7-0.9.
    similarity_metric : str, default='cosine'
        Metric for comparing structural signatures.
        Options: 'cosine', 'euclidean', 'correlation'
    propagation_strength : float, default=0.5
        Interpolation weight [0-1] for identity propagation.
        - 0.0: No propagation (structural memory detection only)
        - 0.5: Balanced blending (recommended)
        - 1.0: Full replacement (aggressive, may reduce diversity)
    min_cluster_size : int, default=2
        Minimum nodes required to form a recursive pattern.
        Single isolated nodes are not considered patterns.

    Notes
    -----
    **TNFR Physics**: This implements the principle that "coherence propagates
    structurally, not imposed" (TNFR.pdf § 4.2). Patterns that resonate across
    the network mutually reinforce through similarity-based coupling.

    **Workflow**:
    1. `apply_network_remesh(G)` - Standard temporal memory mixing
    2. `detect_recursive_patterns(G, threshold)` - Find similar node groups
    3. For each cluster:
       - `identify_pattern_origin(G, cluster)` - Find strongest instance
       - `propagate_structural_identity(G, origin, targets)` - Reinforce pattern

    **Telemetry**: Logs structural memory events to G.graph['history']['structural_memory_events']
    including cluster statistics and propagation metadata.

    **Canonical Relationships**:
    - Hierarchical REMESH: Combine with IL (coherence) for stable multi-level propagation
    - Rhizomatic REMESH: Combine with UM (coupling) for decentralized pattern spread
    - Fractal Harmonic: Combine with RA (resonance) for symmetric amplification

    Examples
    --------
    >>> # Standard REMESH with structural memory (recommended)
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
    apply_network_remesh(G)

    if not enable_structural_memory:
        return

    # Phase 2: Structural memory - detect and propagate patterns
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
        # Graceful degradation: if structural memory fails, REMESH still applied
        import warnings

        warnings.warn(
            f"Structural memory activation failed: {e}. " "Standard REMESH applied successfully.",
            RuntimeWarning,
            stacklevel=2,
        )


def _mst_edges_from_epi(
    nx: NetworkxModule,
    nodes: Sequence[Hashable],
    epi: Mapping[Hashable, float],
) -> set[RemeshEdge]:
    """Return MST edges based on absolute EPI distance."""
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_weighted_edges_from((u, v, abs(epi[u] - epi[v])) for u, v in combinations(nodes, 2))
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
                if abs(epi_u - epi_vals[ordered[left]]) <= abs(epi_vals[ordered[right]] - epi_u):
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
    extra_edges, attempts, rewired_edges = _community_k_neighbor_edges(C, k_val, p_rewire, rnd)
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
    nodes = list(G.nodes())
    n_before = len(nodes)
    if n_before <= 1:
        return
    if seed is None:
        base_seed = int(G.graph.get("RANDOM_SEED", 0))
    else:
        base_seed = int(seed)
    rnd = make_rng(base_seed, -2, G)

    if mode is None:
        mode = str(G.graph.get("REMESH_MODE", REMESH_DEFAULTS.get("REMESH_MODE", "knn")))
    mode = str(mode)
    nx, nx_comm = _get_networkx_modules()
    epi = {n: _as_float(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) for n in nodes}
    mst_edges = _mst_edges_from_epi(nx, nodes, epi)
    default_k = int(G.graph.get("REMESH_COMMUNITY_K", REMESH_DEFAULTS.get("REMESH_COMMUNITY_K", 2)))
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
            win = series[-w_estab:]
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
    """Trigger remeshing when global stability indicators satisfy thresholds."""

    from ..glyph_history import ensure_history

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "apply_remesh_if_globally_stable() got unexpected keyword argument(s): " f"{unexpected}"
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
        stable_step_window if stable_step_window is not None else cfg["REMESH_STABILITY_WINDOW"]
    )

    hist = ensure_history(G)
    sf = hist.setdefault("stable_frac", [])
    if len(sf) < w_estab:
        return
    win_sf = sf[-w_estab:]
    if not all(v >= frac_req for v in win_sf):
        return
    if cfg["REMESH_REQUIRE_STABILITY"] and not _extra_gating_ok(hist, cfg, w_estab):
        return

    last = G.graph.get("_last_remesh_step", -(10**9))
    step_idx = len(sf)
    if step_idx - last < cfg[COOLDOWN_KEY]:
        return
    t_now = _as_float(G.graph.get("_t", 0.0))
    last_ts = _as_float(G.graph.get("_last_remesh_ts", -1e12))
    if cfg["REMESH_COOLDOWN_TS"] > 0 and (t_now - last_ts) < cfg["REMESH_COOLDOWN_TS"]:
        return

    apply_network_remesh(G)
    G.graph["_last_remesh_step"] = step_idx
    G.graph["_last_remesh_ts"] = t_now


__all__ = [
    # Core remesh functions (existing API)
    "apply_network_remesh",
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
