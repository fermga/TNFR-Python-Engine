r"""Auxiliary finite Hermitian matrix construction from TNFR graph read-outs.

This module implements the explicit construction of the internal Hamiltonian:

.. math::
    \hat{H}_{int} = \hat{H}_{coh} + \hat{H}_{freq} + \hat{H}_{coupling}

Mathematical Foundation
-----------------------

This matrix defines an auxiliary unitary model. It does not derive the engine's
nodal pressure, phase law or operator selection. The legacy node-local
commutator read-out is identically zero; no evolving density matrix or mapping
from that matrix to graph EPI is supplied here.

**Components**:

1. **Coherence Potential** :math:`\hat{H}_{coh}`:
   Potential energy from structural alignment between nodes.

   .. math::
       \hat{H}_{coh} = C_0 W

   where :math:`W=(w_{ij})` is the auxiliary structural-affinity matrix and
   :math:`C_0` is normally negative. This term is distinct from canonical
   total coherence ``C(t)``; :math:`W` is symmetric but can be indefinite.

2. **Frequency Operator** :math:`\hat{H}_{freq}`:
   Diagonal operator encoding each node's structural frequency.

   .. math::
       \hat{H}_{freq} = \sum_i \nu_{f,i} |i\rangle\langle i|

3. **Coupling Hamiltonian** :math:`\hat{H}_{coupling}`:
   Network topology-induced interactions.

   .. math::
       \hat{H}_{coupling} = J_0 A_{supp}

   Here :math:`A_{supp}` is the Boolean adjacency matrix of the underlying
   simple undirected support. Direction, reciprocal arcs and parallel edges
   collapse to one coupling. A self-loop contributes one diagonal entry.

The model and sign boundaries are documented in
``theory/STRUCTURAL_STABILITY_AND_DYNAMICS.md``, section 5.

Examples
--------

**Basic Hamiltonian construction**:

>>> import networkx as nx
>>> from tnfr.operators.hamiltonian import InternalHamiltonian
>>> G = nx.Graph()
>>> G.add_edges_from([(0, 1), (1, 2), (2, 0)])
>>> for i, node in enumerate(G.nodes):
...     G.nodes[node].update({
...         'nu_f': 0.5 + 0.1 * i,
...         'phase': 0.0,
...         'epi': 1.0,
...         'si': 0.7
...     })
>>> ham = InternalHamiltonian(G)
>>> print("Total Hamiltonian shape:", ham.H_int.shape)
Total Hamiltonian shape: (3, 3)

**Time evolution**:

>>> U_t = ham.time_evolution_operator(t=1.0)
>>> import numpy as np
>>> is_unitary = np.allclose(U_t @ U_t.conj().T, np.eye(3))
>>> print("Evolution operator is unitary:", is_unitary)
Evolution operator is unitary: True

**Energy spectrum**:

>>> eigenvalues, eigenvectors = ham.get_spectrum()
>>> eigenvalues.shape, eigenvectors.shape
((3,), (3, 3))
"""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Sequence

from ..alias import get_attr
from ..constants.aliases import ALIAS_VF
from ..mathematics.unified_numerical import np
from ..utils.cache import CacheManager, _graph_cache_manager, cached_node_list

if TYPE_CHECKING:  # pragma: no cover
    from ..types import FloatMatrix, TNFRGraph

__all__ = (
    "InternalHamiltonian",
    "build_H_coherence",
    "build_H_frequency",
    "build_H_coupling",
)


class InternalHamiltonian:
    r"""Constructs and manipulates the internal Hamiltonian H_int.

    Mathematical Definition
    -----------------------

    .. math::
        \hat{H}_{int} = \hat{H}_{coh} + \hat{H}_{freq} + \hat{H}_{coupling}

    where each component is an N×N Hermitian matrix (N = number of nodes).

    Attributes
    ----------
    G : TNFRGraph
        Network graph with structural attributes
    H_coh : ndarray, shape (N, N)
        Coherence potential matrix
    H_freq : ndarray, shape (N, N)
        Frequency operator matrix (diagonal)
    H_coupling : ndarray, shape (N, N)
        Coupling matrix from network topology
    H_int : ndarray, shape (N, N)
        Total internal Hamiltonian
    hbar_str : float
        Finite nonzero scale of the auxiliary unitary model
    nodes : list
        Ordered list of node identifiers
    N : int
        Number of nodes in the network

    Notes
    -----

    Matrix construction reuses the node-order and structural-affinity owners.
    The matrices are a snapshot: changing the graph does not update this object.
    The cache-manager argument is retained for compatibility. Components must
    be finite and pass a numerical Hermiticity check; this is not an exact
    algebraic certificate or a theorem about the engine trajectory.
    """

    def __init__(
        self,
        G: TNFRGraph,
        hbar_str: float = 1.0,
        cache_manager: CacheManager | None = None,
    ):
        """Initialize Hamiltonian from graph structure.

        Parameters
        ----------
        G : TNFRGraph
            Graph with nodes containing 'nu_f', 'phase', 'epi', 'si' attributes
        hbar_str : float, default=1.0
            Finite nonzero real scale for the auxiliary unitary model.
        cache_manager : CacheManager, optional
            Retained compatibility argument. If None, retains the graph's
            internal cache manager; matrix construction is not cached here.

        Raises
        ------
        ValueError
            If the scale is zero/nonfinite or a component is nonfinite or
            fails the numerical Hermiticity check
        """
        self.G = G
        self.hbar_str = _finite_real_coefficient(hbar_str, "hbar_str")
        if self.hbar_str == 0.0:
            raise ValueError("hbar_str must be nonzero")

        # Use unified cache infrastructure
        if cache_manager is None:
            cache_manager = _graph_cache_manager(G.graph)
        self._cache_manager = cache_manager

        # Get consistent node ordering using cached utility
        self.nodes = cached_node_list(G)
        self.N = len(self.nodes)

        # Build Hamiltonian components
        self.H_coh = self._build_H_coherence()
        self.H_freq = self._build_H_frequency()
        self.H_coupling = self._build_H_coupling()

        # Combine into total Hamiltonian
        self.H_int = self.H_coh + self.H_freq + self.H_coupling

        # Validate the auxiliary matrix, not a physical identification.
        self._verify_hermitian()

    def _build_H_coherence(self) -> FloatMatrix:
        r"""Construct the auxiliary affinity contribution ``H_coh``.

        Theory
        ------

        .. math::
            \hat{H}_{coh} = C_0 W

        Here ``W`` is the bounded structural-affinity matrix and ``C_0`` is
        normally negative for an attractive contribution. ``W`` is symmetric
        but is not positive semidefinite in general; this auxiliary model is
        distinct from canonical ``C(t)``.

        Returns
        -------
        H_coh : ndarray, shape (N, N)
            Coherence potential matrix (Hermitian)

        Notes
        -----

        Reuses ``coherence_matrix()`` function to avoid code duplication and
        ensure consistency with existing coherence computations.
        """

        strength = self.G.graph.get("H_COH_STRENGTH", -1.0)
        return build_H_coherence(
            self.G,
            nodes=list(self.nodes),
            C_0=strength,
        )

    def _build_H_frequency(self) -> FloatMatrix:
        r"""Construct frequency operator H_freq (diagonal).

        Theory
        ------

        .. math::
            \hat{H}_{freq} = \sum_i \nu_{f,i} |i\rangle\langle i|

        Each node's structural frequency :math:`\nu_{f,i}` becomes its diagonal
        entry of this auxiliary matrix. This is a model choice, not a derived
        kinetic energy for the first-order nodal equation.

        Returns
        -------
        H_freq : ndarray, shape (N, N)
            Diagonal frequency operator (Hermitian)

        Notes
        -----

        Uses ``get_attr()`` with ``ALIAS_VF`` to support attribute aliasing
        and maintain consistency with the rest of the codebase.
        """

        return build_H_frequency(self.G, nodes=list(self.nodes))

    def _build_H_coupling(self) -> FloatMatrix:
        r"""Construct coupling from the graph's undirected Boolean support.

        Theory
        ------

        .. math::
            \hat{H}_{coupling} = J_0 A_{supp}

        ``A_supp`` is the Boolean adjacency of the underlying simple
        undirected graph. Direction, reciprocal arcs and edge multiplicity do
        not change its entries; a self-loop contributes one diagonal entry.

        Returns
        -------
        H_coupling : ndarray, shape (N, N)
            Real-symmetric coupling matrix
        """

        strength = self.G.graph.get("H_COUPLING_STRENGTH", 0.1)
        return build_H_coupling(
            self.G,
            nodes=list(self.nodes),
            J_0=strength,
        )

    def _verify_hermitian(self, tolerance: float = 1e-10) -> None:
        r"""Verify that all Hamiltonian components are Hermitian.

        Parameters
        ----------
        tolerance : float, default=1e-10
            Maximum allowed deviation from Hermiticity

        Raises
        ------
        ValueError
            If any component fails Hermiticity check with detailed diagnostics

        Notes
        -----

        A matrix H is Hermitian if :math:`H = H^\dagger`, where :math:`\dagger`
        denotes conjugate transpose. This ensures:

        1. Real eigenvalues (energy spectrum)
        2. Unitary time evolution
        3. Euclidean norm conservation in the auxiliary complex state space
        """

        # Handle empty graph case
        if self.N == 0:
            return

        components = [
            ("H_coh", self.H_coh),
            ("H_freq", self.H_freq),
            ("H_coupling", self.H_coupling),
            ("H_int", self.H_int),
        ]

        for name, H in components:
            if not np.all(np.isfinite(H)):
                raise ValueError(f"{name} must contain only finite entries")
            # Check Hermiticity: H = H†
            H_dagger = H.conj().T
            deviation = np.max(np.abs(H - H_dagger))

            if deviation > tolerance:
                raise ValueError(
                    f"{name} is not Hermitian: max deviation = {deviation:.2e} "
                    f"(tolerance = {tolerance:.2e})"
                )

    def compute_delta_nfr_operator(self) -> FloatMatrix:
        r"""Return the legacy auxiliary matrix ``+i H_int / hbar_str``.

        Despite its name, this N-by-N matrix is not a commutator superoperator
        on density matrices and does not supply node-local pressure. For a
        Hermitian H it is anti-Hermitian. Its sign is the negative of the
        ket-state generator used by :meth:`time_evolution_operator`; retain
        this convention when consuming the compatibility API.
        """
        return (1j / self.hbar_str) * self.H_int

    def time_evolution_operator(self, t: float) -> FloatMatrix:
        r"""Compute time evolution operator U(t) = exp(-i H_int t / ℏ_str).

        Parameters
        ----------
        t : float
            Evolution time in structural time units

        Returns
        -------
        U_t : ndarray, shape (N, N)
            Unitary time evolution operator

        Raises
        ------
        ValueError
            If the computed operator is not unitary (indicates numerical issues)
        ImportError
            If scipy is not installed

        Notes
        -----

        The time evolution operator propagates states forward in time:

        .. math::
            |\psi(t)\rangle = U(t)|\psi(0)\rangle

        Unitarity :math:`U^\dagger U = I` preserves the auxiliary state norm.
        This method does not write graph EPI or establish physical probabilities.
        """
        try:
            from scipy.linalg import expm
        except ImportError as exc:
            raise ImportError(
                "scipy is required for time evolution computation. "
                "Install with: pip install scipy"
            ) from exc

        # Compute matrix exponential
        exponent = -1j * self.H_int * t / self.hbar_str
        U_t = expm(exponent)

        # Verify unitarity: U†U = I
        U_dag_U = U_t.conj().T @ U_t
        identity = np.eye(self.N)

        if not np.allclose(U_dag_U, identity):
            max_error = np.max(np.abs(U_dag_U - identity))
            raise ValueError(
                f"Evolution operator is not unitary: max error = {max_error:.2e}. "
                "This indicates numerical instability, possibly due to "
                "ill-conditioned Hamiltonian or inappropriate time step."
            )

        return U_t

    def get_spectrum(self) -> tuple[Any, Any]:
        r"""Compute eigenvalues and eigenvectors of H_int.

        Returns
        -------
        eigenvalues : ndarray, shape (N,)
            Energy eigenvalues (sorted in ascending order)
        eigenvectors : ndarray, shape (N, N)
            Eigenvector matrix (columns are eigenstates)

        Notes
        -----

        The eigenvalue equation:

        .. math::
            \hat{H}_{int}|\phi_n\rangle = E_n|\phi_n\rangle

        gives the stationary states :math:`|\phi_n\rangle` with energies :math:`E_n`.
        They are stationary rays of the auxiliary unitary flow, not a proof of
        stable or attracting TNFR graph configurations.
        """

        # Use eigh for Hermitian matrices (more efficient and numerically stable)
        eigenvalues, eigenvectors = np.linalg.eigh(self.H_int)

        return eigenvalues, eigenvectors

    def compute_node_delta_nfr(self, node: Any) -> float:
        r"""Return the identically zero localized-projector read-out.

        Parameters
        ----------
        node : NodeId
            Node identifier

        Returns
        -------
        delta_nfr : float
            Zero for a node in this Hamiltonian's captured support

        Theory
        ------

        For node n, the ΔNFR is computed as:

        .. math::
            \Delta\text{NFR}_n = \frac{i}{\hbar_{str}} \langle n | [\hat{H}_{int}, \rho_n] | n \rangle

        where :math:`\rho_n = |n\rangle\langle n|` is the density matrix for a
        pure state localized on node n.

        Notes
        -----

        For this projector, ``[H, rho_n][n,n] = H[n,n] - H[n,n] = 0`` for any
        matrix H. Evaluating large matrix products cannot add a structural
        rate to this identity. This compatibility read-out is not the engine's
        configured pressure and supplies no density-matrix evolution.
        """
        if node not in self.nodes:
            raise ValueError(f"Node {node} not found in Hamiltonian")
        return 0.0


# Standalone builder functions for modular usage


def _finite_real_coefficient(value: Any, name: str) -> float:
    """Return a finite, non-Boolean real coefficient."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar, not a boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _coherence_strength(value: Any) -> float:
    """Return a validated affinity-potential coefficient."""

    return _finite_real_coefficient(value, "C_0")


def _coupling_strength(value: Any) -> float:
    """Return a validated topological-coupling coefficient."""

    return _finite_real_coefficient(value, "J_0")


def _validated_node_order(
    G: TNFRGraph,
    nodes: Sequence[Any] | None,
) -> tuple[Any, ...]:
    """Return ``nodes`` after validating complete graph support exactly once."""

    requested = tuple(cached_node_list(G) if nodes is None else nodes)
    graph_nodes = tuple(G.nodes)
    try:
        same_nodes = (
            len(requested) == len(graph_nodes)
            and len(set(requested)) == len(requested)
            and set(requested) == set(graph_nodes)
        )
    except TypeError as exc:
        raise TypeError("nodes must contain hashable graph node identifiers") from exc
    if not same_nodes:
        raise ValueError("nodes must be a duplicate-free permutation of graph nodes")
    return requested


def _dense_coherence_affinity(
    G: TNFRGraph,
    nodes: Sequence[Any],
) -> FloatMatrix:
    """Materialize and reorder the auxiliary affinity exactly once."""

    from ..metrics.coherence import _is_sparse_affinity_payload, coherence_matrix

    requested = _validated_node_order(G, nodes)

    affinity_nodes, payload = coherence_matrix(
        G,
        _force_dense=True,
        _record_history=False,
    )
    size = len(requested)
    if affinity_nodes is None or payload is None:
        return np.zeros((size, size), dtype=complex)
    source_nodes = tuple(affinity_nodes)
    if len(source_nodes) != size or set(source_nodes) != set(requested):
        raise RuntimeError("coherence affinity node support changed during assembly")

    if isinstance(payload, list):
        if not payload:
            matrix = np.zeros((size, size), dtype=complex)
        elif _is_sparse_affinity_payload(payload):
            matrix = np.zeros((size, size), dtype=complex)
            for row, column, weight in payload:
                if not 0 <= row < size or not 0 <= column < size:
                    raise ValueError("sparse coherence affinity index is out of range")
                matrix[row, column] = weight
        else:
            matrix = np.asarray(payload, dtype=complex)
    else:
        matrix = np.asarray(payload, dtype=complex)

    if matrix.shape != (size, size):
        raise ValueError(
            f"coherence affinity shape {matrix.shape} does not match "
            f"node count ({size}, {size})"
        )
    source_index = {node: index for index, node in enumerate(source_nodes)}
    permutation = [source_index[node] for node in requested]
    return np.asarray(matrix[np.ix_(permutation, permutation)], dtype=complex)


def build_H_coherence(
    G: TNFRGraph,
    nodes: list | None = None,
    C_0: float = -1.0,
) -> FloatMatrix:
    """Construct the legacy Hamiltonian's structural-affinity contribution.

    This is ``C_0 * W`` for the auxiliary affinity matrix ``W``.  It is not
    canonical total coherence ``C(t)`` and ``W`` need not be positive
    semidefinite.  ``nodes`` must be a permutation of all graph nodes; the
    returned matrix is explicitly reordered to match it.

    Parameters
    ----------
    G : TNFRGraph
        Graph with structural attributes
    nodes : list, optional
        Ordered list of nodes. If None, uses cached_node_list(G)
    C_0 : float, default=-1.0
        Affinity-potential strength (negative for an attractive contribution)

    Returns
    -------
    H_coh : ndarray, shape (N, N)
        Coherence potential matrix
    """
    if nodes is None:
        nodes = list(cached_node_list(G))
    strength = _coherence_strength(C_0)
    return strength * _dense_coherence_affinity(G, nodes)


def build_H_frequency(
    G: TNFRGraph,
    nodes: list | None = None,
) -> FloatMatrix:
    """Construct diagonal frequency operator from graph.

    Parameters
    ----------
    G : TNFRGraph
        Graph with 'nu_f' attributes
    nodes : list, optional
        Ordered list of nodes. If None, uses cached_node_list(G)

    Returns
    -------
    H_freq : ndarray, shape (N, N)
        Diagonal frequency operator
    """
    nodes = _validated_node_order(G, nodes)

    N = len(nodes)
    frequencies = np.zeros(N, dtype=float)

    for i, node in enumerate(nodes):
        frequencies[i] = get_attr(
            G.nodes[node],
            ALIAS_VF,
            0.0,
            strict=True,
            conv=lambda value: _finite_real_coefficient(value, "nu_f"),
        )

    return np.diag(frequencies).astype(complex)


def build_H_coupling(
    G: TNFRGraph,
    nodes: Sequence[Any] | None = None,
    J_0: float = 0.1,
) -> FloatMatrix:
    """Construct coupling from the underlying simple undirected support.

    Each unordered pair with at least one edge receives exactly one ``J_0``
    entry in each symmetric position. Direction, reciprocal arcs and parallel
    edges therefore collapse. A self-loop contributes one diagonal ``J_0``.

    Parameters
    ----------
    G : TNFRGraph
        Graph with edge structure
    nodes : list, optional
        Complete node permutation defining matrix order. If None, uses
        cached_node_list(G)
    J_0 : float, default=0.1
        Finite real coupling strength; booleans are rejected

    Returns
    -------
    H_coupling : ndarray, shape (N, N)
        Real-symmetric support-coupling matrix
    """

    ordered_nodes = _validated_node_order(G, nodes)
    strength = _coupling_strength(J_0)
    size = len(ordered_nodes)
    H_coupling = np.zeros((size, size), dtype=complex)
    node_to_idx = {node: i for i, node in enumerate(ordered_nodes)}

    for u, v in G.edges():
        i = node_to_idx[u]
        j = node_to_idx[v]
        H_coupling[i, j] = strength
        if i != j:
            H_coupling[j, i] = strength

    return H_coupling
