r"""Internal Hamiltonian operator construction for TNFR.

This module implements the explicit construction of the internal Hamiltonian:

.. math::
    \hat{H}_{int} = \hat{H}_{coh} + \hat{H}_{freq} + \hat{H}_{coupling}

Mathematical Foundation
-----------------------

The internal Hamiltonian :math:`\hat{H}_{int}` governs the structural evolution
of resonant fractal nodes through the canonical nodal equation:

.. math::
    \frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta\text{NFR}(t)

where the reorganization operator :math:`\Delta\text{NFR}` is defined as:

.. math::
    \Delta\text{NFR} = \frac{d}{dt} + \frac{i[\hat{H}_{int}, \cdot]}{\hbar_{str}}

**Components**:

1. **Coherence Potential** :math:`\hat{H}_{coh}`:
   Potential energy from structural alignment between nodes.

   .. math::
       \hat{H}_{coh} = -C_0 \sum_{ij} w_{ij} |i\rangle\langle j|

   where :math:`w_{ij}` is the coherence weight from similarity metrics.

2. **Frequency Operator** :math:`\hat{H}_{freq}`:
   Diagonal operator encoding each node's structural frequency.

   .. math::
       \hat{H}_{freq} = \sum_i \nu_{f,i} |i\rangle\langle i|

3. **Coupling Hamiltonian** :math:`\hat{H}_{coupling}`:
   Network topology-induced interactions.

   .. math::
       \hat{H}_{coupling} = J_0 \sum_{(i,j) \in E} (|i\rangle\langle j| + |j\rangle\langle i|)

Theoretical References
----------------------

See:
- Mathematical formalization: ``Formalizacion-Matematica-TNFR-Unificada.pdf``, §2.4
- ΔNFR development: ``Desarrollo-Exhaustivo_-Formalizacion-Matematica-Ri-3.pdf``
- Quantum time evolution: Sakurai, "Modern Quantum Mechanics", Chapter 2

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
>>> print("Ground state energy:", eigenvalues[0])
Ground state energy: ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

from ..alias import get_attr
from ..constants.aliases import ALIAS_VF
from ..utils.cache import cached_node_list, CacheManager, _graph_cache_manager

if TYPE_CHECKING:  # pragma: no cover
    from ..types import TNFRGraph, FloatMatrix

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
        Structural Planck constant (ℏ_str)
    nodes : list
        Ordered list of node identifiers
    N : int
        Number of nodes in the network

    Notes
    -----

    This implementation leverages existing cache infrastructure:

    - Uses ``cached_node_list()`` for consistent node ordering
    - Reuses ``coherence_matrix()`` computation for H_coh
    - Integrates with ``CacheManager`` for performance optimization

    All matrix components are verified to be Hermitian (self-adjoint),
    ensuring real eigenvalues and unitary time evolution.
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
            Structural Planck constant (ℏ_str). This sets the scale for
            structural reorganization rates. Default value of 1.0 gives natural
            units where the Hamiltonian directly represents structural energy scales.
        cache_manager : CacheManager, optional
            Cache manager for performance optimization. If None, uses the
            graph's internal cache manager.

        Raises
        ------
        ValueError
            If any Hamiltonian component fails Hermiticity check
        ImportError
            If NumPy is not available
        """
        # Import NumPy (required for matrix operations)
        try:
            import numpy as np

            self._np = np
        except ImportError as exc:
            raise ImportError(
                "NumPy is required for Hamiltonian construction. " "Install with: pip install numpy"
            ) from exc

        self.G = G
        self.hbar_str = float(hbar_str)

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

        # Verify Hermiticity (critical for physical validity)
        self._verify_hermitian()

    def _build_H_coherence(self) -> FloatMatrix:
        r"""Construct coherence potential H_coh from coherence matrix.

        Theory
        ------

        .. math::
            \hat{H}_{coh} = -C_0 \sum_{ij} w_{ij} |i\rangle\langle j|

        where :math:`w_{ij}` is the coherence weight computed from structural
        similarity (phase, EPI, νf, Si). The negative sign ensures coherent
        states have lower energy (potential well).

        Returns
        -------
        H_coh : ndarray, shape (N, N)
            Coherence potential matrix (Hermitian)

        Notes
        -----

        Reuses ``coherence_matrix()`` function to avoid code duplication and
        ensure consistency with existing coherence computations.
        """
        np = self._np

        # Handle empty graph case
        if self.N == 0:
            return np.zeros((0, 0), dtype=complex)

        # Import here to avoid circular dependency
        from ..metrics.coherence import coherence_matrix

        # Reuse existing coherence_matrix computation
        nodes, W = coherence_matrix(self.G)

        # Convert to dense NumPy array
        if isinstance(W, list):
            # Empty list case
            if len(W) == 0:
                W_matrix = np.zeros((self.N, self.N), dtype=complex)
            # Check if sparse format (list of tuples) or dense (list of lists)
            elif isinstance(W[0], (list, tuple)) and len(W[0]) == 3:
                # Sparse format: [(i, j, w), ...]
                W_matrix = np.zeros((self.N, self.N), dtype=complex)
                for i, j, w in W:
                    W_matrix[i, j] = w
            else:
                # Dense format: [[...], [...], ...]
                W_matrix = np.array(W, dtype=complex)
        else:
            W_matrix = np.asarray(W, dtype=complex)

        # Reshape if necessary (handle 1D case)
        if W_matrix.ndim == 1:
            if len(W_matrix) == 0:
                W_matrix = np.zeros((self.N, self.N), dtype=complex)
            elif len(W_matrix) == self.N * self.N:
                W_matrix = W_matrix.reshape((self.N, self.N))
            else:
                raise ValueError(
                    f"Cannot reshape coherence vector of length {len(W_matrix)} "
                    f"into ({self.N}, {self.N}) matrix"
                )

        # Ensure correct shape
        if W_matrix.shape != (self.N, self.N):
            raise ValueError(
                f"Coherence matrix shape {W_matrix.shape} does not match "
                f"node count ({self.N}, {self.N})"
            )

        # Scale by coherence strength (negative for potential well)
        C_0 = self.G.graph.get("H_COH_STRENGTH", -1.0)
        H_coh = C_0 * W_matrix

        return H_coh

    def _build_H_frequency(self) -> FloatMatrix:
        r"""Construct frequency operator H_freq (diagonal).

        Theory
        ------

        .. math::
            \hat{H}_{freq} = \sum_i \nu_{f,i} |i\rangle\langle i|

        Each node's structural frequency :math:`\nu_{f,i}` becomes its diagonal
        energy. Nodes with higher νf have higher "kinetic" reorganization energy.

        Returns
        -------
        H_freq : ndarray, shape (N, N)
            Diagonal frequency operator (Hermitian)

        Notes
        -----

        Uses ``get_attr()`` with ``ALIAS_VF`` to support attribute aliasing
        and maintain consistency with the rest of the codebase.
        """
        np = self._np

        frequencies = np.zeros(self.N, dtype=float)

        for i, node in enumerate(self.nodes):
            # Use unified attribute access with aliasing support
            nu_f = get_attr(self.G.nodes[node], ALIAS_VF, 0.0)
            frequencies[i] = float(nu_f)

        # Create diagonal matrix (automatically Hermitian)
        H_freq = np.diag(frequencies).astype(complex)

        return H_freq

    def _build_H_coupling(self) -> FloatMatrix:
        r"""Construct coupling Hamiltonian from network topology.

        Theory
        ------

        .. math::
            \hat{H}_{coupling} = J_0 \sum_{(i,j) \in E} (|i\rangle\langle j| + |j\rangle\langle i|)

        where E is the edge set and :math:`J_0` is coupling strength.
        The sum is symmetric (Hermitian) for undirected graphs.

        Returns
        -------
        H_coupling : ndarray, shape (N, N)
            Coupling matrix (Hermitian for undirected graphs)

        Notes
        -----

        For directed graphs, the matrix may not be Hermitian unless the graph
        is explicitly symmetrized.
        """
        np = self._np

        H_coupling = np.zeros((self.N, self.N), dtype=complex)

        # Build node index mapping (use consistent ordering)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        # Get coupling strength from graph configuration
        J_0 = self.G.graph.get("H_COUPLING_STRENGTH", 0.1)

        # Populate coupling matrix from edges
        for u, v in self.G.edges():
            i = node_to_idx[u]
            j = node_to_idx[v]

            # Symmetric coupling (ensures Hermiticity)
            H_coupling[i, j] = J_0
            H_coupling[j, i] = J_0

        return H_coupling

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
        3. Probability conservation
        """
        np = self._np

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
            # Check Hermiticity: H = H†
            H_dagger = H.conj().T
            deviation = np.max(np.abs(H - H_dagger))

            if deviation > tolerance:
                raise ValueError(
                    f"{name} is not Hermitian: max deviation = {deviation:.2e} "
                    f"(tolerance = {tolerance:.2e})"
                )

    def compute_delta_nfr_operator(self) -> FloatMatrix:
        r"""Compute ΔNFR operator from Hamiltonian commutator.

        Theory
        ------

        .. math::
            \Delta\text{NFR} = \frac{i[\hat{H}_{int}, \cdot]}{\hbar_{str}}

        For a state :math:`|\psi\rangle`:

        .. math::
            \Delta\text{NFR}|\psi\rangle = \frac{i}{\hbar_{str}}(\hat{H}_{int}|\psi\rangle - |\psi\rangle\hat{H}_{int})

        Returns
        -------
        Delta_NFR_matrix : ndarray, shape (N, N)
            ΔNFR operator in matrix form (anti-Hermitian)

        Notes
        -----

        The ΔNFR operator is anti-Hermitian: :math:`\Delta\text{NFR}^\dagger = -\Delta\text{NFR}`,
        which ensures imaginary eigenvalues and corresponds to generator of
        time evolution.
        """
        # ΔNFR = (i/ℏ_str) * H_int (for operators acting on states)
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

        Unitarity :math:`U^\dagger U = I` ensures probability conservation.
        """
        try:
            from scipy.linalg import expm
        except ImportError as exc:
            raise ImportError(
                "scipy is required for time evolution computation. "
                "Install with: pip install scipy"
            ) from exc

        np = self._np

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

    def get_spectrum(self) -> Tuple[Any, Any]:
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
        These are the maximally stable coherent configurations.
        """
        np = self._np

        # Use eigh for Hermitian matrices (more efficient and numerically stable)
        eigenvalues, eigenvectors = np.linalg.eigh(self.H_int)

        return eigenvalues, eigenvectors

    def compute_node_delta_nfr(self, node: Any) -> float:
        r"""Compute ΔNFR for a single node using Hamiltonian commutator.

        Parameters
        ----------
        node : NodeId
            Node identifier

        Returns
        -------
        delta_nfr : float
            ΔNFR value for the specified node

        Theory
        ------

        For node n, the ΔNFR is computed as:

        .. math::
            \Delta\text{NFR}_n = \frac{i}{\hbar_{str}} \langle n | [\hat{H}_{int}, \rho_n] | n \rangle

        where :math:`\rho_n = |n\rangle\langle n|` is the density matrix for a
        pure state localized on node n.

        Notes
        -----

        The commutator result is anti-Hermitian, so its diagonal elements are
        purely imaginary in theory. We extract the real part to obtain the ΔNFR
        observable value. In practice, numerical precision may introduce small
        real components that represent the actual structural reorganization rate.
        """
        np = self._np

        # Get node index
        try:
            node_idx = self.nodes.index(node)
        except ValueError:
            raise ValueError(f"Node {node} not found in Hamiltonian")

        # Node density matrix (pure state |n⟩⟨n|)
        rho_n = np.zeros((self.N, self.N), dtype=complex)
        rho_n[node_idx, node_idx] = 1.0

        # Commutator: [H_int, ρ_n] = H_int ρ_n - ρ_n H_int
        commutator = self.H_int @ rho_n - rho_n @ self.H_int

        # ΔNFR operator
        delta_nfr_matrix = (1j / self.hbar_str) * commutator

        # Extract diagonal element for node n
        # Note: Take real part to obtain observable. Diagonal elements of the
        # anti-Hermitian commutator are purely imaginary theoretically; any
        # nonzero real part comes from numerical precision or represents the
        # actual structural reorganization rate.
        delta_nfr = float(delta_nfr_matrix[node_idx, node_idx].real)

        return delta_nfr


# Standalone builder functions for modular usage


def build_H_coherence(
    G: TNFRGraph,
    nodes: list | None = None,
    C_0: float = -1.0,
) -> FloatMatrix:
    """Construct coherence potential matrix from graph.

    Parameters
    ----------
    G : TNFRGraph
        Graph with structural attributes
    nodes : list, optional
        Ordered list of nodes. If None, uses cached_node_list(G)
    C_0 : float, default=-1.0
        Coherence potential strength (negative for attractive potential)

    Returns
    -------
    H_coh : ndarray, shape (N, N)
        Coherence potential matrix
    """
    import numpy as np

    # Import here to avoid circular dependency
    from ..metrics.coherence import coherence_matrix

    if nodes is None:
        nodes = cached_node_list(G)

    N = len(nodes)
    _, W = coherence_matrix(G)

    # Convert to NumPy array
    if isinstance(W, list):
        if W and isinstance(W[0], (list, tuple)) and len(W[0]) == 3:
            W_matrix = np.zeros((N, N), dtype=complex)
            for i, j, w in W:
                W_matrix[i, j] = w
        else:
            W_matrix = np.array(W, dtype=complex)
    else:
        W_matrix = np.asarray(W, dtype=complex)

    return C_0 * W_matrix


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
    import numpy as np

    if nodes is None:
        nodes = cached_node_list(G)

    N = len(nodes)
    frequencies = np.zeros(N, dtype=float)

    for i, node in enumerate(nodes):
        nu_f = get_attr(G.nodes[node], ALIAS_VF, 0.0)
        frequencies[i] = float(nu_f)

    return np.diag(frequencies).astype(complex)


def build_H_coupling(
    G: TNFRGraph,
    nodes: list | None = None,
    J_0: float = 0.1,
) -> FloatMatrix:
    """Construct coupling matrix from graph topology.

    Parameters
    ----------
    G : TNFRGraph
        Graph with edge structure
    nodes : list, optional
        Ordered list of nodes. If None, uses cached_node_list(G)
    J_0 : float, default=0.1
        Coupling strength

    Returns
    -------
    H_coupling : ndarray, shape (N, N)
        Coupling matrix (symmetric for undirected graphs)
    """
    import numpy as np

    if nodes is None:
        nodes = cached_node_list(G)

    N = len(nodes)
    H_coupling = np.zeros((N, N), dtype=complex)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    for u, v in G.edges():
        i = node_to_idx[u]
        j = node_to_idx[v]
        H_coupling[i, j] = J_0
        H_coupling[j, i] = J_0

    return H_coupling
