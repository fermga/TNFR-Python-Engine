r"""Structural diffusion and scoped diagnostics for the nodal EPI channel.

For a fixed, nonnegative adjacency W, let D contain its row strengths and
L_rw = I - D^-1 W, with a zero row at an isolated node. The canonical EPI
pressure is DeltaNFR_epi = -L_rw EPI. With the capacity vector nu_f, the
isolated channel therefore evolves as

    dEPI/dt = -A EPI,    A = diag(nu_f) L_rw.

The full pressure also includes phase, frequency and topology channels.
Wrapped circular phase averaging is nonlinear and is not this scalar
Laplacian identity. These diagnostics hold the graph and capacity fixed;
they do not integrate a changing canonical operator sequence.

Transport and spectral scope
----------------------------
For symmetric adjacency and a common positive capacity nu, each mode
decays at nu*lambda_k. A connected graph relaxes to a uniform field and
conserves sum(d_i*EPI_i). With disconnected components, stationarity need
not be globally uniform. With positive heterogeneous capacity the conserved
weights are d_i/nu_i and the decay rates are eigenvalues of A; replacing
capacity by its mean changes the dynamics. Zero capacity freezes a node
even when its pressure is nonzero. Directed transport uses the actual
nonsymmetric A; real eigenvalue parts describe asymptotic damping, without
certifying normality or absence of transient growth.

The symmetric normalized Laplacian provides an orthonormal geometry basis
only for symmetric adjacency. Its zero mode is proportional to sqrt(d)
on a connected component, corresponding to a uniform EPI field after the
degree-coordinate transformation. Finite dimension gives a finite spectrum;
a nodal-domain upper bound does not imply monotonic domain counts for every
graph or every basis of a degenerate eigenspace.

Adding a scalar reaction rate r gives real growth rates
r - Re(eigenvalues(A)). On a connected homogeneous network the first
nonuniform threshold is nu*lambda_2. This is a spatial-mode threshold,
not a bound on all evolution: the uniform mode already grows for any r>0.
The correspondence is a linear diagnostic, not a derivation of grammar U2
for arbitrary sequences.

Drift, waves and certificates
----------------------------
Under held pressure F the nodal equation is the first-order mobility law
q_dot = nu_f*F. The damped graph-wave model q_ddot + gamma*q_dot + L*q = 0
has a slow diffusion limit with mobility 1/gamma. The wave-to-diffusion
checks below concern that specified graph-wave model. They do not prove
that the isotropic Hamiltonian implemented by symplectic_substrate, or all
13 engine operators, induces that graph wave.

verify_structural_diffusion checks the canonical pressure on a replica and
samples the actual frozen-capacity flow. It reports global uniformity and
degree-weighted conservation separately from stationarity and the complete
left-nullspace invariants of A. Finite-time residuals are diagnostics, not
proofs of eventual convergence.

Random walks and currents
------------------------
P = I - L_rw is row-stochastic, with an absorbing self-transition at an
isolated node. For symmetric adjacency the degree distribution is
stationary; convergence of a discrete walk additionally requires
aperiodicity within an irreducible component. Resistance geometry requires
symmetric nonnegative conductance. Disconnected pairs have infinite
resistance and commute time; finite commute times use the volume of the
pair's component.

For symmetric W, edge currents J_ij = W_ij*(EPI_i-EPI_j) have divergence
(D-W)*EPI. At nodes with positive degree and capacity, the nodal continuity
balance is (d_i/nu_i)*dEPI_i/dt + div(J)_i = 0. This EPI-channel identity
is distinct from the tetrad charge and currents in physics.conservation.

The same conductance model gives the Dirichlet energy
E_D = (1/4)*sum_ij W_ij*(EPI_i-EPI_j)^2. Its gradient is (D-W)*EPI,
and the mobility is diag(nu_i/d_i), zero at isolates. Hence the frozen
EPI-only flow satisfies dE_D/dt = -sum_i mobility_i*gradient_i^2 <= 0.
compute_diffusion_energy reports this balance without evolving the graph.
This energy is distinct from the tetrad potential in physics.variational.

References within the implementation: dynamics.dnfr (canonical pressure),
directed_diffusion (nonsymmetric transport), symplectic_substrate (specified
Hamiltonian), and physics.conservation (tetrad diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..mathematics.unified_numerical import np
from ._conductance import ConductanceSnapshot, read_conductance

__all__ = [
    "DiffusionEnergyBalance",
    "StructuralDiffusionCertificate",
    "OverdampedRegimeCertificate",
    "OverdampedProjectionCertificate",
    "UndampedLimitCertificate",
    "DiscreteModeCertificate",
    "StructuralStabilityCertificate",
    "RandomWalkCertificate",
    "StructuralFlowCertificate",
    "structural_diffusion_operator",
    "symmetric_normalized_laplacian",
    "structural_field",
    "structural_diffusivity",
    "relaxation_spectrum",
    "structural_frequency_rank",
    "degree_weighted_total",
    "compute_diffusion_energy",
    "structural_eigenmodes",
    "nodal_domain_count",
    "compute_emergent_pulse",
    "compute_nodal_pulse",
    "dispersion_relation",
    "instability_threshold",
    "fiedler_partition",
    "random_walk_matrix",
    "stationary_distribution",
    "effective_resistance",
    "commute_time",
    "structural_current",
    "current_divergence",
    "verify_structural_diffusion",
    "verify_overdamped_regime",
    "damped_wave_rates",
    "verify_overdamped_projection",
    "verify_undamped_limit",
    "verify_discrete_modes",
    "verify_structural_stability",
    "verify_structural_random_walk",
    "verify_structural_flow",
]


def _ordered_nodes(G: Any) -> list:
    """Stable node ordering for the matrix representation."""
    return list(G.nodes())


def _weighted_adjacency(G: Any, nodes: list | None = None) -> tuple[list, Any]:
    """One nonnegative adjacency convention, including parallel edges/loops."""
    conductance = read_conductance(G, nodes)
    return conductance.nodes, conductance.dense()


def _nodal_frequencies(G: Any, nodes: list | None = None) -> Any:
    """Read the actual capacity vector; zero capacity freezes a node."""
    if nodes is None:
        nodes = _ordered_nodes(G)
    try:
        frequency = np.array([
            get_attr(G.nodes[node], ALIAS_VF, 0.0, conv=float, strict=True)
            for node in nodes
        ], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Structural frequency must be finite and nonnegative") from exc
    if not np.all(np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise ValueError("Structural frequency must be finite and nonnegative")
    return frequency


def structural_diffusion_operator(G: Any) -> tuple[list, Any]:
    r"""Return the random-walk graph Laplacian L_rw = I − D⁻¹W.

    This is the operator whose action on a field is exactly the canonical
    ΔNFR ``neighbour-mean − self`` gradient: g = −L_rw·field.  Built from
    the (optionally weighted) adjacency; isolated nodes (degree 0) get a
    zero row (no diffusion).

    Finite effective edge weights are normalized in scaled coordinates, so
    a raw row sum need not fit in a float. Raw degree-weighted quantities
    retain their separate representability requirements.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, L_rw) : tuple[list, np.ndarray]
        The node ordering and the N×N random-walk Laplacian.
    """
    conductance = read_conductance(G)
    probability, scale, _ = conductance.normalization()
    laplacian = conductance.dense(-probability)
    diagonal = np.diag_indices(len(conductance.nodes))
    laplacian[diagonal] += (scale > 0.0)
    return conductance.nodes, laplacian


def symmetric_normalized_laplacian(
    G: Any, nodes: list | None = None
) -> tuple[list, Any]:
    r"""Return the symmetric normalized Laplacian L_sym = I − D^{-1/2} W D^{-1/2}.

    For symmetric adjacency, L_sym shares the spectrum of the diffusion operator
    L_rw = I − D⁻¹W (:func:`structural_diffusion_operator`) but is symmetric, so
    it has an orthonormal eigenbasis and real eigenvalues — the canonical choice
    for the relaxation spectrum (its λ₂ is the structural ``diffusion_gap``).
    Isolated nodes (degree 0) get a zero row. Asymmetric adjacency is rejected:
    it cannot be passed to a symmetric eigensolver. Directed damping rates are
    available through :func:`relaxation_spectrum`.

    Scaled conductance avoids overflowing raw row sums; square-root ratios
    retain representable symmetric coefficients at very unequal strengths.

    Parameters
    ----------
    G : TNFRGraph
    nodes : list, optional
        Node ordering; defaults to the stable ``list(G.nodes())`` order.

    Returns
    -------
    (nodes, L_sym) : tuple[list, np.ndarray]
        The node ordering and the N×N symmetric normalized Laplacian.
    """
    conductance = read_conductance(G, nodes, symmetric=True)
    normalized, positive = conductance.symmetric_normalized_weights()
    lap = conductance.dense(-normalized)
    lap[np.diag_indices(len(conductance.nodes))] += positive
    return conductance.nodes, lap


def structural_field(G: Any, nodes: list | None = None) -> Any:
    r"""Return the EPI field as a vector aligned with ``nodes``."""
    if nodes is None:
        nodes = _ordered_nodes(G)
    return np.array(
        [get_attr(G.nodes[n], ALIAS_EPI, 0.0, conv=float, strict=True) for n in nodes],
        dtype=float,
    )


def structural_diffusivity(G: Any) -> float:
    r"""Mean structural frequency, a descriptive capacity summary.

    In ∂EPI/∂t = −νf·L_rw·EPI, νf plays the role of the diffusivity: the
    larger the structural frequency, the faster the form spreads. This mean
    is a diffusion coefficient only when all nodal frequencies are equal;
    heterogeneous transport uses diag(νf)·L_rw, never mean(νf)·L_rw.
    """
    vf = _nodal_frequencies(G)
    return float(np.mean(vf)) if len(vf) else 0.0


def degree_weighted_total(G: Any) -> float:
    r"""The row-strength-weighted total Σ_i d_i·EPI(i).

    For symmetric adjacency and a common frequency this total is conserved.
    For fixed positive heterogeneous frequencies the conserved weights are
    d_i/νf_i instead; a directed graph requires the generator's left nullspace.
    This helper keeps its literal weighted-total meaning in every case.
    Nonfinite EPI or an unrepresentable final total raises ValueError.
    Edge-based summation allows finite cancellation even when an intermediate
    degree or product exceeds float range.

    Under those restrictions this is an **EPI-channel** conserved quantity.
    It is **distinct** from the tetrad Noether charge
    Q = Σ(Φ_s + K_φ)
    (:func:`tnfr.physics.conservation.compute_noether_charge`), conserved under
    grammar U1–U6: TNFR carries two distinct conservation laws, on the EPI
    field and on the tetrad fields respectively (see
    STRUCTURAL_CONSERVATION_THEOREM §8.7).
    """
    conductance = read_conductance(G)
    field = structural_field(G, conductance.nodes)
    if not np.all(np.isfinite(field)):
        raise ValueError("Structural transport requires finite scalar EPI")
    total = conductance.weighted_total(field)
    if not np.isfinite(total):
        raise ValueError("Degree-weighted total exceeds finite floating-point range")
    return total


def _read_edge_flux(G: Any) -> tuple[ConductanceSnapshot, Any, Any]:
    """Share edge differences between current, divergence and energy gradient.

    Capacity is deliberately absent: constitutive current can remain nonzero
    at a frozen node. Effective zero edges are removed before subtraction.
    """
    conductance = read_conductance(G, symmetric=True)
    field = structural_field(G, conductance.nodes)
    if not np.all(np.isfinite(field)):
        raise ValueError("Structural transport requires finite scalar EPI")
    try:
        with np.errstate(over="raise", invalid="raise"):
            difference = field[conductance.source] - field[conductance.target]
            flux = conductance.weight * difference
    except FloatingPointError as exc:
        raise ValueError("Structural current exceeds finite floating-point range") from exc
    return conductance, difference, flux


@dataclass(frozen=True)
class DiffusionEnergyBalance:
    """Instantaneous balance for the fixed symmetric EPI-only channel.

    Arrays follow ``nodes`` and are detached from the graph. ``gradient``
    is the Euclidean derivative of ``energy``; ``epi_rate`` includes the
    actual nodal capacities. Zero mobility is permitted and does not imply
    zero pressure. This is not a certificate for the full tetrad energy,
    changing topology, or arbitrary finite integration steps.
    """

    nodes: list
    energy: float
    gradient: Any
    mobility: Any
    epi_rate: Any
    energy_rate: float


def compute_diffusion_energy(G: Any) -> DiffusionEnergyBalance:
    r"""Read the Dirichlet gradient-flow balance on symmetric conductance.

    With B = D-W and x = EPI, E_D = x^T B x / 2 and grad(E_D) = Bx.
    The canonical pure EPI channel is x' = -M grad(E_D), where
    M_ii = nu_f_i/d_i for positive row strength and zero otherwise.
    Thus E_D' = -grad(E_D)^T M grad(E_D) <= 0. For positive capacity
    and strength this is the gradient flow in metric diag(d_i/nu_f_i);
    zero capacity instead gives a degenerate positive-semidefinite mobility.

    Nonnegative finite weights, finite scalar EPI and nonnegative finite
    capacities are required. Parallel edges and loops follow the shared
    diffusion adjacency convention; loops add strength but no energy.
    Asymmetric adjacency has no such symmetric Dirichlet identity and is
    rejected. No graph attributes, pressure callbacks or caches are changed.
    Unrepresentable floating-point balances raise ValueError.
    A row strength itself may exceed float range when the returned mobility,
    gradient, energy and rates remain representable in scaled coordinates.
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            conductance, difference, flux = _read_edge_flux(G)
            nodes = conductance.nodes
            frequency = _nodal_frequencies(G, nodes)
            mobility = conductance.divide_by_strength(frequency)
            gradient = conductance.divergence(flux)
            energy = float(0.25 * np.sum(flux * difference))
            epi_rate = -mobility * gradient
            energy_rate = float(gradient @ epi_rate)
    except FloatingPointError as exc:
        raise ValueError("Diffusion energy balance exceeds finite floating-point range") from exc
    return DiffusionEnergyBalance(nodes, energy, gradient, mobility, epi_rate, energy_rate)


def relaxation_spectrum(G: Any) -> Any:
    r"""Real decay rates of diag(νf)·L_rw, sorted ascending.

    Equal frequencies reduce to νf·λ_k and reuse the cached symmetric geometry
    on an undirected graph. With heterogeneous frequency, the generator is
    diag(νf)·L_rw. Directed rates are real parts of its possibly complex
    eigenvalues; they do not specify oscillation or transient amplification.

    Returns
    -------
    np.ndarray
        The decay rates sorted ascending (real parts).
    """
    frequency = _nodal_frequencies(G)
    if not len(frequency):
        return np.empty(0, dtype=float)
    if not G.is_directed():
        if np.all(frequency == frequency[0]):
            eig = _cached_eigenvalues(G)
            return frequency[0] * eig
        _, lap_sym = symmetric_normalized_laplacian(G)
        root_frequency = np.sqrt(frequency)
        # AB and BA share eigenvalues even if a frequency is zero.
        rates = np.linalg.eigvalsh(
            root_frequency[:, None] * lap_sym * root_frequency[None, :]
        )
    else:
        _, lap = structural_diffusion_operator(G)
        rates = np.linalg.eigvals(frequency[:, None] * lap).real
    return np.maximum(np.sort(rates), 0.0)


def structural_frequency_rank(G: Any, decimals: int = 8) -> int:
    r"""Number of distinct structural frequencies (the structural rank).

    The distinct eigenvalues of the canonical random-walk Laplacian L_rw are
    the network's structural frequencies (the relaxation rates of
    ∂EPI/∂t = −νf·L_rw·EPI, up to the νf scale). This returns their count s(G)
    — the size of the distinct-frequency spectrum — complementing
    ``relaxation_spectrum`` (which returns the rates themselves).

    For a connected graph, s distinct eigenvalues bound the diameter by s−1,
    and s = 2 iff the graph is complete (regular case). On arithmetic Cayley
    networks the rank is a primality / cyclotomy diagnostic (see
    :mod:`tnfr.mathematics.number_theory`): the quadratic-residue network on an
    odd prime has rank 3, and the k-th power residue network on a prime p has
    rank ``gcd(k, p-1) + 1``.

    Note
    ----
    For large or dense graphs the distinct-eigenvalue count is sensitive to the
    ``decimals`` rounding (floating-point noise in ``eigvals`` can split truly
    equal eigenvalues). For arithmetic residue networks the exact multiplicative
    :func:`tnfr.mathematics.number_theory.quadratic_residue_annotated_rank` is
    the robust closed-form object; this scalar count agrees with it for small
    moduli.

    Parameters
    ----------
    G : TNFRGraph
    decimals : int
        Rounding applied to the real and imaginary parts before counting
        distinct values (the spectrum may be complex for directed graphs).

    Returns
    -------
    int
        The number of distinct eigenvalues of L_rw.
    """
    _, lap = structural_diffusion_operator(G)
    eig = np.linalg.eigvals(lap)
    rounded = np.round(eig.real, decimals) + 1j * np.round(eig.imag, decimals)
    return int(np.unique(rounded).size)


@dataclass(frozen=True)
class StructuralDiffusionCertificate:
    r"""Verification that the nodal equation's EPI channel is graph diffusion.

    Attributes
    ----------
    n_nodes : int
    dnfr_is_graph_laplacian : bool
        The canonical ΔNFR (EPI channel) equals −L_rw·EPI.
    max_laplacian_residual : float
        Max |ΔNFR_epi − (−L_rw·EPI)| over the nodes (≈ 0).
    diffusivity : float
        Mean νf, a capacity summary (a diffusion coefficient only if uniform).
    spectral_gap : float
        Second-smallest real part of the geometry spectrum of L_rw.
    slowest_relaxation_rate : float
        Smallest positive real decay rate of diag(νf)·L_rw.
    degree_weighted_conserved : bool
        Σ deg·EPI is conserved under the diffusion flow.
    max_conservation_drift : float
        Max drift of the degree-weighted total over the sampled flow.
    relaxes_to_uniform : bool
        The field relaxes to a spatially uniform diffusive equilibrium.
    final_field_std : float
        Std of the field after the sampled diffusion flow (≈ 0).
    invariant_weighted_conserved : bool or None
        All left-nullspace coordinates of diag(νf)·L_rw remain constant.
    max_invariant_conservation_drift : float
        Maximum absolute drift in the orthonormal invariant coordinates.
    reaches_stationary_state : bool or None
        Final nodal derivative is below the sampled-flow threshold.
    final_stationarity_residual : float
        Maximum absolute final nodal derivative; distinct from ΔNFR pressure.
    """

    n_nodes: int
    dnfr_is_graph_laplacian: bool
    max_laplacian_residual: float
    diffusivity: float
    spectral_gap: float
    slowest_relaxation_rate: float
    degree_weighted_conserved: bool
    max_conservation_drift: float
    relaxes_to_uniform: bool
    final_field_std: float
    invariant_weighted_conserved: bool | None = None
    max_invariant_conservation_drift: float = 0.0
    reaches_stationary_state: bool | None = None
    final_stationarity_residual: float = 0.0

    @property
    def is_valid_diffusion(self) -> bool:
        """True when the nodal EPI channel verifies as graph diffusion."""
        return (
            self.dnfr_is_graph_laplacian
            and (
                self.degree_weighted_conserved
                if self.invariant_weighted_conserved is None
                else self.invariant_weighted_conserved
            )
            and (
                self.relaxes_to_uniform
                if self.reaches_stationary_state is None
                else self.reaches_stationary_state
            )
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_diffusion else "INVALID"
        return (
            f"Structural diffusion [{ok}]: "
            f"ΔNFR_epi = −L_rw·EPI={self.dnfr_is_graph_laplacian} "
            f"(res {self.max_laplacian_residual:.1e}), "
            f"mean capacity νf={self.diffusivity:.4f}, "
            f"spectral gap λ₂={self.spectral_gap:.4f}, "
            f"slowest positive nodal rate={self.slowest_relaxation_rate:.4f}, "
            f"deg-weighted conserved={self.degree_weighted_conserved} "
            f"(drift {self.max_conservation_drift:.1e}), "
            f"relaxes to uniform={self.relaxes_to_uniform} "
            f"(final std {self.final_field_std:.1e}), "
            f"nodal invariants conserved={self.invariant_weighted_conserved}, "
            f"stationary={self.reaches_stationary_state}"
        )


def _dnfr_epi_channel(G: Any, nodes: list) -> Any:
    r"""Canonical ΔNFR restricted to the EPI channel, on a clean replica.

    Isolates the EPI diffusion channel by computing the canonical ΔNFR with
    weights (phase=0, epi=1, vf=0, topo=0) on a minimal structural replica
    (nodes + edges + EPI/θ/νf only), so the caller's graph is never mutated
    and the non-copyable runtime caches are not duplicated.
    """
    from ..dynamics import default_compute_delta_nfr

    g2 = G.__class__()
    for node in nodes:
        data = G.nodes[node]
        g2.add_node(
            node,
            EPI=float(get_attr(data, ALIAS_EPI, 0.0)),
            theta=float(data.get("theta", 0.0)),
            nu_f=float(get_attr(data, ALIAS_VF, 0.0)),
        )
    for u, v, data in G.edges(data=True):
        g2.add_edge(u, v, weight=float(data.get("weight", 1.0)))
    g2.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0,
        "epi": 1.0,
        "vf": 0.0,
        "topo": 0.0,
    }
    default_compute_delta_nfr(g2)
    return np.array(
        [float(get_attr(g2.nodes[n], ALIAS_DNFR, 0.0)) for n in nodes],
        dtype=float,
    )


def verify_structural_diffusion(
    G: Any,
    *,
    dt: float = 0.1,
    steps: int = 400,
    tolerance: float = 1e-9,
) -> StructuralDiffusionCertificate:
    r"""Verify the nodal equation's EPI channel is graph diffusion.

    Checks the EPI pressure identity and samples the actual frozen-frequency
    nodal flow ``e' = -diag(νf) L_rw e``. Conservation is checked against all
    left-nullspace invariants of that generator. Degree-weighted conservation
    and global uniformity are reported separately: they need not hold with
    heterogeneous frequencies, disconnected components or frozen nodes.

    ``reaches_stationary_state`` tests the final nodal derivative, not pressure
    equilibrium: zero capacity can freeze a nonuniform field. This is a finite
    sampled-flow diagnostic, not a proof of eventual convergence for every
    graph. The explicit step must preserve the nonnegative diffusion transition.

    The caller's graph is never mutated (the ΔNFR check runs on a copy).

    Parameters
    ----------
    G : TNFRGraph
    dt : float
        Forward-Euler step for the diffusion-flow checks.
    steps : int
        Number of diffusion steps for the relaxation / conservation checks.
    tolerance : float
        Maximum allowed Laplacian residual and conservation drift.

    Returns
    -------
    StructuralDiffusionCertificate
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    nodes, lap = structural_diffusion_operator(G)
    n = len(nodes)
    epi = structural_field(G, nodes)
    if not np.all(np.isfinite(epi)):
        raise ValueError("The diffusion certificate requires a finite EPI field")
    frequency = _nodal_frequencies(G, nodes)
    generator = frequency[:, None] * lap
    if dt * float(np.max(np.diag(generator), initial=0.0)) > 1.0:
        raise ValueError("dt exceeds the nonnegative explicit diffusion step bound")

    # (1) ΔNFR (epi channel) == −L_rw·EPI ?
    try:
        dnfr_epi = _dnfr_epi_channel(G, nodes)
        residual = float(np.max(np.abs(dnfr_epi + lap @ epi), initial=0.0))
        is_laplacian = residual < max(tolerance, 1e-12)
    except Exception:
        residual = float("nan")
        is_laplacian = False

    # diffusivity and spectrum
    diffusivity = float(np.mean(frequency)) if n else 0.0
    eig = np.linalg.eigvals(lap).real
    eig.sort()
    spectral_gap = max(0.0, float(eig[1])) if n > 1 else 0.0
    rates = relaxation_spectrum(G)
    slowest_rate = next((float(rate) for rate in rates if rate > tolerance), 0.0)

    # degree vector for the conserved weighted total
    deg = read_conductance(G, nodes).strength
    # Columns of U beyond rank(A) span ker(A.T), hence their scalar products
    # with e are conserved. On an undirected positive-frequency graph this
    # includes the analytic weights d_i/νf_i, component by component.
    left_vectors, singular_values, _ = np.linalg.svd(generator, full_matrices=True)
    rank_tolerance = n * np.finfo(float).eps * float(np.max(singular_values, initial=0.0))
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    invariants = left_vectors[:, rank:].T

    # (2)+(3) integrate the nodal flow at the specified physical time step.
    e = epi.copy()
    w0 = float(deg @ e)
    max_drift = 0.0
    initial_invariants = invariants @ e
    invariant_drift = 0.0
    for _ in range(steps):
        e = e - dt * (generator @ e)
        max_drift = max(max_drift, abs(float(deg @ e) - w0))
        invariant_drift = max(invariant_drift, float(np.max(
            np.abs(invariants @ e - initial_invariants), initial=0.0,
        )))
    conserved = max_drift < max(tolerance, 1e-9 * (abs(w0) + 1e-12))
    invariant_conserved = invariant_drift <= tolerance * max(
        1.0, float(np.max(np.abs(initial_invariants), initial=0.0)),
    )
    final_std = float(np.std(e)) if n else 0.0
    initial_std = float(np.std(epi)) if n else 0.0
    relaxes = final_std < max(1e-3, 1e-2 * (initial_std + 1e-12))
    stationarity_residual = float(np.max(np.abs(generator @ e), initial=0.0))
    stationary = stationarity_residual < max(1e-3, 1e-2 * float(
        np.max(np.abs(generator @ epi), initial=0.0),
    ))

    return StructuralDiffusionCertificate(
        n_nodes=n,
        dnfr_is_graph_laplacian=is_laplacian,
        max_laplacian_residual=residual,
        diffusivity=diffusivity,
        spectral_gap=spectral_gap,
        slowest_relaxation_rate=slowest_rate,
        degree_weighted_conserved=conserved,
        max_conservation_drift=max_drift,
        relaxes_to_uniform=relaxes,
        final_field_std=final_std,
        invariant_weighted_conserved=bool(invariant_conserved),
        max_invariant_conservation_drift=invariant_drift,
        reaches_stationary_state=bool(stationary),
        final_stationarity_residual=stationarity_residual,
    )


# ---------------------------------------------------------------------------
# The overdamped drift regime: the bare nodal equation is first-order
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverdampedRegimeCertificate:
    r"""Verification that the bare nodal equation is the overdamped drift law.

    The nodal equation ∂EPI/∂t = νf·ΔNFR is **first order in time**, so —
    reading EPI as a position q and ΔNFR as the structural pressure F — it
    is the **mobility / drift law** q̇ = νf·F: velocity proportional to
    force, with νf the mobility.  Under a sustained pressure the field
    drifts at *constant* velocity (linear in time), it does not accelerate.

    This is the empirically-demonstrated overdamped regime (Stokes 1851,
    Einstein 1905, terminal velocity, sedimentation, electrophoresis).  The
    inertial Newtonian regime (q̈ = F/m, second order) is the separate
    :mod:`tnfr.physics.symplectic_substrate` Hamiltonian flow. This held-force
    diagnostic does not establish a projection from that Hamiltonian.

    Attributes
    ----------
    drift_velocity : float
        v = νf·F evaluated at the reference (νf, F).
    velocity_is_constant : bool
        Under sustained pressure, dEPI/dt is constant (first-order/drift).
    max_velocity_variation : float
        Max |dEPI/dt − v| over the held-pressure integration (≈ 0).
    position_linear_in_time : bool
        EPI(t) grows linearly (slope = drift), not quadratically.
    position_slope : float
        Measured slope of EPI(t) (= the drift velocity).
    mobility_linear_in_nu_f : bool
        v ∝ νf (the mobility law): v/νf is constant across νf.
    drift_linear_in_pressure : bool
        v ∝ F: v/F is constant across F.
    is_second_order : bool
        Whether the bare equation is second order (always False — it is the
        overdamped, first-order regime).
    """

    drift_velocity: float
    velocity_is_constant: bool
    max_velocity_variation: float
    position_linear_in_time: bool
    position_slope: float
    mobility_linear_in_nu_f: bool
    drift_linear_in_pressure: bool
    is_second_order: bool

    @property
    def is_overdamped_drift(self) -> bool:
        """True when the bare nodal equation verifies as overdamped drift."""
        return (
            self.velocity_is_constant
            and self.position_linear_in_time
            and self.mobility_linear_in_nu_f
            and self.drift_linear_in_pressure
            and not self.is_second_order
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_overdamped_drift else "INVALID"
        return (
            f"Overdamped drift regime [{ok}]: "
            f"q̇ = νf·F = {self.drift_velocity:.4f} "
            f"(mobility law); "
            f"velocity constant={self.velocity_is_constant} "
            f"(var {self.max_velocity_variation:.1e}), "
            f"position linear={self.position_linear_in_time} "
            f"(slope {self.position_slope:.4f}), "
            f"v∝νf={self.mobility_linear_in_nu_f}, "
            f"v∝F={self.drift_linear_in_pressure}, "
            f"second-order={self.is_second_order}"
        )


def verify_overdamped_regime(
    *,
    nu_f: float = 0.7,
    pressure: float = 1.3,
    dt: float = 0.01,
    steps: int = 300,
    tolerance: float = 1e-9,
) -> OverdampedRegimeCertificate:
    r"""Verify the bare nodal equation is the overdamped drift law q̇ = νf·F.

    Integrates the canonical nodal equation
    (:func:`tnfr.dynamics.canonical.compute_canonical_nodal_derivative`)
    under a *sustained* structural pressure and measures that the EPI
    coordinate drifts at constant velocity v = νf·F (first-order, mobility
    law), linear in νf (mobility) and in the pressure F.  Uses the canonical
    nodal-equation function — no formula is re-implemented here.

    Parameters
    ----------
    nu_f : float
        Reference structural frequency (mobility).
    pressure : float
        Sustained structural pressure ΔNFR (= F).
    dt : float
        Integration step.
    steps : int
        Number of integration steps.
    tolerance : float
        Maximum allowed velocity variation / linearity residual.

    Returns
    -------
    OverdampedRegimeCertificate
    """
    from ..dynamics.canonical import compute_canonical_nodal_derivative

    # integrate the bare nodal equation under a held pressure
    epi = 0.0
    velocities = []
    positions = []
    for _ in range(steps):
        v = compute_canonical_nodal_derivative(nu_f, pressure).derivative
        epi = epi + dt * v
        velocities.append(v)
        positions.append(epi)
    vel = np.array(velocities, dtype=float)
    pos = np.array(positions, dtype=float)

    drift = float(vel[0])
    vel_var = float(np.max(np.abs(vel - drift)))
    vel_constant = vel_var < tolerance

    # position grows linearly with slope = drift (first-order, not quadratic)
    t = np.arange(steps, dtype=float) * dt
    slope, _ = np.polyfit(t, pos, 1)
    quad = np.polyfit(t, pos, 2)[0]  # leading quadratic coefficient ≈ 0
    pos_linear = abs(float(slope) - drift) < max(tolerance, 1e-6 * abs(drift)) and abs(
        float(quad)
    ) < max(tolerance, 1e-6 * abs(drift) + 1e-9)

    # mobility law: v ∝ νf (v/νf constant across νf)
    ratios_nu = [
        compute_canonical_nodal_derivative(nf, pressure).derivative / nf
        for nf in (0.2, 0.5, 1.0, 1.5)
    ]
    mobility_linear = float(np.std(ratios_nu)) < tolerance

    # drift ∝ F (v/F constant across F)
    ratios_f = [
        compute_canonical_nodal_derivative(nu_f, f).derivative / f
        for f in (0.3, 0.8, 1.3, 2.0)
    ]
    pressure_linear = float(np.std(ratios_f)) < tolerance

    return OverdampedRegimeCertificate(
        drift_velocity=drift,
        velocity_is_constant=vel_constant,
        max_velocity_variation=vel_var,
        position_linear_in_time=pos_linear,
        position_slope=float(slope),
        mobility_linear_in_nu_f=mobility_linear,
        drift_linear_in_pressure=pressure_linear,
        is_second_order=False,
    )


# ---------------------------------------------------------------------------
# Overdamped projection: the limit of a specified damped graph wave
# to the dissipative structural diffusion
# ---------------------------------------------------------------------------


def damped_wave_rates(G: Any, gamma: float) -> tuple[Any, Any, Any]:
    r"""Per-mode slow/fast rates of the damped graph wave q̈ + γq̇ + Lq = 0.

    The specified graph-wave model has the equation q̈ = −L q
    (second order, mode k oscillating at √λ_k — the
    standing-wave ``discrete modes`` of :func:`verify_discrete_modes`).
    Adding a damping γ gives the damped oscillator q̈ + γq̇ + L q = 0, whose
    per-mode characteristic equation is

        s² + γ s + λ_k = 0   ⟹   s± = ½(−γ ± √(γ² − 4λ_k)).

    For γ² > 4λ_k (overdamped per mode) both roots are real: a **slow** root
    s₋ → −λ_k/γ (the diffusion rate) and a **fast** root s₊ → −γ (a
    transient that dies immediately).  This is the spectral content of the
    overdamped projection: after the fast transient, mode k relaxes at
    λ_k/γ = ν_f·λ_k with ν_f = 1/γ.

    Parameters
    ----------
    G : TNFRGraph
    gamma : float
        Damping coefficient.  Its inverse is the effective diffusivity
        (mobility) ν_f = 1/γ.

    Returns
    -------
    (lambdas, s_slow, s_fast) : tuple[np.ndarray, np.ndarray, np.ndarray]
        Sorted Laplacian eigenvalues and the (real-part) slow/fast roots.
    """
    _, lap = structural_diffusion_operator(G)
    lambdas = np.sort(np.linalg.eigvals(lap).real)
    lambdas = np.clip(lambdas, 0.0, None)
    disc = gamma * gamma - 4.0 * lambdas + 0j
    root = np.sqrt(disc)
    s_slow = ((-gamma + root) / 2.0).real
    s_fast = ((-gamma - root) / 2.0).real
    return lambdas, s_slow, s_fast


@dataclass(frozen=True)
class OverdampedProjectionCertificate:
    r"""Verification of the specified graph-wave-to-diffusion limit.

    The model starts from the graph wave q̈ = −L q. Damping it and taking
    the strong-damping (Smoluchowski) limit collapses it onto the
    first-order structural diffusion q̇ = −(1/γ) L q, with the
    identification **ν_f = 1/γ** (structural frequency = inverse damping =
    mobility). This does not establish a projection of the different isotropic
    Hamiltonian implemented in :mod:`tnfr.physics.symplectic_substrate`.

    Attributes
    ----------
    n_nodes : int
    gamma : float
        Damping coefficient used for the projection.
    nu_f_effective : float
        The effective diffusivity 1/γ recovered by the projection.
    spectral_gap : float
        λ₂ of L_rw (the Fiedler value).
    lambda_max : float
        Largest Laplacian eigenvalue (sets the bridge error scale).
    max_rate_rel_error : float
        Max over modes of |s_slow + λ_k/γ| / (λ_k/γ): how far the damped
        slow rate is from the diffusion rate.
    rate_error_times_gamma_sq : float
        ``max_rate_rel_error · γ²`` — converges to ≈ λ_max, confirming the
        bridge error scales as O(λ_max/γ²).
    slowest_slow_rate : float
        Overdamped slow rate of the Fiedler mode, −s_slow(λ₂).
    slowest_diffusion_rate : float
        The diffusion spectral gap ν_f·λ₂ = λ₂/γ.
    trajectory_max_rel_error : float
        Max relative L² error between the damped-wave trajectory and the
        diffusion trajectory exp(−L t/γ)·q₀ over an overdamped time window.
    projects_to_diffusion : bool
        Whether both the rate and trajectory errors fall within tolerance.
    """

    n_nodes: int
    gamma: float
    nu_f_effective: float
    spectral_gap: float
    lambda_max: float
    max_rate_rel_error: float
    rate_error_times_gamma_sq: float
    slowest_slow_rate: float
    slowest_diffusion_rate: float
    trajectory_max_rel_error: float
    projects_to_diffusion: bool

    @property
    def is_valid_projection(self) -> bool:
        """True when the specified damped graph wave approaches diffusion."""
        return self.projects_to_diffusion

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_projection else "INVALID"
        return (
            f"Overdamped projection [{ok}]: damped graph wave "
            f"projects onto diffusion with nu_f=1/gamma="
            f"{self.nu_f_effective:.4f}; rate error "
            f"{self.max_rate_rel_error:.2e} (x gamma^2="
            f"{self.rate_error_times_gamma_sq:.3f} ~ lambda_max="
            f"{self.lambda_max:.3f}), slow gap {self.slowest_slow_rate:.5f} "
            f"vs diffusion gap {self.slowest_diffusion_rate:.5f}, "
            f"trajectory error {self.trajectory_max_rel_error:.2e}"
        )


def verify_overdamped_projection(
    G: Any,
    *,
    gamma: float = 50.0,
    n_time_samples: int = 40,
    tolerance: float = 1e-2,
) -> OverdampedProjectionCertificate:
    r"""Verify the overdamped diffusion limit of the specified graph wave.

    Builds the canonical random-walk Laplacian L_rw, forms the damped graph
    wave q̈ + γq̇ + L q = 0, and measures two things in the strong-damping
    limit: (i) the per-mode **slow rate** converges to the diffusion rate
    λ_k/γ = ν_f·λ_k (ν_f = 1/γ), with error scaling as O(λ_max/γ²); and
    (ii) the damped-wave **trajectory** (from q₀ at rest) collapses onto the
    structural-diffusion trajectory exp(−L t/γ)·q₀.  No field formula is
    re-implemented — L_rw comes from
    :func:`structural_diffusion_operator` and the orthonormal eigenbasis
    from the symmetric normalized Laplacian.

    Parameters
    ----------
    G : TNFRGraph
    gamma : float
        Damping coefficient (effective diffusivity ν_f = 1/γ).  Must satisfy
        γ² > 4·λ_max for every mode to be overdamped.
    n_time_samples : int
        Time samples in the overdamped window for the trajectory check.
    tolerance : float
        Maximum relative error (rate and trajectory) for a valid projection.

    Returns
    -------
    OverdampedProjectionCertificate
    """
    nodes, lap = structural_diffusion_operator(G)
    n = len(nodes)
    lambdas = np.sort(np.linalg.eigvals(lap).real)
    lambdas = np.clip(lambdas, 0.0, None)
    lam_max = float(lambdas[-1]) if n else 0.0
    nonzero = lambdas[lambdas > 1e-9]
    lam2 = float(nonzero[0]) if nonzero.size else 0.0
    nu_f = 1.0 / gamma

    # (i) per-mode slow rate vs diffusion rate
    disc = gamma * gamma - 4.0 * lambdas + 0j
    s_slow = ((-gamma + np.sqrt(disc)) / 2.0).real
    diff_rate = lambdas / gamma  # = nu_f * lambda_k
    mask = lambdas > 1e-9
    if np.any(mask):
        rel = np.abs(s_slow[mask] + diff_rate[mask]) / diff_rate[mask]
        max_rate_rel = float(np.max(rel))
    else:
        max_rate_rel = 0.0
    # Fiedler (slowest) mode.  Complex-safe: an under-damped gamma (gamma^2 <
    # 4*lam2, reachable when a caller fits gamma from oscillatory data) yields
    # a complex root whose real part -gamma/2 is the envelope decay rate.
    if lam2 > 0.0:
        s_gap = (
            (-gamma + np.sqrt(gamma * gamma - 4.0 * lam2 + 0j)) / 2.0
        ).real
        slow_gap = float(-s_gap)
    else:
        slow_gap = 0.0
    diff_gap = lam2 / gamma

    # (ii) trajectory: damped wave vs diffusion in the orthonormal eigenbasis
    sym_nodes, lsym = _symmetric_normalized_laplacian(G)
    w, V = np.linalg.eigh(lsym)
    w = np.clip(w, 0.0, None)
    q0 = structural_field(G, sym_nodes)
    c0 = V.T @ q0
    disc_w = gamma * gamma - 4.0 * w + 0j
    root_w = np.sqrt(disc_w)
    ss = (-gamma + root_w) / 2.0
    sf = (-gamma - root_w) / 2.0
    denom = sf - ss
    safe = np.abs(denom) > 1e-12
    a = np.where(safe, c0 * sf / np.where(safe, denom, 1.0), c0)
    b = np.where(safe, -c0 * ss / np.where(safe, denom, 1.0), 0.0)
    horizon = 3.0 * gamma / lam2 if lam2 > 0.0 else gamma
    ts = np.linspace(0.05 * horizon, horizon, max(2, n_time_samples))
    traj_err = 0.0
    for t in ts:
        q_wave = V @ (a * np.exp(ss * t) + b * np.exp(sf * t)).real
        q_diff = V @ (c0 * np.exp(-w * t / gamma))
        denom_t = float(np.linalg.norm(q_diff)) + 1e-12
        err = float(np.linalg.norm(q_wave - q_diff)) / denom_t
        traj_err = max(traj_err, err)

    projects = (max_rate_rel < tolerance) and (traj_err < tolerance)

    return OverdampedProjectionCertificate(
        n_nodes=n,
        gamma=float(gamma),
        nu_f_effective=float(nu_f),
        spectral_gap=lam2,
        lambda_max=lam_max,
        max_rate_rel_error=max_rate_rel,
        rate_error_times_gamma_sq=float(max_rate_rel * gamma * gamma),
        slowest_slow_rate=slow_gap,
        slowest_diffusion_rate=float(diff_gap),
        trajectory_max_rel_error=traj_err,
        projects_to_diffusion=bool(projects),
    )


@dataclass(frozen=True)
class UndampedLimitCertificate:
    r"""Verification that the γ→0 limit of the damped graph wave is the
    undamped standing-wave spectrum √λ_k (the discrete modes).

    The overdamped projection (γ→∞, :func:`verify_overdamped_projection`)
    collapses the damped graph wave onto structural diffusion. Its
    opposite end, γ→0, is the **conservative** limit: the roots of
    s² + γs + λ_k = 0 become the pure-imaginary pair s = ±i√λ_k, so every
    mode oscillates undamped at the standing-wave frequency ω_k = √λ_k —
    exactly the discrete modes of :func:`verify_discrete_modes`.  γ is the
    single dial between the two regimes: γ→∞ diffusion, γ→0 standing waves.

    Attributes
    ----------
    n_nodes : int
    gamma : float
        Small damping used to probe the conservative limit.
    max_decay_rate : float
        Max |Re(s)| over the non-uniform modes (→ 0 as γ → 0; equals γ/2
        for underdamped modes — the envelope decay).
    max_freq_rel_error : float
        Max over modes of |Im(s) − √λ_k| / √λ_k: how close the damped
        oscillation frequency is to the undamped standing-wave frequency.
    freq_error_times_inv_gamma_sq : float
        ``max_freq_rel_error / γ²`` — converges to a constant, confirming
        the frequency error scales as O(γ²).
    matches_discrete_modes : bool
        Whether the γ→0 frequencies match the standing-wave spectrum √λ_k.
    standing_wave_frequencies : tuple[float, ...]
        The lowest few undamped frequencies ω_k = √λ_k (in ascending
        eigenvalue order, starting from the uniform mode ω₀ ≈ 0 — the same
        convention as :func:`verify_discrete_modes`).
    """

    n_nodes: int
    gamma: float
    max_decay_rate: float
    max_freq_rel_error: float
    freq_error_times_inv_gamma_sq: float
    matches_discrete_modes: bool
    standing_wave_frequencies: tuple[float, ...]

    @property
    def is_valid_undamped_limit(self) -> bool:
        """True when the γ→0 wave recovers the standing-wave spectrum."""
        return self.matches_discrete_modes

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_undamped_limit else "INVALID"
        return (
            f"Undamped limit [{ok}]: gamma->0 damped wave -> standing waves "
            f"s=+-i*sqrt(lambda_k); at gamma={self.gamma:.3g} decay "
            f"{self.max_decay_rate:.3e} (=gamma/2), freq error "
            f"{self.max_freq_rel_error:.2e} (/gamma^2="
            f"{self.freq_error_times_inv_gamma_sq:.3f}), "
            f"matches discrete modes={self.matches_discrete_modes}"
        )


def verify_undamped_limit(
    G: Any,
    *,
    gamma: float = 1e-3,
    tolerance: float = 1e-2,
) -> UndampedLimitCertificate:
    r"""Verify the γ→0 limit of the damped graph wave is standing waves.

    Forms the damped graph wave q̈ + γq̇ + L q = 0 at a *small* damping γ and
    measures that each underdamped mode's complex root tends to the
    pure-imaginary standing-wave value s = ±i√λ_k: the envelope decay
    Re(s) = −γ/2 → 0, and the oscillation frequency Im(s) → √λ_k (the
    discrete-mode frequency of :func:`verify_discrete_modes`).  This is the
    conservative end of the same γ-dial whose γ→∞ end is the overdamped
    projection onto structural diffusion.

    Parameters
    ----------
    G : TNFRGraph
    gamma : float
        Small damping (γ² < 4·λ₂ keeps the slow modes underdamped /
        oscillatory).
    tolerance : float
        Maximum relative frequency error for a valid undamped limit.

    Returns
    -------
    UndampedLimitCertificate
    """
    nodes, lap = structural_diffusion_operator(G)
    n = len(nodes)
    lambdas, s_slow, _ = damped_wave_rates(G, gamma)
    omega = np.sqrt(np.clip(lambdas, 0.0, None))  # standing-wave frequencies

    # complex roots of s^2 + gamma s + lambda = 0 (full, not just real part)
    disc = gamma * gamma - 4.0 * lambdas + 0j
    root = np.sqrt(disc)
    s_plus = (-gamma + root) / 2.0
    decay = np.abs(s_plus.real)  # envelope decay = gamma/2 (underdamped)
    freq = np.abs(s_plus.imag)  # oscillation frequency

    mask = lambdas > 1e-9
    if np.any(mask):
        rel = np.abs(freq[mask] - omega[mask]) / omega[mask]
        max_freq_rel = float(np.max(rel))
        max_decay = float(np.max(decay[mask]))
    else:
        max_freq_rel = 0.0
        max_decay = 0.0

    freqs = tuple(float(omega[k]) for k in range(min(6, n)))
    matches = max_freq_rel < tolerance

    return UndampedLimitCertificate(
        n_nodes=n,
        gamma=float(gamma),
        max_decay_rate=max_decay,
        max_freq_rel_error=max_freq_rel,
        freq_error_times_inv_gamma_sq=float(max_freq_rel / (gamma * gamma)),
        matches_discrete_modes=bool(matches),
        standing_wave_frequencies=freqs,
    )


# ---------------------------------------------------------------------------
# Discrete modes: the standing waves of the bounded structural manifold
# ---------------------------------------------------------------------------


def _symmetric_normalized_laplacian(G: Any) -> tuple[list, Any]:
    """Compatibility wrapper for the canonical normalized Laplacian.

    Reuse its zero-row convention for isolated nodes so the cached spectrum
    and the nodal diffusion operator have the same stationary modes.
    """
    return symmetric_normalized_laplacian(G)


def _topology_signature(G: Any) -> tuple:
    """A hashable signature of the graph topology (nodes + weighted edges).

    The diffusion Laplacian depends only on the node set and the weighted edge
    set -- never on node state (EPI, nu_f, theta). When this signature is
    unchanged the spectrum is unchanged, so the eigendecomposition can be
    reused across evolution steps and across read-outs.
    """
    conductance = read_conductance(G)
    edges = tuple(sorted(zip(conductance.source.tolist(), conductance.target.tolist(),
                             conductance.weight.tolist())))
    return (G.is_directed(), G.is_multigraph(), tuple(conductance.nodes), edges)


def _cached_eigenvalues(G: Any) -> Any:
    """Cache only the spectrum when a read-out does not require mode shapes.

    A full decomposition, when already cached, supplies the same values.
    Otherwise eigvalsh avoids computing and retaining an unused N-by-N basis.
    The full-spectrum solver and its Laplacian are still dense.
    """
    sig = _topology_signature(G)
    cache = G.graph.get("_tnfr_spectrum_cache")
    if isinstance(cache, dict) and cache.get("sig") == sig:
        return cache["vals"]
    _, lap = _symmetric_normalized_laplacian(G)
    values = np.clip(np.linalg.eigvalsh(lap), 0.0, None)
    G.graph["_tnfr_spectrum_cache"] = {"sig": sig, "vals": values}
    return values


def _cached_eigh(G: Any) -> tuple[Any, Any]:
    """Eigendecomposition of L_sym, memoized on the topology signature.

    Returns ``(eigenvalues, eigenvectors)`` ascending and clipped ``>= 0``.
    The decomposition is invariant under evolution on a fixed graph, so the
    O(N^3) ``eigh`` runs once per topology; the cache lives in ``G.graph`` and
    self-invalidates when the signature changes. ``structural_eigenmodes`` (the
    rhythm path) and ``relaxation_spectrum`` (the spectrum / NFR path) share
    this cache -- L_sym and L_rw have the same spectrum. A value-only cache
    is upgraded on the first request for actual eigenvectors.
    """
    sig = _topology_signature(G)
    cache = G.graph.get("_tnfr_spectrum_cache")
    if isinstance(cache, dict) and cache.get("sig") == sig and "vecs" in cache:
        return cache["vals"], cache["vecs"]
    _, lap = _symmetric_normalized_laplacian(G)
    eigvals, eigvecs = np.linalg.eigh(lap)
    eigvals = np.clip(eigvals, 0.0, None)
    G.graph["_tnfr_spectrum_cache"] = {
        "sig": sig,
        "vals": eigvals,
        "vecs": eigvecs,
    }
    return eigvals, eigvecs


def structural_eigenmodes(G: Any) -> tuple[Any, Any]:
    r"""Return the discrete eigenmodes of the bounded structural manifold.

    Computes the eigenvalues {λ_k} and orthonormal eigenvectors {v_k} of the
    symmetric normalized Laplacian L_sym (same spectrum as the diffusion
    operator L_rw).  The eigenvalues are the discrete mode "energies"; the
    eigenvectors are the standing-wave mode shapes (orthonormal), sorted by
    ascending λ_k. For symmetric adjacency the zero mode is proportional
    to sqrt(row strength), corresponding to uniform EPI in degree coordinates.
    Asymmetric adjacency is rejected. Returned arrays are independent copies.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (eigenvalues, eigenvectors) : tuple[np.ndarray, np.ndarray]
        ``eigenvalues`` shape ``(N,)`` ascending; ``eigenvectors`` shape
        ``(N, N)`` with column ``k`` the k-th standing-wave mode shape.
    """
    eigenvalues, eigenvectors = _cached_eigh(G)
    return eigenvalues.copy(), eigenvectors.copy()


def nodal_domain_count(mode: Any) -> int:
    r"""Number of sign changes (nodal domains − 1) of a standing-wave mode.

    The structural "mode number": the k-th standing wave has k sign changes
    on a 1D manifold (Courant's nodal-domain ordering).  Near-zero entries
    are ignored to avoid spurious sign flips.

    Parameters
    ----------
    mode : np.ndarray
        A mode shape (eigenvector).

    Returns
    -------
    int
        The number of sign changes along the node ordering.
    """
    v = np.asarray(mode, dtype=float)
    sig = np.sign(v[np.abs(v) > 1e-12])
    if sig.size < 2:
        return 0
    return int(np.sum(np.abs(np.diff(sig)) > 0))


def compute_emergent_pulse(G: Any, n_modes: int = 8) -> dict[str, Any]:
    r"""The emergent pulse: the rhythm the substrate plays [CANONICAL].

    The conservative face of the nodal dynamics is a *sustained vibration*:
    every structural mode oscillates at the standing-wave frequency
    :math:`\omega_k = \sqrt{\lambda_k}` (the discrete modes of
    :func:`verify_undamped_limit` / :func:`structural_eigenmodes`).  The rhythm
    is the interference of those resonances -- beats at the differences
    :math:`\omega_j - \omega_k` -- and the equilibria (the ``dNFR = 0``
    coherence states) are the beats the vibration passes through.  This unifies
    the scattered conservative machinery (the spectrum, the standing-wave
    frequencies, the self-similar decimation) into one *pulse* read-out,
    computed closed-form from the structural spectrum (no time integration) --
    the conservative twin of the dissipative coherence read-out.

    Parameters
    ----------
    G : TNFRGraph
    n_modes : int
        Number of leading resonant frequencies to report.

    Returns
    -------
    dict
        ``resonant_spectrum`` (leading :math:`\omega_k = \sqrt{\lambda_k}`),
        ``fundamental`` (the slowest non-uniform resonance), ``dominant_beat``
        (the slowest beat = smallest positive :math:`\omega_j - \omega_k`),
        ``spectral_multiplicity`` (the largest eigenvalue multiplicity = the
        self-similar / fractal signature), ``vibration_energy``
        (:math:`\tfrac12\sum\lambda_k`, the conserved structural-pressure
        energy of the vibration), ``n_modes``.
    """
    eigvals = _cached_eigenvalues(G)
    eigvals = np.asarray(eigvals, dtype=float)
    omega = np.sqrt(np.clip(eigvals, 0.0, None))
    nz = np.sort(omega[eigvals > 1e-9])  # exclude the uniform (lambda~0) mode
    beats = np.diff(nz) if nz.size > 1 else np.asarray([])
    pos = beats[beats > 1e-9] if beats.size else beats
    _, counts = np.unique(np.round(eigvals, 9), return_counts=True)
    return {
        "resonant_spectrum": [float(x) for x in nz[:n_modes]],
        "fundamental": float(nz[0]) if nz.size else 0.0,
        "dominant_beat": float(pos.min()) if pos.size else 0.0,
        "spectral_multiplicity": int(counts.max()) if counts.size else 0,
        "vibration_energy": float(0.5 * np.sum(eigvals)),
        "n_modes": int(nz.size),
    }


def compute_nodal_pulse(G: Any) -> dict[str, Any]:
    r"""The per-NFR pulse and its resonance [CANONICAL].

    The collective rhythm (:func:`compute_emergent_pulse`) is what the NFR
    *bricks* produce: every NFR is itself a phase oscillator -- the
    single-node reduction of the nodal equation
    :math:`\partial\mathrm{EPI}_i/\partial t=\nu_{f,i}\,\Delta\mathrm{NFR}_i`
    -- pulsing at its own structural frequency :math:`\nu_{f,i}` with phase
    :math:`\varphi_i`.  *Resonance* couples those pulses: the local
    phase-synchrony (:func:`~tnfr.metrics.coherence.local_phase_sync`)
    measures how phase-locked each NFR is with its neighbours, the global
    Kuramoto order ``R`` (:func:`~tnfr.gamma.kuramoto_R_psi`) the collective
    locking, and the U3 gate :data:`~tnfr.constants.canonical.DELTA_PHI_MAX`
    sets admissibility.  The collective pulse emerges as these per-NFR pulses
    resonate (``R -> 1``).  This is the *local* face of the rhythm -- the
    pulsing NFRs that generate the network rhythm -- read from canonical
    per-node quantities (no time integration).

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict
        ``mean_frequency`` / ``frequency_spread`` (the per-NFR pulse rates
        nu_f: mean and std in Hz_str), ``phase_coherence`` (the collective
        Kuramoto ``R`` in ``[0, 1]`` -- how locked the pulses are),
        ``mean_local_resonance`` (mean per-NFR local phase synchrony in
        ``[0, 1]``), ``resonance_gate`` (the U3 admissibility bound
        Delta phi_max), ``n_pulsing`` (NFRs with nu_f > 0), ``n_nodes``.
    """
    from ..constants.canonical import DELTA_PHI_MAX

    nodes = list(G.nodes())
    n = len(nodes)
    gate = float(DELTA_PHI_MAX)
    if not n:
        return {
            "mean_frequency": 0.0,
            "frequency_spread": 0.0,
            "phase_coherence": 0.0,
            "mean_local_resonance": 0.0,
            "resonance_gate": gate,
            "n_pulsing": 0,
            "n_nodes": 0,
        }
    vf = np.asarray(
        [float(get_attr(G.nodes[k], ALIAS_VF, 0.0) or 0.0) for k in nodes],
        dtype=float,
    )
    # collective resonance of the per-NFR pulses (Kuramoto order parameter)
    try:
        from ..gamma import kuramoto_R_psi

        phase_coherence = float(kuramoto_R_psi(G)[0])
    except Exception:
        phase_coherence = 0.0
    # per-NFR resonance: mean local phase synchrony (one shared matrix build)
    try:
        from ..metrics.coherence import (
            coherence_matrix,
            local_phase_sync_weighted,
        )

        order, W = coherence_matrix(G)
        if order is None:
            mean_local = 0.0
        else:
            mean_local = float(
                np.mean(
                    [
                        local_phase_sync_weighted(
                            G, k, nodes_order=order, W_row=W
                        )
                        for k in order
                    ]
                )
            )
    except Exception:
        mean_local = 0.0
    return {
        "mean_frequency": float(np.mean(vf)),
        "frequency_spread": float(np.std(vf)),
        "phase_coherence": phase_coherence,
        "mean_local_resonance": mean_local,
        "resonance_gate": gate,
        "n_pulsing": int(np.sum(vf > 1e-9)),
        "n_nodes": n,
    }


@dataclass(frozen=True)
class DiscreteModeCertificate:
    r"""Verification of the discrete standing-wave modes of a bounded manifold.

    A bounded structural manifold (finite graph) supports a discrete
    spectrum of orthonormal standing-wave eigenmodes — the same structure
    as the discrete harmonics of a vibrating string (Pythagoras), a Chladni
    plate, or a molecular vibrational spectrum.

    Attributes
    ----------
    n_modes : int
        Number of discrete modes (= number of nodes; finite/discrete).
    spectrum_is_discrete : bool
        The manifold has a finite, discrete eigenvalue spectrum.
    modes_orthonormal : bool
        The standing-wave mode shapes are orthonormal.
    max_orthonormality_residual : float
        Max |⟨v_i, v_j⟩ − δ_ij| (≈ 0).
    has_uniform_zero_mode : bool
        λ_1 = 0 (the uniform mode / conserved diffusion mode).
    spectral_gap : float
        λ_2 — the first non-trivial mode.
    matches_diffusion_spectrum : bool
        The L_sym spectrum equals the diffusion operator (L_rw) spectrum.
    nodal_domains_grow : bool
        The nodal-domain count grows from the lowest to the highest mode
        (Courant ordering; structural mode number).
    standing_wave_frequencies : tuple
        The first few standing-wave frequencies ω_k = √λ_k.
    """

    n_modes: int
    spectrum_is_discrete: bool
    modes_orthonormal: bool
    max_orthonormality_residual: float
    has_uniform_zero_mode: bool
    spectral_gap: float
    matches_diffusion_spectrum: bool
    nodal_domains_grow: bool
    standing_wave_frequencies: tuple

    @property
    def is_valid_discrete_modes(self) -> bool:
        """True when the manifold verifies as discrete standing waves."""
        return (
            self.spectrum_is_discrete
            and self.modes_orthonormal
            and self.has_uniform_zero_mode
            and self.matches_diffusion_spectrum
            and self.nodal_domains_grow
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_discrete_modes else "INVALID"
        freqs = ", ".join(f"{f:.3f}" for f in self.standing_wave_frequencies)
        return (
            f"Discrete standing-wave modes [{ok}]: "
            f"{self.n_modes} discrete modes, "
            f"orthonormal={self.modes_orthonormal} "
            f"(res {self.max_orthonormality_residual:.1e}), "
            f"uniform λ₁=0={self.has_uniform_zero_mode}, "
            f"spectral gap λ₂={self.spectral_gap:.4f}, "
            f"matches diffusion spectrum={self.matches_diffusion_spectrum}, "
            f"nodal domains grow={self.nodal_domains_grow}; "
            f"ω_k=√λ_k=[{freqs}]"
        )


def verify_discrete_modes(
    G: Any, *, tolerance: float = 1e-9
) -> DiscreteModeCertificate:
    r"""Verify the discrete standing-wave modes of the bounded manifold.

    Confirms that the finite manifold has a discrete spectrum of orthonormal
    standing-wave eigenmodes, with a uniform λ_1 = 0 mode, a spectrum
    matching the diffusion operator (L_rw), and nodal-domain counts growing
    with the mode index (Courant) — the structural origin of "discrete
    modes", the same as the discrete harmonics of a bounded elastic medium.

    Parameters
    ----------
    G : TNFRGraph
    tolerance : float
        Numerical tolerance for the orthonormality / spectrum checks.

    Returns
    -------
    DiscreteModeCertificate
    """
    eigvals, eigvecs = structural_eigenmodes(G)
    n = len(eigvals)

    discrete = n > 0 and np.all(np.isfinite(eigvals))

    gram = eigvecs.T @ eigvecs
    ortho_res = float(np.max(np.abs(gram - np.eye(n)))) if n else 0.0
    orthonormal = ortho_res < max(tolerance, 1e-9)

    uniform_zero = bool(abs(float(eigvals[0])) < 1e-6) if n else False
    gap = float(eigvals[1]) if n > 1 else 0.0

    # spectrum matches the random-walk diffusion operator L_rw
    _, lrw = structural_diffusion_operator(G)
    rw_spec = np.sort(np.linalg.eigvals(lrw).real)
    matches = bool(np.allclose(np.sort(eigvals), rw_spec, atol=1e-7))

    # nodal-domain counts grow from lowest to highest mode (Courant)
    counts = [nodal_domain_count(eigvecs[:, k]) for k in range(n)]
    grow = (counts[0] == 0 and counts[-1] > counts[0]) if n > 1 else True

    freqs = tuple(float(np.sqrt(eigvals[k])) for k in range(min(6, n)))

    return DiscreteModeCertificate(
        n_modes=n,
        spectrum_is_discrete=discrete,
        modes_orthonormal=orthonormal,
        max_orthonormality_residual=ortho_res,
        has_uniform_zero_mode=uniform_zero,
        spectral_gap=gap,
        matches_diffusion_spectrum=matches,
        nodal_domains_grow=grow,
        standing_wave_frequencies=freqs,
    )


# ---------------------------------------------------------------------------
# Structural stability: the dispersion relation and the instability threshold
# ---------------------------------------------------------------------------


def dispersion_relation(G: Any, reaction_rate: float = 0.0) -> Any:
    r"""Real growth rates r − Re(eigenvalues(diag(νf)·L_rw)).

    The growth (σ_k > 0) or decay (σ_k < 0) rate of each structural
    eigenmode under diffusion plus a local reaction rate.  At
    ``reaction_rate = 0`` this is the negative of the relaxation spectrum
    (pure diffusion: nonnegative damping, possibly multiple frozen modes).

    Parameters
    ----------
    G : TNFRGraph
    reaction_rate : float, optional
        A local growth/decay rate r added to every mode (the operators
        supply it: stabilizers lower r, destabilizers raise it).

    Returns
    -------
    np.ndarray
        Growth rates sorted by ascending nodal decay rate.
    """
    return float(reaction_rate) - relaxation_spectrum(G)


def instability_threshold(G: Any) -> float:
    r"""Second nodal decay rate, the first nonuniform reaction threshold.

    On a connected symmetric graph with common positive frequency this is
    νf·λ_2 and its geometry mode is Fiedler. Heterogeneous capacity uses the
    actual nodal generator. Multiple zero modes give a zero threshold.
    For any positive r the constant mode already grows, so this is not
    a global boundedness certificate or an operator-sequence U2 test.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    float
        Second-smallest real decay rate (0 with fewer than two modes).
    """
    rates = relaxation_spectrum(G)
    return float(rates[1]) if len(rates) > 1 else 0.0


def fiedler_partition(G: Any) -> tuple[list, list]:
    r"""The Fiedler-mode 2-partition — the first structural pattern.

    Splits the nodes by the sign of the Fiedler eigenvector (the mode with
    the smallest non-zero λ).  This is the network's **weakest structural
    cut** (the two most weakly-connected communities) — the empirically-
    validated spectral-clustering partition, and the first pattern to grow
    once the reaction rate crosses the instability threshold.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (part_a, part_b) : tuple[list, list]
        Node lists for the two structural communities.
    """
    nodes = _ordered_nodes(G)
    eigvals, eigvecs = structural_eigenmodes(G)
    if eigvecs.shape[1] < 2:
        return list(nodes), []
    fiedler = eigvecs[:, 1]
    part_a = [nodes[i] for i in range(len(nodes)) if fiedler[i] > 0]
    part_b = [nodes[i] for i in range(len(nodes)) if fiedler[i] <= 0]
    return part_a, part_b


@dataclass(frozen=True)
class StructuralStabilityCertificate:
    r"""Verification of the structural linear-stability / dispersion relation.

    The dispersion relation σ_k = r − νf·λ_k governs the growth/decay of
    every structural eigenmode.  Pure diffusion (r = 0) decays every
    non-uniform mode (stable equilibrium); the threshold r_c = νf·λ_2
    separates uniform amplification from structural pattern formation (the
    Fiedler partition). This certificate requires a connected symmetric
    graph and a common positive capacity. It does not certify grammar U2.

    Attributes
    ----------
    n_nodes : int
    diffusivity : float
        νf (the diffusion coefficient).
    spectral_gap : float
        λ_2 (the Fiedler value).
    instability_threshold : float
        r_c = νf·λ_2.
    pure_diffusion_stable : bool
        At r = 0 every non-uniform mode decays (σ_k < 0 for k ≥ 1).
    max_nonuniform_growth_at_zero : float
        Max σ_k over k ≥ 1 at r = 0 (should be < 0).
    dispersion_matches_relaxation : bool
        The r = 0 dispersion equals the negative relaxation spectrum.
    first_unstable_mode : int
        The first non-uniform mode to go unstable above threshold (= 1,
        the Fiedler mode).
    fiedler_partition_sizes : tuple
        Sizes (|A|, |B|) of the Fiedler 2-partition (the first pattern).
    """

    n_nodes: int
    diffusivity: float
    spectral_gap: float
    instability_threshold: float
    pure_diffusion_stable: bool
    max_nonuniform_growth_at_zero: float
    dispersion_matches_relaxation: bool
    first_unstable_mode: int
    fiedler_partition_sizes: tuple

    @property
    def is_valid_stability(self) -> bool:
        """True when the structural-stability picture verifies."""
        return (
            self.pure_diffusion_stable
            and self.dispersion_matches_relaxation
            and self.first_unstable_mode == 1
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_stability else "INVALID"
        a, b = self.fiedler_partition_sizes
        return (
            f"Structural stability [{ok}]: "
            f"νf={self.diffusivity:.4f}, spectral gap λ₂="
            f"{self.spectral_gap:.4f}, "
            f"instability threshold r_c=νf·λ₂="
            f"{self.instability_threshold:.4f}, "
            f"pure diffusion stable={self.pure_diffusion_stable} "
            f"(max non-uniform growth "
            f"{self.max_nonuniform_growth_at_zero:.1e}), "
            f"first unstable mode k={self.first_unstable_mode} (Fiedler), "
            f"Fiedler partition {a}/{b}"
        )


def verify_structural_stability(
    G: Any, *, tolerance: float = 1e-9
) -> StructuralStabilityCertificate:
    r"""Verify the structural linear-stability / dispersion relation.

    Confirms that pure diffusion decays every non-uniform mode (stable
    equilibrium), that the r = 0 dispersion equals the negative relaxation
    spectrum, that the instability threshold is r_c = νf·λ_2, and that the
    first structural mode to go unstable above threshold is the Fiedler
    mode. This homogeneous Fiedler certificate requires symmetric adjacency,
    a connected positive-weight support, at least two nodes, and one common
    positive frequency. For heterogeneous/directed damping use
    :func:`relaxation_spectrum` and :func:`dispersion_relation`.

    Parameters
    ----------
    G : TNFRGraph
    tolerance : float
        Numerical tolerance.

    Returns
    -------
    StructuralStabilityCertificate
    """
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    frequency = _nodal_frequencies(G)
    if len(frequency) < 2 or not np.all(frequency == frequency[0]) or frequency[0] <= 0.0:
        raise ValueError("Fiedler stability requires at least two nodes and a common positive frequency")
    eigvals = _cached_eigenvalues(G)
    n = len(eigvals)
    nu = float(frequency[0])
    gap = float(eigvals[1])
    if gap <= tolerance:
        raise ValueError("Fiedler stability requires connected positive-weight support")
    r_c = nu * gap

    # pure diffusion (r=0): non-uniform modes (k>=1) decay
    sigma0 = dispersion_relation(G, 0.0)
    max_nonuniform = float(np.max(sigma0[1:])) if n > 1 else 0.0
    stable = max_nonuniform < tolerance

    # Independently check the modal readout against the actual nodal matrix.
    _, lap = structural_diffusion_operator(G)
    rates = np.linalg.eigvals(frequency[:, None] * lap).real
    matches = bool(np.allclose(np.sort(-sigma0), np.sort(rates), atol=1e-7))

    # first non-uniform mode to go unstable just above threshold
    sigma = dispersion_relation(G, r_c + max(1e-3, 1e-3 * r_c))
    if n > 1 and np.any(sigma[1:] > tolerance):
        first_unstable = int(np.argmax(sigma[1:] > tolerance)) + 1
    else:
        first_unstable = 0

    # Fiedler partition sizes
    part_a, part_b = fiedler_partition(G)
    sizes = (len(part_a), len(part_b))

    return StructuralStabilityCertificate(
        n_nodes=n,
        diffusivity=nu,
        spectral_gap=gap,
        instability_threshold=r_c,
        pure_diffusion_stable=stable,
        max_nonuniform_growth_at_zero=max_nonuniform,
        dispersion_matches_relaxation=matches,
        first_unstable_mode=first_unstable,
        fiedler_partition_sizes=sizes,
    )


# ---------------------------------------------------------------------------
# The structural random walk: Brownian motion and resistance geometry
# ---------------------------------------------------------------------------


def _adjacency_degree(G: Any) -> tuple[list, Any, Any]:
    """Return (nodes, weighted adjacency W, weighted degree vector)."""
    conductance = read_conductance(G)
    return conductance.nodes, conductance.dense(), conductance.strength


def _symmetric_adjacency_degree(G: Any) -> tuple[list, Any, Any]:
    """Require the reversible conductance model used by resistance geometry."""
    conductance = read_conductance(G, symmetric=True)
    return conductance.nodes, conductance.dense(), conductance.strength


def random_walk_matrix(G: Any) -> tuple[list, Any]:
    r"""Return the random-walk transition matrix P = I - L_rw.

    The diffusion operator is L_rw = I − P, so P is exactly the
    random-walk the structural diffusion generates.  Row-stochastic (each
    row sums to 1); nodes with zero outgoing strength are absorbing. This
    includes isolated nodes, directed sinks and nodes with only zero-weight
    edges. Parallel edge conductances are summed by the shared adjacency.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, P) : tuple[list, np.ndarray]
    """
    conductance = read_conductance(G)
    return conductance.nodes, conductance.transition()


def stationary_distribution(G: Any) -> tuple[list, Any]:
    r"""A stationary distribution of a walk with symmetric conductances.

    Uses row strengths ``π_i = d_i / Σ d`` when any conductance is present,
    and the uniform distribution when every node is absorbing. The empty
    graph returns an empty vector. This is a stationary measure, not a claim
    of uniqueness or convergence on disconnected or periodic walks.

    Asymmetric adjacency is rejected: its stationary measure is generally
    not the degree distribution. The separate directed-diffusion stationary
    solver supports the strictly positive measure on strongly connected
    directed networks. Self-loop conductance is counted once, as in P.
    Scaled strengths avoid overflowing row sums and total volume. Stationary
    probabilities below the floating-point range round to zero.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, pi) : tuple[list, np.ndarray]
    """
    conductance = read_conductance(G, symmetric=True)
    nodes = conductance.nodes
    scaled = conductance.relative_strength()
    if np.any(scaled > 0.0):
        with np.errstate(under="ignore"):
            pi = scaled / scaled.sum()
    else:
        pi = np.full(len(nodes), 1.0 / len(nodes)) if nodes else scaled
    return nodes, pi


def _resistance_geometry(G: Any) -> tuple[list, Any, Any, list]:
    """Return dimensionless resistance, component scales and exact-capable volumes.

    Work component by component so unreachable pairs stay infinite and
    the pseudoinverse's numerical rank is scaled locally. Connectivity is
    determined by positive conductance, including summed parallel edges.
    Non-loop weights set the scale: loops change holding time, not resistance.
    A volume too large for float is retained as a Fraction internally; its
    ratio with the conductance scale can still give a finite commute time.
    """
    import networkx as nx
    import math
    from fractions import Fraction

    conductance = read_conductance(G, symmetric=True)
    nodes, adjacency = conductance.nodes, conductance.dense()
    resistance = np.full(adjacency.shape, np.inf)
    scales = np.ones(len(nodes))
    volumes = [0.0] * len(nodes)
    for component in nx.connected_components(nx.from_numpy_array(adjacency)):
        indices = np.asarray(sorted(component), dtype=int)
        block = np.ix_(indices, indices)
        weights = adjacency[block]
        positive_weights = weights[weights > 0.0]
        try:
            volume = math.fsum(positive_weights)
        except OverflowError:
            volume = sum((Fraction.from_float(float(weight)) for weight in positive_weights),
                         Fraction())
        for index in indices:
            volumes[index] = volume
        weights = weights.copy()
        np.fill_diagonal(weights, 0.0)
        scale = float(np.max(weights, initial=0.0)) or 1.0
        positive = weights > 0.0
        with np.errstate(under="ignore"):
            weights /= scale
        if np.any(positive & (weights == 0.0)):
            raise ValueError("Resistance conductance ratios are below floating-point range")
        scales[indices] = scale
        laplacian = np.diag(weights.sum(axis=1)) - weights
        inverse = np.linalg.pinv(laplacian, hermitian=True)
        diagonal = np.diag(inverse)
        resistance[block] = np.maximum(
            diagonal[:, None] + diagonal[None, :] - 2.0 * inverse, 0.0,
        )
    np.fill_diagonal(resistance, 0.0)
    return nodes, resistance, scales, volumes


def _resistance_scale_ratios(numerators: Any, denominators: Any) -> list:
    """Retain an exact exceptional ratio until the requested matrix is formed."""
    from fractions import Fraction

    result = []
    for numerator, denominator in zip(numerators, denominators):
        if isinstance(numerator, Fraction):
            result.append(numerator / Fraction.from_float(float(denominator)))
        else:
            with np.errstate(over="ignore", under="ignore"):
                ratio = float(numerator) / float(denominator)
            if np.isfinite(ratio) and (ratio != 0.0 or numerator == 0.0):
                result.append(ratio)
            else:
                result.append(Fraction.from_float(float(numerator))
                              / Fraction.from_float(float(denominator)))
    return result


def _commute_from_resistance(resistance: Any, volumes: Any) -> Any:
    """Apply finite or exact scale factors; only unreachable pairs stay infinite."""
    finite = np.isfinite(resistance)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            factors = np.asarray(volumes, dtype=float)
            result = np.multiply(
                factors[:, None], resistance,
                out=np.full_like(resistance, np.inf), where=finite,
            )
        if not np.all(np.isfinite(result[finite])):
            raise OverflowError
    except (FloatingPointError, OverflowError):
        from fractions import Fraction

        result = np.full_like(resistance, np.inf)
        for row, factor in enumerate(volumes):
            exact_factor = factor if isinstance(factor, Fraction) else Fraction.from_float(float(factor))
            for column in np.flatnonzero(finite[row]):
                exact = Fraction.from_float(float(resistance[row, column])) * exact_factor
                try:
                    result[row, column] = float(exact)
                except OverflowError as exc:
                    raise ValueError("Resistance read-out exceeds finite floating-point range") from exc
    if np.any(finite & (resistance > 0.0) & (result == 0.0)):
        raise ValueError("Resistance read-out is below floating-point range")
    return result


def effective_resistance(G: Any) -> tuple[list, Any]:
    r"""Effective-resistance matrix R_eff(i,j) (Ohm's law).

    Treating the network as a resistor network (the combinatorial
    Laplacian L = D − W is the conductance matrix — Kirchhoff 1847), the
    effective resistance between nodes is

        R_eff(i,j) = L⁺_ii + L⁺_jj − 2·L⁺_ij,

    with L⁺ the Moore–Penrose pseudoinverse within the pair's connected
    component. R_eff is an extended metric: different positive-conductance
    components have infinite resistance, including distinct isolated nodes.
    Self-resistance is zero. The conductance matrix must be symmetric;
    directed asymmetric resistance is outside this formula's scope.
    Component scaling avoids raw degree/volume overflow and removes loops
    before forming the Laplacian. Unrepresentable requested finite entries
    raise ValueError; infinity is reserved for unreachable pairs. This remains
    a floating-point pseudoinverse, with its numerical rank-resolution limit.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, R) : tuple[list, np.ndarray]
        ``R[i, j]`` is the effective resistance between node i and node j.
    """
    nodes, resistance, scales, _ = _resistance_geometry(G)
    return nodes, _commute_from_resistance(
        resistance, _resistance_scale_ratios(np.ones(len(nodes)), scales),
    )


def commute_time(G: Any) -> tuple[list, Any]:
    r"""Commute-time matrix C(i,j) = volume(component)·R_eff(i,j).

    The expected round-trip time of the structural random walk between two
    reachable nodes equals their component's total row strength times the
    effective resistance. For a connected, unweighted, loopless graph the
    volume is 2m. Self-loop conductances contribute once to row strengths
    and account for holding time. Unreachable pairs have infinite commute
    time; self-commute time is zero. Symmetric conductance is required.
    A common conductance scale cancels before raw resistance or volume is
    materialized, so their separate overflow need not prevent a finite result.
    A requested reachable commute time outside float range raises ValueError.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, C) : tuple[list, np.ndarray]
    """
    nodes, resistance, scales, volumes = _resistance_geometry(G)
    return nodes, _commute_from_resistance(
        resistance, _resistance_scale_ratios(volumes, scales),
    )


@dataclass(frozen=True)
class RandomWalkCertificate:
    r"""Verification of the structural random walk and resistance geometry.

    The diffusion operator is the generator of a random walk (Brownian
    motion on the network). For symmetric conductance its stationary
    measure follows row strength, and resistance / commute time describe
    transport within components. Disconnected pairs have infinite distance.
    This certificate retains its historical field names; ``2m`` means the
    pair's component volume, and an edgeless graph uses uniform stationarity.

    Attributes
    ----------
    n_nodes : int
    operator_is_walk_generator : bool
        L_rw = I − P exactly (the diffusion operator generates the walk).
    transition_row_stochastic : bool
        Every row of P sums to one, including absorbing zero-strength rows.
    stationary_is_degree : bool
        The normalized degree measure (uniform if edgeless) obeys π·P = π.
    resistance_is_metric : bool
        R_eff is symmetric, non-negative, and obeys the triangle
        inequality, allowing infinity between different components.
    max_resistance : float
        The largest pairwise effective resistance (transport diameter).
    commute_equals_2m_resistance : bool
        C(i,j) = component volume·R_eff(i,j) for reachable pairs; both
        quantities are infinite for unreachable pairs.
    max_walk_generator_residual : float
        Max |L_rw − (I − P)| (≈ 0).
    """

    n_nodes: int
    operator_is_walk_generator: bool
    transition_row_stochastic: bool
    stationary_is_degree: bool
    resistance_is_metric: bool
    max_resistance: float
    commute_equals_2m_resistance: bool
    max_walk_generator_residual: float

    @property
    def is_valid_random_walk(self) -> bool:
        """True when the structural random-walk picture verifies."""
        return (
            self.operator_is_walk_generator
            and self.transition_row_stochastic
            and self.stationary_is_degree
            and self.resistance_is_metric
            and self.commute_equals_2m_resistance
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_random_walk else "INVALID"
        return (
            f"Structural random walk [{ok}]: "
            f"L_rw = I−P={self.operator_is_walk_generator} "
            f"(res {self.max_walk_generator_residual:.1e}), "
            f"P row-stochastic={self.transition_row_stochastic}, "
            f"stationary measure={self.stationary_is_degree}, "
            f"R_eff extended metric={self.resistance_is_metric} "
            f"(max R {self.max_resistance:.4f}), "
            f"commute=component volume·R_eff={self.commute_equals_2m_resistance}"
        )


def verify_structural_random_walk(
    G: Any, *, tolerance: float = 1e-9
) -> RandomWalkCertificate:
    r"""Verify the structural random walk and resistance geometry.

    Confirms that the diffusion operator is the random-walk generator
    (L_rw = I − P), that P is row-stochastic with stationary distribution
    proportional to row strength (uniform if edgeless), that effective
    resistance is an extended transport metric, and that commute time uses
    each component's volume. Requires symmetric conductance. Stationarity
    does not assert convergence or uniqueness; triangle checks are sampled.

    Parameters
    ----------
    G : TNFRGraph
    tolerance : float
        Numerical tolerance.

    Returns
    -------
    RandomWalkCertificate
    """
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    nodes, scaled_resistance, scales, volumes = _resistance_geometry(G)
    n = len(nodes)
    r = _commute_from_resistance(
        scaled_resistance, _resistance_scale_ratios(np.ones(n), scales),
    )
    _, lrw = structural_diffusion_operator(G)
    _, p = random_walk_matrix(G)

    # L_rw = I − P
    gen_res = float(np.max(np.abs(lrw - (np.eye(n) - p)))) if n else 0.0
    is_generator = gen_res < max(tolerance, 1e-12)

    # Every row is stochastic, including absorbing zero-strength rows.
    row_sums = p.sum(axis=1)
    row_stochastic = bool(np.all(np.abs(row_sums - 1.0) < tolerance))

    # stationary distribution π = degree, π·P = π
    _, pi = stationary_distribution(G)
    stationary_ok = bool(
        np.allclose(pi @ p, pi, atol=tolerance, rtol=0.0)
        and np.all(pi >= 0.0)
        and abs(float(pi.sum()) - 1.0) < tolerance
    ) if n else True

    # effective resistance is a metric
    symmetric = bool(np.allclose(r, r.T))
    nonneg = bool(np.all(r >= -1e-9))
    # triangle inequality on a sample of triples
    triangle = True
    if n >= 3:
        rng = np.random.default_rng(0)
        for _ in range(200):
            i, j, k = rng.integers(0, n, size=3)
            if r[i, k] > r[i, j] + r[j, k] + 1e-7:
                triangle = False
                break
    is_metric = symmetric and nonneg and triangle
    max_r = float(np.max(r)) if n else 0.0

    # The volume belongs to the pair's component, never to unreachable nodes.
    _, c = commute_time(G)
    commute_ok = bool(np.allclose(
        c, _commute_from_resistance(r, volumes), atol=tolerance, rtol=0.0,
    ))

    return RandomWalkCertificate(
        n_nodes=n,
        operator_is_walk_generator=is_generator,
        transition_row_stochastic=row_stochastic,
        stationary_is_degree=stationary_ok,
        resistance_is_metric=is_metric,
        max_resistance=max_r,
        commute_equals_2m_resistance=commute_ok,
        max_walk_generator_residual=gen_res,
    )


# ---------------------------------------------------------------------------
# The structural flow: current, Kirchhoff's law, and continuity
# ---------------------------------------------------------------------------
def structural_current(G: Any) -> tuple[list, Any]:
    r"""Structural edge-current matrix J_ij = W_ij (EPI_i − EPI_j).

    The transport carries a current: along each edge i∼j the diffusion
    flux is J_ij = W_ij (EPI_i − EPI_j) (flux from high to low
    concentration).  The matrix is **antisymmetric** (J_ij = −J_ji) and
    supported on positive-conductance edges only. Symmetric conductance is
    required; parallel edges add their conductances and self-loops carry
    zero current. Asymmetric directed transport is outside this contract.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, J) : tuple[list, np.ndarray]
        ``J[i, j]`` is the current from node i to node j across edge i∼j.
    """
    conductance, _, flux = _read_edge_flux(G)
    return conductance.nodes, conductance.dense(flux)


def current_divergence(G: Any) -> tuple[list, Any]:
    r"""Net current outflow per node div(J)(i) = Σ_{j∼i} J_ij = (L·EPI)(i).

    Kirchhoff's current law: the net outflow at a node equals the
    combinatorial Laplacian L = D − W acting on EPI.  This is the discrete
    continuity equation div(J) = L·EPI. At positive row strength d_i and
    frequency νf_i, the nodal equation gives
    ``(d_i / νf_i) ∂EPI_i/∂t + div(J)_i = 0``. This constitutive flux
    does not itself include mobility: a frozen node can carry nonzero
    pressure/current while its nodal derivative vanishes.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, div) : tuple[list, np.ndarray]
        ``div[i]`` is the net current leaving node i.
    """
    conductance, _, flux = _read_edge_flux(G)
    return conductance.nodes, conductance.divergence(flux)


@dataclass(frozen=True)
class StructuralFlowCertificate:
    r"""Verification of the structural flow (current, Kirchhoff, Ohm).

    Symmetric diffusion transport carries the current
    J_ij = W_ij (EPI_i − EPI_j). Its node balance is Kirchhoff's current law —
    the discrete continuity equation div(J) = L·EPI — and under an
    injected current the potential drop is the effective resistance
    (Ohm's law). The latter checks sampled reachable pairs and the injected
    current equation, not injections between disconnected components.

    Attributes
    ----------
    n_nodes : int
    current_antisymmetric : bool
        J_ij = −J_ji (the current is a directed edge flow).
    kirchhoff_holds : bool
        Net outflow Σ_j J_ij = (L·EPI)_i (current law = continuity).
    total_flux_balances : bool
        Σ_i div(J)_i = 0 (closed network, no sources/sinks).
    equilibrium_zero_current : bool
        A uniform EPI field produces zero current everywhere.
    ohm_law_holds : bool
        An injected unit current s→t induces a potential drop equal to
        R_eff(s,t), with L·V equal to the injected source/sink vector,
        for sampled reachable pairs.
    max_kirchhoff_residual : float
        Max |Σ_j J_ij − (L·EPI)_i| (≈ 0).
    """

    n_nodes: int
    current_antisymmetric: bool
    kirchhoff_holds: bool
    total_flux_balances: bool
    equilibrium_zero_current: bool
    ohm_law_holds: bool
    max_kirchhoff_residual: float

    @property
    def is_valid_flow(self) -> bool:
        """True when the structural-flow picture verifies."""
        return (
            self.current_antisymmetric
            and self.kirchhoff_holds
            and self.total_flux_balances
            and self.equilibrium_zero_current
            and self.ohm_law_holds
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_flow else "INVALID"
        return (
            f"Structural flow [{ok}]: "
            f"current antisymmetric={self.current_antisymmetric}, "
            f"Kirchhoff div(J)=L·EPI={self.kirchhoff_holds} "
            f"(res {self.max_kirchhoff_residual:.1e}), "
            f"total flux balances={self.total_flux_balances}, "
            f"equilibrium zero current={self.equilibrium_zero_current}, "
            f"Ohm drop=R_eff={self.ohm_law_holds}"
        )


def verify_structural_flow(
    G: Any, *, tolerance: float = 1e-9
) -> StructuralFlowCertificate:
    r"""Verify the structural flow: current, Kirchhoff's law, Ohm's law.

    Confirms that the diffusion edge current J_ij = W_ij (EPI_i − EPI_j) is
    antisymmetric, that Kirchhoff's current law div(J) = L·EPI holds (the
    discrete continuity equation), that the total flux balances on a closed
    network, that a uniform EPI field carries zero current, and that an
    injected unit current induces a potential drop equal to the effective
    resistance (Ohm's law). Conductance must be symmetric. Up to 20 distinct
    reachable pairs are sampled reproducibly; their voltage fields must
    satisfy L·V = b as well as the resistance drop. When no distinct pair
    is reachable, this sampled Ohm check is vacuous.

    Parameters
    ----------
    G : TNFRGraph
    tolerance : float
        Numerical tolerance.

    Returns
    -------
    StructuralFlowCertificate
    """
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    nodes, j = structural_current(G)
    n = len(nodes)

    # current antisymmetry J_ij = −J_ji
    antisym = bool(np.allclose(j, -j.T, atol=tolerance)) if n else True

    # Kirchhoff: net outflow = (L·EPI) with L the combinatorial Laplacian
    _, adj, deg = _symmetric_adjacency_degree(G)
    lap = np.diag(deg) - adj
    epi = structural_field(G, nodes)
    net_out = j.sum(axis=1)
    kirchhoff_res = float(np.max(np.abs(net_out - lap @ epi))) if n else 0.0
    kirchhoff_ok = kirchhoff_res < max(tolerance, 1e-9)

    # total flux balances (L has zero column sums) — closed network
    total_balances = (
        bool(abs(float(net_out.sum())) < max(tolerance, 1e-9)) if n else True
    )

    # equilibrium: a uniform EPI field carries zero current
    uniform = np.ones(n)
    j_uniform = (uniform[:, None] - uniform[None, :]) * adj
    equilibrium_ok = bool(np.allclose(j_uniform, 0.0, atol=tolerance)) if n else True

    # Ohm's law: injected unit current s→t induces drop V_s − V_t = R_eff
    ohm_ok = True
    if n >= 2:
        _, resistance = effective_resistance(G)
        pairs = np.argwhere(np.triu(np.isfinite(resistance), k=1))
        rng = np.random.default_rng(0)
        if len(pairs) > 20:
            pairs = pairs[rng.choice(len(pairs), size=20, replace=False)]
        inverses = {}
        for s, t in pairs:
            component = tuple(np.flatnonzero(np.isfinite(resistance[s])))
            indices = np.asarray(component, dtype=int)
            if component not in inverses:
                inverses[component] = np.linalg.pinv(
                    lap[np.ix_(indices, indices)], hermitian=True,
                )
            b = np.zeros(n)
            b[s], b[t] = 1.0, -1.0
            v = np.zeros(n)
            v[indices] = inverses[component] @ b[indices]
            drop = v[s] - v[t]
            if not (
                np.allclose(lap @ v, b, atol=tolerance, rtol=0.0)
                and np.isclose(drop, resistance[s, t], atol=tolerance, rtol=tolerance)
            ):
                ohm_ok = False
                break

    return StructuralFlowCertificate(
        n_nodes=n,
        current_antisymmetric=antisym,
        kirchhoff_holds=kirchhoff_ok,
        total_flux_balances=total_balances,
        equilibrium_zero_current=equilibrium_ok,
        ohm_law_holds=ohm_ok,
        max_kirchhoff_residual=kirchhoff_res,
    )
