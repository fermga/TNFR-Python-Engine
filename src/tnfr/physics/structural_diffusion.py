r"""TNFR Structural Diffusion — the transport content of the nodal equation.

This module makes explicit, and verifies, that the TNFR nodal equation

    ∂EPI/∂t = νf · ΔNFR(t)

is **structurally a diffusion equation on the network**.  This is not an
analogy imported from another paradigm: it is the literal content of the
canonical ΔNFR computation.

THE NODAL EQUATION IS GRAPH DIFFUSION
=====================================
The canonical ΔNFR (:func:`tnfr.dynamics.default_compute_delta_nfr`) is a
weighted sum of *neighbour-mean-minus-self* gradients, one per structural
channel (see :mod:`tnfr.dynamics.dnfr`):

    g_epi(i)   = mean_{j∈N(i)} EPI(j) − EPI(i)
    g_phase(i) = −angle_diff(θ(i), mean θ neighbours) / π
    g_vf(i)    = mean νf(neighbours) − νf(i)
    g_topo(i)  = mean deg(neighbours) − deg(i)

Each ``neighbour-mean − self`` term is exactly the action of the
**random-walk graph Laplacian** L_rw = I − D⁻¹W on that field:

    g_epi = −(L_rw · EPI)   (verified to machine precision).

So the EPI channel of the nodal equation is

    ∂EPI/∂t = νf · ΔNFR_epi = −νf · L_rw · EPI,

i.e. the **discrete diffusion (heat) equation** with diffusivity νf.  The
structural form EPI spreads across the network exactly as heat or a
concentration diffuses; ΔNFR is the diffusive gradient (the structural
pressure) driving the flux, and νf is the mobility / diffusivity.

WHAT EMERGES (empirically-grounded, in TNFR's own terms)
========================================================
- **Structural diffusion** (EPI channel): the form relaxes to a uniform
  field; each Laplacian eigenmode decays as exp(−νf·λ_k·t); the slowest
  rate is set by the spectral gap λ₂ (the Fiedler value).
- **Conserved structural total**: the random-walk Laplacian conserves the
  **degree-weighted total** Σ_i deg(i)·EPI(i) (its left null vector is the
  degree vector), the analogue of the conserved amount of diffusing
  substance.
- **Equilibrium ⟺ no gradients**: ΔNFR = 0 ⟺ the field is uniform across
  neighbourhoods — the diffusive steady state.
- **Synchronization** (phase channel): the phase term aligns θ to the
  neighbour mean, driving Kuramoto-type synchronization (R → 1).

These are the registers whose existence is established by the strictest
empirical method — diffusion (Fourier 1822, Fick 1855, Einstein 1905) and
synchronization (Kuramoto; observed in fireflies, pacemaker cells, neurons,
Josephson junctions).  They are reproduced here as the **same mathematics**
(the graph Laplacian is the discrete diffusion operator), not as a
metaphor.

THE MECHANICAL REGIME IS OVERDAMPED DRIFT (not inertial)
========================================================
Because the nodal equation is **first order in time**, the mechanical
regime it produces directly is the **overdamped drift law**, not Newtonian
inertia.  Reading EPI as a position-like coordinate q and ΔNFR as the
structural pressure F, the nodal equation is

    q̇ = νf · F,

i.e. **velocity proportional to applied force**, with νf the **mobility**.
Under a sustained structural pressure the field drifts at *constant*
velocity (linear in time), it does **not** accelerate.  This is the
empirically-demonstrated mobility / drift law — Stokes drag (1851),
Einstein's mobility relation (1905), terminal velocity, sedimentation,
electrophoresis — where νf is the mobility, NOT an inverse inertial mass.

The **inertial** Newtonian regime (second order, q̈ = F/m, oscillation)
is a *different* structure: it lives in the conservative **symplectic
substrate** Hamiltonian flow (:mod:`tnfr.physics.symplectic_substrate`,
where the flow is q̈ = −q per conjugate pair).  The bare nodal equation is
the **overdamped projection** of that substrate flow.  So:

    bare nodal equation (1st order)  →  overdamped drift  v = νf·F
    symplectic substrate (2nd order) →  inertial oscillation  q̈ = −∂V/∂q

both empirically grounded, but distinct regimes — a single first-order
nodal equation cannot, by itself, be Newton's second law.

DISCRETE MODES ARE THE BOUNDED-MANIFOLD STANDING WAVES
======================================================
On a **bounded** structural manifold (a finite graph) the diffusion
operator has a **discrete** spectrum of eigenmodes — the same structure as
the discrete harmonics of a bounded vibrating medium.  The symmetric
normalized Laplacian L_sym = I − D^{-1/2} W D^{-1/2} shares the diffusion
operator L_rw's spectrum {λ_k} but has **orthonormal** eigenvectors v_k:

- **Discrete spectrum**: a finite manifold supports a finite, discrete set
  of eigenvalues {λ_k} (not a continuum) — the structural origin of
  "discrete modes".  λ_1 = 0 is the uniform mode (the conserved diffusion
  mode); λ_2 (the spectral gap) is the first non-trivial mode.
- **Standing-wave shapes**: the eigenvectors v_k are orthonormal standing
  waves.  On a path graph they are exactly the cosine standing waves of a
  vibrating string (overlap 1.0 to machine precision).
- **Nodal-domain ordering** (Courant): the number of sign changes (nodal
  domains) grows with the mode index k — the structural "mode number" k
  emerges from the bounded geometry, not from a postulate.
- **Two time-regimes, same modes**: under diffusion (first order) mode k
  relaxes as exp(−νf·λ_k·t); under the wave/substrate flow (second order)
  it oscillates at the standing-wave frequency ω_k = √λ_k.

This is the discrete-harmonic structure of a bounded elastic medium —
vibrating strings (Pythagoras), Chladni plate modes (1787), molecular
vibrational spectra — all established by the strictest empirical method.
The discreteness is a consequence of the **bounded structural geometry**,
not an imported quantum postulate.

HONEST SCOPE
============
- The identity ΔNFR_epi = −L_rw·EPI is EXACT (machine precision), a
  mathematical fact about the canonical ΔNFR.
- The full ΔNFR is multi-channel: EPI **diffusion** + phase
  **synchronization** + νf/topology **homogenization**.  This module
  isolates and certifies the diffusion (EPI) channel and reports the
  synchronization channel qualitatively.
- This characterises the transport content of the nodal dynamics; it does
  not, by itself, resolve any open program (Riemann G4, Navier–Stokes).

References
----------
- :mod:`tnfr.dynamics.dnfr` — the canonical ΔNFR neighbour-mean gradients
- :func:`tnfr.observers.kuramoto_order` — the synchronization order R
- :mod:`tnfr.physics.conservation` — the structural continuity equation
- AGENTS.md §"Foundational Physics" — the nodal equation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..mathematics.unified_numerical import np
from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR

__all__ = [
    "StructuralDiffusionCertificate",
    "OverdampedRegimeCertificate",
    "DiscreteModeCertificate",
    "structural_diffusion_operator",
    "structural_field",
    "structural_diffusivity",
    "relaxation_spectrum",
    "degree_weighted_total",
    "structural_eigenmodes",
    "nodal_domain_count",
    "verify_structural_diffusion",
    "verify_overdamped_regime",
    "verify_discrete_modes",
]


def _ordered_nodes(G: Any) -> list:
    """Stable node ordering for the matrix representation."""
    return list(G.nodes())


def structural_diffusion_operator(G: Any) -> tuple[list, Any]:
    r"""Return the random-walk graph Laplacian L_rw = I − D⁻¹W.

    This is the operator whose action on a field is exactly the canonical
    ΔNFR ``neighbour-mean − self`` gradient: g = −L_rw·field.  Built from
    the (optionally weighted) adjacency; isolated nodes (degree 0) get a
    zero row (no diffusion).

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, L_rw) : tuple[list, np.ndarray]
        The node ordering and the N×N random-walk Laplacian.
    """
    nodes = _ordered_nodes(G)
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    lap = np.zeros((n, n), dtype=float)
    for node in nodes:
        i = index[node]
        neigh = list(G.neighbors(node))
        if not neigh:
            continue
        # weighted degree (weight defaults to 1.0 when absent)
        weights = [
            float(G[node][m].get("weight", 1.0)) for m in neigh
        ]
        deg = sum(weights)
        if deg <= 0.0:
            continue
        lap[i, i] = 1.0
        for m, w in zip(neigh, weights):
            lap[i, index[m]] -= w / deg
    return nodes, lap


def structural_field(G: Any, nodes: list | None = None) -> Any:
    r"""Return the EPI field as a vector aligned with ``nodes``."""
    if nodes is None:
        nodes = _ordered_nodes(G)
    return np.array(
        [float(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) for n in nodes],
        dtype=float,
    )


def structural_diffusivity(G: Any) -> float:
    r"""Mean structural frequency νf — the diffusion coefficient (mobility).

    In ∂EPI/∂t = −νf·L_rw·EPI, νf plays the role of the diffusivity: the
    larger the structural frequency, the faster the form spreads.
    """
    nodes = _ordered_nodes(G)
    vf = [float(get_attr(G.nodes[n], ALIAS_VF, 0.0)) for n in nodes]
    return float(np.mean(vf)) if vf else 0.0


def degree_weighted_total(G: Any) -> float:
    r"""The conserved structural total Σ_i deg(i)·EPI(i).

    The random-walk Laplacian conserves the degree-weighted total (its left
    null vector is the degree vector), the analogue of the conserved amount
    of a diffusing substance.
    """
    nodes = _ordered_nodes(G)
    total = 0.0
    for node in nodes:
        neigh = list(G.neighbors(node))
        deg = sum(float(G[node][m].get("weight", 1.0)) for m in neigh)
        total += deg * float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    return float(total)


def relaxation_spectrum(G: Any) -> Any:
    r"""Diffusion relaxation rates νf·λ_k (sorted ascending).

    The eigenvalues λ_k of the random-walk Laplacian L_rw scaled by the
    diffusivity νf give the decay rates of the diffusion eigenmodes:
    mode k relaxes as exp(−νf·λ_k·t).  λ₁ = 0 (the conserved uniform mode);
    λ₂ (the spectral gap / Fiedler value) sets the slowest relaxation.

    Returns
    -------
    np.ndarray
        The rates νf·λ_k sorted ascending (real parts).
    """
    _, lap = structural_diffusion_operator(G)
    eig = np.linalg.eigvals(lap).real
    eig.sort()
    return structural_diffusivity(G) * eig


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
        Mean νf (the diffusion coefficient / mobility).
    spectral_gap : float
        λ₂ of L_rw (the Fiedler value); sets the slowest relaxation.
    slowest_relaxation_rate : float
        νf·λ₂ — the slowest diffusion decay rate.
    degree_weighted_conserved : bool
        Σ deg·EPI is conserved under the diffusion flow.
    max_conservation_drift : float
        Max drift of the degree-weighted total over the sampled flow.
    relaxes_to_uniform : bool
        The field relaxes to a spatially uniform diffusive equilibrium.
    final_field_std : float
        Std of the field after the sampled diffusion flow (≈ 0).
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

    @property
    def is_valid_diffusion(self) -> bool:
        """True when the nodal EPI channel verifies as graph diffusion."""
        return (
            self.dnfr_is_graph_laplacian
            and self.degree_weighted_conserved
            and self.relaxes_to_uniform
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_diffusion else "INVALID"
        return (
            f"Structural diffusion [{ok}]: "
            f"ΔNFR_epi = −L_rw·EPI={self.dnfr_is_graph_laplacian} "
            f"(res {self.max_laplacian_residual:.1e}), "
            f"diffusivity νf={self.diffusivity:.4f}, "
            f"spectral gap λ₂={self.spectral_gap:.4f}, "
            f"slowest rate νf·λ₂={self.slowest_relaxation_rate:.4f}, "
            f"deg-weighted conserved={self.degree_weighted_conserved} "
            f"(drift {self.max_conservation_drift:.1e}), "
            f"relaxes to uniform={self.relaxes_to_uniform} "
            f"(final std {self.final_field_std:.1e})"
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

    Confirms (1) the canonical ΔNFR EPI channel equals −L_rw·EPI to machine
    precision, (2) the degree-weighted total is conserved under the
    diffusion flow, and (3) the field relaxes to a uniform diffusive
    equilibrium; and reports the diffusivity νf and the relaxation spectrum.

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
    nodes, lap = structural_diffusion_operator(G)
    n = len(nodes)
    epi = structural_field(G, nodes)

    # (1) ΔNFR (epi channel) == −L_rw·EPI ?
    try:
        dnfr_epi = _dnfr_epi_channel(G, nodes)
        residual = float(np.max(np.abs(dnfr_epi - (-(lap @ epi)))))
        is_laplacian = residual < max(tolerance, 1e-12)
    except Exception:
        residual = float("nan")
        is_laplacian = False

    # diffusivity and spectrum
    diffusivity = structural_diffusivity(G)
    eig = np.linalg.eigvals(lap).real
    eig.sort()
    spectral_gap = float(eig[1]) if n > 1 else 0.0
    slowest_rate = diffusivity * spectral_gap

    # degree vector for the conserved weighted total
    deg = np.array(
        [
            sum(
                float(G[node][m].get("weight", 1.0))
                for m in G.neighbors(node)
            )
            for node in nodes
        ],
        dtype=float,
    )

    # (2)+(3) integrate the pure diffusion flow e ← e − dt·L_rw·e
    e = epi.copy()
    w0 = float(deg @ e)
    max_drift = 0.0
    for _ in range(steps):
        e = e - dt * (lap @ e)
        max_drift = max(max_drift, abs(float(deg @ e) - w0))
    conserved = max_drift < max(tolerance, 1e-9 * (abs(w0) + 1e-12))
    final_std = float(np.std(e))
    relaxes = final_std < max(1e-3, 1e-2 * float(np.std(epi) + 1e-12))

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
    :mod:`tnfr.physics.symplectic_substrate` Hamiltonian flow; the nodal
    equation is its overdamped projection.

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
    pos_linear = (
        abs(float(slope) - drift) < max(tolerance, 1e-6 * abs(drift))
        and abs(float(quad)) < max(tolerance, 1e-6 * abs(drift) + 1e-9)
    )

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
# Discrete modes: the standing waves of the bounded structural manifold
# ---------------------------------------------------------------------------


def _symmetric_normalized_laplacian(G: Any) -> tuple[list, Any]:
    r"""Return the symmetric normalized Laplacian L_sym.

    L_sym = I − D^{-1/2} W D^{-1/2} is symmetric (orthonormal eigenvectors)
    and shares the spectrum of the random-walk diffusion operator L_rw used
    by ΔNFR.
    """
    nodes = _ordered_nodes(G)
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    deg = np.zeros(n, dtype=float)
    adj = np.zeros((n, n), dtype=float)
    for node in nodes:
        i = index[node]
        for m in G.neighbors(node):
            w = float(G[node][m].get("weight", 1.0))
            adj[i, index[m]] = w
            deg[i] += w
    dinv = np.where(deg > 0.0, 1.0 / np.sqrt(deg), 0.0)
    lap = np.eye(n) - (dinv[:, None] * adj * dinv[None, :])
    lap = 0.5 * (lap + lap.T)  # symmetrise residual numerical asymmetry
    return nodes, lap


def structural_eigenmodes(G: Any) -> tuple[Any, Any]:
    r"""Return the discrete eigenmodes of the bounded structural manifold.

    Computes the eigenvalues {λ_k} and orthonormal eigenvectors {v_k} of the
    symmetric normalized Laplacian L_sym (same spectrum as the diffusion
    operator L_rw).  The eigenvalues are the discrete mode "energies"; the
    eigenvectors are the standing-wave mode shapes (orthonormal), sorted by
    ascending λ_k.  λ_1 = 0 is the uniform mode.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (eigenvalues, eigenvectors) : tuple[np.ndarray, np.ndarray]
        ``eigenvalues`` shape ``(N,)`` ascending; ``eigenvectors`` shape
        ``(N, N)`` with column ``k`` the k-th standing-wave mode shape.
    """
    _, lap = _symmetric_normalized_laplacian(G)
    eigvals, eigvecs = np.linalg.eigh(lap)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvals, eigvecs


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
