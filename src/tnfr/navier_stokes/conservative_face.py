"""TNFR-Navier-Stokes — the honest two-face reading and the blow-up frontier.

Reads a faithful NS flow (``operator.TNFRNavierStokes``) through the paradigm's
two-face machinery, without forcing an analogy:

* ``verify_diffusive_face`` / ``face_of_flow`` -- the *linear* NS operator
  (viscous diffusion of ``K_phi``, ``nu_f <-> nu``) is the **over-damped
  (diffusive)** projection of the conservative substrate wave: it calls the
  engine's :func:`~tnfr.physics.structural_diffusion.verify_overdamped_projection`
  with ``gamma = 1/nu``.  For every physical viscosity this is VALID -- linear
  NS carries no oscillatory (under-damped) content; the conservative/inertial
  character is entirely in the nonlinear stretching source.
* ``vorticity_modal_spectrum`` -- the enstrophy distribution across the emergent
  structural modes (the torus L_rw modes are the Fourier shells), i.e. the
  ``K_phi`` cascade spectrum: where the nonlinear stretching pumps enstrophy.
* ``measure_cascade_frontier`` -- the honest Clay frontier: evolve Taylor-Green
  at several viscosities to a matched structural time ``tau_str = nu*t`` and
  record how the peak enstrophy, the peak stretching production and the
  high-mode enstrophy fraction scale with Reynolds number.

Honest scope: this closes NOTHING.  The linear diffusive face is regular by
construction; the open question is whether the *nonlinear* ``K_phi`` cascade
keeps the enstrophy bounded as ``nu -> 0`` (Re -> inf).  Measured over the
accessible range, the peak grows with Re; the asymptotic is undecidable from
finite resolved points.  Global 3D NS regularity (Clay) stays OPEN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metrics.common import is_structural_equilibrium, structural_coherence
from ..physics.structural_diffusion import (
    damped_wave_rates,
    verify_overdamped_projection,
)
from .operator import TNFRNavierStokes, build_torus_graph_3d

__all__ = [
    "vorticity_modal_spectrum",
    "cascade_moment_hierarchy",
    "moment_ladder_closure",
    "flow_coherence",
    "face_of_flow",
    "verify_diffusive_face",
    "measure_cascade_frontier",
    "CascadeFrontierCertificate",
]


def vorticity_modal_spectrum(
    flow: TNFRNavierStokes, n_shells: int = 12
) -> dict[str, Any]:
    r"""Enstrophy per wavenumber shell -- the ``K_phi`` cascade spectrum.

    The periodic-torus structural modes (L_rw eigenvectors) are the Fourier
    modes, and the structural eigenvalue grows monotonically with ``|k|``, so
    the enstrophy-by-shell is exactly the canonical modal-energy spectrum of the
    vorticity field.  Reports the shell spectrum plus the fraction of enstrophy
    living in the upper half of the resolved shells (the cascade indicator).
    """
    w_hat = flow._vorticity_hat(flow.u_hat)
    ens_k = 0.5 * (
        np.abs(w_hat[0]) ** 2
        + np.abs(w_hat[1]) ** 2
        + np.abs(w_hat[2]) ** 2
    )
    kmag = np.sqrt(flow.k2)
    kmax = float(np.max(kmag))
    edges = np.linspace(0.0, kmax + 1e-9, int(n_shells) + 1)
    spectrum = []
    for j in range(int(n_shells)):
        sel = (kmag >= edges[j]) & (kmag < edges[j + 1])
        spectrum.append(float(np.sum(ens_k[sel])))
    total = float(np.sum(spectrum)) or 1.0
    half = int(n_shells) // 2
    high_fraction = float(np.sum(spectrum[half:]) / total)
    return {
        "shell_edges": [float(x) for x in edges],
        "enstrophy_spectrum": spectrum,
        "high_mode_fraction": high_fraction,
        "n_shells": int(n_shells),
    }


def cascade_moment_hierarchy(flow: TNFRNavierStokes) -> dict[str, Any]:
    r"""The lambda-moment hierarchy of the cascade (the newly-unlocked read-out).

    The old diffusive reading saw only the scalar enstrophy.  The emergent modal
    basis (L_rw modes, lambda_k) gives the whole ladder of lambda-moments
    ``M_p = sum lambda_k^p E(lambda_k)``:

      * ``M_0`` = energy       ``(1/2)<|u|^2>``   -- the CONSERVATIVE-face budget
        (bounded by Leray: ``M_0(t) <= M_0(0)``);
      * ``M_1`` = enstrophy    ``(1/2)<|omega|^2>``  (the classical blow-up
        quantity);
      * ``M_2`` = palinstrophy ``(1/2)<|grad omega|^2>``  (weights the
        small-scale / high-lambda tail more).

    The blow-up question in this basis: the low moment ``M_0`` is bounded, so
    Clay is exactly whether the ladder ``M_p`` (``p >= 1``) stays uniformly
    bounded as ``nu -> 0``.  This is the NS twin of the re-founded Riemann
    coherence budget (a low moment bounded, the high-moment tail the open wall).

    Also returns the Kolmogorov resolution flag ``kmax * eta`` at the current
    state (``> 1`` means the small scales are resolved; below that the high
    moments are under-resolved lower bounds).
    """
    energy = flow.energy()
    enstrophy = flow.enstrophy()
    w_hat = flow._vorticity_hat(flow.u_hat)
    palinstrophy = 0.0
    for a in range(3):
        for kk in (flow.kx, flow.ky, flow.kz):
            g = np.fft.ifftn(1j * kk * w_hat[a]).real
            palinstrophy += float(np.mean(g**2))
    palinstrophy *= 0.5
    eps = 2.0 * flow.nu * enstrophy  # ~ energy dissipation rate
    eta = (flow.nu**3 / eps) ** 0.25 if eps > 0 else 0.0
    kmax_eta = (flow.n / 2.0) * eta
    return {
        "energy_m0": energy,
        "enstrophy_m1": enstrophy,
        "palinstrophy_m2": palinstrophy,
        "m1_over_m0": enstrophy / energy if energy else 0.0,
        "m2_over_m1": palinstrophy / enstrophy if enstrophy else 0.0,
        "kmax_eta": float(kmax_eta),
        "resolved": bool(kmax_eta > 1.0),
    }


def moment_ladder_closure(flow: TNFRNavierStokes) -> dict[str, Any]:
    r"""The `H^s` / Foias-Temam ladder rung for enstrophy, in modal language.

    The enstrophy rung is ``dM_1/dt = P - 2*nu*M_2`` with ``P`` the
    vortex-stretching production (nonlinear VAL) and ``2*nu*M_2`` the palinstrophy
    dissipation (diffusive face).  The ladder **closes** (regularity) iff ``P`` is
    dominated by the dissipation uniformly in Re.  Two exact / measured handles:

    * **interpolation saturation** ``s = M_1^2 / (M_0*M_2) in (0, 1]`` -- the exact
      Cauchy-Schwarz coupling on the modal spectrum ``E(lambda_k)``.  ``s = 1`` is a
      single-scale (concentrated) spectrum (dangerous); ``s -> 0`` a spread spectrum.
      It bounds the dissipation below: ``2*nu*M_2 >= 2*nu*M_1^2/M_0``.
    * **closure ratio** ``P / (2*nu*M_2)`` -- ``< 1`` means dissipation dominates
      (enstrophy decreasing), ``= 1`` at the enstrophy peak, ``> 1`` production wins.

    Returns the state's rung data.  Uniform-in-Re closure (the ratio staying ``< 1``
    as ``nu -> 0``) is exactly Clay -- this read-out measures it, it does not close it.
    """
    h = cascade_moment_hierarchy(flow)
    m0 = h["energy_m0"]
    m1 = h["enstrophy_m1"]
    m2 = h["palinstrophy_m2"]
    production = flow.stretching_production()
    dissipation = 2.0 * flow.nu * m2
    s = (m1 * m1) / (m0 * m2) if (m0 > 0.0 and m2 > 0.0) else 0.0
    ratio = production / dissipation if dissipation > 0.0 else float("nan")
    return {
        "interpolation_saturation": float(s),
        "production": float(production),
        "palinstrophy_dissipation": float(dissipation),
        "closure_ratio": float(ratio),
        "interpolation_ok": bool(s <= 1.0 + 1e-9),
        "dissipation_dominates": bool(ratio < 1.0),
        "kmax_eta": h["kmax_eta"],
        "resolved": h["resolved"],
    }


def flow_coherence(flow: TNFRNavierStokes) -> dict[str, Any]:
    r"""Read a static pressure-coherence snapshot of the vorticity field.

    The domain-specific pressure is the random-walk-Laplacian action on
    vorticity magnitude, ``ΔNFR = -L_rw·|ω|``.  The function evaluates the
    shared scalar map :func:`~tnfr.metrics.common.structural_coherence` with
    the explicit static assumption ``dEPI = 0``.  It therefore reports
    ``C_static = 1/(1 + mean|ΔNFR|)`` rather than the engine's dynamic total
    ``C(t)``, which requires a measured or declared EPI-rate channel.

    Reusing the scalar map does not identify Navier--Stokes, graph, arithmetic,
    and chemical state spaces or prove that they share an attractor.  The
    compatibility key ``at_equilibrium`` only tests whether the mean pressure
    magnitude is within the selected tolerance while the rate is held at zero.
    Successive calls along a numerical trajectory can record a trend;
    one snapshot does not certify convergence, regularity, or an asymptotic
    Navier--Stokes bound.
    """
    wx, wy, wz = flow.vorticity()
    mag = np.sqrt(wx**2 + wy**2 + wz**2)
    neigh = (
        np.roll(mag, 1, 0) + np.roll(mag, -1, 0)
        + np.roll(mag, 1, 1) + np.roll(mag, -1, 1)
        + np.roll(mag, 1, 2) + np.roll(mag, -1, 2)
    ) / 6.0
    dnfr = neigh - mag  # = -(L_rw . |omega|): the canonical DeltaNFR realisation
    mean_abs_dnfr = float(np.mean(np.abs(dnfr)))
    coherence = structural_coherence(mean_abs_dnfr, 0.0)
    frag_floor = 1.0 / (math.pi + 1.0)
    strong_cut = math.pi / (math.pi + 1.0)
    mean_pressure_within_tolerance = bool(
        is_structural_equilibrium(mean_abs_dnfr, 0.0, eps_dnfr=1e-2)
    )
    return {
        "coherence": float(coherence),
        "mean_abs_dnfr": mean_abs_dnfr,
        "at_equilibrium": mean_pressure_within_tolerance,
        "mean_pressure_within_tolerance": mean_pressure_within_tolerance,
        "coherence_kind": "static_pressure_only_structural_coherence",
        "depi_assumption": 0.0,
        "dynamic_equilibrium_assessed": False,
        "strong": bool(coherence > strong_cut),
        "coherent": bool(coherence > frag_floor),
    }


def face_of_flow(nu: float, *, n_probe: int = 4) -> dict[str, Any]:
    r"""Which face the *linear* NS operator sits on, at viscosity ``nu``.

    Maps viscosity to the damped-wave damping ``gamma = 1/nu`` (the canonical
    ``nu_f = 1/gamma`` identity) and compares ``gamma^2`` to ``4*lambda_max`` of
    the structural spectrum: ``gamma^2 > 4*lambda_max`` means every mode is
    over-damped, i.e. the linear operator is on the diffusive face.
    """
    graph = build_torus_graph_3d(int(n_probe))
    gamma = 1.0 / float(nu)
    lambdas, _s_slow, _s_fast = damped_wave_rates(graph, gamma)
    lam_max = float(np.max(lambdas)) if len(lambdas) else 0.0
    overdamped = bool(gamma * gamma > 4.0 * lam_max)
    return {
        "nu": float(nu),
        "gamma": gamma,
        "lambda_max": lam_max,
        "overdamped": overdamped,
        "face": "diffusive (over-damped)" if overdamped else "under-damped",
        "note": (
            "linear NS is diffusive; the conservative/inertial content is the "
            "nonlinear stretching source, not a linear wave"
        ),
    }


def verify_diffusive_face(nu: float, *, n_probe: int = 4, gamma: float | None = None):
    r"""Engine certificate that linear NS (``nu_f = nu``) is the over-damped
    projection of the conservative substrate wave (``gamma = 1/nu``).

    Returns the engine's
    :class:`~tnfr.physics.structural_diffusion.OverdampedProjectionCertificate`
    on a small representative torus graph (the face is an operator-level
    property of ``gamma`` vs the L_rw spectrum, resolution-independent).
    """
    graph = build_torus_graph_3d(int(n_probe))
    g = (1.0 / float(nu)) if gamma is None else float(gamma)
    return verify_overdamped_projection(graph, gamma=g)


@dataclass(frozen=True)
class CascadeFrontierCertificate:
    r"""The honest Clay frontier: nonlinear ``K_phi`` cascade vs Reynolds.

    Attributes
    ----------
    reynolds : list[float]
        ``Re = 2*pi/nu`` per run.
    peak_debt : list[float]
        Peak enstrophy over the trajectory divided by the initial enstrophy.
    peak_stretching : list[float]
        Peak vortex-stretching production (the nonlinear VAL source).
    high_mode_fraction : list[float]
        Fraction of enstrophy in the upper resolved shells at the peak.
    saturates : list[bool]
        Whether each run's enstrophy peaks and then decays (bounded debt at
        fixed Re -> the diffusive face regularises).
    tau_target : float
        Matched structural time ``tau_str = nu*t`` reached by every run.
    """

    reynolds: list[float]
    peak_debt: list[float]
    peak_stretching: list[float]
    high_mode_fraction: list[float]
    saturates: list[bool]
    tau_target: float

    @property
    def debt_grows_with_re(self) -> bool:
        """Whether the peak enstrophy debt increases across the Re sweep."""
        d = self.peak_debt
        return len(d) >= 2 and d[-1] > d[0]

    def summary(self) -> str:
        """Human-readable one-line verdict (honest: closes nothing)."""
        pairs = ", ".join(
            f"Re={r:.0f}:{d:.2f}"
            for r, d in zip(self.reynolds, self.peak_debt)
        )
        trend = "GROWS" if self.debt_grows_with_re else "flat/decays"
        allsat = "all saturate" if all(self.saturates) else "NOT all saturate"
        return (
            f"CascadeFrontier[Clay OPEN]: peak enstrophy debt {trend} with Re "
            f"({pairs}); {allsat} at fixed Re (tau_str={self.tau_target}); "
            f"asymptotic Re->inf undecidable from finite resolved points"
        )


def measure_cascade_frontier(
    *,
    n: int = 16,
    viscosities: tuple[float, ...] = (0.05, 0.02, 0.01),
    tau_target: float = 0.2,
    dt: float = 0.01,
    amplitude: float = 1.0,
    n_shells: int = 12,
) -> CascadeFrontierCertificate:
    r"""Measure how the nonlinear ``K_phi`` cascade scales with Reynolds.

    Evolves the Taylor-Green vortex at each viscosity to a matched structural
    time ``tau_str = nu*t = tau_target`` and records the peak enstrophy debt,
    the peak stretching production and the peak high-mode enstrophy fraction.
    Deterministic (no RNG).  Closes nothing; Clay stays OPEN.
    """
    reynolds: list[float] = []
    peak_debt: list[float] = []
    peak_stretch: list[float] = []
    high_frac: list[float] = []
    saturates: list[bool] = []
    for nu in viscosities:
        flow = TNFRNavierStokes(n, nu, amplitude)
        steps = max(1, int(round(tau_target / nu / dt)))
        ens0 = flow.enstrophy()
        peak_e = ens0
        peak_s = abs(flow.stretching_production())
        peak_hf = vorticity_modal_spectrum(flow, n_shells)["high_mode_fraction"]
        last_e = ens0
        for _ in range(steps):
            flow.step(dt)
            e = flow.enstrophy()
            peak_e = max(peak_e, e)
            peak_s = max(peak_s, abs(flow.stretching_production()))
            peak_hf = max(
                peak_hf,
                vorticity_modal_spectrum(flow, n_shells)["high_mode_fraction"],
            )
            last_e = e
        reynolds.append(2.0 * np.pi / nu)
        peak_debt.append(peak_e / ens0 if ens0 else 0.0)
        peak_stretch.append(peak_s)
        high_frac.append(peak_hf)
        saturates.append(bool(last_e < peak_e * 0.999))
    return CascadeFrontierCertificate(
        reynolds=reynolds,
        peak_debt=peak_debt,
        peak_stretching=peak_stretch,
        high_mode_fraction=high_frac,
        saturates=saturates,
        tau_target=float(tau_target),
    )
