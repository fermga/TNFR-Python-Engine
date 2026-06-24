r"""P18: Admissibility / gauge sweep for the Weil-TNFR ratio
:math:`\alpha(\sigma) = W[\sigma] \,/\, E_{\mathrm{TNFR}}[\sigma]`.

Motivation
----------

P17 (:mod:`tnfr.riemann.weil_positivity`) introduced the Weil-TNFR
positivity bridge, observing :math:`\alpha(\sigma) > 0` for a small
Gaussian-width grid under one *canonical* TNFR mapping
:math:`h_\sigma \mapsto (\Delta\!NFR, \phi, \mathrm{EPI})` on the P14
prime-ladder graph.  The honest §14.6 disclaimer noted that this
mapping is canonical-but-not-unique, and that promoting
:math:`\alpha > 0` to a theorem would require, among other things,
**lower-boundedness of** :math:`\alpha(\sigma)` **across a dense
admissible class** of test functions and a wide gauge family of
structural mappings.

This module provides the corresponding stress test:

* A *dense* :math:`\sigma`-grid (log-spaced) covering both the
  exponentially-small regime (:math:`\sigma \lesssim 1`) and the
  classical regime (:math:`\sigma \sim 10`).
* A family of **structural gauges** parametrising how the test profile
  :math:`h_\sigma` is encoded into the tetrad-driving fields
  :math:`(\Delta\!NFR,\ \phi,\ \mathrm{EPI})`.
* A vectorised driver that reuses each Weil functional
  :math:`W[\sigma]` (gauge-independent) across all gauges, producing
  a 2-D :math:`\alpha`-table with aggregate positivity flags.

If :math:`\alpha(\sigma; g) > 0` for every gauge :math:`g` and every
:math:`\sigma` in the grid, the result strengthens the P17 numerical
evidence considerably — the bridge is robust under canonical-mapping
ambiguity.  A negative value would falsify the bridge *as currently
parameterised* (it would not disprove RH, which depends only on
:math:`W[\sigma]`).

Status: EXPERIMENTAL — Research prototype for TNFR-Riemann P18 program.
Like P17, this is an RH-equivalent diagnostic, not a proof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import networkx as nx

from ..mathematics.unified_numerical import np
from ..physics.conservation import compute_energy_functional
from .prime_ladder_hamiltonian import PrimeLadderHamiltonian
from .weil_explicit_formula import gaussian_test_function, weil_zero_side

# ---------------------------------------------------------------------------
# Gauge family — canonical and probe variants
# ---------------------------------------------------------------------------


# Gauge signature: callable(h_val: float) -> (dnfr, phase, epi).
# The phase value is clipped to [-pi, pi] by the builder; callers may
# return any real number.
GaugeFn = Callable[[float], tuple[float, float, float]]


def _gauge_canonical(h: float) -> tuple[float, float, float]:
    """Match the P17 canonical mapping: h drives all three channels."""
    return (h, h, h)


def _gauge_dnfr_only(h: float) -> tuple[float, float, float]:
    """h enters only the structural pressure channel."""
    return (h, 0.0, 1.0)


def _gauge_phase_only(h: float) -> tuple[float, float, float]:
    """h enters only the phase channel."""
    return (0.0, h, 1.0)


def _gauge_epi_only(h: float) -> tuple[float, float, float]:
    """h enters only the EPI amplitude channel."""
    return (0.0, 0.0, h)


def _gauge_dnfr_phase(h: float) -> tuple[float, float, float]:
    """h enters dnfr and phase; EPI fixed to unit amplitude."""
    return (h, h, 1.0)


def _gauge_pressure_amplified(h: float) -> tuple[float, float, float]:
    """Pressure channel boosted by factor 2; phase and EPI canonical."""
    return (2.0 * h, h, h)


#: Default gauge family probed by :func:`sweep_alpha`.
DEFAULT_GAUGES: Mapping[str, GaugeFn] = {
    "canonical": _gauge_canonical,
    "dnfr_only": _gauge_dnfr_only,
    "phase_only": _gauge_phase_only,
    "epi_only": _gauge_epi_only,
    "dnfr_phase": _gauge_dnfr_phase,
    "pressure_amplified": _gauge_pressure_amplified,
}


# ---------------------------------------------------------------------------
# Test-state builder with gauge selection
# ---------------------------------------------------------------------------


def build_test_state_with_gauge(
    bundle: PrimeLadderHamiltonian,
    sigma: float,
    gauge: GaugeFn,
) -> nx.Graph:
    r"""Map :math:`h_\sigma` onto the P14 graph using a custom gauge.

    For every node :math:`(p,k)` with structural energy
    :math:`E_n = k \log p`, compute :math:`h = h_\sigma(E_n) =
    \exp(-E_n^2/(2\sigma^2))` and write the three fields returned by
    ``gauge(h)`` as ``(dnfr, phase, EPI)``.  The ``phase`` value is
    clipped to :math:`[-\pi, \pi]` (consistent with the wrap-angle
    convention used elsewhere in TNFR).  ``nu_f`` is inherited from
    the P14 construction.

    Parameters
    ----------
    bundle
        Prime-ladder Hamiltonian bundle from P14.
    sigma
        Gaussian width (must be > 0).
    gauge
        Callable mapping ``h_val -> (dnfr, phase, epi)``.

    Returns
    -------
    networkx.Graph
        A copy of ``bundle.graph`` with structural attributes set.

    Raises
    ------
    ValueError
        If ``sigma <= 0``.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive")

    G = bundle.graph.copy()
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    pi = math.pi

    for node in G.nodes():
        p, k = node
        E_n = float(k) * math.log(float(p))
        h_val = math.exp(-(E_n * E_n) * inv_two_sigma_sq)
        d_val, phi_val, epi_val = gauge(h_val)
        # Wrap phase into [-pi, pi].
        if phi_val > pi:
            phi_val = pi
        elif phi_val < -pi:
            phi_val = -pi
        G.nodes[node]["dnfr"] = float(d_val)
        G.nodes[node]["phase"] = float(phi_val)
        G.nodes[node]["EPI"] = float(epi_val)
    return G


# ---------------------------------------------------------------------------
# Certificate dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlphaSweepCertificate:
    r"""Outcome of an :math:`(\sigma, \mathrm{gauge})` sweep of
    :math:`\alpha = W / E_{\mathrm{TNFR}}`.

    Attributes
    ----------
    sigmas
        1-D array of Gaussian widths probed (length ``n_sigma``).
    gauges
        Tuple of gauge names probed (length ``n_gauge``).
    weil_values
        1-D array of :math:`W[\sigma]` indexed by sigma (gauge-independent).
    energy_table
        2-D array shape ``(n_gauge, n_sigma)`` of
        :math:`E_{\mathrm{TNFR}}[\sigma; g]`.
    alpha_table
        2-D array shape ``(n_gauge, n_sigma)`` of
        :math:`\alpha(\sigma; g) = W[\sigma] / E_{\mathrm{TNFR}}[\sigma; g]`.
        Entries with ``E = 0`` are reported as ``+inf`` if ``W > 0``,
        ``-inf`` if ``W < 0`` and ``nan`` if both are zero.
    weil_all_positive
        ``True`` if :math:`W[\sigma] \ge 0` for every sigma in the grid.
    alpha_all_positive
        ``True`` if every finite :math:`\alpha(\sigma; g)` is strictly
        positive.  Infinite-positive entries count as positive.
    alpha_min, alpha_max
        Extremes of the finite alpha values across the table.
    alpha_min_sigma, alpha_min_gauge
        Coordinates of the minimum (most demanding) alpha entry.
    """

    sigmas: object
    gauges: tuple[str, ...]
    weil_values: object
    energy_table: object
    alpha_table: object
    weil_all_positive: bool
    alpha_all_positive: bool
    alpha_min: float
    alpha_max: float
    alpha_min_sigma: float
    alpha_min_gauge: str

    def summary(self) -> str:
        n_sigma = len(self.sigmas)
        n_gauge = len(self.gauges)
        return (
            f"AlphaSweepCertificate(n_sigma={n_sigma}, n_gauge={n_gauge}, "
            f"W_all_positive={self.weil_all_positive}, "
            f"alpha_all_positive={self.alpha_all_positive}, "
            f"alpha_min={self.alpha_min:+.4e} "
            f"@(sigma={self.alpha_min_sigma:.3f}, "
            f"gauge='{self.alpha_min_gauge}'), "
            f"alpha_max={self.alpha_max:+.4e})"
        )


# ---------------------------------------------------------------------------
# Top-level sweep driver
# ---------------------------------------------------------------------------


def sweep_alpha(
    bundle: PrimeLadderHamiltonian,
    sigmas: Sequence[float],
    *,
    gauges: Mapping[str, GaugeFn] | None = None,
    n_zeros: int = 60,
    convergence_tol: float = 1e-12,
    max_zeros: int = 200,
) -> AlphaSweepCertificate:
    r"""Tabulate :math:`\alpha(\sigma; g) = W[\sigma] / E_{\mathrm{TNFR}}
    [\sigma; g]` across a :math:`\sigma`-grid and a gauge family.

    For each :math:`\sigma` in ``sigmas`` the Weil functional
    :math:`W[\sigma]` is computed *once* (via the classical zero side,
    reused across gauges).  For each ``(gauge_name, gauge_fn)`` pair in
    ``gauges`` the canonical TNFR Lyapunov energy is computed by
    building a test state with that gauge and evaluating
    :func:`tnfr.physics.conservation.compute_energy_functional`.

    Parameters
    ----------
    bundle
        P14 prime-ladder Hamiltonian bundle.
    sigmas
        Sequence of Gaussian widths (all > 0).
    gauges
        Mapping ``name -> gauge_fn``; defaults to
        :data:`DEFAULT_GAUGES`.
    n_zeros, convergence_tol, max_zeros
        Forwarded to :func:`weil_zero_side` for the zero-side Weil
        evaluation.

    Returns
    -------
    AlphaSweepCertificate
        Dense :math:`(\sigma, \mathrm{gauge})` table plus aggregate
        positivity flags.

    Raises
    ------
    ValueError
        If ``sigmas`` is empty, any sigma is non-positive, or
        ``gauges`` is empty.
    """
    sigma_array = np.asarray(list(sigmas), dtype=float)
    if sigma_array.size == 0:
        raise ValueError("sigmas must be non-empty")
    if not np.all(sigma_array > 0.0):
        raise ValueError("every sigma must be strictly positive")

    gauge_map: Mapping[str, GaugeFn] = (
        dict(gauges) if gauges is not None else dict(DEFAULT_GAUGES)
    )
    if len(gauge_map) == 0:
        raise ValueError("gauges must be non-empty")
    gauge_names = tuple(gauge_map.keys())
    n_gauge = len(gauge_names)
    n_sigma = int(sigma_array.size)

    # ----- Weil functional W[sigma] (gauge-independent) ---------------
    weil_vals = np.empty(n_sigma, dtype=float)
    for j, sigma in enumerate(sigma_array):
        test = gaussian_test_function(float(sigma))
        w_total, _n_used = weil_zero_side(
            test,
            n_zeros=n_zeros,
            convergence_tol=convergence_tol,
            max_zeros=max_zeros,
        )
        weil_vals[j] = float(w_total)

    # ----- TNFR Lyapunov energy table E[sigma; gauge] -----------------
    energy_table = np.empty((n_gauge, n_sigma), dtype=float)
    for i, name in enumerate(gauge_names):
        gauge_fn = gauge_map[name]
        for j, sigma in enumerate(sigma_array):
            G = build_test_state_with_gauge(bundle, float(sigma), gauge_fn)
            energy_table[i, j] = float(compute_energy_functional(G))

    # ----- Alpha table (handle zero-energy edge cases gracefully) -----
    alpha_table = np.empty_like(energy_table)
    for i in range(n_gauge):
        for j in range(n_sigma):
            E = energy_table[i, j]
            W = weil_vals[j]
            if E == 0.0:
                if W > 0.0:
                    alpha_table[i, j] = float("inf")
                elif W < 0.0:
                    alpha_table[i, j] = float("-inf")
                else:
                    alpha_table[i, j] = float("nan")
            else:
                alpha_table[i, j] = W / E

    # ----- Aggregate flags & extrema ----------------------------------
    weil_all_positive = bool(np.all(weil_vals >= 0.0))

    finite_mask = np.isfinite(alpha_table)
    if finite_mask.any():
        finite_vals = alpha_table[finite_mask]
        alpha_min = float(finite_vals.min())
        alpha_max = float(finite_vals.max())
        # Locate minimum (handles multiple minima by taking first occurrence)
        idx_flat = int(np.argmin(np.where(finite_mask, alpha_table, np.inf)))
        i_min, j_min = divmod(idx_flat, n_sigma)
        alpha_min_sigma = float(sigma_array[j_min])
        alpha_min_gauge = gauge_names[i_min]
    else:
        alpha_min = float("nan")
        alpha_max = float("nan")
        alpha_min_sigma = float("nan")
        alpha_min_gauge = "<none>"

    # Positivity: finite values strictly > 0; +inf passes; -inf/nan fail.
    if finite_mask.any():
        finite_positive = bool(np.all(alpha_table[finite_mask] > 0.0))
    else:
        finite_positive = False
    no_negative_inf = not bool(np.any(np.isneginf(alpha_table)))
    no_nan = not bool(np.any(np.isnan(alpha_table)))
    alpha_all_positive = finite_positive and no_negative_inf and no_nan

    return AlphaSweepCertificate(
        sigmas=sigma_array,
        gauges=gauge_names,
        weil_values=weil_vals,
        energy_table=energy_table,
        alpha_table=alpha_table,
        weil_all_positive=weil_all_positive,
        alpha_all_positive=alpha_all_positive,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_min_sigma=alpha_min_sigma,
        alpha_min_gauge=alpha_min_gauge,
    )


__all__ = [
    "GaugeFn",
    "DEFAULT_GAUGES",
    "build_test_state_with_gauge",
    "AlphaSweepCertificate",
    "sweep_alpha",
]
