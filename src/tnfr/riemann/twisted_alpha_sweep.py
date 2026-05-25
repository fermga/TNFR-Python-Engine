r"""P38: chi-twisted admissibility / gauge sweep for the Weil-TNFR
ratio
:math:`\alpha_\chi(\sigma; g) = W_\chi[\sigma]
\,/\, E_{\mathrm{TNFR}}^\chi[\sigma; g]`.

Motivation
----------

P37 (:mod:`tnfr.riemann.twisted_weil_positivity`) extended the P17
Weil-TNFR positivity bridge to primitive real Dirichlet L-functions
``L(s, chi)`` using a single *canonical* TNFR mapping
:math:`h_\sigma \mapsto (\Delta\!NFR, \phi, \mathrm{EPI})` on the P34
chi-twisted prime-ladder graph and a small Gaussian-width grid.  The
honest §13sexiesdecies disclaimer noted that this mapping is
canonical-but-not-unique and that strengthening
:math:`\alpha_\chi(\sigma) > 0` toward a GRH$_\chi$-equivalent
witness would require, among other things, **lower-boundedness of**
:math:`\alpha_\chi(\sigma)` **across a dense admissible class** of
test functions and a wide gauge family of structural mappings -- the
chi-twisted analogue of the P18 stress test.

This module performs that stress test for the chi-twisted bridge:

* A *dense* :math:`\sigma`-grid covering both the exponentially-small
  regime (:math:`\sigma \lesssim 1`) and the classical regime
  (:math:`\sigma \sim 10`).
* The same family of **structural gauges** parametrising how the
  test profile :math:`h_\sigma` is encoded into the tetrad-driving
  fields :math:`(\Delta\!NFR,\ \phi,\ \mathrm{EPI})` introduced in
  P18 (so the gauge family stays canonical across both pistes).
* A vectorised driver that reuses each chi-twisted Weil functional
  :math:`W_\chi[\sigma]` (gauge-independent) across all gauges,
  producing a 2-D :math:`\alpha_\chi`-table with aggregate positivity
  flags.

If :math:`\alpha_\chi(\sigma; g) > 0` for every gauge :math:`g` and
every :math:`\sigma` in the grid, the result strengthens the P37
numerical evidence considerably -- the chi-twisted bridge is robust
under canonical-mapping ambiguity.  A negative value would falsify
the bridge *as currently parameterised* for that character; it would
not disprove GRH$_\chi$, which depends only on
:math:`W_\chi[\sigma]`.

Honesty disclaimer
------------------
P38 does **not** prove GRH for any ``L(s, chi)`` and does **not**
advance G4 = RH.  Like P18 / P37, it is an RH-equivalent diagnostic
on a finite Gaussian admissible grid evaluated under a finite gauge
family.  It is the chi-twisted *robustness audit* of the P37
positivity bridge.

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P38 program.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import networkx as nx

from ..mathematics.unified_numerical import np
from ..physics.conservation import compute_energy_functional
from .alpha_sweep import DEFAULT_GAUGES, GaugeFn
from .dirichlet_l import DirichletCharacter
from .twisted_prime_ladder_hamiltonian import TwistedPrimeLadderHamiltonian
from .twisted_weil_explicit_formula import twisted_weil_zero_side
from .weil_explicit_formula import gaussian_test_function

__all__ = [
    "TwistedAlphaSweepCertificate",
    "build_twisted_test_state_with_gauge",
    "sweep_twisted_alpha",
]


# ---------------------------------------------------------------------------
# Test-state builder with gauge selection (chi-twisted graph)
# ---------------------------------------------------------------------------


def build_twisted_test_state_with_gauge(
    bundle: TwistedPrimeLadderHamiltonian,
    sigma: float,
    gauge: GaugeFn,
) -> nx.Graph:
    r"""Map :math:`h_\sigma` onto the P34 chi-twisted graph via a gauge.

    For every node :math:`(p,k)` with structural energy
    :math:`E_n = k \log p`, compute
    :math:`h = h_\sigma(E_n) = \exp(-E_n^2/(2\sigma^2))` and write the
    three fields returned by ``gauge(h)`` as ``(dnfr, phase, EPI)``.
    The ``phase`` value is clipped to :math:`[-\pi, \pi]` (consistent
    with the wrap-angle convention used elsewhere in TNFR).
    ``nu_f`` is inherited from the P34 construction.

    Parameters
    ----------
    bundle
        P34 chi-twisted prime-ladder Hamiltonian bundle.
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
class TwistedAlphaSweepCertificate:
    r"""Outcome of an :math:`(\sigma, \mathrm{gauge})` sweep of
    :math:`\alpha_\chi = W_\chi / E_{\mathrm{TNFR}}^\chi`.

    Attributes
    ----------
    character_name
        Identifier of the primitive Dirichlet character probed.
    character_modulus
        Modulus ``q`` of the character.
    sigmas
        1-D array of Gaussian widths probed (length ``n_sigma``).
    gauges
        Tuple of gauge names probed (length ``n_gauge``).
    weil_values
        1-D array of :math:`W_\chi[\sigma]` indexed by sigma
        (gauge-independent).
    energy_table
        2-D array shape ``(n_gauge, n_sigma)`` of
        :math:`E_{\mathrm{TNFR}}^\chi[\sigma; g]`.
    alpha_table
        2-D array shape ``(n_gauge, n_sigma)`` of
        :math:`\alpha_\chi(\sigma; g) = W_\chi[\sigma]
        / E_{\mathrm{TNFR}}^\chi[\sigma; g]`.  Entries with
        ``E = 0`` are reported as ``+inf`` if ``W > 0``,
        ``-inf`` if ``W < 0`` and ``nan`` if both are zero.
    weil_all_positive
        ``True`` if :math:`W_\chi[\sigma] \ge 0` for every sigma in
        the grid.
    alpha_all_positive
        ``True`` if every finite :math:`\alpha_\chi(\sigma; g)` is
        strictly positive.  Infinite-positive entries count as
        positive.
    alpha_min, alpha_max
        Extremes of the finite alpha values across the table.
    alpha_min_sigma, alpha_min_gauge
        Coordinates of the minimum (most demanding) alpha entry.
    """

    character_name: str
    character_modulus: int
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
            f"TwistedAlphaSweepCertificate(chi='{self.character_name}', "
            f"q={self.character_modulus}, "
            f"n_sigma={n_sigma}, n_gauge={n_gauge}, "
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


def sweep_twisted_alpha(
    chi: DirichletCharacter,
    bundle: TwistedPrimeLadderHamiltonian,
    sigmas: Sequence[float],
    *,
    gauges: Mapping[str, GaugeFn] | None = None,
    t_min: float = 0.5,
    t_max: float | None = None,
    initial_step: float = 0.25,
    dps: int = 30,
) -> TwistedAlphaSweepCertificate:
    r"""Tabulate :math:`\alpha_\chi(\sigma; g) = W_\chi[\sigma] /
    E_{\mathrm{TNFR}}^\chi[\sigma; g]` across a :math:`\sigma`-grid
    and a gauge family for a primitive real Dirichlet character.

    For each :math:`\sigma` in ``sigmas`` the chi-twisted Weil
    functional :math:`W_\chi[\sigma]` is computed *once* (via the
    classical zero side from P35, reused across gauges).  For each
    ``(gauge_name, gauge_fn)`` pair in ``gauges`` the canonical TNFR
    Lyapunov energy is computed by building a test state with that
    gauge on the P34 chi-twisted graph and evaluating
    :func:`tnfr.physics.conservation.compute_energy_functional`.

    Parameters
    ----------
    chi
        Primitive real Dirichlet character (P32).
    bundle
        P34 chi-twisted prime-ladder Hamiltonian bundle for the same
        character ``chi``.
    sigmas
        Sequence of Gaussian widths (all > 0).
    gauges
        Mapping ``name -> gauge_fn``; defaults to
        :data:`tnfr.riemann.alpha_sweep.DEFAULT_GAUGES` (shared across
        the zeta and L-function pistes for canonical comparability).
    t_min, t_max, initial_step, dps
        Forwarded to :func:`twisted_weil_zero_side` (P35) for the
        zero-side Weil evaluation.  ``t_max`` defaults to
        ``12 * sigma`` per sigma.

    Returns
    -------
    TwistedAlphaSweepCertificate
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

    # ----- chi-twisted Weil functional W_chi[sigma] -------------------
    weil_vals = np.empty(n_sigma, dtype=float)
    for j, sigma in enumerate(sigma_array):
        test = gaussian_test_function(float(sigma))
        local_t_max = (12.0 * float(sigma)) if t_max is None else float(t_max)
        w_total, _n_used, _zeros = twisted_weil_zero_side(
            chi,
            test,
            t_min=t_min,
            t_max=local_t_max,
            initial_step=initial_step,
            dps=dps,
        )
        weil_vals[j] = float(w_total)

    # ----- TNFR Lyapunov energy table E_chi[sigma; gauge] -------------
    energy_table = np.empty((n_gauge, n_sigma), dtype=float)
    for i, name in enumerate(gauge_names):
        gauge_fn = gauge_map[name]
        for j, sigma in enumerate(sigma_array):
            G = build_twisted_test_state_with_gauge(
                bundle, float(sigma), gauge_fn
            )
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
        idx_flat = int(np.argmin(np.where(finite_mask, alpha_table, np.inf)))
        i_min, j_min = divmod(idx_flat, n_sigma)
        alpha_min_sigma = float(sigma_array[j_min])
        alpha_min_gauge = gauge_names[i_min]
    else:
        alpha_min = float("nan")
        alpha_max = float("nan")
        alpha_min_sigma = float("nan")
        alpha_min_gauge = "<none>"

    if finite_mask.any():
        finite_positive = bool(np.all(alpha_table[finite_mask] > 0.0))
    else:
        finite_positive = False
    no_negative_inf = not bool(np.any(np.isneginf(alpha_table)))
    no_nan = not bool(np.any(np.isnan(alpha_table)))
    alpha_all_positive = finite_positive and no_negative_inf and no_nan

    return TwistedAlphaSweepCertificate(
        character_name=str(getattr(chi, "name", "chi")),
        character_modulus=int(getattr(chi, "modulus", 0)),
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
