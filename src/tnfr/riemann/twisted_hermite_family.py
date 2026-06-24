r"""P41: chi-twisted Hermite2-Gaussian eta-parameter sweep for the
Weil-TNFR ratio
:math:`\alpha_\chi(\sigma; \eta, g) = W_\chi[\sigma; h_{\sigma,\eta}]
\,/\, E_{\mathrm{TNFR}}^\chi[\sigma; \eta, g]`.

Motivation
----------

The zeta-track P21 milestone enriches the admissible-family bundle of
P19 with the second-order Hermite-Gaussian profile

.. math::

    h_{\sigma,\eta}(t) = \bigl(1 + \eta (t/\sigma)^2\bigr)\,
                         e^{-t^2/(2\sigma^2)}, \qquad \eta \ge 0,

which preserves evenness and Schwartz decay while injecting a
polynomial envelope of strength :math:`\eta`.  Under the canonical
P19 / P39 sweep the envelope strength is fixed at the default
:math:`\eta = 0.25`.

P41 lifts that single-:math:`\eta` snapshot to a full parameter sweep
on the chi-twisted L-function track.  For every primitive real
Dirichlet character ``chi`` it tabulates
:math:`\alpha_\chi(\sigma; \eta, g)` over a finite
``(eta, gauge, sigma)`` grid, where ``eta = 0.0`` recovers the pure
Gaussian baseline and increasing ``eta`` increasingly biases the
test profile toward the wings.  Positivity across the full table is
the structural P21 analogue of the chi-twisted Hermite2-family
robustness audit.

Reuse
-----
* :class:`tnfr.riemann.admissible_family_sweep.Hermite2GaussianTestFunction`
  -- second-order Hermite-Gaussian admissible profile (unchanged from
  P21).
* :data:`tnfr.riemann.alpha_sweep.DEFAULT_GAUGES`
  -- six canonical structural gauges (unchanged from P18).
* ``build_twisted_test_state_from_test_function`` in
  :mod:`tnfr.riemann.twisted_admissible_family_sweep`
  -- canonical chi-twisted test-state builder (unchanged from P39).
* :func:`tnfr.riemann.twisted_weil_explicit_formula.twisted_weil_zero_side`
  -- chi-twisted Hardy-Z zero-side enumerator (unchanged from P35).
* :func:`tnfr.physics.conservation.compute_energy_functional`
  -- TNFR Lyapunov energy (unchanged from P17).

Honesty disclaimer
------------------
P41 does **not** prove GRH for any ``L(s, chi)`` and does **not**
advance G4 = RH.  Like P21 / P39, it is an RH-equivalent diagnostic
on a *finite* admissible grid (Hermite2 family parameterised by a
finite ``eta`` grid x finite ``sigma`` grid x six gauges).  The full
admissible Schwartz-even space is not exhausted by any finite
``eta``-grid.

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P41
program.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..mathematics.unified_numerical import np
from ..physics.conservation import compute_energy_functional
from .admissible_family_sweep import Hermite2GaussianTestFunction
from .alpha_sweep import DEFAULT_GAUGES, GaugeFn
from .dirichlet_l import DirichletCharacter
from .twisted_admissible_family_sweep import build_twisted_test_state_from_test_function
from .twisted_prime_ladder_hamiltonian import TwistedPrimeLadderHamiltonian
from .twisted_weil_explicit_formula import twisted_weil_zero_side

__all__ = [
    "DEFAULT_HERMITE2_ETAS",
    "TwistedHermite2EtaSweepCertificate",
    "sweep_twisted_hermite2_eta",
]


# ---------------------------------------------------------------------------
# Canonical eta grid
# ---------------------------------------------------------------------------

DEFAULT_HERMITE2_ETAS: tuple[float, ...] = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
)
"""Canonical Hermite2 envelope-strength grid.

* ``eta = 0.0`` recovers the pure Gaussian baseline (the P19
  ``gaussian`` family).
* ``eta = 0.25`` matches the P19 default and the P39 snapshot.
* ``eta in {0.1, 0.5, 1.0, 2.0}`` probes weaker and stronger
  polynomial envelopes; admissibility (evenness + Schwartz decay)
  is preserved for every finite ``eta >= 0``.
"""


# ---------------------------------------------------------------------------
# Certificate dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwistedHermite2EtaSweepCertificate:
    r"""Outcome of a chi-twisted ``(eta, gauge, sigma)`` Hermite2 sweep."""

    character_name: str
    character_modulus: int
    sigmas: object
    etas: tuple[float, ...]
    gauges: tuple[str, ...]
    weil_table: object  # shape (n_eta, n_sigma)
    energy_table: object  # shape (n_eta, n_gauge, n_sigma)
    alpha_table: object  # shape (n_eta, n_gauge, n_sigma)
    weil_all_positive: bool
    alpha_all_positive: bool
    alpha_min: float
    alpha_max: float
    alpha_min_sigma: float
    alpha_min_eta: float
    alpha_min_gauge: str

    def summary(self) -> str:
        return (
            "TwistedHermite2EtaSweepCertificate("
            f"chi='{self.character_name}', "
            f"q={self.character_modulus}, "
            f"n_sigma={len(self.sigmas)}, "
            f"n_eta={len(self.etas)}, "
            f"n_gauge={len(self.gauges)}, "
            f"W_all_positive={self.weil_all_positive}, "
            f"alpha_all_positive={self.alpha_all_positive}, "
            f"alpha_min={self.alpha_min:+.4e} "
            f"@(sigma={self.alpha_min_sigma:.3f}, "
            f"eta={self.alpha_min_eta:.3f}, "
            f"gauge='{self.alpha_min_gauge}'), "
            f"alpha_max={self.alpha_max:+.4e})"
        )


# ---------------------------------------------------------------------------
# Top-level sweep driver
# ---------------------------------------------------------------------------


def sweep_twisted_hermite2_eta(
    chi: DirichletCharacter,
    bundle: TwistedPrimeLadderHamiltonian,
    sigmas: Sequence[float],
    *,
    etas: Sequence[float] | None = None,
    gauges: Mapping[str, GaugeFn] | None = None,
    t_min: float = 0.5,
    t_max: float | None = None,
    initial_step: float = 0.25,
    dps: int = 30,
) -> TwistedHermite2EtaSweepCertificate:
    r"""Tabulate :math:`\alpha_\chi(\sigma; \eta, g)
    = W_\chi[\sigma; h_{\sigma,\eta}] / E_{\mathrm{TNFR}}^\chi[
    \sigma; \eta, g]` across a sigma grid, the Hermite2 envelope
    strength ``eta`` grid, and a structural gauge family, for a
    primitive real Dirichlet character.

    For each ``(eta, sigma)`` pair the chi-twisted Weil functional
    :math:`W_\chi[\sigma; h_{\sigma,\eta}]` is computed *once*
    (gauge-independent) via :func:`twisted_weil_zero_side` (P35).  For
    each ``(eta, gauge, sigma)`` triple the TNFR Lyapunov energy is
    computed by building the test state with the corresponding
    Hermite2 profile and gauge on the P34 chi-twisted graph, then
    evaluating :func:`tnfr.physics.conservation.compute_energy_functional`.

    Parameters
    ----------
    chi
        Primitive real Dirichlet character (P32).
    bundle
        P34 chi-twisted prime-ladder Hamiltonian bundle for ``chi``.
    sigmas
        Sequence of positive Gaussian widths.
    etas
        Sequence of non-negative Hermite2 envelope strengths; defaults
        to :data:`DEFAULT_HERMITE2_ETAS`.
    gauges
        Mapping ``name -> gauge_fn``; defaults to
        :data:`tnfr.riemann.alpha_sweep.DEFAULT_GAUGES` (six canonical
        gauges shared with the zeta-track for canonical comparability).
    t_min, t_max, initial_step, dps
        Forwarded to :func:`twisted_weil_zero_side`.  ``t_max`` is
        ``12 * sigma`` per sigma when ``None`` (matches the Gaussian
        tail discipline used in P35 / P38 / P39; the Hermite2 envelope
        :math:`(1 + \eta x^2) e^{-x^2/2}` decays super-polynomially
        for every finite ``eta``, so the Gaussian bound remains safe).

    Returns
    -------
    TwistedHermite2EtaSweepCertificate
        Dense ``(eta, gauge, sigma)`` tables plus aggregate flags.

    Raises
    ------
    ValueError
        If ``sigmas`` is empty, any sigma is non-positive, ``etas``
        is empty, any eta is negative, or ``gauges`` is empty.
    """
    sigma_array = np.asarray(list(sigmas), dtype=float)
    if sigma_array.size == 0:
        raise ValueError("sigmas must be non-empty")
    if not np.all(sigma_array > 0.0):
        raise ValueError("every sigma must be strictly positive")

    eta_tuple = (
        tuple(float(e) for e in etas) if etas is not None else DEFAULT_HERMITE2_ETAS
    )
    if len(eta_tuple) == 0:
        raise ValueError("etas must be non-empty")
    if any(e < 0.0 for e in eta_tuple):
        raise ValueError("every eta must be non-negative")

    gauge_map = dict(gauges) if gauges is not None else dict(DEFAULT_GAUGES)
    if len(gauge_map) == 0:
        raise ValueError("gauges must be non-empty")
    gauge_names = tuple(gauge_map.keys())

    n_e = len(eta_tuple)
    n_g = len(gauge_names)
    n_s = int(sigma_array.size)

    weil_table = np.empty((n_e, n_s), dtype=float)
    energy_table = np.empty((n_e, n_g, n_s), dtype=float)
    alpha_table = np.empty((n_e, n_g, n_s), dtype=float)

    for i, eta_val in enumerate(eta_tuple):
        for j, sigma in enumerate(sigma_array):
            test = Hermite2GaussianTestFunction(sigma=float(sigma), eta=float(eta_val))
            local_t_max = (12.0 * float(sigma)) if t_max is None else float(t_max)
            w_total, _n_used, _zeros = twisted_weil_zero_side(
                chi,
                test,
                t_min=t_min,
                t_max=local_t_max,
                initial_step=initial_step,
                dps=dps,
            )
            weil_table[i, j] = float(w_total)
            for k, gname in enumerate(gauge_names):
                gfn = gauge_map[gname]
                G = build_twisted_test_state_from_test_function(bundle, test, gfn)
                E = float(compute_energy_functional(G))
                energy_table[i, k, j] = E
                W = weil_table[i, j]
                if E == 0.0:
                    if W > 0.0:
                        alpha_table[i, k, j] = float("inf")
                    elif W < 0.0:
                        alpha_table[i, k, j] = float("-inf")
                    else:
                        alpha_table[i, k, j] = float("nan")
                else:
                    alpha_table[i, k, j] = W / E

    weil_all_positive = bool(np.all(weil_table >= 0.0))

    finite_mask = np.isfinite(alpha_table)
    if finite_mask.any():
        finite_vals = alpha_table[finite_mask]
        alpha_min = float(finite_vals.min())
        alpha_max = float(finite_vals.max())
        idx_flat = int(np.argmin(np.where(finite_mask, alpha_table, np.inf)))
        i_min, rem = divmod(idx_flat, n_g * n_s)
        k_min, j_min = divmod(rem, n_s)
        alpha_min_sigma = float(sigma_array[j_min])
        alpha_min_eta = float(eta_tuple[i_min])
        alpha_min_gauge = gauge_names[k_min]
    else:
        alpha_min = float("nan")
        alpha_max = float("nan")
        alpha_min_sigma = float("nan")
        alpha_min_eta = float("nan")
        alpha_min_gauge = "<none>"

    if finite_mask.any():
        finite_positive = bool(np.all(alpha_table[finite_mask] > 0.0))
    else:
        finite_positive = False
    no_negative_inf = not bool(np.any(np.isneginf(alpha_table)))
    no_nan = not bool(np.any(np.isnan(alpha_table)))
    alpha_all_positive = finite_positive and no_negative_inf and no_nan

    # Silence unused-import noise for ``math`` (kept for callers).
    _ = math.pi

    return TwistedHermite2EtaSweepCertificate(
        character_name=chi.name,
        character_modulus=chi.modulus,
        sigmas=sigma_array,
        etas=eta_tuple,
        gauges=gauge_names,
        weil_table=weil_table,
        energy_table=energy_table,
        alpha_table=alpha_table,
        weil_all_positive=weil_all_positive,
        alpha_all_positive=alpha_all_positive,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_min_sigma=alpha_min_sigma,
        alpha_min_eta=alpha_min_eta,
        alpha_min_gauge=alpha_min_gauge,
    )
