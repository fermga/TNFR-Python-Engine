r"""P39: chi-twisted admissible-family sweep for the Weil-TNFR ratio
:math:`\alpha_\chi(\sigma; f, g) = W_\chi[\sigma; f]
\,/\, E_{\mathrm{TNFR}}^\chi[\sigma; f, g]`.

Motivation
----------

P38 (:mod:`tnfr.riemann.twisted_alpha_sweep`) sweeps the canonical
six-gauge family ``DEFAULT_GAUGES`` against a Gaussian-only test
profile :math:`h_\sigma(t) = \exp(-t^2/(2\sigma^2))` for primitive
real Dirichlet ``L(s, chi)``.  P39 extends that audit beyond the
Gaussian envelope: it reuses the admissible-family bundle introduced
by the zeta-track P19 (:mod:`tnfr.riemann.admissible_family_sweep`)
- ``gaussian``, ``gaussian_mixture``, ``hermite2_gaussian`` - and
combines it with the same gauge family from P18, producing a dense
``(family, gauge, sigma)`` table of chi-twisted alpha values.

If :math:`\alpha_\chi > 0` for every cell of the table the chi-twisted
positivity bridge is robust under *both* canonical-mapping ambiguity
(P38) and test-profile ambiguity (P39), strengthening the P37
diagnostic considerably.

Reuse
-----
* :data:`tnfr.riemann.admissible_family_sweep.DEFAULT_TEST_FAMILIES`
  — three admissible Schwartz-even families (unchanged from P19).
* :data:`tnfr.riemann.alpha_sweep.DEFAULT_GAUGES`
  — six canonical structural gauges (unchanged from P18).
* :func:`tnfr.riemann.twisted_weil_explicit_formula.twisted_weil_zero_side`
  — chi-twisted Hardy-Z zero-side enumerator (unchanged from P35).
* :func:`tnfr.physics.conservation.compute_energy_functional`
  — TNFR Lyapunov energy (unchanged from P17).

Honesty disclaimer
------------------
P39 does **not** prove GRH for any ``L(s, chi)`` and does **not**
advance G4 = RH.  Like P19 / P38, it is an RH-equivalent diagnostic
on a *finite* admissible grid (three families × finite sigma grid ×
six gauges) evaluated for a single character at a time.  It is the
chi-twisted *family + gauge* robustness audit of the P37 bridge.

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P39 program.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import networkx as nx

from ..mathematics.unified_numerical import np
from ..physics.conservation import compute_energy_functional
from .admissible_family_sweep import (
    DEFAULT_TEST_FAMILIES,
    AdmissibleTestFunction,
    FamilyFactory,
)
from .alpha_sweep import DEFAULT_GAUGES, GaugeFn
from .dirichlet_l import DirichletCharacter
from .twisted_prime_ladder_hamiltonian import TwistedPrimeLadderHamiltonian
from .twisted_weil_explicit_formula import twisted_weil_zero_side

__all__ = [
    "TwistedAdmissibleFamilySweepCertificate",
    "build_twisted_test_state_from_test_function",
    "sweep_twisted_admissible_family",
]


# ---------------------------------------------------------------------------
# Test-state builder with arbitrary admissible test function + gauge
# ---------------------------------------------------------------------------


def build_twisted_test_state_from_test_function(
    bundle: TwistedPrimeLadderHamiltonian,
    test: AdmissibleTestFunction,
    gauge: GaugeFn,
) -> nx.Graph:
    r"""Map :math:`h = \mathrm{test}.h(E_n)` onto the P34 chi-twisted graph.

    For every node :math:`(p, k)` with structural energy
    :math:`E_n = k \log p`, evaluate :math:`h = \mathrm{test}.h(E_n)`
    and write the three fields returned by ``gauge(h)`` as
    ``(dnfr, phase, EPI)``.  ``phase`` is clipped to :math:`[-\pi, \pi]`
    per the canonical wrap-angle convention.  ``nu_f`` is inherited
    from the P34 construction.

    Parameters
    ----------
    bundle
        P34 chi-twisted prime-ladder Hamiltonian bundle.
    test
        Admissible Schwartz-even test function (P19 protocol).
    gauge
        Callable mapping ``h_val -> (dnfr, phase, epi)``.

    Returns
    -------
    networkx.Graph
        A copy of ``bundle.graph`` with structural attributes set.
    """
    G = bundle.graph.copy()
    pi = math.pi
    for node in G.nodes():
        p, k = node
        e_n = float(k) * math.log(float(p))
        h_val = float(test.h(e_n))
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
class TwistedAdmissibleFamilySweepCertificate:
    r"""Outcome of a chi-twisted ``(family, gauge, sigma)`` alpha sweep."""

    character_name: str
    character_modulus: int
    sigmas: object
    families: tuple[str, ...]
    gauges: tuple[str, ...]
    weil_table: object
    energy_table: object
    alpha_table: object
    weil_all_positive: bool
    alpha_all_positive: bool
    alpha_min: float
    alpha_max: float
    alpha_min_sigma: float
    alpha_min_family: str
    alpha_min_gauge: str

    def summary(self) -> str:
        return (
            "TwistedAdmissibleFamilySweepCertificate("
            f"chi='{self.character_name}', "
            f"q={self.character_modulus}, "
            f"n_sigma={len(self.sigmas)}, "
            f"n_family={len(self.families)}, "
            f"n_gauge={len(self.gauges)}, "
            f"W_all_positive={self.weil_all_positive}, "
            f"alpha_all_positive={self.alpha_all_positive}, "
            f"alpha_min={self.alpha_min:+.4e} "
            f"@(sigma={self.alpha_min_sigma:.3f}, "
            f"family='{self.alpha_min_family}', "
            f"gauge='{self.alpha_min_gauge}'), "
            f"alpha_max={self.alpha_max:+.4e})"
        )


# ---------------------------------------------------------------------------
# Top-level sweep driver
# ---------------------------------------------------------------------------


def sweep_twisted_admissible_family(
    chi: DirichletCharacter,
    bundle: TwistedPrimeLadderHamiltonian,
    sigmas: Sequence[float],
    *,
    families: Mapping[str, FamilyFactory] | None = None,
    gauges: Mapping[str, GaugeFn] | None = None,
    t_min: float = 0.5,
    t_max: float | None = None,
    initial_step: float = 0.25,
    dps: int = 30,
) -> TwistedAdmissibleFamilySweepCertificate:
    r"""Tabulate :math:`\alpha_\chi(\sigma; f, g) = W_\chi[\sigma; f]
    / E_{\mathrm{TNFR}}^\chi[\sigma; f, g]` across a sigma grid,
    an admissible test-function family bundle, and a structural
    gauge family, for a primitive real Dirichlet character.

    For each ``(family, sigma)`` pair the chi-twisted Weil functional
    :math:`W_\chi[\sigma; f]` is computed *once* (gauge-independent)
    via :func:`twisted_weil_zero_side` (P35).  For each
    ``(family, gauge, sigma)`` triple the TNFR Lyapunov energy is
    computed by building the test state with that family/gauge on the
    P34 chi-twisted graph and evaluating
    :func:`tnfr.physics.conservation.compute_energy_functional`.

    Parameters
    ----------
    chi
        Primitive real Dirichlet character (P32).
    bundle
        P34 chi-twisted prime-ladder Hamiltonian bundle for ``chi``.
    sigmas
        Sequence of positive Gaussian widths driving every family.
    families
        Mapping ``name -> factory(sigma) -> AdmissibleTestFunction``;
        defaults to
        :data:`tnfr.riemann.admissible_family_sweep.DEFAULT_TEST_FAMILIES`
        (``gaussian``, ``gaussian_mixture``, ``hermite2_gaussian``).
    gauges
        Mapping ``name -> gauge_fn``; defaults to
        :data:`tnfr.riemann.alpha_sweep.DEFAULT_GAUGES` (six canonical
        gauges shared with the zeta-track for canonical comparability).
    t_min, t_max, initial_step, dps
        Forwarded to :func:`twisted_weil_zero_side`.  ``t_max`` is
        ``12 * sigma`` per sigma when ``None`` (matches the Gaussian
        tail discipline used in P35 / P38; families with envelopes
        decaying faster than Gaussian inherit the same bound safely).

    Returns
    -------
    TwistedAdmissibleFamilySweepCertificate
        Dense ``(family, gauge, sigma)`` tables plus aggregate flags.

    Raises
    ------
    ValueError
        If ``sigmas`` is empty, any sigma is non-positive, or
        ``families`` / ``gauges`` is empty.
    """
    sigma_array = np.asarray(list(sigmas), dtype=float)
    if sigma_array.size == 0:
        raise ValueError("sigmas must be non-empty")
    if not np.all(sigma_array > 0.0):
        raise ValueError("every sigma must be strictly positive")

    family_map = (
        dict(families) if families is not None else dict(DEFAULT_TEST_FAMILIES)
    )
    if len(family_map) == 0:
        raise ValueError("families must be non-empty")

    gauge_map = dict(gauges) if gauges is not None else dict(DEFAULT_GAUGES)
    if len(gauge_map) == 0:
        raise ValueError("gauges must be non-empty")

    family_names = tuple(family_map.keys())
    gauge_names = tuple(gauge_map.keys())
    n_f = len(family_names)
    n_g = len(gauge_names)
    n_s = int(sigma_array.size)

    weil_table = np.empty((n_f, n_s), dtype=float)
    energy_table = np.empty((n_f, n_g, n_s), dtype=float)
    alpha_table = np.empty((n_f, n_g, n_s), dtype=float)

    for i, fname in enumerate(family_names):
        mk_test = family_map[fname]
        for j, sigma in enumerate(sigma_array):
            test = mk_test(float(sigma))
            local_t_max = (
                (12.0 * float(sigma)) if t_max is None else float(t_max)
            )
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
                G = build_twisted_test_state_from_test_function(
                    bundle, test, gfn
                )
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
        idx_flat = int(
            np.argmin(np.where(finite_mask, alpha_table, np.inf))
        )
        i_min, rem = divmod(idx_flat, n_g * n_s)
        k_min, j_min = divmod(rem, n_s)
        alpha_min_sigma = float(sigma_array[j_min])
        alpha_min_family = family_names[i_min]
        alpha_min_gauge = gauge_names[k_min]
    else:
        alpha_min = float("nan")
        alpha_max = float("nan")
        alpha_min_sigma = float("nan")
        alpha_min_family = "<none>"
        alpha_min_gauge = "<none>"

    if finite_mask.any():
        finite_positive = bool(np.all(alpha_table[finite_mask] > 0.0))
    else:
        finite_positive = False
    no_negative_inf = not bool(np.any(np.isneginf(alpha_table)))
    no_nan = not bool(np.any(np.isnan(alpha_table)))
    alpha_all_positive = finite_positive and no_negative_inf and no_nan

    return TwistedAdmissibleFamilySweepCertificate(
        character_name=chi.name,
        character_modulus=int(chi.modulus),
        sigmas=sigma_array,
        families=family_names,
        gauges=gauge_names,
        weil_table=weil_table,
        energy_table=energy_table,
        alpha_table=alpha_table,
        weil_all_positive=weil_all_positive,
        alpha_all_positive=alpha_all_positive,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_min_sigma=alpha_min_sigma,
        alpha_min_family=alpha_min_family,
        alpha_min_gauge=alpha_min_gauge,
    )
