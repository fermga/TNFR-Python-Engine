r"""P40: chi-twisted node-aware gauge sweep for primitive real L(s, chi).

Structural extension of P20 (`nodeaware_gauge_sweep.py`) to primitive
real Dirichlet L-functions.  Combines the admissible-test-function
family sweep (P19, inherited unchanged) with **node-aware** gauges
that depend on the per-node normalised structural frequency
:math:`\hat\nu_f(n)` and the per-node normalised node-weight
:math:`\hat w(n) = \log p / \max_n \log p`.

Whereas P39 swept the canonical scalar-h gauges (P18 family), P40
sweeps gauges of the form

.. math::
   (d, \phi, \epsilon) = g(h(E_n), \hat\nu_f(n), \hat w(n))

over the P34 chi-twisted prime-ladder graph.  The Weil-Guinand
zero-side numerator :math:`W_\chi[\sigma; f]` (P35) is gauge-
independent and is reused unchanged from P39.

EXPERIMENTAL --- Research prototype for the TNFR-Riemann P40 program.
Does NOT prove GRH for any L(s, chi) and does NOT advance G4 = RH.

References
----------
* P20: ``nodeaware_gauge_sweep.py`` (zeta-track template).
* P19: ``admissible_family_sweep.py`` (DEFAULT_TEST_FAMILIES).
* P34: ``twisted_prime_ladder_hamiltonian.py`` (chi-twisted bundle).
* P35: ``twisted_weil_explicit_formula.py`` (W_chi zero side).
* P37: ``twisted_weil_positivity.py`` (canonical alpha_chi bridge).
* P39: ``twisted_admissible_family_sweep.py`` (scalar-gauge variant).
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
from .dirichlet_l import DirichletCharacter
from .nodeaware_gauge_sweep import (
    DEFAULT_NODEAWARE_GAUGES,
    NodeAwareGaugeFn,
)
from .twisted_prime_ladder_hamiltonian import TwistedPrimeLadderHamiltonian
from .twisted_weil_explicit_formula import twisted_weil_zero_side


def _normalized_node_channels(
    bundle: TwistedPrimeLadderHamiltonian,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Return normalised nu_f and log-p channels on the twisted graph."""
    nodes = list(bundle.graph.nodes())
    nu_vals = np.array(
        [float(bundle.graph.nodes[n].get("nu_f", 0.0)) for n in nodes],
        dtype=float,
    )
    w_vals = np.array(
        [math.log(float(n[0])) for n in nodes],
        dtype=float,
    )

    nu_min = float(nu_vals.min())
    nu_max = float(nu_vals.max())
    w_min = float(w_vals.min())
    w_max = float(w_vals.max())

    if nu_max > nu_min:
        nu_hat_vals = (nu_vals - nu_min) / (nu_max - nu_min)
    else:
        nu_hat_vals = np.zeros_like(nu_vals)

    if w_max > w_min:
        w_hat_vals = (w_vals - w_min) / (w_max - w_min)
    else:
        w_hat_vals = np.zeros_like(w_vals)

    nu_hat = {node: float(val) for node, val in zip(nodes, nu_hat_vals)}
    w_hat = {node: float(val) for node, val in zip(nodes, w_hat_vals)}
    return nu_hat, w_hat


def build_twisted_test_state_nodeaware(
    bundle: TwistedPrimeLadderHamiltonian,
    test: AdmissibleTestFunction,
    gauge: NodeAwareGaugeFn,
) -> nx.Graph:
    """Build structural state on the chi-twisted graph.

    For each node ``(p, k)``, computes the scalar test value
    ``h_val = test.h(k * log p)`` and the normalised channels
    ``nu_hat``, ``w_hat``; then evaluates
    ``(d_val, phi_val, epi_val) = gauge(h_val, nu_hat, w_hat)``
    and writes them onto ``dnfr``, ``phase``, ``EPI`` (with phase
    clipped to ``[-pi, pi]``).
    """
    G = bundle.graph.copy()
    nu_hat, w_hat = _normalized_node_channels(bundle)
    pi = math.pi

    for node in G.nodes():
        p, k = node
        e_n = float(k) * math.log(float(p))
        h_val = float(test.h(e_n))
        d_val, phi_val, epi_val = gauge(
            h_val,
            nu_hat[node],
            w_hat[node],
        )
        if phi_val > pi:
            phi_val = pi
        elif phi_val < -pi:
            phi_val = -pi
        G.nodes[node]["dnfr"] = float(d_val)
        G.nodes[node]["phase"] = float(phi_val)
        G.nodes[node]["EPI"] = float(epi_val)
    return G


@dataclass(frozen=True)
class TwistedNodeAwareGaugeSweepCertificate:
    """Outcome of family x node-aware-gauge x sigma alpha_chi sweep."""

    character_name: str
    character_modulus: int
    sigmas: object
    families: tuple[str, ...]
    node_gauges: tuple[str, ...]
    weil_table: object
    energy_table: object
    alpha_table: object
    weil_all_positive: bool
    alpha_all_positive: bool
    alpha_min: float
    alpha_max: float
    alpha_min_sigma: float
    alpha_min_family: str
    alpha_min_node_gauge: str

    def summary(self) -> str:
        return (
            "TwistedNodeAwareGaugeSweepCertificate("
            f"chi='{self.character_name}', q={self.character_modulus}, "
            f"n_sigma={len(self.sigmas)}, "
            f"n_family={len(self.families)}, "
            f"n_node_gauge={len(self.node_gauges)}, "
            f"W_all_positive={self.weil_all_positive}, "
            f"alpha_all_positive={self.alpha_all_positive}, "
            f"alpha_min={self.alpha_min:+.4e} "
            f"@(sigma={self.alpha_min_sigma:.3f}, "
            f"family='{self.alpha_min_family}', "
            f"node_gauge='{self.alpha_min_node_gauge}'), "
            f"alpha_max={self.alpha_max:+.4e})"
        )


def sweep_twisted_nodeaware_gauge(
    chi: DirichletCharacter,
    bundle: TwistedPrimeLadderHamiltonian,
    sigmas: Sequence[float],
    *,
    families: Mapping[str, FamilyFactory] | None = None,
    node_gauges: Mapping[str, NodeAwareGaugeFn] | None = None,
    t_min: float = 0.5,
    t_max: float | None = None,
    initial_step: float = 0.25,
    dps: int = 30,
) -> TwistedNodeAwareGaugeSweepCertificate:
    """Sweep alpha_chi(sigma; f, g) across families and node-aware gauges.

    Parameters
    ----------
    chi
        Primitive real Dirichlet character (P32) used to label and
        validate the certificate.
    bundle
        P34 chi-twisted prime-ladder Hamiltonian bundle.  Must be the
        bundle built for ``chi``.
    sigmas
        Strictly positive widths of the Gaussian envelope ``h_sigma``.
    families
        Mapping name -> family factory.  Defaults to
        ``DEFAULT_TEST_FAMILIES`` (P19: gaussian, gaussian_mixture,
        hermite2_gaussian).
    node_gauges
        Mapping name -> node-aware gauge.  Defaults to
        ``DEFAULT_NODEAWARE_GAUGES`` (P20: nuf_pressure, nuf_phase,
        weight_pressure, mixed_affine).
    t_min, t_max, initial_step, dps
        Forwarded to ``twisted_weil_zero_side``.  When ``t_max`` is
        ``None`` the local cut-off ``12 * sigma`` is used (matching
        P39 convention).

    Returns
    -------
    TwistedNodeAwareGaugeSweepCertificate
        Frozen dataclass with tables W_chi[i, j], E_chi[i, k, j],
        alpha_chi[i, k, j] and global positivity flags.
    """
    sigma_array = np.asarray(list(sigmas), dtype=float)
    if sigma_array.size == 0:
        raise ValueError("sigmas must be non-empty")
    if not np.all(sigma_array > 0.0):
        raise ValueError("every sigma must be strictly positive")

    family_map = (
        dict(families) if families is not None
        else dict(DEFAULT_TEST_FAMILIES)
    )
    if len(family_map) == 0:
        raise ValueError("families must be non-empty")

    node_gauge_map = (
        dict(node_gauges) if node_gauges is not None
        else dict(DEFAULT_NODEAWARE_GAUGES)
    )
    if len(node_gauge_map) == 0:
        raise ValueError("node_gauges must be non-empty")

    family_names = tuple(family_map.keys())
    node_gauge_names = tuple(node_gauge_map.keys())

    n_f = len(family_names)
    n_g = len(node_gauge_names)
    n_s = int(sigma_array.size)

    weil_table = np.empty((n_f, n_s), dtype=float)
    energy_table = np.empty((n_f, n_g, n_s), dtype=float)
    alpha_table = np.empty((n_f, n_g, n_s), dtype=float)

    for i, family_name in enumerate(family_names):
        mk_test = family_map[family_name]
        for j, sigma in enumerate(sigma_array):
            test = mk_test(float(sigma))
            local_t_max = (
                float(t_max) if t_max is not None
                else 12.0 * float(sigma)
            )
            w_total, _n_used, _zeros = twisted_weil_zero_side(
                chi,
                test,
                t_min=float(t_min),
                t_max=local_t_max,
                initial_step=float(initial_step),
                dps=int(dps),
            )
            w_float = float(w_total)
            weil_table[i, j] = w_float
            for k, gauge_name in enumerate(node_gauge_names):
                gfn = node_gauge_map[gauge_name]
                G = build_twisted_test_state_nodeaware(bundle, test, gfn)
                E = float(compute_energy_functional(G))
                energy_table[i, k, j] = E
                if E == 0.0:
                    if w_float > 0.0:
                        alpha_table[i, k, j] = float("inf")
                    elif w_float < 0.0:
                        alpha_table[i, k, j] = float("-inf")
                    else:
                        alpha_table[i, k, j] = float("nan")
                else:
                    alpha_table[i, k, j] = w_float / E

    weil_all_positive = bool(np.all(weil_table >= 0.0))

    finite_mask = np.isfinite(alpha_table)
    if finite_mask.any():
        finite_vals = alpha_table[finite_mask]
        alpha_min = float(finite_vals.min())
        alpha_max = float(finite_vals.max())
        idx_flat = int(np.argmin(
            np.where(finite_mask, alpha_table, np.inf)
        ))
        i_min, rem = divmod(idx_flat, n_g * n_s)
        k_min, j_min = divmod(rem, n_s)
        alpha_min_sigma = float(sigma_array[j_min])
        alpha_min_family = family_names[i_min]
        alpha_min_node_gauge = node_gauge_names[k_min]
    else:
        alpha_min = float("nan")
        alpha_max = float("nan")
        alpha_min_sigma = float("nan")
        alpha_min_family = "<none>"
        alpha_min_node_gauge = "<none>"

    if finite_mask.any():
        finite_positive = bool(np.all(alpha_table[finite_mask] > 0.0))
    else:
        finite_positive = False
    no_negative_inf = not bool(np.any(np.isneginf(alpha_table)))
    no_nan = not bool(np.any(np.isnan(alpha_table)))
    alpha_all_positive = finite_positive and no_negative_inf and no_nan

    return TwistedNodeAwareGaugeSweepCertificate(
        character_name=str(chi.name),
        character_modulus=int(chi.modulus),
        sigmas=sigma_array,
        families=family_names,
        node_gauges=node_gauge_names,
        weil_table=weil_table,
        energy_table=energy_table,
        alpha_table=alpha_table,
        weil_all_positive=weil_all_positive,
        alpha_all_positive=alpha_all_positive,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_min_sigma=alpha_min_sigma,
        alpha_min_family=alpha_min_family,
        alpha_min_node_gauge=alpha_min_node_gauge,
    )


__all__ = [
    "TwistedNodeAwareGaugeSweepCertificate",
    "build_twisted_test_state_nodeaware",
    "sweep_twisted_nodeaware_gauge",
]
