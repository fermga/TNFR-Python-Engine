r"""P20: node-aware gauge sweep (\nu_f + node-weight dependent).

This module extends P19 by enriching the gauge family with node-aware
maps that depend on local structural frequency and node weight, not only
on the scalar test profile value h(E_n).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import networkx as nx

from ..mathematics.unified_numerical import np
from ..physics.conservation import compute_energy_functional
from .admissible_family_sweep import (
    DEFAULT_TEST_FAMILIES,
    AdmissibleTestFunction,
    FamilyFactory,
)
from .prime_ladder_hamiltonian import PrimeLadderHamiltonian
from .weil_explicit_formula import weil_zero_side


NodeAwareGaugeFn = Callable[[float, float, float], tuple[float, float, float]]


def _gauge_nuf_pressure(
    h: float,
    nu_hat: float,
    w_hat: float,
) -> tuple[float, float, float]:
    """Boost pressure channel with normalized frequency."""
    del w_hat
    return (h * (1.0 + 0.60 * nu_hat), h, h)


def _gauge_nuf_phase(
    h: float,
    nu_hat: float,
    w_hat: float,
) -> tuple[float, float, float]:
    """Twist phase channel with normalized frequency."""
    del w_hat
    return (h, h * (1.0 + 0.80 * nu_hat), 1.0)


def _gauge_weight_pressure(
    h: float,
    nu_hat: float,
    w_hat: float,
) -> tuple[float, float, float]:
    """Boost pressure with normalized node-weight (log p)."""
    del nu_hat
    return (h * (1.0 + 0.75 * w_hat), h, h)


def _gauge_mixed_affine(
    h: float,
    nu_hat: float,
    w_hat: float,
) -> tuple[float, float, float]:
    """Mixed affine gauge combining nu_f and node-weight channels."""
    d_val = h * (1.0 + 0.35 * nu_hat + 0.35 * w_hat)
    phi_val = h * (1.0 + 0.30 * nu_hat)
    epi_val = h * (1.0 + 0.20 * w_hat)
    return (d_val, phi_val, epi_val)


DEFAULT_NODEAWARE_GAUGES: Mapping[str, NodeAwareGaugeFn] = {
    "nuf_pressure": _gauge_nuf_pressure,
    "nuf_phase": _gauge_nuf_phase,
    "weight_pressure": _gauge_weight_pressure,
    "mixed_affine": _gauge_mixed_affine,
}


def _normalized_node_channels(
    bundle: PrimeLadderHamiltonian,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Return normalized nu_f and node-weight channels in [0, 1]."""
    nodes = list(bundle.graph.nodes())
    nu_vals = np.array([
        float(bundle.graph.nodes[n].get("nu_f", 0.0)) for n in nodes
    ], dtype=float)
    w_vals = np.array([
        math.log(float(n[0])) for n in nodes
    ], dtype=float)

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


def build_test_state_nodeaware(
    bundle: PrimeLadderHamiltonian,
    test: AdmissibleTestFunction,
    gauge: NodeAwareGaugeFn,
) -> nx.Graph:
    """Build structural state using h(E_n), nu_hat(node), w_hat(node)."""
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
class NodeAwareGaugeSweepCertificate:
    """Outcome of family × node-aware-gauge × sigma alpha sweep."""

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
            "NodeAwareGaugeSweepCertificate("
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


def sweep_alpha_nodeaware(
    bundle: PrimeLadderHamiltonian,
    sigmas: Sequence[float],
    *,
    families: Mapping[str, FamilyFactory] | None = None,
    node_gauges: Mapping[str, NodeAwareGaugeFn] | None = None,
    n_zeros: int = 60,
    convergence_tol: float = 1e-12,
    max_zeros: int = 200,
) -> NodeAwareGaugeSweepCertificate:
    """Sweep alpha over sigma, admissible families, and node-aware gauges."""
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

    node_gauge_map = (
        dict(node_gauges)
        if node_gauges is not None
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
            w_val, _n_used = weil_zero_side(
                test,
                n_zeros=n_zeros,
                convergence_tol=convergence_tol,
                max_zeros=max_zeros,
            )
            w_float = float(w_val)
            weil_table[i, j] = w_float
            for k, gauge_name in enumerate(node_gauge_names):
                gfn = node_gauge_map[gauge_name]
                G = build_test_state_nodeaware(bundle, test, gfn)
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
        idx_flat = int(np.argmin(np.where(finite_mask, alpha_table, np.inf)))
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

    return NodeAwareGaugeSweepCertificate(
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
    "NodeAwareGaugeFn",
    "DEFAULT_NODEAWARE_GAUGES",
    "build_test_state_nodeaware",
    "NodeAwareGaugeSweepCertificate",
    "sweep_alpha_nodeaware",
]
