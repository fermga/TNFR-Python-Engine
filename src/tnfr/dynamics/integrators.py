from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import networkx as nx  # type: ignore[import-untyped]

from ..constants import (
    DEFAULTS,
    get_aliases,
)
from ..gamma import _get_gamma_spec, eval_gamma
from ..alias import get_attr, get_attr_str, set_attr, set_attr_str

ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_DEPI = get_aliases("DEPI")
ALIAS_EPI = get_aliases("EPI")
ALIAS_EPI_KIND = get_aliases("EPI_KIND")
ALIAS_D2EPI = get_aliases("D2EPI")

__all__ = (
    "prepare_integration_params",
    "update_epi_via_nodal_equation",
)


def prepare_integration_params(
    G,
    dt: float | None = None,
    t: float | None = None,
    method: Literal["euler", "rk4"] | None = None,
):
    """Validate and normalise ``dt``, ``t`` and ``method`` for integration.

    Returns ``(dt_step, steps, t0, method)`` where ``dt_step`` is the
    effective step, ``steps`` the number of substeps and ``t0`` the prepared
    initial time.
    """
    if dt is None:
        dt = float(G.graph.get("DT", DEFAULTS["DT"]))
    else:
        if not isinstance(dt, (int, float)):
            raise TypeError("dt must be a number")
        if dt < 0:
            raise ValueError("dt must be non-negative")
        dt = float(dt)

    if t is None:
        t = float(G.graph.get("_t", 0.0))
    else:
        t = float(t)

    method = (
        method
        or G.graph.get(
            "INTEGRATOR_METHOD", DEFAULTS.get("INTEGRATOR_METHOD", "euler")
        )
    ).lower()
    if method not in ("euler", "rk4"):
        raise ValueError("method must be 'euler' or 'rk4'")

    dt_min = float(G.graph.get("DT_MIN", DEFAULTS.get("DT_MIN", 0.0)))
    if dt_min > 0 and dt > dt_min:
        steps = int(math.ceil(dt / dt_min))
    else:
        steps = 1
    # ``steps`` is guaranteed to be ≥1 at this point
    dt_step = dt / steps

    return dt_step, steps, t, method


def _apply_increments(
    G: Any,
    dt_step: float,
    increments: dict[Any, tuple[float, ...]],
    *,
    method: str,
) -> dict[Any, tuple[float, float, float]]:
    """Combine precomputed increments to update node states."""

    new_states: dict[Any, tuple[float, float, float]] = {}
    for n, nd in G.nodes(data=True):
        vf, dnfr, dEPI_dt_prev, epi_i = _node_state(nd)
        ks = increments[n]
        if method == "rk4":
            k1, k2, k3, k4 = ks
            # RK4: EPIₙ₊₁ = EPIᵢ + Δt/6·(k1 + 2k2 + 2k3 + k4)
            epi = epi_i + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            dEPI_dt = k4
        else:
            (k1,) = ks
            # Euler: EPIₙ₊₁ = EPIᵢ + Δt·k1 where k1 = νf·ΔNFR + Γ
            epi = epi_i + dt_step * k1
            dEPI_dt = k1
        d2epi = (dEPI_dt - dEPI_dt_prev) / dt_step if dt_step != 0 else 0.0
        new_states[n] = (epi, dEPI_dt, d2epi)
    return new_states


def _collect_nodal_increments(
    G: Any,
    gamma_maps: tuple[dict[Any, float], ...],
    *,
    method: str,
) -> dict[Any, tuple[float, ...]]:
    """Combine node base state with staged Γ contributions.

    ``gamma_maps`` must contain one entry for Euler integration and four for
    RK4. The helper merges the structural frequency/ΔNFR base contribution
    with the supplied Γ evaluations.
    """

    increments: dict[Any, tuple[float, ...]] = {}
    for n, nd in G.nodes(data=True):
        vf, dnfr, *_ = _node_state(nd)
        base = vf * dnfr
        gammas = [gm.get(n, 0.0) for gm in gamma_maps]

        if method == "rk4":
            if len(gammas) != 4:
                raise ValueError("rk4 integration requires four gamma maps")
            k1, k2, k3, k4 = gammas
            increments[n] = (
                base + k1,
                base + k2,
                base + k3,
                base + k4,
            )
        else:
            if len(gammas) != 1:
                raise ValueError("euler integration requires one gamma map")
            (k1,) = gammas
            increments[n] = (base + k1,)

    return increments


def _build_gamma_increments(
    G: Any,
    dt_step: float,
    t_local: float,
    *,
    method: str,
) -> dict[Any, tuple[float, ...]]:
    """Evaluate Γ contributions and merge them with ``νf·ΔNFR`` base terms."""

    if method == "rk4":
        gamma_count = 4
    elif method == "euler":
        gamma_count = 1
    else:
        raise ValueError("method must be 'euler' or 'rk4'")

    gamma_spec = G.graph.get("_gamma_spec")
    if gamma_spec is None:
        gamma_spec = _get_gamma_spec(G)

    gamma_type = ""
    if isinstance(gamma_spec, Mapping):
        gamma_type = str(gamma_spec.get("type", "")).lower()

    if gamma_type == "none":
        gamma_maps = tuple({} for _ in range(gamma_count))
        return _collect_nodal_increments(G, gamma_maps, method=method)

    if method == "rk4":
        t_mid = t_local + dt_step / 2.0
        t_end = t_local + dt_step
        g1_map = {n: eval_gamma(G, n, t_local) for n in G.nodes}
        g_mid_map = {n: eval_gamma(G, n, t_mid) for n in G.nodes}
        g4_map = {n: eval_gamma(G, n, t_end) for n in G.nodes}
        gamma_maps = (g1_map, g_mid_map, g_mid_map, g4_map)
    else:  # method == "euler"
        gamma_maps = ({n: eval_gamma(G, n, t_local) for n in G.nodes},)

    return _collect_nodal_increments(G, gamma_maps, method=method)


def _integrate_euler(G, dt_step: float, t_local: float):
    """One explicit Euler integration step."""
    increments = _build_gamma_increments(
        G,
        dt_step,
        t_local,
        method="euler",
    )
    return _apply_increments(G, dt_step, increments, method="euler")


def _integrate_rk4(G, dt_step: float, t_local: float):
    """One Runge–Kutta order-4 integration step."""
    increments = _build_gamma_increments(
        G,
        dt_step,
        t_local,
        method="rk4",
    )
    return _apply_increments(G, dt_step, increments, method="rk4")


def update_epi_via_nodal_equation(
    G,
    *,
    dt: float | None = None,
    t: float | None = None,
    method: Literal["euler", "rk4"] | None = None,
) -> None:
    """TNFR nodal equation.

    Implements the extended nodal equation:
        ∂EPI/∂t = νf · ΔNFR(t) + Γi(R)

    Where:
      - EPI is the node's Primary Information Structure.
      - νf is the node's structural frequency (Hz_str).
      - ΔNFR(t) is the nodal gradient (reorganisation need), typically a mix
        of components (e.g. phase θ, EPI, νf).
      - Γi(R) is the optional network coupling as a function of Kuramoto order
        ``R`` (see :mod:`gamma`), used to modulate network integration.

    TNFR references: nodal equation (manual), νf/ΔNFR/EPI glossary, Γ operator.
    Side effects: caches dEPI and updates EPI via explicit integration.
    """
    if not isinstance(
        G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
    ):
        raise TypeError("G must be a networkx graph instance")

    dt_step, steps, t0, method = prepare_integration_params(G, dt, t, method)

    t_local = t0
    for _ in range(steps):
        if method == "rk4":
            updates = _integrate_rk4(G, dt_step, t_local)
        else:
            updates = _integrate_euler(G, dt_step, t_local)

        for n, (epi, dEPI_dt, d2epi) in updates.items():
            nd = G.nodes[n]
            epi_kind = get_attr_str(nd, ALIAS_EPI_KIND, "")
            set_attr(nd, ALIAS_EPI, epi)
            if epi_kind:
                set_attr_str(nd, ALIAS_EPI_KIND, epi_kind)
            set_attr(nd, ALIAS_DEPI, dEPI_dt)
            set_attr(nd, ALIAS_D2EPI, d2epi)

        t_local += dt_step

    G.graph["_t"] = t_local


def _node_state(nd: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return common node state attributes.

    Extracts ``νf``, ``ΔNFR``, previous ``dEPI/dt`` and current ``EPI``
    using alias helpers, providing ``0.0`` defaults when attributes are
    missing.
    """

    vf = get_attr(nd, ALIAS_VF, 0.0)
    dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
    dEPI_dt_prev = get_attr(nd, ALIAS_DEPI, 0.0)
    epi_i = get_attr(nd, ALIAS_EPI, 0.0)
    return vf, dnfr, dEPI_dt_prev, epi_i
