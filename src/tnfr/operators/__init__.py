"""Network operators."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING, Callable
import math
import heapq
from itertools import islice

from ..constants import DEFAULTS, get_aliases, get_param
from ..helpers.numeric import (
    list_mean,
    angle_diff,
    neighbor_mean,
)
from ..metrics.trig import neighbor_phase_mean
from ..import_utils import get_nodonx
from ..rng import make_rng
from tnfr import glyph_history
from ..types import Glyph

from .jitter import (
    JitterCache,
    JitterCacheManager,
    get_jitter_manager,
    reset_jitter_manager,
    random_jitter,
    _get_jitter_cache,
)
from .remesh import (
    apply_network_remesh,
    apply_topological_remesh,
    apply_remesh_if_globally_stable,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..node import NodoProtocol

ALIAS_EPI = get_aliases("EPI")

__all__ = [
    "JitterCache",
    "JitterCacheManager",
    "get_jitter_manager",
    "reset_jitter_manager",
    "random_jitter",
    "_get_jitter_cache",
    "get_neighbor_epi",
    "get_glyph_factors",
    "GLYPH_OPERATIONS",
    "apply_glyph_obj",
    "apply_glyph",
    "apply_network_remesh",
    "apply_topological_remesh",
    "apply_remesh_if_globally_stable",
]


def get_glyph_factors(node: NodoProtocol) -> dict[str, Any]:
    """Return glyph factors for ``node`` with defaults."""
    return node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"].copy())


def get_factor(gf: dict[str, Any], key: str, default: float) -> float:
    """Return ``gf[key]`` as ``float`` with ``default`` fallback."""
    return float(gf.get(key, default))


# -------------------------
# Glyphs (operadores locales)
# -------------------------


def _any_neighbor_has(node: NodoProtocol, aliases: tuple[str, ...]) -> bool:
    """Return ``True`` if any neighbour defines one of ``aliases``."""
    if hasattr(node, "G"):
        G = node.G
        return any(
            any(a in G.nodes[v] for a in aliases) for v in G.neighbors(node.n)
        )
    return any(
        any(hasattr(neigh, a) for a in aliases) for neigh in node.neighbors()
    )


def get_neighbor_epi(node: NodoProtocol) -> tuple[list[NodoProtocol], float]:
    """Return neighbour list and their mean ``EPI`` without mutating ``node``."""

    epi = node.EPI
    neigh = list(node.neighbors())
    if not neigh:
        return [], epi

    if hasattr(node, "G"):
        if not _any_neighbor_has(node, ALIAS_EPI):
            return [], epi
        epi_bar = neighbor_mean(node.G, node.n, ALIAS_EPI, default=epi)
        NodoNX = get_nodonx()
        if NodoNX is None:
            raise ImportError("NodoNX is unavailable")
        neigh = [
            v if hasattr(v, "EPI") else NodoNX.from_graph(node.G, v)
            for v in neigh
        ]
    else:
        epi_bar = list_mean((v.EPI for v in neigh), default=epi)

    return neigh, epi_bar


def _determine_dominant(
    neigh: list[NodoProtocol], default_kind: str
) -> tuple[str, float]:
    """Return dominant ``epi_kind`` among ``neigh`` and its absolute ``EPI``."""
    best_kind: str | None = None
    best_abs = 0.0
    for v in neigh:
        abs_v = abs(v.EPI)
        if abs_v > best_abs:
            best_abs = abs_v
            best_kind = v.epi_kind
    if not best_kind:
        return default_kind, 0.0
    return best_kind, best_abs


def _mix_epi_with_neighbors(
    node: NodoProtocol, mix: float, default_glyph: Glyph | str
) -> tuple[float, str]:
    """Mix ``EPI`` of ``node`` with the mean of its neighbours."""
    default_kind = (
        default_glyph.value
        if isinstance(default_glyph, Glyph)
        else str(default_glyph)
    )
    epi = node.EPI
    neigh, epi_bar = get_neighbor_epi(node)

    if not neigh:
        node.epi_kind = default_kind
        return epi, default_kind

    dominant, best_abs = _determine_dominant(neigh, default_kind)
    new_epi = (1 - mix) * epi + mix * epi_bar
    node.EPI = new_epi
    final = dominant if best_abs > abs(new_epi) else node.epi_kind
    if not final:
        final = default_kind
    node.epi_kind = final
    return epi_bar, final


def _op_AL(node: NodoProtocol, gf: dict[str, Any]) -> None:  # AL — Emisión
    f = get_factor(gf, "AL_boost", 0.05)
    node.EPI = node.EPI + f


def _op_EN(node: NodoProtocol, gf: dict[str, Any]) -> None:  # EN — Recepción
    mix = get_factor(gf, "EN_mix", 0.25)
    _mix_epi_with_neighbors(node, mix, Glyph.EN)


def _op_IL(node: NodoProtocol, gf: dict[str, Any]) -> None:  # IL — Coherencia
    factor = get_factor(gf, "IL_dnfr_factor", 0.7)
    node.dnfr = factor * getattr(node, "dnfr", 0.0)


def _op_OZ(node: NodoProtocol, gf: dict[str, Any]) -> None:  # OZ — Disonancia
    factor = get_factor(gf, "OZ_dnfr_factor", 1.3)
    dnfr = getattr(node, "dnfr", 0.0)
    if bool(node.graph.get("OZ_NOISE_MODE", False)):
        sigma = float(node.graph.get("OZ_SIGMA", 0.1))
        if sigma <= 0:
            node.dnfr = dnfr
            return
        node.dnfr = dnfr + random_jitter(node, sigma)
    else:
        node.dnfr = factor * dnfr if abs(dnfr) > 1e-9 else 0.1


def _um_candidate_iter(node: NodoProtocol):
    sample_ids = node.graph.get("_node_sample")
    if sample_ids is not None and hasattr(node, "G"):
        NodoNX = get_nodonx()
        if NodoNX is None:
            raise ImportError("NodoNX is unavailable")
        base = (NodoNX.from_graph(node.G, j) for j in sample_ids)
    else:
        base = node.all_nodes()
    for j in base:
        same = (j is node) or (
            getattr(node, "n", None) == getattr(j, "n", None)
        )
        if same or node.has_edge(j):
            continue
        yield j


def _um_select_candidates(
    node: NodoProtocol,
    candidates,
    limit: int,
    mode: str,
    th: float,
):
    """Select a subset of ``candidates`` for UM coupling."""
    rng = make_rng(int(node.graph.get("RANDOM_SEED", 0)), node.offset(), node.G)

    if limit <= 0:
        return list(candidates)

    if mode == "proximity":
        return heapq.nsmallest(
            limit, candidates, key=lambda j: abs(angle_diff(j.theta, th))
        )

    reservoir = list(islice(candidates, limit))
    for i, cand in enumerate(candidates, start=limit):
        j = rng.randint(0, i)
        if j < limit:
            reservoir[j] = cand

    if mode == "sample":
        rng.shuffle(reservoir)

    return reservoir


def _op_UM(node: NodoProtocol, gf: dict[str, Any]) -> None:  # UM — Coupling
    k = get_factor(gf, "UM_theta_push", 0.25)
    th = node.theta
    thL = neighbor_phase_mean(node)
    d = angle_diff(thL, th)
    node.theta = th + k * d

    if bool(node.graph.get("UM_FUNCTIONAL_LINKS", False)):
        thr = float(
            node.graph.get(
                "UM_COMPAT_THRESHOLD",
                DEFAULTS.get("UM_COMPAT_THRESHOLD", 0.75),
            )
        )
        epi_i = node.EPI
        si_i = node.Si

        limit = int(node.graph.get("UM_CANDIDATE_COUNT", 0))
        mode = str(node.graph.get("UM_CANDIDATE_MODE", "sample")).lower()
        candidates = _um_select_candidates(
            node, _um_candidate_iter(node), limit, mode, th
        )

        for j in candidates:
            th_j = j.theta
            dphi = abs(angle_diff(th_j, th)) / math.pi
            epi_j = j.EPI
            si_j = j.Si
            epi_sim = 1.0 - abs(epi_i - epi_j) / (
                abs(epi_i) + abs(epi_j) + 1e-9
            )
            si_sim = 1.0 - abs(si_i - si_j)
            compat = (1 - dphi) * 0.5 + 0.25 * epi_sim + 0.25 * si_sim
            if compat >= thr:
                node.add_edge(j, compat)


def _op_RA(node: NodoProtocol, gf: dict[str, Any]) -> None:  # RA — Resonancia
    diff = get_factor(gf, "RA_epi_diff", 0.15)
    _mix_epi_with_neighbors(node, diff, Glyph.RA)


def _op_SHA(node: NodoProtocol, gf: dict[str, Any]) -> None:  # SHA — Silencio
    factor = get_factor(gf, "SHA_vf_factor", 0.85)
    node.vf = factor * node.vf


factor_val = 1.15
factor_nul = 0.85
_SCALE_FACTORS = {Glyph.VAL: factor_val, Glyph.NUL: factor_nul}


def _op_scale(node: NodoProtocol, glyph: Glyph, factor: float) -> None:
    if glyph is Glyph.VAL:
        node.vf *= factor
    else:
        node.vf *= factor


def _make_scale_op(glyph: Glyph):
    def _op(node: NodoProtocol, gf: dict[str, Any]) -> None:
        factor_val = get_factor(gf, "VAL_scale", _SCALE_FACTORS[Glyph.VAL])
        factor_nul = get_factor(gf, "NUL_scale", _SCALE_FACTORS[Glyph.NUL])
        factor = factor_val if glyph is Glyph.VAL else factor_nul
        _op_scale(node, glyph, factor)

    return _op


def _op_THOL(
    node: NodoProtocol, gf: dict[str, Any]
) -> None:  # THOL — Autoorganización
    a = get_factor(gf, "THOL_accel", 0.10)
    node.dnfr = node.dnfr + a * getattr(node, "d2EPI", 0.0)


def _op_ZHIR(
    node: NodoProtocol, gf: dict[str, Any]
) -> None:  # ZHIR — Mutación
    shift = get_factor(gf, "ZHIR_theta_shift", math.pi / 2)
    node.theta = node.theta + shift


def _op_NAV(
    node: NodoProtocol, gf: dict[str, Any]
) -> None:  # NAV — Transición
    dnfr = node.dnfr
    vf = node.vf
    eta = get_factor(gf, "NAV_eta", 0.5)
    strict = bool(node.graph.get("NAV_STRICT", False))
    if strict:
        base = vf
    else:
        sign = 1.0 if dnfr >= 0 else -1.0
        target = sign * vf
        base = (1.0 - eta) * dnfr + eta * target
    j = get_factor(gf, "NAV_jitter", 0.05)
    if bool(node.graph.get("NAV_RANDOM", True)):
        jitter = random_jitter(node, j)
    else:
        jitter = j * (1 if base >= 0 else -1)
    node.dnfr = base + jitter


def _op_REMESH(
    node: NodoProtocol, gf: dict[str, Any] | None = None
) -> None:  # REMESH — aviso
    step_idx = glyph_history.current_step_idx(node)
    last_warn = node.graph.get("_remesh_warn_step", None)
    if last_warn != step_idx:
        msg = (
            "REMESH es a escala de red. Usa apply_remesh_if_globally_"
            "stable(G) o apply_network_remesh(G)."
        )
        hist = glyph_history.ensure_history(node)
        glyph_history.append_metric(
            hist,
            "events",
            ("warn", {"step": step_idx, "node": None, "msg": msg}),
        )
        node.graph["_remesh_warn_step"] = step_idx
    return


# -------------------------
# Dispatcher
# -------------------------

GLYPH_OPERATIONS: dict[Glyph, Callable[["NodoProtocol", dict[str, Any]], None]] = {
    Glyph.AL: _op_AL,
    Glyph.EN: _op_EN,
    Glyph.IL: _op_IL,
    Glyph.OZ: _op_OZ,
    Glyph.UM: _op_UM,
    Glyph.RA: _op_RA,
    Glyph.SHA: _op_SHA,
    Glyph.VAL: _make_scale_op(Glyph.VAL),
    Glyph.NUL: _make_scale_op(Glyph.NUL),
    Glyph.THOL: _op_THOL,
    Glyph.ZHIR: _op_ZHIR,
    Glyph.NAV: _op_NAV,
    Glyph.REMESH: _op_REMESH,
}


def apply_glyph_obj(
    node: NodoProtocol, glyph: Glyph | str, *, window: int | None = None
) -> None:
    """Apply ``glyph`` to an object satisfying :class:`NodoProtocol`."""

    try:
        g = glyph if isinstance(glyph, Glyph) else Glyph(str(glyph))
    except ValueError:
        step_idx = glyph_history.current_step_idx(node)
        hist = glyph_history.ensure_history(node)
        glyph_history.append_metric(
            hist,
            "events",
            (
                "warn",
                {
                    "step": step_idx,
                    "node": getattr(node, "n", None),
                    "msg": f"glyph desconocido: {glyph}",
                },
            ),
        )
        raise ValueError(f"glyph desconocido: {glyph}")

    op = GLYPH_OPERATIONS.get(g)
    if op is None:
        raise ValueError(f"glyph sin operador: {g}")
    if window is None:
        window = int(get_param(node, "GLYPH_HYSTERESIS_WINDOW"))
    gf = get_glyph_factors(node)
    op(node, gf)
    glyph_history.push_glyph(node._glyph_storage(), g.value, window)
    node.epi_kind = g.value


def apply_glyph(
    G, n, glyph: Glyph | str, *, window: int | None = None
) -> None:
    """Adapter to operate on ``networkx`` graphs."""
    NodoNX = get_nodonx()
    if NodoNX is None:
        raise ImportError("NodoNX is unavailable")
    node = NodoNX(G, n)
    apply_glyph_obj(node, glyph, window=window)
