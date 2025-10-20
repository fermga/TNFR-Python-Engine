"""Network operators."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, TYPE_CHECKING
import math
import heapq
from itertools import islice
from statistics import fmean, StatisticsError

from ..alias import get_attr
from ..constants import DEFAULTS, get_aliases, get_param

from ..helpers.numeric import angle_diff
from ..metrics.trig import neighbor_phase_mean
from ..utils import get_nodenx
from ..rng import make_rng
from tnfr import glyph_history
from ..types import EPIValue, Glyph, NodeId, TNFRGraph

from . import definitions as _definitions
from .jitter import (
    JitterCache,
    JitterCacheManager,
    get_jitter_manager,
    reset_jitter_manager,
    random_jitter,
)
from .registry import OPERATORS, discover_operators, get_operator_class
from .remesh import (
    apply_network_remesh,
    apply_topological_remesh,
    apply_remesh_if_globally_stable,
)

_remesh_doc = (
    "Trigger a remesh once the stability window is satisfied.\n\n"
    "Parameters\n----------\n"
    "stable_step_window : int | None\n"
    "    Number of consecutive stable steps required before remeshing.\n"
    "    Only the English keyword 'stable_step_window' is supported."
)
if apply_remesh_if_globally_stable.__doc__:
    apply_remesh_if_globally_stable.__doc__ += "\n\n" + _remesh_doc
else:
    apply_remesh_if_globally_stable.__doc__ = _remesh_doc

discover_operators()

_DEFINITION_EXPORTS = {
    name: getattr(_definitions, name)
    for name in getattr(_definitions, "__all__", ())
}
globals().update(_DEFINITION_EXPORTS)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..node import NodeProtocol

GlyphFactors = dict[str, Any]
GlyphOperation = Callable[["NodeProtocol", GlyphFactors], None]

ALIAS_EPI = get_aliases("EPI")

__all__ = [
    "JitterCache",
    "JitterCacheManager",
    "get_jitter_manager",
    "reset_jitter_manager",
    "random_jitter",
    "get_neighbor_epi",
    "get_glyph_factors",
    "GLYPH_OPERATIONS",
    "apply_glyph_obj",
    "apply_glyph",
    "apply_network_remesh",
    "apply_topological_remesh",
    "apply_remesh_if_globally_stable",
    "OPERATORS",
    "discover_operators",
    "get_operator_class",
]

__all__.extend(_DEFINITION_EXPORTS.keys())


def get_glyph_factors(node: NodeProtocol) -> GlyphFactors:
    """Return glyph factors for ``node`` with defaults."""
    return node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"].copy())


def get_factor(gf: GlyphFactors, key: str, default: float) -> float:
    """Return ``gf[key]`` as ``float`` with ``default`` fallback."""
    return float(gf.get(key, default))


# -------------------------
# Glyphs (operadores locales)
# -------------------------


def get_neighbor_epi(node: NodeProtocol) -> tuple[list[NodeProtocol], EPIValue]:
    """Return neighbour list and their mean ``EPI`` without mutating ``node``."""

    epi = node.EPI
    neigh = list(node.neighbors())
    if not neigh:
        return [], epi

    if hasattr(node, "G"):
        G = node.G
        total = 0.0
        count = 0
        has_valid_neighbor = False
        needs_conversion = False
        for v in neigh:
            if hasattr(v, "EPI"):
                total += float(v.EPI)
                has_valid_neighbor = True
            else:
                attr = get_attr(G.nodes[v], ALIAS_EPI, None)
                if attr is not None:
                    total += float(attr)
                    has_valid_neighbor = True
                else:
                    total += float(epi)
                needs_conversion = True
            count += 1
        if not has_valid_neighbor:
            return [], epi
        epi_bar = total / count if count else float(epi)
        if needs_conversion:
            NodeNX = get_nodenx()
            if NodeNX is None:
                raise ImportError("NodeNX is unavailable")
            neigh = [
                v if hasattr(v, "EPI") else NodeNX.from_graph(node.G, v)
                for v in neigh
            ]
    else:
        try:
            epi_bar = fmean(v.EPI for v in neigh)
        except StatisticsError:
            epi_bar = epi

    return neigh, epi_bar


def _determine_dominant(
    neigh: list[NodeProtocol], default_kind: str
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
    node: NodeProtocol, mix: float, default_glyph: Glyph | str
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


def _op_AL(node: NodeProtocol, gf: GlyphFactors) -> None:  # AL — Emission
    f = get_factor(gf, "AL_boost", 0.05)
    node.EPI = node.EPI + f


def _op_EN(node: NodeProtocol, gf: GlyphFactors) -> None:  # EN — Reception
    mix = get_factor(gf, "EN_mix", 0.25)
    _mix_epi_with_neighbors(node, mix, Glyph.EN)


def _op_IL(node: NodeProtocol, gf: GlyphFactors) -> None:  # IL — Coherence
    factor = get_factor(gf, "IL_dnfr_factor", 0.7)
    node.dnfr = factor * getattr(node, "dnfr", 0.0)


def _op_OZ(node: NodeProtocol, gf: GlyphFactors) -> None:  # OZ — Dissonance
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


def _um_candidate_iter(node: NodeProtocol) -> Iterator[NodeProtocol]:
    sample_ids = node.graph.get("_node_sample")
    if sample_ids is not None and hasattr(node, "G"):
        NodeNX = get_nodenx()
        if NodeNX is None:
            raise ImportError("NodeNX is unavailable")
        base = (NodeNX.from_graph(node.G, j) for j in sample_ids)
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
    node: NodeProtocol,
    candidates: Iterator[NodeProtocol],
    limit: int,
    mode: str,
    th: float,
) -> list[NodeProtocol]:
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


def _op_UM(node: NodeProtocol, gf: GlyphFactors) -> None:  # UM — Coupling
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


def _op_RA(node: NodeProtocol, gf: GlyphFactors) -> None:  # RA — Resonance
    diff = get_factor(gf, "RA_epi_diff", 0.15)
    _mix_epi_with_neighbors(node, diff, Glyph.RA)


def _op_SHA(node: NodeProtocol, gf: GlyphFactors) -> None:  # SHA — Silence
    factor = get_factor(gf, "SHA_vf_factor", 0.85)
    node.vf = factor * node.vf


factor_val = 1.15
factor_nul = 0.85
_SCALE_FACTORS = {Glyph.VAL: factor_val, Glyph.NUL: factor_nul}


def _op_scale(node: NodeProtocol, factor: float) -> None:
    node.vf *= factor


def _make_scale_op(glyph: Glyph) -> GlyphOperation:
    def _op(node: NodeProtocol, gf: GlyphFactors) -> None:
        key = "VAL_scale" if glyph is Glyph.VAL else "NUL_scale"
        default = _SCALE_FACTORS[glyph]
        factor = get_factor(gf, key, default)
        _op_scale(node, factor)

    return _op


def _op_THOL(
    node: NodeProtocol, gf: GlyphFactors
) -> None:  # THOL — Self-organization
    a = get_factor(gf, "THOL_accel", 0.10)
    node.dnfr = node.dnfr + a * getattr(node, "d2EPI", 0.0)


def _op_ZHIR(
    node: NodeProtocol, gf: GlyphFactors
) -> None:  # ZHIR — Mutation
    shift = get_factor(gf, "ZHIR_theta_shift", math.pi / 2)
    node.theta = node.theta + shift


def _op_NAV(
    node: NodeProtocol, gf: GlyphFactors
) -> None:  # NAV — Transition
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
    node: NodeProtocol, gf: GlyphFactors | None = None
) -> None:  # REMESH — advisory
    step_idx = glyph_history.current_step_idx(node)
    last_warn = node.graph.get("_remesh_warn_step", None)
    if last_warn != step_idx:
        msg = (
            "REMESH operates at network scale. Use apply_remesh_if_globally_"
            "stable(G) or apply_network_remesh(G)."
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

GLYPH_OPERATIONS: dict[Glyph, GlyphOperation] = {
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
    node: NodeProtocol, glyph: Glyph | str, *, window: int | None = None
) -> None:
    """Apply ``glyph`` to an object satisfying :class:`NodeProtocol`."""

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
                    "msg": f"unknown glyph: {glyph}",
                },
            ),
        )
        raise ValueError(f"unknown glyph: {glyph}")

    op = GLYPH_OPERATIONS.get(g)
    if op is None:
        raise ValueError(f"glyph has no registered operator: {g}")
    if window is None:
        window = int(get_param(node, "GLYPH_HYSTERESIS_WINDOW"))
    gf = get_glyph_factors(node)
    op(node, gf)
    glyph_history.push_glyph(node._glyph_storage(), g.value, window)
    node.epi_kind = g.value


def apply_glyph(
    G: TNFRGraph, n: NodeId, glyph: Glyph | str, *, window: int | None = None
) -> None:
    """Adapter to operate on ``networkx`` graphs."""
    NodeNX = get_nodenx()
    if NodeNX is None:
        raise ImportError("NodeNX is unavailable")
    node = NodeNX(G, n)
    apply_glyph_obj(node, glyph, window=window)
