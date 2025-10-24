"""Network operators."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterator
from itertools import islice
from statistics import StatisticsError, fmean
from typing import TYPE_CHECKING, Any

from tnfr import glyph_history

from ..alias import get_attr
from ..constants import DEFAULTS, get_aliases, get_param
from ..helpers.numeric import angle_diff
from ..metrics.trig import neighbor_phase_mean
from ..rng import make_rng
from ..types import EPIValue, Glyph, NodeId, TNFRGraph
from ..utils import get_nodenx
from . import definitions as _definitions
from .jitter import (
    JitterCache,
    JitterCacheManager,
    get_jitter_manager,
    random_jitter,
    reset_jitter_manager,
)
from .registry import OPERATORS, discover_operators, get_operator_class
from .remesh import (
    apply_network_remesh,
    apply_remesh_if_globally_stable,
    apply_topological_remesh,
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
    name: getattr(_definitions, name) for name in getattr(_definitions, "__all__", ())
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
    """Fetch glyph tuning factors for a node.

    The glyph factors expose per-operator coefficients that modulate how an
    operator reorganizes a node's Primary Information Structure (EPI),
    structural frequency (νf), internal reorganization differential (ΔNFR), and
    phase. Missing factors fall back to the canonical defaults stored at the
    graph level.

    Parameters
    ----------
    node : NodeProtocol
        TNFR node providing a ``graph`` mapping where glyph factors may be
        cached under ``"GLYPH_FACTORS"``.

    Returns
    -------
    GlyphFactors
        Mapping with operator-specific coefficients merged with the canonical
        defaults. Mutating the returned mapping does not affect the graph.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self):
    ...         self.graph = {"GLYPH_FACTORS": {"AL_boost": 0.2}}
    >>> node = MockNode()
    >>> factors = get_glyph_factors(node)
    >>> factors["AL_boost"]
    0.2
    >>> factors["EN_mix"]  # Fallback to the default reception mix
    0.25
    """
    return node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"].copy())


def get_factor(gf: GlyphFactors, key: str, default: float) -> float:
    """Return a glyph factor as ``float`` with a default fallback.

    Parameters
    ----------
    gf : GlyphFactors
        Mapping of glyph names to numeric factors.
    key : str
        Factor identifier to look up.
    default : float
        Value used when ``key`` is absent. This typically corresponds to the
        canonical operator tuning and protects structural invariants.

    Returns
    -------
    float
        The resolved factor converted to ``float``.

    Examples
    --------
    >>> get_factor({"AL_boost": 0.3}, "AL_boost", 0.05)
    0.3
    >>> get_factor({}, "IL_dnfr_factor", 0.7)
    0.7
    """
    return float(gf.get(key, default))


# -------------------------
# Glyphs (local operators)
# -------------------------


def get_neighbor_epi(node: NodeProtocol) -> tuple[list[NodeProtocol], EPIValue]:
    """Collect neighbour nodes and their mean EPI.

    The neighbour EPI is used by reception-like glyphs (e.g., EN, RA) to
    harmonise the node's EPI with the surrounding field without mutating νf,
    ΔNFR, or phase. When a neighbour lacks a direct ``EPI`` attribute the
    function resolves it from NetworkX metadata using known aliases.

    Parameters
    ----------
    node : NodeProtocol
        Node whose neighbours participate in the averaging.

    Returns
    -------
    list of NodeProtocol
        Concrete neighbour objects that expose TNFR attributes.
    EPIValue
        Arithmetic mean of the neighbouring EPIs. Equals the node EPI when no
        valid neighbours are found, allowing glyphs to preserve the node state.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, epi, neighbors):
    ...         self.EPI = epi
    ...         self._neighbors = neighbors
    ...         self.graph = {}
    ...     def neighbors(self):
    ...         return self._neighbors
    >>> neigh_a = MockNode(1.0, [])
    >>> neigh_b = MockNode(2.0, [])
    >>> node = MockNode(0.5, [neigh_a, neigh_b])
    >>> neighbors, epi_bar = get_neighbor_epi(node)
    >>> len(neighbors), round(epi_bar, 2)
    (2, 1.5)
    """

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
                v if hasattr(v, "EPI") else NodeNX.from_graph(node.G, v) for v in neigh
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
    """Resolve the dominant ``epi_kind`` across neighbours.

    The dominant kind guides glyphs that synchronise EPI, ensuring that
    reshaping a node's EPI also maintains a coherent semantic label for the
    structural phase space.

    Parameters
    ----------
    neigh : list of NodeProtocol
        Neighbouring nodes providing EPI magnitude and semantic kind.
    default_kind : str
        Fallback label when no neighbour exposes an ``epi_kind``.

    Returns
    -------
    tuple of (str, float)
        The dominant ``epi_kind`` together with the maximum absolute EPI. The
        amplitude assists downstream logic when choosing between the node's own
        label and the neighbour-driven kind.

    Examples
    --------
    >>> class Mock:
    ...     def __init__(self, epi, kind):
    ...         self.EPI = epi
    ...         self.epi_kind = kind
    >>> _determine_dominant([Mock(0.2, "seed"), Mock(-1.0, "pulse")], "seed")
    ('pulse', 1.0)
    """
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
    """Blend node EPI with the neighbour field and update its semantic label.

    The routine is shared by reception-like glyphs. It interpolates between the
    node EPI and the neighbour mean while selecting a dominant ``epi_kind``.
    ΔNFR, νf, and phase remain untouched; the function focuses on reconciling
    form.

    Parameters
    ----------
    node : NodeProtocol
        Node that exposes ``EPI`` and ``epi_kind`` attributes.
    mix : float
        Interpolation weight for the neighbour mean. ``mix = 0`` preserves the
        current EPI, while ``mix = 1`` adopts the average neighbour field.
    default_glyph : Glyph or str
        Glyph driving the mix. Its value informs the fallback ``epi_kind``.

    Returns
    -------
    tuple of (float, str)
        The neighbour mean EPI and the resolved ``epi_kind`` after mixing.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, epi, kind, neighbors):
    ...         self.EPI = epi
    ...         self.epi_kind = kind
    ...         self.graph = {}
    ...         self._neighbors = neighbors
    ...     def neighbors(self):
    ...         return self._neighbors
    >>> neigh = [MockNode(0.8, "wave", []), MockNode(1.2, "wave", [])]
    >>> node = MockNode(0.0, "seed", neigh)
    >>> _, kind = _mix_epi_with_neighbors(node, 0.5, Glyph.EN)
    >>> round(node.EPI, 2), kind
    (0.5, 'wave')
    """
    default_kind = (
        default_glyph.value if isinstance(default_glyph, Glyph) else str(default_glyph)
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
    """Amplify the node EPI via the Emission glyph.

    Emission injects additional coherence into the node by boosting its EPI
    without touching νf, ΔNFR, or phase. The boost amplitude is controlled by
    ``AL_boost``.

    Parameters
    ----------
    node : NodeProtocol
        Node whose EPI is increased.
    gf : GlyphFactors
        Factor mapping used to resolve ``AL_boost``.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, epi):
    ...         self.EPI = epi
    >>> node = MockNode(0.8)
    >>> _op_AL(node, {"AL_boost": 0.2})
    >>> node.EPI
    1.0
    """
    f = get_factor(gf, "AL_boost", 0.05)
    node.EPI = node.EPI + f


def _op_EN(node: NodeProtocol, gf: GlyphFactors) -> None:  # EN — Reception
    """Mix the node EPI with the neighbour field via Reception.

    Reception reorganizes the node's EPI towards the neighbourhood mean while
    choosing a coherent ``epi_kind``. νf, ΔNFR, and phase remain unchanged.

    Parameters
    ----------
    node : NodeProtocol
        Node whose EPI is being reconciled.
    gf : GlyphFactors
        Source of the ``EN_mix`` blending coefficient.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, epi, neighbors):
    ...         self.EPI = epi
    ...         self.epi_kind = "seed"
    ...         self.graph = {}
    ...         self._neighbors = neighbors
    ...     def neighbors(self):
    ...         return self._neighbors
    >>> neigh = [MockNode(1.0, []), MockNode(0.0, [])]
    >>> node = MockNode(0.4, neigh)
    >>> _op_EN(node, {"EN_mix": 0.5})
    >>> round(node.EPI, 2)
    0.7
    """
    mix = get_factor(gf, "EN_mix", 0.25)
    _mix_epi_with_neighbors(node, mix, Glyph.EN)


def _op_IL(node: NodeProtocol, gf: GlyphFactors) -> None:  # IL — Coherence
    """Dampen ΔNFR magnitudes through the Coherence glyph.

    Coherence contracts the internal reorganization differential (ΔNFR) while
    leaving EPI, νf, and phase untouched. The contraction preserves the sign of
    ΔNFR, increasing structural stability.

    Parameters
    ----------
    node : NodeProtocol
        Node whose ΔNFR is being scaled.
    gf : GlyphFactors
        Provides ``IL_dnfr_factor`` controlling the contraction strength.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, dnfr):
    ...         self.dnfr = dnfr
    >>> node = MockNode(0.5)
    >>> _op_IL(node, {"IL_dnfr_factor": 0.2})
    >>> node.dnfr
    0.1
    """
    factor = get_factor(gf, "IL_dnfr_factor", 0.7)
    node.dnfr = factor * getattr(node, "dnfr", 0.0)


def _op_OZ(node: NodeProtocol, gf: GlyphFactors) -> None:  # OZ — Dissonance
    """Excite ΔNFR through the Dissonance glyph.

    Dissonance amplifies ΔNFR or injects jitter, testing the node's stability.
    EPI, νf, and phase remain unaffected while ΔNFR grows to trigger potential
    bifurcations.

    Parameters
    ----------
    node : NodeProtocol
        Node whose ΔNFR is being stressed.
    gf : GlyphFactors
        Supplies ``OZ_dnfr_factor`` and optional noise parameters.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, dnfr):
    ...         self.dnfr = dnfr
    ...         self.graph = {}
    >>> node = MockNode(0.2)
    >>> _op_OZ(node, {"OZ_dnfr_factor": 2.0})
    >>> node.dnfr
    0.4
    """
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
        same = (j is node) or (getattr(node, "n", None) == getattr(j, "n", None))
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
    """Align node phase with neighbours and optionally create links.

    Coupling shifts the node phase ``theta`` towards the neighbour mean while
    respecting νf and EPI. When functional links are enabled it may add edges
    based on combined phase, EPI, and sense-index similarity.

    Parameters
    ----------
    node : NodeProtocol
        Node whose phase is being synchronised.
    gf : GlyphFactors
        Provides ``UM_theta_push`` and optional selection parameters.

    Examples
    --------
    >>> import math
    >>> class MockNode:
    ...     def __init__(self, theta, neighbors):
    ...         self.theta = theta
    ...         self.EPI = 1.0
    ...         self.Si = 0.5
    ...         self.graph = {}
    ...         self._neighbors = neighbors
    ...     def neighbors(self):
    ...         return self._neighbors
    ...     def offset(self):
    ...         return 0
    ...     def all_nodes(self):
    ...         return []
    ...     def has_edge(self, _):
    ...         return False
    ...     def add_edge(self, *_):
    ...         raise AssertionError("not used in example")
    >>> neighbor = MockNode(math.pi / 2, [])
    >>> node = MockNode(0.0, [neighbor])
    >>> _op_UM(node, {"UM_theta_push": 0.5})
    >>> round(node.theta, 2)
    0.79
    """
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
            epi_sim = 1.0 - abs(epi_i - epi_j) / (abs(epi_i) + abs(epi_j) + 1e-9)
            si_sim = 1.0 - abs(si_i - si_j)
            compat = (1 - dphi) * 0.5 + 0.25 * epi_sim + 0.25 * si_sim
            if compat >= thr:
                node.add_edge(j, compat)


def _op_RA(node: NodeProtocol, gf: GlyphFactors) -> None:  # RA — Resonance
    """Diffuse EPI to the node through the Resonance glyph.

    Resonance propagates EPI along existing couplings without affecting νf,
    ΔNFR, or phase. The glyph nudges the node towards the neighbour mean using
    ``RA_epi_diff``.

    Parameters
    ----------
    node : NodeProtocol
        Node harmonising with its neighbourhood.
    gf : GlyphFactors
        Provides ``RA_epi_diff`` as the mixing coefficient.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, epi, neighbors):
    ...         self.EPI = epi
    ...         self.epi_kind = "seed"
    ...         self.graph = {}
    ...         self._neighbors = neighbors
    ...     def neighbors(self):
    ...         return self._neighbors
    >>> neighbor = MockNode(1.0, [])
    >>> node = MockNode(0.2, [neighbor])
    >>> _op_RA(node, {"RA_epi_diff": 0.25})
    >>> round(node.EPI, 2)
    0.4
    """
    diff = get_factor(gf, "RA_epi_diff", 0.15)
    _mix_epi_with_neighbors(node, diff, Glyph.RA)


def _op_SHA(node: NodeProtocol, gf: GlyphFactors) -> None:  # SHA — Silence
    """Reduce νf while preserving EPI, ΔNFR, and phase.

    Silence decelerates a node by scaling νf (structural frequency) towards
    stillness. EPI, ΔNFR, and phase remain unchanged, signalling a temporary
    suspension of structural evolution.

    Parameters
    ----------
    node : NodeProtocol
        Node whose νf is being attenuated.
    gf : GlyphFactors
        Provides ``SHA_vf_factor`` to scale νf.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, vf):
    ...         self.vf = vf
    >>> node = MockNode(1.0)
    >>> _op_SHA(node, {"SHA_vf_factor": 0.5})
    >>> node.vf
    0.5
    """
    factor = get_factor(gf, "SHA_vf_factor", 0.85)
    node.vf = factor * node.vf


factor_val = 1.15
factor_nul = 0.85
_SCALE_FACTORS = {Glyph.VAL: factor_val, Glyph.NUL: factor_nul}


def _op_scale(node: NodeProtocol, factor: float) -> None:
    """Scale νf with the provided factor.

    Parameters
    ----------
    node : NodeProtocol
        Node whose νf is being updated.
    factor : float
        Multiplicative change applied to νf.
    """
    node.vf *= factor


def _make_scale_op(glyph: Glyph) -> GlyphOperation:
    def _op(node: NodeProtocol, gf: GlyphFactors) -> None:
        key = "VAL_scale" if glyph is Glyph.VAL else "NUL_scale"
        default = _SCALE_FACTORS[glyph]
        factor = get_factor(gf, key, default)
        _op_scale(node, factor)

    _op.__doc__ = (
        """{} glyph scales νf to modulate expansion or contraction.

        VAL (expansion) increases νf, whereas NUL (contraction) decreases it.
        EPI, ΔNFR, and phase remain fixed, isolating the change to temporal
        cadence.

        Parameters
        ----------
        node : NodeProtocol
            Node whose νf is updated.
        gf : GlyphFactors
            Provides the respective scale factor (``VAL_scale`` or
            ``NUL_scale``).

        Examples
        --------
        >>> class MockNode:
        ...     def __init__(self, vf):
        ...         self.vf = vf
        >>> node = MockNode(1.0)
        >>> op = _make_scale_op(Glyph.VAL)
        >>> op(node, {{"VAL_scale": 1.5}})
        >>> node.vf
        1.5
        """.format(glyph.name)
    )
    return _op


def _op_THOL(node: NodeProtocol, gf: GlyphFactors) -> None:  # THOL — Self-organization
    """Inject curvature from ``d2EPI`` into ΔNFR to trigger self-organization.

    The glyph keeps EPI, νf, and phase fixed while increasing ΔNFR according to
    the second derivative of EPI, accelerating structural rearrangement.

    Parameters
    ----------
    node : NodeProtocol
        Node contributing ``d2EPI`` to ΔNFR.
    gf : GlyphFactors
        Source of the ``THOL_accel`` multiplier.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, dnfr, curvature):
    ...         self.dnfr = dnfr
    ...         self.d2EPI = curvature
    >>> node = MockNode(0.1, 0.5)
    >>> _op_THOL(node, {"THOL_accel": 0.2})
    >>> node.dnfr
    0.2
    """
    a = get_factor(gf, "THOL_accel", 0.10)
    node.dnfr = node.dnfr + a * getattr(node, "d2EPI", 0.0)


def _op_ZHIR(node: NodeProtocol, gf: GlyphFactors) -> None:  # ZHIR — Mutation
    """Shift phase by a fixed offset to enact mutation.

    Mutation changes the node's phase (θ) while preserving EPI, νf, and ΔNFR.
    The glyph encodes discrete structural transitions between coherent states.

    Parameters
    ----------
    node : NodeProtocol
        Node whose phase is rotated.
    gf : GlyphFactors
        Supplies ``ZHIR_theta_shift`` defining the rotation.

    Examples
    --------
    >>> import math
    >>> class MockNode:
    ...     def __init__(self, theta):
    ...         self.theta = theta
    >>> node = MockNode(0.0)
    >>> _op_ZHIR(node, {"ZHIR_theta_shift": math.pi / 2})
    >>> round(node.theta, 2)
    1.57
    """
    shift = get_factor(gf, "ZHIR_theta_shift", math.pi / 2)
    node.theta = node.theta + shift


def _op_NAV(node: NodeProtocol, gf: GlyphFactors) -> None:  # NAV — Transition
    """Rebalance ΔNFR towards νf while permitting jitter.

    Transition pulls ΔNFR towards a νf-aligned target, optionally adding jitter
    to explore nearby states. EPI and phase remain untouched; νf may be used as
    a reference but is not directly changed.

    Parameters
    ----------
    node : NodeProtocol
        Node whose ΔNFR is redirected.
    gf : GlyphFactors
        Supplies ``NAV_eta`` and ``NAV_jitter`` tuning parameters.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, dnfr, vf):
    ...         self.dnfr = dnfr
    ...         self.vf = vf
    ...         self.graph = {"NAV_RANDOM": False}
    >>> node = MockNode(-0.6, 0.4)
    >>> _op_NAV(node, {"NAV_eta": 0.5, "NAV_jitter": 0.0})
    >>> round(node.dnfr, 2)
    -0.1
    """
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
    """Record an advisory requesting network-scale remeshing.

    REMESH does not change node-level EPI, νf, ΔNFR, or phase. Instead it
    annotates the glyph history so orchestrators can trigger global remesh
    procedures once the stability conditions are met.

    Parameters
    ----------
    node : NodeProtocol
        Node whose history records the advisory.
    gf : GlyphFactors, optional
        Unused but accepted for API symmetry.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self):
    ...         self.graph = {}
    >>> node = MockNode()
    >>> _op_REMESH(node)
    >>> "_remesh_warn_step" in node.graph
    True
    """
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
