"""Network operators.

Operator helpers interact with TNFR graphs adhering to
:class:`tnfr.types.GraphLike`, relying on ``nodes``/``neighbors`` views,
``number_of_nodes`` and the graph-level ``.graph`` metadata when applying
structural transformations.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterator
from itertools import islice
from statistics import StatisticsError, fmean
from typing import TYPE_CHECKING, Any

from tnfr import glyph_history

from ..alias import get_attr
from ..constants import DEFAULTS, get_param
from ..constants.aliases import ALIAS_EPI
from ..utils import angle_diff
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

from .grammar import apply_glyph_with_grammar  # noqa: E402
from .health_analyzer import SequenceHealthAnalyzer, SequenceHealthMetrics  # noqa: E402
from .hamiltonian import InternalHamiltonian, build_H_coherence, build_H_frequency, build_H_coupling  # noqa: E402

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
    "apply_glyph_with_grammar",
    "apply_network_remesh",
    "apply_topological_remesh",
    "apply_remesh_if_globally_stable",
    "OPERATORS",
    "discover_operators",
    "get_operator_class",
    "SequenceHealthMetrics",
    "SequenceHealthAnalyzer",
    "InternalHamiltonian",
    "build_H_coherence",
    "build_H_frequency",
    "build_H_coupling",
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

    Notes
    -----
    This function performs defensive validation to ensure numeric safety.
    Invalid values (non-numeric, nan, inf) are silently replaced with the
    default to prevent operator failures. For strict validation, use
    ``validate_glyph_factors`` before passing factors to operators.

    Examples
    --------
    >>> get_factor({"AL_boost": 0.3}, "AL_boost", 0.05)
    0.3
    >>> get_factor({}, "IL_dnfr_factor", 0.7)
    0.7
    """
    value = gf.get(key, default)
    # Defensive validation: ensure the value is numeric and finite
    # Use default for invalid values to prevent operator failures
    if not isinstance(value, (int, float, str)):
        return default
    try:
        value = float(value)
    except (ValueError, TypeError):
        return default
    if not math.isfinite(value):
        return default
    return value


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
    _set_epi_with_boundary_check(node, new_epi)
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
    ...         self.graph = {}
    >>> node = MockNode(0.8)
    >>> _op_AL(node, {"AL_boost": 0.2})
    >>> node.EPI <= 1.0  # Bounded by structural_clip
    True
    """
    f = get_factor(gf, "AL_boost", 0.05)
    new_epi = node.EPI + f
    _set_epi_with_boundary_check(node, new_epi)


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


factor_val = 1.05  # Conservative scale prevents EPI overflow near boundaries
factor_nul = 0.85
_SCALE_FACTORS = {Glyph.VAL: factor_val, Glyph.NUL: factor_nul}


def _set_epi_with_boundary_check(
    node: NodeProtocol, new_epi: float, *, apply_clip: bool = True
) -> None:
    """Canonical EPI assignment with structural boundary preservation.
    
    This is the unified function all operators should use when modifying EPI
    to ensure structural boundaries are respected. Provides single point of
    enforcement for TNFR canonical invariant: EPI ∈ [EPI_MIN, EPI_MAX].
    
    Parameters
    ----------
    node : NodeProtocol
        Node whose EPI is being updated
    new_epi : float
        New EPI value to assign
    apply_clip : bool, default True
        If True, applies structural_clip to enforce boundaries.
        If False, assigns value directly (use only when boundaries
        are known to be satisfied, e.g., from edge-aware pre-computation).
        
    Notes
    -----
    TNFR Principle: This function embodies the canonical invariant that EPI
    must remain within structural boundaries. All operator EPI modifications
    should flow through this function to maintain coherence.
    
    The function uses the graph-level configuration for EPI_MIN, EPI_MAX,
    and CLIP_MODE to ensure consistent boundary enforcement across all operators.
    
    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, epi):
    ...         self.EPI = epi
    ...         self.graph = {"EPI_MAX": 1.0, "EPI_MIN": -1.0}
    >>> node = MockNode(0.5)
    >>> _set_epi_with_boundary_check(node, 1.2)  # Will be clipped to 1.0
    >>> float(node.EPI)
    1.0
    """
    from ..dynamics.structural_clip import structural_clip
    
    if not apply_clip:
        node.EPI = new_epi
        return
    
    # Ensure new_epi is float (in case it's a BEPI or other structure)
    new_epi_float = float(new_epi)
    
    # Get boundary configuration from graph (with defensive fallback)
    graph_attrs = getattr(node, 'graph', {})
    epi_min = float(graph_attrs.get("EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)))
    epi_max = float(graph_attrs.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)))
    clip_mode_str = str(graph_attrs.get("CLIP_MODE", "hard"))
    
    # Validate clip mode
    if clip_mode_str not in ("hard", "soft"):
        clip_mode_str = "hard"
    
    # Apply structural boundary preservation
    clipped_epi = structural_clip(
        new_epi_float,
        lo=epi_min,
        hi=epi_max,
        mode=clip_mode_str,  # type: ignore[arg-type]
        record_stats=False,
    )
    
    node.EPI = clipped_epi


def _compute_val_edge_aware_scale(
    epi_current: float, scale: float, epi_max: float, epsilon: float
) -> float:
    """Compute edge-aware scale factor for VAL (Expansion) operator.
    
    Adapts the expansion scale to prevent EPI overflow beyond EPI_MAX.
    When EPI is near the upper boundary, the effective scale is reduced
    to ensure EPI * scale_eff <= EPI_MAX.
    
    Parameters
    ----------
    epi_current : float
        Current EPI value
    scale : float
        Desired expansion scale factor (e.g., VAL_scale = 1.05)
    epi_max : float
        Upper EPI boundary (typically 1.0)
    epsilon : float
        Small value to prevent division by zero (e.g., 1e-12)
        
    Returns
    -------
    float
        Effective scale factor, adapted to respect EPI_MAX boundary
        
    Notes
    -----
    TNFR Principle: This implements "resonance to the edge" - expansion
    scales adaptively to explore volume while respecting structural envelope.
    The adaptation is a dynamic compatibility check, not a fixed constant.
    
    Examples
    --------
    >>> # Normal case: EPI far from boundary
    >>> _compute_val_edge_aware_scale(0.5, 1.05, 1.0, 1e-12)
    1.05
    
    >>> # Edge case: EPI near boundary, scale adapts
    >>> scale = _compute_val_edge_aware_scale(0.96, 1.05, 1.0, 1e-12)
    >>> abs(scale - 1.0417) < 0.001  # Roughly 1.0/0.96
    True
    """
    abs_epi = abs(epi_current)
    if abs_epi < epsilon:
        # EPI near zero, full scale can be applied safely
        return scale
    
    # Compute maximum safe scale that keeps EPI within bounds
    max_safe_scale = epi_max / abs_epi
    
    # Return the minimum of desired scale and safe scale
    return min(scale, max_safe_scale)


def _compute_nul_edge_aware_scale(
    epi_current: float, scale: float, epi_min: float, epsilon: float
) -> float:
    """Compute edge-aware scale factor for NUL (Contraction) operator.
    
    Adapts the contraction scale to prevent EPI underflow below EPI_MIN.
    
    Parameters
    ----------
    epi_current : float
        Current EPI value
    scale : float
        Desired contraction scale factor (e.g., NUL_scale = 0.85)
    epi_min : float
        Lower EPI boundary (typically -1.0)
    epsilon : float
        Small value to prevent division by zero (e.g., 1e-12)
        
    Returns
    -------
    float
        Effective scale factor, adapted to respect EPI_MIN boundary
        
    Notes
    -----
    TNFR Principle: Contraction concentrates structure toward core while
    maintaining coherence.
    
    For typical NUL_scale < 1.0, contraction naturally moves EPI toward zero
    (the center), which is always safe regardless of whether EPI is positive
    or negative. Edge-awareness is only needed if scale could somehow push
    EPI beyond boundaries.
    
    In practice, with NUL_scale = 0.85 < 1.0:
    - Positive EPI contracts toward zero: safe
    - Negative EPI contracts toward zero: safe
    
    Edge-awareness is provided for completeness and future extensibility.
    
    Examples
    --------
    >>> # Normal contraction (always safe with scale < 1.0)
    >>> _compute_nul_edge_aware_scale(0.5, 0.85, -1.0, 1e-12)
    0.85
    >>> _compute_nul_edge_aware_scale(-0.5, 0.85, -1.0, 1e-12)
    0.85
    """
    # With NUL_scale < 1.0, contraction moves toward zero (always safe)
    # No adaptation needed in typical case
    return scale


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
        
        # Always scale νf (existing behavior)
        _op_scale(node, factor)
        
        # Edge-aware EPI scaling (new behavior) if enabled
        edge_aware_enabled = bool(node.graph.get("EDGE_AWARE_ENABLED", DEFAULTS.get("EDGE_AWARE_ENABLED", True)))
        
        if edge_aware_enabled:
            epsilon = float(node.graph.get("EDGE_AWARE_EPSILON", DEFAULTS.get("EDGE_AWARE_EPSILON", 1e-12)))
            epi_min = float(node.graph.get("EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)))
            epi_max = float(node.graph.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)))
            
            epi_current = node.EPI
            
            # Compute edge-aware scale factor
            if glyph is Glyph.VAL:
                scale_eff = _compute_val_edge_aware_scale(epi_current, factor, epi_max, epsilon)
            else:  # Glyph.NUL
                scale_eff = _compute_nul_edge_aware_scale(epi_current, factor, epi_min, epsilon)
            
            # Apply edge-aware EPI scaling with boundary check
            # Edge-aware already computed safe scale, but use unified function
            # for consistency (with apply_clip=True as safety net)
            new_epi = epi_current * scale_eff
            _set_epi_with_boundary_check(node, new_epi, apply_clip=True)
            
            # Record telemetry if scale was adapted
            if abs(scale_eff - factor) > epsilon:
                telemetry = node.graph.setdefault("edge_aware_interventions", [])
                telemetry.append({
                    "glyph": glyph.name if hasattr(glyph, "name") else str(glyph),
                    "epi_before": epi_current,
                    "epi_after": float(node.EPI),  # Get actual value after boundary check
                    "scale_requested": factor,
                    "scale_effective": scale_eff,
                    "adapted": True,
                })

    _op.__doc__ = """{} glyph scales νf and EPI with edge-aware adaptation.

        VAL (expansion) increases νf and EPI, whereas NUL (contraction) decreases them.
        Edge-aware scaling adapts the scale factor near EPI boundaries to prevent
        overflow/underflow, maintaining structural coherence within [-1.0, 1.0].

        When EDGE_AWARE_ENABLED is True (default), the effective scale is computed as:
        - VAL: scale_eff = min(VAL_scale, EPI_MAX / |EPI_current|)
        - NUL: scale_eff = min(NUL_scale, |EPI_MIN| / |EPI_current|) for negative EPI

        This implements TNFR principle: "resonance to the edge" without breaking
        the structural envelope. Telemetry records adaptation events.

        Parameters
        ----------
        node : NodeProtocol
            Node whose νf and EPI are updated.
        gf : GlyphFactors
            Provides the respective scale factor (``VAL_scale`` or
            ``NUL_scale``).

        Examples
        --------
        >>> class MockNode:
        ...     def __init__(self, vf, epi):
        ...         self.vf = vf
        ...         self.EPI = epi
        ...         self.graph = {{"EDGE_AWARE_ENABLED": True, "EPI_MAX": 1.0}}
        >>> node = MockNode(1.0, 0.96)
        >>> op = _make_scale_op(Glyph.VAL)
        >>> op(node, {{"VAL_scale": 1.05}})
        >>> node.vf  # νf scaled normally
        1.05
        >>> node.EPI <= 1.0  # EPI kept within bounds
        True
        """.format(
        glyph.name
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

    from .grammar import function_name_to_glyph
    from ..validation.input_validation import ValidationError, validate_glyph

    # Validate glyph parameter
    try:
        if not isinstance(glyph, Glyph):
            validated_glyph = validate_glyph(glyph)
            glyph = (
                validated_glyph.value
                if isinstance(validated_glyph, Glyph)
                else str(glyph)
            )
        else:
            glyph = glyph.value
    except ValidationError as e:
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
                    "msg": f"invalid glyph: {e}",
                },
            ),
        )
        raise ValueError(f"invalid glyph: {e}") from e

    # Try direct glyph code first
    try:
        g = Glyph(str(glyph))
    except ValueError:
        # Try structural function name mapping
        g = function_name_to_glyph(glyph)
        if g is None:
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
    from ..validation.input_validation import (
        ValidationError,
        validate_node_id,
        validate_tnfr_graph,
    )

    # Validate graph and node parameters
    try:
        validate_tnfr_graph(G)
        validate_node_id(n)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for apply_glyph: {e}") from e

    NodeNX = get_nodenx()
    if NodeNX is None:
        raise ImportError("NodeNX is unavailable")
    node = NodeNX(G, n)
    apply_glyph_obj(node, glyph, window=window)
