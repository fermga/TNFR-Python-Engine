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
from ..constants.aliases import ALIAS_EPI, ALIAS_VF
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
from .hamiltonian import (
    InternalHamiltonian,
    build_H_coherence,
    build_H_frequency,
    build_H_coupling,
)  # noqa: E402
from .pattern_detection import (  # noqa: E402
    PatternMatch,
    UnifiedPatternDetector,
    detect_pattern,
    analyze_sequence,
)

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
    # Pattern detection (unified module)
    "PatternMatch",
    "UnifiedPatternDetector",
    "detect_pattern",
    "analyze_sequence",
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
            neigh = [v if hasattr(v, "EPI") else NodeNX.from_graph(node.G, v) for v in neigh]
    else:
        try:
            epi_bar = fmean(v.EPI for v in neigh)
        except StatisticsError:
            epi_bar = epi

    return neigh, epi_bar


def _determine_dominant(neigh: list[NodeProtocol], default_kind: str) -> tuple[str, float]:
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
    default_kind = default_glyph.value if isinstance(default_glyph, Glyph) else str(default_glyph)
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
        return heapq.nsmallest(limit, candidates, key=lambda j: abs(angle_diff(j.theta, th)))

    reservoir = list(islice(candidates, limit))
    for i, cand in enumerate(candidates, start=limit):
        j = rng.randint(0, i)
        if j < limit:
            reservoir[j] = cand

    if mode == "sample":
        rng.shuffle(reservoir)

    return reservoir


def compute_consensus_phase(phases: list[float]) -> float:
    """Compute circular mean (consensus phase) from a list of phase angles.

    This function calculates the consensus phase using the circular mean
    formula: arctan2(mean(sin), mean(cos)). This ensures proper handling
    of phase wrapping at ±π boundaries.

    Parameters
    ----------
    phases : list[float]
        List of phase angles in radians.

    Returns
    -------
    float
        Consensus phase angle in radians, in the range [-π, π).

    Notes
    -----
    The consensus phase represents the central tendency of a set of angular
    values, accounting for the circular nature of phase space. This is
    critical for bidirectional phase synchronization in the UM operator.

    Examples
    --------
    >>> import math
    >>> phases = [0.0, math.pi/2, math.pi]
    >>> result = compute_consensus_phase(phases)
    >>> -math.pi <= result < math.pi
    True
    """
    if not phases:
        return 0.0

    cos_sum = sum(math.cos(ph) for ph in phases)
    sin_sum = sum(math.sin(ph) for ph in phases)
    return math.atan2(sin_sum, cos_sum)


def _op_UM(node: NodeProtocol, gf: GlyphFactors) -> None:  # UM — Coupling
    """Align node phase and frequency with neighbours and optionally create links.

    Coupling shifts the node phase ``theta`` towards the neighbour mean while
    respecting νf and EPI. When bidirectional mode is enabled (default), both
    the node and its neighbors synchronize their phases mutually. Additionally,
    structural frequency (νf) synchronization causes coupled nodes to converge
    their reorganization rates. Coupling also reduces ΔNFR through mutual
    stabilization, decreasing reorganization pressure proportional to phase
    alignment strength. When functional links are enabled it may add edges
    based on combined phase, EPI, and sense-index similarity.

    Parameters
    ----------
    node : NodeProtocol
        Node whose phase and frequency are being synchronised.
    gf : GlyphFactors
        Provides ``UM_theta_push``, ``UM_vf_sync``, ``UM_dnfr_reduction`` and
        optional selection parameters.

    Notes
    -----
    Bidirectional synchronization (UM_BIDIRECTIONAL=True, default) implements
    the canonical TNFR requirement φᵢ(t) ≈ φⱼ(t) by mutually adjusting phases
    of both the node and its neighbors towards a consensus phase. This ensures
    true coupling as defined in the theory.

    Structural frequency synchronization (UM_SYNC_VF=True, default) implements
    the TNFR requirement that coupling synchronizes not only phases but also
    structural frequencies (νf). This enables coupled nodes to converge their
    reorganization rates, which is essential for sustained resonance and coherent
    network evolution as described by the nodal equation: ∂EPI/∂t = νf · ΔNFR(t).

    ΔNFR stabilization (UM_STABILIZE_DNFR=True, default) implements the canonical
    effect where coupling reduces reorganization pressure through mutual stabilization.
    The reduction is proportional to phase alignment: well-coupled nodes (high phase
    alignment) experience stronger ΔNFR reduction, promoting structural coherence.

    Legacy unidirectional mode (UM_BIDIRECTIONAL=False) only adjusts the node's
    phase towards its neighbors, preserving backward compatibility.

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
    k_vf = get_factor(gf, "UM_vf_sync", 0.10)
    th_i = node.theta

    # Check if bidirectional synchronization is enabled (default: True)
    bidirectional = bool(node.graph.get("UM_BIDIRECTIONAL", True))

    if bidirectional:
        # Bidirectional mode: mutually synchronize node and neighbors
        neighbor_ids = list(node.neighbors())
        if neighbor_ids:
            # Get NodeNX wrapper for accessing neighbor attributes
            NodeNX = get_nodenx()
            if NodeNX is None or not hasattr(node, "G"):
                # Fallback to unidirectional if NodeNX unavailable
                thL = neighbor_phase_mean(node)
                d = angle_diff(thL, th_i)
                node.theta = th_i + k * d
            else:
                # Wrap neighbor IDs to access theta attribute
                neighbors = [NodeNX.from_graph(node.G, nid) for nid in neighbor_ids]

                # Collect all phases (node + neighbors)
                phases = [th_i] + [n.theta for n in neighbors]
                target_phase = compute_consensus_phase(phases)

                # Adjust node phase towards consensus
                node.theta = th_i + k * angle_diff(target_phase, th_i)

                # Adjust neighbor phases towards consensus
                for neighbor in neighbors:
                    th_j = neighbor.theta
                    neighbor.theta = th_j + k * angle_diff(target_phase, th_j)
    else:
        # Legacy unidirectional mode: only adjust node towards neighbors
        thL = neighbor_phase_mean(node)
        d = angle_diff(thL, th_i)
        node.theta = th_i + k * d

    # Structural frequency (νf) synchronization
    # According to TNFR theory, coupling synchronizes both phase and frequency
    sync_vf = bool(node.graph.get("UM_SYNC_VF", True))
    if sync_vf:
        neighbor_ids = list(node.neighbors())
        if neighbor_ids and hasattr(node, "G"):
            # Canonical access to vf through alias system
            vf_i = node.vf
            vf_neighbors = [get_attr(node.G.nodes[nid], ALIAS_VF, 0.0) for nid in neighbor_ids]

            if vf_neighbors:
                vf_mean = sum(vf_neighbors) / len(vf_neighbors)

                # Gradual convergence towards mean (similar to phase sync)
                node.vf = vf_i + k_vf * (vf_mean - vf_i)

    # ΔNFR reduction by mutual stabilization
    # Coupling produces a stabilizing effect that reduces reorganization pressure
    stabilize_dnfr = bool(node.graph.get("UM_STABILIZE_DNFR", True))

    if stabilize_dnfr:
        k_dnfr = get_factor(gf, "UM_dnfr_reduction", 0.15)

        # Calculate compatibility with neighbors based on phase alignment
        neighbor_ids = list(node.neighbors())
        if neighbor_ids:
            # Get NodeNX wrapper for accessing neighbor attributes
            NodeNX = get_nodenx()
            if NodeNX is not None and hasattr(node, "G"):
                neighbors = [NodeNX.from_graph(node.G, nid) for nid in neighbor_ids]

                # Compute phase alignments with each neighbor
                phase_alignments = []
                # Compute phase alignment using canonical formula
                from ..metrics.phase_compatibility import compute_phase_coupling_strength

                for neighbor in neighbors:
                    alignment = compute_phase_coupling_strength(node.theta, neighbor.theta)
                    phase_alignments.append(alignment)

                # Mean alignment represents coupling strength
                mean_alignment = sum(phase_alignments) / len(phase_alignments)

                # Reduce ΔNFR proportionally to coupling strength
                # reduction_factor < 1.0 when well-coupled (high alignment)
                reduction_factor = 1.0 - (k_dnfr * mean_alignment)
                node.dnfr = node.dnfr * reduction_factor

    if bool(node.graph.get("UM_FUNCTIONAL_LINKS", True)):
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
        candidates = _um_select_candidates(node, _um_candidate_iter(node), limit, mode, th_i)

        # Use canonical phase coupling strength formula
        from ..metrics.phase_compatibility import compute_phase_coupling_strength

        for j in candidates:
            phase_coupling = compute_phase_coupling_strength(th_i, j.theta)

            epi_j = j.EPI
            si_j = j.Si
            epi_sim = 1.0 - abs(epi_i - epi_j) / (abs(epi_i) + abs(epi_j) + 1e-9)
            si_sim = 1.0 - abs(si_i - si_j)
            # Compatibility combines phase coupling (50%), EPI similarity (25%), Si similarity (25%)
            compat = phase_coupling * 0.5 + 0.25 * epi_sim + 0.25 * si_sim
            if compat >= thr:
                node.add_edge(j, compat)


def _op_RA(node: NodeProtocol, gf: GlyphFactors) -> None:  # RA — Resonance
    """Propagate coherence through resonance with νf amplification.

    Resonance (RA) propagates EPI along existing couplings while amplifying
    the structural frequency (νf) to reflect network coherence propagation.
    According to TNFR theory, RA creates "resonant cascades" where coherence
    amplifies across the network, increasing collective νf and global C(t).

    **Canonical Effects (always active):**

    - **EPI Propagation**: Diffuses EPI to neighbors (identity-preserving)
    - **νf Amplification**: Increases structural frequency when propagating coherence
    - **Phase Alignment**: Strengthens phase synchrony across propagation path
    - **Network C(t)**: Contributes to global coherence increase
    - **Identity Preservation**: Maintains structural identity during propagation

    Parameters
    ----------
    node : NodeProtocol
        Node harmonising with its neighbourhood.
    gf : GlyphFactors
        Provides ``RA_epi_diff`` (mixing coefficient, default 0.15),
        ``RA_vf_amplification`` (νf boost factor, default 0.05), and
        ``RA_phase_coupling`` (phase alignment factor, default 0.10).

    Notes
    -----
    **νf Amplification (Canonical)**: When neighbors have coherence (|epi_bar| > 1e-9),
    node.vf is multiplied by (1.0 + RA_vf_amplification). This reflects
    the canonical TNFR property that resonance amplifies collective νf.
    This is NOT optional - it is a fundamental property of resonance per TNFR theory.

    **Phase Alignment Strengthening (Canonical)**: RA strengthens phase alignment
    with neighbors by applying a small phase correction toward the network mean.
    This ensures that "Phase alignment: Strengthens across propagation path" as
    stated in the theoretical foundations. Uses existing phase utility functions
    to avoid code duplication.

    **Network Coherence Tracking (Optional)**: If ``TRACK_NETWORK_COHERENCE`` is enabled,
    global C(t) is measured before/after RA application to quantify network-level
    coherence increase.

    **Identity Preservation (Canonical)**: EPI structure (kind and sign) are preserved
    during propagation to ensure structural identity is maintained as required by theory.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self, epi, neighbors):
    ...         self.EPI = epi
    ...         self.epi_kind = "seed"
    ...         self.vf = 1.0
    ...         self.theta = 0.0
    ...         self.graph = {}
    ...         self._neighbors = neighbors
    ...     def neighbors(self):
    ...         return self._neighbors
    >>> neighbor = MockNode(1.0, [])
    >>> neighbor.theta = 0.1
    >>> node = MockNode(0.2, [neighbor])
    >>> _op_RA(node, {"RA_epi_diff": 0.25, "RA_vf_amplification": 0.05})
    >>> round(node.EPI, 2)
    0.4
    >>> node.vf  # Amplified due to neighbor coherence (canonical effect)
    1.05
    """
    # Get configuration factors
    diff = get_factor(gf, "RA_epi_diff", 0.15)
    vf_boost = get_factor(gf, "RA_vf_amplification", 0.05)
    phase_coupling = get_factor(gf, "RA_phase_coupling", 0.10)  # Canonical phase strengthening

    # Track network C(t) before RA if enabled (optional telemetry)
    track_coherence = bool(node.graph.get("TRACK_NETWORK_COHERENCE", False))
    c_before = None
    if track_coherence and hasattr(node, "G"):
        try:
            from ..metrics.coherence import compute_network_coherence

            c_before = compute_network_coherence(node.G)
            if "_ra_c_tracking" not in node.graph:
                node.graph["_ra_c_tracking"] = []
        except ImportError:
            pass  # Metrics module not available

    # Capture state before for metrics
    vf_before = node.vf
    epi_before = node.EPI
    kind_before = node.epi_kind
    theta_before = node.theta if hasattr(node, "theta") else None

    # EPI diffusion (existing behavior)
    neigh, epi_bar = get_neighbor_epi(node)
    epi_bar_result, kind_result = _mix_epi_with_neighbors(node, diff, Glyph.RA)

    # CANONICAL EFFECT 1: νf amplification through resonance
    # This is always active - it's a fundamental property of resonance per TNFR theory
    # Only amplify if neighbors have coherence to propagate
    if abs(epi_bar_result) > 1e-9 and len(neigh) > 0:
        node.vf *= 1.0 + vf_boost

    # CANONICAL EFFECT 2: Phase alignment strengthening
    # Per theory: "Phase alignment: Strengthens across propagation path"
    # Uses existing phase locking logic from IL operator (avoid duplication)
    phase_strengthened = False
    if len(neigh) > 0 and hasattr(node, "theta") and hasattr(node, "G"):
        try:
            # Use existing phase locking utility from IL operator
            from ..alias import get_attr
            from ..constants.aliases import ALIAS_THETA
            import cmath
            import math

            # Get neighbor phases using existing utilities
            neighbor_phases = []
            for n in neigh:
                try:
                    theta_n = float(get_attr(n, ALIAS_THETA, 0.0))
                    neighbor_phases.append(theta_n)
                except (KeyError, ValueError, TypeError):
                    continue

            if neighbor_phases:
                # Circular mean using the same method as in phase_coherence.py
                complex_phases = [cmath.exp(1j * theta) for theta in neighbor_phases]
                mean_real = sum(z.real for z in complex_phases) / len(complex_phases)
                mean_imag = sum(z.imag for z in complex_phases) / len(complex_phases)
                mean_complex = complex(mean_real, mean_imag)
                mean_phase = cmath.phase(mean_complex)

                # Ensure positive phase [0, 2π]
                if mean_phase < 0:
                    mean_phase += 2 * math.pi

                # Calculate phase difference (shortest arc)
                delta_theta = mean_phase - node.theta
                if delta_theta > math.pi:
                    delta_theta -= 2 * math.pi
                elif delta_theta < -math.pi:
                    delta_theta += 2 * math.pi

                # Apply phase strengthening (move toward network mean)
                # Same approach as IL operator phase locking
                node.theta = node.theta + phase_coupling * delta_theta

                # Normalize to [0, 2π]
                node.theta = node.theta % (2 * math.pi)
                phase_strengthened = True
        except (AttributeError, ImportError):
            pass  # Phase alignment not possible in this context

    # Track identity preservation (canonical validation)
    identity_preserved = (kind_result == kind_before or kind_result == Glyph.RA.value) and (
        float(epi_before) * float(node.EPI) >= 0
    )  # Sign preserved

    # Collect propagation metrics if enabled (optional telemetry)
    collect_metrics = bool(node.graph.get("COLLECT_RA_METRICS", False))
    if collect_metrics:
        metrics = {
            "operator": "RA",
            "epi_propagated": epi_bar_result,
            "vf_amplification": node.vf / vf_before if vf_before > 0 else 1.0,
            "neighbors_influenced": len(neigh),
            "identity_preserved": identity_preserved,
            "epi_before": epi_before,
            "epi_after": float(node.EPI),
            "vf_before": vf_before,
            "vf_after": node.vf,
            "phase_before": theta_before,
            "phase_after": node.theta if hasattr(node, "theta") else None,
            "phase_alignment_strengthened": phase_strengthened,
        }
        if "ra_metrics" not in node.graph:
            node.graph["ra_metrics"] = []
        node.graph["ra_metrics"].append(metrics)

    # Track network C(t) after RA if enabled (optional telemetry)
    if track_coherence and c_before is not None and hasattr(node, "G"):
        try:
            from ..metrics.coherence import compute_network_coherence

            c_after = compute_network_coherence(node.G)
            node.graph["_ra_c_tracking"].append(
                {
                    "node": getattr(node, "n", None),
                    "c_before": c_before,
                    "c_after": c_after,
                    "c_delta": c_after - c_before,
                }
            )
        except ImportError:
            pass


def _op_SHA(node: NodeProtocol, gf: GlyphFactors) -> None:  # SHA — Silence
    """Reduce νf while preserving EPI, ΔNFR, and phase.

    Silence decelerates a node by scaling νf (structural frequency) towards
    stillness. EPI, ΔNFR, and phase remain unchanged, signalling a temporary
    suspension of structural evolution.

    **TNFR Canonical Behavior:**

    According to the nodal equation ∂EPI/∂t = νf · ΔNFR(t), reducing νf → νf_min ≈ 0
    causes structural evolution to freeze (∂EPI/∂t → 0) regardless of ΔNFR magnitude.
    This implements **structural silence** - a state where the node's form (EPI) is
    preserved intact despite external pressures, enabling memory consolidation and
    protective latency.

    Parameters
    ----------
    node : NodeProtocol
        Node whose νf is being attenuated.
    gf : GlyphFactors
        Provides ``SHA_vf_factor`` to scale νf (default 0.85 for gradual reduction).

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
    # Canonical SHA effect: reduce structural frequency toward zero
    # This implements: νf → νf_min ≈ 0 ⇒ ∂EPI/∂t → 0 (structural preservation)
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
    graph_attrs = getattr(node, "graph", {})
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

        # NUL canonical ΔNFR densification (implements structural pressure concentration)
        if glyph is Glyph.NUL:
            # Volume reduction: V' = V · scale_factor (where scale_factor < 1.0)
            # Density increase: ρ_ΔNFR = ΔNFR / V' = ΔNFR / (V · scale_factor)
            # Result: ΔNFR' = ΔNFR · densification_factor
            #
            # Physics: When volume contracts by factor λ < 1, structural pressure
            # concentrates by factor 1/λ > 1. For NUL_scale = 0.85, densification ≈ 1.176
            #
            # Default densification_factor from config (typically 1.3-1.5) provides
            # additional canonical amplification beyond geometric 1/λ to account for
            # nonlinear structural effects at smaller scales.
            densification_key = "NUL_densification_factor"
            densification_default = 1.35  # Canonical default: moderate amplification
            densification_factor = get_factor(gf, densification_key, densification_default)

            # Apply densification to ΔNFR (use lowercase dnfr for NodeProtocol)
            current_dnfr = node.dnfr
            node.dnfr = current_dnfr * densification_factor

            # Record densification telemetry for traceability
            telemetry = node.graph.setdefault("nul_densification_log", [])
            telemetry.append(
                {
                    "dnfr_before": current_dnfr,
                    "dnfr_after": float(node.dnfr),
                    "densification_factor": densification_factor,
                    "contraction_scale": factor,
                }
            )

        # Edge-aware EPI scaling (new behavior) if enabled
        edge_aware_enabled = bool(
            node.graph.get("EDGE_AWARE_ENABLED", DEFAULTS.get("EDGE_AWARE_ENABLED", True))
        )

        if edge_aware_enabled:
            epsilon = float(
                node.graph.get("EDGE_AWARE_EPSILON", DEFAULTS.get("EDGE_AWARE_EPSILON", 1e-12))
            )
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
                telemetry.append(
                    {
                        "glyph": glyph.name if hasattr(glyph, "name") else str(glyph),
                        "epi_before": epi_current,
                        "epi_after": float(node.EPI),  # Get actual value after boundary check
                        "scale_requested": factor,
                        "scale_effective": scale_eff,
                        "adapted": True,
                    }
                )

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
    """Apply canonical phase transformation θ → θ' based on structural dynamics.

    ZHIR (Mutation) implements the canonical TNFR phase transformation that depends on
    the node's reorganization state (ΔNFR). Unlike a fixed rotation, the transformation
    magnitude and direction are determined by structural pressure, implementing the
    physics: θ → θ' when ΔEPI/Δt > ξ (AGENTS.md §11, TNFR.pdf §2.2.11).

    **Canonical Behavior**:
    - Direction: Based on ΔNFR sign (positive → forward phase, negative → backward)
    - Magnitude: Proportional to theta_shift_factor and |ΔNFR|
    - Regime detection: Identifies quadrant crossings (π/2 boundaries)
    - Deterministic: Same seed produces same transformation

    The transformation preserves structural identity (epi_kind) while shifting the
    operational regime, enabling adaptation without losing coherence.

    Parameters
    ----------
    node : NodeProtocol
        Node whose phase is transformed based on its structural state.
    gf : GlyphFactors
        Supplies ``ZHIR_theta_shift_factor`` (default: 0.3) controlling transformation
        magnitude. Can override with explicit ``ZHIR_theta_shift`` for fixed rotation.

    Examples
    --------
    >>> import math
    >>> class MockNode:
    ...     def __init__(self, theta, dnfr):
    ...         self.theta = theta
    ...         self.dnfr = dnfr
    ...         self.graph = {}
    >>> # Positive ΔNFR → forward phase shift
    >>> node = MockNode(0.0, 0.5)
    >>> _op_ZHIR(node, {"ZHIR_theta_shift_factor": 0.3})
    >>> 0.2 < node.theta < 0.3  # ~π/4 * 0.3 ≈ 0.24
    True
    >>> # Negative ΔNFR → backward phase shift
    >>> node2 = MockNode(math.pi, -0.5)
    >>> _op_ZHIR(node2, {"ZHIR_theta_shift_factor": 0.3})
    >>> 2.9 < node2.theta < 3.0  # π - 0.24 ≈ 2.90
    True
    >>> # Fixed shift overrides dynamic behavior
    >>> node3 = MockNode(0.0, 0.5)
    >>> _op_ZHIR(node3, {"ZHIR_theta_shift": math.pi / 2})
    >>> round(node3.theta, 2)
    1.57
    """
    # Check for explicit fixed shift (backward compatibility)
    if "ZHIR_theta_shift" in gf:
        shift = get_factor(gf, "ZHIR_theta_shift", math.pi / 2)
        node.theta = node.theta + shift
        # Store telemetry for fixed shift mode
        storage = node._glyph_storage()
        storage["_zhir_theta_shift"] = shift
        storage["_zhir_fixed_mode"] = True
        return

    # Canonical transformation: θ → θ' based on ΔNFR
    theta_before = node.theta
    dnfr = node.dnfr

    # Transformation magnitude controlled by factor
    theta_shift_factor = get_factor(gf, "ZHIR_theta_shift_factor", 0.3)

    # Direction based on ΔNFR sign (coherent with structural pressure)
    # Magnitude based on |ΔNFR| (stronger pressure → larger shift)
    # Base shift is π/4, scaled by factor and ΔNFR
    base_shift = math.pi / 4
    shift = theta_shift_factor * math.copysign(1.0, dnfr) * base_shift

    # Apply transformation with phase wrapping [0, 2π)
    theta_new = (theta_before + shift) % (2 * math.pi)
    node.theta = theta_new

    # Detect regime change (crossing quadrant boundaries)
    regime_before = int(theta_before // (math.pi / 2))
    regime_after = int(theta_new // (math.pi / 2))
    regime_changed = regime_before != regime_after

    # Store telemetry for metrics collection
    storage = node._glyph_storage()
    storage["_zhir_theta_shift"] = shift
    storage["_zhir_theta_before"] = theta_before
    storage["_zhir_theta_after"] = theta_new
    storage["_zhir_regime_changed"] = regime_changed
    storage["_zhir_regime_before"] = regime_before
    storage["_zhir_regime_after"] = regime_after
    storage["_zhir_fixed_mode"] = False


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


def _op_REMESH(node: NodeProtocol, gf: GlyphFactors | None = None) -> None:  # REMESH — advisory
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


def apply_glyph_obj(node: NodeProtocol, glyph: Glyph | str, *, window: int | None = None) -> None:
    """Apply ``glyph`` to an object satisfying :class:`NodeProtocol`."""

    from .grammar import function_name_to_glyph
    from ..validation.input_validation import ValidationError, validate_glyph

    # Validate glyph parameter
    try:
        if not isinstance(glyph, Glyph):
            validated_glyph = validate_glyph(glyph)
            glyph = validated_glyph.value if isinstance(validated_glyph, Glyph) else str(glyph)
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


def apply_glyph(G: TNFRGraph, n: NodeId, glyph: Glyph | str, *, window: int | None = None) -> None:
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
