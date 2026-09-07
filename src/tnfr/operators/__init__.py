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
from contextlib import contextmanager
from itertools import islice
from statistics import StatisticsError
from typing import TYPE_CHECKING, Any

from tnfr import glyph_history

from ..alias import get_attr, set_attr_str
from ..constants import DEFAULTS, get_param
from ..constants.aliases import (
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..constants.canonical import UM_COMPAT_THRESHOLD as _UM_COMPAT_CANONICAL
from ..constants.canonical import (
    COHERENCE_RETENTION,
    COUPLING_FINE,
    COUPLING_GENTLE,
    COUPLING_MODERATE,
    DISSONANCE_AMPLIFICATION,
    EN_MIX_FACTOR,
    INV_PI,
    NUL_DENSIFICATION_FACTOR,
    NUL_SCALE_FACTOR,
    SHA_VF_FACTOR,
    VAL_SCALE_FACTOR,
)
from ..errors import TNFRValueError
from ..rng import make_rng, resolve_graph_seed, validate_graph_seed
from ..types import EPIValue, Glyph, NodeId, TNFRGraph
from ..utils import angle_diff, get_nodenx
from . import definitions as _definitions
from ._epi_domain import require_real_scalar_epi, validate_affine_epi_graph_input
from ._neighbor_epi_kernel import (
    neighbor_epi_blend_value,
    neighbor_epi_unweighted_mean,
)
from ._phase_gate import (
    U3PhaseGateError,
    U3PhaseNeighborSet,
    resolve_u3_phase_neighbors,
    select_u3_phase_neighbors,
)
from ._resonance_identity import (
    RA_RUNTIME_AMPLIFICATION_TRIGGER,
    normalize_resonance_epi_kind,
    resonance_identity_failures,
    resonance_neighbor_circular_mean,
    resonance_proposed_epi_kind,
    validate_resonance_runtime_factors,
)
from .factor_contracts import (
    GLYPH_FACTOR_SPECS,
    GLYPH_FACTORS_BY_GLYPH,
    GlyphFactorSpec,
    GlyphFactorValidationError,
    canonical_glyph_factor_defaults,
    resolve_operator_factors,
    resolve_runtime_operator_factors,
    runtime_active_glyph_factor_keys,
    validate_glyph_factor,
    validate_glyph_factors,
)
from .jitter import (
    _JITTER_PROGRESS_KEY,
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
from .grammar_observations import GrammarObservation, observe_grammar
from .grammar_u6 import (
    StructuralPotentialConfinementObservation,
    observe_structural_potential_confinement,
)
from .hamiltonian import (  # noqa: E402
    InternalHamiltonian,
    build_H_coherence,
    build_H_coupling,
    build_H_frequency,
)
from .health_analyzer import SequenceHealthAnalyzer, SequenceHealthMetrics  # noqa: E402
from .pattern_detection import (  # noqa: E402
    PatternMatch,
    UnifiedPatternDetector,
    analyze_sequence,
    detect_pattern,
)
from .word_execution import (  # noqa: E402
    preflight_network_mutation_sequence,
    run_network_sequence,
)

__all__ = [
    "JitterCache",
    "JitterCacheManager",
    "get_jitter_manager",
    "reset_jitter_manager",
    "random_jitter",
    "get_neighbor_epi",
    "get_glyph_factors",
    "GLYPH_FACTOR_SPECS",
    "GLYPH_FACTORS_BY_GLYPH",
    "GlyphFactorSpec",
    "GlyphFactorValidationError",
    "canonical_glyph_factor_defaults",
    "resolve_operator_factors",
    "resolve_runtime_operator_factors",
    "runtime_active_glyph_factor_keys",
    "validate_glyph_factor",
    "validate_glyph_factors",
    "GLYPH_OPERATIONS",
    "apply_glyph_obj",
    "apply_glyph",
    "apply_glyph_with_grammar",
    "GrammarObservation",
    "observe_grammar",
    "StructuralPotentialConfinementObservation",
    "observe_structural_potential_confinement",
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
    "preflight_network_mutation_sequence",
    "run_network_sequence",
]

__all__.extend(_DEFINITION_EXPORTS.keys())


def get_glyph_factors(
    node: NodeProtocol, glyph: Glyph | str | None = None
) -> GlyphFactors:
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
    glyph : Glyph or str, optional
        When supplied, validate only the factors used by that operator and
        resolve its derived relations. This is the runtime path: an unrelated
        pending override cannot block the current operator.

    Returns
    -------
    GlyphFactors
        Detached mapping with graph overrides merged onto the canonical
        defaults. Known factors are validated against their operator contracts.

    Examples
    --------
    >>> class MockNode:
    ...     def __init__(self):
    ...         self.graph = {"GLYPH_FACTORS": {"AL_boost": 0.2}}
    >>> node = MockNode()
    >>> factors = get_glyph_factors(node)
    >>> factors["AL_boost"]
    0.2
    >>> round(factors["EN_mix"], 6)  # Canonical 1/(pi + 1) fallback
    0.241453
    """
    raw = node.graph.get("GLYPH_FACTORS")
    if glyph is not None:
        return resolve_runtime_operator_factors(raw, glyph, node.graph)
    factors = canonical_glyph_factor_defaults()
    validated = validate_glyph_factors(raw)
    factors.update(validated)
    # Keep the derived NUL relation coherent in the context-free public view as
    # well as during execution.  An explicit densification remains subject to
    # the registry's equality check above.
    if isinstance(raw, dict) and "NUL_scale" in raw:
        factors["NUL_densification_factor"] = 1.0 / factors["NUL_scale"]
    return validate_glyph_factors(factors)


def get_factor(gf: GlyphFactors, key: str, default: float) -> float:
    """Return one finite glyph factor without hiding an explicit bad value.

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

    Known canonical factors are checked by the shared contract registry. An
    absent key uses the supplied default; an explicit invalid value raises and
    can therefore never silently change the requested operation.

    Examples
    --------
    >>> get_factor({"AL_boost": 0.3}, "AL_boost", 0.05)
    0.3
    >>> get_factor({}, "IL_dnfr_factor", 0.7)
    0.7
    """
    value = gf.get(key, default)
    if key in GLYPH_FACTOR_SPECS:
        return validate_glyph_factor(key, value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GlyphFactorValidationError(
            f"{key} must be a finite real scalar, got {value!r}"
        )
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise GlyphFactorValidationError(
            f"{key} must be representable as a finite real scalar, got {value!r}"
        ) from exc
    if not math.isfinite(resolved):
        raise GlyphFactorValidationError(f"{key} must be finite, got {value!r}")
    return resolved


def _finite_operator_scalar(value: Any, label: str) -> float:
    """Materialize a finite runtime state/proposal or reject before commit."""

    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"{label} must be representable as a finite scalar",
            context={"field": label, "value": repr(value)},
        ) from exc
    if not math.isfinite(resolved):
        raise TNFRValueError(
            f"{label} must remain finite",
            context={"field": label, "value": repr(value)},
        )
    return resolved


def _finite_real_epi(value: Any, label: str, *, operator: str) -> float:
    """Require the exact signed EPI chart used by an affine glyph."""

    return require_real_scalar_epi(value, operator=operator, label=label)


@contextmanager
def _rollback_jitter_progress_on_error(node: NodeProtocol) -> Iterator[None]:
    """Restore deterministic jitter progress when an operation is rejected."""

    storage = node._glyph_storage()
    had_progress = _JITTER_PROGRESS_KEY in storage
    progress_before = storage.get(_JITTER_PROGRESS_KEY)
    try:
        yield
    except BaseException:
        if had_progress:
            storage[_JITTER_PROGRESS_KEY] = progress_before
        else:
            storage.pop(_JITTER_PROGRESS_KEY, None)
        raise


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
        values: list[float] = []
        has_valid_neighbor = False
        needs_conversion = False
        for v in neigh:
            if hasattr(v, "EPI"):
                values.append(
                    _finite_real_epi(v.EPI, "neighbor EPI state", operator="Reception")
                )
                has_valid_neighbor = True
            else:
                attr = get_attr(G.nodes[v], ALIAS_EPI, None)
                if attr is not None:
                    values.append(
                        _finite_real_epi(
                            attr, "neighbor EPI state", operator="Reception"
                        )
                    )
                    has_valid_neighbor = True
                else:
                    values.append(
                        _finite_real_epi(epi, "target EPI state", operator="Reception")
                    )
                needs_conversion = True
        if not has_valid_neighbor:
            return [], epi
        epi_bar = neighbor_epi_unweighted_mean(values)
        if needs_conversion:
            NodeNX = get_nodenx()
            if NodeNX is None:
                raise ImportError("NodeNX is unavailable")
            neigh = [
                v if hasattr(v, "EPI") else NodeNX.from_graph(node.G, v) for v in neigh
            ]
    else:
        try:
            epi_bar = neighbor_epi_unweighted_mean(
                _finite_real_epi(
                    v.EPI, "neighbor EPI state", operator="Reception"
                )
                for v in neigh
            )
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
    epi = _finite_real_epi(node.EPI, "target EPI state", operator="Reception")
    neigh, epi_bar = get_neighbor_epi(node)

    if not neigh:
        node.epi_kind = str(default_kind)
        return epi, default_kind

    epi_bar = _finite_operator_scalar(epi_bar, "EN neighbor EPI mean")
    for index, neighbor in enumerate(neigh):
        _finite_real_epi(
            neighbor.EPI,
            f"neighbor EPI state[{index}]",
            operator="Reception",
        )
    dominant, best_abs = _determine_dominant(neigh, default_kind)
    new_epi = _finite_operator_scalar(
        neighbor_epi_blend_value(epi, epi_bar, mix), "EN EPI proposal"
    )
    final = dominant if best_abs > abs(new_epi) else node.epi_kind
    if not final:
        final = default_kind
    final = str(final)

    # Validate both proposals before changing either form or identity.  Retain a
    # best-effort rollback for custom NodeProtocol setters that reject the kind
    # after accepting EPI; NodeNX's ordinary mapping setters cannot hit it.
    bounded_epi = _validated_epi_assignment_value(node, new_epi)
    epi_before = node.EPI
    kind_before = node.epi_kind
    node.EPI = bounded_epi
    try:
        node.epi_kind = final
    except BaseException:
        try:
            node.EPI = epi_before
            node.epi_kind = kind_before
        except BaseException:
            pass
        raise
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
    f = get_factor(gf, "AL_boost", COUPLING_GENTLE)
    epi = _finite_real_epi(node.EPI, "target EPI state", operator="Emission")
    new_epi = _finite_operator_scalar(epi + f, "AL EPI proposal")
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
    mix = get_factor(gf, "EN_mix", EN_MIX_FACTOR)
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
    factor = get_factor(gf, "IL_dnfr_factor", COHERENCE_RETENTION)
    dnfr = _finite_operator_scalar(getattr(node, "dnfr", 0.0), "IL DeltaNFR state")
    proposal = _finite_operator_scalar(factor * dnfr, "IL DeltaNFR proposal")
    node.dnfr = proposal


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
    dnfr = _finite_operator_scalar(getattr(node, "dnfr", 0.0), "OZ DeltaNFR state")
    if bool(node.graph.get("OZ_NOISE_MODE", False)):
        sigma = _finite_operator_scalar(
            node.graph.get("OZ_SIGMA", 0.1), "OZ noise sigma"
        )
        if sigma <= 0:
            return
        with _rollback_jitter_progress_on_error(node):
            jitter = _finite_operator_scalar(
                random_jitter(node, sigma), "OZ noise sample"
            )
            proposal = _finite_operator_scalar(
                dnfr + jitter, "OZ DeltaNFR proposal"
            )
            node.dnfr = proposal
    else:
        factor = get_factor(gf, "OZ_dnfr_factor", DISSONANCE_AMPLIFICATION)
        proposal = factor * dnfr if abs(dnfr) > 1e-9 else 0.1
        node.dnfr = _finite_operator_scalar(proposal, "OZ DeltaNFR proposal")


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


def _raw_runtime_phase(node: NodeProtocol, subject: Any) -> Any:
    """Read a concrete graph phase without adapter coercion or fallback."""

    if hasattr(node, "G"):
        subject_id = getattr(subject, "n", subject)
        try:
            attributes = node.G.nodes[subject_id]
        except (KeyError, TypeError):
            pass
        else:
            return get_attr(
                attributes,
                ALIAS_THETA,
                None,
                strict=True,
                conv=lambda value: value,
            )
    return getattr(subject, "theta")


def _runtime_u3_neighbors(
    node: NodeProtocol, operator_code: str
) -> tuple[U3PhaseNeighborSet, tuple[NodeProtocol, ...]]:
    """Resolve one immutable U3 snapshot before an UM/RA state mutation."""

    operator_name = "Coupling" if operator_code == "UM" else "Resonance"
    try:
        raw_neighbors = tuple(node.neighbors())
        selection = resolve_u3_phase_neighbors(
            getattr(node, "graph", {}),
            _raw_runtime_phase(node, node),
            raw_neighbors,
            phase_getter=lambda neighbor: _raw_runtime_phase(node, neighbor),
            operator_code=operator_code,
        )
    except U3PhaseGateError as exc:
        raise TNFRValueError(
            f"{operator_name} phase gate rejected the operation: {exc}",
            context={
                "operator": operator_name,
                "failed_condition": exc.failed_condition,
            },
        ) from exc
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"{operator_name} phase gate could not read the runtime phase state",
            context={
                "operator": operator_name,
                "failed_condition": "phase_state",
            },
        ) from exc

    if not hasattr(node, "G"):
        return selection, tuple(selection.neighbors)

    NodeNX = get_nodenx()
    if NodeNX is None:
        raise ImportError("NodeNX is unavailable")
    runtime_neighbors = tuple(
        neighbor
        if hasattr(neighbor, "theta")
        else NodeNX.from_graph(node.G, neighbor)
        for neighbor in selection.neighbors
    )
    return selection, runtime_neighbors


def _um_select_candidates(
    node: NodeProtocol,
    candidates: Iterator[NodeProtocol],
    limit: int,
    mode: str,
    th: float,
) -> list[NodeProtocol]:
    """Select a subset of ``candidates`` for UM coupling."""
    rng = make_rng(resolve_graph_seed(node), node.offset(), node.G)

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


def compute_consensus_phase(phases: list[float]) -> float:
    """Compute circular mean (consensus phase) from a list of phase angles.

    This function calculates the consensus phase using the circular mean
    formula: arctan2(mean(sin), mean(cos)). This ensures proper handling
    of phase wrapping at ±π boundaries.

    Parameters
    ----------
    phases : list[float]
        list of phase angles in radians.

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
    # Resolve every factor and the immutable U3 subset before any node, edge,
    # history, metric, or telemetry mutation.
    k = get_factor(gf, "UM_theta_push", EN_MIX_FACTOR)
    k_vf = get_factor(gf, "UM_vf_sync", COUPLING_GENTLE)
    stabilize_dnfr = bool(node.graph.get("UM_STABILIZE_DNFR", True))
    k_dnfr = (
        get_factor(gf, "UM_dnfr_reduction", COUPLING_MODERATE)
        if stabilize_dnfr
        else 0.0
    )
    selection, neighbors = _runtime_u3_neighbors(node, "UM")
    th_i = selection.target_phase
    neighbor_phases = list(selection.phases)
    bidirectional = bool(node.graph.get("UM_BIDIRECTIONAL", True))

    consensus_inputs = [th_i, *neighbor_phases] if bidirectional else neighbor_phases
    target_phase = compute_consensus_phase(consensus_inputs)
    proposed_theta = _finite_operator_scalar(
        th_i + k * angle_diff(target_phase, th_i), "UM target phase proposal"
    )
    if bidirectional:
        proposed_neighbor_phases = [
            _finite_operator_scalar(
                phase + k * angle_diff(target_phase, phase),
                "UM neighbor phase proposal",
            )
            for phase in neighbor_phases
        ]
    else:
        proposed_neighbor_phases = neighbor_phases

    proposed_vf: float | None = None
    sync_vf = bool(node.graph.get("UM_SYNC_VF", True))
    if sync_vf and hasattr(node, "G"):
        vf_i = _finite_operator_scalar(node.vf, "UM target structural frequency")
        vf_neighbors = [
            _finite_operator_scalar(
                neighbor.vf, "UM compatible-neighbor structural frequency"
            )
            for neighbor in neighbors
        ]
        vf_mean = sum(vf_neighbors) / len(vf_neighbors)
        proposed_vf = _finite_operator_scalar(
            vf_i + k_vf * (vf_mean - vf_i),
            "UM structural-frequency proposal",
        )

    proposed_dnfr: float | None = None
    if stabilize_dnfr and hasattr(node, "G"):
        from ..metrics.phase_compatibility import compute_phase_coupling_strength

        phase_alignments = [
            compute_phase_coupling_strength(proposed_theta, phase)
            for phase in proposed_neighbor_phases
        ]
        mean_alignment = sum(phase_alignments) / len(phase_alignments)
        reduction_factor = 1.0 - k_dnfr * mean_alignment
        proposed_dnfr = _finite_operator_scalar(
            _finite_operator_scalar(node.dnfr, "UM target DeltaNFR")
            * reduction_factor,
            "UM DeltaNFR proposal",
        )

    proposed_links: list[tuple[NodeProtocol, float]] = []
    if bool(node.graph.get("UM_FUNCTIONAL_LINKS", True)) and hasattr(node, "G"):
        thr = _finite_operator_scalar(
            node.graph.get(
                "UM_COMPAT_THRESHOLD",
                DEFAULTS.get("UM_COMPAT_THRESHOLD", _UM_COMPAT_CANONICAL),
            ),
            "UM_COMPAT_THRESHOLD",
        )
        limit = int(node.graph.get("UM_CANDIDATE_COUNT", 0))
        mode = str(node.graph.get("UM_CANDIDATE_MODE", "sample")).lower()

        # The hard U3 gate is applied to prospective functional links before
        # sampling or scoring, so a high EPI/Si score cannot admit an antiphase
        # edge. The optional UM limit is the same effective limit used above.
        all_candidates = tuple(_um_candidate_iter(node))
        try:
            _, phase_candidates, phase_candidate_values = select_u3_phase_neighbors(
                th_i,
                all_candidates,
                phase_getter=lambda candidate: _raw_runtime_phase(node, candidate),
                phase_limit=selection.effective_limit,
                require_compatible=False,
            )
        except U3PhaseGateError as exc:
            raise TNFRValueError(
                f"Coupling functional-link phase gate rejected a candidate: {exc}",
                context={
                    "operator": "Coupling",
                    "failed_condition": exc.failed_condition,
                },
            ) from exc
        phase_by_identity = {
            id(candidate): phase
            for candidate, phase in zip(
                phase_candidates, phase_candidate_values, strict=True
            )
        }
        candidates = _um_select_candidates(
            node, iter(phase_candidates), limit, mode, th_i
        )

        from ..metrics.phase_compatibility import compute_phase_coupling_strength

        epi_i = node.EPI
        si_i = _finite_operator_scalar(node.Si, "UM target sense index")
        for candidate in candidates:
            phase_coupling = compute_phase_coupling_strength(
                th_i, phase_by_identity[id(candidate)]
            )
            epi_j = candidate.EPI
            si_j = _finite_operator_scalar(
                candidate.Si, "UM candidate sense index"
            )
            epi_sim = 1.0 - abs(epi_i - epi_j) / (
                abs(epi_i) + abs(epi_j) + 1e-9
            )
            si_sim = 1.0 - abs(si_i - si_j)
            compat = _finite_operator_scalar(
                phase_coupling * 0.5 + 0.25 * epi_sim + 0.25 * si_sim,
                "UM functional-link compatibility",
            )
            if compat >= thr:
                proposed_links.append((candidate, compat))

    # Atomic channel commit follows complete validation and proposal building.
    node.theta = proposed_theta
    if bidirectional:
        for neighbor, phase in zip(
            neighbors, proposed_neighbor_phases, strict=True
        ):
            neighbor.theta = phase
    if proposed_vf is not None:
        node.vf = proposed_vf
    if proposed_dnfr is not None:
        node.dnfr = proposed_dnfr
    for candidate, compatibility in proposed_links:
        node.add_edge(candidate, compatibility)


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
        Provides ``RA_epi_diff`` (mixing coefficient, default ``1/(2π)``),
        ``RA_vf_amplification`` (νf boost factor, default ``1/(8π)``), and
        ``RA_phase_coupling`` (phase alignment factor, default ``1/(4π)``).

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
    # Resolve configuration before any state or telemetry mutation.
    try:
        diff = get_factor(gf, "RA_epi_diff", COUPLING_MODERATE)
        vf_boost = get_factor(gf, "RA_vf_amplification", COUPLING_FINE)
        phase_coupling = get_factor(
            gf, "RA_phase_coupling", COUPLING_GENTLE
        )  # Canonical phase strengthening
    except GlyphFactorValidationError as exc:
        raise TNFRValueError(
            f"Resonance factor gate rejected the proposed propagation: {exc}",
            context={"operator": "Resonance", "failed_condition": "factor_domain"},
        ) from exc
    invalid_factors = validate_resonance_runtime_factors(
        diff, vf_boost, phase_coupling
    )
    if invalid_factors:
        raise TNFRValueError(
            "Resonance factor gate rejected the proposed propagation: "
            + "; ".join(invalid_factors),
            context={
                "operator": "Resonance",
                "RA_epi_diff": diff,
                "RA_vf_amplification": vf_boost,
                "RA_phase_coupling": phase_coupling,
                "failed_conditions": tuple(invalid_factors),
            },
        )

    # Resolve the same hard U3 neighbor snapshot used by UM before reading any
    # propagation input that could otherwise bypass the phase filter.
    selection, compatible_neighbors = _runtime_u3_neighbors(node, "RA")
    theta_before = selection.target_phase
    neigh = list(compatible_neighbors)
    compatible_neighbor_phases = list(selection.phases)
    graph_attrs = getattr(node, "graph", {})

    # Capture the complete identity state and compute the clipped EPI proposal
    # before touching EPI, frequency, phase, history, or telemetry.  Resonance
    # may reorganize the scalar value, but its canonical contract forbids a
    # strict nonzero sign inversion and forbids replacing an established kind.
    vf_before = node.vf
    epi_before = node.EPI
    from ..types import real_scalar_epi

    epi_before_scalar = real_scalar_epi(epi_before)
    if epi_before_scalar is None:
        raise TNFRValueError(
            "Resonance identity requires a raw scalar or uniform-real BEPI target",
            context={"operator": "Resonance", "failed_condition": "scalar_epi"},
        )
    raw_kind: Any = node.epi_kind
    if hasattr(node, "G") and hasattr(node, "n"):
        raw_kind = get_attr(
            node.G.nodes[node.n],
            ALIAS_EPI_KIND,
            "",
            strict=True,
            conv=lambda value: value,
        )
    kind_before = normalize_resonance_epi_kind(raw_kind)

    if neigh:
        neighbor_value_kinds_list: list[tuple[float, str]] = []
        for neighbor in neigh:
            scalar = real_scalar_epi(neighbor.EPI)
            if scalar is None:
                raise TNFRValueError(
                    "Resonance identity requires raw scalar or uniform-real BEPI neighbors",
                    context={
                        "operator": "Resonance",
                        "failed_condition": "scalar_neighbor_epi",
                    },
                )
            raw_neighbor_kind: Any = neighbor.epi_kind
            if hasattr(neighbor, "G") and hasattr(neighbor, "n"):
                raw_neighbor_kind = get_attr(
                    neighbor.G.nodes[neighbor.n],
                    ALIAS_EPI_KIND,
                    "",
                    strict=True,
                    conv=lambda value: value,
                )
            neighbor_value_kinds_list.append(
                (scalar, normalize_resonance_epi_kind(raw_neighbor_kind))
            )
        neighbor_value_kinds = tuple(neighbor_value_kinds_list)
        epi_bar = neighbor_epi_unweighted_mean(
            value for value, _kind in neighbor_value_kinds
        )
        unclipped_epi = neighbor_epi_blend_value(epi_before, epi_bar, diff)
        from ..dynamics.structural_clip import structural_clip

        epi_min = float(graph_attrs.get("EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)))
        epi_max = float(graph_attrs.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)))
        clip_mode = str(graph_attrs.get("CLIP_MODE", "hard"))
        if clip_mode not in ("hard", "soft"):
            clip_mode = "hard"
        proposed_epi = float(
            structural_clip(
                unclipped_epi,
                lo=epi_min,
                hi=epi_max,
                mode=clip_mode,
                record_stats=False,
            )
        )
    else:
        epi_bar = epi_before_scalar
        proposed_epi = epi_before_scalar
        neighbor_value_kinds = ()

    proposed_theta = theta_before
    neighbor_phase_mean: float | None = None
    if compatible_neighbor_phases:
        neighbor_phase_mean, phase_mean_defined = resonance_neighbor_circular_mean(
            compatible_neighbor_phases
        )
        if phase_mean_defined and neighbor_phase_mean is not None:
            proposed_theta = (
                theta_before
                + phase_coupling * angle_diff(neighbor_phase_mean, theta_before)
            ) % (2.0 * math.pi)
            if not math.isfinite(proposed_theta):
                raise TNFRValueError(
                    "Resonance phase proposal must remain finite",
                    context={
                        "operator": "Resonance",
                        "failed_condition": "finite_phase_proposal",
                    },
                )

    proposed_kind = resonance_proposed_epi_kind(
        kind_before,
        neighbor_value_kinds,
        proposed_epi,
        fallback_kind=Glyph.RA.value,
    )
    identity_failures = resonance_identity_failures(
        epi_before_scalar, proposed_epi, kind_before, proposed_kind
    )
    if identity_failures:
        raise TNFRValueError(
            "Resonance identity gate rejected the proposed propagation: "
            + ", ".join(identity_failures),
            context={
                "operator": "Resonance",
                "epi_before": epi_before_scalar,
                "epi_proposed": proposed_epi,
                "epi_kind_before": kind_before,
                "epi_kind_proposed": proposed_kind,
                "failed_conditions": identity_failures,
            },
        )

    amplification_active = bool(
        neigh and abs(epi_bar) > RA_RUNTIME_AMPLIFICATION_TRIGGER
    )
    proposed_vf = float(vf_before)
    if amplification_active:
        proposed_vf = float(vf_before) * (1.0 + vf_boost)
    if not math.isfinite(proposed_vf) or proposed_vf < float(vf_before):
        raise TNFRValueError(
            "Resonance capacity proposal must be finite and nondecreasing",
            context={
                "operator": "Resonance",
                "vf_before": float(vf_before),
                "vf_proposed": proposed_vf,
                "failed_condition": "finite_nondecreasing_capacity",
            },
        )

    # Track network C(t) before RA if enabled (optional telemetry).  This starts
    # only after the hard identity gate, so rejected inputs leave no RA metadata.
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

    # Commit the already validated proposal.  The setter remains the canonical
    # storage boundary, while clipping is not repeated after the gate.
    if neigh:
        _set_epi_with_boundary_check(node, proposed_epi, apply_clip=False)
    node.epi_kind = proposed_kind
    epi_bar_result = epi_bar
    kind_result = proposed_kind

    # CANONICAL EFFECT 1: νf amplification through resonance
    # This is always active - it's a fundamental property of resonance per TNFR theory
    # Only amplify if neighbors have coherence to propagate
    if amplification_active:
        node.vf = proposed_vf

    # CANONICAL EFFECT 2: align only with the U3-admissible propagation subset.
    phase_strengthened = bool(neigh and proposed_theta != theta_before)
    if neigh:
        node.theta = proposed_theta

    # The hard gate above establishes both independent identity clauses.
    identity_preserved = True

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
        Provides ``SHA_vf_factor`` to scale νf (default = SHA_VF_FACTOR, 0.9).

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
    factor = get_factor(gf, "SHA_vf_factor", SHA_VF_FACTOR)  # canonical ν_f↓ gain
    # Canonical SHA effect: reduce structural frequency toward zero
    # This implements: νf → νf_min ≈ 0 ⇒ ∂EPI/∂t → 0 (structural preservation)
    vf = _finite_operator_scalar(node.vf, "SHA nu_f state")
    proposal = _finite_operator_scalar(factor * vf, "SHA nu_f proposal")
    node.vf = proposal


factor_val = VAL_SCALE_FACTOR  # canonical Expansion ν_f↑ gain
factor_nul = NUL_SCALE_FACTOR  # canonical Contraction ν_f↓ gain
_SCALE_FACTORS = {Glyph.VAL: factor_val, Glyph.NUL: factor_nul}


def _validated_epi_assignment_value(
    node: NodeProtocol, new_epi: float, *, apply_clip: bool = True
) -> float:
    """Return a finite, structurally bounded EPI proposal without committing it."""

    from ..dynamics.structural_clip import structural_clip

    new_epi_float = _finite_operator_scalar(new_epi, "EPI proposal")
    if not apply_clip:
        return new_epi_float

    graph_attrs = getattr(node, "graph", {})
    epi_min = _finite_operator_scalar(
        graph_attrs.get("EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)), "EPI_MIN"
    )
    epi_max = _finite_operator_scalar(
        graph_attrs.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)), "EPI_MAX"
    )
    if epi_min > epi_max:
        raise TNFRValueError(
            "EPI_MIN must be less than or equal to EPI_MAX",
            context={"EPI_MIN": epi_min, "EPI_MAX": epi_max},
        )

    clipped_epi = structural_clip(
        new_epi_float,
        lo=epi_min,
        hi=epi_max,
        mode=(
            str(graph_attrs.get("CLIP_MODE", "hard"))
            if str(graph_attrs.get("CLIP_MODE", "hard")) in ("hard", "soft")
            else "hard"
        ),
        record_stats=False,
    )
    return _finite_operator_scalar(clipped_epi, "clipped EPI proposal")


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
    proposal = _validated_epi_assignment_value(
        node, new_epi, apply_clip=apply_clip
    )
    node.EPI = proposal


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
        Desired contraction scale factor (e.g., NUL_scale = 0.9)
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

    In practice, with NUL_scale = 0.9 < 1.0:
    - Positive EPI contracts toward zero: safe
    - Negative EPI contracts toward zero: safe

    Edge-awareness is provided for completeness and future extensibility.

    Examples
    --------
    >>> # Normal contraction (always safe with scale < 1.0)
    >>> _compute_nul_edge_aware_scale(0.5, 0.9, -1.0, 1e-12)
    0.9
    >>> _compute_nul_edge_aware_scale(-0.5, 0.9, -1.0, 1e-12)
    0.9
    """
    # With NUL_scale < 1.0, contraction moves toward zero (always safe)
    # No adaptation needed in typical case
    return scale


def _make_scale_op(glyph: Glyph) -> GlyphOperation:
    def _op(node: NodeProtocol, gf: GlyphFactors) -> None:
        key = "VAL_scale" if glyph is Glyph.VAL else "NUL_scale"
        default = _SCALE_FACTORS[glyph]
        factor = get_factor(gf, key, default)

        # Resolve every state-dependent proposal before committing any channel.
        vf_before = _finite_operator_scalar(node.vf, "nu_f before scale operator")
        vf_after = _finite_operator_scalar(
            vf_before * factor, f"{glyph.value} nu_f proposal"
        )
        if vf_after < 0.0:
            raise TNFRValueError(
                f"{glyph.value} must preserve nonnegative structural frequency"
            )

        dnfr_before: float | None = None
        dnfr_after: float | None = None
        densification_factor: float | None = None
        inverse_residual: float | None = None
        if glyph is Glyph.NUL:
            # The pressure coefficient is the materialized reciprocal of the
            # resolved capacity contraction. It is a derived relation, not a
            # second tuning degree of freedom.
            densification_factor = _finite_operator_scalar(
                1.0 / factor, "NUL inverse densification coefficient"
            )
            if "NUL_densification_factor" in gf:
                configured = get_factor(
                    gf, "NUL_densification_factor", densification_factor
                )
                if configured != densification_factor:
                    raise GlyphFactorValidationError(
                        "NUL_densification_factor is derived, not independent: "
                        f"expected {densification_factor!r}, got {configured!r}"
                    )
            dnfr_before = _finite_operator_scalar(
                node.dnfr, "DeltaNFR before Contraction"
            )
            dnfr_after = _finite_operator_scalar(
                dnfr_before * densification_factor,
                "NUL DeltaNFR proposal",
            )
            # This is a binary64 diagnostic. The ideal-real coefficient product
            # is one; its materialized multiplication need not equal one exactly.
            inverse_residual = factor * densification_factor - 1.0

        edge_aware_enabled = bool(
            node.graph.get(
                "EDGE_AWARE_ENABLED", DEFAULTS.get("EDGE_AWARE_ENABLED", True)
            )
        )
        epi_before: float | None = None
        epi_after: float | None = None
        scale_eff: float | None = None
        epsilon: float | None = None
        if edge_aware_enabled:
            epsilon = _finite_operator_scalar(
                node.graph.get(
                    "EDGE_AWARE_EPSILON",
                    DEFAULTS.get("EDGE_AWARE_EPSILON", 1e-12),
                ),
                "EDGE_AWARE_EPSILON",
            )
            if epsilon <= 0.0:
                raise TNFRValueError("EDGE_AWARE_EPSILON must be positive")
            epi_min = _finite_operator_scalar(
                node.graph.get("EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)),
                "EPI_MIN",
            )
            epi_max = _finite_operator_scalar(
                node.graph.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)),
                "EPI_MAX",
            )
            if epi_min > epi_max:
                raise TNFRValueError("EPI_MIN must not exceed EPI_MAX")
            epi_before = _finite_real_epi(
                node.EPI,
                "target EPI state",
                operator="Expansion" if glyph is Glyph.VAL else "Contraction",
            )

            # Compute edge-aware scale factor
            if glyph is Glyph.VAL:
                magnitude_bound = epi_max if epi_before >= 0.0 else abs(epi_min)
                scale_eff = _compute_val_edge_aware_scale(
                    epi_before, factor, magnitude_bound, epsilon
                )
            else:  # Glyph.NUL
                scale_eff = _compute_nul_edge_aware_scale(
                    epi_before, factor, epi_min, epsilon
                )
            scale_eff = _finite_operator_scalar(
                scale_eff, f"{glyph.value} effective EPI scale"
            )
            raw_epi_after = _finite_operator_scalar(
                epi_before * scale_eff, f"{glyph.value} EPI proposal"
            )
            from ..dynamics.structural_clip import structural_clip

            clip_mode = str(node.graph.get("CLIP_MODE", "hard"))
            if clip_mode not in ("hard", "soft"):
                clip_mode = "hard"
            epi_after = _finite_operator_scalar(
                structural_clip(
                    raw_epi_after,
                    lo=epi_min,
                    hi=epi_max,
                    mode=clip_mode,  # type: ignore[arg-type]
                    record_stats=False,
                ),
                f"{glyph.value} bounded EPI proposal",
            )

        # Atomic commit after every factor, bound and proposal has passed.
        node.vf = vf_after
        if glyph is Glyph.NUL:
            assert dnfr_before is not None
            assert dnfr_after is not None
            assert densification_factor is not None
            node.dnfr = dnfr_after
            node.graph.setdefault("nul_densification_log", []).append(
                {
                    "dnfr_before": dnfr_before,
                    "dnfr_after": dnfr_after,
                    "densification_factor": densification_factor,
                    "contraction_scale": factor,
                    "derived_inverse_coefficient": True,
                    "binary64_inverse_product_residual": inverse_residual,
                }
            )
        if epi_after is not None:
            _set_epi_with_boundary_check(node, epi_after, apply_clip=False)
            assert epi_before is not None
            assert scale_eff is not None
            assert epsilon is not None
            if abs(scale_eff - factor) > epsilon:
                node.graph.setdefault("edge_aware_interventions", []).append(
                    {
                        "glyph": glyph.name if hasattr(glyph, "name") else str(glyph),
                        "epi_before": epi_before,
                        "epi_after": epi_after,
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

    The glyph keeps EPI, νf, and phase fixed while reorganizing ΔNFR by the
    signed second derivative of EPI. Positive and negative acceleration therefore
    move structural pressure in their respective directions.

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
    a = get_factor(gf, "THOL_accel", COUPLING_GENTLE)
    dnfr = _finite_operator_scalar(node.dnfr, "THOL DeltaNFR state")
    d2_epi = _finite_operator_scalar(
        getattr(node, "d2EPI", 0.0), "THOL d2EPI state"
    )
    contribution = _finite_operator_scalar(
        a * d2_epi, "THOL DeltaNFR contribution"
    )
    proposal = _finite_operator_scalar(
        dnfr + contribution, "THOL DeltaNFR proposal"
    )
    node.dnfr = proposal


def _op_ZHIR(node: NodeProtocol, gf: GlyphFactors) -> None:  # ZHIR — Mutation
    """Apply canonical phase transformation θ → θ' based on structural dynamics.

    ZHIR (Mutation) implements the canonical TNFR phase transformation whose
    DIRECTION and firing are governed by the node's reorganization state (ΔNFR),
    implementing the physics: θ → θ' when ΔEPI/Δt > ξ (AGENTS.md §11,
    TNFR.pdf §2.2.11). Per the operator contract, ZHIR acts on the θ (phase)
    channel; ΔNFR enters only through its SIGN (rotation direction) and the
    bifurcation threshold (whether ZHIR fires, U4b). The shift MAGNITUDE is a
    calibration constant, not a function of |ΔNFR| — only channel and direction
    are canonical (the shift magnitude is an operational calibration value;
    only π is a genuine structural scale).

    **Canonical Behavior**:
    - Direction: Based on ΔNFR sign (positive → forward phase, negative → backward)
    - Magnitude: Calibrated constant theta_shift_factor · (π/4); independent of |ΔNFR|
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
        theta_before = _finite_operator_scalar(node.theta, "ZHIR phase state")
        theta_new = _finite_operator_scalar(
            ((theta_before % math.tau) + (shift % math.tau)) % math.tau,
            "ZHIR phase proposal",
        )
        node.theta = theta_new
        # Store telemetry for fixed shift mode
        storage = node._glyph_storage()
        storage["_zhir_theta_shift"] = shift
        storage["_zhir_fixed_mode"] = True
        return

    # Canonical transformation: θ → θ' based on ΔNFR
    theta_before = _finite_operator_scalar(node.theta, "ZHIR phase state")
    theta_before = theta_before % math.tau
    dnfr = _finite_operator_scalar(node.dnfr, "ZHIR DeltaNFR state")

    # Transformation magnitude controlled by factor
    theta_shift_factor = get_factor(gf, "ZHIR_theta_shift_factor", INV_PI)

    # Direction based on ΔNFR sign (coherent with structural pressure)
    # Magnitude is a calibration constant (theta_shift_factor · π/4); ΔNFR enters
    # only via its SIGN (direction) and the U4b firing threshold, NOT as |ΔNFR|.
    base_shift = math.pi / 4
    shift = _finite_operator_scalar(
        theta_shift_factor * math.copysign(1.0, dnfr) * base_shift,
        "ZHIR phase shift",
    )

    # Apply transformation with phase wrapping [0, 2π)
    theta_new = _finite_operator_scalar(
        (theta_before + (shift % math.tau)) % math.tau,
        "ZHIR phase proposal",
    )
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
    dnfr = _finite_operator_scalar(node.dnfr, "NAV DeltaNFR state")
    vf = _finite_operator_scalar(node.vf, "NAV nu_f state")
    strict = bool(node.graph.get("NAV_STRICT", False))
    if strict:
        base = vf
    else:
        eta = get_factor(gf, "NAV_eta", 0.5)
        sign = 1.0 if dnfr >= 0 else -1.0
        target = sign * vf
        base = _finite_operator_scalar(
            (1.0 - eta) * dnfr + eta * target,
            "NAV deterministic proposal",
        )
    base = _finite_operator_scalar(base, "NAV deterministic proposal")
    j = get_factor(gf, "NAV_jitter", COUPLING_FINE)
    if bool(node.graph.get("NAV_RANDOM", True)):
        with _rollback_jitter_progress_on_error(node):
            jitter = _finite_operator_scalar(
                random_jitter(node, j), "NAV jitter sample"
            )
            proposal = _finite_operator_scalar(
                base + jitter, "NAV DeltaNFR proposal"
            )
            if proposal == dnfr:
                raise TNFRValueError(
                    "NAV must change DeltaNFR",
                    context={"DeltaNFR_before": dnfr, "DeltaNFR_after": proposal},
                )
            node.dnfr = proposal
    else:
        jitter = j * (1 if base >= 0 else -1)
        proposal = _finite_operator_scalar(
            base + jitter, "NAV DeltaNFR proposal"
        )
        if proposal == dnfr:
            raise TNFRValueError(
                "NAV must change DeltaNFR",
                context={"DeltaNFR_before": dnfr, "DeltaNFR_after": proposal},
            )
        node.dnfr = proposal


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


def _resolve_glyph_operation(glyph: Glyph | str) -> tuple[Glyph, GlyphOperation]:
    """Resolve public names and glyph aliases before touching runtime state."""
    from .grammar_types import function_name_to_glyph, glyph_function_name

    g = function_name_to_glyph(glyph_function_name(glyph))
    if g is None:
        raise TNFRValueError(f"unknown glyph: {glyph}", context={"glyph": glyph})
    op = GLYPH_OPERATIONS.get(g)
    if op is None:
        raise TNFRValueError(
            f"glyph has no registered operator: {g}", context={"glyph": g}
        )
    return g, op


def _validated_execution_window(subject: Any, window: int | None) -> int:
    """Validate the trace bound before an operator can change the node."""
    from ..validation.window import validate_window

    if window is None:
        window = get_param(subject, "GLYPH_HYSTERESIS_WINDOW")
    return validate_window(window)


def _validate_u3_graph_application(
    G: TNFRGraph, node: NodeId, glyph: Glyph
) -> None:
    """Reject an inadmissible concrete UM/RA request before NodeNX caching."""
    if glyph not in (Glyph.UM, Glyph.RA):
        return
    from .grammar_types import glyph_function_name
    from .preconditions import OperatorPreconditionError, validate_phase_gate_u3

    operator = glyph_function_name(glyph)
    try:
        validate_phase_gate_u3(G, node, operator)
    except OperatorPreconditionError as exc:
        raise TNFRValueError(
            str(exc),
            context={
                "operator": operator,
                "failed_condition": "u3_phase_compatibility",
            },
        ) from exc


def apply_glyph_obj(
    node: NodeProtocol, glyph: Glyph | str, *, window: int | None = None
) -> None:
    """Apply a canonical name or glyph to a :class:`NodeProtocol` object.

    Argument and history validation precede the structural operation. This
    low-level primitive does not make arbitrary operator failures transactional.
    """
    from .grammar_debt import require_replayable_history

    g, op = _resolve_glyph_operation(glyph)
    window = _validated_execution_window(node, window)
    validate_graph_seed(node)
    require_replayable_history(node._glyph_storage().get("glyph_history"))
    # Resolve the current operator's complete factor domain before any channel,
    # history, metric, or cache can be changed.
    gf = get_glyph_factors(node, g)
    if g is Glyph.ZHIR:
        from ._mutation_gate import validate_mutation_runtime_gate

        validate_mutation_runtime_gate(node._glyph_storage(), node.graph)
    op(node, gf)
    storage = node._glyph_storage()
    glyph_history.push_glyph(storage, g.value, window)
    # Structural identity and operator provenance are independent channels.
    # The ordered history is authoritative; source_glyph is the serialized
    # single-value fallback and must never overwrite epi_kind.
    set_attr_str(storage, ALIAS_SOURCE_GLYPH, g.value)


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
        raise TNFRValueError(
            f"Invalid parameters for apply_glyph: {e}", context={"error": str(e)}
        ) from e

    # Reject invalid requests before constructing/caching a NodeNX wrapper.
    glyph, _ = _resolve_glyph_operation(glyph)
    window = _validated_execution_window(G, window)
    validate_graph_seed(G)
    from .grammar_debt import require_replayable_history

    require_replayable_history(G.nodes[n].get("glyph_history"))
    # Validate the operator's numerical contract before NodeNX construction,
    # because the adapter is cached in graph metadata.
    resolve_runtime_operator_factors(
        G.graph.get("GLYPH_FACTORS"), glyph, G.graph
    )
    _validate_u3_graph_application(G, n, glyph)
    validate_affine_epi_graph_input(G, n, glyph)
    if glyph is Glyph.ZHIR:
        from ._mutation_gate import validate_mutation_runtime_gate

        # Validate before NodeNX construction so a rejected mutation leaves no
        # adapter cache or other graph metadata behind.
        validate_mutation_runtime_gate(G.nodes[n], G.graph)
    NodeNX = get_nodenx()
    if NodeNX is None:
        raise ImportError("NodeNX is unavailable")
    node = NodeNX(G, n)
    apply_glyph_obj(node, glyph, window=window)
