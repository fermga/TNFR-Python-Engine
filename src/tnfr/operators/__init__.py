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
from ._mutation_stage_kernel import propose_mutation_stage
from ._neighbor_epi_kernel import (
    dominant_neighbor_epi_kind,
    neighbor_epi_blend_value,
    neighbor_epi_unweighted_mean,
    reception_proposed_epi_kind,
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
from ._scale_operator_kernel import (
    compute_nul_edge_aware_scale as _compute_nul_edge_aware_scale,
    compute_val_edge_aware_scale as _compute_val_edge_aware_scale,
    edge_aware_intervention_event,
    nul_densification_event,
    propose_scale_operator,
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
from .event_remesh_runtime import (
    EventRemeshCycleResult,
    RemeshHistoryTransitionObservation,
    WeightedEPIObservation,
    execute_event_remesh_cycle,
)
from .event_remesh_sequence import (
    EventRemeshCycleBoundaryObservation,
    ObservedEventRemeshCycleSequence,
    compose_event_remesh_cycle_observations,
)
from .event_runtime import (
    ExecutedGlyphStage,
    ExecutedNodalFlowInterval,
    ExecutedOperatorEvent,
    ExecutedPressureRefreshedFlowPartition,
    ObservedRepresentedEPIScheduleComposition,
    PhysicalEulerModalObservation,
    PressureRefreshBoundaryObservation,
    RepresentedEPIScheduleOperation,
    OperatorEventExecutionResult,
    execute_operator_event_schedule,
)
from .event_timing import (
    OperatorEventRuntimeClockDiagnostic,
    OperatorEventSchedule,
    PhysicalFlowPartition,
    ScheduledOperatorEvent,
    StructuralFlowInterval,
    build_physical_flow_partition,
    build_operator_event_schedule,
    diagnose_operator_event_runtime_clock,
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
    DelayedRemeshNodeProposal,
    DelayedRemeshPlan,
    DelayedRemeshResult,
    DelayedRemeshStabilityEvidence,
    apply_network_remesh,
    apply_remesh_if_globally_stable,
    apply_topological_remesh,
    plan_network_remesh,
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
    "EventRemeshCycleResult",
    "RemeshHistoryTransitionObservation",
    "WeightedEPIObservation",
    "execute_event_remesh_cycle",
    "EventRemeshCycleBoundaryObservation",
    "ObservedEventRemeshCycleSequence",
    "compose_event_remesh_cycle_observations",
    "ExecutedGlyphStage",
    "ExecutedNodalFlowInterval",
    "ExecutedOperatorEvent",
    "ExecutedPressureRefreshedFlowPartition",
    "ObservedRepresentedEPIScheduleComposition",
    "PhysicalEulerModalObservation",
    "PressureRefreshBoundaryObservation",
    "RepresentedEPIScheduleOperation",
    "OperatorEventExecutionResult",
    "execute_operator_event_schedule",
    "OperatorEventRuntimeClockDiagnostic",
    "OperatorEventSchedule",
    "PhysicalFlowPartition",
    "ScheduledOperatorEvent",
    "StructuralFlowInterval",
    "build_physical_flow_partition",
    "build_operator_event_schedule",
    "diagnose_operator_event_runtime_clock",
    "GLYPH_OPERATIONS",
    "apply_glyph_obj",
    "apply_glyph",
    "apply_glyph_with_grammar",
    "GrammarObservation",
    "observe_grammar",
    "StructuralPotentialConfinementObservation",
    "observe_structural_potential_confinement",
    "apply_network_remesh",
    "plan_network_remesh",
    "DelayedRemeshNodeProposal",
    "DelayedRemeshPlan",
    "DelayedRemeshResult",
    "DelayedRemeshStabilityEvidence",
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
    return dominant_neighbor_epi_kind(
        ((float(v.EPI), v.epi_kind) for v in neigh), default_kind
    )


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
    new_epi = _finite_operator_scalar(
        neighbor_epi_blend_value(epi, epi_bar, mix), "EN EPI proposal"
    )
    final = reception_proposed_epi_kind(
        node.epi_kind,
        ((float(neighbor.EPI), neighbor.epi_kind) for neighbor in neigh),
        unclipped_target_epi=new_epi,
        fallback_kind=default_kind,
    )

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
    from ._coherence_stage_kernel import propose_coherence_pressure

    dnfr = _finite_operator_scalar(
        getattr(node, "dnfr", 0.0), "IL DeltaNFR state"
    )
    _before, proposal = propose_coherence_pressure(dnfr, factor)
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
    """Compute the finite circular mean through the shared UM kernel."""

    from ._coupling_stage_kernel import compute_consensus_phase as _kernel_mean

    return _kernel_mean(phases)


def _op_um_protocol_fallback(
    node: NodeProtocol, gf: GlyphFactors
) -> None:
    """Preserve the phase-only behavior of graphless NodeProtocol objects."""

    theta_push = get_factor(gf, "UM_theta_push", EN_MIX_FACTOR)
    selection, neighbors = _runtime_u3_neighbors(node, "UM")
    bidirectional = bool(node.graph.get("UM_BIDIRECTIONAL", True))
    inputs = (
        [selection.target_phase, *selection.phases]
        if bidirectional
        else list(selection.phases)
    )
    consensus = compute_consensus_phase(inputs)
    target_phase = _finite_operator_scalar(
        (
            selection.target_phase
            + theta_push
            * angle_diff(consensus, selection.target_phase)
        )
        % math.tau,
        "UM target phase proposal",
    )
    if bidirectional:
        neighbor_phases = tuple(
            _finite_operator_scalar(
                (
                    phase
                    + theta_push * angle_diff(consensus, phase)
                )
                % math.tau,
                "UM neighbor phase proposal",
            )
            for phase in selection.phases
        )
    else:
        neighbor_phases = ()

    node.theta = target_phase
    for neighbor, phase in zip(
        neighbors, neighbor_phases, strict=True
    ):
        neighbor.theta = phase


def _op_UM(node: NodeProtocol, gf: GlyphFactors) -> None:  # UM - Coupling
    """Synchronize U3-compatible phases and optionally form valid links.

    NetworkX-backed nodes use the same pure proposal and merge policy as the
    all-target stage. A direct call has one target, so its phase result matches
    the established single-target rule apart from canonical normalization and
    final U3 revalidation. Graphless protocol objects retain the historical
    phase-only behavior.
    """

    if not hasattr(node, "G"):
        _op_um_protocol_fallback(node, gf)
        return

    from ._coupling_stage_kernel import propose_coupling_stage
    from .network_stage import (
        GraphTransactionSnapshot,
        _commit_coupling_structure,
    )

    transaction = GraphTransactionSnapshot(node.G)
    try:
        functional_links = bool(
            node.graph.get("UM_FUNCTIONAL_LINKS", True)
        )
        # Direct and staged public entry points both validate the graph seed
        # before dispatch, even when links are disabled. Keep that shared
        # argument contract here for standalone calls of this private helper.
        configured_seed = validate_graph_seed(node)
        resolved_seed: int | None = None
        node_offsets: dict[Any, int] = {}
        if functional_links:
            if configured_seed is None:
                import random

                resolved_seed = random.SystemRandom().getrandbits(64)
            else:
                resolved_seed = configured_seed
            node_offsets[node.n] = node.offset()

        stage = propose_coupling_stage(
            node.G,
            (node.n,),
            gf,
            resolved_seed=resolved_seed,
            node_offsets=node_offsets,
        )

        if resolved_seed is not None:
            if validate_graph_seed(node) != configured_seed:
                raise RuntimeError(
                    "stale UM random seed: configuration changed before commit"
                )
            if configured_seed is None:
                node.graph["RANDOM_SEED"] = resolved_seed
        _commit_coupling_structure(node.G, stage)
    except BaseException as failure:
        transaction.restore_after_failure(node.G, failure)
        raise


def _op_RA(node: NodeProtocol, gf: GlyphFactors) -> None:  # RA — Resonance
    """Propagate scalar EPI through U3-admissible resonance.

    Resonance (RA) blends the target EPI with its compatible neighborhood,
    proposes a phase step toward that subset, and may amplify structural
    frequency when a nonzero signal is present. Structural C(t) is an observed
    pressure/rate read-out: RA has no unconditional monotonic C(t) guarantee.

    **Canonical effects:**

    - **EPI propagation**: blends scalar EPI while preserving sign/kind identity
    - **νf amplification**: raises capacity when the propagation trigger is active
    - **Phase step**: moves the target toward the compatible circular mean
    - **C(t) telemetry**: optionally records the signed before/after difference
    - **Identity preservation**: maintains structural identity during propagation

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
    **νf amplification**: When the compatible-neighbor scalar EPI mean satisfies
    the configured nonzero trigger, ``node.vf`` is multiplied by
    ``1 + RA_vf_amplification``. This changes reorganization capacity; it does
    not by itself establish a change in C(t).

    **Phase step**: RA moves the target toward the compatible-neighbor circular
    mean by ``RA_phase_coupling``. Phase alignment and structural C(t) remain
    separate diagnostics because phase is not a direct argument of the C kernel.

    **Network C(t) tracking (optional)**: With ``TRACK_NETWORK_COHERENCE``, RA
    records canonical C(t) before and after execution plus their signed
    difference. The difference may be negative, zero, or positive.

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
        from ..metrics.common import compute_coherence

        c_before = float(compute_coherence(node.G))
        node.graph.setdefault("_ra_c_tracking", [])

    # Commit the already validated proposal.  The setter remains the canonical
    # storage boundary, while clipping is not repeated after the gate.
    if neigh:
        _set_epi_with_boundary_check(node, proposed_epi, apply_clip=False)
    node.epi_kind = proposed_kind
    epi_bar_result = epi_bar
    kind_result = proposed_kind

    # CANONICAL EFFECT 1: νf amplification through resonance
    # This is always active - it's a fundamental property of resonance per TNFR theory
    # Only amplify when the compatible-neighbor EPI trigger is active
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

    # Record the signed network C(t) change if optional telemetry is enabled
    if track_coherence and c_before is not None and hasattr(node, "G"):
        from ..metrics.common import compute_coherence

        c_after = float(compute_coherence(node.G))
        node.graph["_ra_c_tracking"].append(
            {
                "node": getattr(node, "n", None),
                "c_before": c_before,
                "c_after": c_after,
                "c_delta": c_after - c_before,
            }
        )


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


def _make_scale_op(glyph: Glyph) -> GlyphOperation:
    def _op(node: NodeProtocol, gf: GlyphFactors) -> None:
        key = "VAL_scale" if glyph is Glyph.VAL else "NUL_scale"
        default = _SCALE_FACTORS[glyph]
        factor = get_factor(gf, key, default)
        edge_aware_enabled = bool(
            node.graph.get(
                "EDGE_AWARE_ENABLED", DEFAULTS.get("EDGE_AWARE_ENABLED", True)
            )
        )
        proposal = propose_scale_operator(
            glyph=glyph,
            factor=factor,
            vf_before=node.vf,
            dnfr_before=node.dnfr if glyph is Glyph.NUL else None,
            configured_densification_factor=(
                gf.get("NUL_densification_factor")
                if glyph is Glyph.NUL
                and "NUL_densification_factor" in gf
                else None
            ),
            edge_aware_enabled=edge_aware_enabled,
            epi_before=node.EPI if edge_aware_enabled else None,
            epi_min=node.graph.get(
                "EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)
            ),
            epi_max=node.graph.get(
                "EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)
            ),
            epsilon=node.graph.get(
                "EDGE_AWARE_EPSILON",
                DEFAULTS.get("EDGE_AWARE_EPSILON", 1e-12),
            ),
            clip_mode=str(node.graph.get("CLIP_MODE", "hard")),
        )

        # Atomic commit after every factor, bound and proposal has passed.
        node.vf = proposal.vf_after
        if glyph is Glyph.NUL:
            assert proposal.dnfr_before is not None
            assert proposal.dnfr_after is not None
            assert proposal.densification_factor is not None
            node.dnfr = proposal.dnfr_after
            node.graph.setdefault("nul_densification_log", []).append(
                nul_densification_event(proposal, getattr(node, "n", None))
            )
        if proposal.write_epi:
            assert proposal.epi_after is not None
            _set_epi_with_boundary_check(
                node, proposal.epi_after, apply_clip=False
            )
            if proposal.edge_aware_adapted:
                assert proposal.epi_before is not None
                assert proposal.effective_epi_scale is not None
                node.graph.setdefault("edge_aware_interventions", []).append(
                    edge_aware_intervention_event(
                        proposal, getattr(node, "n", None)
                    )
                )

    _op.__doc__ = """{} glyph proposes a target-local capacity scale.

        VAL increases νf and scales signed EPI away from zero; NUL decreases νf,
        scales EPI toward zero, and densifies |ΔNFR|. EPI is changed only when
        EDGE_AWARE_ENABLED is true. Boundary projection then keeps it within the
        configured structural interval.

        When EDGE_AWARE_ENABLED is True (default), the effective scale is computed as:
        - VAL: scale_eff = min(VAL_scale, sign-specific bound / |EPI_current|)
        - NUL: scale_eff = NUL_scale

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
    - RNG-free: Same validated state and configuration produce the same result

    The transformation preserves structural identity (epi_kind) while shifting the
    operational regime, enabling adaptation without losing coherence.

    Parameters
    ----------
    node : NodeProtocol
        Node whose phase is transformed based on its structural state.
    gf : GlyphFactors
        Supplies ``ZHIR_theta_shift_factor`` (default: 1/π) controlling
        transformation magnitude. Can override with explicit
        ``ZHIR_theta_shift`` for fixed rotation.

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
    # Resolve only the active branch before building the pure phase proposal.
    if "ZHIR_theta_shift" in gf:
        proposal = propose_mutation_stage(
            node.theta,
            fixed_shift=get_factor(gf, "ZHIR_theta_shift", math.pi / 2),
        )
    else:
        proposal = propose_mutation_stage(
            node.theta,
            node.dnfr,
            theta_shift_factor=get_factor(
                gf, "ZHIR_theta_shift_factor", INV_PI
            ),
        )

    node.theta = proposal.theta_after
    storage = node._glyph_storage()
    for key, value in proposal.telemetry_items:
        storage[key] = value


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
    from ._recursivity_stage_kernel import (
        commit_recursivity_advisory,
        propose_recursivity_advisory,
    )

    proposal = propose_recursivity_advisory(node)
    commit_recursivity_advisory(node, proposal)
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
    """Resolve executable/display names or a glyph before touching state."""
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
