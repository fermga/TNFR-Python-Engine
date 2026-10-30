"""True-history compatibility of the actual connected post-SHA child.

This read-only analytical consumer acquires no samples and executes no event.
It keeps constitutive acceleration, prospective sample errors, current grammar
and strict form admission separate. A large derivative does not make the child
an admitted THOL actor.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I
from ..operators._thol_config import (
    resolve_thol_bifurcation_threshold,
    resolve_thol_min_epi,
)
from ..operators.grammar_dynamics import validate_candidate
from ..operators.nodal_equation import (
    StructuralAccelerationObservation,
    observe_structural_acceleration,
)
from ..operators.preconditions import OperatorPreconditionError
from ..operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from ..physics.relational_observations import (
    RelationalSampleJetBudget,
    bound_relational_sample_jet_budget,
)
from .maintained_parent_thol import _represented, _signature
from .maintained_sha_motion_source import validate_live_sha_successor
from .maintained_successor_chamber import _certify_chamber
from .maintained_temporal_window import NodeHistoryWindow, bound_node_history_window

CHILD = "2_sub_0"
PARENT = 2
CHILD_INDEX = 20
SAMPLE_STEP = Q(1, 2**16)
HORIZON = 2 * SAMPLE_STEP
SAMPLE_ERROR_ALLOWANCE = Q(1, 2**30)


@dataclass(frozen=True)
class ChildHistoryPrediction:
    """A true window and a conditional sample budget, not measured readiness.

    ``prospective_exact_sample_acceleration_lower_bound`` covers sample-value
    errors only. It is not a certified runtime-reader result: acquisition,
    arithmetic and live endpoint/clock association remain separate obligations.
    The sample budget also retains an actual global third-derivative bound;
    its derivative-at-one-time truncation term is not added to a theorem that
    already bounds acceleration over the complete observation window.
    """

    nodes: tuple
    edges: tuple[tuple[int, int], ...]
    shifts: tuple[int, ...]
    child: str
    parent: int
    source_time: Q
    sample_step: Q
    elapsed_sample_times: tuple[Q, Q, Q]
    sample_times: tuple[Q, Q, Q]
    window: NodeHistoryWindow
    threshold: Q
    minimum_form: Q
    sample_error_allowance: Q
    sample_budget: RelationalSampleJetBudget
    prospective_exact_sample_acceleration_lower_bound: Q
    source_observation: StructuralAccelerationObservation
    source_child_glyph_history: tuple[str, ...]
    source_child_hierarchy_level: int
    grammar_allowed: bool
    grammar_error_rules: tuple[str, ...]
    grammar_errors: tuple[str, ...]
    blocked_u4b: bool
    strict_precondition_error: str | None
    minimum_form_excluded_whole_window: bool
    common_principal_identity_retained: bool
    temporal_capability: bool
    measured_history: bool
    reader_arithmetic_error_certified: bool
    actual_event: bool
    full_thol_readiness: bool


def predict_child_history(admitted):
    """Re-admit SHA once and bound one fixed child-owned history window.

    All 42 coordinates and actual held coefficients supply the bound. The
    freshly rebuilt SHA assessment supplies primitives, never cached caller
    flags or earlier trajectory coefficients. Fresh principal branches are
    reconstructed from the retained real lifts. No observation or grammar
    context is borrowed from the parent, and no pressure refresh is executed.
    """
    admitted = validate_live_sha_successor(admitted)
    graph = admitted.graph
    before = _signature(graph)
    fields = admitted.assessment["actual_fields"]
    nodes = fields["nodes"]
    positions = {node: index for index, node in enumerate(nodes)}
    weights = {
        tuple(sorted((positions[a], positions[b]))): weight
        for (a, b), weight in zip(fields["edges"], fields["conductances"], strict=True)
    }
    edges = tuple(sorted(weights))
    child_data = graph.nodes[CHILD]
    if (
        len(nodes) != 21
        or len(edges) != 53
        or nodes[CHILD_INDEX] != CHILD
        or child_data["parent_node"] != PARENT
        or tuple(graph[CHILD]) != (PARENT,)
    ):
        raise ValueError("child history requires the actual connected SHA successor")
    chamber = _certify_chamber(tuple(map(I, fields["phase"])), edges)
    window = bound_node_history_window(
        epi=fields["epi"],
        phases=fields["phase"],
        capacities=fields["capacity"],
        edges=edges,
        conductances=tuple(weights[edge] for edge in edges),
        shifts=chamber.shifts,
        horizon=HORIZON,
        target_index=CHILD_INDEX,
    )
    source_time = _represented(graph.graph["_t"], "source structural time")
    threshold = _represented(
        resolve_thol_bifurcation_threshold(graph.graph), "THOL threshold"
    )
    minimum_form = _represented(resolve_thol_min_epi(graph.graph), "THOL minimum EPI")
    budget = bound_relational_sample_jet_budget(
        sample_step=SAMPLE_STEP,
        sample_error_bounds=(SAMPLE_ERROR_ALLOWANCE,) * 3,
        third_derivative_bound=window.third_form_derivative_bound,
        observation_time=source_time,
    )
    lower = window.form_acceleration.lo - budget.acceleration_sample_error_bound
    observation = observe_structural_acceleration(graph, CHILD)
    grammar = validate_candidate(graph, CHILD, "THOL")
    errors = tuple(value for value in grammar.violations if value.severity == "error")
    strict_error = None
    try:
        validate_self_organization_strict(graph, CHILD, emit_warnings=False)
    except OperatorPreconditionError as exc:
        strict_error = str(exc)
    identity_retained = len(window.retained_edge_indices) == len(edges)
    capability = identity_retained and lower > threshold and window.pressure.lo > 0
    if _signature(graph) != before:
        raise ValueError("read-only child history admission changed persistent state")
    return ChildHistoryPrediction(
        nodes=nodes,
        edges=edges,
        shifts=chamber.shifts,
        child=CHILD,
        parent=PARENT,
        source_time=source_time,
        sample_step=SAMPLE_STEP,
        elapsed_sample_times=(Q(0), SAMPLE_STEP, HORIZON),
        sample_times=budget.sample_times,
        window=window,
        threshold=threshold,
        minimum_form=minimum_form,
        sample_error_allowance=SAMPLE_ERROR_ALLOWANCE,
        sample_budget=budget,
        prospective_exact_sample_acceleration_lower_bound=lower,
        source_observation=observation,
        source_child_glyph_history=tuple(child_data.get("glyph_history", ())),
        source_child_hierarchy_level=child_data["hierarchy_level"],
        grammar_allowed=grammar.allowed,
        grammar_error_rules=tuple(value.rule for value in errors),
        grammar_errors=tuple(value.message for value in errors),
        blocked_u4b=any(value.rule == "U4b" for value in errors),
        strict_precondition_error=strict_error,
        minimum_form_excluded_whole_window=window.form.hi < minimum_form,
        common_principal_identity_retained=identity_retained,
        temporal_capability=capability,
        measured_history=False,
        reader_arithmetic_error_certified=False,
        actual_event=False,
        full_thol_readiness=False,
    )
