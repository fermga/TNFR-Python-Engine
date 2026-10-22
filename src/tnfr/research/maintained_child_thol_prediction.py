"""Detached THOL prediction on the child's freshly acquired own history.

No proposal, metabolic implementation, event, refresh or trajectory is run.
Ideal algebra, represented arithmetic and a prospective implementation
acceptance budget remain separate. The successor law is an explicitly
supplied frozen-isolate completion, not a transferred future certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from math import tau

from .._binary64 import uses_ieee_binary64_rounding
from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR
from ..constants.canonical import COUPLING_MODERATE
from ..mathematics._rational_interval import I, pi_interval
from ..operators._thol_config import (
    resolve_thol_bifurcation_threshold,
    resolve_thol_min_epi,
)
from ..operators._thol_constants import THOL_CHILD_VF_DAMPING, THOL_SUB_EPI_SCALING
from ..operators.definitions import Dissonance, SelfOrganization, Silence
from ..operators.factor_contracts import resolve_runtime_operator_factors
from ..operators.grammar_dynamics import validate_candidate
from ..operators.grammar_execution import ValidatedSequence
from ..operators.nodal_equation import observe_structural_acceleration
from ..operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from ..types import Glyph
from .local_capacity_history import (
    CONTEXT,
    PRESSURE_ERROR_BUDGET,
    _fresh_pressure_bounds,
    capture,
)
from .maintained_child_history import CHILD as ACTOR
from .maintained_child_history import CHILD_INDEX, PARENT
from .maintained_child_history_source import load_child_history_source
from .maintained_parent_thol import _represented, _signature
from .maintained_parent_thol_prediction import (
    NUMERICAL_ALLOWANCE,
    MetabolicPrediction,
    _metabolic_prediction,
    _with_allowance,
)
from .maintained_successor_chamber import _certify_chamber

CHILD = "2_sub_0_sub_0"
SAMPLE_TIMES = tuple(Q(n, 524288) for n in (3682305, 3682817, 3683329))


@dataclass(frozen=True)
class ChildTholPrediction:
    """Conditional default reset, with no observed birth or future enclosure.

    Capacity and pressure represented predictions follow separate binary64
    operations. ``child_form_bounds`` instead adds a supplied acceptance
    allowance to independent mathematical bounds; it does not prove a libm
    error bound. All zero jumps below are algebraic unchanged-argument
    identities, conditional on the specified isolated birth and full refresh.
    ``source_edges`` and conductances retain graph order. Principal shifts and
    margins instead align with ``source_principal_edges``, canonical sorted
    pairs of indices into ``source_nodes``.
    The separate pressure-square diagnostic uses ``raw_pressure_ideal``;
    pressure is not an argument of the selected edge-storage law.
    """

    actor: str
    child: str
    source_time: Q
    source_nodes: tuple
    source_edges: tuple
    source_conductances: tuple[Q, ...]
    source_forms: tuple[Q, ...]
    source_phases: tuple[Q, ...]
    source_capacities: tuple[Q, ...]
    source_degrees: tuple[int, ...]
    source_stored_pressures: tuple[Q, ...]
    source_pressure_defects: tuple[I, ...]
    source_principal_edges: tuple[tuple[int, int], ...]
    source_principal_shifts: tuple[int, ...]
    source_principal_margins: tuple[Q, ...]
    successor_nodes: tuple
    successor_edges: tuple
    neighbors: tuple
    actor_form: Q
    actor_phase: Q
    actor_capacity: Q
    actor_pressure: Q
    sample_times: tuple[Q, Q, Q]
    observed_acceleration: Q
    threshold: Q
    minimum_form: Q
    strict_preconditions_admitted: bool
    grammar_allowed: bool
    pressure_gain: Q
    exact_pressure_increment: Q
    represented_pressure_increment: Q
    raw_pressure_ideal: Q
    raw_pressure_represented: Q
    pressure_rounding_defect: Q
    raw_pressure_bounds: I
    pressure_square_storage_jump: Q
    metabolic: MetabolicPrediction
    ideal_phase_variance: Q
    ideal_child_form: Q
    child_form_bounds: I
    child_capacity_ideal: Q
    child_capacity: Q
    capacity_rounding_defect: Q
    child_phase: Q
    child_pressure: Q
    child_degree: int
    child_hierarchy_level: int
    child_hierarchy_path: tuple
    child_record_operator_step: int
    expected_hierarchy: tuple
    expected_actor_glyph_history: tuple[str, ...]
    expected_actor_u2_debt: int
    child_legacy_history_length: int
    child_timed_history_length: int
    child_glyph_history: tuple[str, ...]
    child_immediate_thol_admitted: bool
    event_storage_jump: Q
    old_component_mean_weights: tuple[Q, ...]
    old_component_mean_jumps: tuple[Q, Q]
    old_component_rows_unchanged: bool
    isolate_form_rate: Q
    isolate_phase_rate: Q
    global_positive_mean_metric_available: bool
    numerical_allowance: Q = NUMERICAL_ALLOWANCE
    successor_law: str = "supplied_frozen_isolate_completion"
    actual_event: bool = False
    complete_word_executed: bool = False
    original_retention_transferred_to_acquired_point: bool = False
    future_from_acquired_point_certified: bool = False
    cross_birth_u5_u6_certified: bool = False


def _admit_context(graph):
    data, actor = graph.graph, graph.nodes[ACTOR]
    if any(str(key).startswith("THOL_") for key in data):
        raise ValueError("selected child prediction requires absent THOL overrides")
    if "BIFURCATION_THRESHOLD_TAU" in data:
        raise ValueError("selected child prediction requires the default threshold")
    factors = data.get("GLYPH_FACTORS")
    if (
        type(factors) is not dict
        or set(factors) != {"VAL_scale"}
        or _represented(factors["VAL_scale"], "VAL factor") != 128
        or data.get("GRAMMAR_REJECTION_MODE") != "raise"
        or data.get("VALIDATE_OPERATOR_PRECONDITIONS") is not True
        or data.get("EDGE_AWARE_ENABLED") is not False
        or data.get("VALIDATE_NODAL_EQUATION", False) is not False
    ):
        raise ValueError(
            "selected child prediction requires unchanged execution policy"
        )
    if (
        _represented(data.get("EPI_MIN"), "EPI_MIN") != -1
        or _represented(data.get("EPI_MAX"), "EPI_MAX") != 1
        or _represented(data.get("_t"), "structural time") != SAMPLE_TIMES[-1]
    ):
        raise ValueError(
            "selected child prediction requires its rails and endpoint clock"
        )
    hierarchy = data.get("hierarchy")
    if (
        type(hierarchy) is not dict
        or tuple(hierarchy) != (PARENT,)
        or type(hierarchy[PARENT]) is not list
        or hierarchy[PARENT] != [ACTOR]
        or graph.nodes[PARENT].get("sub_nodes") != [ACTOR]
        or type(actor.get("parent_node")) is not int
        or actor["parent_node"] != PARENT
        or type(actor.get("_hierarchy_path")) is not list
        or actor["_hierarchy_path"] != [PARENT]
        or any(type(x) is not int for x in actor["_hierarchy_path"])
    ):
        raise ValueError("selected source requires the retained parent-child hierarchy")
    for key in (
        "hierarchy_level",
        "_bifurcation_level",
        "_operator_step",
        "_grammar_u2_debt",
    ):
        if type(actor.get(key)) is not int or actor[key] != 1:
            raise ValueError(f"selected actor requires actual {key}=1")
    for key in ("sub_nodes", "sub_epis"):
        value = actor.get(key, [])
        if type(value) is not list or value:
            raise ValueError("selected actor requires no existing child records")
    if (
        tuple(actor.get("glyph_history", ())) != ("OZ",)
        or actor.get("_grammar_prior_coherence") is not False
        or CHILD in graph
    ):
        raise ValueError(
            "selected actor requires its own OZ prefix and unused child ID"
        )


def predict_child_thol(evidence_dir, *, repository_root):
    """Reload primitive evidence before predicting the selected child THOL.

    A mutable report is deliberately not a public input. Artifact association
    and acquired-history certification belong to the source reader; this
    consumer rebuilds the live fields, observation and context once more.
    """
    admitted = load_child_history_source(evidence_dir, repository_root=repository_root)
    return _predict_child_thol(admitted)


def _predict_child_thol(admitted):
    """Internal same-call composition, not admission of an arbitrary cached report."""
    graph = admitted.graph
    before = _signature(graph)
    fields = capture(graph)
    if (
        fields.nodes != (*range(20), ACTOR)
        or len(fields.edges) != 53
        or tuple(graph.neighbors(ACTOR)) != (PARENT,)
        or fields.degrees[CHILD_INDEX] != 1
        or any(value <= 0 for value in fields.capacity)
        or any(value <= 0 for value in fields.conductances)
    ):
        raise ValueError(
            "selected prediction requires the actual degree-one child source"
        )
    _admit_context(graph)
    if not uses_ieee_binary64_rounding():
        raise ValueError(
            "selected represented prediction requires IEEE binary64 rounding"
        )
    pressure = tuple(
        _represented(
            get_attr(
                graph.nodes[node], ALIAS_DNFR, strict=True, conv=lambda value: value
            ),
            "stored pressure",
        )
        for node in fields.nodes
    )
    defects = tuple(
        I(value) - bound
        for value, bound in zip(pressure, _fresh_pressure_bounds(fields), strict=True)
    )
    if any(bound.abs_max > PRESSURE_ERROR_BUDGET for bound in defects):
        raise ValueError(
            "stored pressure is inconsistent with the admitted full sine field"
        )
    observation = observe_structural_acceleration(graph, ACTOR)
    if (
        not observation.available
        or observation.source != "epi_time_history"
        or observation.history_length != 3
        or observation.current_endpoint_matches_state is not True
        or tuple(_represented(row[0], "sample time") for row in observation.samples)
        != SAMPLE_TIMES
    ):
        raise ValueError("selected actor requires its live three-sample acquisition")
    acceleration = _represented(observation.value, "observed acceleration")
    threshold = _represented(
        resolve_thol_bifurcation_threshold(graph.graph), "threshold"
    )
    minimum_form = _represented(resolve_thol_min_epi(graph.graph), "minimum form")
    if abs(acceleration) <= threshold:
        raise ValueError("actual acceleration does not trigger the selected birth")
    validate_self_organization_strict(graph, ACTOR, emit_warnings=False)
    sequence = ValidatedSequence(
        (Dissonance(), SelfOrganization(), Silence()), context=CONTEXT
    )
    grammar = validate_candidate(
        graph, ACTOR, "THOL", sequence_context=sequence.step(1)
    )
    if not grammar.allowed:
        raise ValueError("actual child context does not admit THOL")
    x, theta, nu = (
        values[CHILD_INDEX] for values in (fields.epi, fields.phase, fields.capacity)
    )
    if not Q(0) <= theta < Q(tau):
        raise ValueError("selected birth requires the unchanged positive phase chart")
    metabolic = _metabolic_prediction(
        x, theta, (fields.epi[PARENT],), (fields.phase[PARENT],)
    )
    # One mathematical unit phasor has modulus one, without interval dependency.
    ideal_form = Q(THOL_SUB_EPI_SCALING) * x + Q(COUPLING_MODERATE) * (
        fields.epi[PARENT] - x
    )
    form_bounds = _with_allowance(metabolic.child_form)
    if not (0 < form_bounds.lo <= ideal_form <= form_bounds.hi < minimum_form):
        raise ValueError("selected child amplitude leaves its admitted interior domain")
    gain = _represented(
        resolve_runtime_operator_factors(
            graph.graph.get("GLYPH_FACTORS"), Glyph.THOL, graph.graph
        )["THOL_accel"],
        "THOL gain",
    )
    increment = gain * acceleration
    increment_rn = finite_represented_real(increment, "predicted pressure increment")[1]
    raw = pressure[CHILD_INDEX] + increment
    raw_rn = finite_represented_real(
        pressure[CHILD_INDEX] + increment_rn, "predicted raw pressure"
    )[1]
    ideal_capacity = nu * Q(THOL_CHILD_VF_DAMPING)
    capacity = finite_represented_real(ideal_capacity, "predicted child capacity")[1]
    if capacity <= 0 or abs(raw_rn - raw) > NUMERICAL_ALLOWANCE:
        raise ValueError(
            "selected represented reset leaves the prospective numeric budget"
        )
    positions = {node: index for index, node in enumerate(fields.nodes)}
    edges = tuple(
        sorted(tuple(sorted((positions[a], positions[b]))) for a, b in fields.edges)
    )
    chamber = _certify_chamber(tuple(map(I, fields.phase)), edges)
    margins = tuple(pi_interval().lo - gap.abs_max for gap in chamber.differences)
    if _signature(graph) != before:
        raise ValueError("read-only child prediction changed persistent state")
    return ChildTholPrediction(
        actor=ACTOR,
        child=CHILD,
        source_time=SAMPLE_TIMES[-1],
        source_nodes=fields.nodes,
        source_edges=fields.edges,
        source_conductances=fields.conductances,
        source_forms=fields.epi,
        source_phases=fields.phase,
        source_capacities=fields.capacity,
        source_degrees=fields.degrees,
        source_stored_pressures=pressure,
        source_pressure_defects=defects,
        source_principal_shifts=chamber.shifts,
        source_principal_edges=edges,
        source_principal_margins=margins,
        successor_nodes=(*fields.nodes, CHILD),
        successor_edges=fields.edges,
        neighbors=(PARENT,),
        actor_form=x,
        actor_phase=theta,
        actor_capacity=nu,
        actor_pressure=pressure[CHILD_INDEX],
        sample_times=SAMPLE_TIMES,
        observed_acceleration=acceleration,
        threshold=threshold,
        minimum_form=minimum_form,
        strict_preconditions_admitted=True,
        grammar_allowed=grammar.allowed,
        pressure_gain=gain,
        exact_pressure_increment=increment,
        represented_pressure_increment=increment_rn,
        raw_pressure_ideal=raw,
        raw_pressure_represented=raw_rn,
        pressure_rounding_defect=raw_rn - raw,
        raw_pressure_bounds=_with_allowance(I(raw)),
        pressure_square_storage_jump=(raw**2 - pressure[CHILD_INDEX] ** 2) / 2,
        metabolic=metabolic,
        ideal_phase_variance=Q(0),
        ideal_child_form=ideal_form,
        child_form_bounds=form_bounds,
        child_capacity_ideal=ideal_capacity,
        child_capacity=capacity,
        capacity_rounding_defect=capacity - ideal_capacity,
        child_phase=theta,
        child_pressure=Q(0),
        child_degree=0,
        child_hierarchy_level=2,
        child_hierarchy_path=(PARENT, ACTOR),
        child_record_operator_step=2,
        expected_hierarchy=((PARENT, (ACTOR,)), (ACTOR, (CHILD,))),
        expected_actor_glyph_history=("OZ", "THOL"),
        expected_actor_u2_debt=0,
        child_legacy_history_length=1,
        child_timed_history_length=0,
        child_glyph_history=(),
        child_immediate_thol_admitted=False,
        event_storage_jump=Q(0),
        old_component_mean_weights=tuple(
            Q(degree) / capacity
            for degree, capacity in zip(fields.degrees, fields.capacity, strict=True)
        ),
        old_component_mean_jumps=(Q(0), Q(0)),
        old_component_rows_unchanged=True,
        isolate_form_rate=Q(0),
        isolate_phase_rate=Q(0),
        global_positive_mean_metric_available=False,
    )
