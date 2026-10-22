"""Primitive, read-only checks for the selected nested THOL/SHA continuation.

Expected records reconstruct the declared footprint without invoking either
operator or its proposal. Observed metabolic outputs enter that footprint only
after admission against independently predicted bounds. No trajectory or
future maintenance certificate follows from these finite event checks.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from fractions import Fraction as Q

from ..alias import set_attr, set_dnfr, set_vf
from ..constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from ..constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SOURCE_GLYPH,
    ALIAS_VF,
)
from ..mathematics._rational_interval import I
from ..operators import _validated_execution_window, network_stage
from ..operators.definitions import SelfOrganization, Silence
from ..operators.network_stage import NetworkStageResult
from ..operators.nodal_equation import observe_structural_acceleration
from ..operators.stage_contracts import stage_schedule_metadata
from ..physics.relational_sine_support import (
    bound_relational_sine_frozen_isolates,
    bound_relational_sine_support_exchange,
)
from .local_capacity_history import MODEL, PRESSURE_ERROR_BUDGET, encode_result
from .maintained_parent_sha_source import _same
from .maintained_parent_thol import _represented, _signature
from .maintained_parent_um_execution import _maximum_matches, _read
from .maintained_parent_um_work import _bound_state
from .retained_graph_state import record_graph_state, restore_recorded_graph_state

EVENT_TIMESTAMP = "2030-01-02T03:04:10+00:00"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _clone(graph):
    return restore_recorded_graph_state(record_graph_state(graph)).graph


def _prediction(admitted, supplied):
    """Rebuild consumed law and reset fields from the fresh initial graph."""
    from .maintained_child_word_prediction import _predict_child_word

    rebuilt = _predict_child_word(admitted)
    _require(
        _same(encode_result(rebuilt), encode_result(supplied)),
        "child word checker prediction differs from its primitive source",
    )
    return rebuilt


def _history(graph, node, glyph, step):
    """Reconstruct this selected prefix, not a general grammar implementation."""
    data = graph.nodes[node]
    data["glyph_history"] = deque(
        (*data["glyph_history"], glyph), maxlen=_validated_execution_window(graph, None)
    )
    data["_operator_step"] = step
    data["_grammar_u2_debt"] = 0
    data["_grammar_prior_coherence"] = False
    # The string setter has the same first-present alias contract.
    key = next(
        (key for key in ALIAS_SOURCE_GLYPH if key in data), ALIAS_SOURCE_GLYPH[0]
    )
    data[key] = glyph


def _signals(record, prediction):
    expected = prediction.metabolic
    budget = prediction.numerical_allowance
    _require(
        type(record) is dict
        and set(record)
        == {
            "epi_gradient",
            "phase_variance",
            "neighbor_count",
            "coupling_strength_mean",
            "mean_neighbor_epi",
        },
        "THOL metabolic signal fields differ",
    )
    _require(
        type(record["neighbor_count"]) is int
        and record["neighbor_count"] == expected.neighbor_count,
        "THOL metabolic neighbor count differs",
    )
    for name, value in (
        ("epi_gradient", expected.epi_gradient),
        ("mean_neighbor_epi", expected.mean_neighbor_epi),
    ):
        actual = _represented(record[name], name)
        _require(
            abs(actual - value) <= budget, f"THOL {name} exceeds predicted allowance"
        )
    for name, bounds in (
        ("phase_variance", expected.phase_variance),
        ("coupling_strength_mean", expected.coupling_strength_mean),
    ):
        actual = _represented(record[name], name)
        _require(
            0 <= actual <= 1 and bounds.lo - budget <= actual <= bounds.hi + budget,
            f"THOL {name} exceeds predicted bounds",
        )
    return deepcopy(record)


def _expected_thol_raw(before, raw, prediction):
    actor, child = prediction.actor, prediction.child
    _require(
        tuple(raw) == prediction.successor_nodes, "THOL node identity/order differs"
    )
    form = _read(raw, child, ALIAS_EPI)
    _require(
        prediction.child_form_bounds.lo <= form <= prediction.child_form_bounds.hi,
        "THOL child form exceeds predicted bounds",
    )
    records = raw.nodes[actor].get("sub_epis")
    _require(
        type(records) is list and len(records) == 1 and type(records[0]) is dict,
        "THOL must create exactly one child record",
    )
    signals = _signals(records[0].get("network_signals"), prediction)
    expected = _clone(before)
    expected._last_operator_applied = SelfOrganization().name
    set_attr(
        expected.nodes[actor], ALIAS_D2EPI, float(prediction.observed_acceleration)
    )
    set_dnfr(expected, actor, float(prediction.raw_pressure_represented))
    expected.add_node(
        child,
        **{
            EPI_PRIMARY: float(form),
            VF_PRIMARY: float(prediction.child_capacity),
            THETA_PRIMARY: float(prediction.child_phase),
            DNFR_PRIMARY: 0.0,
            "parent_node": actor,
            "hierarchy_level": prediction.child_hierarchy_level,
            "_bifurcation_level": prediction.child_hierarchy_level,
            "_hierarchy_path": list(prediction.child_hierarchy_path),
            "epi_history": [float(form)],
            "glyph_history": [],
        },
    )
    target = expected.nodes[actor]
    target["sub_nodes"] = [child]
    expected.graph["hierarchy"][actor] = [child]
    target["sub_epis"] = [
        {
            "epi": float(form),
            "vf": float(prediction.child_capacity),
            "timestamp": prediction.child_record_operator_step,
            "d2_epi": float(prediction.observed_acceleration),
            "tau": float(prediction.threshold),
            "node_id": child,
            "metabolized": True,
            "network_signals": signals,
            "bifurcation_level": prediction.child_hierarchy_level,
            "hierarchy_path": list(prediction.child_hierarchy_path),
        }
    ]
    _history(expected, actor, "THOL", prediction.child_record_operator_step)
    target["_thol_subepi_amplitude_alignment"] = 0.0
    target["_mutation_context"] = {
        "destabilizer_operator": "dissonance",
        "destabilizer_distance": 1,
        "recent_history": ["dissonance"],
    }
    target["_thol_no_bifurcation_expected"] = False
    _require(_signature(expected) == _signature(raw), "THOL raw footprint differs")
    return expected


def _expected_sha_raw(before, raw, prediction):
    expected = _clone(before)
    actor = prediction.thol.actor
    expected._last_operator_applied = Silence().name
    target = expected.nodes[actor]
    form = _read(before, actor, ALIAS_EPI)
    target.update(
        {
            "latent": True,
            "latency_start_time": EVENT_TIMESTAMP,
            "preserved_epi": float(form),
            "silence_duration": 0.0,
            "was_initial_on_silence": abs(float(form)) < 1e-6,
        }
    )
    set_vf(expected, actor, float(prediction.actor_capacity_after))
    _history(expected, actor, "SHA", prediction.expected_actor_operator_step)
    _require(_signature(expected) == _signature(raw), "SHA raw footprint differs")
    return expected


def _stage_endpoint(raw_expected, endpoint, stage, original, glyph):
    operator = SelfOrganization() if glyph == "THOL" else Silence()
    expected_stage = NetworkStageResult(
        operator=operator.name,
        glyph=glyph,
        schedule="two_phase_jacobi",
        nodes_processed=1,
    )
    _require(
        _same(encode_result(stage), encode_result(expected_stage)),
        f"{glyph} did not return the canonical single-target stage",
    )
    expected = _clone(raw_expected)
    for node in expected:
        value = _read(original, node, ALIAS_DNFR) if node in original else Q(0)
        set_dnfr(expected, node, float(value))
    expected.graph[network_stage.STAGE_SCHEDULE_KEY] = {
        "operator": operator.name,
        "glyph": glyph,
        "schedule": "two_phase_jacobi",
        "nodes_processed": 1,
    }
    expected.graph[network_stage.STAGE_CONTRACT_KEY] = stage_schedule_metadata(
        operator, observed_schedule="two_phase_jacobi"
    )
    _require(
        _signature(expected) == _signature(endpoint),
        f"{glyph} refresh or stage metadata footprint differs",
    )


def _law_checks(original, raw, endpoint, prediction, *, closed):
    thol = prediction.thol
    source = bound_relational_sine_support_exchange(original, reference_model=MODEL)
    fields = tuple(
        bound_relational_sine_frozen_isolates(graph, reference_model=MODEL)
        for graph in (raw, endpoint)
    )
    actual = fields[-1]
    _require(
        all(
            field.nodes == thol.successor_nodes
            and field.epi[:-1] == source.epi
            and field.phase[:-1] == source.phase
            and field.edges == source.edges
            and field.conductances == source.conductances
            for field in fields
        ),
        "birth/closure changed old storage arguments or support",
    )
    component = bound_relational_sine_support_exchange(
        endpoint.subgraph(thol.source_nodes).copy(), reference_model=MODEL
    )
    positions = {node: index for index, node in enumerate(component.nodes)}
    weights = {
        tuple(sorted((positions[a], positions[b]))): weight
        for (a, b), weight in zip(component.edges, component.conductances, strict=True)
    }
    edges = tuple(sorted(weights))
    state = _bound_state(
        component.epi,
        tuple(map(I, component.phase)),
        tuple(map(I, component.capacity)),
        edges,
        tuple(I(weights[edge]) for edge in edges),
    )
    expected = prediction.after if closed else prediction.before
    _require(
        _same(encode_result(state), encode_result(expected)),
        "complete component rows or means differ from prediction",
    )
    pressure_errors = tuple(
        max(
            abs(_read(endpoint, node, ALIAS_DNFR) - bound.lo),
            abs(_read(endpoint, node, ALIAS_DNFR) - bound.hi),
        )
        for node, bound in zip(component.nodes, state.pressure, strict=True)
    )
    _require(
        max(pressure_errors) <= PRESSURE_ERROR_BUDGET,
        "refreshed constitutive pressure exceeds allowance",
    )
    index = actual.nodes.index(thol.child)
    _require(
        actual.degrees[index] == 0
        and _read(endpoint, thol.child, ALIAS_DNFR) == 0
        and actual.form_rates[index] == actual.phase_rates[index] == I(0),
        "new child does not obey the declared frozen-isolate completion",
    )
    _require(
        actual.form_storage == source.form_storage
        and actual.phase_storage == source.phase_storage,
        "unchanged edge-storage identity failed",
    )
    _require(
        all(
            _represented(graph.graph["_t"], "structural clock")
            == prediction.structural_time
            for graph in (raw, endpoint)
        ),
        "event changed structural clock",
    )
    _require(
        all(
            _maximum_matches(graph, aliases, key)
            for graph in (raw, endpoint)
            for aliases, key in ((ALIAS_DNFR, "_dnfrmax"), (ALIAS_VF, "_vfmax"))
        ),
        "declared scalar maxima differ from actual state",
    )
    _require(
        "epi_time_history" not in endpoint.nodes[thol.child]
        and not observe_structural_acceleration(endpoint, thol.child).available,
        "birth inherited an unavailable temporal baseline",
    )
    return {
        "complete_component_state": encode_result(state),
        "pressure_error_bounds": pressure_errors,
        "edge_storage_jump": Q(0),
        "all_checks_passed": True,
        "complete_word_executed": closed,
        "newborn_history_available": False,
        "future_retention_certified": False,
        "numerical_u5_observation_available": False,
        "cross_birth_u6_observation_available": False,
        "physical_identification": False,
    }


def check_child_thol_endpoint(admitted, raw, endpoint, *, prediction, stage):
    """Check one THOL result within a freshly rebuilt prediction orchestration."""
    original = admitted.graph
    protected = tuple(_signature(graph) for graph in (original, raw, endpoint))
    prediction = _prediction(admitted, prediction)
    expected = _expected_thol_raw(original, raw, prediction.thol)
    _stage_endpoint(expected, endpoint, stage, original, "THOL")
    result = _law_checks(original, raw, endpoint, prediction, closed=False)
    _require(
        protected == tuple(_signature(graph) for graph in (original, raw, endpoint)),
        "read-only THOL check changed its inputs",
    )
    return result


def check_child_sha_endpoint(
    admitted, thol_endpoint, raw, endpoint, *, prediction, stage
):
    """Check SHA against the separately checked committed THOL prefix."""
    original = admitted.graph
    protected = tuple(
        _signature(graph) for graph in (original, thol_endpoint, raw, endpoint)
    )
    prediction = _prediction(admitted, prediction)
    expected = _expected_sha_raw(thol_endpoint, raw, prediction)
    _stage_endpoint(expected, endpoint, stage, original, "SHA")
    result = _law_checks(original, raw, endpoint, prediction, closed=True)
    _require(
        protected
        == tuple(
            _signature(graph) for graph in (original, thol_endpoint, raw, endpoint)
        ),
        "read-only SHA check changed its inputs",
    )
    return result
