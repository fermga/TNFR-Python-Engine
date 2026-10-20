"""Detached default SHA reset and complete-law admission after committed UM.

The source reader re-admits the actual UM endpoint. This module predicts the
target-local capacity product through the shared scalar kernel, without an
operator proposal, event, pressure refresh or trajectory. The ideal product
and the represented implementation product remain separate. The complete
post-event law holds the new capacities, support and conductances fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR
from ..mathematics._rational_interval import I, pi_interval
from ..operators.al_sha_stage_proposals import silence_capacity_proposal
from ..operators.definitions import Coupling, Expansion, SelfOrganization, Silence
from ..operators.factor_contracts import (
    resolve_runtime_operator_factors,
    validate_glyph_factor,
)
from ..operators.grammar_debt import node_debt
from ..operators.grammar_dynamics import validate_candidate
from ..operators.grammar_execution import ValidatedSequence
from ..operators.preconditions import validate_silence
from ..physics.relational_sine_support import bound_relational_sine_support_exchange
from ..types import Glyph
from .local_capacity_history import CONTEXT, MODEL, TIMES
from .maintained_parent_thol import _signature
from .maintained_parent_um import PARENT, _represented
from .maintained_parent_um_work import StateBounds, _bound_state


@dataclass(frozen=True)
class ShaResetPrediction:
    """Exact scalar reset and rebuilt instantaneous complete-law bounds.

    Edge endpoints index ``nodes``; phase means use the source's retained real
    lifts. ``represented`` is a predicted runtime product, not an observed
    event. Neither latency metadata nor grammar closure freezes later flow.
    Storage omits capacity, so its jump is the algebraic identity zero, not a
    subtraction of independently rounded nonzero storage enclosures.
    """

    nodes: tuple
    parent_index: int
    edges: tuple[tuple[int, int], ...]
    epi: tuple[Q, ...]
    phases: tuple[Q, ...]
    conductances: tuple[Q, ...]
    capacity_before: tuple[Q, ...]
    ideal_capacity_after: tuple[Q, ...]
    represented_capacity_after: tuple[Q, ...]
    raw_pressure_before: tuple[Q, ...]
    factor: Q
    ideal_parent_capacity: Q
    represented_parent_capacity: Q
    capacity_absolute_error: Q
    represented_capacity_ratio: Q
    before: StateBounds
    ideal: StateBounds
    represented: StateBounds
    mean_weights_before: tuple[Q, ...]
    mean_weights_after: tuple[Q, ...]
    form_mean_before: Q
    form_mean_after: Q
    phase_mean_before: Q
    phase_mean_after: Q
    mobility_before: tuple[I, ...]
    mobility_after: tuple[I, ...]
    event_storage_jump: Q
    source_um_parent_capacity: I
    represented_capacity_inside_um_family: bool
    expected_parent_history: tuple[str, ...]
    expected_parent_operator_step: int
    structural_time: Q
    preserved_epi: Q
    was_initial_on_silence: bool
    complete_word_executed: bool = False
    numerical_u5_observation_available: bool = False
    cross_birth_u6_observation_available: bool = False
    autonomous_occurrence_derived: bool = False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _capacity_reset(capacity, factor):
    """Separate the exact ideal product from the shared binary64 product.

    Inputs must already have the declared runtime representation. General
    SHA admits an intentional zero factor; the selected positive-capacity
    successor is admitted separately by ``predict_parent_sha``.
    """
    capacity = _represented(capacity, "SHA source capacity")
    factor_value = validate_glyph_factor("SHA_vf_factor", factor)
    factor = _represented(factor, "SHA capacity factor")
    _require(capacity >= 0, "SHA source capacity must be nonnegative")
    before, after = silence_capacity_proposal(float(capacity), factor_value)
    _require(Q(before) == capacity, "SHA scalar kernel changed its admitted source")
    represented = _represented(after, "SHA represented capacity product")
    ideal = capacity * factor
    return ideal, represented, abs(represented - ideal)


def _mean(values, weights):
    return sum((x * w for x, w in zip(values, weights, strict=True)), Q(0)) / sum(
        weights, Q(0)
    )


def predict_parent_sha(admitted):
    """Re-admit the committed source and predict the default parent SHA.

    Cached verdicts, a mutable admission graph and its cached prediction cannot
    replace retained primitives. This calculation changes no state and does
    not extend the UM motion certificate past a capacity-changing event.
    """
    from .maintained_parent_sha_source import validate_live_um_successor

    admitted = validate_live_um_successor(admitted)
    graph = admitted.graph
    signature = _signature(graph)
    data, parent = graph.graph, graph.nodes[PARENT]
    _require(
        not any(str(key).startswith("SHA_") for key in data)
        and data.get("GLYPH_FACTORS") == {"VAL_scale": 128.0},
        "selected SHA prediction requires the unchanged default attenuation",
    )
    _require(
        data.get("GRAMMAR_REJECTION_MODE") == "raise"
        and data.get("VALIDATE_OPERATOR_PRECONDITIONS") is True
        and data.get("EDGE_AWARE_ENABLED") is False,
        "selected SHA prediction requires strict live execution policy",
    )
    _require(
        tuple(parent.get("glyph_history", ())) == ("VAL", "THOL", "UM")
        and type(parent.get("_grammar_u2_debt")) is int
        and parent["_grammar_u2_debt"] == node_debt(parent) == 0
        and type(parent.get("_operator_step")) is int
        and parent["_operator_step"] == 3,
        "SHA source requires the actual VAL,THOL,UM prefix and zero debt",
    )
    clock = _represented(data.get("_t"), "SHA structural clock")
    _require(clock == TIMES[-1], "SHA requires the retained structural clock")
    sequence = ValidatedSequence(
        (Expansion(), SelfOrganization(), Coupling(), Silence()), context=CONTEXT
    )
    _require(
        type(admitted.sequence) is ValidatedSequence
        and admitted.sequence.names == sequence.names,
        "SHA requires the selected core word",
    )
    fields = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    nodes = fields.nodes
    positions = {node: index for index, node in enumerate(nodes)}
    edges = tuple(
        sorted(tuple(sorted((positions[a], positions[b]))) for a, b in fields.edges)
    )
    _require(
        nodes == tuple(range(20)) + ("2_sub_0",)
        and len(edges) == 53
        and all(degree > 0 for degree in fields.degrees),
        "SHA requires the full connected committed UM support",
    )
    _require(
        all(value > 0 for value in fields.capacity + fields.conductances),
        "selected successor law requires positive capacities and conductances",
    )
    conductance_map = {
        tuple(sorted((positions[a], positions[b]))): value
        for (a, b), value in zip(fields.edges, fields.conductances, strict=True)
    }
    conductances = tuple(conductance_map[edge] for edge in edges)
    factor = _represented(
        resolve_runtime_operator_factors(data.get("GLYPH_FACTORS"), Glyph.SHA, data)[
            "SHA_vf_factor"
        ],
        "resolved SHA factor",
    )
    _require(0 < factor < 1, "selected successor requires positive SHA attenuation")
    validate_silence(graph, PARENT)
    decision = validate_candidate(
        graph, PARENT, "SHA", sequence_context=sequence.step(3)
    )
    _require(decision.allowed, "source live SHA grammar rejected")
    index = positions[PARENT]
    ideal_capacity, represented_capacity, defect = _capacity_reset(
        fields.capacity[index], factor
    )
    _require(
        ideal_capacity > 0 and represented_capacity > 0,
        "selected SHA successor must retain positive capacity",
    )
    ideal_capacities, represented_capacities = (
        list(fields.capacity),
        list(fields.capacity),
    )
    ideal_capacities[index], represented_capacities[index] = (
        ideal_capacity,
        represented_capacity,
    )
    ideal_capacities, represented_capacities = (
        tuple(ideal_capacities),
        tuple(represented_capacities),
    )

    def state(capacities):
        return _bound_state(
            fields.epi,
            tuple(map(I, fields.phase)),
            tuple(map(I, capacities)),
            edges,
            tuple(map(I, conductances)),
        )

    before, ideal, represented = (
        state(fields.capacity),
        state(ideal_capacities),
        state(represented_capacities),
    )
    weights_before, weights_after = (
        tuple(
            Q(degree) / capacity
            for degree, capacity in zip(before.degrees, capacities, strict=True)
        )
        for capacities in (fields.capacity, represented_capacities)
    )
    pi = pi_interval()
    mobility_before, mobility_after = (
        tuple(
            capacity / (pi * degree)
            for degree, capacity in zip(before.degrees, capacities, strict=True)
        )
        for capacities in (fields.capacity, represented_capacities)
    )
    source_capacity = admitted.prediction.chamber.conditional_capacities[index]
    pressure = tuple(
        _represented(
            get_attr(graph.nodes[node], ALIAS_DNFR, strict=True, conv=lambda x: x),
            f"source pressure at {node!r}",
        )
        for node in nodes
    )
    _require(_signature(graph) == signature, "SHA prediction mutated its source")
    return ShaResetPrediction(
        nodes=nodes,
        parent_index=index,
        edges=edges,
        epi=fields.epi,
        phases=fields.phase,
        conductances=conductances,
        capacity_before=fields.capacity,
        ideal_capacity_after=ideal_capacities,
        represented_capacity_after=represented_capacities,
        raw_pressure_before=pressure,
        factor=factor,
        ideal_parent_capacity=ideal_capacity,
        represented_parent_capacity=represented_capacity,
        capacity_absolute_error=defect,
        represented_capacity_ratio=represented_capacity / fields.capacity[index],
        before=before,
        ideal=ideal,
        represented=represented,
        mean_weights_before=weights_before,
        mean_weights_after=weights_after,
        form_mean_before=_mean(fields.epi, weights_before),
        form_mean_after=_mean(fields.epi, weights_after),
        phase_mean_before=_mean(fields.phase, weights_before),
        phase_mean_after=_mean(fields.phase, weights_after),
        mobility_before=mobility_before,
        mobility_after=mobility_after,
        event_storage_jump=Q(0),
        source_um_parent_capacity=source_capacity,
        represented_capacity_inside_um_family=source_capacity.contains(
            represented_capacity
        ),
        expected_parent_history=("VAL", "THOL", "UM", "SHA"),
        expected_parent_operator_step=4,
        structural_time=clock,
        preserved_epi=fields.epi[index],
        was_initial_on_silence=abs(float(fields.epi[index])) < 1e-6,
    )
