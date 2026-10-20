"""New AL/SHA preparations on the retained weighted-star successor.

Only the producer executes these four declared words. The reader rebuilds
complete instantaneous laws from recorded primitives without replay or flow.
Record association and contract checks do not authenticate event chronology.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction as Q
from unittest.mock import patch

from .._exact_time import exact_or_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR
from ..operators import network_stage
from ..operators.definitions import Emission, Silence
from ..operators.grammar_execution import ValidatedSequence
from ..operators.grammar_patterns import validate_sequence
from ..operators.stage_contracts import stage_schedule_metadata
from ..operators.word_execution import execute_network_operator_stage
from ..physics.relational_sine_support import (
    SineSupportExchangeComparison,
    bound_relational_sine_support_exchange,
)
from ..types import serialize_bepi
from ..utils.io import json_dumps
from ._recorded_graph_containers import (
    decode_recorded_container,
    encode_recorded_container,
)
from .artifact_io import decode_exact_tree, encode_exact_tree
from .retained_graph_state import record_graph_state, restore_recorded_graph_state
from .weighted_star_constitutive_handoff import (
    compare_weighted_star_constitutive_handoff,
)
from .weighted_star_contact import MODEL

DONORS = (5, 0)
BOOSTS = (Q(1, 1024), Q(1, 512))
SHA_FACTOR = Q(1, 2)
PROBE_NOW = datetime(2030, 1, 2, 3, 4, 6, tzinfo=timezone.utc)
_CORE = {"compatibility_profile": "core"}
_SCHEDULE = "two_phase_jacobi"
_OMISSIONS = {
    "omitted_rebuildable_caches",
    "omitted_networkx_views",
    "omitted_networkx_cache",
}


@dataclass(frozen=True)
class RelativeProbeBranch:
    """Fresh complete prefix fields and exact static event work."""

    donor: int
    boost: Q
    prefixes: tuple[SineSupportExchangeComparison, ...]
    actual_increment: Q
    event_work: tuple[Q, Q]
    stored_pressure: tuple[tuple[Q, ...], ...]


@dataclass(frozen=True)
class WeightedStarRelativeProbe:
    """A detached preparation, not a response or a future identity verdict.

    Consumers re-admit ``record`` with the original preparation. Frozen
    dataclass attributes alone cannot establish premises for another report.
    """

    record: dict
    source: SineSupportExchangeComparison
    branches: tuple[RelativeProbeBranch, ...]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _signature(value):
    # Tagged scalar/container spelling distinguishes bool/int and list/tuple.
    # Mapping insertion order is immaterial; support and neighbor order remain
    # explicit tuples in the graph record.
    def ordered(item):
        if type(item) is dict:
            return {
                key: ordered(item[key])
                for key in sorted(item, key=lambda key: (type(key).__name__, str(key)))
            }
        if type(item) is list:
            return [ordered(part) for part in item]
        if type(item) is tuple:
            return tuple(ordered(part) for part in item)
        if type(item) is deque:
            return deque((ordered(part) for part in item), maxlen=item.maxlen)
        return item

    return json_dumps(
        encode_exact_tree(encode_recorded_container(ordered(value))), sort_keys=True
    )


def _same(actual, expected, message):
    _require(_signature(actual) == _signature(expected), message)


def _persistent(graph):
    data = decode_recorded_container(record_graph_state(graph))
    return {key: value for key, value in data.items() if key not in _OMISSIONS}


def _protocol():
    return {
        "schema": "weighted_star_relative_probe_v1",
        "word": ("AL", "SHA"),
        "context": dict(_CORE),
        "structural_time": 7,
        "event_timestamp": PROBE_NOW.isoformat(),
        "sha_factor": SHA_FACTOR,
        "pressure_refresh": "absent",
        "continuation": "not_executed",
    }


def _stage_record(operator, donor, index):
    return {
        "operator": operator.name,
        "glyph": operator.glyph.value,
        "schedule": _SCHEDULE,
        "nodes_processed": 1,
        "target": donor,
        "word_index": index,
        "structural_time": 7,
        "event_timestamp": PROBE_NOW.isoformat(),
    }


def _word():
    operators = (Emission(), Silence())
    admission = validate_sequence([op.name for op in operators], context=_CORE)
    _require(admission.passed, "the declared AL/SHA word must pass core grammar")
    _require(
        admission.metadata["compatibility_profile"] == "core",
        "the declared word requires core grammar",
    )
    return operators, ValidatedSequence(operators, context=_CORE)


def _expected_prefix(source, donor, boost, index, operators):
    """Exact retained-record association, not an alternative event executor.

    The narrow contract records the first emission of this unactivated donor,
    its two history entries and SHA lifecycle. No callbacks or histories are
    replayed here. Structural values are separately admitted before comparison.
    """
    expected = deepcopy(source)
    config = expected["graph"]
    config["GLYPH_FACTORS"]["AL_boost"] = float(boost)
    config["GLYPH_FACTORS"]["SHA_vf_factor"] = float(SHA_FACTOR)
    if index == 0:
        return expected
    operator = operators[index - 1]
    config[network_stage.STAGE_SCHEDULE_KEY] = {
        key: _stage_record(operator, donor, index - 1)[key]
        for key in ("operator", "glyph", "schedule", "nodes_processed")
    }
    config[network_stage.STAGE_CONTRACT_KEY] = stage_schedule_metadata(
        operator, observed_schedule=_SCHEDULE
    )
    expected["object_state"] = {"_last_operator_applied": operator.name}
    target = dict(expected["nodes"])[donor]
    value = float(Q(5, 16) + boost)
    timestamp = PROBE_NOW.isoformat()
    target.update(
        EPI=serialize_bepi(value),
        glyph_history=deque(("AL", "SHA")[:index], maxlen=8),
        emission_timestamp=timestamp,
        _emission_activated=True,
        _emission_origin=timestamp,
        _structural_lineage={
            "origin": timestamp,
            "activation_count": 1,
            "derived_nodes": [],
            "parent_emission": None,
        },
        _operator_step=index,
        _grammar_u2_debt=0,
        _grammar_prior_coherence=False,
        source_glyph=operator.glyph.value,
    )
    if index == 2:
        target.update(
            nu_f=2.0,
            latent=True,
            latency_start_time=timestamp,
            preserved_epi=value,
            silence_duration=0.0,
            was_initial_on_silence=False,
        )
    return expected


def _fields(graph):
    source = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    pressure = tuple(
        get_attr(
            graph.nodes[node],
            ALIAS_DNFR,
            strict=True,
            conv=lambda value: exact_or_represented_real(value, "stored pressure"),
        )
        for node in source.nodes
    )
    _require(all(value is not None for value in pressure), "stored pressure required")
    return source, pressure


def admit_weighted_star_relative_probe(preparation, record):
    """Re-admit a captured four-branch preparation without any event or flow.

    Full persistent context is checked against the fixed source plus declared
    word effects; normalized tagged equality retains scalar and container
    types. Shared omission admission permits only declared rebuildable caches.
    The resulting equality is association, not provenance authentication.
    """
    handoff = compare_weighted_star_constitutive_handoff(preparation)
    base = restore_recorded_graph_state(handoff.preparation["static_states"][-1]).graph
    source, _ = _fields(base)
    source_record = _persistent(base)
    _require(type(record) is dict, "probe record must be a mapping")
    _require(
        set(record) == {"source_record_sha256", "protocol", "branches"},
        "invalid probe record fields",
    )
    _same(record["protocol"], _protocol(), "probe protocol differs")
    _same(
        record["source_record_sha256"],
        handoff.preparation_record_sha256,
        "probe source association differs",
    )
    raw_branches = record["branches"]
    _require(
        type(raw_branches) is tuple and len(raw_branches) == 4,
        "four ordered branches required",
    )
    operators, _ = _word()
    branches = []
    for raw, (donor, boost) in zip(
        raw_branches, ((d, a) for d in DONORS for a in BOOSTS), strict=True
    ):
        _require(
            type(raw) is dict
            and set(raw) == {"donor", "boost", "static_states", "stages"},
            "invalid branch fields",
        )
        _same(
            (raw["donor"], raw["boost"]), (donor, boost), "branch intervention differs"
        )
        _same(
            raw["stages"],
            tuple(_stage_record(op, donor, i) for i, op in enumerate(operators)),
            "branch stage context differs",
        )
        states = raw["static_states"]
        _require(
            type(states) is tuple and len(states) == 3,
            "three prefix snapshots required",
        )
        prefixes, pressures = [], []
        for index, state in enumerate(states):
            graph = restore_recorded_graph_state(state).graph
            fields, pressure = _fields(graph)
            # Primitive admission precedes record comparison, including EPI,
            # capacities, support, phases and authoritative stored pressure.
            _same(
                _persistent(graph),
                _expected_prefix(source_record, donor, boost, index, operators),
                "branch persistent prefix differs from declared event contract",
            )
            prefixes.append(fields)
            pressures.append(pressure)
        initial, after_al, final = prefixes
        branches.append(
            RelativeProbeBranch(
                donor=donor,
                boost=boost,
                prefixes=tuple(prefixes),
                actual_increment=final.epi[donor] - initial.epi[donor],
                event_work=(
                    after_al.form_storage - initial.form_storage,
                    final.form_storage - after_al.form_storage,
                ),
                stored_pressure=tuple(pressures),
            )
        )
    return WeightedStarRelativeProbe(deepcopy(record), source, tuple(branches))


class _SuppliedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return PROBE_NOW if tz is None else PROBE_NOW.astimezone(tz)


def prepare_weighted_star_relative_probe(preparation):
    """Execute only four new AL/SHA words on independent retained-source copies.

    Strict source configuration and complete historical context remain live.
    These instantaneous hybrid events leave structural time at seven. No
    pressure callback, continuous integration or reserved reading is invoked.
    """
    handoff = compare_weighted_star_constitutive_handoff(preparation)
    operators, sequence = _word()
    branches = []
    for donor in DONORS:
        for boost in BOOSTS:
            graph = restore_recorded_graph_state(
                handoff.preparation["static_states"][-1]
            ).graph
            graph.graph["GLYPH_FACTORS"]["AL_boost"] = float(boost)
            graph.graph["GLYPH_FACTORS"]["SHA_vf_factor"] = float(SHA_FACTOR)
            states, stages = [record_graph_state(graph)], []
            with patch.object(network_stage, "datetime", _SuppliedClock):
                for index, operator in enumerate(operators):
                    result = execute_network_operator_stage(
                        graph, operator, (donor,), sequence_context=sequence.step(index)
                    )
                    stage = _stage_record(operator, donor, index)
                    for key in ("operator", "glyph", "schedule", "nodes_processed"):
                        _same(
                            getattr(result, key),
                            stage[key],
                            "unexpected stage execution",
                        )
                    states.append(record_graph_state(graph))
                    stages.append(stage)
            branches.append(
                {
                    "donor": donor,
                    "boost": boost,
                    "static_states": tuple(states),
                    "stages": tuple(stages),
                }
            )
    record = decode_exact_tree(
        encode_exact_tree(
            {
                "source_record_sha256": handoff.preparation_record_sha256,
                "protocol": _protocol(),
                "branches": tuple(branches),
            }
        )
    )
    return admit_weighted_star_relative_probe(preparation, record)
