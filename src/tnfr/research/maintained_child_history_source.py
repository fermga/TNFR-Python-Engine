"""Fresh admission of the retained child-owned OZ/history endpoint.

This reader rebuilds consumed sample errors, held-law state and live grammar.
The archived numerical trace and derivative jets retain their earlier audit
and frozen-producer provenance: their arithmetic is not replayed here. Byte
associations do not authenticate acquisition or occurrence. No event, pressure
refresh, numerical trajectory or new temporal observation is executed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from ..alias import get_attr, set_attr, set_dnfr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..dynamics.runtime import (
    _MUTATION_TIME_HISTORY_MAXLEN,
    _validated_mutation_time_history,
)
from ..operators import _validated_execution_window
from ..operators._thol_config import resolve_thol_min_epi
from ..operators.definitions import Dissonance, SelfOrganization, Silence
from ..operators.grammar_debt import node_debt
from ..operators.grammar_dynamics import validate_candidate
from ..operators.grammar_execution import ValidatedSequence
from ..operators.nodal_equation import observe_structural_acceleration
from ..operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from ..operators.stage_contracts import stage_schedule_metadata
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)
from .local_capacity_history import CONTEXT, PRESSURE_ERROR_BUDGET, capture
from .maintained_child_history import CHILD, CHILD_INDEX, PARENT
from .maintained_child_joint_source import CELL_END, CELL_START
from .maintained_child_oz_timing import (
    PRESSURE_ERROR_ALLOWANCE,
    SAMPLE_ERROR_ALLOWANCE,
    SAMPLE_STEP,
    _admit_child_oz_timing_source,
)
from .maintained_child_sample_certificate import (
    _assess_child_samples,
    prepare_child_sample_certificate,
)
from .maintained_parent_sha_source import _same
from .maintained_parent_thol import _represented, _signature
from .maintained_sha_retention_evidence import _receipt, _receipt_fields, _require
from .retained_graph_state import record_graph_state, restore_recorded_graph_state

_MAX_BYTES = 128 * 1024**2
_JOINT_PATH = "docs/assets/maintained_child_joint_window"
_FROZEN_NAMES = frozenset({"protocol.json", "source-manifest.json", "source.zip"})
_RESULT_NAMES = frozenset(
    {
        "source.json",
        "event_before.json",
        "event_raw.json",
        "event_after.json",
        "sample_0.json",
        "sample_1.json",
        "sample_2.json",
        "response.json",
        "steps.zip",
    }
)


@dataclass(frozen=True)
class ChildHistoryAdmission:
    """Fresh mutable endpoint and rebuilt pre-THOL evidence, not a future law.

    Public consumers reload this reader. Private compositions may use one
    fresh result in the same call but must not trust later mutated graphs or
    cached certificates. The original SHA enclosures bound acquisition error
    at the recorded instants; they are not a future tube for this numerical
    endpoint or for a changed-support successor.
    """

    graph: object
    certificate: object
    assessment: object
    fields: object
    observation: object
    sequence: ValidatedSequence
    threshold: Q
    receipts: tuple[dict, ...] = ()


def _inventory(rows, expected, label):
    _require(
        type(rows) is tuple
        and len(rows) == len(expected)
        and {_receipt_fields(row) for row in rows} == set(expected),
        f"{label} artifact inventory differs",
    )


def _read_child_history_artifacts(folder, root):
    """Associate all retained bytes without executing an archived source."""
    data, receipts = {}, {}

    def raw(name):
        if name not in data:
            path = (folder / name).resolve()
            _require(path.is_relative_to(folder), "child artifact leaves root")
            value = read_bytes_bounded(path, max_bytes=_MAX_BYTES)
            data[name] = value
            receipts[name] = {
                "path": name,
                "bytes": len(value),
                "sha256": sha256_bytes(value),
            }
        return data[name]

    def verify(row, expected=None):
        name = _receipt_fields(row)
        _require(expected is None or name == expected, "child artifact role differs")
        raw(name)
        _require(_same(receipts[name], row), "child artifact receipt differs")

    names = (
        _FROZEN_NAMES | _RESULT_NAMES | {"freeze.json", "attempt.json", "result.json"}
    ) - {"source.zip", "steps.zip"}
    documents = {
        name.removesuffix(".json"): decode_exact_tree(json_loads(raw(name)))
        for name in sorted(names)
    }
    freeze, attempt, result = (
        documents[key] for key in ("freeze", "attempt", "result")
    )
    _require(
        freeze["schema"] == "tnfr.maintained-child-history-freeze.v1"
        and freeze["evaluation_status_at_freeze"] == "not_evaluated",
        "child freeze declaration differs",
    )
    _inventory(freeze["artifacts"], _FROZEN_NAMES, "frozen child")
    for row in freeze["artifacts"]:
        verify(row)
    prior = freeze["prior_artifacts"]
    _require(
        type(prior) is tuple
        and prior
        and len(prior) == len({_receipt_fields(row) for row in prior}),
        "child prior artifact inventory differs",
    )
    for row in prior:
        _receipt(root, row)
    manifest = documents["source-manifest"]
    _require(
        manifest["schema"] == "tnfr.source-manifest.v1", "child source manifest schema"
    )
    members = manifest["files"]
    _require(
        type(members) is tuple
        and members
        and len(members) == len({_receipt_fields(row) for row in members}),
        "child frozen source inventory differs",
    )
    hashes = {row["path"]: row["sha256"] for row in members}
    _require(
        {
            "src/tnfr/research/maintained_child_history_acquisition.py",
            "src/tnfr/research/maintained_child_sample_certificate.py",
            "src/tnfr/research/maintained_child_oz_timing.py",
            "src/tnfr/mathematics/_validated_taylor.py",
            "src/tnfr/dynamics/integrators.py",
            "src/tnfr/dynamics/runtime.py",
            "src/tnfr/operators/network_stage.py",
            "instrument.py",
            "pyproject.toml",
        }
        <= hashes.keys(),
        "child frozen implementation owner missing",
    )
    archive = raw("source.zip")
    verify_archive_members(BytesIO(archive), hashes, max_bytes=_MAX_BYTES)
    with ZipFile(BytesIO(archive)) as zipped:
        _require(
            _same(
                {item.filename: item.file_size for item in zipped.infolist()},
                {row["path"]: row["bytes"] for row in members},
            ),
            "child source archive member size differs",
        )
    _require(
        attempt["schema"] == "tnfr.maintained-child-history-attempt.v1"
        and attempt["evaluation_status_at_start"] == "not_evaluated"
        and type(attempt["attempt_number"]) is int
        and attempt["attempt_number"] == 1,
        "child attempt declaration differs",
    )
    verify(attempt["freeze"], "freeze.json")
    _require(
        result["schema"] == "tnfr.maintained-child-history-result.v1"
        and result["status"] == "acquired"
        and result["failure"] is None
        and result["complete_word_executed"] is False
        and result["second_birth_executed"] is False,
        "child result declaration differs",
    )
    _require(
        _represented(result["elapsed_seconds"], "recorded elapsed time") >= 0,
        "negative recorded elapsed time",
    )
    verify(result["attempt"], "attempt.json")
    _inventory(result["artifacts"], _RESULT_NAMES, "child result")
    for row in result["artifacts"]:
        verify(row)
    steps = result["step_members"]
    expected = tuple(f"step-{index:04d}.json" for index in range(416))
    _inventory(steps, expected, "child numerical trace")
    _require(
        tuple(row["path"] for row in steps) == expected, "child trace order differs"
    )
    # The original numerical trace audit is inherited. Here only archive/member
    # associations are checked; no recorded field or RK4 stage is reevaluated.
    archived_steps = raw("steps.zip")
    verify_archive_members(
        BytesIO(archived_steps),
        {row["path"]: row["sha256"] for row in steps},
        max_bytes=_MAX_BYTES,
    )
    with ZipFile(BytesIO(archived_steps)) as zipped:
        _require(tuple(zipped.namelist()) == expected, "child ZIP trace order differs")
        _require(
            tuple(item.file_size for item in zipped.infolist())
            == tuple(row["bytes"] for row in steps),
            "child trace member size differs",
        )
    return documents, tuple(receipts[name] for name in sorted(receipts))


def _values(graph, aliases):
    return tuple(
        _represented(
            get_attr(graph.nodes[node], aliases, strict=True, conv=lambda value: value),
            f"child evidence coordinate at {node!r}",
        )
        for node in graph
    )


def _state(graph):
    return _values(graph, ALIAS_EPI) + _values(graph, ALIAS_THETA)


def _restore(record):
    return restore_recorded_graph_state(record).graph


def _same_clock(graph, time):
    _require(
        _represented(graph.graph["_t"], "live clock") == time,
        "child evidence clock differs",
    )


def _flow_footprint(before, after, time, *, sample=False):
    """Compare a detached expected footprint; do not run flow or the recorder."""
    expected = _restore(record_graph_state(before))
    state, pressures = _state(after), _values(after, ALIAS_DNFR)
    _require(
        tuple(after) == tuple(expected) and len(after) == 21,
        "child flow node order differs",
    )
    _same_clock(after, time)
    for index, node in enumerate(expected):
        set_attr(expected.nodes[node], ALIAS_EPI, float(state[index]))
        set_attr(expected.nodes[node], ALIAS_THETA, float(state[21 + index]))
        set_dnfr(expected, node, float(pressures[index]))
        if sample:
            old = _validated_mutation_time_history(
                before.nodes[node].get("epi_time_history"), node=node
            )
            _require(not old or Q(old[-1][0]) < time, "sample clock does not advance")
            expected.nodes[node]["epi_time_history"] = deque(
                (*old, (float(time), float(state[index]))),
                maxlen=_MUTATION_TIME_HISTORY_MAXLEN,
            )
    expected.graph["_t"] = float(time)
    _require(
        _signature(expected) == _signature(after),
        "child flow changed undeclared configuration, hierarchy or context",
    )


def _protocol(protocol, certificate, timing):
    source_graph = certificate.source.retention.admitted.graph
    expected = {
        "schema": "tnfr.maintained-child-history-protocol.v1",
        "evidence_kind": "new_numerical_acquisition_against_existing_exact_flow_certificate",
        "child": CHILD,
        "parent": PARENT,
        "node_order": tuple(range(20)) + (CHILD,),
        "state_dimension": 42,
        "original_edges": 53,
        "law": "x_dot=M*S(theta); theta_dot=M*L_W*x; M=diag(nu/(pi*d))",
        "held": ("support", "conductances", "positive capacities"),
        "forcing": 0,
        "loss": 0,
        "clock": "original structural clock; unchanged real phase lifts",
        "source_time": timing.source_time,
        "elapsed_sample_times": (CELL_START, CELL_START + SAMPLE_STEP, CELL_END),
        "sample_times": certificate.sample_times,
        "event_time": certificate.sample_times[0],
        "integrator": "tnfr.dynamics.integrators.rk4_coupled_step",
        "field": "tnfr.research.local_capacity_history.numerical_field",
        "step": Q(1, 2**14),
        "maximum_steps": 416,
        "segment_steps": (384, 16, 16),
        "field_evaluations": 1664,
        "maximum_reserved_evaluations": 1,
        "maximum_events": 1,
        "execution_wall_budget_seconds": 600,
        "wall_budget_scope": "cooperative execution checks; source re-admission excluded",
        "context_word": ("OZ", "THOL", "SHA"),
        "executed_prefix": ("OZ",),
        "propagation_mode": "phase_weighted",
        "random_seed": 17,
        "oz_noise": False,
        "full_state_error_allowance": SAMPLE_ERROR_ALLOWANCE,
        "child_sample_error_allowance": SAMPLE_ERROR_ALLOWANCE,
        "exact_flow_pressure_error_allowance": PRESSURE_ERROR_ALLOWANCE,
        "represented_point_pressure_error_allowance": PRESSURE_ERROR_BUDGET,
        "timestamp_error": 0,
        "reader_roundoff_allowance": 0,
        "minimum_form": _represented(
            resolve_thol_min_epi(source_graph.graph), "form policy"
        ),
        "threshold": timing.threshold,
    }
    _require(
        type(protocol) is dict
        and all(
            key in protocol and _same(protocol[key], value)
            for key, value in expected.items()
        ),
        "child acquisition protocol differs from admitted source and policies",
    )
    receipt = next(
        row for row in certificate.source.receipts if row["path"] == "result.json"
    )
    _require(
        _same(
            protocol["source_result"], {**receipt, "path": f"{_JOINT_PATH}/result.json"}
        ),
        "child protocol source receipt differs",
    )


def _oz_footprint(before, raw, after, timing):
    """Reconstruct only the declared pressure/context records on a detached copy."""
    incoming = _values(before, ALIAS_DNFR)
    outgoing = _values(raw, ALIAS_DNFR)
    expected = _restore(record_graph_state(before))
    proposal = float(incoming[CHILD_INDEX]) * float(timing.gain)
    _require(outgoing[CHILD_INDEX] == Q(proposal), "raw child OZ pressure differs")
    set_dnfr(expected, CHILD, proposal)
    expected._last_operator_applied = Dissonance().name
    target = expected.nodes[CHILD]
    target["glyph_history"] = deque(
        ("OZ",), maxlen=_validated_execution_window(before, None)
    )
    target.update(
        {
            "_oz_precondition_context": {
                "epi": float(_values(before, ALIAS_EPI)[CHILD_INDEX]),
                "dnfr": float(incoming[CHILD_INDEX]),
                "vf": float(_values(before, ALIAS_VF)[CHILD_INDEX]),
                "degree": 1,
                "validation_passed": True,
            },
            "_operator_step": 1,
            "_grammar_u2_debt": 1,
            "_grammar_prior_coherence": False,
            "source_glyph": "OZ",
        }
    )
    events = list(before.graph.get("_oz_propagation_events", ()))
    events.append(
        {
            "source": CHILD,
            "magnitude": proposal - float(incoming[CHILD_INDEX]),
            "affected_nodes": [],
            "affected_count": 0,
        }
    )
    expected.graph["_oz_propagation_events"] = events
    _require(
        _signature(expected) == _signature(raw),
        "OZ changed undeclared state or context",
    )
    for node, pressure in zip(expected, incoming, strict=True):
        set_dnfr(expected, node, float(pressure))
    expected.graph["_last_network_stage_schedule"] = {
        "operator": "dissonance",
        "glyph": "OZ",
        "schedule": "two_phase_jacobi",
        "nodes_processed": 1,
    }
    expected.graph["_last_network_stage_contract"] = stage_schedule_metadata(
        Dissonance(), observed_schedule="two_phase_jacobi"
    )
    _require(
        _signature(expected) == _signature(after), "OZ refresh/stage footprint differs"
    )


def _admit_child_history(documents, certificate):
    """Consume raw records within one freshly admitted source orchestration."""
    protected = _signature(certificate.source.retention.admitted.graph)
    timing = _admit_child_oz_timing_source(certificate.source)
    _protocol(documents["protocol"], certificate, timing)
    record, response = documents["source"], documents["response"]
    source = _restore(record["endpoint"])
    _require(
        _signature(source) == protected,
        "acquired source differs from actual SHA endpoint",
    )
    _require(
        _same(record["source_receipts"], certificate.source.receipts)
        and _same(response["source_receipts"], certificate.source.receipts),
        "child source associations differ",
    )
    _require(_same(record["timing_admission"], timing), "cached child timing differs")
    expected = {
        "schema": "tnfr.maintained-child-history-response.v1",
        "status": "acquired",
        "numerical_steps": 416,
        "field_evaluations": 1664,
        "sample_times": certificate.sample_times,
        "executed_prefix": ("OZ",),
        "outstanding_u2_debt": 1,
        "complete_word_executed": False,
        "second_birth_executed": False,
        "physical_observation": False,
    }
    _require(
        all(_same(response[key], value) for key, value in expected.items()),
        "child response scope or chronology differs",
    )
    assessment = _assess_child_samples(
        certificate,
        states=response["states"],
        pressures=response["pressures"],
        sample_times=response["sample_times"],
    )
    _require(assessment.passed, "child acquired sample certificate failed")
    _require(
        _same(response["assessment"], assessment), "cached child assessment differs"
    )
    _require(
        _same(response["boundary_associations"], assessment.boundaries),
        "cached child boundary associations differ",
    )
    _require(
        all(
            max(row.point_pressure_error_bounds) <= PRESSURE_ERROR_BUDGET
            for row in assessment.boundaries
        ),
        "represented-point pressure budget failed",
    )
    before, raw = (_restore(documents[name]) for name in ("event_before", "event_raw"))
    after = _restore(documents["event_after"]["endpoint"])
    _flow_footprint(source, before, certificate.sample_times[0])
    _oz_footprint(before, raw, after, timing)
    stage = {
        "operator": "dissonance",
        "glyph": "OZ",
        "schedule": "two_phase_jacobi",
        "nodes_processed": 1,
        "pointwise_epi_jump_certificate": None,
        "neighbor_epi_jump_certificate": None,
        "epi_jump_certificate_abstention_reason": None,
        "mutation_decision_observations": (),
        "reception_observations": (),
    }
    _require(
        _same(documents["event_after"]["stage"], stage)
        and _same(response["stage"], stage),
        "child completed-stage record differs",
    )
    previous = after
    for index, boundary in enumerate(assessment.boundaries):
        sample = documents[f"sample_{index}"]
        _require(
            type(sample["index"]) is int and sample["index"] == index,
            "child sample index differs",
        )
        _require(
            _represented(sample["time"], "sample clock") == boundary.sample_time,
            "child sample time differs",
        )
        graph = _restore(sample["endpoint"])
        _require(
            _same(sample["state"], boundary.state)
            and _same(sample["pressures"], boundary.pressures)
            and _state(graph) == boundary.state
            and _values(graph, ALIAS_DNFR) == boundary.pressures,
            "child sample differs from its full graph state",
        )
        _require(
            _same(sample["association"], boundary), "cached sample association differs"
        )
        if index == 0:
            _require(
                _state(after) == boundary.state
                and _values(after, ALIAS_DNFR) == boundary.pressures,
                "first sample is not the actual post-OZ boundary",
            )
        _flow_footprint(previous, graph, boundary.sample_time, sample=True)
        previous = graph
    graph = _restore(response["endpoint"])
    _require(
        _signature(graph) == _signature(previous),
        "final child endpoint differs from final sample",
    )
    signature = _signature(graph)
    fields = capture(graph)  # Includes current live Gamma and primitive admission.
    observation = observe_structural_acceleration(graph, CHILD)
    validate_self_organization_strict(graph, CHILD, emit_warnings=False)
    sequence = ValidatedSequence(
        (Dissonance(), SelfOrganization(), Silence()), context=CONTEXT
    )
    grammar = validate_candidate(
        graph, CHILD, "THOL", sequence_context=sequence.step(1)
    )
    _require(
        _same(response["observation"], observation)
        and _same(response["grammar"], grammar),
        "cached live readiness differs",
    )
    exact_stencil = (
        assessment.states[0][CHILD_INDEX]
        - 2 * assessment.states[1][CHILD_INDEX]
        + assessment.states[2][CHILD_INDEX]
    ) / SAMPLE_STEP**2
    _require(
        observation.available
        and observation.current_endpoint_matches_state
        and _represented(observation.value, "observed acceleration") == exact_stencil
        and exact_stencil > timing.threshold
        and grammar.allowed
        and node_debt(graph.nodes[CHILD]) == 1,
        "live child history or grammar does not admit the retained trigger",
    )
    _require(
        _signature(graph) == signature
        and _signature(certificate.source.retention.admitted.graph) == protected,
        "child source reader changed persistent state",
    )
    return ChildHistoryAdmission(
        graph, certificate, assessment, fields, observation, sequence, timing.threshold
    )


def load_child_history_source(evidence_dir, *, repository_root):
    """Reload and rebuild the retained child source before analytical THOL work.

    This preserves the inherited full-trace audit and uniform derivative-jet
    provenance. It does not rerun their producers or infer a future trajectory
    from the three certified numerical samples. Context/policies are checked
    on the actual mutable endpoint, not on an earlier parent's history.
    """
    folder, root = Path(evidence_dir).resolve(), Path(repository_root).resolve()
    documents, receipts = _read_child_history_artifacts(folder, root)
    certificate = prepare_child_sample_certificate(
        root / _JOINT_PATH, repository_root=root
    )
    prior = {_receipt_fields(row) for row in documents["freeze"]["prior_artifacts"]}
    _require(
        {f"{_JOINT_PATH}/{row['path']}" for row in certificate.source.receipts}
        <= prior,
        "child freeze omits joint source artifacts",
    )
    admitted = _admit_child_history(documents, certificate)
    return replace(admitted, receipts=receipts)
