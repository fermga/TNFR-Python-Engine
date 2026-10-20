"""Two frozen Silence suffixes on retained Coherence endpoints.

Only the explicitly declared consumed state is reconstructed; the upstream
response is not a complete runtime checkpoint. There is no Reception or
Coherence replay, continuous step, source acquisition or event selector.
An exclusive attempt precedes the two atomic events and retains partial failure.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import warnings
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402

from tnfr._exact_time import (  # noqa: E402
    exact_or_represented_real,
    finite_represented_real,
)
from tnfr.alias import get_attr, get_attr_str, set_attr_str, set_dnfr  # noqa: E402
from tnfr.constants.aliases import (  # noqa: E402
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.mathematics._rational_interval import (  # noqa: E402
    INTERVAL_BITS,
    I,
    pi_interval,
)
from tnfr.operators.definitions import Coherence, Reception, Silence  # noqa: E402
from tnfr.operators.factor_contracts import (  # noqa: E402
    resolve_runtime_operator_factors,
)
from tnfr.operators.grammar_debt import (  # noqa: E402
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
    canonical_operator_code,
    read_grammar_state,
    reset_grammar_state_from_history,
)
from tnfr.operators.grammar_execution import ValidatedSequence  # noqa: E402
from tnfr.operators.word_execution import execute_network_operator_stage  # noqa: E402
from tnfr.physics import _organized_source_capture as capture  # noqa: E402
from tnfr.physics import _organized_source_port as owner  # noqa: E402
from tnfr.research.artifact_io import (  # noqa: E402
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    validate_artifact_path,
    verify_archive_members,
    write_json_once,
)
from tnfr.types import Glyph, require_finite_real_scalar_epi  # noqa: E402
from tnfr.utils.io import json_loads  # noqa: E402

TARGETS = (0, 1)
PREPARATIONS = ("reception_handoff", "moving_control")
LATENCY_KEYS = (
    "latent",
    "latency_start_time",
    "preserved_epi",
    "silence_duration",
    "was_initial_on_silence",
)
RECONSTRUCTION_SUPPLEMENT = {
    "declared_absent": ("epi_history_during_silence",),
    "scope": (
        "configured absence in reconstructed runtime, not inferred from old snapshot",
        "legacy variance and preservation metrics have no observed temporal history",
        "legacy metrics are uninterpreted telemetry, not continuous maintenance evidence",
    ),
}


def _read(path):
    return decode_exact_tree(json_loads(read_bytes_bounded(path, max_bytes=2**27)))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_freeze(output):
    freeze = _read(output / "freeze.json")
    _require(
        freeze["protocol"]
        == file_receipt(output / "protocol.json", label="protocol.json"),
        "frozen protocol bytes changed",
    )
    archive = freeze["archive"]
    archive_path = output / validate_artifact_path(archive["path"])
    _require(
        archive == file_receipt(archive_path, label=archive["path"]), "archive changed"
    )
    verify_archive_members(archive_path, freeze["members"])
    for name, digest in freeze["members"].items():
        _require(
            file_receipt(ROOT / validate_artifact_path(name), label=name)["sha256"]
            == digest,
            f"live source changed: {name}",
        )


def _upstream(protocol):
    selected = protocol["upstream"]
    base = ROOT / validate_artifact_path(selected["directory"])
    _require(
        set(selected["receipts"])
        == {
            "protocol.json",
            "freeze.json",
            "attempt.json",
            "response.json",
            "source.zip",
        },
        "upstream inventory changed",
    )
    for name, receipt in selected["receipts"].items():
        _require(
            receipt == file_receipt(base / name, label=name),
            f"upstream bytes changed: {name}",
        )
    freeze, attempt = _read(base / "freeze.json"), _read(base / "attempt.json")
    for name in ("protocol", "freeze"):
        _require(
            attempt[name] == file_receipt(base / f"{name}.json", label=f"{name}.json"),
            f"upstream attempt receipt changed: {name}",
        )
    _require(freeze["archive"] == selected["receipts"]["source.zip"], "archive receipt")
    _require(
        freeze["protocol"] == selected["receipts"]["protocol.json"],
        "upstream protocol receipt",
    )
    verify_archive_members(base / "source.zip", freeze["members"])
    response = _read(base / "response.json")
    _require(response["status"] == "evaluated", "upstream response unavailable")
    snapshots = {}
    for name, preparation in protocol["preparations"].items():
        index = preparation["upstream_stage_index"]
        _require(type(index) is int, "invalid upstream stage index")
        stage = response["stages"][index]
        _require(
            stage["glyph"] == "IL" and stage["preparation"] == name,
            "wrong upstream endpoint",
        )
        snapshots[name] = stage["after"]
    return _read(base / "protocol.json"), snapshots


def _validate_protocol(protocol, upstream, geometry):
    config, support = protocol["configuration"], protocol["support"]
    _require(protocol["status_at_freeze"] == "not_evaluated", "invalid freeze")
    _require(tuple(protocol["preparations"]) == PREPARATIONS, "preparations changed")
    _require(support == upstream["support"], "support differs from upstream")
    _require(support["nodes"] == geometry.node_order, "node order changed")
    _require(support["weighted_edges"] == geometry.weighted_edges, "support changed")
    _require(support["targets"] == TARGETS, "targets changed")
    _require(config["word"] == ("EN", "IL", "SHA"), "word changed")
    _require(config["sha_word_index"] == 2, "word index changed")
    _require(config["phase_mobility"] == "shared_capacity", "phase law changed")
    _require(config["grammar_rejection_mode"] == "raise", "strict grammar required")
    _require(
        config["validate_operator_preconditions"] is True, "preconditions required"
    )
    _require(config["collect_operator_metrics"] is True, "metric recording required")
    _require(
        Q(resolve_runtime_operator_factors(None, Glyph.SHA, {})["SHA_vf_factor"])
        == config["silence_capacity_factor"],
        "resolved Silence factor changed",
    )
    for name, value in upstream["configuration"].items():
        _require(config[name] == value, f"upstream configuration changed: {name}")
    _require(
        protocol["phase_representation"]["turns"]
        == upstream["phase_representation"]["output_turns"],
        "retained lift changed",
    )
    _require(
        protocol["execution_counts"]
        == dict(
            atomic_silence_responses=2,
            reception=0,
            coherence=0,
            flow_steps=0,
            attempts=1,
        ),
        "event budget changed",
    )


def _neighbors(edges, node):
    return tuple(b if a == node else a for a, b, _ in edges if node in (a, b))


def _field(snapshot, name):
    return tuple(row[name] for row in snapshot["nodes"])


def _admit_source_snapshot(snapshot, geometry, protocol):
    """Normalize primitives before binary64 reconstruction or equality checks."""
    _require(
        set(snapshot) == {"nodes", "weighted_edges", "neighbors", "time"},
        "unexpected source snapshot fields",
    )

    def represented(value, label):
        exact = exact_or_represented_real(value, label)
        _materialized, normalized = finite_represented_real(exact, label)
        _require(normalized == exact, f"{label} is not the retained binary64 value")
        return normalized

    def ordinary(value, label):
        _require(type(value) is int and value >= 0, f"invalid ordinary {label}")
        return value

    expected_fields = {
        "node",
        "form",
        "phase",
        "phase_hex",
        "capacity",
        "pressure",
        "kind",
        "source_glyph",
        "history",
        "ordinal",
        "debt",
        "prior_coherence",
        "reception_sources",
        "stored_dEPI",
        "temporal_history_present",
    }
    rows = []
    _require(type(snapshot["nodes"]) is tuple and len(snapshot["nodes"]) == 45, "nodes")
    for row in snapshot["nodes"]:
        _require(type(row) is dict and set(row) == expected_fields, "node fields")
        node = ordinary(row["node"], "node")
        _require(node < 45, "node outside support")
        normalized = dict(row, node=node)
        for name in ("form", "phase", "capacity", "pressure", "stored_dEPI"):
            if name == "form":
                require_finite_real_scalar_epi(row[name])
            normalized[name] = represented(row[name], f"node {node} {name}")
        _require(normalized["capacity"] >= 0, "negative source capacity")
        _require(type(row["phase_hex"]) is str, "phase hex is not text")
        _require(float(normalized["phase"]).hex() == row["phase_hex"], "phase hex")
        for name in ("kind", "source_glyph"):
            _require(type(row[name]) is str, f"invalid {name}")
        _require(
            (
                canonical_operator_code(row["source_glyph"]) == row["source_glyph"]
                if row["source_glyph"]
                else True
            ),
            "source glyph must be a simple canonical code",
        )
        _require(type(row["history"]) is tuple, "source history must be ordered")
        history = tuple(canonical_operator_code(code) for code in row["history"])
        _require(history == row["history"], "history must use simple canonical codes")
        normalized["history"] = history
        normalized["ordinal"] = ordinary(row["ordinal"], "ordinal")
        normalized["debt"] = ordinary(row["debt"], "debt")
        if node in TARGETS:
            _require(
                history == protocol["acceptance"]["history_before"],
                "target history differs from frozen preparation",
            )
            _require(
                normalized["ordinal"] == protocol["acceptance"]["ordinal_before"],
                "target ordinal differs from frozen preparation",
            )
        for name in ("prior_coherence", "temporal_history_present"):
            _require(type(row[name]) is bool, f"{name} must be Boolean")
        read_grammar_state(
            {
                "glyph_history": history,
                OPERATOR_STEP_KEY: normalized["ordinal"],
                U2_DEBT_KEY: normalized["debt"],
                PRIOR_COHERENCE_KEY: row["prior_coherence"],
            }
        )
        sources = row["reception_sources"]
        _require(sources is None or type(sources) is tuple, "reception source metadata")
        if sources is not None:
            normalized["reception_sources"] = tuple(
                ordinary(source, "reception source") for source in sources
            )
            _require(
                all(source < 45 for source in sources),
                "reception source outside support",
            )
        rows.append(normalized)
    _require(
        tuple(row["node"] for row in rows) == geometry.node_order, "source node order"
    )
    _require(
        tuple(row["capacity"] for row in rows) == protocol["support"]["capacities"],
        "source capacities differ from the declared incoming model",
    )
    _require(
        all(row["kind"] == protocol["configuration"]["kind"] for row in rows),
        "source form kind differs from the declared preparation",
    )
    edges = []
    _require(type(snapshot["weighted_edges"]) is tuple, "ordered source edges required")
    for edge in snapshot["weighted_edges"]:
        _require(type(edge) is tuple and len(edge) == 3, "edge shape")
        a, b = ordinary(edge[0], "edge endpoint"), ordinary(edge[1], "edge endpoint")
        edges.append((a, b, represented(edge[2], "edge conductance")))
    _require(tuple(edges) == geometry.weighted_edges, "source support differs")
    _require(type(snapshot["neighbors"]) is tuple, "ordered neighbor lists required")
    neighbors = []
    for values in snapshot["neighbors"]:
        _require(type(values) is tuple, "neighbor row must be ordered")
        neighbors.append(tuple(ordinary(node, "neighbor") for node in values))
    _require(
        tuple(neighbors) == tuple(_neighbors(edges, n) for n in geometry.node_order),
        "source neighbor order differs",
    )
    time = represented(snapshot["time"], "structural time")
    _require(time == protocol["configuration"]["structural_time_tau"], "source time")
    return dict(
        nodes=tuple(rows),
        weighted_edges=tuple(edges),
        neighbors=tuple(neighbors),
        time=time,
    )


def _graph(protocol, snapshot):
    """Reconstruct the declared consumed subset; do not infer a checkpoint."""
    config = protocol["configuration"]
    graph = nx.Graph()
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=config["validate_operator_preconditions"],
        GRAMMAR_REJECTION_MODE=config["grammar_rejection_mode"],
        COLLECT_OPERATOR_METRICS=config["collect_operator_metrics"],
        EPI_MIN=float(config["epi_min"]),
        EPI_MAX=float(config["epi_max"]),
        CLIP_MODE=config["clip_mode"],
        IL_PRECONDITIONS={"activity_policy": config["activity_policy"]},
        SHA_MIN_VF=float(config["silence_min_capacity"]),
        RANDOM_SEED=config["seed"],
        _t=float(config["structural_time_tau"]),
    )
    for row in snapshot["nodes"]:
        _require(not row["temporal_history_present"], "unrecorded temporal history")
        graph.add_node(
            row["node"],
            EPI=float(row["form"]),
            EPI_kind=row["kind"],
            theta=float(row["phase"]),
            nu_f=float(row["capacity"]),
            delta_nfr=float(row["pressure"]),
            dEPI=float(row["stored_dEPI"]),
            glyph_history=list(row["history"]),
            **{OPERATOR_STEP_KEY: row["ordinal"]},
        )
        data = graph.nodes[row["node"]]
        set_attr_str(data, ALIAS_SOURCE_GLYPH, row["source_glyph"])
        if row["reception_sources"] is not None:
            data["_reception_sources"] = deepcopy(row["reception_sources"])
        reset_grammar_state_from_history(data)
        _require(data[U2_DEBT_KEY] == row["debt"], "grammar debt reconstruction")
        _require(
            data[PRIOR_COHERENCE_KEY] == row["prior_coherence"],
            "prior coherence reconstruction",
        )
    for a, b, weight in snapshot["weighted_edges"]:
        graph.add_edge(a, b, weight=float(weight))
    return graph


def _snapshot(graph, geometry):
    _require(not graph.is_directed() and not graph.is_multigraph(), "wrong graph")
    _require(tuple(graph) == geometry.node_order, "node order changed")
    _require(all(type(n) is int for n in graph), "invalid node identifiers")
    _require(graph.number_of_edges() == 90, "support changed")
    for a, b, weight in geometry.weighted_edges:
        _require(type(graph[a][b]["weight"]) is float, "invalid conductance")
        _require(Q(graph[a][b]["weight"]) == weight, "conductance changed")
    rows = []
    for n in graph:
        data = graph.nodes[n]

        def raw(aliases):
            return get_attr(data, aliases, strict=True, conv=lambda value: value)

        for aliases in (ALIAS_THETA, ALIAS_DNFR, ALIAS_VF):
            _require(type(raw(aliases)) is float, "invalid scalar representation")
            _require(math.isfinite(raw(aliases)), "nonfinite scalar")
        _require(
            tuple(graph.neighbors(n)) == _neighbors(geometry.weighted_edges, n),
            "neighbor order",
        )
        rows.append(
            dict(
                node=n,
                form=Q(require_finite_real_scalar_epi(raw(ALIAS_EPI))),
                phase=Q(raw(ALIAS_THETA)),
                phase_hex=raw(ALIAS_THETA).hex(),
                capacity=Q(raw(ALIAS_VF)),
                pressure=Q(raw(ALIAS_DNFR)),
                kind=get_attr_str(data, ALIAS_EPI_KIND),
                source_glyph=get_attr_str(data, ALIAS_SOURCE_GLYPH, default=""),
                history=tuple(data["glyph_history"]),
                ordinal=data[OPERATOR_STEP_KEY],
                debt=data.get(U2_DEBT_KEY, 0),
                prior_coherence=data.get(PRIOR_COHERENCE_KEY, False),
                reception_sources=deepcopy(data.get("_reception_sources")),
                stored_dEPI=Q(data["dEPI"]),
                temporal_history_present="epi_time_history" in data,
                lifecycle={
                    key: deepcopy(data[key]) for key in LATENCY_KEYS if key in data
                },
            )
        )
    return dict(
        nodes=tuple(rows),
        weighted_edges=geometry.weighted_edges,
        neighbors=tuple(tuple(graph.neighbors(n)) for n in graph),
        time=Q(graph.graph["_t"]),
    )


def _source_subset(snapshot):
    return {
        **snapshot,
        "nodes": tuple(
            {key: value for key, value in row.items() if key != "lifecycle"}
            for row in snapshot["nodes"]
        ),
    }


def _observe(snapshot):
    return owner._observe_organized_source_raw_capacity(
        form=_field(snapshot, "form"),
        phases=_field(snapshot, "phase"),
        capacities=_field(snapshot, "capacity"),
        phase_mobility="shared_capacity",
    )


def _diagnostics(value):
    """Retain unavailable diagnostic values explicitly, never as a zero."""
    if isinstance(value, float) and not math.isfinite(value):
        return dict(
            available=False,
            reason="nonfinite_runtime_diagnostic",
            representation=(
                "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"
            ),
        )
    if isinstance(value, dict):
        return {key: _diagnostics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_diagnostics(item) for item in value)
    return value


def _refresh(graph, geometry):
    before = _snapshot(graph, geometry)
    observation = _observe(before)
    metrics = _diagnostics(deepcopy(graph.graph.get("operator_metrics", [])))
    # Pressure is constitutive p, not the capacity-scaled nodal derivative.
    for node, pressure in zip(graph, observation.pressure_tau_bounds):
        set_dnfr(graph, node, float(pressure.midpoint))
    return dict(
        before=before,
        observation=asdict(observation),
        metrics_before_refresh=metrics,
        after=_snapshot(graph, geometry),
    )


def _stage(graph, geometry, context, record):
    record["before"] = _snapshot(graph, geometry)
    record["callback"] = []

    def refresh(candidate):
        record["callback"].append(_refresh(candidate, geometry))

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        stage = execute_network_operator_stage(
            graph,
            Silence(),
            TARGETS,
            sequence_context=context,
            compute_delta_nfr=refresh,
        )
    record["warnings"] = tuple(str(w.message) for w in captured)
    record["stage"] = dict(
        operator=stage.operator,
        glyph=stage.glyph,
        schedule=stage.schedule,
        nodes_processed=stage.nodes_processed,
    )
    record["after"] = _snapshot(graph, geometry)


def _analyze(record, protocol, prediction):
    before, after = record["before"], record["after"]
    old, new = _observe(before), _observe(after)
    retention = capture._assess_organized_source_raw_capacity_retention(
        form=_field(after, "form"),
        phases=_field(after, "phase"),
        capacities=_field(after, "capacity"),
        phase_turns=protocol["phase_representation"]["turns"],
    )
    held = all(
        _field(before, name) == _field(after, name)
        for name in (
            "form",
            "phase",
            "phase_hex",
            "kind",
            "stored_dEPI",
            "temporal_history_present",
            "reception_sources",
            "prior_coherence",
        )
    ) and all(
        before[name] == after[name] for name in ("weighted_edges", "neighbors", "time")
    )
    budget = protocol["budgets"]["pressure_materialization_tau"]
    acceptance = protocol["acceptance"]
    turns = protocol["phase_representation"]["turns"]
    weights = tuple(1 / nu for nu in new.capacities)
    weight_sum = sum(weights, Q(0))
    raw_sum = sum((w * p for w, p in zip(weights, _field(after, "phase"))), Q(0))
    turn_sum = sum((w * turn for w, turn in zip(weights, turns)), Q(0))
    slot_sum = sum((w * (n % 9 - 4) for n, w in enumerate(weights)), Q(0))
    weighted_origin = (
        I(raw_sum) + pi_interval() * (2 * turn_sum - 2 * slot_sum / 9)
    ) / weight_sum
    slow = 2 / new.unit_capacity_reference.geometry.gamma_bounds**2
    original = Q(1023, 1024)
    checks = dict(
        stage=record["stage"]
        == dict(
            operator=Silence.name,
            glyph="SHA",
            schedule=acceptance["schedule"],
            nodes_processed=len(acceptance["targets"]),
        ),
        held_fields=held,
        capacity=_field(after, "capacity") == prediction["capacities"],
        unchanged_storage=old.unit_capacity_reference == new.unit_capacity_reference,
        pressure_before_refresh=(
            len(record["callback"]) == 1
            and _field(record["callback"][0]["before"], "pressure")
            == _field(before, "pressure")
        ),
        constitutive_pressure_unchanged=old.pressure_tau_bounds
        == new.pressure_tau_bounds,
        refreshed_pressure=all(
            (I(value) - bounds).abs_max <= budget
            for value, bounds in zip(_field(after, "pressure"), new.pressure_tau_bounds)
        ),
        inverse_capacity_zero_form=new.inverse_capacity_weighted_form_total == 0,
        continuous_retention=retention.admitted,
        energy_ceiling=retention.energy_ceiling == acceptance["energy_ceiling"],
        strong_form_bound=(
            retention.form_squared_upper_bound is not None
            and retention.form_squared_upper_bound
            < acceptance["squared_form_bound_ceiling"]
        ),
        windings=retention.retained_windings == acceptance["retained_windings"],
        convergence=retention.converges_to_target is True,
        weighted_origin=retention.weighted_phase_origin_bounds == weighted_origin,
        fresh_rows_all_clocks=all(
            getattr(new, f"{channel}_{clock}_rate_bounds")
            == tuple(
                factor * value for value in getattr(new, f"{channel}_tau_rate_bounds")
            )
            for channel in ("form", "phase")
            for clock, factor in (("slow", slow), ("original", original))
        ),
        nonzero_fresh_target_rates=all(
            bounds.lo > 0 or bounds.hi < 0
            for n in TARGETS
            for bounds in (new.form_tau_rate_bounds[n], new.phase_tau_rate_bounds[n])
        ),
        common_wall_timestamp=(
            after["nodes"][0]["lifecycle"].get("latency_start_time")
            == after["nodes"][1]["lifecycle"].get("latency_start_time")
        ),
    )
    for n, (previous, current) in enumerate(zip(before["nodes"], after["nodes"])):
        if n in TARGETS:
            checks[f"target_{n}_grammar"] = (
                current["history"] == (*previous["history"], "SHA")
                and previous["ordinal"] == 4
                and current["ordinal"] == 5
                and current["source_glyph"] == "SHA"
                and current["debt"] == previous["debt"] == 0
            )
            lifecycle = current["lifecycle"]
            checks[f"target_{n}_latency"] = (
                lifecycle.get("latent") is True
                and Q(lifecycle["preserved_epi"]) == current["form"]
                and lifecycle.get("silence_duration") == 0
                and lifecycle.get("was_initial_on_silence")
                == (abs(float(current["form"])) < 1e-6)
                and isinstance(lifecycle.get("latency_start_time"), str)
            )
        else:
            checks[f"untargeted_{n}"] = {
                key: value for key, value in current.items() if key != "pressure"
            } == {key: value for key, value in previous.items() if key != "pressure"}
    record["analysis"] = dict(
        observation=asdict(new),
        retention=asdict(retention),
        event_storage_work=Q(0) if held else None,
        checks=checks,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    for name in ("attempt.json", "response.json"):
        _require(
            not (output / name).exists(), "existing attempt forbids another evaluation"
        )
    protocol = _read(output / "protocol.json")
    _verify_freeze(output)
    upstream, snapshots = _upstream(protocol)
    geometry = owner._derive_organized_source_port()
    _validate_protocol(protocol, upstream, geometry)
    snapshots = {
        name: _admit_source_snapshot(snapshot, geometry, protocol)
        for name, snapshot in snapshots.items()
    }
    factor = float(protocol["configuration"]["silence_capacity_factor"])
    predictions = {
        name: dict(
            capacities=tuple(
                Q(float(value) * factor) if n in TARGETS else value
                for n, value in enumerate(_field(snapshot, "capacity"))
            ),
            event_storage_work=Q(0),
            phase_turns=protocol["phase_representation"]["turns"],
        )
        for name, snapshot in snapshots.items()
    }
    write_json_once(
        output / "attempt.json",
        dict(
            schema="tnfr.organized-source-silence-handoff.attempt.v1",
            status="not_evaluated",
            protocol=file_receipt(output / "protocol.json", label="protocol.json"),
            freeze=file_receipt(output / "freeze.json", label="freeze.json"),
            driver=file_receipt(
                Path(__file__), label="benchmarks/organized_source_silence_handoff.py"
            ),
            upstream=protocol["upstream"],
            python=platform.python_version(),
            platform=platform.platform(),
            networkx=nx.__version__,
            interval_bits=INTERVAL_BITS,
            predictions=predictions,
            execution_counts=protocol["execution_counts"],
            reconstruction_supplement=RECONSTRUCTION_SUPPLEMENT,
        ),
    )
    response = dict(
        schema="tnfr.organized-source-silence-handoff.response.v1",
        status="incomplete",
        stages=[],
    )
    try:
        sequence = ValidatedSequence(
            (Reception(), Coherence(), Silence()),
            context={
                "initial_epi_nonzero": True,
                "compatibility_profile": protocol["configuration"][
                    "compatibility_profile"
                ],
            },
        )
        for name, source in snapshots.items():
            graph = _graph(protocol, source)
            initial = _snapshot(graph, geometry)
            _require(
                _source_subset(initial) == source, "source subset was not reconstructed"
            )
            _require(not any(_field(initial, "lifecycle")), "unexpected source latency")
            record = dict(glyph="SHA", preparation=name)
            response["stages"].append(record)
            _stage(graph, geometry, sequence.step(2), record)
            _analyze(record, protocol, predictions[name])
        response["status"] = "evaluated"
        response["all_analysis_checks"] = all(
            all(record["analysis"]["checks"].values()) for record in response["stages"]
        )
    except BaseException as exc:
        response["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json_once(output / "response.json", response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
