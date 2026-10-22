"""One frozen Reception prerequisite and two atomic Coherence responses.

This instrument consumes the separately frozen protocol. It never advances a
continuous trajectory or selects another preparation. Predictions and an
exclusive attempt record precede every event; failure retains a partial response
and cannot trigger a retry. Read-only evidence tests do not import this driver.
"""

from __future__ import annotations

import argparse
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

from tnfr.alias import get_attr, get_attr_str, set_dnfr  # noqa: E402
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
    arg,
    cos,
    pi_interval,
    sin,
)
from tnfr.operators._coherence_stage_kernel import (  # noqa: E402
    DEFAULT_PHASE_LOCKING_COEFFICIENT,
)
from tnfr.operators.definitions import Coherence, Reception, Silence  # noqa: E402
from tnfr.operators.factor_contracts import (  # noqa: E402
    resolve_runtime_operator_factors,
)
from tnfr.operators.grammar_debt import (  # noqa: E402
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
    reset_grammar_state_from_history,
)
from tnfr.operators.grammar_execution import ValidatedSequence  # noqa: E402
from tnfr.operators.word_execution import execute_network_operator_stage  # noqa: E402
from tnfr.physics import _organized_source_port as owner  # noqa: E402
from tnfr.research.artifact_io import (  # noqa: E402
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    validate_artifact_path,
    verify_archive_members,
    write_json_once,
)
from tnfr.research.phase_reset_certificate import bound_phase_reset  # noqa: E402
from tnfr.types import Glyph, require_finite_real_scalar_epi  # noqa: E402
from tnfr.utils.io import json_loads  # noqa: E402

TARGETS = (0, 1)


def _read(path):
    return decode_exact_tree(json_loads(read_bytes_bounded(path, max_bytes=2**24)))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _neighbors(edges, node):
    return tuple(b if a == node else a for a, b, _ in edges if node in (a, b))


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
        path = ROOT / validate_artifact_path(name)
        _require(
            file_receipt(path, label=name)["sha256"] == digest,
            f"live source changed: {name}",
        )


def _validate_protocol(protocol, geometry):
    """Fail before an attempt if a consumed frozen coefficient has drifted."""
    config, support = protocol["configuration"], protocol["support"]
    _require(protocol["status_at_freeze"] == "not_evaluated", "invalid freeze")
    _require(support["nodes"] == geometry.node_order, "node order changed")
    _require(support["weighted_edges"] == geometry.weighted_edges, "support changed")
    _require(support["targets"] == TARGETS, "targets changed")
    _require(support["capacities"] == (Q(1),) * 45, "capacity changed")
    _require(config["word"] == ("EN", "IL", "SHA"), "word changed")
    _require(config["il_word_index"] == 1, "word index changed")
    _require(config["activity_policy"] == "event_domain", "policy changed")
    _require(
        config["phase_locking_coefficient"] == Q(DEFAULT_PHASE_LOCKING_COEFFICIENT),
        "phase coefficient changed",
    )
    for glyph, factor, key in (
        (Glyph.EN, "EN_mix", "reception_mix"),
        (Glyph.IL, "IL_dnfr_factor", "pressure_retention"),
    ):
        _require(
            Q(resolve_runtime_operator_factors(None, glyph, {})[factor]) == config[key],
            f"resolved {factor} changed",
        )
    _require(
        protocol["execution_counts"]
        == dict(
            atomic_reception_preparation=1,
            atomic_coherence_responses=2,
            silence=0,
            serial_controls=0,
            flow_steps=0,
            attempts=1,
        ),
        "event budget changed",
    )
    turns = protocol["phase_representation"]
    _require(turns["input_turns"] == (0,) * 45, "input lift changed")
    _require(turns["output_turns"] == (-1, -1) + (0,) * 43, "output lift changed")


def _ideal_phases(geometry, offsets):
    return tuple(geometry.alpha_bounds * (n % 9 - 4) + v for n, v in enumerate(offsets))


def _phase_prediction(phases, edges, coefficient):
    """Independent simultaneous resultant and nine-edge work before dispatch."""
    displacement = [I(0)] * 45
    resultants = []
    for n in TARGETS:
        gaps = tuple(phases[j] - phases[n] for j in _neighbors(edges, n))
        real = sum((cos(gap) for gap in gaps), I(0))
        imaginary = sum((sin(gap) for gap in gaps), I(0))
        displacement[n] = coefficient * arg(real, imaginary)
        resultants.append((real, imaginary))
    after = tuple(value + delta for value, delta in zip(phases, displacement))
    work = sum(
        (
            weight * (cos(phases[b] - phases[a]) - cos(after[b] - after[a]))
            for a, b, weight in edges
            if a in TARGETS or b in TARGETS
        ),
        I(0),
    )
    return dict(
        phases=phases,
        displacement=tuple(displacement),
        after=after,
        resultants=tuple(resultants),
        work=work,
    )


def _freeze_predictions(protocol, geometry):
    config = protocol["configuration"]
    result = {}
    for name, preparation in protocol["preparations"].items():
        phases = _ideal_phases(geometry, preparation["phase_offsets"])
        result[name] = _phase_prediction(
            phases, geometry.weighted_edges, config["phase_locking_coefficient"]
        )
    form = protocol["preparations"]["reception_handoff"]["form"]
    contraction = 1 - 6 * config["reception_mix"] / 5
    result["reception"] = dict(
        form_after=(contraction * form[0], contraction * form[1]) + form[2:],
        work=6 * form[0] ** 2 * (contraction**2 - 1),
        form_error_budget=Q(1, 10**17),
    )
    return result


def _tree(value):
    """Detach only declared interval/prediction trees, retaining exact endpoints."""
    if isinstance(value, I):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_tree(item) for item in value)
    return value


def _graph(protocol, geometry, name):
    preparation, config = protocol["preparations"][name], protocol["configuration"]
    graph = nx.Graph()
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=config["validate_operator_preconditions"],
        GRAMMAR_REJECTION_MODE=config["grammar_rejection_mode"],
        COLLECT_OPERATOR_METRICS=config["collect_operator_metrics"],
        EPI_MIN=float(config["epi_min"]),
        EPI_MAX=float(config["epi_max"]),
        CLIP_MODE=config["clip_mode"],
        IL_PRECONDITIONS={"activity_policy": config["activity_policy"]},
        RANDOM_SEED=config["seed"],
        _t=float(config["structural_time_tau"]),
    )
    for node in geometry.node_order:
        prefix = "target" if node in TARGETS else "other"
        history = preparation.get(f"{prefix}_history", preparation.get("history"))
        ordinal = preparation.get(f"{prefix}_ordinal", preparation.get("ordinal"))
        graph.add_node(
            node,
            EPI=float(preparation["form"][node]),
            EPI_kind=config["kind"],
            theta=float(
                geometry.alpha_bounds.midpoint * (node % 9 - 4)
                + preparation["phase_offsets"][node]
            ),
            nu_f=1.0,
            delta_nfr=0.0,
            dEPI=float(config["stored_dEPI"]),
            glyph_history=list(history),
            **{OPERATOR_STEP_KEY: ordinal},
        )
        reset_grammar_state_from_history(graph.nodes[node])
    for a, b, weight in geometry.weighted_edges:
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
            return get_attr(data, aliases, strict=True, conv=lambda v: v)

        _require(type(raw(ALIAS_THETA)) is float, "invalid phase representation")
        _require(type(raw(ALIAS_DNFR)) is float, "invalid pressure representation")
        _require(
            tuple(graph.neighbors(n)) == _neighbors(geometry.weighted_edges, n),
            "neighbor order",
        )
        _require(
            type(raw(ALIAS_VF)) is float and raw(ALIAS_VF) == 1.0, "capacity changed"
        )
        form = require_finite_real_scalar_epi(raw(ALIAS_EPI))
        rows.append(
            dict(
                node=n,
                form=Q(form),
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
            )
        )
    return dict(
        nodes=tuple(rows),
        weighted_edges=geometry.weighted_edges,
        neighbors=tuple(tuple(graph.neighbors(n)) for n in graph),
        time=Q(graph.graph["_t"]),
    )


def _observe(snapshot):
    return owner._observe_organized_source_phase_state(
        form=tuple(row["form"] for row in snapshot["nodes"]),
        phases=tuple(row["phase"] for row in snapshot["nodes"]),
    )


def _refresh(graph, geometry):
    before = _snapshot(graph, geometry)
    observation = _observe(before)
    metrics_before_refresh = deepcopy(graph.graph.get("operator_metrics", []))
    for node, pressure in zip(graph, observation.pressure_tau_bounds):
        set_dnfr(graph, node, float(pressure.midpoint))
    return dict(
        before=before,
        observation=asdict(observation),
        metrics_before_refresh=metrics_before_refresh,
        after=_snapshot(graph, geometry),
    )


def _stage(graph, geometry, operator, context, record):
    record["before"] = _snapshot(graph, geometry)
    record["callback"] = []

    def refresh(candidate):
        record["callback"].append(_refresh(candidate, geometry))

    # Actual shared atomic stage: no replacement executor, selector or law.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        stage = execute_network_operator_stage(
            graph,
            operator,
            TARGETS,
            sequence_context=context,
            compute_delta_nfr=refresh,
        )
    record["warnings"] = tuple(str(w.message) for w in captured)
    record["telemetry"] = {
        key: deepcopy(graph.graph.get(key, []))
        for key in ("IL_phase_locking", "IL_dnfr_reductions", "IL_coherence_tracking")
    }
    record["stage"] = dict(
        operator=stage.operator,
        glyph=stage.glyph,
        schedule=stage.schedule,
        nodes_processed=stage.nodes_processed,
        reception_observations=tuple(
            {
                key: value
                for key, value in asdict(observation).items()
                if not key.startswith("_") and key != "glyph"
            }
            for observation in stage.reception_observations
        ),
    )
    record["after"] = _snapshot(graph, geometry)


def _analyze_il(record, protocol, geometry, prediction):
    before, after = record["before"], record["after"]
    old = tuple(row["phase"] for row in before["nodes"])
    new = tuple(row["phase"] for row in after["nodes"])
    coefficient = protocol["configuration"]["phase_locking_coefficient"]
    budgets = protocol["budgets"]
    certificates = bound_phase_reset(
        old,
        new,
        before["neighbors"],
        TARGETS,
        coefficient=coefficient,
        budget=budgets["runtime_reset"],
    )
    lifted = tuple(
        I(value) + 2 * pi_interval() * turn
        for value, turn in zip(new, protocol["phase_representation"]["output_turns"])
    )
    defects = tuple(
        actual - ideal for actual, ideal in zip(lifted, prediction["after"])
    )
    input_prediction = _phase_prediction(
        tuple(I(v) for v in old), geometry.weighted_edges, coefficient
    )
    old_report, new_report = _observe(before), _observe(after)
    work = new_report.joint_storage_bounds - old_report.joint_storage_bounds
    edge_angles = []
    for a, b, _ in geometry.weighted_edges:
        gap = lifted[b] - lifted[a]
        if a // 9 == b // 9 and a % 9 == 8 and b % 9 == 0:
            gap += 2 * pi_interval()
        edge_angles.append(gap)
    baseline = 45 * (1 - cos(geometry.alpha_bounds))
    excess = new_report.joint_storage_bounds - baseline
    pair_increment = lifted[0] + lifted[1] - old[0] - old[1]
    record["analysis"] = _tree(
        dict(
            certificates=tuple(asdict(certificate) for certificate in certificates),
            lifted_after=lifted,
            source_endpoint_defects=defects,
            work=work,
            actual_input_ideal_work=input_prediction["work"],
            work_vs_actual_input_ideal=work - input_prediction["work"],
            work_vs_exact_source=work - prediction["work"],
            edge_angles=tuple(edge_angles),
            excess=excess,
            pair_increment_defect=pair_increment,
            exact_reflection_certified=False,
            future_retention_certified=False,
            checks=dict(
                reset=all(
                    c.classification == "certified_within_budget" for c in certificates
                ),
                source_endpoint=all(
                    d.abs_max <= budgets["exact_source_endpoint"] for d in defects
                ),
                fixed_nearest_lift=all(
                    (lifted[n] - old[n] - c.ideal_displacement).abs_max
                    < pi_interval().lo
                    for n, c in zip(TARGETS, certificates)
                ),
                acute=all(v.abs_max < pi_interval().lo / 2 for v in edge_angles),
                energy=excess.hi < Q(1, 100),
                work_source=(work - prediction["work"]).abs_max
                <= budgets["work_vs_exact_source"],
                work_actual=(work - input_prediction["work"]).abs_max
                <= budgets["work_vs_actual_input_ideal"],
                work_sign_or_quiet=(
                    work.hi < 0
                    if record["preparation"] == "moving_control"
                    else work.abs_max <= budgets["work_vs_exact_source"]
                ),
            ),
        )
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
    geometry = owner._derive_organized_source_port()
    _validate_protocol(protocol, geometry)
    predictions = _freeze_predictions(protocol, geometry)
    attempt = dict(
        schema="tnfr.organized-source-coherence-handoff.attempt.v1",
        status="not_evaluated",
        protocol=file_receipt(output / "protocol.json", label="protocol.json"),
        freeze=file_receipt(output / "freeze.json", label="freeze.json"),
        driver=file_receipt(
            Path(__file__), label="benchmarks/organized_source_coherence_handoff.py"
        ),
        python=platform.python_version(),
        platform=platform.platform(),
        networkx=nx.__version__,
        interval_bits=INTERVAL_BITS,
        predictions=_tree(predictions),
        execution_counts=protocol["execution_counts"],
    )
    write_json_once(output / "attempt.json", attempt)
    response = dict(
        schema="tnfr.organized-source-coherence-handoff.response.v1",
        status="incomplete",
        stages=[],
        initial={},
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
        graphs = {}
        for name in protocol["preparations"]:
            graph = _graph(protocol, geometry, name)
            graphs[name] = graph
            response["initial"][name] = _refresh(graph, geometry)
            snapshot = response["initial"][name]["after"]
            defects = tuple(
                I(row["phase"]) - ideal
                for row, ideal in zip(snapshot["nodes"], predictions[name]["phases"])
            )
            response["initial"][name]["rendering_defects"] = _tree(defects)
            _require(
                all(
                    defect.abs_max <= protocol["budgets"]["rendering"]
                    for defect in defects
                ),
                "prepared phase rendering exceeds frozen budget before events",
            )
        graph = graphs["reception_handoff"]
        preparation = dict(glyph="EN", preparation="reception_handoff")
        response["stages"].append(preparation)
        _stage(graph, geometry, Reception(), sequence.step(0), preparation)
        for name in ("reception_handoff", "moving_control"):
            record = dict(glyph="IL", preparation=name)
            response["stages"].append(record)
            _stage(graphs[name], geometry, Coherence(), sequence.step(1), record)
            _analyze_il(record, protocol, geometry, predictions[name])
        response["status"] = "evaluated"
        response["all_analysis_checks"] = all(
            all(row["analysis"]["checks"].values())
            for row in response["stages"]
            if row["glyph"] == "IL"
        )
    except Exception as exc:
        response["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json_once(output / "response.json", response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
