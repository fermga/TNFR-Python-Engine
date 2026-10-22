"""Evaluate one frozen complete NUL/IL/SHA word and nonlinear continuation.

No previous producer is imported. The retained graph is a new preparation
from explicitly recorded inputs, not an authenticated process checkpoint.
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from fractions import Fraction as Q
from importlib.metadata import version
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from tnfr._exact_time import finite_represented_real  # noqa: E402
from tnfr.alias import set_dnfr  # noqa: E402
from tnfr.dynamics.relational import RelationalExchangeModel  # noqa: E402
from tnfr.mathematics._rational_interval import INTERVAL_BITS, I  # noqa: E402
from tnfr.operators import network_stage  # noqa: E402
from tnfr.operators.definitions import Coherence, Contraction, Silence  # noqa: E402
from tnfr.operators.grammar_debt import node_debt  # noqa: E402
from tnfr.operators.grammar_dynamics import validate_candidate  # noqa: E402
from tnfr.operators.grammar_execution import ValidatedSequence  # noqa: E402
from tnfr.operators.grammar_u6 import (  # noqa: E402
    observe_structural_potential_confinement,
)
from tnfr.operators.preconditions import (  # noqa: E402
    validate_coherence,
    validate_contraction,
    validate_silence,
)
from tnfr.operators.word_execution import execute_network_operator_stage  # noqa: E402
from tnfr.physics.fields import compute_structural_potential  # noqa: E402
from tnfr.physics.relational_sine_support import (  # noqa: E402
    bound_relational_sine_support_exchange,
)
from tnfr.research.artifact_io import (  # noqa: E402
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
    write_json_once,
)
from tnfr.research.collective_form_capacity_maintenance import (  # noqa: E402
    forecast_contact,
)
from tnfr.research.retained_graph_state import (  # noqa: E402
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils.io import json_loads  # noqa: E402

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
NOW = datetime(2030, 1, 2, 3, 4, 6, tzinfo=timezone.utc)


class SuppliedClock:
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)


def read(path):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(path, max_bytes=64 * 1024**2))
    )


def encoded(value):
    if isinstance(value, I):
        return {"lo": value.lo, "hi": value.hi}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: encoded(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): encoded(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(encoded(item) for item in value)
    return value


def capture_graph(graph):
    """Acquire canonical fields and explicit persistent-state provenance."""
    potential = compute_structural_potential(graph)
    return {
        "state": record_graph_state(graph),
        "fields": bound_relational_sine_support_exchange(graph, reference_model=MODEL),
        "structural_potential": tuple((node, Q(potential[node])) for node in graph),
    }


def u6_observation(graph, before, after, label):
    """Use the declared canonical finite drift monitor without a controller."""
    return observe_structural_potential_confinement(
        graph,
        dict(before["structural_potential"]),
        dict(after["structural_potential"]),
        reference=label,
        time_coverage="same_time_two_snapshot_event_observation",
    )


def verify():
    freeze = read(EVIDENCE / "freeze.json")
    if freeze["evaluation_status_at_freeze"] != "not_evaluated":
        raise ValueError("the protocol must precede evaluation")
    for entry in freeze["artifacts"]:
        if file_receipt(EVIDENCE / entry["path"], label=entry["path"]) != entry:
            raise ValueError(f"frozen artifact changed: {entry['path']}")
    for entry in freeze["inputs"]:
        if file_receipt(ROOT / entry["path"], label=entry["path"]) != entry:
            raise ValueError(f"retained input changed: {entry['path']}")
    entries = read(EVIDENCE / "source-manifest.json")["files"]
    verify_archive_members(
        EVIDENCE / "source.zip",
        {entry["path"]: entry["sha256"] for entry in entries},
        max_bytes=64 * 1024**2,
    )
    for entry in entries:
        if file_receipt(ROOT / entry["path"], label=entry["path"]) != entry:
            raise ValueError(f"working source differs from freeze: {entry['path']}")
    if any(
        (EVIDENCE / name).exists()
        for name in ("attempt.json", "response.json", "failure.json")
    ):
        raise FileExistsError("this one-shot protocol already has an attempt")


def calculate():
    protocol = read(EVIDENCE / "protocol.json")
    if protocol["interval_bits"] != INTERVAL_BITS:
        raise ValueError("interval precision differs from the protocol")
    original = read(ROOT / protocol["source_response"])
    reconstruction = restore_recorded_graph_state(
        original["captures"]["OZ_completed"]["state"]
    )
    graph = reconstruction.graph
    if (
        graph.graph.get("GRAMMAR_REJECTION_MODE") != "raise"
        or graph.graph.get("VALIDATE_OPERATOR_PRECONDITIONS") is not True
    ):
        raise ValueError("retained live rejection/precondition policy changed")
    if tuple(graph) != tuple(range(20)) or graph.graph.get("_t") != 7:
        raise ValueError("retained source order or clock changed")
    events, refreshes = [], []

    def capture():
        return capture_graph(graph)

    def u6(before, after, label):
        return u6_observation(graph, before, after, label)

    def refresh(subject, label):
        if subject is not graph:
            raise ValueError("pressure callback received another graph")
        before = capture()
        state = before["fields"]
        values = tuple(
            finite_represented_real(value.midpoint, "fresh selected-law pressure")
            for value in state.pressure
        )
        for node, (materialized, _) in zip(state.nodes, values, strict=True):
            set_dnfr(graph, node, materialized)
        after = capture()
        other = after["fields"]
        if (
            state.epi,
            state.phase,
            state.capacity,
            state.edges,
            state.conductances,
        ) != (other.epi, other.phase, other.capacity, other.edges, other.conductances):
            raise ValueError("pressure callback altered a different consumed channel")
        refreshes.append(
            {
                "label": label,
                "before": before,
                "after": after,
                "residuals": tuple(
                    I(represented) - expected
                    for (_, represented), expected in zip(
                        values, state.pressure, strict=True
                    )
                ),
                "u6_raw_to_refreshed": u6(
                    before, after, f"{label}_raw_before_pressure_refresh"
                ),
            }
        )

    initial = capture()
    if any(initial["fields"].epi[node] == 0 for node in (0, 1)):
        raise ValueError("the source does not supply initialized-EPI grammar context")
    operators = (Contraction(), Coherence(), Silence())
    preconditions = (validate_contraction, validate_coherence, validate_silence)
    context = {"initial_epi_nonzero": True, "compatibility_profile": "core"}
    sequence = ValidatedSequence(operators, context=context)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", SuppliedClock)
        refresh(graph, "initial")
        for index, operator in enumerate(operators):
            label = operator.glyph.value
            before = capture()
            candidates = tuple(
                validate_candidate(graph, node, label) for node in (0, 1)
            )
            if not all(candidate.allowed for candidate in candidates):
                raise ValueError(f"live grammar rejected {label}")
            debt_before = tuple(node_debt(graph.nodes[node]) for node in (0, 1))
            for node in (0, 1):
                preconditions[index](graph, node)
            stage = execute_network_operator_stage(
                graph,
                operator,
                (0, 1),
                sequence_context=sequence.step(index),
                compute_delta_nfr=lambda subject, label=label: refresh(subject, label),
                include_epi_jump_certificate=False,
            )
            if (
                stage.glyph != label
                or stage.schedule != "two_phase_jacobi"
                or stage.nodes_processed != 2
            ):
                raise ValueError(
                    "the actual stage differs from the frozen admitted action"
                )
            events.append(
                {
                    "glyph": label,
                    "before": before,
                    "stage": stage,
                    "live_candidates": candidates,
                    "scalar_preconditions": "passed_on_both_live_targets_before_stage",
                    "u2_debt_before": debt_before,
                    "u2_debt_after": tuple(
                        node_debt(graph.nodes[node]) for node in (0, 1)
                    ),
                    "completed": capture(),
                }
            )
            events[-1]["u6_stage"] = u6(
                before, events[-1]["completed"], f"before_{label}_stage"
            )
            print(f"stage {label} completed", flush=True)
    completed = capture()
    if any(node_debt(graph.nodes[node]) != 0 for node in (0, 1)):
        raise ValueError("the complete word left unresolved live U2 debt")
    if (
        tuple(graph) != tuple(range(20))
        or initial["fields"].edges != completed["fields"].edges
        or initial["fields"].conductances != completed["fields"].conductances
    ):
        raise ValueError("the word changed the fixed support or conductances")

    def progress(number, record):
        if (
            number == 1
            or number % 10 == 0
            or record["endpoint_identity_squared"].lo > Q(1, 256)
        ):
            print(
                f"step {number}/160 t={float(number / 4):.2f} Z2=[{float(record['endpoint_identity_squared'].lo):.9g},{float(record['endpoint_identity_squared'].hi):.9g}] radius={float(record['endpoint_metric_radius']):.3g}",
                flush=True,
            )

    forecast = forecast_contact(completed["fields"], progress=progress)
    return {
        "schema": "tnfr.collective-form-capacity-maintenance.v1",
        "initial": initial,
        "whole_word_admission": {
            "names": sequence.names,
            "context": context,
            "validated_sequence_constructed": True,
        },
        "events": tuple(events),
        "refreshes": tuple(refreshes),
        "completed": completed,
        "u6_complete_word": u6(initial, completed, "retained_source_before_NUL_IL_SHA"),
        "forecast": forecast,
        "scope": "one reconstructed represented preparation; complete NUL/IL/SHA word; exact selected-law continuation; no all-time or source-family claim",
    }


def main():
    verify()
    write_json_once(
        EVIDENCE / "attempt.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": file_receipt(EVIDENCE / "freeze.json")["sha256"],
            "python": sys.version,
            "executable": sys.executable,
            "versions": {
                name: version(name) for name in ("numpy", "networkx", "pytest")
            },
            "maximum_evaluations": 1,
        },
    )
    try:
        result = calculate()
        write_json_once(EVIDENCE / "response.json", encoded(result))
        forecast = result["forecast"]
        write_json_once(
            EVIDENCE / "report.json",
            encoded(
                {
                    "status": forecast["status"],
                    "reason": forecast["reason"],
                    "validated_duration": forecast["validated_duration"],
                    "complete_word_executed": True,
                    "stage_count": len(result["events"]),
                    "all_time_maintenance_claimed": False,
                    "maximum_evaluations": 1,
                }
            ),
        )
        print(f"result: {forecast['status']}", flush=True)
    except Exception as exc:
        write_json_once(
            EVIDENCE / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
