"""Matched finite C6 event schedules with freshly generated branch pressures.

Both branches reuse the three existing preparations, canonical UM/IL/SHA
word and eight explicit physical segments. The carried and ordinary solvers
each refresh pressure on their own live displayed EPI. Their accumulated
nodal sources are recorded separately from executor rounding defects.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.c6_winding_joint_domain import CASES  # noqa: E402
from benchmarks.c6_winding_phase_response import prepare_c6_phase_response  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.operators.event_runtime import execute_operator_event_schedule  # noqa: E402
from tnfr.operators.event_timing import (  # noqa: E402
    build_operator_event_schedule, build_physical_flow_partition,
)
from tnfr.operators.nodal_remainder_runtime import execute_nodal_remainder_event_schedule  # noqa: E402
from tnfr.physics.forcing_realization import capture_non_epi_forcing  # noqa: E402
from tnfr.physics.nodal_remainder_pressure import observe_nodal_remainder_pressure_readout  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402

SOURCE_SCOPE = ("src/tnfr", "benchmarks")
WORD = ("coupling", "coherence", "coupling", "coherence", "silence")
DURATIONS = (0.0, 0.0, .25, 0.0, .25, 0.0)
SEGMENT_DURATIONS = (.0625,) * 4


def _mean(values):
    values = tuple(values)
    return sum(values, Fraction(0)) / len(values)


def _schedule():
    schedule = build_operator_event_schedule(WORD, start_time=0.0, flow_durations=DURATIONS)
    partitions = tuple(build_physical_flow_partition(interval, SEGMENT_DURATIONS)
                       for interval in schedule.intervals if interval.exact_duration > 0)
    return schedule, partitions


def _ordinary_budget(execution):
    if not execution._proof_fields_are_intact():
        raise RuntimeError("ordinary reference lost its finite executor evidence")
    if len(execution.glyph_stage_evidence) != len(WORD):
        raise RuntimeError("ordinary comparison requires every declared event boundary")
    for stage in execution.glyph_stage_evidence:
        if (stage.left is None or stage.right is None or stage.left.nodes != stage.right.nodes
                or stage.left.exact_epi != stage.right.exact_epi):
            raise RuntimeError("ordinary comparison requires EPI-preserving executed events")
    records = []
    cumulative_area, cumulative_visible = Fraction(0), Fraction(0)
    for partition in execution.physical_flow_partition_evidence:
        for flow in partition.segment_flow_evidence:
            certificate = flow.certificate
            if (certificate is None or not flow.integrator_provenance_certified
                    or flow.resolved_substeps != 1 or flow.resolved_method != "euler"
                    or flow.clipping_applied is not False):
                raise RuntimeError("matched ordinary comparison requires one unclipped default Euler step")
            before, after = certificate.left, certificate.right
            area = tuple(certificate.exact_duration * capacity * pressure for capacity, pressure
                         in zip(before.exact_nu_f, before.exact_delta_nfr, strict=True))
            visible = tuple(y - x for x, y in zip(before.exact_epi, after.exact_epi, strict=True))
            cumulative_area += _mean(area)
            cumulative_visible += _mean(visible)
            records.append({
                "left": asdict(before), "right": asdict(after),
                "duration": certificate.exact_duration, "nodal_area": area,
                "visible_change": visible, "cumulative_mean_area": cumulative_area,
                "cumulative_visible_mean_change": cumulative_visible,
                "cumulative_executor_mean_defect": cumulative_visible - cumulative_area,
            })
    return {
        "flows": records, "mean_area": cumulative_area, "visible_mean_change": cumulative_visible,
        "executor_mean_defect": cumulative_visible - cumulative_area,
        "pressure_refresh_count": execution.pressure_refresh_callback_invocations,
        "event_names": tuple(event.operator_name for event in execution.events),
        "runtime_provenance_certified": True,
    }


def run_c6_remainder_runtime_case(mode, epsilon):
    if type(epsilon) is not float or (mode, epsilon) not in CASES:
        raise ValueError("mode and epsilon must identify one of the three inherited C6 controls")
    graph = prepare_c6_phase_response(mode, epsilon)
    ordinary_graph = prepare_c6_phase_response(mode, epsilon)
    initial = capture_non_epi_forcing(graph)
    ordinary_initial = capture_non_epi_forcing(ordinary_graph)
    if initial != ordinary_initial:
        raise RuntimeError("matched solver preparations differ")
    schedule, partitions = _schedule()
    execution = execute_nodal_remainder_event_schedule(
        graph, schedule, physical_flow_partitions=partitions,
    )
    ordinary_execution = execute_operator_event_schedule(
        ordinary_graph, schedule, context={"initial_epi_nonzero": min(initial.snapshot.epi) > 0},
        method="euler", physical_flow_partitions=partitions, include_stage_certificates=True,
    )
    if not execution.runtime_provenance_certified or execution.prefix_budget is None:
        raise RuntimeError("carried comparison lacks sealed runtime or prefix evidence")
    ordinary = _ordinary_budget(ordinary_execution)
    flows = []
    for flow in execution.flows:
        readout = observe_nodal_remainder_pressure_readout(
            state=flow.step.before, conductance=flow.before_snapshot.conductance,
            capacity=flow.step.capacity, stored_pressure=flow.step.pressure,
            epi_weight=flow.epi_weight,
        )
        flows.append({
            "segment": asdict(flow.segment), "parent_interval_index": flow.parent_interval_index,
            "before_snapshot": asdict(flow.before_snapshot), "step": asdict(flow.step),
            "pressure_readout": asdict(readout),
        })
    terminal = execution.prefix_budget.prefixes[-1]
    if len(flows) != 8 or len(ordinary["flows"]) != 8:
        raise RuntimeError("matched comparison changed its declared eight-segment horizon")
    pressure_differences = tuple(
        tuple(Fraction(carried) - reference for carried, reference
              in zip(flow.step.pressure, old["left"]["exact_delta_nfr"], strict=True))
        for flow, old in zip(execution.flows, ordinary["flows"], strict=True)
    )
    visible_gap = terminal.mean_visible_change - ordinary["visible_mean_change"]
    source_gap = terminal.mean_nodal_area - ordinary["mean_area"]
    executor_gap = terminal.mean_rounding_defect - ordinary["executor_mean_defect"]
    if visible_gap != source_gap + executor_gap:
        raise RuntimeError("branch comparison lost its exact source/executor decomposition")
    event_records = tuple({
        "event": event.event.as_record(), "before_binding": asdict(event.before_binding.state),
        "after_binding": asdict(event.after_binding.state),
        "exact_epi_jump": tuple(y - x for x, y in zip(
            event.before_binding.state.exact_epi, event.after_binding.state.exact_epi, strict=True,
        )),
    } for event in execution.events)
    return {
        "mode": mode, "epsilon": Fraction(epsilon), "initial_capture": asdict(initial),
        "schedule": asdict(schedule), "physical_partitions": tuple(map(asdict, partitions)),
        "carried": {
            "flows": flows, "events": event_records, "prefix_budget": asdict(execution.prefix_budget),
            "mean_area": terminal.mean_nodal_area, "visible_mean_change": terminal.mean_visible_change,
            "reconstructed_mean_change": terminal.mean_reconstructed_change,
            "executor_mean_defect": terminal.mean_rounding_defect,
            "runtime_provenance_certified": execution.runtime_provenance_certified,
            "final_state": asdict(execution.final_binding.state),
        },
        "ordinary": ordinary,
        "comparison": {
            "pressure_differences": pressure_differences,
            "any_pressure_difference": any(any(row) for row in pressure_differences),
            "visible_mean_gap": visible_gap, "source_mean_gap": source_gap,
            "executor_mean_gap": executor_gap, "identity_residual": visible_gap - source_gap - executor_gap,
        },
        "final_time": graph.graph["_t"],
        "scope": "Matched finite live executions; each branch generates its own visible-state pressure",
    }


def run_c6_remainder_runtime_comparison():
    return {
        "cases": [run_c6_remainder_runtime_case(*case) for case in CASES],
        "runtime_executed": True, "case_count": 3, "cycle_count_per_branch": 2,
        "physical_segments_per_branch": 8, "pressure_refreshed_on_live_epi": True,
        "production_default_integrator_modified": False, "future_stability_certified": False,
        "empirical_correspondence_tested": False,
        "scope": (
            "Fixed inherited preparations and canonical factors, now using an explicit "
            "pressure refresh mesh in both branches. Each source budget is measured on "
            "its own live state. Finite carry/event integration does not prove generated "
            "pressure summability, an invariant full-runtime domain or future behavior."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/c6_winding_remainder_runtime.json")
    args = parser.parse_args()
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = run_c6_remainder_runtime_comparison()
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance:
        raise RuntimeError("source changed during the finite runtime comparison")
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-live-nodal-remainder-event-comparison", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__, "networkx": nx.__version__},
        graph_construction="Inherited prepared C6 null/k1/k3, unchanged triad and unit conductance",
        capacity_specification="Initial unit capacity, subsequent canonical UM/IL/SHA",
        solver="Matched eight physical 1/16 segments with canonical visible-pressure refresh",
        seed=17, timestep=.0625, operator_sequence=WORD, result_status=ClaimStatus.MEASURED,
        telemetry=("sealed finite carried state", "actual nodal source and executor mean budgets",
                   "EPI-preserving event carry", "exact pressure-readout decomposition"),
        controls=("independently refreshed ordinary DefaultIntegrator branch",),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report.update(manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote finite C6 carried runtime comparison to {args.output}")


if __name__ == "__main__":
    main()
