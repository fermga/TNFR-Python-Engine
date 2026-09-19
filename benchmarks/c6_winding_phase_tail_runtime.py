"""Finite graph-owned passage from the original C6 preparation to its phase tail.

The carried numerical solver is explicit: four refreshed h=1/16 steps
follow every UM/IL pair. A single terminal SHA closes the complete word
after all unit-capacity measurements. No detached carry is imported.
"""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.c6_winding_phase_response import prepare_c6_phase_response  # noqa: E402
from tnfr.constants.aliases import ALIAS_THETA, ALIAS_VF  # noqa: E402
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)  # noqa: E402
from tnfr.operators.nodal_remainder_runtime import (
    execute_nodal_remainder_event_schedule,
)  # noqa: E402
from tnfr.physics.c6_carried_closure import derive_c6_carried_closure  # noqa: E402
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile  # noqa: E402
from tnfr.physics.c6_phase_orbit import (
    observe_c6_coupling_coherence_phase_step,
)  # noqa: E402
from tnfr.physics.c6_pressure_lattice import (  # noqa: E402
    derive_c6_pressure_lattice,
    _observe_rebuilt_c6_pressure_lattice,
)

TAIL_ENTRY_CYCLE = 89
SEGMENT_DURATION = 0.0625
INTERVAL_DURATION = 0.25
EPI_LOWER, EPI_UPPER = 0.375, 0.625


def _signature(values):
    return tuple(value.hex() for value in values)


def _declared_schedule(cycles):
    if type(cycles) is not int or not 1 <= cycles <= TAIL_ENTRY_CYCLE:
        raise ValueError("cycles must be an integer in the declared finite range 1..89")
    names = ("coupling", "coherence") * cycles + ("silence",)
    durations = tuple(
        INTERVAL_DURATION if 0 < i <= 2 * cycles and i % 2 == 0 else 0.0
        for i in range(len(names) + 1)
    )
    schedule = build_operator_event_schedule(
        names, start_time=0.0, flow_durations=durations
    )
    partitions = tuple(
        build_physical_flow_partition(interval, (SEGMENT_DURATION,) * 4)
        for interval in schedule.intervals
        if interval.duration > 0
    )
    return schedule, partitions


def _event_record(event):
    return {
        "operator_name": event.event.operator_name,
        "time": event.event.event_time,
        "phase_before": event.phase_before,
        "phase_after": event.phase_after,
        "capacity_before": event.before_snapshot.nu_f,
        "capacity_after": event.after_snapshot.nu_f,
        "state_before": asdict(event.before_binding.state),
        "state_after": asdict(event.after_binding.state),
        "binding_preserved": event.before_binding is event.after_binding,
        "stage_schedule": event.stage.schedule,
        "target_count": event.stage.nodes_processed,
    }


def run_c6_phase_tail_runtime_bridge(*, cycles=TAIL_ENTRY_CYCLE):
    """Execute and bind one finite original-preparation word through existing owners.

    Prefix counts below 89 exercise the same finite contract but do not
    claim that the phase tail was reached. At 89, the final actual phase
    is separately replayed through the shared phase proposal kernel to
    verify its conditional fixed-point identity. The full runtime word
    ends with SHA, which changes capacity; unit capacity is claimed only
    for its preceding measured flows, never for a subsequent invocation.
    """
    schedule, partitions = _declared_schedule(cycles)
    graph = prepare_c6_phase_response("null", 0.0)
    initial_phase = tuple(graph.nodes[node][ALIAS_THETA[0]] for node in graph)
    initial_capacity = tuple(graph.nodes[node][ALIAS_VF[0]] for node in graph)
    initial_support = tuple(graph.edges)
    result = execute_nodal_remainder_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=partitions,
        epi_lower=EPI_LOWER,
        epi_upper=EPI_UPPER,
    )
    if (
        not result.runtime_provenance_certified
        or len(result.events) != 2 * cycles + 1
        or len(result.flows) != 4 * cycles
        or result.initial_binding.state.epi != (0.5,) * 6
        or any(result.initial_binding.state.remainder)
        or initial_capacity != (1.0,) * 6
        or tuple(graph.edges) != initial_support
        or any(result.exact_nodal_balance_residual)
    ):
        raise RuntimeError(
            "the live preparation bridge lost its executor-owned finite contract"
        )
    rows = []
    current_phase = initial_phase
    for index in range(cycles):
        um, il = result.events[2 * index : 2 * index + 2]
        expected = observe_c6_coupling_coherence_phase_step(phase=current_phase)
        if (
            um.event.operator_name != "coupling"
            or il.event.operator_name != "coherence"
            or _signature(um.phase_before) != _signature(current_phase)
            or _signature(um.phase_after) != _signature(expected.phase_after_coupling)
            or _signature(il.phase_before) != _signature(um.phase_after)
            or _signature(il.phase_after) != _signature(expected.phase_after_coherence)
            or um.before_binding is not um.after_binding
            or il.before_binding is not il.after_binding
            or any(
                item.stage.schedule != "two_phase_jacobi"
                or item.stage.nodes_processed != 6
                for item in (um, il)
            )
            or any(
                item.before_snapshot.nu_f != (1.0,) * 6
                or item.after_snapshot.nu_f != (1.0,) * 6
                for item in (um, il)
            )
        ):
            raise RuntimeError(
                "a live UM/IL phase or carry observation differs from the shared phase map"
            )
        flows = result.flows[4 * index : 4 * index + 4]
        weights = dict(flows[0].normalized_weights)
        lattice = derive_c6_pressure_lattice(
            phase=il.phase_after,
            epi_weight=float(weights["epi"]),
            phase_weight=float(weights["phase"]),
            epi_lower=EPI_LOWER,
            epi_upper=EPI_UPPER,
        )
        records = []
        for flow in flows:
            pressure = _observe_rebuilt_c6_pressure_lattice(
                lattice, flow.step.before.epi
            )
            if (
                flow.step.capacity != (1.0,) * 6
                or flow.step.timestep != SEGMENT_DURATION
                or _signature(flow.phase_before) != _signature(il.phase_after)
                or _signature(flow.phase_after) != _signature(il.phase_after)
                or dict(flow.normalized_weights) != weights
                or flow.step.pressure != pressure.pressure
                or not all(value for _name, value in flow.pressure_refresh_checks)
            ):
                raise RuntimeError(
                    "the live flow differs from its actual phase-bound canonical pressure"
                )
            records.append(
                {
                    "step": asdict(flow.step),
                    "phase_before": flow.phase_before,
                    "phase_after": flow.phase_after,
                    "start_time": flow.segment.start_time,
                    "end_time": flow.segment.end_time,
                    "fresh_pressure_matches": True,
                }
            )
        rows.append(
            {
                "ordinal": index + 1,
                "coupling": _event_record(um),
                "coherence": _event_record(il),
                "phase_projection": asdict(expected),
                "flows": tuple(records),
            }
        )
        current_phase = il.phase_after
    terminal = result.events[-1]
    actual_phase = tuple(graph.nodes[node][ALIAS_THETA[0]] for node in graph)
    if (
        terminal.event.operator_name != "silence"
        or terminal.before_binding is not terminal.after_binding
        or _signature(terminal.phase_before) != _signature(current_phase)
        or _signature(terminal.phase_after) != _signature(current_phase)
        or _signature(actual_phase) != _signature(current_phase)
        or terminal.before_snapshot.nu_f != (1.0,) * 6
        or not all(0.0 <= value < 1.0 for value in terminal.after_snapshot.nu_f)
    ):
        raise RuntimeError(
            "the terminal SHA boundary lost its phase/carry preservation or capacity change"
        )
    following_phase_step = observe_c6_coupling_coherence_phase_step(phase=current_phase)
    fixed = _signature(following_phase_step.phase_after_coherence) == _signature(
        current_phase
    )
    if cycles == TAIL_ENTRY_CYCLE and not fixed:
        raise RuntimeError(
            "the declared 89-cycle live phase endpoint did not enter its fixed projection"
        )
    tail_entry = result.events[-2].after_binding.state
    endpoint = terminal.before_binding.state
    profile = derive_c6_carried_profile(lattice)
    closure = derive_c6_carried_closure(
        profile, state=tail_entry, timestep=SEGMENT_DURATION
    )
    closure_record = asdict(closure)
    closure_record.pop("base_tube")
    return {
        "source": {
            "preparation": "canonical null C6 winding",
            "initial_phase": initial_phase,
            "initial_state": asdict(result.initial_binding.state),
            "initial_capacity": initial_capacity,
            "ordered_nodes": result.nodes,
            "support": initial_support,
            "normalized_weights": result.flows[0].normalized_weights,
            "random_seed": 17,
            "declared_carried_band": (EPI_LOWER, EPI_UPPER),
            "carry_imported_or_reset": False,
        },
        "cycle_count": cycles,
        "event_count": len(result.events),
        "flow_count": len(result.flows),
        "schedule": asdict(schedule),
        "cycles": tuple(rows),
        "terminal_silence": _event_record(terminal),
        "runtime_provenance_certified_at_capture": result.runtime_provenance_certified,
        "phase_observations_retained_in_executor_seal": True,
        "phase_projection_fixed_at_endpoint": fixed,
        "declared_phase_tail_reached": cycles == TAIL_ENTRY_CYCLE and fixed,
        "tail_entry_after_last_IL": asdict(tail_entry),
        "endpoint_before_SHA": asdict(endpoint),
        "tail_entry_time": result.events[-2].after_binding.time,
        "endpoint_time": terminal.before_binding.time,
        "endpoint_phase": current_phase,
        "tail_entry_closure": closure_record,
        "tail_entry_profile": asdict(profile.forced_balance),
        "exact_nodal_area": result.exact_nodal_area,
        "exact_reconstructed_change": result.exact_reconstructed_change,
        "exact_nodal_balance_residual": result.exact_nodal_balance_residual,
        "mean_area": sum(result.exact_nodal_area, F(0)) / 6,
        "all_measured_flows_have_unit_capacity": True,
        "unit_capacity_after_terminal_SHA": False,
        "historical_detached_endpoint_reachability_certified": False,
        "future_runtime_certified": False,
        "indefinite_trapping_certified": False,
        "scope": (
            "One admitted finite live word from the original null preparation, with a newly declared carried "
            "and pressure-refreshed solver. The historical detached B27-B40 endpoint is not retroactively "
            "made reachable by this distinct causal history. Serialized observations retain measured "
            "evidence; they do not recreate the live executor seal. SHA changes capacity after measurement."
        ),
    }
