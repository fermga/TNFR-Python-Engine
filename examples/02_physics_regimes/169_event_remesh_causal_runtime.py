"""Execute one finite causal event/REMESH cycle sequence.

Two distinct zero-duration Transition schedules provide represented identity maps
while lag-one REMESH with ``alpha=1`` advances the delayed history.  The outer
executor binds both cycles to one graph and one transaction.  The resulting
trajectory alternates, so finite causal provenance does not imply repeated or
future stability.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
import json
from typing import Any

import networkx as nx

from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.operators import (
    EventRemeshCycleExecutionSpec,
    build_operator_event_schedule,
    execute_event_remesh_cycle_sequence,
)


def causal_remesh_graph() -> nx.Graph:
    """Build the deterministic two-node lag-one witness."""

    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=23,
        _gamma_spec={"type": "none"},
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=1.0,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        CLIP_MODE="hard",
        DT_MIN=0.0,
        compute_delta_nfr=default_compute_delta_nfr,
    )
    for node, epi in enumerate((2.0, 0.0)):
        graph.nodes[node].update(
            EPI=epi,
            # Exactly 0.8 keeps both nodes in NAV's active regime, so the
            # zero-duration schedule leaves EPI and the reversible metric fixed.
            nu_f=0.8,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.graph["_epi_hist"] = deque([{0: 0.0, 1: 2.0}], maxlen=64)
    return graph


def _cycle_specs() -> tuple[EventRemeshCycleExecutionSpec, ...]:
    """Declare two distinct schedules forming one exact clock chain."""

    first = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.0, 0.0),
    )
    second = build_operator_event_schedule(
        ("transition",),
        start_time=first.end_time,
        flow_durations=(0.0, 0.0),
    )
    return (
        EventRemeshCycleExecutionSpec(first),
        EventRemeshCycleExecutionSpec(second),
    )


def run_protocol():
    """Execute the finite causal sequence through the public facade."""

    graph = causal_remesh_graph()
    return execute_event_remesh_cycle_sequence(
        graph,
        _cycle_specs(),
        metric_weights=(1.0, 1.0),
    )


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def build_report(result: Any) -> dict[str, Any]:
    """Return a compact JSON-compatible statement of the certified scope."""

    trajectory = [result.cycles[0].pre_schedule_epi.epi_values]
    trajectory.extend(cycle.post_remesh_epi.epi_values for cycle in result.cycles)
    return {
        "claim": "one finite graph-owned causal event/REMESH sequence",
        "cycle_indices": list(result.cycle_indices),
        "epi_trajectory": [list(row) for row in trajectory],
        "causal_cycle_order_certified": result.causal_cycle_order_certified,
        "same_graph_execution_provenance_certified": (
            result.same_graph_execution_provenance_certified
        ),
        "whole_sequence_graph_state_atomic": (
            result.whole_sequence_graph_state_atomic
        ),
        "exact_recorded_boundary_continuity_certified": (
            result.exact_recorded_boundary_continuity_certified
        ),
        "exact_finite_energy_telescope_certified": (
            result.exact_finite_energy_telescope_certified
        ),
        "exact_total_energy_drop": _fraction_text(
            result.runtime_telescope.exact_total_energy_drop
        ),
        "receipts": [
            {
                "cycle_index": receipt.cycle_index,
                "binding_certified": receipt.receipt_binding_certified,
                "spec_schedule_identity": receipt.spec.schedule is receipt.schedule,
                "executed_schedule_identity": (
                    receipt.cycle_result.event_execution.schedule
                    is receipt.schedule
                ),
            }
            for receipt in result.receipts
        ],
        "nested_offline_scope": {
            "cycle_sequence_shared_graph_provenance": (
                result.observed_sequence
                .shared_graph_execution_provenance_certified
            ),
            "runtime_telescope_shared_graph_provenance": (
                result.runtime_telescope
                .shared_graph_execution_provenance_certified
            ),
            "runtime_telescope_whole_sequence_atomicity": (
                result.runtime_telescope.whole_sequence_atomicity_certified
            ),
        },
        "scope": {
            "runtime_global_gain": result.runtime_global_gain_certified,
            "uniform_repeated_margin": False,
            "repeated_runtime_stability": (
                result.repeated_runtime_stability_certified
            ),
            "future_stability": result.future_stability_certified,
            "solver_accuracy": result.solver_accuracy_certified,
            "solver_order": result.solver_order_certified,
            "mesh_convergence": result.mesh_convergence_certified,
            "full_tnfr_stability": result.full_tnfr_stability_certified,
            "external_side_effects_rolled_back": (
                result.external_side_effects_rolled_back
            ),
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
