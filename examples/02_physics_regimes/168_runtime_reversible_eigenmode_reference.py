"""Bind executed binary64 P3 partitions to one exact modal reference.

Three independently executed, explicitly pressure-refreshed Euler partitions
start from the same nonuniform mode of the nonregular path ``P3``.  The
observer rationalizes every captured binary64 value, derives the exact
reversible reference, and separates the pressure-realization defect ``rho``,
the held-input execution defect ``eta`` and their local combination
``epsilon = h*diag(nu_f)*rho + eta``.  It propagates ``epsilon`` through the
complete Euler matrices because represented defects need not remain modal.

This is a finite offline comparison of individually executor-certified runs.
It does not prove binary64 or runtime mesh convergence, solver accuracy or
order, common causal provenance, glyph or REMESH dynamics, repeated behavior,
or future TNFR stability.
"""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

import networkx as nx

from tnfr.operators.event_runtime import execute_operator_event_schedule
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)
from tnfr.physics import (
    ExecutedReversibleSingleEigenmodeEulerReferenceObservation,
    observe_executed_reversible_single_eigenmode_euler_reference,
)


F = Fraction
INITIAL_EPI = (1.0e16, 1.0, -9_999_999_999_999_998.0)
PARTITIONS = (
    (0.25, 0.25),
    (0.125,) * 4,
    (0.0625,) * 8,
)


def _execute_partition(durations: tuple[float, ...]):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1.0, "weight")
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        _gamma_spec={"type": "none"},
        EPI_MIN=-1.0e300,
        EPI_MAX=1.0e300,
        DNFR_WEIGHTS={
            "phase": 0.0,
            "epi": 1.0,
            "vf": 0.0,
            "topo": 0.0,
        },
    )
    for node, epi in zip(graph, INITIAL_EPI, strict=True):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="runtime-reference-example",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )

    exact_total = sum((F.from_float(value) for value in durations), F(0))
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(float(exact_total),),
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0],
        durations,
    )
    result = execute_operator_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
        suppress_birth_warnings=True,
    )
    return result.physical_flow_partition_evidence[0]


def run_protocol(
) -> ExecutedReversibleSingleEigenmodeEulerReferenceObservation:
    """Execute and bind the finite two-, four- and eight-segment family."""

    executions = tuple(_execute_partition(row) for row in PARTITIONS)
    return observe_executed_reversible_single_eigenmode_euler_reference(
        executions
    )


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def build_report(
    observation: ExecutedReversibleSingleEigenmodeEulerReferenceObservation,
) -> dict[str, Any]:
    """Return a compact JSON-compatible report with explicit false scope."""

    reference = observation.reference_certificate
    return {
        "claim": "finite executed reversible single-eigenmode binding on P3",
        "runtime_reference_binding_certified": (
            observation.runtime_reference_binding_certified
        ),
        "nodes": list(observation.nodes),
        "mu": _fraction_text(reference.exact_mode_eigenvalue),
        "reversible_metric": [
            _fraction_text(value)
            for value in reference.exact_reversible_metric
        ],
        "partitions": [
            {
                "segment_count": len(row.exact_segment_durations),
                "binding_certified": row.runtime_partition_binding_certified,
                "nonzero_rho": any(
                    value != 0
                    for residual in row.exact_pressure_realization_residuals
                    for value in residual
                ),
                "nonzero_eta": any(
                    value != 0
                    for residual in row.exact_held_input_execution_residuals
                    for value in residual
                ),
                "nonzero_epsilon": any(
                    value != 0
                    for defect in row.exact_local_runtime_defects
                    for value in defect
                ),
                "endpoint_runtime_defect_linf": _fraction_text(
                    row.exact_endpoint_runtime_defect_linf
                ),
                "runtime_continuous_linf_error_interval": [
                    _fraction_text(
                        row.exact_runtime_continuous_linf_error_lower_bound
                    ),
                    _fraction_text(
                        row.exact_runtime_continuous_linf_error_upper_bound
                    ),
                ],
            }
            for row in observation.partition_observations
        ],
        "scope": {
            "binary64_asymptotic_convergence": (
                observation.binary64_asymptotic_convergence_certified
            ),
            "runtime_mesh_convergence": (
                observation.runtime_mesh_convergence_certified
            ),
            "solver_accuracy": observation.solver_accuracy_certified,
            "solver_order": observation.solver_order_certified,
            "common_causal_execution_provenance": (
                observation.common_causal_execution_provenance_certified
            ),
            "glyph_or_remesh_dynamics": (
                observation.glyph_or_remesh_dynamics_certified
            ),
            "future_or_repeated_stability": (
                observation.future_or_repeated_stability_certified
            ),
            "full_tnfr_stability": observation.full_tnfr_stability_certified,
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
