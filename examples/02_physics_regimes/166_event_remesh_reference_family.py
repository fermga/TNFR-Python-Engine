"""Certify one finite P2 three-mesh event/REMESH reference family.

The graph has two nodes, homogeneous capacity and the nonuniform initial mode
``(1, -1)``.  Three pressure-refreshed Euler executions split the same physical
duration into 2, 4 and 8 segments.  The exact P2 certificate compares their
Euler factors with a rational enclosure of ``exp(-1)``, proves the quadratic
finite-mesh bound, and carries that error through the unit-delay REMESH map.

This is a compatible finite reference problem.  It does not promote generic
mesh convergence, solver order, repeated runtime stability or future behavior.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
import json
from typing import Any

import networkx as nx

from tnfr.operators.event_remesh_runtime import execute_event_remesh_cycle
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)
from tnfr.physics.event_remesh_reference import (
    observe_p2_event_remesh_reference_family,
)


MESH_DURATIONS = (
    (0.25, 0.25),
    (0.125,) * 4,
    (0.0625,) * 8,
)


def _refresh_p2_pressure(graph: nx.Graph) -> None:
    left = float(graph.nodes[0]["EPI"])
    right = float(graph.nodes[1]["EPI"])
    graph.nodes[0]["delta_nfr"] = right - left
    graph.nodes[1]["delta_nfr"] = left - right


def _graph() -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        RANDOM_SEED=23,
        GAMMA={"type": "none"},
        _gamma_spec={"type": "none"},
        DNFR_WEIGHTS={
            "phase": 0.0,
            "epi": 1.0,
            "vf": 0.0,
            "topo": 0.0,
        },
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=0.5,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        CLIP_MODE="hard",
        compute_delta_nfr=_refresh_p2_pressure,
    )
    for node, epi in enumerate((1.0, -1.0)):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="reference",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.graph["_epi_hist"] = deque(
        [{0: 1.0, 1: -1.0}],
        maxlen=64,
    )
    _refresh_p2_pressure(graph)
    return graph


def _cycle(durations: tuple[float, ...]):
    graph = _graph()
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.5,),
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0],
        durations,
    )
    return execute_event_remesh_cycle(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
        include_stage_certificates=True,
        suppress_birth_warnings=True,
    )


def run_protocol() -> dict[str, Any]:
    """Execute the three finite meshes and build the exact reference record."""

    cycles = tuple(_cycle(durations) for durations in MESH_DURATIONS)
    reference = observe_p2_event_remesh_reference_family(*cycles)
    return {"cycles": cycles, "reference": reference}


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-compatible report of the certified identities."""

    reference = protocol["reference"]
    certified = reference.reference_family_certified
    return {
        "claim": "finite exact P2 event/REMESH reference family",
        "reference_family_certified": certified,
        "model": {
            "initial_field": [
                _fraction_text(value) for value in reference.exact_initial_field
            ],
            "mean": _fraction_text(reference.exact_mean),
            "amplitude": _fraction_text(reference.exact_initial_amplitude),
            "nu_f": _fraction_text(reference.exact_nu_f),
            "lambda": _fraction_text(reference.exact_lambda),
            "duration": _fraction_text(reference.exact_total_duration),
            "alpha": _fraction_text(reference.exact_alpha),
            "beta": _fraction_text(reference.exact_beta),
            "continuous_factor_interval": [
                float(reference.exact_continuous_factor_lower_bound),
                float(reference.exact_continuous_factor_upper_bound),
            ],
        },
        "meshes": [
            {
                "name": mesh.mesh_name,
                "segment_count": len(mesh.exact_segment_durations),
                "euler_factor": _fraction_text(mesh.exact_euler_factor),
                "factor_error_interval": [
                    float(mesh.exact_factor_error_lower_bound),
                    float(mesh.exact_factor_error_upper_bound),
                ],
                "quadratic_error_bound": _fraction_text(
                    mesh.exact_quadratic_factor_error_upper_bound
                ),
                "hmax_error_bound": _fraction_text(
                    mesh.exact_hmax_factor_error_upper_bound
                ),
                "beta_scaled_post_error_upper_bound": float(
                    mesh.exact_ideal_post_remesh_error_upper_bound
                ),
                "runtime_residual_linf": _fraction_text(
                    mesh.exact_total_residual_linf
                ),
                "runtime_post_error_upper_bound": float(
                    mesh.exact_runtime_post_remesh_error_upper_bound
                ),
            }
            for mesh in reference.meshes
        ],
        "strict_proper_subdivision_improvement": (
            reference.strict_pre_remesh_error_improvement
        ),
        "scope": {
            "compatible_finite_p2_problem": certified,
            "binary64_asymptotic_convergence": (
                reference.binary64_asymptotic_convergence_certified
            ),
            "arbitrary_glyph_or_mixed_mode": (
                reference.arbitrary_glyph_or_mixed_mode_certified
            ),
            "soft_clipping": reference.soft_clipping_certified,
            "changing_support_or_metric": (
                reference.changing_support_or_metric_certified
            ),
            "generic_mesh_convergence": (
                reference.generic_mesh_convergence_certified
            ),
            "solver_order": reference.solver_order_certified,
            "repeated_runtime_stability": (
                reference.repeated_runtime_stability_certified
            ),
            "future_stability": reference.future_stability_certified,
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
