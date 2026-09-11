"""Verify a robust REMESH/schedule envelope on finite causal executions.

The dyadic three-cycle witness has zero represented REMESH head defect and
spans one complete universal block.  A second witness retains a positive
binary64 head-energy defect and selects its exact minimum relative budget.
Both observations are finite; neither proves that the budget is invariant on
future runtime states.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
import json
from typing import Any

import networkx as nx

from tnfr.operators import (
    EventRemeshCycleExecutionSpec,
    build_operator_event_schedule,
    execute_event_remesh_cycle_sequence,
)
from tnfr.physics import (
    certify_uniform_remesh_history_stability,
    certify_uniform_remesh_schedule_policy_stability,
    certify_uniform_remesh_schedule_relative_defect_stability,
    observe_executed_event_remesh_relative_defect_block,
)


def _set_pure_epi_pressure(graph: nx.Graph) -> None:
    values = {node: float(graph.nodes[node]["EPI"]) for node in graph}
    for node in graph:
        neighbours = tuple(graph.neighbors(node))
        graph.nodes[node]["delta_nfr"] = (
            sum(values[item] for item in neighbours) / len(neighbours)
            - values[node]
        )


def _graph(
    *,
    current: tuple[float, float],
    past: tuple[float, float],
    alpha: float,
) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=23,
        _gamma_spec={"type": "none"},
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=alpha,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        CLIP_MODE="hard",
        DT_MIN=0.0,
        compute_delta_nfr=_set_pure_epi_pressure,
    )
    for node, epi in enumerate(current):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.graph["_epi_hist"] = deque(
        [{node: value for node, value in enumerate(past)}],
        maxlen=64,
    )
    _set_pure_epi_pressure(graph)
    return graph


def _specs(count: int) -> tuple[EventRemeshCycleExecutionSpec, ...]:
    result: list[EventRemeshCycleExecutionSpec] = []
    start = 0.0
    for _index in range(count):
        schedule = build_operator_event_schedule(
            (),
            start_time=start,
            flow_durations=(0.125,),
        )
        result.append(EventRemeshCycleExecutionSpec(schedule))
        start = schedule.end_time
    return tuple(result)


def _execute(
    *,
    count: int,
    current: tuple[float, float],
    past: tuple[float, float],
    alpha: float,
):
    return execute_event_remesh_cycle_sequence(
        _graph(current=current, past=past, alpha=alpha),
        _specs(count),
        metric_weights=(1.0, 1.0),
    )


def _certificate(*, alpha: float | Fraction, eta: Fraction):
    remesh = certify_uniform_remesh_history_stability(
        alpha=alpha,
        tau_local=1,
        tau_global=1,
    )
    policy = certify_uniform_remesh_schedule_policy_stability(
        remesh,
        Fraction(9, 16),
    )
    return certify_uniform_remesh_schedule_relative_defect_stability(
        policy,
        eta,
    )


def _minimum_eta(execution: Any) -> Fraction:
    boundary = execution.runtime_telescope.boundaries[0]
    transition = boundary.exact_transition
    jensen = sum(
        (
            coefficient * transition.exact_history_energies[delay]
            for delay, coefficient in (
                transition.certificate.combined_delay_coefficients
            )
        ),
        Fraction(0),
    )
    defect = (
        boundary.schedule_balance.exact_runtime_bounded_head_energy
        - transition.exact_next_energy
    )
    if jensen <= 0 or defect <= 0:
        raise RuntimeError("positive-defect witness did not materialize")
    return defect / jensen


def run_protocol() -> dict[str, Any]:
    """Build exact-zero and positive-defect finite causal witnesses."""

    dyadic_execution = _execute(
        count=3,
        current=(2.0, 0.0),
        past=(0.0, 2.0),
        alpha=0.5,
    )
    dyadic_certificate = _certificate(alpha=Fraction(1, 2), eta=Fraction(0))
    positive_execution = _execute(
        count=2,
        current=(-2.0, -1.0),
        past=(-2.0, -1.0),
        alpha=0.4,
    )
    minimum_eta = _minimum_eta(positive_execution)
    positive_certificate = _certificate(alpha=0.4, eta=minimum_eta)
    return {
        "dyadic_certificate": dyadic_certificate,
        "dyadic_observation": (
            observe_executed_event_remesh_relative_defect_block(
                dyadic_execution,
                dyadic_certificate,
            )
        ),
        "positive_certificate": positive_certificate,
        "positive_observation": (
            observe_executed_event_remesh_relative_defect_block(
                positive_execution,
                positive_certificate,
            )
        ),
    }


def _fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def _row(certificate: Any, observation: Any) -> dict[str, Any]:
    return {
        "certificate_valid": (
            certificate.relative_defect_stability_certificate_certified
        ),
        "finite_observation_valid": (
            observation.relative_defect_block_observation_certified
        ),
        "q": _fraction_text(certificate.schedule_energy_gain_upper_bound),
        "eta": _fraction_text(
            certificate.pre_schedule_relative_energy_defect_upper_bound
        ),
        "q_eff": _fraction_text(
            certificate.exact_effective_head_energy_gain_upper_bound
        ),
        "boundary_count": observation.boundary_count,
        "defects": [
            _fraction_text(value)
            for value in observation.exact_pre_schedule_energy_defects
        ],
        "relative_defect_ratios": [
            _fraction_text(value)
            for value in observation.exact_relative_energy_defect_ratios
        ],
        "endpoint_gain_upper_bound": _fraction_text(
            observation.exact_finite_endpoint_energy_gain_upper_bound
        ),
        "finite_endpoint_bound": (
            observation.exact_finite_endpoint_bound_certified
        ),
    }


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return the finite theorem binding and its explicit open boundary."""

    dyadic = protocol["dyadic_observation"]
    positive = protocol["positive_observation"]
    return {
        "claim": "finite causal verification of a relative-defect envelope",
        "zero_defect_complete_block": _row(
            protocol["dyadic_certificate"],
            dyadic,
        ),
        "positive_binary64_defect": _row(
            protocol["positive_certificate"],
            positive,
        ),
        "scope": {
            "same_execution_provenance": (
                dyadic.same_graph_execution_provenance_certified
            ),
            "runtime_forward_invariant_class": (
                dyadic.runtime_forward_invariant_class_certified
            ),
            "repeated_binary64_stability": (
                dyadic.repeated_binary64_runtime_stability_certified
            ),
            "future_runtime_stability": (
                dyadic.future_runtime_stability_certified
            ),
            "solver_accuracy": dyadic.solver_accuracy_certified,
            "full_tnfr_stability": dyadic.full_tnfr_stability_certified,
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
