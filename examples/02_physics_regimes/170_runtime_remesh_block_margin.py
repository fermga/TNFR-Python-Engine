"""Measure exact normalized margins on two causal finite REMESH blocks.

The first two-cycle block has the exact finite lower drop fraction
``kappa=139/256``.  A separate lag-one ``alpha=1`` block uses zero-duration
represented identity schedules and has ``kappa=0``.  These observations do not
provide a uniform margin over a forward-invariant class or control energy at
prefixes inside a repeated block.
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
from tnfr.physics import observe_executed_event_remesh_block_margin


def _graph(
    *,
    alpha: float,
    nu_f: float,
    initialize_pressure: bool,
) -> nx.Graph:
    """Build a deterministic P2 graph with the canonical pure-EPI pressure."""

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
        compute_delta_nfr=default_compute_delta_nfr,
    )
    for node, epi in enumerate((2.0, 0.0)):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=nu_f,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.graph["_epi_hist"] = deque([{0: 0.0, 1: 2.0}], maxlen=64)
    if initialize_pressure:
        graph.graph["DNFR_WEIGHTS"] = {
            "phase": 0.0,
            "epi": 1.0,
            "vf": 0.0,
            "topo": 0.0,
        }
        default_compute_delta_nfr(graph)
    return graph


def _positive_margin_specs() -> tuple[EventRemeshCycleExecutionSpec, ...]:
    specs: list[EventRemeshCycleExecutionSpec] = []
    start = 0.0
    for _index in range(2):
        schedule = build_operator_event_schedule(
            (),
            start_time=start,
            flow_durations=(0.125,),
        )
        specs.append(EventRemeshCycleExecutionSpec(schedule))
        start = schedule.end_time
    return tuple(specs)


def _zero_margin_specs() -> tuple[EventRemeshCycleExecutionSpec, ...]:
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


def _execute(*, alpha: float, nu_f: float, identity_schedules: bool):
    graph = _graph(
        alpha=alpha,
        nu_f=nu_f,
        initialize_pressure=not identity_schedules,
    )
    specs = _zero_margin_specs() if identity_schedules else _positive_margin_specs()
    return execute_event_remesh_cycle_sequence(
        graph,
        specs,
        metric_weights=(1.0, 1.0),
    )


def run_protocol() -> dict[str, Any]:
    """Execute and observe the positive and zero finite-margin blocks."""

    positive_execution = _execute(
        alpha=0.5,
        nu_f=1.0,
        identity_schedules=False,
    )
    zero_execution = _execute(
        alpha=1.0,
        nu_f=0.8,
        identity_schedules=True,
    )
    return {
        "positive_execution": positive_execution,
        "positive_margin": observe_executed_event_remesh_block_margin(
            positive_execution
        ),
        "zero_execution": zero_execution,
        "zero_margin": observe_executed_event_remesh_block_margin(
            zero_execution
        ),
    }


def _fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def _margin_report(observation: Any) -> dict[str, Any]:
    return {
        "block_observation_certified": observation.block_observation_certified,
        "exact_energy_before": _fraction_text(
            observation.exact_augmented_energy_before
        ),
        "exact_energy_after": _fraction_text(
            observation.exact_augmented_energy_after
        ),
        "exact_lower_bound": _fraction_text(
            observation.exact_gain_based_energy_drop_lower_bound
        ),
        "exact_gain_based_energy_drop_fraction_lower_bound": _fraction_text(
            observation.exact_gain_based_energy_drop_fraction_lower_bound
        ),
        "exact_endpoint_energy_gain_upper_bound": _fraction_text(
            observation.exact_endpoint_energy_gain_upper_bound
        ),
        "positive_normalized_block_margin_certified": (
            observation.positive_normalized_block_margin_certified
        ),
        "strict_energy_contraction_observed": (
            observation.strict_energy_contraction_observed
        ),
    }


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible statement of evidence and open scope."""

    positive = protocol["positive_margin"]
    zero = protocol["zero_margin"]
    return {
        "claim": "exact margins for two causally executed finite blocks",
        "positive_block": _margin_report(positive),
        "alpha_one_boundary": _margin_report(zero),
        "scope": {
            "absolute_uniform_margin": False,
            "uniform_normalized_forward_invariant_class_margin": (
                positive.uniform_class_coercivity_certified
            ),
            "intrablock_prefix_bound": False,
            "repeated_runtime_stability": (
                positive.repeated_runtime_stability_certified
            ),
            "future_stability": positive.future_stability_certified,
            "runtime_global_gain": positive.runtime_global_gain_certified,
            "solver_accuracy": positive.solver_accuracy_certified,
            "solver_order": positive.solver_order_certified,
            "mesh_convergence": positive.mesh_convergence_certified,
            "full_tnfr_stability": positive.full_tnfr_stability_certified,
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
