"""Certify finite causal P2 Reception/REMESH disagreement extinction.

Two graph-owned ``EN -> IL -> REMESH`` cycles are executed under one outer
transaction.  The finite adapter binds every half-Reception stage and delayed
REMESH result to the global P2 numeric theorem.  The active history reaches
exact spatial consensus after ``tau_global + 1`` observed cycles.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
import json
from typing import Any

import networkx as nx

from tnfr.operators.event_remesh_causal_runtime import (
    EventRemeshCycleExecutionSpec,
    execute_event_remesh_cycle_sequence,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.physics import (
    certify_alpha_one_hard_clip_remesh_class,
    certify_executed_p2_half_reception_remesh_sequence,
    certify_p2_half_reception_remesh_stability,
)


def _refresh_zero_pressure(graph: nx.Graph) -> None:
    for node in graph:
        graph.nodes[node]["delta_nfr"] = 0.0


def _runtime_graph() -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        _gamma_spec={"type": "none"},
        RANDOM_SEED=7,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        GLYPH_FACTORS={
            "EN_mix": 0.5,
            "IL_lambda": 0.1,
            "REMESH_alpha": 1.0,
        },
        REMESH_TAU_LOCAL=1,
        REMESH_TAU_GLOBAL=1,
        REMESH_ALPHA=1.0,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        compute_delta_nfr=_refresh_zero_pressure,
    )
    for node, epi in enumerate((-1.0, 0.5)):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="example",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            latent=False,
            glyph_history=["AL"],
            epi_history=[epi, epi],
        )
    graph.graph["_epi_hist"] = deque([{0: 0.5, 1: -0.5}], maxlen=64)
    return graph


def _cycle_specs() -> tuple[EventRemeshCycleExecutionSpec, ...]:
    word = ("reception", "coherence", "recursivity")
    return tuple(
        EventRemeshCycleExecutionSpec(
            build_operator_event_schedule(
                word,
                start_time=0.0,
                flow_durations=(0.0, 0.0, 0.0, 0.0),
            )
        )
        for _index in range(2)
    )


def run_protocol() -> dict[str, Any]:
    """Execute two cycles and bind their finite active-history extinction."""

    source = certify_alpha_one_hard_clip_remesh_class(
        (0, 1),
        (1.0, 1.0),
        tau_local=1,
        tau_global=1,
        epi_min=-1.0,
        epi_max=1.0,
    )
    kernel = certify_p2_half_reception_remesh_stability(source)
    execution = execute_event_remesh_cycle_sequence(
        _runtime_graph(),
        _cycle_specs(),
        metric_weights=(1.0, 1.0),
        context={"initial_epi_nonzero": True},
        suppress_birth_warnings=True,
        require_runtime_telescope=False,
    )
    certificate = certify_executed_p2_half_reception_remesh_sequence(
        kernel,
        execution,
    )
    return {"execution": execution, "certificate": certificate}


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _history_text(
    histories: tuple[tuple[tuple[Fraction, Fraction], ...], ...],
) -> list[list[list[str]]]:
    return [
        [
            [_fraction_text(left), _fraction_text(right)]
            for left, right in history
        ]
        for history in histories
    ]


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return the finite positive result and all withheld promotions."""

    execution = protocol["execution"]
    certificate = protocol["certificate"]
    return {
        "claim": "finite causal P2 active-history disagreement extinction",
        "certificate_valid": (
            certificate
            .executed_p2_half_reception_remesh_sequence_certificate_certified
        ),
        "cycle_count": certificate.cycle_count,
        "extinction_horizon": certificate.active_history_extinction_horizon,
        "extinction_cycle_index": certificate.guaranteed_extinction_cycle_index,
        "post_reception_energies": [
            _fraction_text(value)
            for value in certificate.exact_post_reception_energies
        ],
        "post_remesh_energies": [
            _fraction_text(value)
            for value in certificate.exact_post_remesh_energies
        ],
        "active_history_suffixes": _history_text(
            certificate.exact_active_history_suffixes
        ),
        "scope": {
            "same_invocation_causal_provenance": (
                certificate.same_invocation_causal_provenance_certified
            ),
            "whole_sequence_graph_state_atomicity": (
                certificate.whole_sequence_graph_state_atomicity_certified
            ),
            "runtime_telescope_included": execution.runtime_telescope is not None,
            "every_reception_stage_q_zero": (
                certificate.every_reception_stage_q_zero_certified
            ),
            "every_remesh_global_delay_copy_eta_zero": (
                certificate.every_remesh_global_delay_copy_eta_zero_certified
            ),
            "observed_active_history_extinction": (
                certificate.observed_active_history_extinction_certified
            ),
            "future_runtime_stability": (
                certificate.future_runtime_stability_certified
            ),
            "unobserved_repetition_stability": (
                certificate.unobserved_repetition_stability_certified
            ),
            "auxiliary_state_stability": (
                certificate.auxiliary_state_stability_certified
            ),
            "solver_accuracy": certificate.solver_accuracy_certified,
            "full_tnfr_stability": certificate.full_tnfr_stability_certified,
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
