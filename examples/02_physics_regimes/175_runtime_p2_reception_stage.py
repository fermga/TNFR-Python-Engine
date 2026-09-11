"""Bind one executed Reception event to the global P2 half-mix kernel.

The event executor validates and commits a grammar-aware two-phase Reception
stage on a real two-node graph.  The finite adapter then proves that its
captured EPI endpoints realize the restricted global ``q=0`` kernel.  This
does not bind REMESH history to the graph or prove repeated runtime stability.
"""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

import networkx as nx

from tnfr.operators.event_runtime import execute_operator_event_schedule
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.physics import (
    certify_alpha_one_hard_clip_remesh_class,
    certify_executed_p2_half_reception_stage,
    certify_p2_half_reception_remesh_stability,
)


def _runtime_graph() -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        RANDOM_SEED=7,
        GLYPH_FACTORS={
            "EN_mix": 0.5,
            "IL_lambda": 0.1,
            "REMESH_alpha": 1.0,
        },
        REMESH_TAU_LOCAL=1,
        REMESH_TAU_GLOBAL=1,
    )
    for node, epi in zip(graph, (-1.0, 0.5), strict=True):
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
    return graph


def run_protocol() -> dict[str, Any]:
    """Execute the stage and construct its finite causal certificate."""

    source = certify_alpha_one_hard_clip_remesh_class(
        (0, 1),
        (1.0, 1.0),
        tau_local=1,
        tau_global=1,
        epi_min=-1.0,
        epi_max=1.0,
    )
    kernel = certify_p2_half_reception_remesh_stability(source)
    schedule = build_operator_event_schedule(
        ("reception", "coherence", "recursivity"),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.0),
    )
    execution = execute_operator_event_schedule(
        _runtime_graph(),
        schedule,
        context={"initial_epi_nonzero": True},
        include_stage_certificates=True,
        suppress_birth_warnings=True,
    )
    certificate = certify_executed_p2_half_reception_stage(kernel, execution)
    return {"execution": execution, "certificate": certificate}


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return the observed endpoint theorem and its explicit scope."""

    certificate = protocol["certificate"]
    return {
        "claim": "one executed P2 half-Reception EPI stage realizes global q=0",
        "certificate_valid": (
            certificate.executed_p2_half_reception_stage_certificate_certified
        ),
        "event_index": certificate.event_index,
        "node_order": list(certificate.node_order),
        "epi_before": list(certificate.binary64_epi_before),
        "epi_after": list(certificate.binary64_epi_after),
        "normalized_metric": [
            _fraction_text(value)
            for value in certificate.exact_normalized_metric
        ],
        "energy_before": _fraction_text(
            certificate.exact_centered_energy_before
        ),
        "energy_after": _fraction_text(
            certificate.exact_centered_energy_after
        ),
        "q": _fraction_text(
            certificate.exact_global_kernel_energy_gain_upper_bound
        ),
        "scope": {
            "finite_executor_bound_epi_stage": (
                certificate.finite_executor_bound_epi_stage_certified
            ),
            "finite_grammar_admission": (
                certificate.finite_grammar_admission_observed
            ),
            "two_phase_reception_stage": (
                certificate.canonical_two_phase_reception_stage_observed
            ),
            "observed_numeric_consensus": (
                certificate.observed_numeric_consensus_certified
            ),
            "source_q_zero_observed_epi": (
                certificate.source_global_q_zero_applies_to_observed_epi_transition
            ),
            "schedule_graph_state_atomicity": (
                certificate.finite_schedule_graph_state_atomicity_certified
            ),
            "executed_remesh_configuration": (
                certificate.executed_remesh_configuration_bound
            ),
            "future_or_repeated_runtime": (
                certificate.future_or_repeated_live_graph_stability_certified
            ),
            "solver_accuracy": certificate.solver_accuracy_certified,
            "full_tnfr_stability": certificate.full_tnfr_stability_certified,
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
