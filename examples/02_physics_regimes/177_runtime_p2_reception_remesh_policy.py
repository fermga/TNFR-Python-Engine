"""Execute two independently validated finite P2 policy invocations.

The reusable entry point accepts only the zero-flow canonical
``EN -> IL -> REMESH`` cycle used by the restricted P2 theorem.  It checks the
live graph, metric, factors, REMESH controls and active history on every call,
then executes and post-certifies inside one outer graph transaction.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
import json
from typing import Any

import networkx as nx

from tnfr.operators.event_remesh_causal_runtime import (
    EventRemeshCycleExecutionSpec,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.physics import (
    certify_alpha_one_hard_clip_remesh_class,
    certify_p2_half_reception_remesh_stability,
    execute_p2_half_reception_remesh_policy_invocation,
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
    graph.graph["_epi_hist"] = deque(
        [{0: 0.5, 1: 0.25}],
        maxlen=64,
    )
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
    """Run two calls whose finite certificates have independent provenance."""

    graph = _runtime_graph()
    source = certify_alpha_one_hard_clip_remesh_class(
        (0, 1),
        (1.0, 1.0),
        tau_local=1,
        tau_global=1,
        epi_min=-1.0,
        epi_max=1.0,
    )
    kernel = certify_p2_half_reception_remesh_stability(source)

    certificates = tuple(
        execute_p2_half_reception_remesh_policy_invocation(
            graph,
            kernel,
            _cycle_specs(),
            metric_weights=(1.0, 1.0),
            suppress_birth_warnings=True,
        )
        for _invocation in range(2)
    )
    return {"graph": graph, "certificates": certificates}


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Expose both finite results and the promotions they still withhold."""

    first, second = protocol["certificates"]
    return {
        "claim": "two independently revalidated finite P2 invocations",
        "certificate_valid": [
            certificate
            .executed_p2_half_reception_remesh_sequence_certificate_certified
            for certificate in (first, second)
        ],
        "distinct_executions": first.execution is not second.execution,
        "cycle_counts": [first.cycle_count, second.cycle_count],
        "post_remesh_energies": [
            [_fraction_text(value) for value in certificate.exact_post_remesh_energies]
            for certificate in (first, second)
        ],
        "retained_history_length": len(protocol["graph"].graph["_epi_hist"]),
        "scope": {
            "finite_causal_extinction": [
                first.finite_causal_extinction_certified,
                second.finite_causal_extinction_certified,
            ],
            "future_runtime_stability": second.future_runtime_stability_certified,
            "unobserved_repetition_stability": (
                second.unobserved_repetition_stability_certified
            ),
            "auxiliary_state_stability": second.auxiliary_state_stability_certified,
            "full_tnfr_stability": second.full_tnfr_stability_certified,
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
