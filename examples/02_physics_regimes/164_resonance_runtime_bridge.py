"""Audit the identity-gated runtime realization of Resonance (RA).

The hub has one U3-compatible neighbour and one antiphase neighbour.  Only the
compatible edge contributes to RA's unweighted EPI and phase means.  The local
capacity boost changes the post-RA diffusion metric, so the report exposes a
fixed post-RA certificate while abstaining from a common-metric switching
claim across the pre/post generators.
"""

from __future__ import annotations

import json
import math
from typing import Any

import networkx as nx

from tnfr.physics import certify_resonance_epi_realization


TARGET = ("hub", 1)
COMPATIBLE = "aligned"
INCOMPATIBLE = 17


def resonance_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge(TARGET, COMPATIBLE, weight=9.0)
    graph.add_edge(TARGET, INCOMPATIBLE, weight=1.0)
    values = {
        TARGET: (0.0, 10.0, 0.0),
        COMPATIBLE: (1.0, 9.0, 0.2),
        INCOMPATIBLE: (-1.0, 1.0, math.pi),
    }
    for node, (epi, frequency, phase) in values.items():
        graph.nodes[node].update(
            EPI=epi,
            nu_f=frequency,
            theta=phase,
            epi_kind="seed",
        )
    return graph


def run_protocol():
    return certify_resonance_epi_realization(
        resonance_graph(),
        TARGET,
        fixed_support_declared=True,
        mix_factor=0.5,
        vf_amplification_factor=0.25,
        phase_coupling_factor=0.5,
    )


def build_report(result) -> dict[str, Any]:
    post_flow = result.post_diffusion_certificate
    if post_flow is None or result.affine_jump_certificate is None:
        raise RuntimeError("the declared RA snapshot must admit both fixed certificates")
    return {
        "claim": "conditional local Resonance runtime realization",
        "runtime": {
            "target": repr(result.target),
            "graph_neighbors": [repr(node) for node in result.graph_neighbors],
            "propagating_neighbors": [repr(node) for node in result.runtime_neighbors],
            "excluded_by_u3": [
                repr(node) for node in result.phase_incompatible_neighbors
            ],
            "target_epi_before": float(result.state_before[result.target_index]),
            "target_epi_after": result.runtime_target_value,
            "target_frequency_after": float(
                result.frequency_after[result.target_index]
            ),
            "target_phase_after": result.phase_after,
            "identity_gate_passed": result.identity_gate_passed,
        },
        "transport": {
            "ra_unweighted_neighbor_mean": result.unweighted_runtime_neighbor_mean,
            "conductance_weighted_neighbor_mean": (
                result.transport_weighted_neighbor_mean
            ),
            "post_flow_certified": post_flow.is_certified,
            "pressure_refresh_required": result.pressure_refresh_required,
        },
        "hybrid_boundary": {
            "represented_jump_eligible": result.represented_affine_jump_eligible,
            "pre_post_metric_exactly_proportional": (
                result.pre_post_metric_exactly_proportional
            ),
            "pre_post_switching_certificate_available": (
                result.pre_post_switching_certificate is not None
            ),
            "switching_abstention_reasons": list(
                result.switching_abstention_reasons
            ),
            "global_binary64_runtime_affinity_certified": (
                result.global_binary64_runtime_affinity_certified
            ),
        },
        "scope": result.scope,
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
