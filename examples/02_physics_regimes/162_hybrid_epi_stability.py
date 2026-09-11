"""Certify three affine-reset boundaries for hybrid pure-EPI dynamics.

The fixed flow is the exact pure-EPI diffusion channel on a two-node weighted
graph.  Its heterogeneous capacities produce the represented common metric
``h = (1, 3)`` and the rationally certified energy-rate lower bound ``r = 8``.
Three declared
affine resets then separate the conclusions that are often conflated:

1. a weighted-mean-preserving jump whose amplification is overcome by
   diffusion, while the represented flow's displayed mean is not exact;
2. a uniform translation that contracts disagreement but shifts consensus; and
3. a local offset that sends zero disagreement to positive disagreement and
   therefore has no finite global multiplicative gain.

Canonical operator names are metadata for the declared affine maps.  This
example does not identify those maps with runtime implementations or infer a
gain from grammar roles.
"""

from __future__ import annotations

import json
import math
import platform
from fractions import Fraction
from pathlib import Path
from typing import Any

import networkx as nx

import tnfr
from tnfr.mathematics.unified_numerical import np
from tnfr.physics import (
    certify_affine_epi_jump_gain,
    compose_hybrid_epi_stability,
    verify_heterogeneous_diffusion_stability,
)
from tnfr.research import (
    ClaimStatus,
    CoreExperimentManifest,
    current_git_source_provenance,
)


NODES = ("left", "right")
EDGE_WEIGHT = 3.0
INITIAL_EPI = (2.0, -1.0)
FREQUENCY = (3.0, 1.0)
EXPECTED_METRIC = (1.0, 3.0)
FLOW_DURATIONS = (0.125, 0.125)
AMPLIFICATION_MATRIX = (
    (1.75, -0.75),
    (-0.25, 1.25),
)
UNIFORM_TRANSLATION = (1.0, 1.0)
LOCAL_OFFSET = (1.0, 0.0)
SOURCE_SNAPSHOT_PATHS = (
    "src/tnfr",
    "examples/02_physics_regimes/162_hybrid_epi_stability.py",
)


def weighted_graph() -> nx.Graph:
    """Build the fixed two-node flow with common metric ``h=(1, 3)``."""
    graph = nx.Graph()
    graph.add_edge(*NODES, weight=EDGE_WEIGHT)
    for node, epi, frequency in zip(NODES, INITIAL_EPI, FREQUENCY):
        graph.nodes[node].update(EPI=epi, nu_f=frequency, theta=0.0)
    return graph


def run_protocol() -> dict[str, Any]:
    """Construct the flow, three reset certificates, and two hybrid words."""
    flow = verify_heterogeneous_diffusion_stability(weighted_graph())
    metric = flow.metric_weights

    amplification = certify_affine_epi_jump_gain(
        "Reception",
        AMPLIFICATION_MATRIX,
        metric,
        nodes=flow.nodes,
    )
    amplification_word = compose_hybrid_epi_stability(
        flow,
        (amplification,),
        FLOW_DURATIONS,
        repeat_schedule=True,
    )

    uniform_translation = certify_affine_epi_jump_gain(
        "Emission",
        np.eye(2),
        metric,
        offset=UNIFORM_TRANSLATION,
        nodes=flow.nodes,
    )
    translation_word = compose_hybrid_epi_stability(
        flow,
        (uniform_translation,),
        FLOW_DURATIONS,
        repeat_schedule=True,
    )

    local_offset = certify_affine_epi_jump_gain(
        "Emission",
        np.eye(2),
        metric,
        offset=LOCAL_OFFSET,
        nodes=flow.nodes,
    )

    return {
        "flow": flow,
        "amplification": amplification,
        "amplification_word": amplification_word,
        "uniform_translation": uniform_translation,
        "translation_word": translation_word,
        "local_offset": local_offset,
    }


def _fraction_text(value: Fraction) -> str:
    """Serialize one exact rational without losing its denominator."""
    return f"{value.numerator}/{value.denominator}"


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Build a compact JSON-compatible report with source provenance."""
    flow = protocol["flow"]
    amplification = protocol["amplification"]
    amplification_word = protocol["amplification_word"]
    uniform_translation = protocol["uniform_translation"]
    translation_word = protocol["translation_word"]
    local_offset = protocol["local_offset"]

    git_sha, source_dirty, dirty_source_hash = current_git_source_provenance(
        Path(__file__).resolve().parents[2], SOURCE_SNAPSHOT_PATHS
    )
    manifest = CoreExperimentManifest(
        claim_id="S1-S2-AFFINE-HYBRID-EPI-BOUNDARY",
        git_sha=git_sha,
        versions={"python": platform.python_version(), "tnfr": tnfr.__version__},
        graph_construction="two nodes joined by symmetric conductance weight 3",
        capacity_specification="positive heterogeneous nu_f=(3,1); h=(1,3)",
        solver="exact-rational affine hypotheses plus certified flow composition",
        result_status=ClaimStatus.DERIVED,
        seed=None,
        timestep=None,
        operator_sequence=(),
        telemetry=("EPI", "nu_f", "common-metric disagreement energy"),
        controls=(
            "insufficient flow duration",
            "uniform consensus translation",
            "nonuniform local offset",
        ),
        artifacts=("stdout compact JSON certificate",),
        source_dirty=source_dirty,
        dirty_source_hash=dirty_source_hash,
    )
    manifest.validate_for_admission()

    minimum_amplification_flow = math.log(
        amplification.energy_gain_bound_for_composition
    ) / flow.certified_exponential_rate_lower_bound
    return {
        "claim": "conditional affine-reset hybrid pure-EPI stability boundary",
        "flow": {
            "nodes": list(flow.nodes),
            "edge_weight": EDGE_WEIGHT,
            "nu_f": list(FREQUENCY),
            "metric_weights": [float(value) for value in flow.metric_weights],
            "spectral_energy_decay_rate_estimate": float(flow.exponential_rate),
            "certified_energy_decay_rate_lower_bound": float(
                flow.certified_exponential_rate_lower_bound
            ),
            "exact_consensus_subspace_preservation": bool(
                flow.exact_consensus_subspace_preservation
            ),
            "exact_uniform_fixed_point_preservation": bool(
                flow.exact_uniform_fixed_point_preservation
            ),
            "exact_weighted_mean_preservation": bool(
                flow.exact_weighted_mean_preservation
            ),
            "flow_hypotheses_pass": bool(flow.is_certified),
        },
        "cases": {
            "mean_preserving_amplification": {
                "operator_metadata": amplification.operator_name,
                "exact_consensus_subspace_preservation": (
                    amplification.exact_consensus_subspace_preservation
                ),
                "exact_weighted_mean_preservation": (
                    amplification.exact_weighted_mean_preservation
                ),
                "exact_Gamma_F": _fraction_text(
                    amplification.exact_weighted_frobenius_energy_bound
                ),
                "total_flow_duration": amplification_word.total_flow_duration,
                "minimum_flow_duration_for_strict_bound": (
                    minimum_amplification_flow
                ),
                "energy_multiplier_bound": (
                    amplification_word.energy_multiplier_bound
                ),
                "repeated_disagreement_convergence_certified": (
                    amplification_word.
                    repeated_schedule_disagreement_convergence_certified
                ),
                "initial_weighted_consensus_convergence_certified": (
                    amplification_word.
                    repeated_schedule_initial_weighted_consensus_convergence_certified
                ),
            },
            "uniform_translation": {
                "operator_metadata": uniform_translation.operator_name,
                "exact_consensus_subspace_preservation": (
                    uniform_translation.exact_consensus_subspace_preservation
                ),
                "exact_weighted_mean_preservation": (
                    uniform_translation.exact_weighted_mean_preservation
                ),
                "exact_Gamma_F": _fraction_text(
                    uniform_translation.exact_weighted_frobenius_energy_bound
                ),
                "energy_multiplier_bound": translation_word.energy_multiplier_bound,
                "repeated_disagreement_convergence_certified": (
                    translation_word.
                    repeated_schedule_disagreement_convergence_certified
                ),
                "initial_weighted_consensus_convergence_certified": (
                    translation_word.
                    repeated_schedule_initial_weighted_consensus_convergence_certified
                ),
                "weighted_consensus_shift_per_jump": 1.0,
            },
            "local_offset": {
                "operator_metadata": local_offset.operator_name,
                "exact_consensus_subspace_preservation": (
                    local_offset.exact_consensus_subspace_preservation
                ),
                "finite_global_energy_gain": local_offset.finite_global_energy_gain,
                "global_energy_gain": "infinite",
                "zero_energy_input_level": local_offset.consensus_counterexample_level,
                "exact_output_energy": _fraction_text(
                    local_offset.exact_consensus_counterexample_energy_after
                ),
            },
        },
        "manifest": manifest.to_dict(),
        "scope": {
            "operator_names_are_metadata_only": True,
            "runtime_operator_realization_claimed": False,
            "grammar_conclusion_claimed": False,
            "excluded": (
                "nonlinear clipping, phase, independent pressure, topology/history "
                "changes, REMESH, and unsupplied consensus dynamics"
            ),
        },
    }


def main() -> None:
    """Run the protocol and print its compact reproducibility record."""
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
