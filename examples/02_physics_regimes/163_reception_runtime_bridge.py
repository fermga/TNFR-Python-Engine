"""Expose the exact and numerical boundaries of one local Reception update.

The target has two neighbours with conductances 9 and 1. Reception uses their
unweighted runtime mean, while pure-EPI diffusion uses the conductance-weighted
mean. The uniform real-scalar state, convex half-mix and inactive hard clipping
place this snapshot inside the declared affine realization domain.

The report keeps four conclusions separate: the runtime value, the represented
affine jump bound, drift of the common weighted mean, and the pressure refresh
required before a subsequent pure-EPI flow segment. The optional recovery
Boolean comes from the exact hybrid composer; the displayed break-even time is
only a numerical estimate.
"""

from __future__ import annotations

from fractions import Fraction
import json
import platform
from pathlib import Path
from typing import Any

import networkx as nx

import tnfr
from tnfr.physics import certify_reception_epi_realization
from tnfr.research import (
    ClaimStatus,
    CoreExperimentManifest,
    current_git_source_provenance,
)


NODES = ("hub", "strong", "weak")
EDGE_WEIGHTS = (9.0, 1.0)
EPI = (0.0, 1.0, 0.0)
FREQUENCY = (10.0, 9.0, 1.0)
MIX_FACTOR = 0.5
SOURCE_SNAPSHOT_PATHS = (
    "src/tnfr/operators/_reception_kernel.py",
    "src/tnfr/physics/reception_realization.py",
    "src/tnfr/physics/hybrid_operator_stability.py",
    "src/tnfr/physics/structural_diffusion.py",
    "examples/02_physics_regimes/163_reception_runtime_bridge.py",
)


def weighted_reception_graph() -> nx.Graph:
    """Build a weighted star whose diffusion metric is exactly uniform."""

    graph = nx.Graph()
    graph.add_edge(NODES[0], NODES[1], weight=EDGE_WEIGHTS[0])
    graph.add_edge(NODES[0], NODES[2], weight=EDGE_WEIGHTS[1])
    for node, epi, frequency in zip(NODES, EPI, FREQUENCY):
        graph.nodes[node].update(EPI=epi, nu_f=frequency, theta=0.0)
    return graph


def run_protocol() -> dict[str, Any]:
    """Audit Reception and certify a deliberately sufficient recovery time."""

    graph = weighted_reception_graph()
    baseline = certify_reception_epi_realization(
        graph,
        NODES[0],
        fixed_support_declared=True,
        mix_factor=MIX_FACTOR,
    )
    estimate = baseline.recovery_break_even_duration_estimate
    if estimate is None or estimate <= 0.0:
        raise RuntimeError("the declared Reception case must have a positive estimate")
    recovery_duration = 2.0 * estimate
    recovered = certify_reception_epi_realization(
        graph,
        NODES[0],
        fixed_support_declared=True,
        mix_factor=MIX_FACTOR,
        recovery_flow_duration=recovery_duration,
    )
    return {
        "baseline": baseline,
        "recovered": recovered,
        "recovery_duration": recovery_duration,
    }


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, JSON-compatible reproducibility record."""

    baseline = protocol["baseline"]
    recovered = protocol["recovered"]
    jump = baseline.affine_jump_certificate
    hybrid = recovered.hybrid_certificate
    if jump is None or hybrid is None:
        raise RuntimeError("the declared in-domain realization must be certifiable")

    git_sha, source_dirty, dirty_source_hash = current_git_source_provenance(
        Path(__file__).resolve().parents[2], SOURCE_SNAPSHOT_PATHS
    )
    manifest = CoreExperimentManifest(
        claim_id="S1-S2-RECEPTION-RUNTIME-REALIZATION",
        git_sha=git_sha,
        versions={"python": platform.python_version(), "tnfr": tnfr.__version__},
        graph_construction="three-node weighted star with conductances 9 and 1",
        capacity_specification="nu_f equals weighted degree, giving displayed h=(1,1,1)",
        solver="read-only EN realization plus exact-rational hybrid certificates",
        result_status=ClaimStatus.DERIVED,
        seed=None,
        timestep=None,
        operator_sequence=("Reception",),
        telemetry=(
            "EPI",
            "nu_f",
            "pure-EPI pressure defect",
            "common-metric disagreement energy",
        ),
        controls=(
            "unweighted Reception mean versus weighted transport mean",
            "weighted-mean drift",
            "pressure refresh",
        ),
        artifacts=("stdout compact JSON certificate",),
        source_dirty=source_dirty,
        dirty_source_hash=dirty_source_hash,
    )
    manifest.validate_for_admission()

    return {
        "claim": "conditional local Reception runtime realization",
        "runtime": {
            "nodes": list(baseline.nodes),
            "target": baseline.target,
            "mix_factor": baseline.mix_factor,
            "unweighted_neighbor_mean": baseline.unweighted_runtime_neighbor_mean,
            "conductance_weighted_neighbor_mean": (
                baseline.transport_weighted_neighbor_mean
            ),
            "target_before": float(baseline.state_before[baseline.target_index]),
            "target_after": baseline.runtime_target_value,
            "runtime_matches_represented_matrix_at_snapshot": (
                baseline.runtime_matches_represented_affine_exactly_at_snapshot
            ),
        },
        "affine_boundary": {
            "ideal_real_domain": baseline.ideal_real_affine_regime,
            "ideal_real_hard_clipping_inactive_by_convexity": (
                baseline.ideal_real_hard_clipping_inactive_by_convexity
            ),
            "runtime_hard_clipping_inactive_at_snapshot": (
                baseline.runtime_hard_clipping_inactive_at_snapshot
            ),
            "represented_jump_eligible": baseline.represented_affine_jump_eligible,
            "exact_target_row_sum": _fraction_text(
                baseline.exact_represented_target_row_sum
            ),
            "exact_frobenius_gain_bound": _fraction_text(
                jump.exact_weighted_frobenius_energy_bound
            ),
            "global_binary64_runtime_affinity_certified": (
                baseline.global_binary64_runtime_affinity_certified
            ),
        },
        "consensus_and_pressure": {
            "ideal_map_preserves_weighted_mean": (
                baseline.ideal_real_global_weighted_mean_preservation
            ),
            "exact_snapshot_weighted_mean_shift": _fraction_text(
                baseline.exact_current_weighted_mean_shift
            ),
            "pressure_refresh_required": baseline.pressure_refresh_required,
            "exact_refresh_iff_nontrivial_theorem": (
                baseline.exact_pressure_refresh_iff_nontrivial_theorem
            ),
            "observed_pressure_defect_norm": (
                baseline.post_reset_pressure_manifold_defect_norm
            ),
        },
        "recovery": {
            "break_even_duration_estimate": (
                baseline.recovery_break_even_duration_estimate
            ),
            "declared_flow_duration": protocol["recovery_duration"],
            "exact_hybrid_contraction_decision": (
                hybrid.disagreement_contracts_over_declared_horizon
            ),
            "initial_weighted_mean_preserved": (
                hybrid.initial_weighted_mean_preserved
            ),
        },
        "manifest": manifest.to_dict(),
        "scope": {
            "fixed_support": True,
            "uniform_real_scalar_embedding": True,
            "ideal_hard_clipping_controlled_by_convexity": True,
            "runtime_clip_observation_is_snapshot_only": True,
            "excluded": (
                "genuinely non-scalar or complex BEPI states, active or soft "
                "clipping, repeated words, phase, multichannel pressure, topology "
                "and history mutation"
            ),
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
