"""Run the restricted S16 endpoint certificate with explicit provenance.

This example constructs two frozen pure-EPI states on the same weighted path,
uses a reversible reflection partition, and evaluates the shared stability,
observability, scale-closure and structural-distance boundary.  A passing result
certifies only those endpoint conditions.  It does not certify a trajectory,
phase dynamics, operator identification, REMESH, or changing topology.
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import networkx as nx

import tnfr
from tnfr.mathematics.unified_numerical import np
from tnfr.physics import (
    StructuralChannelScales,
    certify_core_research_integration,
    structural_diffusion_operator,
)
from tnfr.research import (
    ClaimStatus,
    CoreExperimentManifest,
    current_git_source_provenance,
)


PARTITION = ((0, 3), (1, 2))
SOURCE_SNAPSHOT_PATHS = (
    "src/tnfr",
    "examples/02_physics_regimes/160_core_research_integration.py",
)
SCALES = StructuralChannelScales(
    epi=1.0,
    frequency=1.0,
    phase=math.pi,
    pressure=1.0,
    epi_rate=1.0,
    edge_conductance=1.0,
    edge_length=1.0,
)


def frozen_state(
    epi: tuple[float, ...],
    frequency: tuple[float, ...],
    phase: tuple[float, ...],
) -> nx.Graph:
    """Build one state satisfying the pure-EPI pressure and nodal channels."""
    graph = nx.path_graph(4)
    graph.edges[0, 1]["weight"] = 2.0
    graph.edges[1, 2]["weight"] = 0.75
    graph.edges[2, 3]["weight"] = 2.0
    for node, form, rate, angle in zip(graph, epi, frequency, phase):
        graph.nodes[node].update(EPI=form, nu_f=rate, theta=angle)

    nodes, laplacian = structural_diffusion_operator(graph)
    field = np.asarray([graph.nodes[node]["EPI"] for node in nodes], dtype=float)
    pressure = -(laplacian @ field)
    for node, value in zip(nodes, pressure):
        graph.nodes[node]["delta_nfr"] = float(value)
        graph.nodes[node]["dEPI_dt"] = float(
            graph.nodes[node]["nu_f"] * value
        )
    return graph


def main() -> None:
    left = frozen_state(
        (2.0, -1.0, 3.0, 0.5),
        (0.5, 1.5, 1.5, 0.5),
        (0.0, 0.2, 0.4, 0.6),
    )
    right = frozen_state(
        (1.5, -0.5, 2.0, 0.25),
        (0.8, 1.2, 1.2, 0.8),
        (0.1, 0.3, 0.5, 0.7),
    )
    certificate = certify_core_research_integration(
        left,
        right,
        PARTITION,
        scales=SCALES,
    )
    git_sha, source_dirty, dirty_source_hash = current_git_source_provenance(
        Path(__file__).resolve().parents[2], SOURCE_SNAPSHOT_PATHS
    )

    manifest = CoreExperimentManifest(
        claim_id="S16-RESTRICTED-ENDPOINT-INTERSECTION",
        git_sha=git_sha,
        versions={"python": platform.python_version(), "tnfr": tnfr.__version__},
        graph_construction="four-node weighted path; two frozen states",
        capacity_specification="positive heterogeneous nu_f at both endpoints",
        solver="read-only analytic/numerical certificate composition",
        result_status=ClaimStatus.MEASURED,
        seed=None,
        timestep=None,
        operator_sequence=(),
        telemetry=("EPI", "nu_f", "phase", "DeltaNFR", "dEPI"),
        controls=("nonclosing partition", "pressure mismatch", "rate mismatch"),
        artifacts=("stdout JSON summary",),
        source_dirty=source_dirty,
        dirty_source_hash=dirty_source_hash,
    )
    manifest.validate_for_admission()

    print(
        json.dumps(
            {
                "joint_numerical_conditions_pass": (
                    certificate.joint_numerical_conditions_pass
                ),
                "failed_conditions": certificate.failed_conditions,
                "structural_state_distance": (
                    certificate.structural_distance.distance
                ),
                "conditions": dict(certificate.numerical_conditions),
                "manifest": manifest.to_dict(),
                "scope": certificate.scope,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
