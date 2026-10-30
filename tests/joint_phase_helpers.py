"""Shared graph preparation for bounded joint phase/form model controls.

Preparation and a thin adapter reuse the existing execution owners; no new
solver, pressure fit or substrate generator is introduced.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import networkx as nx
import numpy as np

from tnfr.alias import set_theta
from tnfr.constants import DEFAULTS
from tnfr.constants.aliases import ALIAS_SI
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import DefaultIntegrator
from tnfr.dynamics.phase_evolution import propose_u3_gated_phase_step
from tnfr.operators._coupling_stage_kernel import propose_coupling_stage
from tnfr.operators.definitions import Coupling
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.network_stage import _detached_stage_graph, execute_coupling_stage
from tnfr.physics.forcing_realization import (
    NonEpiForcingObservation,
    capture_non_epi_forcing,
)
from tnfr.physics.structural_diffusion import structural_eigenmodes
from tnfr.physics.support_transport import (
    SupportTransportSnapshot,
    observe_support_transport,
)
from tnfr.types import Glyph

WEIGHTS = {"phase": 1.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}


def exact_phase_cycle_state(graph, phase_turns):
    """Prepare declared exact circular geometry, without a runtime phase write."""
    from tnfr.physics.phase_cycle_geometry import (
        derive_phase_cycle_geometry,
        reconstruct_phase_cycle_state,
    )

    geometry = derive_phase_cycle_geometry(graph)
    values = tuple(Fraction(phase_turns[node]) for node in geometry.nodes)
    gaps = tuple(
        (values[j] - values[i] + Fraction(1, 2)) % 1 - Fraction(1, 2)
        for i, j in geometry.edges
    )
    return reconstruct_phase_cycle_state(geometry, edge_turns=gaps)


def configure(graph, *, weights=WEIGHTS):
    graph.graph.update(
        DNFR_WEIGHTS=dict(weights),
        DELTA_PHI_MAX=math.pi / 2,
        UM_MAX_PHASE_DIFF=math.pi / 2,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        vectorized_dnfr=True,
        _t=0.0,
    )


def triangle(
    *, epi=(0.25, -0.125, 0.375), phase=(0.25, 0.75), capacity=1.0, weights=WEIGHTS
):
    graph = nx.complete_graph(3)
    configure(graph, weights=weights)
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 1.0, "length")
    phases = (phase[0], phase[1], phase[1]) if len(phase) == 2 else phase
    for node, x, theta in zip(graph, epi, phases, strict=True):
        graph.nodes[node].update(
            EPI=float(x),
            theta=float(theta),
            nu_f=float(capacity),
            delta_nfr=0.0,
            dEPI=0.0,
        )
    default_compute_delta_nfr(graph)
    return graph


def project(values):
    """Independent metric projection for common-capacity unit K3."""
    return values[0], (values[1] + values[2]) / 2


def unit_barbell(*, capacities=(1, 1)):
    """Prepare zero form and uniform phase on the shared six-node support."""
    graph = nx.barbell_graph(3, 0)
    configure(graph)
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 1.0, "length")
    for node in graph:
        graph.nodes[node].update(
            EPI=0.0,
            theta=math.pi,
            nu_f=float(capacities[int(node >= 3)]),
            delta_nfr=0.0,
            dEPI=0.0,
        )
    default_compute_delta_nfr(graph)
    return graph


def barbell():
    """Freeze the two-mode regional-window preparation before any execution."""
    graph = unit_barbell()
    _, vectors = structural_eigenmodes(graph)
    modes = vectors / np.sqrt([graph.degree(node) for node in graph])[:, None]
    selected = modes[:, [1, 2]].copy()
    for column in range(2):
        pivot = next(value for value in selected[:, column] if abs(value) > 1e-12)
        if pivot < 0:
            selected[:, column] *= -1
    offset = np.sum(selected, axis=1) / 32
    for node, value in zip(graph, offset, strict=True):
        graph.nodes[node].update(
            EPI=-float(value) / math.pi,
            theta=0.125 + float(value),
            nu_f=1.0,
            delta_nfr=0.0,
            dEPI=0.0,
        )
    default_compute_delta_nfr(graph)
    return graph


@dataclass(frozen=True)
class JointExecutionStep:
    """Same-snapshot input and observed endpoints of existing engine owners."""

    before: NonEpiForcingObservation
    after_epi: SupportTransportSnapshot
    phase_after: tuple[float, ...]


def execute_joint_step(graph, *, dt, coupling_strength, time):
    """Compose the shared phase proposal and nodal Euler in a finite test.

    The caller must have refreshed the initial pressure. This adapter adds no
    dynamics, controller, chart certificate or numerical convergence claim.
    """
    before = capture_non_epi_forcing(graph)
    phase_after = propose_u3_gated_phase_step(
        graph,
        before.snapshot.nodes,
        before.phase,
        before.snapshot.capacity,
        dt=float(dt),
        coupling_strength=float(coupling_strength),
    )
    DefaultIntegrator().integrate(
        graph, dt=float(dt), t=float(time), method="euler", n_jobs=1
    )
    after = observe_support_transport(graph)
    for node, value in zip(graph, phase_after, strict=True):
        set_theta(graph, node, float(value))
    default_compute_delta_nfr(graph)
    return JointExecutionStep(before, after, tuple(map(float, phase_after)))


def prepared_coupling_path(*, candidate_si=0.8, functional_links=True):
    graph = nx.path_graph(5)
    configure(graph)
    graph.graph.update(
        RANDOM_SEED=17,
        UM_FUNCTIONAL_LINKS=functional_links,
        UM_CANDIDATE_COUNT=0,
        UM_CANDIDATE_MODE="sample",
    )
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 1.0, "length")
    for node in graph:
        graph.nodes[node].update(
            EPI=0.125,
            theta=math.tau * node / 5,
            nu_f=1.0,
            delta_nfr=0.0,
            dEPI=0.0,
            glyph_history=["AL"],
        )
        graph.nodes[node][ALIAS_SI[0]] = candidate_si if node == 4 else 0.8
    default_compute_delta_nfr(graph)
    return graph


def execute_coupling_cycle_birth(*, candidate_si=0.8, functional_links=True):
    """Run the declared default UM preparation once with prospective controls."""
    graph = prepared_coupling_path(
        candidate_si=candidate_si, functional_links=functional_links
    )
    before = _detached_stage_graph(graph)
    factors = resolve_runtime_operator_factors(None, Glyph.UM, graph.graph)
    # Freeze the independent analytic admission prediction before execution.
    # The compatibility kernel is linear in wrapped angular separation.
    push = factors["UM_theta_push"]
    max_gap = (2 + push) * math.pi / 5
    score = 1 - (2 + push) / 10
    threshold = DEFAULTS["UM_COMPAT_THRESHOLD"]
    assert max_gap < math.pi / 2
    assert score > threshold > score - 0.2
    prediction = propose_coupling_stage(
        before,
        (0,),
        factors,
        resolved_seed=17,
        node_offsets={node: node for node in graph},
    )
    admission = validate_candidate(graph, 0, Glyph.UM)
    # This is the actual atomic stage, with live grammar authoritative.
    # The separate pure proposal above is not executor-retained evidence.
    result = execute_coupling_stage(graph, Coupling(), (0,))
    raw = capture_non_epi_forcing(graph)
    raw_graph = _detached_stage_graph(graph)
    default_compute_delta_nfr(graph)
    refreshed = capture_non_epi_forcing(graph)
    return {
        "before": before,
        "graph": graph,
        "raw_graph": raw_graph,
        "raw": raw,
        "refreshed": refreshed,
        "factors": factors,
        "threshold": threshold,
        "score": score,
        "max_gap": max_gap,
        "prediction": prediction,
        "admission": admission,
        "result": result,
    }
