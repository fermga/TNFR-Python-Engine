"""Regression tests for circular phase semantics across operator surfaces."""

from __future__ import annotations

import math
import warnings
from argparse import Namespace

import networkx as nx
import pytest

from tnfr.cli import execution as cli_execution
from tnfr.errors.contextual import PhaseError
from tnfr.operators.coherence import Coherence
from tnfr.operators.lifecycle import (
    LifecycleState,
    check_collapse_conditions,
    get_lifecycle_state,
)
from tnfr.operators.metrics_basic import dissonance_metrics
from tnfr.operators.metrics_network import coupling_metrics
from tnfr.operators.metrics_structural import (
    expansion_metrics,
    mutation_metrics,
    transition_metrics,
)
from tnfr.operators.network_analysis.source_detection import detect_emission_sources
from tnfr.operators.postconditions import OperatorContractViolation
from tnfr.operators.postconditions.mutation import verify_phase_transformed
from tnfr.operators.preconditions.resonance import (
    diagnose_resonance_readiness,
    validate_resonance_strict,
)
from tnfr.operators.remesh import StructuralIdentity
from tnfr.physics.cell import apply_membrane_flux
from tnfr.utils import angle_diff


def _phase_graph(phases: list[float]) -> nx.Graph:
    graph = nx.star_graph(len(phases) - 1)
    for node, phase in enumerate(phases):
        graph.nodes[node].update(
            EPI=1.0,
            vf=1.0,
            nu_f=1.0,
            DNFR=0.3,
            delta_nfr=0.3,
            dnfr=0.3,
            d2epi=0.0,
            theta=phase,
            phase=phase,
        )
    return graph


def test_lifecycle_uses_circular_neighbor_mean_at_phase_wrap() -> None:
    graph = _phase_graph([0.0, math.tau - 0.02, 0.02])

    assert get_lifecycle_state(graph, 0) is LifecycleState.PROPAGATION
    assert check_collapse_conditions(graph, 0) == (False, None)


def test_operator_metrics_report_shortest_arc_changes() -> None:
    graph = _phase_graph([0.01, math.tau - 0.01])
    before = math.tau - 0.01

    coupling = coupling_metrics(graph, 0, before)
    dissonance = dissonance_metrics(graph, 0, 0.1, before)
    graph.nodes[0]["epi_history"] = [0.5, 1.0]
    mutation = mutation_metrics(graph, 0, before, 1.0)
    transition = transition_metrics(graph, 0, 0.3, 1.0, before, 1.0)
    expansion = expansion_metrics(graph, 0, 0.5, 0.5)

    for metrics in (coupling, dissonance, mutation):
        assert metrics["theta_shift"] == pytest.approx(0.02)
    assert coupling["phase_alignment"] > 0.99
    assert transition["delta_theta"] == pytest.approx(0.02)
    assert expansion["phase_coherence_neighbors"] > 0.99


def test_coupling_dispersion_does_not_split_the_phase_wrap() -> None:
    graph = _phase_graph([0.0, math.tau - 0.02, 0.02])

    metrics = coupling_metrics(graph, 0, math.tau - 0.01)

    assert metrics["phase_alignment"] == pytest.approx(1.0)
    assert metrics["phase_dispersion"] == pytest.approx(0.02)


def test_coherence_locking_moves_along_the_shortest_arc() -> None:
    graph = _phase_graph([math.tau - 0.04, 0.02])
    before = graph.nodes[0]["theta"]

    Coherence()._apply_phase_locking(graph, 0, locking_coefficient=0.5)

    after = graph.nodes[0]["theta"]
    assert abs(angle_diff(after, 0.02)) < abs(angle_diff(before, 0.02))


def test_source_detection_distinguishes_wrap_equivalence_from_antiphase() -> None:
    graph = nx.complete_graph(3)
    for node, phase in enumerate((0.01, math.tau - 0.01, math.pi + 0.01)):
        graph.nodes[node].update(EPI=1.0, vf=1.0, theta=phase)

    sources = {
        node: compatibility
        for node, compatibility, _ in detect_emission_sources(graph, 0)
    }

    assert sources[1] > 0.99
    assert sources[2] == pytest.approx(0.0)


@pytest.mark.parametrize("override, warning", [(None, False), (1.0, True)])
def test_resonance_readiness_and_validator_share_phase_warning_policy(
    override: float | None, warning: bool
) -> None:
    graph = _phase_graph([0.0, 1.2])
    if override is not None:
        graph.graph["RA_MAX_PHASE_DIFF"] = override

    diagnostic = diagnose_resonance_readiness(graph, 0)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        validate_resonance_strict(graph, 0)

    assert diagnostic["checks"]["phase_alignment"] == (
        "warning" if warning else "passed"
    )
    assert (
        any("RA phase misalignment" in str(item.message) for item in emitted) is warning
    )


def test_structural_identity_rejects_antiphase_multiturn_representative() -> None:
    identity = StructuralIdentity(1.0, (0.5, 1.5), phase_pattern=0.0)

    assert identity.matches({"EPI": 1.0, "nu_f": 1.0, "theta": 4.0 * math.pi + 0.01})
    assert not identity.matches({"EPI": 1.0, "nu_f": 1.0, "theta": 3.0 * math.pi})


def test_mutation_postcondition_treats_complete_turn_as_same_phase() -> None:
    graph = nx.empty_graph(1)
    graph.nodes[0]["theta"] = math.tau

    with pytest.raises(OperatorContractViolation, match="Phase was not transformed"):
        verify_phase_transformed(graph, 0, 0.0)


def test_membrane_flux_uses_shortest_arc_gate() -> None:
    compatible = nx.Graph([(0, 1)])
    compatible.nodes[0].update(EPI=0.1, theta=0.01, nu_f=1.0, delta_nfr=0.0)
    compatible.nodes[1].update(EPI=0.8, theta=math.tau - 0.01, nu_f=1.0, delta_nfr=0.0)
    apply_membrane_flux(compatible, [1], [0], phase_threshold=0.1)

    incompatible = nx.Graph([(0, 1)])
    incompatible.nodes[0].update(EPI=0.1, theta=4.0 * math.pi, nu_f=1.0, delta_nfr=0.0)
    incompatible.nodes[1].update(EPI=0.8, theta=math.pi, nu_f=1.0, delta_nfr=0.0)
    apply_membrane_flux(incompatible, [1], [0], phase_threshold=0.1)

    assert compatible.nodes[0]["EPI"] > 0.1
    assert incompatible.nodes[0]["EPI"] == 0.1


def test_phase_error_reports_wrapped_distance_and_canonical_default() -> None:
    error = PhaseError("a", "b", 0.01, math.tau - 0.01)

    assert error.context["phase_difference"] == "0.020 rad"
    assert error.context["threshold"] == f"{math.pi / 2.0:.3f} rad"


@pytest.mark.parametrize(
    ("phase", "expected_status"),
    [(math.tau - 0.01, 0), (math.pi, 1)],
)
def test_cli_epi_validation_enforces_wrapped_u3_gate(
    monkeypatch: pytest.MonkeyPatch,
    phase: float,
    expected_status: int,
) -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0]["theta"] = 0.0
    graph.nodes[1]["theta"] = phase
    monkeypatch.setattr(cli_execution, "_run_cli_program", lambda _args: (0, graph))
    args = Namespace(
        check_coherence=False,
        check_frequency=False,
        check_phase=True,
        tolerance=1e-6,
    )

    assert cli_execution.cmd_epi_validate(args) == expected_status


@pytest.mark.parametrize(
    "gate", [True, -0.1, math.nan, math.inf, math.pi, "bad", 10**1000]
)
@pytest.mark.parametrize("with_edge", [True, False])
def test_cli_epi_validation_rejects_invalid_graph_phase_gate(
    monkeypatch, gate, with_edge
) -> None:
    graph = _phase_graph([0.0, 0.1]) if with_edge else nx.Graph()
    graph.graph["DELTA_PHI_MAX"] = gate
    monkeypatch.setattr(cli_execution, "_run_cli_program", lambda _args: (0, graph))
    args = Namespace(check_coherence=False, check_frequency=False, check_phase=True)

    assert cli_execution.cmd_epi_validate(args) == 1


@pytest.mark.parametrize("phase", [None, True, math.nan, math.inf, "bad"])
def test_cli_epi_validation_rejects_unavailable_edge_phase(monkeypatch, phase):
    graph = nx.Graph([(0, 1)])
    graph.nodes[0]["theta"] = 0.0
    if phase is not None:
        graph.nodes[1]["theta"] = phase
    monkeypatch.setattr(cli_execution, "_run_cli_program", lambda _args: (0, graph))
    args = Namespace(check_coherence=False, check_frequency=False, check_phase=True)

    assert cli_execution.cmd_epi_validate(args) == 1


def test_cli_validation_tolerance_cannot_weaken_configured_u3_gate(monkeypatch):
    graph = _phase_graph([0.0, 0.100001])
    graph.graph["DELTA_PHI_MAX"] = 0.1
    monkeypatch.setattr(cli_execution, "_run_cli_program", lambda _args: (0, graph))
    args = Namespace(
        check_coherence=False,
        check_frequency=False,
        check_phase=True,
        tolerance=1.0,
    )

    assert cli_execution.cmd_epi_validate(args) == 1


@pytest.mark.parametrize(
    "tolerance",
    [True, False, None, -1e-6, math.nan, math.inf, -math.inf, "bad", 10**1000],
)
def test_cli_rejects_invalid_tolerance_before_executing_program(monkeypatch, tolerance):
    def unexpected_execution(_args):
        pytest.fail("invalid tolerance must be rejected before executing a program")

    monkeypatch.setattr(cli_execution, "_run_cli_program", unexpected_execution)
    assert cli_execution.cmd_epi_validate(Namespace(tolerance=tolerance)) == 1
