"""Runtime-bound certificates for canonical operator-event flow intervals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from types import MethodType
from typing import Any

import networkx as nx
import pytest

from tnfr.dynamics.integrators import AbstractIntegrator, DefaultIntegrator
from tnfr.operators.event_runtime import (
    ExecutedNodalFlowInterval,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import build_operator_event_schedule


def _graph(
    *,
    epi: tuple[float, float] = (1.0, -1.0),
    pressure: tuple[float, float] = (-2.0, 2.0),
) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
    )
    for node, epi_value, pressure_value in zip(
        graph,
        epi,
        pressure,
    ):
        graph.nodes[node].update(
            EPI=epi_value,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=pressure_value,
        )
    return graph


def _schedule(duration: float):
    return build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(duration,),
    )


class HeldPressureEulerIntegrator(AbstractIntegrator):
    """Custom implementation that happens to match one Euler formula."""

    def __init__(self) -> None:
        self.calls = 0

    def integrate(
        self,
        graph,
        *,
        dt,
        t,
        method,
        n_jobs,
    ) -> None:
        self.calls += 1
        for node in graph:
            data = graph.nodes[node]
            data["EPI"] = float(data["EPI"]) + float(dt) * (
                float(data["nu_f"]) * float(data["delta_nfr"])
            )
        graph.graph["_t"] = float(t) + float(dt)


def test_opt_in_binds_one_exact_dyadic_default_euler_interval() -> None:
    graph = _graph()
    schedule = _schedule(0.25)

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_flow_certificates=True,
    )

    assert result.flow_certification_requested
    assert len(result.flow_interval_evidence) == 1
    evidence = result.flow_interval_evidence[0]
    assert isinstance(evidence, ExecutedNodalFlowInterval)
    assert evidence.interval is schedule.intervals[0]
    assert evidence.abstention_reason is None
    assert evidence.integrator_name == "DefaultIntegrator"
    assert evidence.integrator_provenance_certified
    assert evidence.resolved_method == "euler"
    assert evidence.resolved_substeps == 1
    assert evidence.gamma_is_none is True
    assert evidence.clipping_applied is False
    assert evidence.extended_dynamics_requested is False

    certificate = evidence.certificate
    assert certificate is not None
    assert certificate.exact_nodal_equation_realized
    assert certificate.binary64_euler_replay_matches
    assert certificate.binary64_runtime_interval_identified
    assert not certificate.integrator_provenance_certified
    assert certificate.explicit_euler_map_identified
    assert certificate.exact_quotient_energy_gain_upper_bound == Fraction(1, 4)
    assert certificate.global_disagreement_contraction_certified
    assert evidence._proof_fields_are_intact()
    assert evidence.runtime_bound_binary64_interval_identified
    assert evidence.runtime_bound_binary64_held_pressure_interval_identified
    assert evidence.runtime_bound_exact_affine_map_identified
    assert evidence.runtime_bound_global_disagreement_contraction_certified
    assert not evidence.solver_accuracy_certified
    assert not evidence.future_or_repeated_schedule_stability_certified
    assert result.all_positive_flow_intervals_binary64_identified is True
    assert (
        result.all_positive_flow_intervals_binary64_held_pressure_identified
        is True
    )
    assert result.all_positive_flow_intervals_exact_affine is True
    assert result.all_positive_flow_intervals_contracting is True
    assert not result.solver_accuracy_certified
    assert not result.future_or_repeated_schedule_stability_certified
    assert "flow_interval_evidence" not in graph.graph

    with pytest.raises(FrozenInstanceError):
        evidence.clipping_applied = True  # type: ignore[misc]


def test_runtime_flow_wrapper_fails_closed_after_provenance_replacement() -> None:
    result = execute_operator_event_schedule(
        _graph(),
        _schedule(0.25),
        include_flow_certificates=True,
    )
    evidence = result.flow_interval_evidence[0]

    forged = replace(evidence, resolved_substeps=2)

    assert not forged._proof_fields_are_intact()
    assert not forged.runtime_bound_binary64_interval_identified
    assert not forged.runtime_bound_binary64_held_pressure_interval_identified
    assert not forged.runtime_bound_exact_affine_map_identified


def test_directly_constructed_runtime_flow_wrapper_has_no_provenance_seal() -> None:
    result = execute_operator_event_schedule(
        _graph(),
        _schedule(0.25),
        include_flow_certificates=True,
    )
    evidence = result.flow_interval_evidence[0]
    fields = {
        name: getattr(evidence, name)
        for name in (
            "interval",
            "certificate",
            "abstention_reason",
            "integrator_name",
            "integrator_provenance_certified",
            "resolved_method",
            "resolved_substeps",
            "gamma_is_none",
            "clipping_applied",
            "extended_dynamics_requested",
        )
    }

    copied = ExecutedNodalFlowInterval(**fields)

    assert not copied._proof_fields_are_intact()
    assert not copied.integrator_provenance_certified
    assert not copied.runtime_bound_binary64_held_pressure_interval_identified
    object.__setattr__(copied, "integrator_provenance_certified", True)
    assert not copied.integrator_provenance_certified


def test_default_path_remains_uncertified_when_opt_in_is_disabled() -> None:
    result = execute_operator_event_schedule(_graph(), _schedule(0.25))

    assert not result.flow_certification_requested
    assert result.flow_interval_evidence == ()
    assert result.all_positive_flow_intervals_binary64_identified is None
    assert (
        result.all_positive_flow_intervals_binary64_held_pressure_identified
        is None
    )
    assert result.all_positive_flow_intervals_exact_affine is None
    assert result.all_positive_flow_intervals_contracting is None


def test_stale_pressure_keeps_nodal_replay_but_blocks_diffusion_map() -> None:
    graph = _graph(pressure=(-1.0, 1.0))

    result = execute_operator_event_schedule(
        graph,
        _schedule(0.25),
        include_flow_certificates=True,
    )

    evidence = result.flow_interval_evidence[0]
    certificate = evidence.certificate
    assert certificate is not None
    assert certificate.exact_nodal_equation_realized
    assert certificate.binary64_euler_replay_matches
    assert not certificate.exact_pure_epi_pressure_realized
    assert not certificate.binary64_pure_epi_pressure_realized
    assert evidence.runtime_bound_binary64_interval_identified
    assert not evidence.runtime_bound_exact_affine_map_identified
    assert "exact_pure_epi_pressure_realized" in (
        certificate.failed_diffusion_conditions
    )


def test_large_euler_interval_is_identified_without_false_contraction() -> None:
    result = execute_operator_event_schedule(
        _graph(),
        _schedule(1.5),
        include_flow_certificates=True,
    )

    evidence = result.flow_interval_evidence[0]
    certificate = evidence.certificate
    assert certificate is not None
    assert evidence.runtime_bound_binary64_interval_identified
    assert evidence.runtime_bound_exact_affine_map_identified
    assert not evidence.runtime_bound_global_disagreement_contraction_certified
    assert certificate.exact_quotient_energy_gain_upper_bound == Fraction(4, 1)
    assert result.all_positive_flow_intervals_contracting is False


@pytest.mark.parametrize("kind", ["custom", "subclass", "instance_override"])
def test_noncanonical_integrator_identity_cannot_supply_provenance(
    kind: str,
) -> None:
    graph = _graph()
    if kind == "custom":
        integrator: AbstractIntegrator = HeldPressureEulerIntegrator()
    elif kind == "subclass":

        class DefaultSubclass(DefaultIntegrator):
            pass

        integrator = DefaultSubclass()
    else:
        default = DefaultIntegrator()

        def integrate_override(
            self,
            live_graph,
            *,
            dt,
            t,
            method,
            n_jobs,
        ) -> None:
            DefaultIntegrator.integrate(
                self,
                live_graph,
                dt=dt,
                t=t,
                method=method,
                n_jobs=n_jobs,
            )

        default.integrate = MethodType(  # type: ignore[method-assign]
            integrate_override,
            default,
        )
        integrator = default
    graph.graph["integrator"] = integrator

    result = execute_operator_event_schedule(
        graph,
        _schedule(0.25),
        include_flow_certificates=True,
    )

    evidence = result.flow_interval_evidence[0]
    certificate = evidence.certificate
    assert certificate is not None
    assert certificate.exact_nodal_equation_realized
    assert not evidence.integrator_provenance_certified
    assert not evidence.runtime_bound_binary64_interval_identified
    assert not evidence.runtime_bound_exact_affine_map_identified
    assert evidence.resolved_method is None
    assert evidence.resolved_substeps is None
    assert evidence.gamma_is_none is None


def test_class_level_integrator_override_cannot_self_certify(monkeypatch) -> None:
    graph = _graph()

    def class_override(
        self,
        live_graph,
        *,
        dt,
        t,
        method,
        n_jobs,
    ) -> None:
        for node in live_graph:
            data = live_graph.nodes[node]
            data["EPI"] = float(data["EPI"]) + float(dt) * (
                float(data["nu_f"]) * float(data["delta_nfr"])
            )
        live_graph.graph["_t"] = float(t) + float(dt)

    monkeypatch.setattr(DefaultIntegrator, "integrate", class_override)
    graph.graph["integrator"] = DefaultIntegrator()

    result = execute_operator_event_schedule(
        graph,
        _schedule(0.25),
        include_flow_certificates=True,
    )

    evidence = result.flow_interval_evidence[0]
    assert evidence.certificate is not None
    assert evidence.certificate.exact_nodal_equation_realized
    assert not evidence.integrator_provenance_certified
    assert not evidence.runtime_bound_binary64_interval_identified
    assert not evidence.runtime_bound_exact_affine_map_identified


def test_live_gamma_configuration_blocks_runtime_promotion() -> None:
    graph = _graph()
    graph.graph["GAMMA"] = {
        "type": "harmonic",
        "beta": 0.25,
        "omega": 1.0,
        "phi": 1.5707963267948966,
    }
    graph.graph["_gamma_spec"] = {"type": "none"}

    result = execute_operator_event_schedule(
        graph,
        _schedule(0.25),
        include_flow_certificates=True,
    )

    evidence = result.flow_interval_evidence[0]
    assert evidence.gamma_is_none is False
    assert evidence.clipping_applied is None
    assert evidence.certificate is not None
    assert "gamma_none" in evidence.certificate.failed_runtime_conditions
    assert not evidence.runtime_bound_binary64_interval_identified


def test_clipping_intervention_is_observed_without_second_integration() -> None:
    graph = _graph(epi=(0.9, -0.9), pressure=(1.0, -1.0))
    graph.graph.update(EPI_MIN=-1.0, EPI_MAX=1.0)

    result = execute_operator_event_schedule(
        graph,
        _schedule(0.5),
        include_flow_certificates=True,
    )

    evidence = result.flow_interval_evidence[0]
    assert evidence.clipping_applied is True
    assert evidence.certificate is not None
    assert "clipping_inactive" in evidence.certificate.failed_runtime_conditions
    assert not evidence.runtime_bound_binary64_interval_identified
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == (1.0, -1.0)


@pytest.mark.parametrize(
    ("configuration", "keyword", "expected_method", "expected_substeps"),
    [
        ({}, "rk4", "rk4", 1),
        ({"DT_MIN": 0.25}, None, "euler", 2),
        ({"use_extended_dynamics": True}, None, "euler", 1),
    ],
)
def test_nonpromotable_default_modes_abstain_explicitly(
    configuration: dict[str, Any],
    keyword: str | None,
    expected_method: str,
    expected_substeps: int,
) -> None:
    graph = _graph()
    graph.graph.update(configuration)

    result = execute_operator_event_schedule(
        graph,
        _schedule(0.5),
        method=keyword,
        include_flow_certificates=True,
    )

    evidence = result.flow_interval_evidence[0]
    assert evidence.resolved_method == expected_method
    assert evidence.resolved_substeps == expected_substeps
    assert evidence.certificate is not None
    assert not evidence.runtime_bound_binary64_interval_identified
    held_pressure_expected = bool(
        expected_method == "euler"
        and expected_substeps >= 1
        and not configuration.get("use_extended_dynamics")
    )
    assert (
        evidence.runtime_bound_binary64_held_pressure_interval_identified
        is held_pressure_expected
    )
    assert (
        result.all_positive_flow_intervals_binary64_held_pressure_identified
        is held_pressure_expected
    )
    if expected_method != "euler":
        assert "euler_method" in evidence.certificate.failed_runtime_conditions
    if expected_substeps != 1:
        assert "one_substep" in evidence.certificate.failed_runtime_conditions
    if configuration.get("use_extended_dynamics"):
        assert evidence.extended_dynamics_requested
        assert "extended_dynamics_not_requested" in (
            evidence.certificate.failed_runtime_conditions
        )


def test_multistep_clipping_blocks_held_pressure_runtime_identification() -> None:
    graph = _graph(epi=(0.9, -0.9), pressure=(1.0, -1.0))
    graph.graph.update(DT_MIN=0.25, EPI_MIN=-1.0, EPI_MAX=1.0)

    result = execute_operator_event_schedule(
        graph,
        _schedule(0.5),
        include_flow_certificates=True,
    )

    evidence = result.flow_interval_evidence[0]
    assert evidence.resolved_substeps == 2
    assert evidence.clipping_applied is True
    assert evidence.certificate is not None
    assert evidence.certificate.binary64_held_pressure_replay_matches is False
    assert not evidence.runtime_bound_binary64_held_pressure_interval_identified
    assert (
        result.all_positive_flow_intervals_binary64_held_pressure_identified
        is False
    )
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == (1.0, -1.0)


def test_unsupported_capture_abstains_without_changing_runtime() -> None:
    graph = _graph()
    graph.edges[0, 1]["weight"] = -1.0

    result = execute_operator_event_schedule(
        graph,
        _schedule(0.25),
        include_flow_certificates=True,
    )

    evidence = result.flow_interval_evidence[0]
    assert evidence.certificate is None
    assert evidence.abstention_reason == "left_and_right_state_capture_failed"
    assert graph.graph["_t"] == 0.25
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == (0.5, -0.5)


def test_custom_integrator_is_invoked_exactly_once_with_certification() -> None:
    graph = _graph()
    integrator = HeldPressureEulerIntegrator()
    graph.graph["integrator"] = integrator

    execute_operator_event_schedule(
        graph,
        _schedule(0.25),
        include_flow_certificates=True,
    )

    assert integrator.calls == 1


def test_late_event_failure_publishes_no_partial_flow_evidence() -> None:
    graph = _graph()
    graph.graph["marker"] = "before"
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    callback_calls = 0

    def fail_refresh(live_graph: nx.Graph) -> None:
        nonlocal callback_calls
        callback_calls += 1
        live_graph.graph["marker"] = "during"
        raise RuntimeError("late refresh failure")

    graph.graph["compute_delta_nfr"] = fail_refresh
    schedule = build_operator_event_schedule(
        ("emission", "coupling", "coherence", "silence"),
        start_time=0.0,
        flow_durations=(0.25, 0.0, 0.0, 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="late refresh failure"):
        execute_operator_event_schedule(
            graph,
            schedule,
            include_flow_certificates=True,
        )

    assert callback_calls == 0
    assert graph.graph["marker"] == "before"
    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph["_t"] == 0.0
    assert "hybrid_event_log" not in graph.graph
    assert "flow_interval_evidence" not in graph.graph


def test_zero_positive_intervals_have_vacuous_requested_aggregates() -> None:
    result = execute_operator_event_schedule(
        _graph(),
        _schedule(0.0),
        include_flow_certificates=True,
    )

    assert result.flow_interval_evidence == ()
    assert result.all_positive_flow_intervals_binary64_identified is True
    assert (
        result.all_positive_flow_intervals_binary64_held_pressure_identified
        is True
    )
    assert result.all_positive_flow_intervals_exact_affine is True
    assert result.all_positive_flow_intervals_contracting is True


def test_include_flow_certificates_requires_a_strict_bool() -> None:
    graph = _graph()

    with pytest.raises(TypeError, match="include_flow_certificates"):
        execute_operator_event_schedule(
            graph,
            _schedule(0.25),
            include_flow_certificates=1,  # type: ignore[arg-type]
        )

    assert graph.graph["_t"] == 0.0
    assert all(graph.nodes[node]["EPI"] in (1.0, -1.0) for node in graph)

def test_unexpected_certificate_failure_rolls_back_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tnfr.physics.runtime_flow_stability as flow_stability

    graph = _graph()
    before_graph = deepcopy(graph.graph)
    before_nodes = deepcopy(dict(graph.nodes(data=True)))

    def fail_capture(_graph: nx.Graph) -> object:
        raise RuntimeError("certificate implementation failed")

    monkeypatch.setattr(
        flow_stability,
        "capture_nodal_flow_state",
        fail_capture,
    )

    with pytest.raises(RuntimeError, match="certificate implementation failed"):
        execute_operator_event_schedule(
            graph,
            _schedule(0.25),
            include_flow_certificates=True,
        )

    assert graph.graph == before_graph
    assert dict(graph.nodes(data=True)) == before_nodes
    assert "flow_interval_evidence" not in graph.graph



def test_forced_signed_zero_interval_mutation_invalidates_runtime_wrapper() -> None:
    result = execute_operator_event_schedule(
        _graph(epi=(0.0, 0.0), pressure=(0.0, 0.0)),
        _schedule(1.0),
        include_flow_certificates=True,
    )
    evidence = result.flow_interval_evidence[0]
    assert evidence.interval.start_time.hex() == 0.0.hex()
    assert evidence._proof_fields_are_intact()

    object.__setattr__(evidence.interval, "start_time", -0.0)

    assert evidence.interval.start_time.hex() == (-0.0).hex()
    assert not evidence._proof_fields_are_intact()
    assert not evidence.runtime_bound_binary64_interval_identified
    assert not evidence.runtime_bound_binary64_held_pressure_interval_identified


def test_mutable_identity_node_invalidates_nested_and_wrapper_proofs() -> None:
    calls = {"repr": 0}

    class MutableIdentityNode:
        def __init__(self, label: str) -> None:
            self.label = label

        def __repr__(self) -> str:
            calls["repr"] += 1
            return f"MutableIdentityNode({self.label!r})"

    left = MutableIdentityNode("left")
    right = MutableIdentityNode("right")
    graph = nx.Graph()
    graph.add_edge(left, right)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
    )
    for node, epi in ((left, 0.0), (right, 1.0)):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.25,
        )
    evidence = execute_operator_event_schedule(
        graph,
        _schedule(0.25),
        include_flow_certificates=True,
    ).flow_interval_evidence[0]
    certificate = evidence.certificate
    assert certificate is not None
    assert certificate._proof_fields_are_intact()
    assert evidence._proof_fields_are_intact()
    calls_before_validation = dict(calls)

    assert certificate._proof_fields_are_intact()
    assert evidence._proof_fields_are_intact()
    assert calls == calls_before_validation
    stamp_before = certificate._proof_stamp
    left.label = "mutated-left"

    assert certificate._proof_stamp == stamp_before
    assert not certificate._proof_fields_are_intact()
    assert not certificate.binary64_held_pressure_runtime_identified
    assert not evidence._proof_fields_are_intact()
    assert not evidence.runtime_bound_binary64_held_pressure_interval_identified
    assert calls == calls_before_validation
