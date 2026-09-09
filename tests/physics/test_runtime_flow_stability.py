"""Focused tests for one-interval nodal-flow stability evidence."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.runtime_flow_stability import (
    capture_nodal_flow_state,
    certify_observed_nodal_flow_interval,
)


_RUNTIME = {
    "integrator_name": "DefaultIntegrator",
    "method": "euler",
    "substeps": 1,
    "gamma_is_none": True,
    "clipping_applied": False,
    "extended_dynamics_requested": False,
}


def _two_node_graph(epi, pressure, capacity=(1.0, 1.0), weight=1.0):
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=weight)
    for node, value, nu_f, delta_nfr in zip(
        graph,
        epi,
        capacity,
        pressure,
    ):
        graph.nodes[node].update(
            EPI=value,
            nu_f=nu_f,
            delta_nfr=delta_nfr,
        )
    return graph


def _certificate(
    left_graph,
    right_graph,
    duration,
    **runtime,
):
    metadata = dict(_RUNTIME)
    metadata.update(runtime)
    return certify_observed_nodal_flow_interval(
        capture_nodal_flow_state(left_graph),
        capture_nodal_flow_state(right_graph),
        duration=duration,
        **metadata,
    )


def test_exact_dyadic_interval_identifies_contracting_euler_map():
    left = _two_node_graph([1.0, -1.0], [-2.0, 2.0])
    right = _two_node_graph([0.5, -0.5], [-2.0, 2.0])

    result = _certificate(left, right, 0.25)

    assert result.nodes == ("a", "b")
    assert result.left_epi == (1.0, -1.0)
    assert result.right_epi == (0.5, -0.5)
    assert result.exact_duration == Fraction(1, 4)
    assert result.exact_nodal_equation_residual == (0, 0)
    assert result.exact_nodal_equation_realized
    assert result.pure_epi_diffusion_eligible
    assert result.runtime_euler_eligible
    assert result.binary64_euler_replay_matches
    assert result.binary64_runtime_interval_identified
    assert result.explicit_euler_map_identified
    assert result.exact_metric_weights == (1, 1)
    assert result.exact_left_disagreement_energy == 1
    assert result.exact_right_disagreement_energy == Fraction(1, 4)
    assert result.exact_observed_disagreement_energy_gain == Fraction(1, 4)
    assert result.exact_quotient_energy_gain_upper_bound == Fraction(1, 4)
    assert result.global_disagreement_contraction_certified
    assert not result.integrator_provenance_certified
    assert not result.future_or_repeated_schedule_stability_certified
    assert result.euler_map_abstention_reasons == ()


def test_interval_theorem_claim_fails_closed_after_proof_field_tamper():
    left = _two_node_graph([1.0, -1.0], [-2.0, 2.0])
    right = _two_node_graph([0.5, -0.5], [-2.0, 2.0])
    result = _certificate(left, right, 0.25)

    replaced = replace(
        result,
        exact_quotient_energy_gain_upper_bound=Fraction(0),
    )
    assert not replaced._proof_fields_are_intact()
    assert not replaced.global_disagreement_contraction_certified

    object.__setattr__(result, "exact_quotient_energy_gain_upper_bound", Fraction(0))
    assert not result._proof_fields_are_intact()
    assert not result.global_disagreement_contraction_certified


def test_stale_pressure_passes_nodal_balance_but_blocks_diffusion():
    left = _two_node_graph([1.0, -1.0], [-1.0, 1.0])
    right = _two_node_graph([0.75, -0.75], [-1.0, 1.0])

    result = _certificate(left, right, 0.25)

    assert result.exact_nodal_equation_realized
    assert result.binary64_euler_replay_matches
    assert not result.exact_pure_epi_pressure_realized
    assert not result.binary64_pure_epi_pressure_realized
    assert not result.pure_epi_diffusion_eligible
    assert result.binary64_runtime_interval_identified
    assert not result.explicit_euler_map_identified
    assert result.exact_quotient_energy_gain_upper_bound is None
    assert "exact_pure_epi_pressure_realized" in (
        result.euler_map_abstention_reasons
    )


def test_large_euler_step_is_realized_but_not_contracting():
    left = _two_node_graph([1.0, -1.0], [-2.0, 2.0])
    right = _two_node_graph([-1.5, 1.5], [-2.0, 2.0])

    result = _certificate(left, right, 1.25)

    assert result.explicit_euler_map_identified
    assert result.exact_observed_disagreement_energy_gain == Fraction(9, 4)
    assert result.exact_quotient_energy_gain_upper_bound == Fraction(9, 4)
    assert not result.global_disagreement_contraction_certified


@pytest.mark.parametrize("change", ["capacity", "pressure", "conductance"])
def test_changed_frozen_data_causes_explicit_abstention(change):
    left = _two_node_graph([1.0, -1.0], [-2.0, 2.0])
    capacity = (2.0, 1.0) if change == "capacity" else (1.0, 1.0)
    pressure = (-1.0, 1.0) if change == "pressure" else (-2.0, 2.0)
    weight = 2.0 if change == "conductance" else 1.0
    right = _two_node_graph(
        [0.5, -0.5],
        pressure,
        capacity=capacity,
        weight=weight,
    )

    result = _certificate(left, right, 0.25)

    expected = {
        "capacity": "capacity_unchanged",
        "pressure": "pressure_unchanged",
        "conductance": "fixed_conductance",
    }[change]
    assert expected in result.failed_diffusion_conditions
    assert not result.pure_epi_diffusion_eligible
    assert not result.explicit_euler_map_identified
    assert result.exact_quotient_energy_gain_upper_bound is None


@pytest.mark.parametrize(
    "runtime_change, expected",
    [
        ({"integrator_name": "CustomIntegrator"}, "default_integrator"),
        ({"method": "rk4"}, "euler_method"),
        ({"substeps": 2}, "one_substep"),
        ({"gamma_is_none": False}, "gamma_none"),
        ({"clipping_applied": True}, "clipping_inactive"),
        (
            {"extended_dynamics_requested": True},
            "extended_dynamics_not_requested",
        ),
    ],
)
def test_runtime_metadata_never_infers_an_unsupported_integrator(
    runtime_change,
    expected,
):
    left = _two_node_graph([1.0, -1.0], [-2.0, 2.0])
    right = _two_node_graph([0.5, -0.5], [-2.0, 2.0])

    result = _certificate(left, right, 0.25, **runtime_change)

    assert result.exact_nodal_equation_realized
    assert result.pure_epi_diffusion_eligible
    assert expected in result.failed_runtime_conditions
    assert not result.runtime_euler_eligible
    assert not result.explicit_euler_map_identified
    assert result.exact_quotient_energy_gain_upper_bound is None
    assert not result.integrator_provenance_certified


def test_binary64_replay_is_separate_from_exact_rational_map():
    epi = np.asarray([0.1, -0.1], dtype=float)
    capacity = np.asarray([0.3, 0.3], dtype=float)
    pressure = np.asarray([-0.2, 0.2], dtype=float)
    right_epi = np.add(
        epi,
        np.multiply(
            0.25,
            np.multiply(capacity, pressure),
        ),
    )
    left = _two_node_graph(epi, pressure, capacity)
    right = _two_node_graph(right_epi, pressure, capacity)

    result = _certificate(left, right, 0.25)

    assert result.binary64_pure_epi_pressure_realized
    assert result.binary64_euler_replay_matches
    assert result.binary64_runtime_interval_identified
    assert not result.exact_nodal_equation_realized
    assert not result.explicit_euler_map_identified
    assert result.exact_quotient_energy_gain_upper_bound is None
    assert result.exact_nodal_equation_residual != (0, 0)


def test_zero_duration_is_identity_without_strict_contraction():
    graph = _two_node_graph([1.0, -1.0], [-2.0, 2.0])

    result = _certificate(graph, graph.copy(), 0.0)

    assert result.explicit_euler_map_identified
    assert result.exact_quotient_energy_gain_upper_bound == 1
    assert result.exact_observed_disagreement_energy_gain == 1
    assert not result.global_disagreement_contraction_certified


@pytest.mark.parametrize("node_count", [0, 1])
def test_empty_or_singleton_support_abstains_without_division(node_count):
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    if node_count:
        graph.nodes[0].update(EPI=1.0, nu_f=1.0, DeltaNFR=0.0)

    result = _certificate(graph, graph.copy(), 0.0)

    assert not result.pure_epi_diffusion_eligible
    assert not result.explicit_euler_map_identified
    assert result.exact_metric_weights is None
    assert result.exact_quotient_energy_gain_upper_bound is None
    assert "at_least_two_nodes" in result.failed_diffusion_conditions


def test_capture_and_certificate_payloads_are_immutable_and_detached():
    graph = _two_node_graph([1.0, -1.0], [-2.0, 2.0])
    left = capture_nodal_flow_state(graph)
    graph.nodes["a"]["EPI"] = 99.0
    graph.edges["a", "b"]["weight"] = 7.0
    right = _two_node_graph([0.5, -0.5], [-2.0, 2.0])
    result = certify_observed_nodal_flow_interval(
        left,
        capture_nodal_flow_state(right),
        duration=0.25,
        **_RUNTIME,
    )

    assert left.epi == (1.0, -1.0)
    assert left.conductance == ((0, 1), (1, 0))
    with pytest.raises(FrozenInstanceError):
        left.epi = (2.0, -2.0)
    with pytest.raises(TypeError):
        left.conductance[0][0] = Fraction(3)
    with pytest.raises(FrozenInstanceError):
        result.scope = "changed"


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda graph: graph.nodes["a"].update(EPI=float("nan")), "EPI"),
        (lambda graph: graph.nodes["a"].update(EPI=True), "EPI"),
        (lambda graph: graph.nodes["a"].update(nu_f=True), "nu_f"),
        (
            lambda graph: graph.edges["a", "b"].update(weight=-1.0),
            "nonnegative",
        ),
    ],
)
def test_capture_rejects_malformed_binary64_state(mutation, message):
    graph = _two_node_graph([1.0, -1.0], [-2.0, 2.0])
    mutation(graph)

    with pytest.raises(ValueError, match=message):
        capture_nodal_flow_state(graph)


@pytest.mark.parametrize("duration", [True, -0.25, float("inf")])
def test_interval_rejects_invalid_duration(duration):
    graph = _two_node_graph([1.0, -1.0], [-2.0, 2.0])
    snapshot = capture_nodal_flow_state(graph)

    with pytest.raises(ValueError, match="duration"):
        certify_observed_nodal_flow_interval(
            snapshot,
            snapshot,
            duration=duration,
            **_RUNTIME,
        )
