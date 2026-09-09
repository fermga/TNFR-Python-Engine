"""Held-pressure binary64 evidence for internal nodal-flow partitions."""

from __future__ import annotations

from dataclasses import fields, replace
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.integrators import prepare_integration_params
from tnfr.physics.runtime_flow_stability import (
    capture_nodal_flow_state,
    certify_observed_nodal_flow_interval,
)


_RUNTIME = {
    "integrator_name": "DefaultIntegrator",
    "method": "euler",
    "gamma_is_none": True,
    "clipping_applied": False,
    "extended_dynamics_requested": False,
}


def _graph(
    epi: tuple[float, float],
    pressure: tuple[float, float],
    *,
    capacity: tuple[float, float] = (1.0, 1.0),
) -> nx.Graph:
    graph = nx.path_graph(2)
    for node, value, nu_f, delta_nfr in zip(
        graph,
        epi,
        capacity,
        pressure,
        strict=True,
    ):
        graph.nodes[node].update(
            EPI=value,
            nu_f=nu_f,
            delta_nfr=delta_nfr,
        )
    return graph


def _certificate(
    left: nx.Graph,
    right: nx.Graph,
    *,
    duration: float,
    substeps: object,
):
    return certify_observed_nodal_flow_interval(
        capture_nodal_flow_state(left),
        capture_nodal_flow_state(right),
        duration=duration,
        substeps=substeps,
        **_RUNTIME,
    )


def _sequential_endpoint(
    epi: tuple[float, float],
    pressure: tuple[float, float],
    capacity: tuple[float, float],
    duration: float,
    substeps: int,
) -> tuple[float, float]:
    base = np.multiply(
        np.asarray(capacity, dtype=float),
        np.asarray(pressure, dtype=float),
    )
    base = np.add(base, np.zeros_like(base))
    increment = np.multiply(duration / substeps, base)
    state = np.asarray(epi, dtype=float)
    for _ in range(substeps):
        state = np.add(state, increment)
    return tuple(float(value) for value in state)


def test_new_partition_fields_follow_the_previous_public_positional_surface() -> None:
    result = _certificate(
        _graph((1.0, -1.0), (-2.0, 2.0)),
        _graph((0.0, 0.0), (-2.0, 2.0)),
        duration=0.5,
        substeps=2,
    )
    names = tuple(item.name for item in fields(type(result)))

    assert names.index("scope") < names.index("binary64_substep_duration")
    assert names[-1] == "_proof_stamp"


def test_dyadic_multistep_replay_is_identified_without_euler_map_promotion() -> None:
    left = _graph((1.0, -1.0), (-2.0, 2.0))
    right = _graph((0.0, 0.0), (-2.0, 2.0))

    result = _certificate(left, right, duration=0.5, substeps=2)

    assert result.binary64_substep_duration == 0.25
    assert result.exact_binary64_substep_duration == Fraction(1, 4)
    assert result.exact_binary64_substep_duration_sum == Fraction(1, 2)
    assert result.exact_substep_duration_sum_matches_interval
    assert result.binary64_held_pressure_replay == (0.0, 0.0)
    assert result.exact_binary64_held_pressure_replay_residual == (0, 0)
    assert result.binary64_held_pressure_replay_matches
    assert result.binary64_held_pressure_runtime_identified
    assert result.failed_held_pressure_runtime_conditions == ()

    assert not result.runtime_euler_eligible
    assert not result.binary64_runtime_interval_identified
    assert not result.explicit_euler_map_identified
    assert result.exact_explicit_euler_map is None
    assert result.exact_quotient_energy_gain_upper_bound is None
    assert "one_substep" in result.failed_runtime_conditions


def test_binary64_partition_endpoints_can_diverge_under_the_same_held_rate() -> None:
    epi = (1.0e16, 0.0)
    pressure = (1.0, 0.0)
    left = _graph(epi, pressure)
    one_step_endpoint = _sequential_endpoint(
        epi,
        pressure,
        (1.0, 1.0),
        2.0,
        1,
    )
    two_step_endpoint = _sequential_endpoint(
        epi,
        pressure,
        (1.0, 1.0),
        2.0,
        2,
    )

    one_step = _certificate(
        left,
        _graph(one_step_endpoint, pressure),
        duration=2.0,
        substeps=1,
    )
    two_steps = _certificate(
        left,
        _graph(two_step_endpoint, pressure),
        duration=2.0,
        substeps=2,
    )

    assert one_step_endpoint == (1.0000000000000002e16, 0.0)
    assert two_step_endpoint == epi
    assert one_step.binary64_held_pressure_runtime_identified
    assert two_steps.binary64_held_pressure_runtime_identified
    assert one_step.binary64_held_pressure_replay != (
        two_steps.binary64_held_pressure_replay
    )
    assert not two_steps.exact_nodal_equation_realized
    assert not two_steps.explicit_euler_map_identified


def test_nondyadic_thirds_expose_exact_substep_duration_sum_mismatch() -> None:
    epi = (0.1, -0.1)
    pressure = (0.2, -0.2)
    endpoint = _sequential_endpoint(
        epi,
        pressure,
        (1.0, 1.0),
        0.3,
        3,
    )
    result = _certificate(
        _graph(epi, pressure),
        _graph(endpoint, pressure),
        duration=0.3,
        substeps=3,
    )

    assert result.binary64_substep_duration == 0.09999999999999999
    assert result.exact_binary64_substep_duration_sum != result.exact_duration
    assert not result.exact_substep_duration_sum_matches_interval
    assert result.binary64_held_pressure_replay_matches
    assert not result.binary64_held_pressure_runtime_identified
    assert result.failed_held_pressure_runtime_conditions == (
        "substep_duration_sum_matches_interval",
    )


def test_dt_min_float_floor_does_not_request_three_decimal_substeps() -> None:
    graph = _graph((0.0, 0.0), (0.0, 0.0))
    graph.graph["DT_MIN"] = 0.1

    step, count, _, method = prepare_integration_params(graph, dt=0.3)

    assert 0.3 / 0.1 < 3.0
    assert count == 2
    assert step == 0.15
    assert method == "euler"


@pytest.mark.parametrize(
    "substeps",
    [None, True, np.bool_(True), 0, -1, 1.5],
)
def test_invalid_or_missing_substep_metadata_causes_explicit_abstention(
    substeps: object,
) -> None:
    graph = _graph((1.0, -1.0), (-2.0, 2.0))

    result = _certificate(
        graph,
        graph.copy(),
        duration=0.5,
        substeps=substeps,
    )

    assert result.binary64_substep_duration is None
    assert result.exact_binary64_substep_duration is None
    assert result.exact_binary64_substep_duration_sum is None
    assert not result.exact_substep_duration_sum_matches_interval
    assert result.binary64_held_pressure_replay is None
    assert result.exact_binary64_held_pressure_replay_residual is None
    assert not result.binary64_held_pressure_replay_matches
    assert not result.binary64_held_pressure_runtime_identified
    assert "positive_substeps" in (
        result.failed_held_pressure_runtime_conditions
    )


def test_sequential_replay_preserves_actual_gamma_none_signed_zero_bits() -> None:
    left = _graph((-0.0, 0.0), (-0.0, 0.0))
    right = _graph((0.0, 0.0), (-0.0, 0.0))

    result = _certificate(left, right, duration=1.0, substeps=2)

    assert result.binary64_held_pressure_replay is not None
    assert result.binary64_held_pressure_replay[0].hex() == 0.0.hex()
    assert result.binary64_held_pressure_replay_matches
    assert result.binary64_held_pressure_runtime_identified

    forged = replace(
        result,
        binary64_held_pressure_replay=(-0.0, 0.0),
    )
    assert forged.binary64_held_pressure_replay[0].hex() == (-0.0).hex()
    assert not forged._proof_fields_are_intact()
    assert not forged.binary64_held_pressure_runtime_identified


def test_held_pressure_identification_fails_closed_after_proof_tamper() -> None:
    result = _certificate(
        _graph((1.0, -1.0), (-2.0, 2.0)),
        _graph((0.0, 0.0), (-2.0, 2.0)),
        duration=0.5,
        substeps=2,
    )
    assert result.binary64_held_pressure_runtime_identified

    replaced = replace(
        result,
        exact_binary64_substep_duration_sum=Fraction(0),
    )

    assert not replaced._proof_fields_are_intact()
    assert not replaced.binary64_held_pressure_runtime_identified
