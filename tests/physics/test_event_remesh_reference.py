"""Exact P2 three-mesh event/REMESH reference-family contracts."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from fractions import Fraction
import math

import networkx as nx
import pytest

import tnfr.physics.event_remesh_reference as reference_module
import tnfr.physics.runtime_remesh_history_stability as bridge_module
from tnfr.errors import TNFRValueError
from tnfr.operators.event_remesh_runtime import execute_event_remesh_cycle
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)
from tnfr.physics.event_remesh_reference import (
    P2EventRemeshReferenceFamilyObservation,
    observe_p2_event_remesh_reference_family,
)
from tnfr.physics.reversible_eigenmode_reference import (
    certify_reversible_single_eigenmode_euler_reference,
)


def _refresh_p2_pressure(graph: nx.Graph) -> None:
    left = float(graph.nodes[0]["EPI"])
    right = float(graph.nodes[1]["EPI"])
    graph.nodes[0]["delta_nfr"] = right - left
    graph.nodes[1]["delta_nfr"] = left - right


def _cycle(
    durations: tuple[float, ...],
    *,
    clip_mode: str = "hard",
    alpha: float = 0.5,
    epi_min: float = -10.0,
    epi_max: float = 10.0,
):
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        RANDOM_SEED=23,
        GAMMA={"type": "none"},
        _gamma_spec={"type": "none"},
        DNFR_WEIGHTS={
            "phase": 0.0,
            "epi": 1.0,
            "vf": 0.0,
            "topo": 0.0,
        },
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=alpha,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=epi_min,
        EPI_MAX=epi_max,
        CLIP_MODE=clip_mode,
        compute_delta_nfr=_refresh_p2_pressure,
    )
    for node, epi in enumerate((1.0, -1.0)):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.graph["_epi_hist"] = deque(
        [{0: 1.0, 1: -1.0}],
        maxlen=64,
    )
    _refresh_p2_pressure(graph)
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.5,),
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0],
        durations,
    )
    return execute_event_remesh_cycle(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
        include_stage_certificates=True,
        suppress_birth_warnings=True,
    )


@pytest.fixture(scope="module")
def cycles():
    return (
        _cycle((0.25, 0.25)),
        _cycle((0.125,) * 4),
        _cycle((0.0625,) * 8),
    )


@pytest.fixture(scope="module")
def observation(cycles) -> P2EventRemeshReferenceFamilyObservation:
    return observe_p2_event_remesh_reference_family(*cycles)


def test_reference_family_certifies_the_exact_p2_problem(
    observation,
    cycles,
) -> None:
    result = observation

    assert type(result) is P2EventRemeshReferenceFamilyObservation
    assert result.reference_family_certified
    assert all(passed for _, passed in result.conditions)
    assert result.nodes == (0, 1)
    assert result.exact_initial_field == (Fraction(1), Fraction(-1))
    assert result.exact_mean == 0
    assert result.exact_initial_amplitude == 1
    assert result.exact_nu_f == 1
    assert result.exact_lambda == 2
    assert result.exact_total_duration == Fraction(1, 2)
    assert result.exact_alpha == Fraction(1, 2)
    assert result.exact_beta == Fraction(1, 4)
    assert tuple(mesh.mesh_name for mesh in result.meshes) == (
        "coarse",
        "intermediate",
        "fine",
    )
    assert all(
        mesh.cycle_result is cycle
        and mesh.runtime_bridge.cycle_result is cycle
        for mesh, cycle in zip(result.meshes, cycles, strict=True)
    )


def test_exact_euler_products_and_bounds_improve_strictly(observation) -> None:
    result = observation

    assert result.exact_euler_factors == (
        Fraction(1, 4),
        Fraction(81, 256),
        Fraction(5_764_801, 16_777_216),
    )
    assert result.exact_euler_factor_improvements == (
        Fraction(17, 256),
        Fraction(456_385, 16_777_216),
    )
    assert result.exact_quadratic_factor_error_upper_bounds == (
        Fraction(1, 4),
        Fraction(1, 8),
        Fraction(1, 16),
    )
    assert result.exact_quadratic_bound_improvements == (
        Fraction(1, 8),
        Fraction(1, 16),
    )
    assert result.strict_pre_remesh_error_improvement
    assert result.strict_ideal_post_remesh_error_improvement
    assert float(result.exact_continuous_factor_lower_bound) == pytest.approx(
        math.exp(-1.0)
    )
    assert float(result.exact_continuous_factor_upper_bound) == pytest.approx(
        math.exp(-1.0)
    )


def test_p2_fields_coincide_with_the_general_eigenmode_kernel(observation) -> None:
    result = observation
    conductance = (
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(0)),
    )
    kernel = certify_reversible_single_eigenmode_euler_reference(
        conductance,
        nu_f=(result.exact_nu_f, result.exact_nu_f),
        initial_epi=result.exact_initial_field,
        partitions=tuple(
            mesh.exact_segment_durations for mesh in result.meshes
        ),
    )

    assert result.exact_mean == kernel.exact_weighted_mean
    assert result.exact_lambda == kernel.exact_mode_eigenvalue
    assert result.exact_total_duration == kernel.exact_total_duration
    assert result.exact_euler_factors == kernel.exact_euler_factors
    assert (
        result.exact_euler_factor_improvements
        == kernel.exact_euler_factor_improvements
    )
    assert (
        result.exact_quadratic_factor_error_upper_bounds
        == kernel.exact_quadratic_factor_error_upper_bounds
    )
    assert (
        result.exact_quadratic_bound_improvements
        == kernel.exact_quadratic_bound_improvements
    )
    for index, mesh in enumerate(result.meshes):
        assert (
            mesh.exact_factor_error_lower_bound
            == kernel.exact_factor_error_lower_bounds[index]
        )
        assert (
            mesh.exact_factor_error_upper_bound
            == kernel.exact_factor_error_upper_bounds[index]
        )


def test_each_mesh_proves_the_two_error_inequalities(observation) -> None:
    for mesh in observation.meshes:
        assert mesh.exact_factor_error_lower_bound >= 0
        assert (
            mesh.exact_factor_error_upper_bound
            <= mesh.exact_quadratic_factor_error_upper_bound
            <= mesh.exact_hmax_factor_error_upper_bound
        )
        assert (
            mesh.exact_observed_pre_remesh_field
            == mesh.exact_euler_reference_pre_remesh_field
        )
        assert mesh.exact_factor_error_symbolic_coefficients == (
            -mesh.exact_euler_factor,
            Fraction(1),
        )


def test_remesh_scales_ideal_error_and_adds_the_runtime_residual(
    observation,
) -> None:
    for mesh in observation.meshes:
        beta = mesh.exact_beta
        assert mesh.exact_ideal_post_remesh_error_symbolic_coefficients == tuple(
            beta * item
            for item in mesh.exact_factor_error_symbolic_coefficients
        )
        assert mesh.exact_ideal_post_remesh_error_lower_bound == (
            beta * mesh.exact_pre_remesh_error_lower_bound
        )
        assert mesh.exact_ideal_post_remesh_error_upper_bound == (
            beta * mesh.exact_pre_remesh_error_upper_bound
        )
        assert mesh.exact_signed_total_residual == tuple(
            rounding + clipping
            for rounding, clipping in zip(
                mesh.exact_signed_rounding_residual,
                mesh.exact_signed_clipping_residual,
                strict=True,
            )
        )
        assert mesh.exact_runtime_post_remesh_error_upper_bound == (
            mesh.exact_ideal_post_remesh_error_upper_bound
            + mesh.exact_total_residual_linf
        )
        assert mesh.exact_total_residual_linf == 0


def test_scope_keeps_generic_and_future_claims_false(observation) -> None:
    result = observation

    assert "effective two-node path" in result.scope
    assert not result.binary64_asymptotic_convergence_certified
    assert not result.arbitrary_glyph_or_mixed_mode_certified
    assert not result.soft_clipping_certified
    assert not result.changing_support_or_metric_certified
    assert not result.solver_order_certified
    assert not result.generic_mesh_convergence_certified
    assert not result.repeated_runtime_stability_certified
    assert not result.future_stability_certified


def test_direct_and_privately_resealed_derivative_tampering_fails_closed(
    observation,
) -> None:
    directly_changed = replace(
        observation,
        exact_euler_factor_improvements=(Fraction(9), Fraction(9)),
    )
    assert not directly_changed.reference_family_certified

    forged = replace(
        observation,
        exact_beta=Fraction(7),
        _proof_stamp=(),
    )
    resealed = reference_module._seal_family(forged)
    assert not resealed.reference_family_certified
    assert resealed.failed_conditions == (
        "p2_event_remesh_reference_family_proof_fields_intact",
    )


def test_privately_resealed_nested_bridge_cannot_be_promoted(observation) -> None:
    original_mesh = observation.meshes[0]
    forged_bridge = bridge_module._seal(
        replace(
            original_mesh.runtime_bridge,
            exact_max_abs_total_residual=Fraction(99),
            _proof_stamp=(),
        )
    )
    forged_mesh = replace(
        original_mesh,
        runtime_bridge=forged_bridge,
        _proof_stamp=(),
    )
    values = reference_module._object_values(
        forged_mesh,
        reference_module.P2EventRemeshMeshReferenceObservation,
    )
    object.__setattr__(
        forged_mesh,
        "_proof_stamp",
        reference_module._stamp_from_values(
            reference_module._MESH_PROOF_VERSION,
            reference_module._MESH_FIELD_NAMES,
            values,
        ),
    )
    forged_family = reference_module._seal_family(
        replace(
            observation,
            meshes=(forged_mesh, *observation.meshes[1:]),
            _proof_stamp=(),
        )
    )

    assert not forged_mesh.mesh_reference_certified
    assert not forged_family.reference_family_certified


def test_in_place_nested_proof_tampering_fails_closed(observation) -> None:
    refinement = observation.refinement
    original_support_flag = refinement.supports_equal_across_meshes
    object.__setattr__(
        refinement,
        "supports_equal_across_meshes",
        not original_support_flag,
    )
    try:
        assert not refinement._proof_fields_are_intact()
        assert not observation.reference_family_certified
    finally:
        object.__setattr__(
            refinement,
            "supports_equal_across_meshes",
            original_support_flag,
        )

    mesh = observation.meshes[0]
    original_residual = mesh.exact_signed_rounding_residual
    object.__setattr__(
        mesh,
        "exact_signed_rounding_residual",
        (Fraction(99), Fraction(99)),
    )
    try:
        assert not mesh.mesh_reference_certified
        assert not observation.reference_family_certified
    finally:
        object.__setattr__(
            mesh,
            "exact_signed_rounding_residual",
            original_residual,
        )

    assert observation.reference_family_certified


def test_privately_resealed_refinement_tampering_fails_closed(observation) -> None:
    refinement = observation.refinement
    forged_refinement = reference_module._refinement_module._seal(
        replace(
            refinement,
            supports_equal_across_meshes=False,
            _proof_stamp=(),
        ),
        type(refinement),
        reference_module._refinement_module._THREE_MESH_PROOF_VERSION,
    )
    forged_family = reference_module._seal_family(
        replace(
            observation,
            refinement=forged_refinement,
            _proof_stamp=(),
        )
    )

    assert forged_refinement._proof_fields_are_intact()
    assert not forged_family.reference_family_certified


def test_soft_clipping_reference_family_is_rejected() -> None:
    soft_cycles = (
        _cycle((0.25, 0.25), clip_mode="soft"),
        _cycle((0.125,) * 4, clip_mode="soft"),
        _cycle((0.0625,) * 8, clip_mode="soft"),
    )

    with pytest.raises(TNFRValueError, match="requires hard clipping"):
        observe_p2_event_remesh_reference_family(*soft_cycles)


def test_active_hard_clipping_residual_stays_inside_runtime_bound() -> None:
    clipped_cycles = tuple(
        _cycle(
            durations,
            alpha=0.9,
            epi_min=-0.95,
            epi_max=0.95,
        )
        for durations in (
            (0.25, 0.25),
            (0.125,) * 4,
            (0.0625,) * 8,
        )
    )
    result = observe_p2_event_remesh_reference_family(*clipped_cycles)

    assert result.reference_family_certified
    for mesh in result.meshes:
        assert mesh.exact_signed_clipping_residual != (Fraction(0),) * 2
        assert mesh.exact_total_residual_linf > 0
        assert mesh.exact_runtime_post_remesh_error_upper_bound == (
            mesh.exact_ideal_post_remesh_error_upper_bound
            + mesh.exact_total_residual_linf
        )


def test_p2_module_does_not_duplicate_the_general_exponential_kernel() -> None:
    assert not hasattr(reference_module, "_negative_exp_bounds")


def test_p2_observer_delegates_once_to_runtime_reference_per_family(
    cycles,
    monkeypatch,
) -> None:
    original = (
        reference_module.observe_executed_reversible_single_eigenmode_euler_reference
    )
    calls = []

    def tracked(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(
        reference_module,
        "observe_executed_reversible_single_eigenmode_euler_reference",
        tracked,
    )
    result = observe_p2_event_remesh_reference_family(*cycles)

    assert len(calls) == 1
    partitions = calls[0][0][0]
    runtime_reference = calls[0][2]
    assert tuple(
        tuple(segment.exact_duration for segment in item.partition.segments)
        for item in partitions
    ) == tuple(
        mesh.exact_segment_durations for mesh in result.meshes
    )
    assert runtime_reference.reference_certificate.exact_partitions == tuple(
        mesh.exact_segment_durations for mesh in result.meshes
    )
    assert all(
        row.all_segment_exact_affine_maps_identified
        and all(
            not any(residual)
            for residual in (
                *row.exact_pressure_realization_residuals,
                *row.exact_held_input_execution_residuals,
                *row.exact_local_runtime_defects,
                row.exact_endpoint_runtime_defect,
            )
        )
        for row in runtime_reference.partition_observations
    )

    calls.clear()
    assert result.reference_family_certified
    assert len(calls) == 1


def test_wrong_public_input_type_is_rejected(cycles) -> None:
    with pytest.raises(TypeError, match="cycle results"):
        observe_p2_event_remesh_reference_family(
            object(),  # type: ignore[arg-type]
            cycles[1],
            cycles[2],
        )
