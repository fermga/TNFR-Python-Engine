"""Exact temporal Lyapunov certificate for repeated uniform REMESH."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

import tnfr.physics.remesh_history_stability as stability_module
from tnfr.errors import TNFRValueError
from tnfr.physics.remesh_history_stability import (
    UniformRemeshHistoryStabilityCertificate,
    UniformRemeshHistoryTransitionObservation,
    certify_uniform_remesh_history_stability,
    observe_uniform_remesh_history_transition,
)


def test_public_physics_facade_reexports_the_stability_api() -> None:
    import tnfr.physics as physics

    public_names = {
        "UniformRemeshHistoryStabilityCertificate",
        "UniformRemeshHistoryTransitionObservation",
        "certify_uniform_remesh_history_stability",
        "observe_uniform_remesh_history_transition",
    }

    assert public_names <= set(physics.__all__)

    assert (
        physics.UniformRemeshHistoryStabilityCertificate
        is UniformRemeshHistoryStabilityCertificate
    )
    assert (
        physics.UniformRemeshHistoryTransitionObservation
        is UniformRemeshHistoryTransitionObservation
    )
    assert (
        physics.certify_uniform_remesh_history_stability
        is certify_uniform_remesh_history_stability
    )
    assert (
        physics.observe_uniform_remesh_history_transition
        is observe_uniform_remesh_history_transition
    )


def _strict_certificate() -> UniformRemeshHistoryStabilityCertificate:
    return certify_uniform_remesh_history_stability(
        alpha=Fraction(1, 2),
        tau_local=1,
        tau_global=2,
    )


def _left_action(vector, matrix):
    return tuple(
        sum(
            (vector[row] * matrix[row][column] for row in range(len(matrix))),
            Fraction(0),
        )
        for column in range(len(matrix))
    )


def test_exact_companion_and_stationary_distribution() -> None:
    certificate = _strict_certificate()

    assert type(certificate) is UniformRemeshHistoryStabilityCertificate
    assert certificate.stability_certificate_certified
    assert certificate.beta == Fraction(1, 4)
    assert certificate.gamma == Fraction(1, 4)
    assert certificate.delta == Fraction(1, 2)
    assert certificate.combined_delay_coefficients == (
        (0, Fraction(1, 4)),
        (1, Fraction(1, 4)),
        (2, Fraction(1, 2)),
    )
    assert certificate.active_max_delay == 2
    assert certificate.companion_matrix == (
        (Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
    )
    assert certificate.stationary_denominator == Fraction(9, 4)
    assert certificate.stationary_distribution == (
        Fraction(4, 9),
        Fraction(1, 3),
        Fraction(2, 9),
    )
    assert all(
        sum(row, Fraction(0)) == 1
        for row in certificate.companion_matrix
    )
    assert _left_action(
        certificate.stationary_distribution,
        certificate.companion_matrix,
    ) == certificate.stationary_distribution
    assert certificate.companion_row_stochastic_certified
    assert certificate.stationary_distribution_certified
    assert certificate.jensen_lyapunov_nonincrease_certified


def test_exact_dissipation_identity_and_monotonicity() -> None:
    certificate = _strict_certificate()
    observation = observe_uniform_remesh_history_transition(
        certificate,
        (
            (Fraction(3), Fraction(-1)),
            (Fraction(0), Fraction(2)),
            (Fraction(1), Fraction(4)),
        ),
        (Fraction(2), Fraction(1)),
        nodes=("left", "right"),
    )

    assert type(observation) is UniformRemeshHistoryTransitionObservation
    assert observation.transition_observation_certified
    assert observation.exact_next_field == (Fraction(5, 4), Fraction(9, 4))
    assert observation.exact_energy_drop > 0
    assert (
        observation.exact_augmented_energy_before
        - observation.exact_augmented_energy_after
        == observation.exact_energy_drop
        == observation.exact_jensen_dissipation
    )

    coefficient_by_delay = dict(certificate.combined_delay_coefficients)
    exact_rhs = Fraction(0)
    for left_index, (left_delay, left) in enumerate(
        observation.active_centered_fields
    ):
        for right_delay, right in observation.active_centered_fields[
            left_index + 1 :
        ]:
            squared_distance = sum(
                weight * (left_value - right_value) ** 2
                for weight, left_value, right_value in zip(
                    observation.exact_metric_weights,
                    left,
                    right,
                    strict=True,
                )
            )
            exact_rhs += (
                certificate.stationary_distribution[0]
                * coefficient_by_delay[left_delay]
                * coefficient_by_delay[right_delay]
                * squared_distance
                / 2
            )
    assert exact_rhs == observation.exact_energy_drop
    assert observation.lyapunov_nonincreasing
    assert not observation.lyapunov_equality
    assert not observation.active_centered_fields_pairwise_equal
    assert observation.exact_dissipation_identity_certified
    assert observation.equality_iff_active_centered_fields_agree_certified


def test_equality_uses_centered_active_fields() -> None:
    certificate = _strict_certificate()
    observation = observe_uniform_remesh_history_transition(
        certificate,
        (
            (Fraction(1), Fraction(3)),
            (Fraction(2), Fraction(4)),
            (Fraction(-5), Fraction(-3)),
        ),
        (Fraction(1), Fraction(3)),
    )

    assert len(set(observation.exact_history)) == 3
    assert len(set(observation.exact_centered_history)) == 1
    assert observation.active_centered_fields_pairwise_equal
    assert observation.exact_energy_drop == 0
    assert observation.exact_jensen_dissipation == 0
    assert observation.lyapunov_equality


def test_coincident_delays_are_combined_before_dissipation() -> None:
    certificate = certify_uniform_remesh_history_stability(
        alpha=Fraction(1, 2),
        tau_local=2,
        tau_global=2,
    )

    assert certificate.combined_delay_coefficients == (
        (0, Fraction(1, 4)),
        (2, Fraction(3, 4)),
    )
    assert certificate.active_delays == (0, 2)
    assert certificate.stationary_denominator == Fraction(5, 2)
    assert certificate.stationary_distribution == (
        Fraction(2, 5),
        Fraction(3, 10),
        Fraction(3, 10),
    )
    observation = observe_uniform_remesh_history_transition(
        certificate,
        ((2, 0), (100, -100), (0, 2)),
        (1, 1),
    )
    assert tuple(delay for delay, _ in observation.active_centered_fields) == (0, 2)
    assert observation.exact_energy_drop == observation.exact_jensen_dissipation
    assert observation.exact_energy_drop > 0


def test_alpha_zero_reduces_to_the_identity_state() -> None:
    certificate = certify_uniform_remesh_history_stability(
        alpha=0,
        tau_local=7,
        tau_global=11,
    )
    assert certificate.combined_delay_coefficients == ((0, Fraction(1)),)
    assert certificate.active_max_delay == 0
    assert certificate.companion_matrix == ((Fraction(1),),)
    assert certificate.stationary_distribution == (Fraction(1),)
    assert certificate.alpha_zero_identity_map_certified
    assert not certificate.alpha_one_pure_delay_map_certified
    assert not certificate.strict_mixing_companion_primitive_certified

    observation = observe_uniform_remesh_history_transition(
        certificate,
        ((Fraction(2), Fraction(-1)),),
        (1, 2),
    )
    assert observation.exact_next_field == observation.exact_history[0]
    assert observation.exact_augmented_energy_before == (
        observation.exact_augmented_energy_after
    )
    assert observation.exact_energy_drop == 0
    assert observation.lyapunov_equality


def test_alpha_one_is_a_pure_delay_with_possible_alternation() -> None:
    certificate = certify_uniform_remesh_history_stability(
        alpha=1,
        tau_local=3,
        tau_global=1,
    )
    assert certificate.combined_delay_coefficients == ((1, Fraction(1)),)
    assert certificate.active_max_delay == 1
    assert certificate.companion_matrix == (
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(0)),
    )
    assert certificate.stationary_distribution == (
        Fraction(1, 2),
        Fraction(1, 2),
    )
    assert certificate.alpha_one_pure_delay_map_certified
    assert certificate.alpha_one_augmented_energy_conservation_certified
    assert certificate.periodic_temporal_cycles_possible
    assert certificate.pure_delay_period == 2
    assert not certificate.strict_mixing_pointwise_temporal_convergence_certified

    observation = observe_uniform_remesh_history_transition(
        certificate,
        ((1, -1), (-1, 1)),
        (1, 1),
    )
    assert observation.exact_next_field == (Fraction(-1), Fraction(1))
    assert observation.exact_energy_drop == 0
    assert observation.exact_augmented_energy_before == (
        observation.exact_augmented_energy_after
    )
    assert observation.exact_strict_mixing_temporal_limit is None
    assert observation.lyapunov_equality

    fixed_orbit = observe_uniform_remesh_history_transition(
        certificate,
        ((2, -3), (2, -3)),
        (1, 1),
    )
    assert fixed_orbit.exact_next_field == (Fraction(2), Fraction(-3))
    assert certificate.pure_delay_period == 2
    assert fixed_orbit.exact_history[0] == fixed_orbit.exact_history[1]


def test_strict_mixing_gives_only_a_pointwise_temporal_limit() -> None:
    certificate = _strict_certificate()
    observation = observe_uniform_remesh_history_transition(
        certificate,
        ((3, -1), (0, 2), (1, 4)),
        (2, 1),
    )

    assert certificate.strict_mixing_companion_primitive_certified
    assert certificate.strict_mixing_pointwise_temporal_convergence_certified
    assert observation.exact_stationary_history_barycenter == (
        Fraction(14, 9),
        Fraction(10, 9),
    )
    assert observation.exact_strict_mixing_temporal_limit == (
        observation.exact_stationary_history_barycenter
    )
    assert observation.exact_stationary_history_barycenter[0] != (
        observation.exact_stationary_history_barycenter[1]
    )
    assert not certificate.spatial_consensus_certified
    assert not certificate.zero_pressure_equilibrium_certified
    assert not observation.spatial_consensus_certified
    assert not observation.zero_pressure_equilibrium_certified


def test_certificate_and_nested_observation_fail_closed_after_tampering() -> None:
    certificate = _strict_certificate()
    observation = observe_uniform_remesh_history_transition(
        certificate,
        ((3, -1), (0, 2), (1, 4)),
        (2, 1),
    )
    object.__setattr__(certificate, "beta", Fraction(9))

    assert not certificate.stability_certificate_certified
    assert certificate.failed_conditions == ("remesh_history_proof_fields_intact",)
    assert not observation.transition_observation_certified
    assert observation.failed_conditions == (
        "remesh_history_transition_proof_fields_intact",
    )
    with pytest.raises(TNFRValueError, match="unsealed, tampered"):
        observe_uniform_remesh_history_transition(
            certificate,
            ((3, -1), (0, 2), (1, 4)),
            (2, 1),
        )


def test_transition_fails_closed_after_direct_field_tampering() -> None:
    observation = observe_uniform_remesh_history_transition(
        _strict_certificate(),
        ((3, -1), (0, 2), (1, 4)),
        (2, 1),
    )
    object.__setattr__(observation, "exact_energy_drop", Fraction(-1))

    assert not observation.transition_observation_certified
    assert not observation.exact_dissipation_identity_certified


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("beta", Fraction(9)),
        ("combined_delay_coefficients", ((0, Fraction(1)),)),
        ("companion_matrix", ((Fraction(1),),)),
        ("stationary_distribution", (Fraction(1),)),
        ("conditions", (("forged", True),)),
    ],
)
def test_privately_resealed_inconsistent_certificate_derivatives_fail_closed(
    field_name: str,
    forged_value: object,
) -> None:
    certificate = _strict_certificate()
    forged = replace(
        certificate,
        **{field_name: forged_value},
        _proof_stamp=(),
    )
    resealed = stability_module._seal(
        forged,
        UniformRemeshHistoryStabilityCertificate,
        stability_module._CERTIFICATE_PROOF_VERSION,
    )

    assert not resealed.stability_certificate_certified
    assert resealed.failed_conditions == ("remesh_history_proof_fields_intact",)


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("exact_next_field", (Fraction(99), Fraction(99))),
        ("exact_history_energies", (Fraction(0), Fraction(0), Fraction(0))),
        ("exact_jensen_dissipation", Fraction(999)),
        ("active_centered_fields_pairwise_equal", True),
        (
            "exact_stationary_history_barycenter",
            (Fraction(0), Fraction(0)),
        ),
        ("conditions", (("forged", True),)),
    ],
)
def test_privately_resealed_inconsistent_transition_derivatives_fail_closed(
    field_name: str,
    forged_value: object,
) -> None:
    observation = observe_uniform_remesh_history_transition(
        _strict_certificate(),
        ((3, -1), (0, 2), (1, 4)),
        (2, 1),
    )
    forged = replace(
        observation,
        **{field_name: forged_value},
        _proof_stamp=(),
    )
    resealed = stability_module._seal(
        forged,
        UniformRemeshHistoryTransitionObservation,
        stability_module._TRANSITION_PROOF_VERSION,
    )

    assert not resealed.transition_observation_certified
    assert resealed.failed_conditions == (
        "remesh_history_transition_proof_fields_intact",
    )


def test_privately_resealed_always_equal_derivatives_do_not_dispatch_equality() -> None:
    marker: list[str] = []

    class AlwaysEqual:
        def __eq__(self, other):
            del other
            marker.append("caller equality dispatched")
            return True

    certificate = _strict_certificate()
    forged_certificate = replace(
        certificate,
        beta=AlwaysEqual(),
        _proof_stamp=(),
    )
    resealed_certificate = stability_module._seal(
        forged_certificate,
        UniformRemeshHistoryStabilityCertificate,
        stability_module._CERTIFICATE_PROOF_VERSION,
    )

    observation = observe_uniform_remesh_history_transition(
        certificate,
        ((3, -1), (0, 2), (1, 4)),
        (2, 1),
    )
    forged_observation = replace(
        observation,
        active_centered_fields=AlwaysEqual(),
        _proof_stamp=(),
    )
    resealed_observation = stability_module._seal(
        forged_observation,
        UniformRemeshHistoryTransitionObservation,
        stability_module._TRANSITION_PROOF_VERSION,
    )

    assert not resealed_certificate.stability_certificate_certified
    assert not resealed_observation.transition_observation_certified
    assert marker == []


@pytest.mark.parametrize("forged_nodes", [("same", "same"), ([], [])])
def test_privately_resealed_invalid_node_order_fails_closed(
    forged_nodes: object,
) -> None:
    observation = observe_uniform_remesh_history_transition(
        _strict_certificate(),
        ((3, -1), (0, 2), (1, 4)),
        (2, 1),
    )
    forged = replace(
        observation,
        nodes=forged_nodes,
        _proof_stamp=(),
    )
    resealed = stability_module._seal(
        forged,
        UniformRemeshHistoryTransitionObservation,
        stability_module._TRANSITION_PROOF_VERSION,
    )

    assert not resealed.transition_observation_certified
    assert resealed.failed_conditions == (
        "remesh_history_transition_proof_fields_intact",
    )


def test_certificate_tamper_is_rejected_before_hostile_numeric_dispatch() -> None:
    marker: list[str] = []

    class HostileAlpha:
        def __rsub__(self, other):
            del other
            marker.append("numeric protocol dispatched")
            raise SystemExit("must not escape proof validation")

    certificate = _strict_certificate()
    object.__setattr__(certificate, "alpha", HostileAlpha())

    assert not certificate.stability_certificate_certified
    assert marker == []


def test_transition_tamper_is_rejected_before_nested_property_dispatch() -> None:
    marker: list[str] = []

    class HostileCertificate:
        @property
        def strict_mixing_pointwise_temporal_convergence_certified(self):
            marker.append("nested property dispatched")
            raise SystemExit("must not escape proof validation")

    observation = observe_uniform_remesh_history_transition(
        _strict_certificate(),
        ((3, -1), (0, 2), (1, 4)),
        (2, 1),
    )
    object.__setattr__(observation, "certificate", HostileCertificate())

    assert not observation.transition_observation_certified
    assert marker == []


def test_hostile_proof_stamp_is_rejected_without_equality_dispatch() -> None:
    marker: list[str] = []

    class HostileStampElement:
        def __eq__(self, other):
            del other
            marker.append("proof stamp equality dispatched")
            raise SystemExit("must not escape proof validation")

    certificate = _strict_certificate()
    object.__setattr__(
        certificate,
        "_proof_stamp",
        ("uniform_remesh_history_stability_v1", HostileStampElement()),
    )

    assert not certificate.stability_certificate_certified
    assert certificate.failed_conditions == ("remesh_history_proof_fields_intact",)
    assert marker == []


def test_malformed_condition_tampering_fails_closed() -> None:
    certificate = _strict_certificate()
    observation = observe_uniform_remesh_history_transition(
        certificate,
        ((3, -1), (0, 2), (1, 4)),
        (2, 1),
    )
    object.__setattr__(observation, "conditions", (("truncated",),))

    assert not observation.transition_observation_certified
    assert observation.failed_conditions == (
        "remesh_history_transition_proof_fields_intact",
    )


@pytest.mark.parametrize(
    "alpha",
    [True, -0.01, 1.01, float("nan"), float("inf"), "0.5"],
)
def test_invalid_alpha_is_rejected(alpha) -> None:
    with pytest.raises(TNFRValueError, match="alpha"):
        certify_uniform_remesh_history_stability(
            alpha=alpha,
            tau_local=1,
            tau_global=2,
        )


@pytest.mark.parametrize("delay", [True, 0, -1, 1.5, "2"])
def test_invalid_delays_are_rejected(delay) -> None:
    with pytest.raises(TNFRValueError, match="positive integer"):
        certify_uniform_remesh_history_stability(
            alpha=Fraction(1, 2),
            tau_local=delay,
            tau_global=2,
        )


@pytest.mark.parametrize(
    ("history", "weights", "nodes", "message"),
    [
        (((1, 2), (3, 4)), (1, 1), None, "exactly 3 fields"),
        (
            ((1, 2), (3,), (4, 5)),
            (1, 1),
            None,
            "common spatial shape",
        ),
        (((1, 2), (3, 4), (5, 6)), (1,), None, "match the spatial"),
        (((1, 2), (3, 4), (5, 6)), (1, 0), None, "strictly positive"),
        (((1, 2), (3, 4), (5, 6)), (1, 1), ("a",), "match the spatial"),
        (
            ((1, 2), (3, 4), (5, 6)),
            (1, 1),
            ("a", "a"),
            "unique identifiers",
        ),
        (
            ((1, 2), (3, 4), (5, 6)),
            (1, 1),
            ([], "b"),
            "hashable identifiers",
        ),
    ],
)
def test_invalid_transition_shapes_are_rejected(
    history,
    weights,
    nodes,
    message,
) -> None:
    with pytest.raises(TNFRValueError, match=message):
        observe_uniform_remesh_history_transition(
            _strict_certificate(),
            history,
            weights,
            nodes=nodes,
        )


@pytest.mark.parametrize(
    ("history", "weights"),
    [
        (((1, 2), (3, 4), (5, float("nan"))), (1, 1)),
        (((1, 2), (3, 4), (5, 6)), (1, float("inf"))),
        ({0: (1, 2), 1: (3, 4), 2: (5, 6)}, (1, 1)),
    ],
)
def test_nonfinite_or_nonsequence_transition_inputs_are_rejected(
    history,
    weights,
) -> None:
    with pytest.raises(TNFRValueError):
        observe_uniform_remesh_history_transition(
            _strict_certificate(),
            history,
            weights,
        )
