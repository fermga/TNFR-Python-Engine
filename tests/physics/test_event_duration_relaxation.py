"""Continuous physical duration mapped to the fixed diffusion theorem."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math

import networkx as nx
import numpy as np
import pytest

import tnfr.physics.event_duration as event_duration
from tnfr.physics.event_duration import (
    ContinuousRelaxationDurationDiagnostic,
    diagnose_continuous_relaxation_duration,
)
from tnfr.physics.structural_diffusion import (
    diagnose_euler_relaxation_window,
    verify_heterogeneous_diffusion_stability,
)


def _state(graph, *, capacity=1.0):
    capacities = (
        [capacity] * len(graph)
        if isinstance(capacity, (int, float))
        else list(capacity)
    )
    for index, (node, nu_f) in enumerate(zip(graph, capacities)):
        graph.nodes[node].update(
            EPI=float(index),
            nu_f=float(nu_f),
            theta=0.0,
        )
    return graph


def test_diagnostic_uses_the_fixed_certificate_energy_rate() -> None:
    graph = _state(nx.complete_graph(4))
    duration = 0.75
    target = 0.2
    source = verify_heterogeneous_diffusion_stability(graph)

    result = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=duration,
        target_fraction=target,
    )

    assert isinstance(result, ContinuousRelaxationDurationDiagnostic)
    assert not result.abstained
    assert result.fixed_flow_certificate_available
    assert result.fixed_flow_theorem_certified
    assert result.exact_energy_decay_rate_lower_bound == (
        2 * source.exact_quotient_gap_lower_bound
    )
    assert result.certified_energy_decay_rate_lower_bound == pytest.approx(8 / 3)
    assert result.spectral_energy_decay_rate_estimate == pytest.approx(8 / 3)
    assert not result.spectral_estimate_is_proof_input
    assert result.spectral_estimate_provenance == (
        "unsealed_source_eigensolver_diagnostic"
    )
    assert result.exact_flow_duration == Fraction(3, 4)
    assert result.exact_target_fraction == Fraction.from_float(target)
    rate = result.certified_energy_decay_rate_lower_bound
    assert rate is not None
    assert result.required_flow_duration_estimate == pytest.approx(
        -math.log(target) / rate
    )
    assert result.required_flow_duration == pytest.approx(
        result.required_flow_duration_estimate
    )
    assert result.decay_factor_estimate == pytest.approx(
        math.exp(-rate * duration)
    )
    assert result.certified_decay_factor_upper_bound is not None
    assert result.exact_log_target_upper_bound is not None
    assert result.exact_required_flow_duration_upper_bound is not None
    exact_rate = result.exact_energy_decay_rate_lower_bound
    assert exact_rate is not None
    assert (
        exact_rate * result.exact_flow_duration
        >= result.exact_log_target_upper_bound
    )
    assert result.duration_reaches_target
    assert result.solver_timestep_independent
    assert result._proof_fields_are_intact()
    with pytest.raises(FrozenInstanceError):
        result.flow_duration = 2.0  # type: ignore[misc]


def test_continuous_diagnostic_is_invariant_to_euler_grid_choice() -> None:
    graph = _state(nx.path_graph(21))
    before = diagnose_continuous_relaxation_duration(
        graph, flow_duration=10.0
    )

    coarse = diagnose_euler_relaxation_window(graph, dt=0.5)
    fine = diagnose_euler_relaxation_window(graph, dt=0.25)
    after = diagnose_continuous_relaxation_duration(
        graph, flow_duration=10.0
    )

    assert coarse.dt != fine.dt
    assert coarse.modal_steps != fine.modal_steps
    assert before == after
    assert not hasattr(before, "dt")
    assert before._proof_fields_are_intact()


def test_path_21_requires_more_certified_flow_time_than_k4() -> None:
    path = diagnose_continuous_relaxation_duration(
        _state(nx.path_graph(21)), flow_duration=1.0
    )
    complete = diagnose_continuous_relaxation_duration(
        _state(nx.complete_graph(4)), flow_duration=1.0
    )

    assert not path.abstained
    assert not complete.abstained
    assert path.certified_energy_decay_rate_lower_bound is not None
    assert complete.certified_energy_decay_rate_lower_bound is not None
    assert path.certified_energy_decay_rate_lower_bound < (
        complete.certified_energy_decay_rate_lower_bound
    )
    assert path.required_flow_duration is not None
    assert complete.required_flow_duration is not None
    assert path.required_flow_duration > complete.required_flow_duration


def test_capacity_changes_the_fixed_continuous_clock() -> None:
    base = diagnose_continuous_relaxation_duration(
        _state(nx.complete_graph(4), capacity=1.0), flow_duration=1.0
    )
    faster = diagnose_continuous_relaxation_duration(
        _state(nx.complete_graph(4), capacity=2.0), flow_duration=1.0
    )

    assert base.certified_energy_decay_rate_lower_bound is not None
    assert faster.certified_energy_decay_rate_lower_bound == pytest.approx(
        2.0 * base.certified_energy_decay_rate_lower_bound
    )
    assert base.required_flow_duration is not None
    assert faster.required_flow_duration == pytest.approx(
        0.5 * base.required_flow_duration
    )


@pytest.mark.parametrize(
    "graph, detail",
    [
        (_state(nx.disjoint_union(nx.path_graph(2), nx.path_graph(2))), "connected"),
        (_state(nx.path_graph(3), capacity=[1.0, 0.0, 1.0]), "frequency"),
        (_state(nx.path_graph(1)), "at least two"),
    ],
)
def test_out_of_domain_fixed_flows_return_explicit_abstention(graph, detail) -> None:
    result = diagnose_continuous_relaxation_duration(
        graph, flow_duration=1.0
    )

    assert result.abstained
    assert result.abstention_reason == "fixed_flow_domain_error"
    assert detail in (result.abstention_detail or "")
    assert not result.fixed_flow_certificate_available
    assert not result.fixed_flow_theorem_certified
    assert result.required_flow_duration is None
    assert result.duration_reaches_target is None
    assert result._proof_fields_are_intact()


def test_rounded_uniform_fixed_point_failure_abstains_after_certificate() -> None:
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 0.1), (0, 2, 0.2), (1, 2, 0.3)]
    )
    graph = _state(graph)

    result = diagnose_continuous_relaxation_duration(
        graph, flow_duration=1.0
    )

    assert result.abstained
    assert result.fixed_flow_certificate_available
    assert not result.fixed_flow_theorem_certified
    assert result.abstention_reason == "fixed_flow_theorem_not_certified"
    assert result.required_flow_duration is None
    assert result._proof_fields_are_intact()


def test_zero_duration_is_valid_and_cannot_reach_a_strict_decay_target() -> None:
    result = diagnose_continuous_relaxation_duration(
        _state(nx.complete_graph(4)),
        flow_duration=0.0,
        target_fraction=0.5,
    )

    assert not result.abstained
    assert result.certified_decay_factor_upper_bound == 1.0
    assert result.decay_factor_estimate == 1.0
    assert result.required_flow_duration is not None
    assert result.required_flow_duration > 0.0
    assert result.duration_reaches_target is False


def test_required_duration_is_the_inclusive_continuous_threshold() -> None:
    graph = _state(nx.complete_graph(4))
    probe = diagnose_continuous_relaxation_duration(
        graph, flow_duration=0.0, target_fraction=0.25
    )
    threshold = probe.required_flow_duration
    assert threshold is not None

    below = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=math.nextafter(threshold, 0.0),
        target_fraction=0.25,
    )
    at = diagnose_continuous_relaxation_duration(
        graph, flow_duration=threshold, target_fraction=0.25
    )

    assert below.duration_reaches_target is False
    assert at.duration_reaches_target is True
    assert at.certified_decay_factor_upper_bound == pytest.approx(0.25)


@pytest.mark.parametrize(
    "parameter, value, message",
    [
        ("flow_duration", True, "flow_duration"),
        ("flow_duration", -0.1, "flow_duration"),
        ("flow_duration", float("nan"), "flow_duration"),
        ("flow_duration", float("inf"), "flow_duration"),
        ("target_fraction", True, "target_fraction"),
        ("target_fraction", 0.0, "target_fraction"),
        ("target_fraction", 1.0, "target_fraction"),
        ("target_fraction", float("nan"), "target_fraction"),
        ("tolerance", True, "tolerance"),
        ("tolerance", 0.0, "tolerance"),
        ("tolerance", -1.0, "tolerance"),
        ("tolerance", float("inf"), "tolerance"),
    ],
)
def test_invalid_controls_raise_before_fixed_flow_verification(
    parameter, value, message
) -> None:
    controls = {
        "flow_duration": 1.0,
        "target_fraction": 0.5,
        "tolerance": 1e-10,
    }
    controls[parameter] = value

    with pytest.raises((TypeError, ValueError), match=message):
        diagnose_continuous_relaxation_duration(
            _state(nx.complete_graph(4)), **controls
        )


def test_positive_duration_underflow_is_not_relabelled_as_zero() -> None:
    with pytest.raises(ValueError, match="underflows"):
        diagnose_continuous_relaxation_duration(
            _state(nx.complete_graph(4)),
            flow_duration=Fraction(1, 10**1000),
        )


def test_proof_stamp_rejects_result_promotion_and_rate_tampering() -> None:
    success = diagnose_continuous_relaxation_duration(
        _state(nx.complete_graph(4)), flow_duration=1.0
    )
    abstention = diagnose_continuous_relaxation_duration(
        _state(nx.path_graph(3), capacity=[1.0, 0.0, 1.0]),
        flow_duration=1.0,
    )

    forged_decision = replace(
        success,
        duration_reaches_target=not success.duration_reaches_target,
    )
    forged_rate = replace(
        success,
        certified_energy_decay_rate_lower_bound=999.0,
    )
    changed_unsealed_estimate = replace(
        success,
        spectral_energy_decay_rate_estimate=999.0,
    )
    forged_promotion = replace(
        abstention,
        abstained=False,
        fixed_flow_theorem_certified=True,
        duration_reaches_target=True,
    )

    assert success._proof_fields_are_intact()
    assert abstention._proof_fields_are_intact()
    assert not forged_decision._proof_fields_are_intact()
    assert not forged_rate._proof_fields_are_intact()
    assert not forged_promotion._proof_fields_are_intact()
    assert changed_unsealed_estimate._proof_fields_are_intact()


@pytest.mark.parametrize(
    "replacement",
    [
        {"exact_quotient_gap_lower_bound": Fraction(0)},
        {"exact_quotient_gap_lower_bound": float("nan")},
        {"certified_exponential_rate_lower_bound": float("nan")},
    ],
    ids=("changed-rate", "invalid-exact-rate", "invalid-rate-display"),
)
def test_forged_fixed_flow_source_cannot_promote_a_duration(
    monkeypatch,
    replacement,
) -> None:
    graph = _state(nx.complete_graph(4))
    source = verify_heterogeneous_diffusion_stability(graph)
    forged = replace(source, **replacement)
    assert not forged._proof_fields_are_intact()
    monkeypatch.setattr(
        event_duration,
        "verify_heterogeneous_diffusion_stability",
        lambda *_args, **_kwargs: forged,
    )

    result = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=1.0,
        target_fraction=0.5,
    )

    assert result.abstained
    assert result.abstention_reason == "fixed_flow_certificate_integrity_failed"
    assert not result.fixed_flow_theorem_certified
    assert result.exact_energy_decay_rate_lower_bound is None
    assert result.certified_energy_decay_rate_lower_bound is None
    assert result.duration_reaches_target is None
    assert result._proof_fields_are_intact()


def test_non_certificate_source_abstains_before_reading_payload(
    monkeypatch,
) -> None:
    class CertificateImpostor:
        @property
        def nodes(self):
            raise AssertionError("invalid certificate payload was consumed")

        def _proof_fields_are_intact(self):
            raise AssertionError("invalid certificate method was called")

    graph = _state(nx.complete_graph(4))
    monkeypatch.setattr(
        event_duration,
        "verify_heterogeneous_diffusion_stability",
        lambda *_args, **_kwargs: CertificateImpostor(),
    )

    result = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=1.0,
        target_fraction=0.5,
    )

    assert result.abstained
    assert result.abstention_reason == "fixed_flow_certificate_integrity_failed"
    assert result.nodes == tuple(graph.nodes())
    assert result.exact_energy_decay_rate_lower_bound is None
    assert result.certified_energy_decay_rate_lower_bound is None
    assert result.spectral_energy_decay_rate_estimate is None
    assert result._proof_fields_are_intact()

def test_invalid_unsealed_spectral_estimate_does_not_block_exact_proof(
    monkeypatch,
) -> None:
    graph = _state(nx.complete_graph(4))
    source = verify_heterogeneous_diffusion_stability(graph)
    unsealed = replace(source, exponential_rate=float("nan"))
    assert unsealed._proof_fields_are_intact()
    monkeypatch.setattr(
        event_duration,
        "verify_heterogeneous_diffusion_stability",
        lambda *_args, **_kwargs: unsealed,
    )

    result = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=1.0,
        target_fraction=0.5,
    )

    assert not result.abstained
    assert result.fixed_flow_theorem_certified
    assert result.spectral_energy_decay_rate_estimate is None
    assert result.duration_reaches_target is True
    assert result._proof_fields_are_intact()


def test_positive_required_duration_cannot_underflow_to_a_false_positive() -> None:
    graph = _state(nx.complete_graph(4), capacity=2e307)
    for node in graph:
        graph.nodes[node]["EPI"] = 0.0
    target = math.nextafter(1.0, 0.0)

    zero = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=0.0,
        target_fraction=target,
    )

    assert not zero.abstained
    assert zero.required_flow_duration_estimate == 0.0
    assert zero.exact_required_flow_duration_upper_bound is not None
    assert zero.exact_required_flow_duration_upper_bound > 0
    assert zero.required_flow_duration == math.ulp(0.0)
    assert zero.duration_reaches_target is False
    assert zero.certified_decay_factor_upper_bound == 1.0
    assert zero.certified_decay_factor_upper_bound > target

    first_positive = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=math.ulp(0.0),
        target_fraction=target,
    )
    assert first_positive.duration_reaches_target is True
    assert first_positive.certified_decay_factor_upper_bound <= target


def test_libm_threshold_estimate_cannot_promote_a_duration() -> None:
    graph = _state(nx.complete_graph(4))
    target = 1e-300
    probe = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=0.0,
        target_fraction=target,
    )
    estimate = probe.required_flow_duration_estimate
    certified = probe.required_flow_duration
    assert estimate is not None
    assert certified is not None
    assert estimate < certified

    at_estimate = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=estimate,
        target_fraction=target,
    )
    at_certified = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=certified,
        target_fraction=target,
    )
    assert at_estimate.duration_reaches_target is False
    assert at_certified.duration_reaches_target is True


def test_smallest_target_retains_a_nonzero_certified_upper_bound() -> None:
    graph = _state(nx.complete_graph(4))
    target = math.ulp(0.0)
    probe = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=0.0,
        target_fraction=target,
    )
    threshold = probe.required_flow_duration
    assert threshold is not None

    result = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=threshold,
        target_fraction=target,
    )

    assert result.duration_reaches_target is True
    assert result.certified_decay_factor_upper_bound == target


def test_exact_rate_survives_operational_and_spectral_underflow() -> None:
    graph = nx.path_graph(5)
    minimum = math.ulp(0.0)
    for source, target in graph.edges:
        graph.edges[source, target]["weight"] = minimum
    for index, node in enumerate(graph):
        graph.nodes[node].update(
            EPI=float(index),
            nu_f=minimum * graph.degree[node],
            theta=0.0,
        )

    result = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=1e308,
        target_fraction=math.nextafter(1.0, 0.0),
    )

    assert not result.abstained
    assert result.fixed_flow_theorem_certified
    assert result.exact_energy_decay_rate_lower_bound is not None
    assert result.exact_energy_decay_rate_lower_bound > 0
    assert result.certified_energy_decay_rate_lower_bound == 0.0
    assert result.certified_energy_decay_rate_lower_bound.hex() == "0x0.0p+0"
    assert result.spectral_energy_decay_rate_estimate == 0.0
    assert result.spectral_energy_decay_rate_estimate.hex() == "0x0.0p+0"
    assert result.required_flow_duration_estimate is None
    assert result.decay_factor_estimate is None
    assert result.exact_required_flow_duration_upper_bound is not None
    assert result.required_flow_duration is not None
    assert math.isfinite(result.required_flow_duration)
    assert result.duration_reaches_target is True
    factor = result.certified_decay_factor_upper_bound
    assert factor is not None
    assert 0.0 < factor <= result.target_fraction
    assert result._proof_fields_are_intact()


def test_negative_zero_duration_is_canonicalized_to_positive_zero() -> None:
    result = diagnose_continuous_relaxation_duration(
        _state(nx.complete_graph(4)),
        flow_duration=-0.0,
    )

    assert result.flow_duration.hex() == "0x0.0p+0"
    assert result.exact_flow_duration == 0


@pytest.mark.parametrize(
    "field, replacement, message",
    [
        ("nodes", [0, 1, 2, 3], "immutable tuple"),
        ("flow_duration", True, "binary64 float"),
        ("exact_flow_duration", 1, "exact Fraction"),
        ("fixed_flow_theorem_certified", 2, "boolean"),
        ("abstained", [], "boolean"),
        ("duration_reaches_target", 1, "boolean or None"),
        ("_source_certificate_stamp", [], "immutable tuple"),
    ],
)
def test_sealed_result_rejects_type_coercing_replacements(
    field, replacement, message
) -> None:
    result = diagnose_continuous_relaxation_duration(
        _state(nx.complete_graph(4)), flow_duration=1.0
    )

    with pytest.raises(TypeError, match=message):
        replace(result, **{field: replacement})
