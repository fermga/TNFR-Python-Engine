"""Adversarial boundaries of the conditional exact C6 joint-domain observer."""

from dataclasses import replace
from fractions import Fraction
import math

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.preconditions import OperatorPreconditionError, validate_coupling
from tnfr.operators.preconditions.coherence import validate_coherence_strict
from tnfr.physics.coupling_winding import (
    derive_c6_winding_joint_domain, observe_c6_winding_joint_domain,
)
from tnfr.physics.winding_certificates import certify_phase_winding


F = Fraction
ZERO = (F(0),) * 6
PHASE = (F(1, 1000), F(-1, 1000)) * 3
UNEQUAL_EPI = (F(1, 4), F(3, 8), F(1, 2), F(5, 8), F(3, 4), F(3, 8))
IDENTITY = tuple(tuple(F(i == j) for j in range(6)) for i in range(6))


def _reference(**overrides):
    parameters = dict(
        coupling_phase_factor=F(1, 4), coherence_phase_factor=F(3, 10),
        capacity=1, epi_weight=1, phase_weight=F(1, 8), timestep=F(1, 4),
        epi_lower=F(1, 8), epi_upper=F(7, 8),
    )
    parameters.update(overrides)
    return derive_c6_winding_joint_domain(**parameters)


def _step(reference, *, before=PHASE, after=None, epi=UNEQUAL_EPI):
    if after is None:
        after = tuple(value / 2 for value in before)
    return observe_c6_winding_joint_domain(
        reference, phase_before_pi=before, phase_after_pi=after, epi=epi,
    )


def test_euler_boundary_retains_an_exact_two_step_alternating_epi_orbit():
    reference = _reference(timestep=1)
    initial = (F(1, 4), F(3, 4)) * 3
    first = _step(reference, before=ZERO, after=ZERO, epi=initial)
    second = _step(reference, before=ZERO, after=ZERO, epi=first.epi_after)
    assert reference.epi_disagreement_factor == 1
    assert not reference.strict_epi_convergence
    assert first.epi_after == (F(3, 4), F(1, 4)) * 3
    assert second.epi_after == initial
    assert first.epi_after != initial
    assert first.phase_pressure == second.phase_pressure == ZERO
    assert first.modeled_pressure == tuple(F(1, 2) if i % 2 == 0 else F(-1, 2) for i in range(6))
    assert first.mean_before == first.mean_after == second.mean_after == F(1, 2)
    assert first.lower_reserve_gain == first.upper_reserve_gain == 0
    assert max(first.epi_after) - min(first.epi_after) == F(1, 2)


def test_zero_duration_keeps_unequal_epi_despite_nonzero_modeled_rate():
    reference = _reference(timestep=0)
    result = _step(reference)
    assert reference.nodal_euler_matrix == IDENTITY
    assert reference.forcing_step_factor == reference.epi_phase_budget_weight == 0
    assert reference.epi_disagreement_factor == 1 and not reference.strict_epi_convergence
    assert result.epi_after == UNEQUAL_EPI
    assert any(result.phase_pressure) and any(result.modeled_rate)
    assert result.lower_reserve_before == result.lower_reserve_after == min(UNEQUAL_EPI)
    assert result.upper_reserve_before == result.upper_reserve_after == max(UNEQUAL_EPI)


def test_zero_phase_weight_leaves_nontrivial_pure_epi_diffusion():
    reference = _reference(phase_weight=0)
    result = _step(reference)
    expected = tuple(3 * value / 4 + (UNEQUAL_EPI[(i - 1) % 6] + UNEQUAL_EPI[(i + 1) % 6]) / 8
                     for i, value in enumerate(UNEQUAL_EPI))
    assert reference.forcing_step_factor == reference.epi_phase_budget_weight == 0
    assert reference.strict_epi_convergence
    assert any(result.phase_pressure)
    assert result.epi_after == expected and expected != UNEQUAL_EPI
    assert result.mean_before == result.mean_after == sum(UNEQUAL_EPI) / 6
    other_endpoint = _step(reference, after=ZERO)
    assert other_endpoint.phase_pressure == ZERO
    assert other_endpoint.epi_after == result.epi_after


@pytest.mark.parametrize("constant", (F(0), F(1, 2000)))
def test_supplied_phase_contraction_does_not_authenticate_a_canonical_phase_step(constant):
    reference = _reference()
    supplied = (constant,) * 6
    result = _step(reference, after=supplied)
    # For this exact alternating chart, actual UM/IL has the finite multiplier
    # (1-t)*(1-2*alpha)=3/10. Neither supplied constant is that phase endpoint.
    actual_multiplier = (1 - reference.coupling_phase_factor) * (1 - 2 * reference.coherence_phase_factor)
    assert actual_multiplier == F(3, 10)
    assert supplied != tuple(actual_multiplier * value for value in PHASE)
    assert result.phase_after_pi == supplied
    assert result.phase_contraction_slack > 0
    assert all(value >= 0 for value in result.phase_interval_nesting_slack)
    assert result.phase_pressure == ZERO
    assert result.modeled_pressure == tuple(
        (UNEQUAL_EPI[(i - 1) % 6] + UNEQUAL_EPI[(i + 1) % 6]) / 2 - value
        for i, value in enumerate(UNEQUAL_EPI)
    )
    assert result.mean_identity_residual == 0


@pytest.mark.parametrize("field,value", (
    ("nonlinear_oscillation_factor", F(0)),
    ("epi_phase_budget_weight", F(0)),
    ("forcing_step_factor", F(99)),
    ("nodal_euler_matrix", IDENTITY),
    ("max_phase_oscillation_pi", F(100)),
    ("strict_epi_convergence", False),
))
def test_forged_public_cache_fields_are_rederived(field, value):
    reference = _reference()
    forged = replace(reference, **{field: value})
    assert getattr(forged, field) != getattr(reference, field)
    assert _step(forged) == _step(reference)


def test_forged_nested_tangent_reference_cannot_replace_the_nonlinear_domain():
    reference = _reference()
    forged_phase = replace(reference.phase_reference, product_matrix=IDENTITY, quotient_energy_bound=F(0))
    forged = replace(reference, phase_reference=forged_phase)
    result = _step(forged)
    assert result == _step(reference)
    assert result.reference.phase_reference == reference.phase_reference
    assert result.reference.nonlinear_oscillation_factor != result.reference.phase_reference.quotient_energy_bound


def test_zeroed_reserve_cache_cannot_admit_an_unfunded_boundary_field():
    reference = _reference()
    forged = replace(reference, epi_phase_budget_weight=F(0))
    initial = (reference.epi_lower,) * 6
    with pytest.raises(ValueError, match="future phase-forcing reserve"):
        _step(forged, epi=initial)


@pytest.mark.parametrize("field,value", (("capacity", F(0)), ("coupling_phase_factor", F(0))))
def test_forged_invalid_source_coefficient_is_rejected_despite_valid_old_caches(field, value):
    forged = replace(_reference(), **{field: value})
    with pytest.raises(ValueError):
        _step(forged)


@pytest.mark.parametrize("epi,capacity,reason", (
    (F(1, 100), F(1), "EPI too low"),
    (F(1, 2), F(1, 1000), "frequency too low"),
))
def test_positive_joint_band_and_winding_do_not_supply_strict_um_floors(epi, capacity, reason):
    reference = _reference(capacity=capacity, epi_lower=epi / 2, epi_upper=(1 + epi) / 2)
    modeled = _step(reference, before=ZERO, after=ZERO, epi=(epi,) * 6)
    assert modeled.epi_after == (epi,) * 6
    assert reference.epi_lower > 0 and reference.capacity > 0
    # This graph is initial data for read-only gate checks; no operator runs.
    graph = nx.cycle_graph(6)
    for node in graph:
        graph.nodes[node].update({
            ALIAS_THETA[0]: node * math.pi / 3,
            ALIAS_EPI[0]: float(epi), ALIAS_VF[0]: float(capacity),
        })
    winding = certify_phase_winding(graph, range(6))
    assert winding.is_defined and winding.winding == 1 and winding.u3_admissible
    for node in graph:
        assert validate_coherence_strict(graph, node, emit_warnings=False) is None
        with pytest.raises(OperatorPreconditionError, match=reason):
            validate_coupling(graph, node)
