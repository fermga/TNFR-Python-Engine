"""Independent rational boundaries of the conditional C6 defect class."""

import math
from dataclasses import replace
from fractions import Fraction as F

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.preconditions import OperatorPreconditionError, validate_coupling
from tnfr.physics.coupling_winding import (
    bound_c6_winding_uniform_defects,
    derive_c6_winding_joint_domain,
)

UNIFORM = (F(1, 2),) * 6
ZERO = (F(0),) * 6


def _reference(**overrides):
    values = dict(
        coupling_phase_factor=F(1, 2),
        coherence_phase_factor=F(3, 10),
        capacity=1,
        epi_weight=1,
        phase_weight=F(1, 4),
        timestep=F(1, 2),
        epi_lower=F(0.05),
        epi_upper=1,
    )
    values.update(overrides)
    return derive_c6_winding_joint_domain(**values)


def _bound(reference=None, **overrides):
    values = dict(
        initial_epi=UNIFORM,
        initial_phase_oscillation=0,
        phase_defect_bound=0,
        centered_epi_defect_bound=0,
        mean_prefix_bound=0,
    )
    values.update(overrides)
    return bound_c6_winding_uniform_defects(reference or _reference(), **values)


def _osc(values):
    return max(values) - min(values)


def _euler(values, s):
    return tuple(
        (1 - s) * values[i] + s * (values[(i - 1) % 6] + values[(i + 1) % 6]) / 2
        for i in range(6)
    )


def _functional(values, s):
    return _osc(values) + _osc(_euler(values, s))


@pytest.mark.parametrize("s", (F(1, 4), F(4, 5), F(9, 10)))
def test_two_step_range_factor_is_sharp_on_both_sides_of_the_four_fifths_switch(s):
    bound = _bound(_reference(timestep=s, phase_weight=0))
    a0, a1, a2 = (1 - s) ** 2 + s**2 / 2, s * (1 - s), s**2 / 4
    row = (a0, a1, a2, F(0), a2, a1)
    expected_matrix = tuple(tuple(row[(j - i) % 6] for j in range(6)) for i in range(6))
    overlap = min(s**2, 4 * s * (1 - s))
    assert bound.diffusion_two_step_matrix == expected_matrix
    assert bound.diffusion_two_step_factor == 1 - overlap
    assert bound.range_functional_factor == 1 - overlap / 2
    assert bound.strict_epi_range_contraction
    # These are sharp range witnesses, not infinitesimal or spectral modes.
    witness = (F(1), F(1), F(0), F(0), F(0), F(1)) if s <= F(4, 5) else (F(1), F(0)) * 3
    second = _euler(_euler(witness, s), s)
    assert _osc(witness) == 1
    assert _osc(second) == bound.diffusion_two_step_factor
    assert second[0] - second[3] == 1 - overlap


def test_one_step_range_functional_contracts_even_when_one_step_range_does_not():
    s = F(1, 4)
    initial = (F(3, 5), F(3, 5), F(2, 5), F(2, 5), F(2, 5), F(3, 5))
    bound = _bound(_reference(timestep=s, phase_weight=0), initial_epi=initial)
    following = _euler(initial, s)
    assert _osc(following) == _osc(initial) == F(1, 5)
    assert bound.initial_range_functional == _functional(initial, s) == F(2, 5)
    assert (
        _functional(following, s)
        == bound.range_functional_factor * bound.initial_range_functional
    )
    assert _functional(following, s) < _functional(initial, s)


def test_uniform_error_tube_uses_range_and_mean_budgets_without_square_roots():
    bound = _bound(
        phase_defect_bound=F(1, 15000), centered_epi_defect_bound=F(1, 10000)
    )
    assert bound.reference.nonlinear_oscillation_factor == F(14, 15)
    assert bound.phase_bound == F(1, 1000)
    assert bound.forcing_oscillation_bound == F(7, 20000)
    assert bound.diffusion_two_step_factor == F(3, 4)
    assert bound.range_functional_factor == F(7, 8)
    assert bound.range_functional_bound == F(7, 1250)
    assert bound.epi_envelope_lower == F(1, 2) - F(7, 1500)
    assert bound.epi_envelope_upper == F(1, 2) + F(7, 1500)
    assert (
        bound.phase_class_preserved
        and bound.epi_band_preserved
        and bound.joint_domain_preserved
    )


def test_phase_error_allowance_at_the_geometric_boundary_is_admitted_exactly():
    reference = _reference(phase_weight=0)
    epsilon = (1 - reference.nonlinear_oscillation_factor) / 12
    bound = _bound(reference, phase_defect_bound=epsilon)
    assert bound.phase_bound == F(1, 12)
    assert bound.phase_class_preserved and bound.joint_domain_preserved
    assert bound.range_functional_bound == 0


def test_exceeding_the_phase_allowance_cannot_be_hidden_by_uniform_epi():
    reference = _reference(phase_weight=0)
    epsilon = (1 - reference.nonlinear_oscillation_factor) / 12 + F(1, 10**9)
    bound = _bound(reference, phase_defect_bound=epsilon)
    assert bound.phase_bound > F(1, 12)
    assert not bound.phase_class_preserved and not bound.joint_domain_preserved
    assert bound.epi_band_preserved


def test_zero_phase_error_allowance_retains_the_initial_nonzero_phase_width():
    bound = _bound(_reference(phase_weight=0), initial_phase_oscillation=F(1, 24))
    assert bound.phase_defect_bound == 0 and bound.phase_bound == F(1, 24)
    assert bound.forcing_oscillation_bound == bound.range_functional_bound == 0
    assert bound.epi_envelope_lower == bound.epi_envelope_upper == F(1, 2)
    assert bound.joint_domain_preserved


def test_declared_class_bounds_two_adversarial_exact_updates_including_signed_mean():
    epsilon, error, mean_limit = F(1, 15000), F(1, 10000), F(1, 10**6)
    bound = _bound(
        phase_defect_bound=epsilon,
        centered_epi_defect_bound=error,
        mean_prefix_bound=mean_limit,
    )
    reference = bound.reference
    s = reference.timestep * reference.capacity * reference.epi_weight
    b, diameter = reference.forcing_step_factor, bound.phase_bound
    phase = (diameter / 2, -diameter / 2) * 3
    values = UNIFORM
    for mean_error in (mean_limit, -mean_limit):
        assert (
            _osc(phase)
            == reference.nonlinear_oscillation_factor * _osc(phase) + epsilon
        )
        centered = (error / 2, -error / 2) * 3
        defect = tuple(value + mean_error for value in centered)
        pressure = tuple(
            (phase[(i - 1) % 6] + phase[(i + 1) % 6]) / 2 - phase[i] for i in range(6)
        )
        diffused = _euler(values, s)
        following = tuple(diffused[i] + b * pressure[i] + defect[i] for i in range(6))
        assert _functional(following, s) <= (
            bound.range_functional_factor * _functional(values, s)
            + 2 * bound.forcing_oscillation_bound
        )
        assert _functional(following, s) <= bound.range_functional_bound
        assert abs(sum(following) / 6 - F(1, 2)) <= mean_limit
        assert (
            bound.epi_envelope_lower
            <= min(following)
            <= max(following)
            <= bound.epi_envelope_upper
        )
        values = following
    assert sum(values) / 6 == F(1, 2)


def test_bounded_centered_error_can_maintain_nonuniform_epi_despite_strict_diffusion():
    reference = _reference(phase_weight=0)
    s = reference.timestep * reference.capacity * reference.epi_weight
    initial = (F(1, 2) + F(1, 10000), F(1, 2) - F(1, 10000)) * 3
    diffused = _euler(initial, s)
    defect = tuple(x - y for x, y in zip(initial, diffused, strict=True))
    bound = _bound(
        reference, initial_epi=initial, centered_epi_defect_bound=_osc(defect)
    )
    assert sum(defect) == 0 and _osc(defect) > 0
    assert bound.strict_epi_range_contraction and bound.joint_domain_preserved
    assert tuple(x + y for x, y in zip(diffused, defect, strict=True)) == initial
    assert _functional(initial, s) <= bound.range_functional_bound
    # Repeating this exact admitted error tuple keeps the same nonuniform x;
    # strict contraction of T alone therefore does not establish convergence.
    assert _osc(initial) > 0


def test_uniform_endpoint_error_crosses_the_um_floor_with_zero_centered_disagreement():
    reference = _reference(phase_weight=0)
    eta = F(1, 1024)
    before, after = F(1, 2) - 460 * eta, F(1, 2) - 461 * eta
    safe = _bound(reference, mean_prefix_bound=460 * eta)
    unsafe = _bound(reference, mean_prefix_bound=461 * eta)
    assert before == F(52, 1024) and after == F(51, 1024) > 0
    assert safe.range_functional_bound == unsafe.range_functional_bound == 0
    assert safe.epi_band_preserved and not unsafe.epi_band_preserved
    assert safe.epi_envelope_lower == before and unsafe.epi_envelope_lower == after
    # This is a counterexample to a class specified only by a local error
    # bound; it does not claim production arithmetic generates this error.
    for value, admitted in ((before, True), (after, False)):
        graph = nx.cycle_graph(6)
        for node in graph:
            graph.nodes[node].update(
                {
                    ALIAS_THETA[0]: node * math.pi / 3,
                    ALIAS_EPI[0]: float(value),
                    ALIAS_VF[0]: 1.0,
                }
            )
        if admitted:
            assert validate_coupling(graph, 0) is None
        else:
            with pytest.raises(OperatorPreconditionError, match="EPI too low"):
                validate_coupling(graph, 0)


def test_zero_defects_on_a_finite_prefix_do_not_bound_a_later_mean_increment():
    reference = _reference(phase_weight=0)
    bound = _bound(reference)
    assert bound.joint_domain_preserved and bound.mean_prefix_bound == 0
    # Any finite number of exact uniform steps has these zero observed errors.
    assert _euler(UNIFORM, F(1, 2)) == UNIFORM
    next_error = (F(-1, 1024),) * 6
    assert _osc(next_error) == bound.centered_epi_defect_bound == 0
    actual_mean_increment = sum(next_error) / 6
    assert abs(actual_mean_increment) > bound.mean_prefix_bound
    # The claimed all-prefix mean assumption fails despite unchanged centered
    # and phase error bounds; finite extrema cannot authenticate that premise.


@pytest.mark.parametrize("s", (F(0), F(1)))
def test_noncontracting_diffusion_rejects_a_positive_uniform_forcing_allowance(s):
    with pytest.raises(ValueError):
        _bound(
            _reference(timestep=s, phase_weight=0),
            centered_epi_defect_bound=F(1, 10**6),
        )


@pytest.mark.parametrize("s", (F(0), F(1)))
def test_noncontracting_zero_forcing_class_remains_bounded_without_strict_decay(s):
    initial = (F(49, 100), F(51, 100)) * 3
    bound = _bound(_reference(timestep=s, phase_weight=0), initial_epi=initial)
    assert bound.diffusion_two_step_factor == bound.range_functional_factor == 1
    assert not bound.strict_epi_range_contraction
    assert bound.range_functional_bound == bound.initial_range_functional == F(1, 25)
    assert bound.joint_domain_preserved
    assert _euler(_euler(initial, s), s) == initial


@pytest.mark.parametrize(
    "field,value",
    (
        ("nonlinear_oscillation_factor", F(0)),
        ("forcing_step_factor", F(0)),
        (
            "nodal_euler_matrix",
            tuple(tuple(F(i == j) for j in range(6)) for i in range(6)),
        ),
    ),
)
def test_uniform_bound_rebuilds_forged_reference_caches_before_derivation(field, value):
    reference = _reference()
    forged = replace(reference, **{field: value})
    options = dict(
        phase_defect_bound=F(1, 15000), centered_epi_defect_bound=F(1, 10000)
    )
    assert _bound(forged, **options) == _bound(reference, **options)
