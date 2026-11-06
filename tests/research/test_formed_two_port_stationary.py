"""Small exact static controls, without fields, roots, scans or trajectories."""

from fractions import Fraction as Q

import numpy as np
import pytest

from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_symmetric_semidefinite,
)
from tnfr.research import formed_two_port_stationary as owner


def _edges():
    """Independent original-label support for full-matrix comparisons."""
    return tuple(
        (base + i, base + (i + 1) % 9) for base in (0, 9) for i in range(9)
    ) + ((4, 13), (5, 14))


@pytest.fixture(autouse=True)
def no_dynamic_or_transcendental_evaluation(monkeypatch):
    from tnfr.mathematics import _validated_metric
    from tnfr.physics import relational_sine_two_port_compatibility
    from tnfr.research import formed_two_port_reference

    def forbidden(*args, **kwargs):
        pytest.fail("static rational controls must not evaluate dynamics or roots")

    for name in ("build_formed_two_port_reference", "pi_interval", "sin", "cos"):
        monkeypatch.setattr(formed_two_port_reference, name, forbidden)
    monkeypatch.setattr(
        formed_two_port_reference.FormedTwoPortReference, "field", forbidden
    )
    monkeypatch.setattr(_validated_metric, "validated_metric_taylor_step", forbidden)
    monkeypatch.setattr(
        relational_sine_two_port_compatibility, "_root_enclosures", forbidden
    )


def test_circulation_uses_original_order_and_all_port_balances():
    p, q, j = Q(1, 3), Q(-1, 4), Q(1, 5)
    result = owner.assign_formed_two_port_circulation(p, q, j)
    expected = (p,) * 4 + (p - j,) + (p,) * 4
    expected += (q,) * 4 + (q + j,) + (q,) * 4 + (j, -j)
    assert result.edge_currents == expected
    divergence = [Q(0)] * 18
    for (left, right), current in zip(_edges(), expected, strict=True):
        divergence[left] += current
        divergence[right] -= current
    assert tuple(divergence) == result.nodal_divergence == (Q(0),) * 18
    assert "phase_closure_not_established" in result.scope
    boundary = owner.assign_formed_two_port_circulation(1, -1, 1)
    assert max(map(abs, boundary.edge_currents)) == 1


@pytest.mark.parametrize(
    "parameters", [(Q(2), 0, 0), (0, Q(-2), 0), (0, 0, Q(2)), (1, 0, -1), (0, 1, 1)]
)
def test_each_polytope_inequality_is_required(parameters):
    with pytest.raises(ValueError, match="five current magnitudes"):
        owner.assign_formed_two_port_circulation(*parameters)


def _phasors_from_nodes(nodes):
    return tuple(
        (
            nodes[left][0] * nodes[right][0] + nodes[left][1] * nodes[right][1],
            nodes[left][0] * nodes[right][1] - nodes[left][1] * nodes[right][0],
        )
        for left, right in _edges()
    )


def test_exact_stationary_phasors_include_consensus_and_antipodal_saddle():
    consensus = owner.admit_formed_two_port_rational_phases([(1, 0)] * 20)
    assert consensus.phase_stationary and consensus.phase_potential == 0
    nodes = [(Q(1), Q(0))] * 18
    nodes[2] = (Q(-1), Q(0))  # One degree-two antipodal node, not a target claim.
    result = owner.admit_formed_two_port_rational_phases(_phasors_from_nodes(nodes))
    assert result.node_phasors == tuple(nodes)
    assert result.cycle_products == ((Q(1), Q(0)),) * 3
    assert result.phase_stationary and result.phase_potential == 4
    assert "full_A2_equilibrium_also_requires_uniform_form" in result.scope


def test_closed_rational_geometry_need_not_be_stationary():
    nodes = [(Q(1), Q(0))] * 18
    nodes[2] = (Q(3, 5), Q(4, 5))
    result = owner.admit_formed_two_port_rational_phases(_phasors_from_nodes(nodes))
    assert result.node_phasors == tuple(nodes)
    assert not result.phase_stationary
    assert result.nodal_divergence[2] == Q(-8, 5)
    assert result.phase_potential == Q(4, 5)


@pytest.mark.parametrize("broken_edge", [0, 9, 19])
def test_each_independent_cycle_closure_is_required(broken_edge):
    phasors = [(1, 0)] * 20
    phasors[broken_edge] = (-1, 0)
    with pytest.raises(ValueError, match="three independent cycle"):
        owner.admit_formed_two_port_rational_phases(phasors)


def test_divergence_free_current_bounds_do_not_imply_phase_closure():
    current = owner.assign_formed_two_port_circulation(Q(3, 5), 0, 0)
    assert not any(current.nodal_divergence)
    # Nine identical rational unit phasors do not close the donor ring.
    phasors = [(Q(4, 5), Q(3, 5))] * 9 + [(1, 0)] * 11
    with pytest.raises(ValueError, match="cycle products"):
        owner.admit_formed_two_port_rational_phases(phasors)


def _full_hessian(weights):
    matrix = [[Q(0)] * 18 for _ in range(18)]
    for (left, right), weight in zip(_edges(), weights, strict=True):
        matrix[left][left] += weight
        matrix[right][right] += weight
        matrix[left][right] -= weight
        matrix[right][left] -= weight
    return tuple(tuple(row) for row in matrix)


def _independent_schur(matrix):
    ports = (4, 5, 13, 14)
    interior = tuple(i for i in range(18) if i not in ports)
    block = tuple(tuple(matrix[i][j] for j in interior) for i in interior)
    assert exact_symmetric_semidefinite(block, strict=True)
    inverse = exact_matrix_inverse(block)
    return tuple(
        tuple(
            matrix[i][j]
            - sum(
                (
                    matrix[i][r] * inverse[k][l] * matrix[s][j]
                    for k, r in enumerate(interior)
                    for l, s in enumerate(interior)
                ),
                Q(0),
            )
            for j in ports
        )
        for i in ports
    )


@pytest.mark.parametrize(
    "short_d,short_r,psd,pd",
    [
        (Q(1), Q(1), True, True),
        (Q(-1, 4), Q(1), True, True),  # Contacts compensate a negative A_D.
        (Q(-1, 8), Q(-1, 8), True, False),
        (Q(-49, 104), Q(1), True, False),  # Nontrivial exact determinant zero.
        (Q(-1), Q(1), False, False),
    ],
)
def test_condensation_matches_full_matrix_and_relative_signs(short_d, short_r, psd, pd):
    weights = [Q(1)] * 20
    weights[4], weights[13] = short_d, short_r
    result = owner.condense_formed_two_port_hessian(weights)
    matrix = _full_hessian(weights)
    assert result.port_order == (4, 5, 13, 14)
    assert result.bulk_conductances == (Q(1, 8), Q(1, 8))
    assert result.four_port_schur == _independent_schur(matrix)
    grounded = tuple(tuple(row[1:]) for row in matrix[1:])
    assert exact_symmetric_semidefinite(grounded) == psd
    assert exact_symmetric_semidefinite(grounded, strict=True) == pd
    assert result.relative_positive_semidefinite == psd
    assert result.relative_positive_definite == pd
    assert all(sum(row) == 0 for row in result.four_port_schur)


def test_unequal_bulk_conductances_and_difference_coordinates():
    weights = [Q(i % 3 + 1, 3) for i in range(18)] + [Q(1, 2)] * 2
    weights[4], weights[13] = Q(-1, 20), Q(1, 4)
    result = owner.condense_formed_two_port_hessian(weights)
    assert result.four_port_schur == _independent_schur(_full_hessian(weights))
    ports = (Q(1, 3), Q(-2, 5), Q(3, 7), Q(4, 9))
    condensed_energy = sum(
        (
            ports[i] * result.four_port_schur[i][j] * ports[j]
            for i in range(4)
            for j in range(4)
        ),
        Q(0),
    )
    differences = (ports[1] - ports[0], ports[3] - ports[2])
    mean_difference = (ports[2] + ports[3] - ports[0] - ports[1]) / 2
    separated_energy = 2 * result.contact_weight * mean_difference**2 + sum(
        (
            differences[i] * result.difference_matrix[i][j] * differences[j]
            for i in range(2)
            for j in range(2)
        ),
        Q(0),
    )
    assert condensed_energy == separated_energy
    assert (
        "supplied_edge_coefficients_do_not_establish_phase_feasibility_or_stationarity"
        in result.scope
    )


@pytest.mark.parametrize(
    "index,value", [(0, 0), (9, -1), (18, 0), (19, Q(1, 2)), (4, 2)]
)
def test_condensation_rejects_missing_weight_premises(index, value):
    weights = [1] * 20
    weights[index] = value
    with pytest.raises(ValueError):
        owner.condense_formed_two_port_hessian(weights)


@pytest.mark.parametrize(
    "value", [True, False, 1.0, 0.0, "1", 1j, {"numerator": 1, "denominator": 1}]
)
def test_exact_scalar_admission_precedes_equal_value_comparisons(value):
    with pytest.raises(TypeError):
        owner.assign_formed_two_port_circulation(value, 0, 0)
    weights = [1] * 20
    weights[4] = value
    with pytest.raises(TypeError):
        owner.condense_formed_two_port_hessian(weights)
    phasors = [(1, 0)] * 20
    phasors[0] = (value, 0)
    with pytest.raises(TypeError):
        owner.admit_formed_two_port_rational_phases(phasors)


def test_fraction_payloads_are_normalized_before_arithmetic():
    half = Q(np.int64(1), np.int64(2))
    result = owner.assign_formed_two_port_circulation(half, -half, half)
    assert all(type(value.numerator) is int for value in result.edge_currents)
    assert all(type(value.denominator) is int for value in result.edge_currents)
    condensed = owner.condense_formed_two_port_hessian([half] * 20)
    assert condensed.bulk_conductances == (Q(1, 16),) * 2
    assert all(type(value.numerator) is int for value in condensed.edge_cosines)


def test_shapes_order_and_unit_norm_are_admitted():
    with pytest.raises(ValueError, match="exactly 20"):
        owner.condense_formed_two_port_hessian([1] * 19)
    with pytest.raises(TypeError, match="ordered"):
        owner.condense_formed_two_port_hessian({1})
    with pytest.raises(ValueError, match="exactly 20"):
        owner.admit_formed_two_port_rational_phases([(1, 0)] * 19)
    phasors = [(1, 0)] * 20
    phasors[0] = (1, 1)
    with pytest.raises(ValueError, match="unit norm"):
        owner.admit_formed_two_port_rational_phases(phasors)
    phasors[0] = (1, 0, 0)
    with pytest.raises(ValueError, match="exactly 2"):
        owner.admit_formed_two_port_rational_phases(phasors)
