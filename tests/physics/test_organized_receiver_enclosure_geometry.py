"""Independent static geometry checks; no receiver trajectory is evaluated."""

from fractions import Fraction as Q
from math import cos as scalar_cos
from math import pi
from math import sin as scalar_sin
from math import sqrt as scalar_sqrt

import numpy as np
import pytest

from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import _organized_receiver_enclosure_geometry as owner


def _independent_edges():
    edges = [(9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in range(3) for j in range(9)]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for j in range(9):
            for offset in (0, 1):
                edges.append((9 * a + j, 9 * b + (j + offset) % 9, Q(1, 2)))
    return tuple(edges)


def _target():
    return tuple(2 * pi_interval() * (node % 9 - 4) / 9 for node in range(27))


def test_reflection_preserves_support_and_exact_zero_mean_reference():
    edges = _independent_edges()
    assert owner._receiver_weighted_edges() == edges
    reflection = owner._REFERENCE_REFLECTION
    assert tuple(reflection[node] for node in reflection) == tuple(range(27))
    assert [node for node in range(27) if reflection[node] == node] == [22]
    values = tuple(Q(i - 6, 8) for i in range(13))
    full = owner._lift_receiver_reference(values)
    assert sum(value.lo for value in full) == 0
    assert full[22] == I(0)
    for node in range(27):
        assert full[reflection[node]] == -full[node]


def test_restricted_flow_matches_independent_full_edge_sum_and_symmetry():
    values = tuple(Q((14 * i) % 17 - 8, 3) for i in range(13))
    full = owner._lift_receiver_reference(values)
    phases = tuple(float(value.midpoint) for value in full)
    rates = np.zeros(27)
    for a, b, weight in _independent_edges():
        current = float(weight) * scalar_sin(phases[b] - phases[a]) / 4
        rates[a] += current
        rates[b] -= current
    enclosed = owner._receiver_reference_flow(values)
    for node, value in zip(owner._REFERENCE_REPRESENTATIVES, enclosed):
        assert float(value.midpoint) == pytest.approx(rates[node], abs=1e-14)
    assert rates[22] == pytest.approx(0, abs=1e-14)
    assert rates.sum() == pytest.approx(0, abs=1e-14)
    for node in range(27):
        assert rates[owner._REFERENCE_REFLECTION[node]] == pytest.approx(
            -rates[node], abs=1e-14
        )


def test_flow_jets_retain_independent_directional_derivative():
    values = tuple(Jet((I(0), I(Q(i - 6, 8)))) for i in range(13))
    direction = owner._lift_receiver_reference(tuple(Q(i - 6, 8) for i in range(13)))
    expected = [Q(0)] * 27
    for a, b, weight in _independent_edges():
        derivative = weight * (direction[b].lo - direction[a].lo) / 4
        expected[a] += derivative
        expected[b] -= derivative
    for node, value in zip(
        owner._REFERENCE_REPRESENTATIVES, owner._receiver_reference_flow(values)
    ):
        assert value.coeffs[0].contains(0)
        assert value.coeffs[1].contains(expected[node])


def test_full_mean_zero_osl_keeps_the_nonsymmetric_error_directions():
    # Here the symmetric mean-zero block grows faster than the nominal
    # antisymmetric reference block; the latter alone is an invalid bound.
    phases = owner._lift_receiver_reference(
        tuple(Q((14 * i) % 17 - 8, 3) for i in range(13))
    )
    report = owner._receiver_local_osl(phases, phase_radius=Q(1, 10000))
    assert report.certified and report.bisections_performed == 8
    jacobian = [[Q(0) for _ in range(27)] for _ in range(27)]
    for (a, b, weight), cosine_lower in zip(
        _independent_edges(), report.edge_cosine_lower_bounds
    ):
        coefficient = weight * cosine_lower / 4
        jacobian[a][a] -= coefficient
        jacobian[b][b] -= coefficient
        jacobian[a][b] += coefficient
        jacobian[b][a] += coefficient
    # Independent 26-coordinate basis e_i-e_26, with Gram I+11^T.
    rate = report.growth_rate_upper_bound
    slack = tuple(
        tuple(
            rate * (1 + int(i == j))
            - (jacobian[i][j] - jacobian[i][26] - jacobian[26][j] + jacobian[26][26])
            for j in range(26)
        )
        for i in range(26)
    )
    assert exact_symmetric_semidefinite(slack)
    matrix = np.asarray(jacobian, dtype=float)
    anti = np.zeros((27, 13))
    for column, node in enumerate(owner._REFERENCE_REPRESENTATIVES):
        anti[node, column] = 1
        anti[owner._REFERENCE_REFLECTION[node], column] = -1
    anti_rate = np.linalg.eigvalsh(anti.T @ matrix @ anti / 2)[-1]
    full_rate = np.linalg.eigvalsh(matrix)[-1]
    assert anti_rate < 0.55 < 0.69 < full_rate <= float(rate)


def test_osl_accepts_nonsymmetric_boxes_and_signed_relative_contraction():
    flat = owner._receiver_local_osl((I(0),) * 27, phase_radius=Q(1, 1000))
    assert flat.growth_rate_upper_bound < 0
    box = tuple(I(Q(j % 7, 5), Q(j % 7, 5) + Q(1, 100)) for j in range(27))
    report = owner._receiver_local_osl(box, phase_radius=Q(1, 20))
    cosine_by_edge = {
        frozenset((a, b)): value
        for (a, b, _), value in zip(
            _independent_edges(), report.edge_cosine_lower_bounds
        )
    }
    for a, b, _ in _independent_edges():
        reflected = frozenset(
            (owner._REFERENCE_REFLECTION[a], owner._REFERENCE_REFLECTION[b])
        )
        assert cosine_by_edge[frozenset((a, b))] == cosine_by_edge[reflected]
        assert cosine_by_edge[frozenset((a, b))] * (1 << 20) % 1 == 0


def test_target_capture_retains_full_form_and_phase_error():
    report = owner._observe_receiver_tube(
        _target(), phase_error_bound=Q(1, 10000), form_norm_upper_bound=Q(1, 10000)
    )
    assert report.windings == (1, 1, 1)
    assert min(report.donor_registered_branch_margins) > 2
    assert report.capture_inequalities_pass
    assert float(report.complete_reference_potential_bounds.midpoint) == pytest.approx(
        40.5 * (1 - scalar_cos(2 * pi / 9)), abs=1e-13
    )
    assert float(report.complete_capture_excess_upper_bound) == pytest.approx(
        8e-8, abs=1e-25
    )
    high_form = owner._observe_receiver_tube(
        _target(), phase_error_bound=Q(0), form_norm_upper_bound=Q(1, 2)
    )
    assert high_form.capture_cell_margin > 0
    assert high_form.capture_energy_margin < 0
    assert not high_form.capture_inequalities_pass


def test_capture_keeps_the_nonzero_linear_gradient_term():
    phase = list(_target())
    phase[0] += Q(1, 20)
    radius, form = Q(1, 100), Q(1, 1000)
    report = owner._observe_receiver_tube(
        phase, phase_error_bound=radius, form_norm_upper_bound=form
    )
    gradient = np.zeros(27)
    potential = 0
    for a, b, weight in _independent_edges():
        difference = float(phase[b].midpoint - phase[a].midpoint)
        current = float(weight) * scalar_sin(difference)
        gradient[a] -= current
        gradient[b] += current
        potential += float(weight) * (1 - scalar_cos(difference))
    norm = scalar_sqrt(sum(value**2 for value in gradient))
    assert norm > 0.1
    assert float(report.reference_gradient_norm_upper_bound) == pytest.approx(norm)
    expected = (
        potential
        - 40.5 * (1 - scalar_cos(2 * pi / 9))
        + norm * float(radius)
        + 4 * float(radius**2 + form**2)
    )
    assert float(report.complete_capture_excess_upper_bound) == pytest.approx(
        expected, abs=1e-13
    )


def test_winding_does_not_replace_continuous_lift_capture_registration():
    phase = list(_target())
    phase[0] += 2 * pi_interval()
    report = owner._observe_receiver_tube(
        phase, phase_error_bound=Q(0), form_norm_upper_bound=Q(0)
    )
    assert report.windings == (1, 1, 1)
    assert report.capture_energy_margin > 0
    assert report.capture_cell_margin < 0
    assert not report.capture_inequalities_pass


def test_branch_ambiguity_is_unavailable_not_zero_winding():
    phase = [I(0)] * 27
    phase[18], phase[26] = -pi_interval() / 2, pi_interval() / 2
    report = owner._observe_receiver_tube(
        phase, phase_error_bound=Q(0), form_norm_upper_bound=Q(0)
    )
    assert report.windings == (0, 0, None)


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), Q(-1)])
def test_invalid_radius_is_rejected_before_geometry(monkeypatch, invalid):
    def forbidden():
        raise AssertionError("geometry must follow radius admission")

    monkeypatch.setattr(owner, "_receiver_weighted_edges", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._receiver_local_osl((I(0),) * 27, phase_radius=invalid)
    with pytest.raises((TypeError, ValueError)):
        owner._observe_receiver_tube(
            (I(0),) * 27, phase_error_bound=invalid, form_norm_upper_bound=Q(0)
        )


def test_shape_and_jet_admission_are_explicit():
    with pytest.raises(ValueError, match="dimension 13"):
        owner._lift_receiver_reference((I(0),) * 12)
    with pytest.raises(ValueError, match="27 phase"):
        owner._receiver_local_osl((I(0),) * 26, phase_radius=Q(0))
    with pytest.raises(TypeError, match="all intervals or all jets"):
        owner._lift_receiver_reference((Jet.constant(0, 1),) + (I(0),) * 12)
    with pytest.raises(ValueError, match="common order"):
        owner._lift_receiver_reference(
            (Jet.constant(0, 1),) + (Jet.constant(0, 2),) * 12
        )
