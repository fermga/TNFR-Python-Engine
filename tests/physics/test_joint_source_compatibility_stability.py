"""A compatible source point need not admit a stable joint stationary point.

These exact local controls concern the stated C2 potential family on an open
EPI/phase domain, with positive block mobility and held support/capacity.
They select neither Psi nor a constitutive law. Symbolic jets and detached
exact profiles are not graph updates or derivatives of binary64 execution.
"""

from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    _exact_generator,
    _graph,
    _prepared_phase_metric,
)
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.support_transport import observe_support_transport


def _prepared_forcing_jet():
    s, graph, phases, metric, source, jacobian = _prepared_phase_metric()
    forcing = source / 4
    forcing_jacobian = jacobian / 4
    compatibility_gradient = forcing_jacobian.T * (3 * s.ones(6, 1))
    return s, graph, phases, metric, forcing, forcing_jacobian, compatibility_gradient


def test_compatible_epi_offsets_preserve_pressure_but_change_reciprocal_phase_rate():
    s, graph, _, metric, forcing, jacobian, gamma = _prepared_forcing_jet()
    source = observe_support_transport(graph)
    reference = derive_forced_support_balance(
        source,
        epi_weight=Q(1, 2),
        forcing=(Q(1, 24), Q(-1, 24), Q(0)) * 2,
    )
    assert reference.has_zero_pressure_equilibrium
    assert reference.compatibility_residual == 0
    assert reference.relative_profile == (Q(1, 12), Q(-1, 12), Q(0)) * 2
    assert s.Matrix(reference.forcing) == forcing
    first = s.ones(6, 1) / 2 + s.Matrix(reference.relative_profile)
    shift = s.Rational(1, 8)
    second = first + shift * s.ones(6, 1)
    generator = s.Matrix(_exact_generator(source)) / 2
    assert generator * first + forcing == generator * second + forcing == s.zeros(6, 1)
    # These are exact reference profiles; neither is materialized as live EPI.
    assert (3 * s.ones(1, 6) * forcing)[0] == 0
    assert gamma != s.zeros(6, 1)
    psi_gradient = s.Matrix(s.symbols("psi_theta0:6", real=True))
    velocities = tuple(
        metric.inv() * (jacobian.T * (3 * x) - psi_gradient) for x in (first, second)
    )
    difference = s.simplify(velocities[1] - velocities[0])
    assert s.simplify(difference - shift * metric.inv() * gamma) == s.zeros(6, 1)
    assert difference != s.zeros(6, 1)
    # Every EPI-independent Psi cancels from this comparison. C=0 at the
    # shared phase point does not make the complete joint flow shift-invariant.


def test_centering_a_source_changes_its_derivative_even_at_a_compatible_point():
    s, _, _, _, forcing, jacobian, gamma = _prepared_forcing_jet()
    ones = s.ones(6, 1)
    degree = 3 * s.eye(6)
    total_degree = 18
    compatibility = (ones.T * degree * forcing)[0]
    centered_forcing = forcing - ones * compatibility / total_degree
    assert centered_forcing == forcing
    corrected_jacobian = jacobian - ones * gamma.T / total_degree
    assert s.simplify(corrected_jacobian.T * degree * ones) == s.zeros(6, 1)
    assert s.simplify(corrected_jacobian - jacobian) != s.zeros(6)
    original_mixed = -degree * jacobian
    centered_mixed = -degree * corrected_jacobian
    expected_correction = ones * gamma.T / 6
    assert s.simplify(centered_mixed - original_mixed - expected_correction) == s.zeros(
        6
    )
    assert s.simplify(ones.T * original_mixed + gamma.T) == s.zeros(1, 6)
    assert s.simplify(ones.T * centered_mixed) == s.zeros(1, 6)
    # Replacing -x^T D F with -(x-mu_D(x)*1)^T D F adds mu_D*C.
    # Its x derivative adds d*C/sum(d), and its mixed derivative remains
    # nonzero at C=0. This is a changed source law, not a harmless gauge fix.


def test_any_stationary_c2_joint_potential_has_an_indefinite_mean_phase_plane():
    s, graph, phases, _, _, _, gamma = _prepared_forcing_jet()
    amplitude, time, curvature = s.symbols("a t k", real=True)
    phase_direction = s.Matrix((0, 0, 1) * 2)
    # Move the middle phase of both triangles. Rotating each neighbor sum
    # by -pi/6 gives sqrt(3)+exp(i*t), with positive real part near t=0.
    mean_change = s.atan(s.sin(time) / (s.sqrt(3) + s.cos(time)))
    compatibility = (18 * mean_change - 6 * time) / (4 * s.pi)
    transverse = s.simplify((gamma.T * phase_direction)[0])
    assert compatibility.subs(time, 0) == 0
    assert s.simplify(s.diff(compatibility, time).subs(time, 0) - transverse) == 0
    assert transverse.is_positive
    # At any full stationary point the phase-axis linear term vanishes.
    # Its arbitrary second derivative k includes every possible C2 Psi.
    local_jet = -amplitude * compatibility + curvature * time**2 / 2
    hessian = s.hessian(local_jet, (amplitude, time)).subs({amplitude: 0, time: 0})
    expected = s.Matrix([[0, -transverse], [-transverse, curvature]])
    assert s.simplify(hessian - expected) == s.zeros(2)
    assert s.simplify(hessian.det() + transverse**2) == 0
    assert (-(transverse**2)).is_negative
    descent = s.Matrix([(curvature + 1) / (2 * transverse), 1])
    assert s.simplify((descent.T * hessian * descent)[0]) == -1
    # A positive source-balanced EPI reference and strict U3 leave an open
    # neighborhood for sufficiently small multiples of this finite direction.
    epi_reference = s.ones(6, 1) / 2 + s.Matrix((1, -1, 0) * 2) / 12
    assert min(epi_reference) == s.Rational(5, 12) > 0
    assert max(epi_reference) == s.Rational(7, 12) < 1
    index = {node: i for i, node in enumerate(NODES)}
    maximum_gap = max(abs(phases[index[i]] - phases[index[j]]) for i, j in graph.edges)
    assert maximum_gap == s.pi / 3 < s.pi / 2
    # With zero first derivative, the C2 remainder is o(step^2), so the
    # negative Hessian direction gives actual local descent. If the full
    # gradient is nonzero, an interior point already cannot be a minimum.


def test_trial_alignment_psi_leaves_the_initial_compatible_source_set():
    s, _, _, metric, forcing, jacobian, gamma = _prepared_forcing_jet()
    mean = s.Rational(1, 2)
    internal = s.Matrix((1, -1, 0) * 2) / 12
    epi = mean * s.ones(6, 1) + internal
    phase_source = 4 * forcing
    assert s.simplify((gamma.T * phase_source)[0]) == 0
    assert s.simplify((gamma.T * metric.inv() * jacobian.T * (3 * internal))[0]) == 0
    # A hypothetical Psi=V_phi block-gradient completion, not an installed law.
    phase_velocity = phase_source + metric.inv() * jacobian.T * (3 * epi)
    compatibility_rate = s.simplify((gamma.T * phase_velocity)[0])
    norm_squared = sum(gamma[i] ** 2 / metric[i, i] for i in range(6))
    assert all(value.is_positive for value in metric.diagonal())
    assert norm_squared.is_positive
    assert s.simplify(compatibility_rate - mean * norm_squared) == 0
    assert compatibility_rate.is_positive
    # Being compatible initially supplies no invariant-domain certificate;
    # this exact tangent leaves it despite unchanged support and capacity.


def test_only_consensus_repeated_phase_triples_have_derivative_compatibility():
    s = pytest.importorskip("sympy")
    graph = _graph()
    resultant, weight = s.symbols("resultant weight", positive=True)
    cosines = s.symbols("c0:3", real=True)
    # Both fibers carry the same arbitrary triple. Every neighbor row sees
    # each of those three phases exactly once, so all resultants agree.
    neighbors = tuple(tuple(graph.neighbors(node)) for node in NODES)
    assert all(sorted(node[1] for node in row) == [0, 1, 2] for row in neighbors)
    response = s.Matrix(
        6,
        6,
        lambda i, j: (
            cosines[NODES[j][1]] / resultant if NODES[j] in neighbors[i] else 0
        ),
    )
    column_sums = s.ones(1, 6) * response
    assert column_sums == s.Matrix([[3 * value / resultant for value in cosines] * 2])
    compatibility_gradient = (
        weight * (response.T - s.eye(6)) * (3 * s.ones(6, 1)) / s.pi
    )
    expected = s.Matrix(
        [weight * (9 * value / resultant - 3) / s.pi for value in cosines] * 2
    )
    assert s.simplify(compatibility_gradient - expected) == s.zeros(6, 1)
    solution = s.solve(list(compatibility_gradient[:3]), cosines, dict=True)
    assert solution == [dict.fromkeys(cosines, resultant / 3)]
    assert compatibility_gradient.subs(weight, 0) == s.zeros(6, 1)
    # The implication therefore requires w_phi>0 and a nonzero resultant.
    # At a regular phase-source point, c_j=cos(theta_j-beta), with beta the
    # shared resultant direction. Equal c_j imply equal sine squares h.
    sine0, sine1, sine2, squared_sine = s.symbols("s0 s1 s2 h", real=True)
    constraints = (
        sine0 + sine1 + sine2,
        sine0**2 - squared_sine,
        sine1**2 - squared_sine,
        sine2**2 - squared_sine,
    )
    ideal = s.groebner(constraints, sine0, sine1, sine2, squared_sine)
    assert ideal.reduce(squared_sine**2)[1] == 0
    # This polynomial consequence proves h^2=0, hence h=0: three signed
    # equal nonzero sine magnitudes cannot sum to zero. No phases are sampled.
    assert s.solveset(squared_sine**2, squared_sine, domain=s.S.Reals) == s.FiniteSet(0)
    assert s.solveset(
        1 - resultant**2 / 9,
        resultant,
        domain=s.Interval.open(0, s.oo),
    ) == s.FiniteSet(3)
    assert {key: value.subs(resultant, 3) for key, value in solution[0].items()} == (
        dict.fromkeys(cosines, 1)
    )
    # Thus every phase equals beta on the circle. Positive equal cosines
    # already enforce beta's open half-pi domain; no semicircle assumption
    # was added. This does not cover arbitrary unequal six-phase states,
    # zero-resultant/branch points, or w_phi=0.
