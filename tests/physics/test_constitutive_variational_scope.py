"""Exact conditional variational constraints from canonical pressure owners."""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.physics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
)
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from tnfr.physics.phase_response import derive_phase_response
from tnfr.physics.support_transport import _from_data

_EPI = (Q(1, 4), Q(1), Q(3, 4))
_CAPACITY = (Q(1, 2), Q(1), Q(3, 2))
_XI0 = Q(1, 4)
_ETA0 = Q(-1, 2)
_E = Q(1, 2)
_V = Q(1, 4)


def _source(epi=_EPI, capacity=_CAPACITY, edge_weights=(1, 2)):
    graph = nx.path_graph(3)
    for (left, right), weight in zip(graph.edges, edge_weights, strict=True):
        graph.edges[left, right]["weight"] = float(weight)
    for node, x, nu in zip(graph, epi, capacity, strict=True):
        graph.nodes[node].update(EPI=float(x), nu_f=float(nu), theta=0.0)
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0,
        "epi": 0.5,
        "vf": 0.25,
        "topo": 0.25,
    }
    observation = capture_non_epi_forcing(graph)
    reference = derive_forced_support_balance(
        observation.snapshot,
        epi_weight=observation.epi_weight,
        forcing=observation.forcing,
    )
    return observation, reference


def _replace_epi(snapshot, epi):
    return _from_data(
        snapshot.nodes,
        snapshot.conductance,
        snapshot.support_neighbors,
        epi,
        snapshot.capacity,
        snapshot.stored_pressure,
    )


def _forced_potential(reference, snapshot):
    # Conditional Psi=0 slice, used only to check the forced x dependence.
    # Any x-independent Psi cancels in all mixed differences below.
    return reference.epi_weight * snapshot.dirichlet_energy - sum(
        d * x * forcing
        for d, x, forcing in zip(
            reference.strengths,
            snapshot.epi,
            reference.forcing,
            strict=True,
        )
    )


def test_forced_potential_realizes_the_weighted_multichannel_epi_equation():
    observation, reference = _source()
    snapshot = observation.snapshot
    channels = dict(decompose_non_epi_forcing(observation))
    assert reference.strengths == (1, 3, 2)
    assert channels == {
        "phase": (0, 0, 0),
        "vf": (Q(1, 8), 0, Q(-1, 8)),
        "topo": (Q(1, 4), Q(-1, 4), Q(1, 4)),
    }
    gradient = tuple(
        reference.epi_weight * bx - d * forcing
        for bx, d, forcing in zip(
            snapshot.dirichlet_gradient,
            reference.strengths,
            reference.forcing,
            strict=True,
        )
    )
    rate_from_gradient = tuple(
        -nu * derivative / d
        for nu, derivative, d in zip(
            snapshot.capacity,
            gradient,
            reference.strengths,
            strict=True,
        )
    )
    rate_from_pressure = tuple(
        nu * (reference.epi_weight * epi_gradient + forcing)
        for nu, epi_gradient, forcing in zip(
            snapshot.capacity,
            snapshot.epi_gradient,
            reference.forcing,
            strict=True,
        )
    )
    assert rate_from_gradient == rate_from_pressure

    direction = (1, -2, 3)
    positive = _replace_epi(
        snapshot, tuple(x + h for x, h in zip(snapshot.epi, direction))
    )
    negative = _replace_epi(
        snapshot, tuple(x - h for x, h in zip(snapshot.epi, direction))
    )
    # A centered finite difference is exact for this quadratic polynomial;
    # no numerical derivative, epsilon or trajectory is involved.
    difference = (
        _forced_potential(reference, positive) - _forced_potential(reference, negative)
    ) / 2
    assert difference == sum(g * h for g, h in zip(gradient, direction))


def test_capacity_pressure_forces_nonzero_reciprocal_mixed_partial():
    # Vary x_0 and nu_1 independently. The exact mixed rectangle measures
    # partial_nu1 partial_x0 E = v*d_0*(L_U)_01 = -1/4.
    epi_shifted = (_EPI[0] + 1, *_EPI[1:])
    capacity_shifted = (_CAPACITY[0], _CAPACITY[1] + 1, _CAPACITY[2])
    values = []
    for epi, capacity in (
        (epi_shifted, capacity_shifted),
        (epi_shifted, _CAPACITY),
        (_EPI, capacity_shifted),
        (_EPI, _CAPACITY),
    ):
        observation, reference = _source(epi, capacity)
        values.append(_forced_potential(reference, observation.snapshot))
    assert values[0] - values[1] - values[2] + values[3] == Q(-1, 4)
    # An x-independent capacity force under an x-independent invertible
    # block mobility has zero opposite mixed derivative, so cannot equal it.


def test_incompatible_source_gives_exact_unbounded_common_offset_tilt():
    observation, reference = _source()
    assert reference.forcing == (Q(3, 8), Q(-1, 4), Q(1, 8))
    assert reference.compatibility_residual == Q(-1, 8)
    assert not reference.has_zero_pressure_equilibrium
    baseline = _forced_potential(reference, observation.snapshot)
    for offset in (-2, 1, 4):
        shifted = _replace_epi(
            observation.snapshot,
            tuple(x + offset for x in observation.snapshot.epi),
        )
        assert shifted.dirichlet_energy == observation.snapshot.dirichlet_energy
        assert (
            _forced_potential(reference, shifted) - baseline
            == -offset * reference.compatibility_residual
        )
    # The formula holds for arbitrary real offsets; the unboundedness proof
    # is the nonzero linear coefficient, not extrapolation from these values.


def test_compatible_source_uses_existing_poisson_profile_to_complete_square():
    observation, reference = _source(edge_weights=(1, 1))
    assert reference.compatibility_residual == 0
    assert reference.has_zero_pressure_equilibrium
    snapshot = observation.snapshot
    profile = _replace_epi(snapshot, reference.relative_profile)
    centered = _replace_epi(
        snapshot,
        tuple(x - z for x, z in zip(snapshot.epi, reference.relative_profile)),
    )
    completed = reference.epi_weight * (
        centered.dirichlet_energy - profile.dirichlet_energy
    )
    assert _forced_potential(reference, snapshot) == completed
    assert centered.dirichlet_energy >= 0


def _star_phase_response(cosine):
    gram = tuple(
        tuple(Q(1) if (i == 3) == (j == 3) else cosine for j in range(4))
        for i in range(4)
    )
    return derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=((1, 2, 3), (0,), (0,), (0,)),
        receiver_sources=((0,), (1,), (2,), (3,)),
        phase_factor=1,
    )


def test_strict_u3_star_refutes_constant_diagonal_phase_gradient_metric():
    consensus = _star_phase_response(Q(1))
    changed = _star_phase_response(Q(3, 5))
    assert consensus.mean_response[0] == (0, Q(1, 3), Q(1, 3), Q(1, 3))
    assert changed.mean_response[0] == (0, Q(13, 37), Q(13, 37), Q(11, 37))
    assert changed.mean_resultant_squared == (Q(37, 5), 1, 1, 1)
    assert all(changed.cosine_gram[0][j] > 0 for j in (1, 2, 3))
    for reference in (consensus, changed):
        assert tuple(reference.mean_response[j][0] for j in (1, 2, 3)) == (1, 1, 1)

    # Consensus reciprocity uniquely fixes the positive diagonal metric ray.
    metric = (3, 1, 1, 1)
    assert all(
        metric[i] * consensus.mean_response[i][j]
        == metric[j] * consensus.mean_response[j][i]
        for i in range(4)
        for j in range(4)
    )
    defects = tuple(
        metric[0] * changed.mean_response[0][j]
        - metric[j] * changed.mean_response[j][0]
        for j in (1, 2, 3)
    )
    assert defects == (Q(2, 37), Q(2, 37), Q(-4, 37))
    # The analytic identity (1-c)/(5+4*c)>0 for every c in (0,1)
    # gives the open-neighborhood obstruction; this is its exact owner control.
    assert defects[0] == (1 - Q(3, 5)) / (5 + 4 * Q(3, 5))


def _p2_source(xi=_XI0, eta=_ETA0):
    """Detached dyadic states, with fixed common means and zero phase."""
    graph = nx.path_graph(2)
    epi = (Q(1, 8) + xi / 2, Q(1, 8) - xi / 2)
    capacity = (Q(5, 4) + eta / 2, Q(5, 4) - eta / 2)
    for node, x, nu in zip(graph, epi, capacity, strict=True):
        graph.nodes[node].update(EPI=float(x), nu_f=float(nu), theta=0.0, delta_nfr=0.0)
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.25,
        "epi": 0.5,
        "vf": 0.25,
        "topo": 0.0,
    }
    observation = capture_non_epi_forcing(graph)
    assert observation.snapshot.epi == epi
    assert observation.snapshot.capacity == capacity
    assert observation.kernel_pressure_defect == (0, 0)
    reference = derive_forced_support_balance(
        observation.snapshot,
        epi_weight=observation.epi_weight,
        forcing=observation.forcing,
    )
    return graph, observation, reference


def _p2_base_potential(xi, eta):
    _, observation, reference = _p2_source(xi, eta)
    return _forced_potential(reference, observation.snapshot)


def _p2_conditional_potential(xi, eta, curvature):
    # A logical completion of the forced potential, not an installed law.
    # The linear term is required for joint stationarity at the baseline.
    b = eta - _ETA0
    psi = -_V * _XI0 * b + curvature * b * b / 2
    return _p2_base_potential(xi, eta) + psi


def _centered_p2_jet(potential):
    # These differences are exact for the degree-two potentials used here.
    # Every capture remains in a small, positive-capacity dyadic neighborhood.
    h = Q(1, 16)
    center = potential(_XI0, _ETA0)
    xp = potential(_XI0 + h, _ETA0)
    xm = potential(_XI0 - h, _ETA0)
    yp = potential(_XI0, _ETA0 + h)
    ym = potential(_XI0, _ETA0 - h)
    mixed = (
        potential(_XI0 + h, _ETA0 + h)
        - potential(_XI0 + h, _ETA0 - h)
        - potential(_XI0 - h, _ETA0 + h)
        + potential(_XI0 - h, _ETA0 - h)
    ) / (4 * h * h)
    return (
        ((xp - xm) / (2 * h), (yp - ym) / (2 * h)),
        (
            ((xp - 2 * center + xm) / (h * h), mixed),
            (mixed, (yp - 2 * center + ym) / (h * h)),
        ),
    )


def test_zero_pressure_does_not_make_the_uncompleted_joint_potential_stationary():
    _, observation, reference = _p2_source()
    snapshot = observation.snapshot
    assert snapshot.epi == (Q(1, 4), 0)
    assert snapshot.capacity == (1, Q(3, 2))
    assert observation.epi_weight == _E
    assert dict(observation.normalized_weights)["vf"] == _V
    assert dict(decompose_non_epi_forcing(observation)) == {
        "phase": (0, 0),
        "vf": (Q(1, 8), Q(-1, 8)),
        "topo": (0, 0),
    }
    assert observation.forcing == (-_V * _ETA0, _V * _ETA0)
    assert observation.full_kernel_pressure == snapshot.stored_pressure == (0, 0)
    assert observation.stored_pressure_residual == snapshot.rate == (0, 0)
    assert reference.compatibility_residual == 0
    assert _forced_potential(reference, snapshot) == (
        _E * _XI0 * _XI0 / 2 + _V * _XI0 * _ETA0
    )

    gradient, hessian = _centered_p2_jet(_p2_base_potential)
    assert gradient == (0, Q(1, 16))
    assert hessian == ((_E, _V), (_V, 0))
    # Psi=0 is not a joint critical point: it needs Psi_eta=-1/16.
    # The reciprocal mixed curvature comes from fresh canonical source reads.
    assert -gradient[1] == -_V * _XI0 == Q(-1, 16)


@pytest.mark.parametrize(
    ("curvature", "semidefinite", "definite"),
    (
        (Q(0), False, False),
        (Q(1, 16), False, False),
        (Q(1, 8), True, False),
        (Q(1, 4), True, True),
    ),
)
def test_stationary_joint_potential_requires_curvature_for_a_restricted_minimum(
    curvature,
    semidefinite,
    definite,
):
    gradient, hessian = _centered_p2_jet(
        lambda xi, eta: _p2_conditional_potential(xi, eta, curvature)
    )
    assert gradient == (0, 0)
    assert hessian == ((_E, _V), (_V, curvature))
    assert exact_symmetric_semidefinite(hessian) is semidefinite
    assert exact_symmetric_semidefinite(hessian, strict=True) is definite
    assert _V * _V / _E == Q(1, 8)
    assert semidefinite is (curvature >= _V * _V / _E)
    # This is necessary for a minimum on the fixed-mean, fixed-phase slice;
    # positive definiteness here makes no claim about other nodal directions.


@pytest.mark.parametrize("curvature", (Q(0), Q(1, 16)))
def test_insufficient_restoring_curvature_has_an_admissible_descent_direction(
    curvature,
):
    direction = (-_V / _E, Q(1))
    assert direction == (Q(-1, 2), 1)
    hessian = ((_E, _V), (_V, curvature))
    directional_form = sum(
        direction[i] * hessian[i][j] * direction[j] for i in range(2) for j in range(2)
    )
    assert directional_form == curvature - Q(1, 8) < 0
    base = _p2_conditional_potential(_XI0, _ETA0, curvature)
    for s in (Q(-1, 16), Q(1, 16)):
        a, b = (s * value for value in direction)
        xi, eta = _XI0 + a, _ETA0 + b
        _, observation, _ = _p2_source(xi, eta)
        assert all(Q(3, 4) < nu < Q(7, 4) for nu in observation.snapshot.capacity)
        assert observation.full_kernel_pressure == (0, 0)
        completed = _E * (a + _V * b / _E) ** 2 / 2
        completed += (curvature - _V * _V / _E) * b * b / 2
        difference = _p2_conditional_potential(xi, eta, curvature) - base
        assert difference == completed == directional_form * s * s / 2 < 0
    # The exact quadratic identity holds for every sufficiently small nonzero
    # s. These are detached nearby states, not a zero-pressure EPI trajectory.


def test_equal_hessians_at_the_restoring_boundary_allow_minimum_saddle_or_valley():
    sympy = pytest.importorskip("sympy")
    a, b = sympy.symbols("a b", real=True)
    threshold = _V * _V / _E
    base = _E * (_XI0 + a) ** 2 / 2 + _V * (_XI0 + a) * (_ETA0 + b)
    # All three conditional Psi choices have the necessary nonzero slope.
    psi = -_V * _XI0 * b + threshold * b * b / 2
    quadratic = sympy.expand(base + psi - base.subs({a: 0, b: 0}))
    assert sympy.expand(quadratic - _E * (a + _V * b / _E) ** 2 / 2) == 0
    for sign in (-1, 0, 1):
        polynomial = quadratic + sign * b**4
        assert sympy.diff(psi + sign * b**4, b).subs(b, 0) == Q(-1, 16)
        assert tuple(sympy.diff(polynomial, q).subs({a: 0, b: 0}) for q in (a, b)) == (
            0,
            0,
        )
        assert sympy.hessian(polynomial, (a, b)).subs({a: 0, b: 0}) == (
            sympy.Matrix(((_E, _V), (_V, threshold)))
        )
        assert sympy.expand(polynomial.subs(a, -_V * b / _E)) == sign * b**4
        assert polynomial.subs(b, 0) == _E * a * a / 2
    # +b^4 is a strict minimum: both nonnegative summands vanish only at 0.
    # -b^4 is a saddle, and the quadratic alone has an entire null valley.
    # Same gradient/Hessian cannot distinguish these local behaviors; these
    # polynomials are logical counterexamples, not proposed capacity laws.


def test_zero_pressure_curve_and_complete_tetrad_do_not_detect_joint_energy_drop():
    first_graph, first, _ = _p2_source()
    s = Q(1, 4)
    second_graph, second, _ = _p2_source(_XI0 - s / 2, _ETA0 + s)
    assert second.snapshot.epi == (Q(1, 4) - s / 4, s / 4)
    assert second.snapshot.capacity == (1 + s / 2, Q(3, 2) - s / 2)
    for observation in (first, second):
        assert observation.full_kernel_pressure == (0, 0)
        assert observation.kernel_pressure_defect == (0, 0)
        assert observation.stored_pressure_residual == (0, 0)
        assert observation.snapshot.rate == (0, 0)
        assert all(Q(3, 4) <= nu <= Q(7, 4) for nu in observation.snapshot.capacity)
    first_tetrad = tuple(
        observer(first_graph)
        for observer in (
            compute_structural_potential,
            compute_phase_gradient,
            compute_phase_curvature,
            estimate_coherence_length_with_provenance,
        )
    )
    second_tetrad = tuple(
        observer(second_graph)
        for observer in (
            compute_structural_potential,
            compute_phase_gradient,
            compute_phase_curvature,
            estimate_coherence_length_with_provenance,
        )
    )
    assert first_tetrad == second_tetrad
    assert first_tetrad[:3] == ({0: 0, 1: 0},) * 3
    assert first_tetrad[3].method == "spectral_gap"
    assert first_tetrad[3].value > 0
    curvature = Q(1, 16)
    difference = _p2_conditional_potential(
        _XI0 - s / 2, _ETA0 + s, curvature
    ) - _p2_conditional_potential(_XI0, _ETA0, curvature)
    assert difference == (curvature - _V * _V / _E) * s * s / 2 < 0
    # This curve is not a trajectory: x varies with s while the nodal rate
    # vanishes everywhere on it. The tetrad is a state diagnostic, not Psi.
