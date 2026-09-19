"""Source matching precedes phase tangency for an inherited prism model.

The candidate coordinate maps below are detached counterexamples, not new
phase laws. A derived observation cannot be fed back as extra pressure while
silently retaining the fine vector field from which it was derived.
"""

from fractions import Fraction as Q
from math import atan2, pi, sqrt

import pytest

from tests.physics._internal_mode_fixture import (
    INDUCED,
    LIFT,
    NODES,
    PROJECTION,
    _apply,
    _exact_generator,
    _graph,
)
from tnfr.physics.forced_support import (
    derive_forced_support_balance,
    observe_forced_support_shape,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing

_COEFFICIENTS = (Q(1, 16), Q(0), Q(1, 16), Q(0))


def _source(coefficients=_COEFFICIENTS):
    graph = _graph(coefficients)
    graph.graph["DNFR_WEIGHTS"] = {
        "epi": 0.5,
        "phase": 0.25,
        "vf": 0.25,
        "topo": 0.0,
    }
    return graph


def _inherited_rate(observation):
    # Retain the same EPI coefficient instead of renormalizing it after
    # removing other channels. Capacity is uniform one in these fixtures.
    assert observation.snapshot.capacity == (1,) * 6
    return tuple(
        observation.epi_weight * value
        for value in _apply(
            _exact_generator(observation.snapshot), observation.snapshot.epi
        )
    )


def _modeled_rate(observation):
    return tuple(
        inherited + source
        for inherited, source in zip(
            _inherited_rate(observation), observation.forcing, strict=True
        )
    )


def test_prepared_phase_source_fails_the_retained_fine_pushforward_identity():
    graph = _source()
    for a, i in NODES:
        graph.nodes[a, i]["theta"] = (0.0, pi / 3, pi / 6)[i]
    observation = capture_non_epi_forcing(graph)
    inherited = _apply(PROJECTION, _inherited_rate(observation))
    induced = tuple(
        observation.epi_weight * value for value in _apply(INDUCED, _COEFFICIENTS)
    )
    assert inherited == induced == (Q(-1, 32), 0, Q(-1, 32), 0)
    added = _apply(PROJECTION, observation.forcing)
    coefficient = Q(1501199875790165, 2**55)
    assert added == (coefficient, 0, coefficient, 0)
    assert observation.kernel_pressure_defect == (0,) * 6
    claimed = _apply(PROJECTION, observation.full_kernel_pressure)
    assert claimed == _apply(PROJECTION, _modeled_rate(observation))
    assert tuple(b - a for a, b in zip(inherited, claimed, strict=True)) == added
    assert claimed == (Q(375299968947541, 2**55), 0) * 2
    assert claimed[0] > 0 > inherited[0]
    # Ideal source projection is (1/24,0,1/24,0), giving u'=1/96.
    assert coefficient - Q(1, 24) == Q(-1, 3 * 2**55)
    # Choosing theta' afterward cannot remove this existing EPI-row defect.


def test_zero_internal_source_projection_does_not_match_the_full_fine_law():
    graph = _source()
    for a, i in NODES:
        graph.nodes[a, i]["theta"] = a * pi / 3
    observation = capture_non_epi_forcing(graph)
    assert any(observation.forcing)
    assert len(set(observation.forcing[:3])) == 1
    assert len(set(observation.forcing[3:])) == 1
    assert _apply(PROJECTION, observation.forcing) == (0,) * 4
    inherited, full = _inherited_rate(observation), _modeled_rate(observation)
    assert full != inherited
    assert _apply(PROJECTION, full) == _apply(PROJECTION, inherited)
    # The four retained coordinates cannot observe the changed fiber means.
    mean_defect = tuple(
        sum(observation.forcing[3 * a : 3 * a + 3], Q(0)) / 3 for a in range(2)
    )
    assert all(mean_defect)
    assert _apply(LIFT, _apply(PROJECTION, observation.forcing)) == (0,) * 6
    # This is only a projected row identity, not full embedding closure or
    # evidence that the prepared primitive phases remain fiberwise constant.


def test_regular_phase_from_form_chart_cannot_repair_a_changed_epi_row():
    symbolic = pytest.importorskip("sympy")
    u, kappa, e, w = symbolic.symbols("u kappa e w", positive=True)
    source = capture_non_epi_forcing(_source()).snapshot
    # Every prism neighbor set contains one node of each within-fiber index.
    assert all(
        sorted(source.nodes[j][1] for j in neighbors) == [0, 1, 2]
        for neighbors in source.support_neighbors
    )
    # Candidate chart: theta=beta-pi*kappa*y, y=(u,-u,0) in both fibers.
    # For |kappa*u|<1/4, all U3 separations are strictly below pi/2 and
    # exp(-i*beta)*S=1+2*cos(pi*kappa*u)>0. Thus g=kappa*y exactly.
    a = symbolic.pi * kappa * u
    relative_phasors = tuple(
        symbolic.cos(angle) + symbolic.I * symbolic.sin(angle)
        for angle in (-a, a, symbolic.Integer(0))
    )
    assert symbolic.simplify(sum(relative_phasors) - (1 + 2 * symbolic.cos(a))) == 0
    y = symbolic.Matrix([u, -u, 0] * 2)
    g = kappa * y
    inherited = -e * y
    feedback = inherited + w * g
    inherited_phase_rate_over_pi = -kappa * inherited
    feedback_image_rate_over_pi = -kappa * feedback
    assert symbolic.simplify(
        inherited_phase_rate_over_pi - feedback_image_rate_over_pi - kappa * w * g
    ) == symbolic.zeros(6, 1)
    # Differentiating the assigned phase along the original fine law gives
    # valid chart tangency, but it does not make the two EPI vector fields equal.
    source_rate = (w * g).diff(u) * (-e * u)
    assert source_rate == -e * w * g
    values = {
        u: symbolic.Rational(1, 16),
        kappa: symbolic.Rational(8, 3),
        e: symbolic.Rational(1, 2),
        w: symbolic.Rational(1, 4),
    }
    assert (kappa * u).subs(values) == symbolic.Rational(1, 6)
    assert inherited[0].subs(values) == symbolic.Rational(-1, 32)
    assert feedback[0].subs(values) == symbolic.Rational(1, 96)
    defect = inherited_phase_rate_over_pi - feedback_image_rate_over_pi
    assert (
        tuple(value.subs(values) for value in defect)
        == (symbolic.Rational(1, 9), symbolic.Rational(-1, 9), 0) * 2
    )
    # beta=pi/6 gives the already-captured (0,pi/3,pi/6) phase snapshot.
    # These are exact-real chart identities, not derivatives of float kernels.


def test_amplitude_normalization_adds_no_canonical_sustaining_source():
    observation = capture_non_epi_forcing(_source())
    assert observation.forcing == observation.kernel_pressure_defect == (0,) * 6
    reference = derive_forced_support_balance(
        observation.snapshot,
        epi_weight=observation.epi_weight,
        forcing=observation.forcing,
    )
    shape = observe_forced_support_shape(reference, observation.snapshot)
    assert shape.relaxation_rate == Q(1, 2)
    assert shape.norm_squared == Q(3, 64)
    assert shape.norm_squared_rate == Q(-3, 64)
    assert shape.shape_tangent_scaled == (0,) * 6
    assert shape.stationary_shape
    inherited = _inherited_rate(observation)
    correction = tuple(
        shape.relaxation_rate * value for value in shape.state.relative_error
    )
    assert any(correction)
    assert tuple(a + b for a, b in zip(inherited, correction, strict=True)) == (
        shape.shape_tangent_scaled
    )
    # q=y/r: r*q'=y'-(r'/r)*y. Recovering y' puts the radial term back.
    recovered = tuple(
        tangent - radial
        for tangent, radial in zip(shape.shape_tangent_scaled, correction, strict=True)
    )
    assert recovered == inherited == observation.full_kernel_pressure
    assert any(inherited)
    # Feeding +kappa*y into physical EPI would cancel its decay and define a
    # different law; the shared observation has neither supplied nor written it.


def test_homogeneous_linear_pair_angles_cannot_match_a_nonzero_scaled_source():
    observations = []
    phases = []
    for u in (Q(1, 16), Q(1, 8)):
        # Deliberately supplied linear observation pairs, not a selected law:
        # (u,0), (u,sqrt(3)*u), (sqrt(3)*u,u). Their ideal arguments are
        # 0, pi/3, pi/6 for every u>0, with no branch or zero-pair crossing.
        pairs = (
            (float(u), 0.0),
            (float(u), sqrt(3) * float(u)),
            (sqrt(3) * float(u), float(u)),
        )
        angles = tuple(atan2(imaginary, real) for real, imaginary in pairs)
        graph = _source((u, 0, u, 0))
        for a, i in NODES:
            graph.nodes[a, i]["theta"] = angles[i]
        phases.append(angles)
        observations.append(capture_non_epi_forcing(graph))
    first, second = observations
    assert phases[0] == phases[1]
    assert first.phase == second.phase
    assert first.normalized_weights == second.normalized_weights
    assert first.forcing == second.forcing
    assert any(first.forcing)
    assert any(_apply(PROJECTION, first.forcing))
    assert _inherited_rate(second) == tuple(
        2 * value for value in _inherited_rate(first)
    )
    full0, full1 = _modeled_rate(first), _modeled_rate(second)
    assert tuple(b - 2 * a for a, b in zip(full0, full1, strict=True)) == tuple(
        -value for value in first.forcing
    )
    # For any fixed original coefficient e0, retaining its fine law requires
    # F=(e0-e)*g_EPI. That right side doubles with this centered form, whereas
    # a homogeneous argument source is unchanged. Nonzero F cannot match both.
    assert second.forcing != tuple(2 * value for value in first.forcing)
    # This obstruction excludes neither zero source nor other fine models;
    # state-dependent coefficients would be extra premises to justify.
