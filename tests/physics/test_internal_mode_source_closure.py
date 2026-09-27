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
from tests.physics.test_joint_quotient_contract import _ordered_phasor_gradient
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.forced_support import (
    derive_forced_support_balance,
    observe_forced_support_shape,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.form_geometry import (
    derive_regional_affine_closure,
    observe_regional_form,
)

_COEFFICIENTS = (Q(1, 16), Q(0), Q(1, 16), Q(0))
_REGIONS = (NODES[:3], NODES[3:])


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
    phases = (0.0, pi / 3, pi / 6)
    for a, i in NODES:
        graph.nodes[a, i]["theta"] = phases[i]
    observation = capture_non_epi_forcing(graph)
    inherited = _apply(PROJECTION, _inherited_rate(observation))
    induced = tuple(
        observation.epi_weight * value for value in _apply(INDUCED, _COEFFICIENTS)
    )
    assert inherited == induced == (Q(-1, 32), 0, Q(-1, 32), 0)
    added = _apply(PROJECTION, observation.forcing)
    # Ideal symmetry does not certify correctly rounded three-neighbor means.
    # Build the represented source from primitive ordered phasors instead.
    phase_gradient = _ordered_phasor_gradient(
        tuple(graph.nodes[node]["theta"] for node in NODES),
        tuple(tuple(NODES.index(other) for other in graph[node]) for node in NODES),
    )
    forcing = tuple(value / 4 for value in phase_gradient)
    fine_inherited = (Q(-1, 32), Q(1, 32), Q(0)) * 2
    fresh_pressure = tuple(
        Q(float(drift) + float(force))
        for drift, force in zip(fine_inherited, forcing, strict=True)
    )
    kernel_defect = tuple(
        fresh - drift - force
        for fresh, drift, force in zip(
            fresh_pressure, fine_inherited, forcing, strict=True
        )
    )
    assert observation.phase_gradient == phase_gradient
    assert observation.forcing == forcing
    assert added == _apply(PROJECTION, forcing)
    assert any(added)
    assert observation.full_kernel_pressure == fresh_pressure
    assert observation.kernel_pressure_defect == kernel_defect
    projected_defect = _apply(PROJECTION, kernel_defect)
    claimed = _apply(PROJECTION, observation.full_kernel_pressure)
    modeled = _apply(PROJECTION, _modeled_rate(observation))
    assert modeled == tuple(a + b for a, b in zip(inherited, added, strict=True))
    assert claimed == tuple(
        a + b for a, b in zip(modeled, projected_defect, strict=True)
    )
    assert tuple(b - a for a, b in zip(inherited, modeled, strict=True)) == added
    assert claimed[0] > 0 > inherited[0]
    # Ideal source projection is (1/24,0,1/24,0), giving u'=1/96.
    assert claimed[0] - Q(1, 96) == added[0] - Q(1, 24) + projected_defect[0]
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


def test_held_capacity_source_moves_means_without_losing_sign_quotient_dynamics():
    reports = []
    for sign in (1, -1):
        graph = _graph(tuple(sign * value for value in _COEFFICIENTS))
        graph.graph["DNFR_WEIGHTS"] = {
            "epi": 0.5,
            "phase": 0.0,
            "vf": 0.5,
            "topo": 0.0,
        }
        for a, i in NODES:
            graph.nodes[a, i]["nu_f"] = (0.25, 1.0)[a]
        default_compute_delta_nfr(graph)
        captured = capture_non_epi_forcing(graph)
        report = observe_regional_form(graph, _REGIONS)
        # One cross-fiber neighbor among three supplies capacity pressure
        # +/- (1/2)*(3/4)/3. Capacity multiplies it only after assembly.
        assert captured.forcing == (Q(1, 8),) * 3 + (Q(-1, 8),) * 3
        source_rate = tuple(
            nu * force
            for nu, force in zip(report.capacity, captured.forcing, strict=True)
        )
        assert source_rate == (Q(1, 32),) * 3 + (Q(-1, 8),) * 3
        assert _apply(PROJECTION, source_rate) == (0,) * 4
        closure = derive_regional_affine_closure(
            NODES,
            _REGIONS,
            generator=tuple(
                tuple(captured.epi_weight * value for value in row)
                for row in _exact_generator(captured.snapshot)
            ),
            source=source_rate,
        )
        assert closure.all_state_closed
        assert closure.source == source_rate != captured.forcing
        assert closure.source_contrast_a == closure.source_contrast_b == (0, 0)
        assert closure.block_circulant_defect == ((0,) * 6,) * 6
        assert closure.mean_generator == (
            (Q(-1, 24), Q(1, 24)),
            (Q(1, 6), Q(-1, 6)),
        )
        assert closure.mean_source == (Q(1, 32), Q(-1, 8))
        assert closure.contrast_generator_real == (
            (Q(-1, 6), Q(1, 24)),
            (Q(1, 6), Q(-2, 3)),
        )
        assert closure.contrast_generator_imag_over_sqrt3 == ((0, 0),) * 2
        expected_rate = (
            Q(1, 32) - sign * Q(1, 128),
            Q(1, 32) + sign * Q(1, 128),
            Q(1, 32),
            Q(-1, 8) - sign * Q(1, 32),
            Q(-1, 8) + sign * Q(1, 32),
            Q(-1, 8),
        )
        assert report.nodal_rate == expected_rate
        assert captured.stored_pressure_residual == (0,) * 6
        assert captured.kernel_pressure_defect == (0,) * 6
        assert report.nodal_rate_rounding_defect == (0,) * 6
        assert tuple(row.mean for row in report.regions) == (Q(1, 2),) * 2
        assert tuple(row.mean_rate for row in report.regions) == (Q(1, 32), Q(-1, 8))
        assert report.gram_real == ((Q(1, 128),) * 2,) * 2
        assert report.gram_imag_numerator == ((0, 0),) * 2
        # Both regional contrasts initially agree. Their linear rates are
        # -nu_a*z_a/2, giving Qdot_ab=-(nu_a+nu_b)*Q_ab/2.
        assert report.gram_rate_real == (
            (Q(-1, 512), Q(-5, 1024)),
            (Q(-5, 1024), Q(-1, 128)),
        )
        assert report.gram_rate_imag_numerator == ((0, 0),) * 2
        reports.append(report)
    assert reports[0].epi != reports[1].epi
    assert reports[0].gram_real == reports[1].gram_real
    assert reports[0].gram_rate_real == reports[1].gram_rate_real
    # This exact dyadic execution control does not choose a phase/capacity law
    # or assert that arbitrary future native events preserve this quotient.


def test_held_contrast_source_separates_equal_gram_states_with_rate_defects():
    reports, modeled_gram_rates, sources = [], [], []
    nu = Q(0.1)
    for sign in (1, -1):
        graph = _source(tuple(sign * value for value in _COEFFICIENTS))
        for a, i in NODES:
            graph.nodes[a, i].update(nu_f=0.1, theta=(0.0, pi / 3, pi / 6)[i])
        phases = tuple(graph.nodes[node]["theta"] for node in NODES)
        rows = tuple(
            tuple(NODES.index(other) for other in graph[node]) for node in NODES
        )
        expected_phase = _ordered_phasor_gradient(phases, rows)
        expected_forcing = tuple(value / 4 for value in expected_phase)
        default_compute_delta_nfr(graph)
        captured = capture_non_epi_forcing(graph)
        report = observe_regional_form(graph, _REGIONS)
        assert captured.phase_gradient == expected_phase
        assert captured.forcing == expected_forcing
        f = expected_forcing[0]
        assert f > Q(1, 32)
        assert expected_forcing == (f, -f, 0) * 2
        source_rate = tuple(nu * value for value in expected_forcing)
        assert _apply(PROJECTION, source_rate) == (nu * f, 0, nu * f, 0)
        closure = derive_regional_affine_closure(
            NODES,
            _REGIONS,
            generator=tuple(
                tuple(captured.epi_weight * value for value in row)
                for row in _exact_generator(captured.snapshot)
            ),
            source=source_rate,
        )
        assert not closure.all_state_closed
        assert closure.block_circulant_defect == ((0,) * 6,) * 6
        assert closure.source == source_rate != captured.forcing
        assert closure.source_contrast_a == (2 * nu * f,) * 2
        assert closure.source_contrast_b == closure.mean_source == (0, 0)
        assert closure.contrast_generator_real is None
        assert closure.contrast_generator_imag_over_sqrt3 is None
        inherited_pressure = (sign * Q(-1, 32), sign * Q(1, 32), 0) * 2
        modeled_pressure = tuple(
            drift + force
            for drift, force in zip(inherited_pressure, expected_forcing, strict=True)
        )
        expected_pressure = tuple(
            Q(float(drift) + float(force))
            for drift, force in zip(inherited_pressure, expected_forcing, strict=True)
        )
        expected_rate = tuple(Q(0.1 * float(p)) for p in expected_pressure)
        pressure_defect = tuple(
            materialized - modeled
            for materialized, modeled in zip(
                expected_pressure, modeled_pressure, strict=True
            )
        )
        rate_defect = tuple(
            rate - nu * pressure
            for rate, pressure in zip(expected_rate, expected_pressure, strict=True)
        )
        assert (
            report.stored_pressure == captured.full_kernel_pressure == expected_pressure
        )
        assert report.nodal_rate == expected_rate
        assert captured.kernel_pressure_defect == pressure_defect
        assert captured.stored_pressure_residual == (0,) * 6
        assert report.nodal_rate_rounding_defect == rate_defect
        assert any(rate_defect)
        assert tuple(row.mean for row in report.regions) == (Q(1, 2),) * 2
        assert tuple(row.mean_rate for row in report.regions) == (0, 0)
        assert report.gram_real == ((Q(1, 128),) * 2,) * 2
        assert report.gram_imag_numerator == ((0, 0),) * 2
        # Qdot=-nu*Q + c*z^*+z*c^*. Reversing the hidden common
        # contrast orientation reverses only the additive-source cross term.
        modeled = -nu / 128 + sign * nu * f / 4
        defect = sign * (nu * pressure_defect[0] + rate_defect[0]) / 4
        assert report.gram_rate_real == ((modeled + defect,) * 2,) * 2
        assert report.gram_rate_imag_numerator == ((0, 0),) * 2
        reports.append(report)
        modeled_gram_rates.append(modeled)
        sources.append(captured.forcing)
    assert sources[0] == sources[1]
    assert modeled_gram_rates[0] - modeled_gram_rates[1] == nu * sources[0][0] / 2
    assert reports[0].gram_real == reports[1].gram_real
    assert reports[0].gram_rate_real[0][0] > 0 > reports[1].gram_rate_real[0][0]
    # This held law is well-defined on the full signed form state. The failed
    # all-state quotient is the attempt to discard its source-relative angle.
