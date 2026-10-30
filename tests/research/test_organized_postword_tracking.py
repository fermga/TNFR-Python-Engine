"""Finite ideal postword tracking with conserved means and actual capacities.

These static controls reconstruct the admitted ideal successors, prove the
weighted full-row estimates, and evaluate only rational proof allowances.
They execute no event, trajectory, guard search or retained certificate.
"""

from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from tests.research import test_organized_word_phase_memory as memory
from tnfr.mathematics._comparison_flow import decay_convolution_upper
from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.physics._organized_source_capture import (
    _derive_organized_source_capture_barrier,
)
from tnfr.physics._organized_source_port import (
    _organized_source_capacity_rows,
    _source_emission_geometry,
)
from tnfr.physics._sine_lyapunov import (
    _sine_common_quadratic_coefficients,
    _sine_common_quadratic_initial_upper,
)
from tnfr.physics.reversible_eigenmode_reference import (
    _MAX_RATIONAL_EXPONENT,
    _negative_exp_bounds,
)

preparation = memory.preparation
increments = memory.increments
no_event_flow_or_source_execution = memory.no_event_flow_or_source_execution


def _project(row):
    return tuple((row[i] + row[memory.SWAP[i]]) / 2 for i in range(45))


def _divergence(edges, currents):
    row = [0 for _ in range(45)]
    for (left, right, _), current in zip(edges, currents):
        row[left] += current
        row[right] -= current
    return tuple(row)


def _laplacian(edges, row):
    return _divergence(edges, tuple(row[a] - row[b] for a, b, _ in edges))


def _matrix(rows):
    return tuple(tuple(Q(value) for value in row) for row in rows)


@pytest.fixture(scope="module")
def postword(preparation, increments):
    incoming, _, projected, observed, _, factors = preparation
    geometry = observed.unit_capacity_reference.geometry
    kappa = factors["SHA_vf_factor"]
    capacity = (kappa, kappa) + memory.UNIT[2:]
    rho = tuple(1 / value for value in capacity)
    successors = tuple(
        tuple(I(value) + (change if i == 0 else I(0)) for i, value in enumerate(row))
        for row, change in (
            (incoming, increments[1][0][0]),
            (projected.phase_offsets, increments[1][1][0]),
        )
    )
    return SimpleNamespace(
        geometry=geometry,
        edges=geometry.weighted_edges,
        gamma=geometry.gamma_bounds,
        capacity=capacity,
        rho=rho,
        total=sum(rho),
        kappa=kappa,
        phases=successors,
        delta=increments[2],
    )


def test_actual_successors_retain_two_passive_acute_domains(postword, preparation):
    barrier = _derive_organized_source_capture_barrier()
    assert barrier.geometry == postword.geometry and barrier.barrier_certified
    assert barrier.boundary_energy_lower_bound > Q(1, 100)
    assert _source_emission_geometry(postword.geometry)[0]
    assert len(postword.edges) == 90 and all(
        weight == 1 for _, _, weight in postword.edges
    )
    assert Q(1, 2) < postword.kappa < 1
    assert all(
        postword.capacity[i] == postword.capacity[memory.SWAP[i]] for i in range(45)
    )
    for old, new in zip(
        (preparation[0], preparation[2].phase_offsets), postword.phases
    ):
        _, old_storage, _ = memory._fields(postword.geometry, tuple(I(x) for x in old))
        _, storage, angles = memory._fields(postword.geometry, new)
        assert (storage - old_storage).hi < 0
        assert (storage - barrier.target_storage_bounds).hi < Q(1, 100)
        assert all(angle.abs_max < barrier.cell_edge_gap_bounds.lo for angle in angles)
    # Passive complete flow retains both admitted sublevels. Their full
    # projection chord therefore shares the collar, including after the jump.
    collar = postword.geometry.alpha_bounds.hi + Q(1, 5)
    assert collar < pi_interval().lo / 3
    # Cosine is even and decreasing on [0, pi]; the point endpoint gives
    # the collar minimum. The generic wide-interval Lipschitz enclosure
    # is deliberately coarser and need not certify this same lower bound.
    assert cos(I(collar)).lo > Q(1, 2)
    assert 1 - barrier.cycle_storage_minimum_bounds.hi / 9 > Q(3, 4)


def test_weighted_centering_retains_actual_jump_error_and_orbit_masses(postword):
    s = pytest.importorskip("sympy")
    rho = s.Matrix(tuple(s.Rational(x) for x in postword.rho))
    total = s.Rational(postword.total)
    weight = s.diag(*rho)
    one = s.ones(45, 1)
    center = s.eye(45) - one * rho.T / total
    exchange = s.zeros(45)
    for i, j in enumerate(memory.SWAP):
        exchange[i, j] = 1
    project = (s.eye(45) + exchange) / 2
    assert center**2 == center
    assert rho.T * center == s.zeros(1, 45)
    assert center.T * weight == weight * center
    assert project * center == center * project
    assert project * weight == weight * project
    e0 = s.eye(45)[:, 0]
    relative = center * e0
    coefficient = (relative.T * weight * relative)[0]
    kappa = s.Rational(postword.kappa)
    assert s.factor(coefficient - (43 * kappa + 1) / (kappa * (43 * kappa + 2))) == 0
    assert coefficient > 0
    assert sum(relative) != 0  # This is not the ordinary mean-zero leaf.
    mean = postword.delta / (43 * postword.kappa + 2)
    assert mean.hi < -Q(1, 2**61)
    assert postword.delta.abs_max < memory.HIDDEN**2 / 4
    assert postword.delta.hi < 0
    # Form is zero in both successors; the nonzero phase jump must remain
    # in the homogeneous initial-error term of the comparison bound.
    assert coefficient * postword.delta.hi**2 > 0
    values = tuple(Q((7 * i) % 13 - 6, 4096) for i in range(45))
    lifted = _project(values)
    fine_norm = sum(postword.rho[i] * lifted[i] ** 2 for i in range(45))
    representatives = (*range(18), *range(27, 45))
    orbit_norm = sum(
        (2 if 9 <= i < 18 else 1) * postword.rho[i] * lifted[i] ** 2
        for i in representatives
    )
    assert fine_norm == orbit_norm
    assert fine_norm != sum(postword.rho[i] * lifted[i] ** 2 for i in representatives)


def test_full_support_weighted_poincare_controls_all_forty_five_coordinates(postword):
    basis = tuple(tuple(Q(i == j) for i in range(45)) for j in range(45))
    laplacian = tuple(zip(*(_laplacian(postword.edges, row) for row in basis)))
    centered_weight = tuple(
        tuple(
            (postword.rho[i] if i == j else 0)
            - postword.rho[i] * postword.rho[j] / postword.total
            for j in range(45)
        )
        for i in range(45)
    )
    gap = 4 * postword.kappa / 15
    difference = tuple(
        tuple(laplacian[i][j] - gap * centered_weight[i][j] for j in range(45))
        for i in range(45)
    )
    assert exact_symmetric_semidefinite(difference)
    assert all(sum(row) == 0 for row in laplacian)
    assert all(sum(row) == 0 for row in centered_weight)
    # D and L do not commute; treating the mobility as a scalar would erase
    # the actual Silence targets. The inequality covers twin-breaking rows.
    assert any(
        (postword.capacity[i] - postword.capacity[j]) * laplacian[i][j]
        for i in range(45)
        for j in range(45)
    )


def test_full_weighted_error_rows_and_forced_metric_identity_are_exact(postword):
    s = pytest.importorskip("sympy")
    p, q = s.symbols("p0:45"), s.symbols("q0:45")
    currents = s.symbols("j0:90")
    secants = s.symbols("c0:90")
    gamma = s.Symbol("gamma", positive=True)
    lp = _laplacian(postword.edges, p)
    kq = _divergence(
        postword.edges,
        tuple(c * (q[a] - q[b]) for (a, b, _), c in zip(postword.edges, secants)),
    )
    bj = _divergence(postword.edges, currents)
    dp = tuple(
        s.Rational(nu) * (-x - gamma * y + gamma * j) / 4
        for nu, x, y, j in zip(postword.capacity, lp, kq, bj)
    )
    dq = tuple(gamma * s.Rational(nu) * x / 4 for nu, x in zip(postword.capacity, lp))
    for row in (dp, dq):
        assert s.expand(sum(s.Rational(w) * x for w, x in zip(postword.rho, row))) == 0
    derivative = 2 * sum(
        s.Rational(w) * ((x + gamma * y) * dx + (y + gamma * x) * dy)
        for w, x, y, dx, dy in zip(postword.rho, p, q, dp, dq)
    )
    edge_identity = 0
    for (a, b, _), c, j in zip(postword.edges, secants, currents):
        x, y = p[a] - p[b], q[a] - q[b]
        edge_identity += (
            -(1 - gamma**2) * x**2 / 2
            - gamma * c * x * y / 2
            - gamma**2 * c * y**2 / 2
            + gamma * (x + gamma * y) * j / 2
        )
    assert s.expand(derivative - edge_identity) == 0


def test_edge_metric_and_forcing_bounds_cover_continuous_secants():
    # In coordinates (edge p, gamma*edge q), gamma<=1/2 gives the
    # following smaller homogeneous dissipation matrix. Its affine c
    # dependence makes endpoint PSD a proof for every c in [1/2,1].
    for c in (Q(1, 2), Q(1)):
        remainder = _matrix(((Q(1, 4), c / 4), (c / 4, c / 2 - Q(1, 8))))
        assert exact_symmetric_semidefinite(remainder)
    s = pytest.importorskip("sympy")
    x, y, j, gamma = s.symbols("x y j gamma", real=True)
    assert s.expand(x**2 / 16 + gamma**2 * j**2 - gamma * x * j / 2) == s.expand(
        (x / 4 - gamma * j) ** 2
    )
    assert s.expand(gamma**2 * (y**2 / 16 + j**2 - y * j / 2)) == s.expand(
        gamma**2 * (y / 4 - j) ** 2
    )
    # Thus W' <= -pLp/16-gamma^2*qLq/16+2*gamma^2*||j||^2,
    # without differentiating a moving secant or assuming common eigenvectors.


def test_actual_ninety_edge_forcing_keeps_both_postword_rows(postword):
    fine, core = postword.phases
    projected = _project(fine)
    hidden = tuple((fine[i] - fine[memory.SWAP[i]]) / 2 for i in range(45))
    alpha = postword.geometry.alpha_bounds
    edge_forcing = []
    cycle_count = matching_count = 0
    for a, b, _ in postword.edges:
        angle = alpha * ((b % 9) - (a % 9)) + projected[b] - projected[a]
        odd_gap = hidden[b] - hidden[a]
        # The stable identity avoids subtracting cosine from one near zero.
        edge_forcing.append(-2 * sin(angle) * sin(odd_gap / 2) ** 2)
        cycle_count += a // 9 == b // 9 and a // 9 in (1, 2)
        matching_count += a // 9 != b // 9 and {a // 9, b // 9} != {0, 4}
    assert (cycle_count, matching_count) == (18, 36)
    force = _divergence(postword.edges, edge_forcing)
    fine_current, _, _ = memory._fields(postword.geometry, fine)
    projected_current, _, _ = memory._fields(postword.geometry, projected)
    core_current, _, _ = memory._fields(postword.geometry, core)
    for actual, base, defect in zip(_project(fine_current), projected_current, force):
        assert (actual - base - defect).contains(0)
    squared = sum(value.abs_max**2 for value in edge_forcing)
    assert 0 < squared < 5 * memory.HIDDEN**4
    actual_rows = []
    for current in (fine_current, core_current):
        actual_rows.append(
            _organized_source_capacity_rows(
                reference=SimpleNamespace(
                    geometry=postword.geometry,
                    form_gradient=memory.ZERO,
                    phase_current_bounds=current,
                ),
                mobilities=postword.capacity,
                phase_mobility="shared_capacity",
            )
        )
    for i, (left, right) in enumerate(
        zip(
            _project(actual_rows[0]["form_tau_rate_bounds"]),
            actual_rows[1]["form_tau_rate_bounds"],
        )
    ):
        predicted = (
            postword.capacity[i]
            * postword.gamma
            * (projected_current[i] - core_current[i] + force[i])
            / 4
        )
        assert (left - right - predicted).contains(0)
    assert all(row["phase_tau_rate_bounds"] == (I(0),) * 45 for row in actual_rows)
    assert any(
        not value.contains(0) for value in actual_rows[0]["form_tau_rate_bounds"]
    )


def test_full_hidden_forcing_multiplicity_and_relaxation_are_re_admitted(postword):
    # Exact C9 inequalities apply to every hidden vector, not only the
    # single nonzero coordinate in the prospective witness.
    cycle_laplacian = [[Q(0) for _ in range(9)] for _ in range(9)]
    for i in range(9):
        j = (i + 1) % 9
        cycle_laplacian[i][i] += 1
        cycle_laplacian[j][j] += 1
        cycle_laplacian[i][j] -= 1
        cycle_laplacian[j][i] -= 1
    assert exact_symmetric_semidefinite(
        tuple(
            tuple(4 * Q(i == j) - cycle_laplacian[i][j] for j in range(9))
            for i in range(9)
        )
    )
    # max(edge difference)^2<=2||b||^2 and sum differences^2<=4||b||^2
    # give half the fourth-power cycle sum <=4||b||^4; the matching
    # contribution is sum b_j^4<=||b||^4. Both copies are retained.
    assert Q(1, 2) * 2 * 4 + 1 == 5
    assert postword.capacity[9:27] == (Q(1),) * 18
    lower, upper, hidden_rate = _sine_common_quadratic_coefficients(
        gamma_lower=postword.gamma.lo,
        gamma_upper=postword.gamma.hi,
        cosine_lower=Q(1, 2),
        gap_lower=Q(2),
        capacity_lower=Q(1),
        capacity_upper=Q(1),
    )
    initial = _sine_common_quadratic_initial_upper(
        gamma_lower=postword.gamma.lo,
        gamma_upper=postword.gamma.hi,
        weighted_square=memory.HIDDEN**2,
        weighted_cross=Q(0),
    )
    assert initial == memory.HIDDEN**2
    assert lower == 1 - postword.gamma.hi and upper == 1 + postword.gamma.hi
    assert hidden_rate == postword.gamma.lo**2 / (4 * (1 + postword.gamma.hi))


@pytest.fixture(scope="module")
def allowances(postword):
    g = postword.gamma
    lower, upper, homogeneous_rate = _sine_common_quadratic_coefficients(
        gamma_lower=g.lo,
        gamma_upper=g.hi,
        cosine_lower=Q(1, 2),
        gap_lower=4 * postword.kappa / 15,
        capacity_lower=Q(1),
        capacity_upper=Q(1),
    )
    # Apply the kernel to y=D^(-1/2)*p and z=D^(-1/2)*q, with unit
    # mobility and Htilde=D^(1/2)*L*D^(1/2) on (D^(-1/2)*1)^perp.
    # The proved weighted Poincare inequality gives its gap 4*kappa/15;
    # raw L need not have gap 4/15 on the inverse-capacity mean-zero leaf.
    # Half of the homogeneous rate pays for the two forcing Young terms.
    rate = homogeneous_rate / 2
    hidden_rate = g.lo**2 / (4 * (1 + g.hi))
    forcing_rate = 2 * hidden_rate
    duration = 40 / rate
    hidden_coefficient = memory.HIDDEN**2 / (1 - g.hi)
    forcing = 10 * g.hi**2 * hidden_coefficient**2
    initial = postword.delta.abs_max**2 * (
        1 / postword.kappa - 1 / (postword.kappa**2 * postword.total)
    )
    # Keep small initial data and returned scalar bounds outside the fixed
    # 128-bit interval grid: positivity and quartic source scaling survive.
    decay = _negative_exp_bounds(rate * duration)
    hidden_decay = _negative_exp_bounds(hidden_rate * duration)
    forcing_decay = _negative_exp_bounds(forcing_rate * duration)
    convolution = decay_convolution_upper(
        rate, forcing_rate, duration, decay, forcing_decay
    )
    metric = initial * decay[1] + forcing * convolution
    return SimpleNamespace(
        lower=lower,
        upper=upper,
        rate=rate,
        hidden_rate=hidden_rate,
        forcing_rate=forcing_rate,
        duration=duration,
        hidden_coefficient=hidden_coefficient,
        forcing=forcing,
        initial=initial,
        decay=decay,
        hidden_decay=hidden_decay,
        convolution=convolution,
        metric=metric,
        projected=metric / lower,
        hidden=2 * hidden_coefficient * hidden_decay[1],
    )


def test_frozen_finite_time_allowance_includes_initial_and_forced_errors(
    postword, allowances
):
    a, g = allowances, postword.gamma
    assert a.rate == postword.kappa * g.lo**2 / (60 * (1 + g.hi))
    assert a.hidden_rate == 15 * a.rate / postword.kappa
    assert a.forcing_rate == 30 * a.rate / postword.kappa
    assert a.rate * a.duration == 40
    assert a.forcing_rate * a.duration < _MAX_RATIONAL_EXPONENT
    assert a.lower == 1 - g.hi
    assert a.upper == 1 + g.hi
    assert 0 < a.initial < memory.HIDDEN**4 / 8
    assert 0 < a.convolution
    assert a.metric > a.forcing * a.convolution
    assert a.projected < Q(1, 2**124)
    assert a.hidden < Q(1, 2**124)
    assert a.projected + a.hidden < Q(1, 2**123)
    assert (-postword.delta / (43 * postword.kappa + 2)).lo > Q(1, 2**61)
    # At time zero the convolution vanishes but the actual jump error does
    # not. Dropping the homogeneous term would claim an impossible zero.
    assert (
        decay_convolution_upper(a.rate, a.forcing_rate, Q(0), (Q(1),) * 2, (Q(1),) * 2)
        == 0
    )
    assert _negative_exp_bounds(Q(0)) == (Q(1), Q(1))
    assert a.initial / a.lower > 0


def test_independent_coarse_budget_proves_the_frozen_thresholds(postword, allowances):
    a, g, b = allowances, postword.gamma, memory.HIDDEN
    assert g.hi < Q(1, 2) and g.hi / g.lo < 2
    assert postword.kappa > Q(1, 2)
    assert a.hidden_coefficient < 2 * b**2
    assert a.forcing / (a.forcing_rate - a.rate) < 14400 * b**4 / 29
    uniform = (a.initial + a.forcing / (a.forcing_rate - a.rate)) / a.lower
    assert uniform < 1024 * b**4 == Q(1, 2**70)
    # The independent positive Taylor polynomial proves exp(5)>128,
    # hence exp(-40)<2^-56 without fitting the computed allowance.
    factorial, polynomial = 1, Q(1)
    for order in range(1, 9):
        factorial *= order
        polynomial += Q(5**order, factorial)
    assert polynomial > 128
    assert a.decay[1] < Q(1, 2**56)
    assert a.projected <= uniform * a.decay[1] < Q(1, 2**126)
    assert a.hidden_rate * a.duration > 600
    assert a.hidden_decay[1] < Q(1, 2**840)
    assert a.hidden < Q(1, 2**878)


def test_weighted_reconstruction_separates_hidden_relative_and_mean_coordinates(
    postword,
):
    s = pytest.importorskip("sympy")
    rho = s.Matrix(tuple(s.Rational(value) for value in postword.rho))
    weight, one = s.diag(*rho), s.ones(45, 1)
    exchange = s.zeros(45)
    for i, j in enumerate(memory.SWAP):
        exchange[i, j] = 1
    project = (s.eye(45) + exchange) / 2
    mean_projector = one * rho.T / s.Rational(postword.total)
    relative = project - mean_projector
    hidden = s.eye(45) - project
    assert hidden.T * weight * relative == s.zeros(45)
    assert hidden.T * weight * mean_projector == s.zeros(45)
    assert relative.T * weight * mean_projector == s.zeros(45)
    assert (
        hidden.T * weight * hidden
        + relative.T * weight * relative
        + mean_projector.T * weight * mean_projector
        == weight
    )
    embedding = s.zeros(45, 9)
    for j in range(9):
        embedding[9 + j, j], embedding[18 + j, j] = 1, -1
    assert embedding.T * weight * embedding == 2 * s.eye(9)
    assert all(value >= 1 for value in postword.rho)
    # The same metric division bounds the weighted relative norm. Thus
    # subtracting the conserved means gives corrected full squared error
    # <= projected allowance + 2N. Raw error retains R*(mx^2+mtheta^2).


def test_constant_mean_error_cannot_be_assigned_a_decaying_relative_bound(postword):
    constant = (Q(1, 7),) * 45
    assert _laplacian(postword.edges, constant) == (Q(0),) * 45
    mean = sum(w * value for w, value in zip(postword.rho, constant)) / postword.total
    assert mean == Q(1, 7)
    assert tuple(value - mean for value in constant) == (Q(0),) * 45
    raw_weighted_square = sum(w * value**2 for w, value in zip(postword.rho, constant))
    assert raw_weighted_square == postword.total * mean**2 > 0
    # Constant phase shifts leave all edge currents and both full rows
    # unchanged; their raw error persists even with zero hidden forcing.
