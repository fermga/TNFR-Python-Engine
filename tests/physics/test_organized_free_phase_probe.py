"""Independent static controls for the freely reacting two-C9 phase probe.

These controls differentiate identities and bound primitive inputs. They do
not evolve a source, reconstruct a measured record or evaluate a trajectory.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from math import factorial

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import cos, pi_interval
from tnfr.physics._organized_free_phase_probe import _derive_organized_free_phase_probe


@pytest.fixture(scope="module", autouse=True)
def no_response_source_or_trajectory_evaluation():
    with forbid_organized_response_evaluation():
        yield


@pytest.fixture(scope="module")
def graph():
    """Assemble the complete eighteen-node support by individual edges."""
    a = sp.Symbol("a", positive=True)
    edges = tuple(
        (9 * child + slot, 9 * child + (slot + 1) % 9, sp.Integer(1))
        for child in range(2)
        for slot in range(9)
    ) + tuple((slot, 9 + slot, a) for slot in range(9))
    incidence = sp.zeros(27, 18)
    for row, (left, right, _) in enumerate(edges):
        incidence[row, left], incidence[row, right] = 1, -1
    weights = sp.diag(*(weight for _, _, weight in edges))
    laplacian = incidence.T * weights * incidence
    means = sp.zeros(2, 18)
    for child in range(2):
        for slot in range(9):
            means[child, 9 * child + slot] = sp.Rational(1, 9)
    fine = sp.eye(18) - 9 * means.T * means
    return a, incidence, weights, laplacian, means, fine


def test_complete_fine_rows_retain_both_mean_reactions_and_storage_loss(graph):
    a, incidence, weights, laplacian, means, fine = graph
    gamma = sp.Symbol("gamma", positive=True)
    form = sp.Matrix(sp.symbols("x0:18", real=True))
    # Every edge current is independent here: cancellation of internal
    # reactions does not use a carrier, small error or sinusoidal identity.
    currents = sp.Matrix(sp.symbols("j0:27", real=True))
    gradient = incidence.T * weights * currents
    dx = (-laplacian * form - gamma * gradient) / 4
    dtheta = gamma * laplacian * form / 4
    relative_mean = (means[0, :] - means[1, :]) * form
    mean_current = sum(currents[18:]) / 9
    assert sp.expand(((means[0, :] - means[1, :]) * dx)[0]) == sp.expand(
        -a * (relative_mean[0] + gamma * mean_current) / 2
    )
    assert sp.expand(((means[0, :] - means[1, :]) * dtheta)[0]) == sp.expand(
        gamma * a * relative_mean[0] / 2
    )
    assert sp.expand(sum(dx)) == sp.expand(sum(dtheta)) == 0
    assert fine * laplacian == laplacian * fine
    assert means * fine == sp.zeros(2, 18)
    assert fine * fine == fine
    storage_rate = (laplacian * form).dot(dx) + gradient.dot(dtheta)
    loss = (laplacian * form).dot(laplacian * form) / 4
    assert sp.expand(storage_rate + loss) == 0
    # A frozen receiver would erase one reaction and halve the relative row.
    assert sp.expand((means[0, :] * dtheta)[0] - gamma * a * relative_mean[0] / 4) == 0
    assert sp.expand((means[1, :] * dtheta)[0] + gamma * a * relative_mean[0] / 4) == 0


def test_exact_carrier_mean_scale_storage_and_attachment_work(graph):
    a, incidence, weights, laplacian, means, _ = graph
    gamma, v, j_alpha, j_delta = sp.symbols("gamma v j_alpha j_delta", real=True)
    form = sp.Matrix([v / (2 * gamma)] * 9 + [-v / (2 * gamma)] * 9)
    gradient = incidence.T * weights * sp.Matrix([j_alpha] * 18 + [j_delta] * 9)
    dx = (-laplacian * form - gamma * gradient) / 4
    dtheta = gamma * laplacian * form / 4
    mean_difference = means[0, :] - means[1, :]
    assert (
        sp.simplify(gamma * (mean_difference * dx)[0] * 2 / a)
        == -v - gamma**2 * j_delta
    )
    assert sp.simplify((mean_difference * dtheta)[0] * 2 / a) == v
    form_storage = (form.T * laplacian * form)[0] / 2
    assert sp.simplify(form_storage) == 9 * a * v**2 / (2 * gamma**2)
    # The finite work is due to the explicitly changed conductance, not to
    # treating the large form preparation as free on the joined support.
    assert form_storage.subs({a: gamma**2 / 1000, v: 12}).simplify() == sp.Rational(
        81, 125
    )
    energy, current, speed = sp.symbols("V J v", real=True)
    dv = -speed - gamma**2 * current
    assert sp.expand(speed * dv + gamma**2 * current * speed) == -(speed**2)
    assert (9 * a / gamma**2 * (v**2 / 2 + gamma**2 * energy)).expand() == (
        form_storage + 9 * a * energy
    ).expand()


def test_zero_child_means_remove_linear_current_and_attachment_errors(graph):
    _, _, _, _, means, fine = graph
    phase = fine * sp.Matrix([sp.Rational((7 * k) % 17 - 8, 1000) for k in range(18)])
    form = fine * sp.Matrix([sp.Rational((5 * k) % 13 - 6, 700) for k in range(18)])
    assert means * phase == means * form == sp.zeros(2, 1)
    gaps = phase[:9, :] - phase[9:, :]
    form_gaps = form[:9, :] - form[9:, :]
    assert sum(gaps) == sum(form_gaps) == 0
    h, j0, j1, j2, v = sp.symbols("h j0 j1 j2 v", real=True)
    mean_taylor = sum(j0 + j1 * h * gap + j2 * h**2 * gap**2 / 2 for gap in gaps) / 9
    assert sp.expand(mean_taylor - j0) == j2 * h**2 * gaps.dot(gaps) / 18
    assert gaps.dot(gaps) <= 2 * phase.dot(phase)
    actual_square_sum = sum((v + gap) ** 2 for gap in form_gaps)
    assert sp.expand(actual_square_sum) == 9 * v**2 + form_gaps.dot(form_gaps)
    assert form_gaps.dot(form_gaps) <= 2 * form.dot(form)
    # Mean errors cannot be silently assigned the quadratic fine allowance.
    shifted_gaps = gaps + sp.ones(9, 1) / 1000
    assert sum(shifted_gaps) == sp.Rational(9, 1000)
    assert sp.diff(sum(j0 + j1 * h * gap for gap in shifted_gaps) / 9, h) == j1 / 1000


def test_common_fine_lyapunov_identity_and_rational_coercivity():
    gamma = sp.Symbol("gamma", positive=True)
    q = sp.Matrix(sp.symbols("q0:2", real=True))
    r = sp.Matrix(sp.symbols("r0:2", real=True))
    # Noncommuting symmetric blocks ensure the identity does not assume a
    # simultaneous mode basis for the moving Hessian and fixed Laplacian.
    laplacian = sp.Matrix([[1, sp.Rational(1, 5)], [sp.Rational(1, 5), 2]])
    hessian = sp.Matrix([[1, sp.Rational(1, 7)], [sp.Rational(1, 7), 3]])
    assert laplacian * hessian != hessian * laplacian
    dq, dr = laplacian * r / 4, -laplacian * r / 4 - gamma**2 * hessian * q / 4
    energy = ((q + r).dot(q + r) + r.dot(r)) / 2
    derivative = sum(
        sp.diff(energy, q[i]) * dq[i] + sp.diff(energy, r[i]) * dr[i] for i in range(2)
    )
    identity = (
        -(
            r.dot(laplacian * r)
            + gamma**2 * q.dot(hessian * q)
            + 2 * gamma**2 * r.dot(hessian * q)
        )
        / 4
    )
    assert sp.expand(derivative - identity) == 0
    assert sp.expand(4 * energy - q.dot(q)) == (q + 2 * r).dot(q + 2 * r).expand()
    assert (
        sp.expand(q.dot(q) + 3 * r.dot(r) / 2 - energy)
        == ((q - r).dot(q - r) / 2).expand()
    )
    qnorm, rnorm = sp.symbols("qnorm rnorm", real=True)
    young = qnorm**2 / 8 + 200 * rnorm**2 - 10 * qnorm * rnorm
    assert sp.expand(young - (qnorm - 40 * rnorm) ** 2 / 8) == 0

    gamma_bound = 1 / (1023 * pi_interval())
    g2 = gamma_bound.hi**2
    assert Q(2, 5) - 200 * g2 > 3 * g2 / 16
    # Thus 4 F' <= -gamma^2*q^2/8 -(2/5-200 gamma^2)*r^2
    # <= -gamma^2*F/8 on the full admitted moving-Hessian domain.
    alpha = 2 * pi_interval() / 9
    assert 2 * (1 - cos(alpha)).lo > Q(2, 5)
    assert alpha.hi + Q(1, 50) < Q(18, 25)
    assert cos(alpha + Q(1, 50)).lo - Q(2, 243) > Q(7, 10)
    a = gamma_bound**2 / 1000
    assert Q(7, 10) * Q(2, 5) - 2 * a.hi * Q(245, 243) > Q(1, 4)
    assert Q(245, 243) * (4 + 2 * a.hi) < 5
    assert alpha.hi + Q(1, 50) < pi_interval().lo


def test_full_turn_access_uses_rational_exponential_bounds_and_finite_horizon():
    # Alternating Taylor sums bound exp(-1), independently of the source's
    # literal endpoint margins and without stepping any state.
    exp_lower = sum((Q((-1) ** k, factorial(k)) for k in range(6)), Q(0))
    exp_upper = sum((Q((-1) ** k, factorial(k)) for k in range(5)), Q(0))
    assert Q(1, 3) < exp_lower < exp_upper < Q(2, 5)
    gamma = 1 / (1023 * pi_interval())
    forcing = gamma.hi**2 * Q(1459, 1458)
    assert forcing < Q(1, 2)
    assert (12 + Q(1, 2)) * exp_lower - Q(1, 2) > Q(11, 3)
    assert 12 - 13 * exp_upper > Q(34, 5) > 2 * pi_interval().hi
    assert forcing < Q(11, 3)  # v_s<0 whenever the lower bound holds.
    report = _derive_organized_free_phase_probe(
        deformation=1, form_error_bound=0, phase_error_bound=0
    )
    assert report.scaled_form_lower_bound == (12 + Q(1, 2)) / 3 - Q(1, 2)
    assert report.scaled_form_upper_bound == 12
    assert report.endpoint_phase_lower_bound == 12 - (12 + 1) * Q(2, 5)
    contact = gamma**2 / 1000
    horizon = 2 / contact
    assert horizon.lo > 10**10
    assert (gamma**2 / 32 * horizon).lo > 62


def test_derivative_free_harmonic_identity_requires_endpoint_correction():
    v, delta, gamma, current = sp.symbols("v delta gamma current", real=True)
    n = sp.Symbol("n", integer=True, positive=True)
    boundary = sp.sin(n * delta) * v**2 / 2
    boundary_rate = sp.diff(boundary, delta) * v + sp.diff(boundary, v) * (
        -v - gamma**2 * current
    )
    integrand = n * sp.cos(n * delta) * v**3 / 2 - sp.sin(n * delta) * v**2
    assert sp.simplify(integrand - boundary_rate) == gamma**2 * current * v * sp.sin(
        n * delta
    )
    # The missing endpoint term is genuinely nonzero for a full turn whose
    # initial registration is not an integer half-period of the tested mode.
    start = boundary.subs({delta: sp.pi / (2 * n), v: 12})
    end = boundary.subs({delta: sp.pi / (2 * n) + 2 * sp.pi, v: 4})
    assert sp.simplify(end - start) == -64
    deformation = sp.Symbol("deformation", real=True)
    law = (
        sp.sin(delta)
        + deformation * (sp.sin(9 * delta) - sp.sin(18 * delta) / 2) / 2187
    )
    for order, expected in ((1, 1), (9, deformation / 2187), (18, -deformation / 4374)):
        assert (
            sp.integrate(law * sp.sin(order * delta), (delta, 0, 2 * sp.pi)) / sp.pi
            == expected
        )


@pytest.mark.parametrize("deformation", (Q(0), Q(1, 3), Q(1)))
def test_report_bounds_are_rebuilt_from_primitive_allowances(deformation):
    rho_x, rho_theta = Q(1, 10), Q(1, 1000)
    report = _derive_organized_free_phase_probe(
        deformation=deformation, form_error_bound=rho_x, phase_error_bound=rho_theta
    )
    gamma = 1 / (1023 * pi_interval())
    contact = gamma**2 / 1000
    f0 = ((rho_theta + gamma.hi * rho_x) ** 2 + (gamma.hi * rho_x) ** 2) / 2
    j0, j1, j2 = 1 + deformation / 1458, 1 + 2 * deformation / 243, 1 + deformation / 9
    assert report.current_derivative_bounds == (j0, j1, j2)
    assert report.fine_energy_upper_bound == f0 < Q(1, 40000)
    assert report.fine_energy_margin == Q(1, 40000) - f0
    assert report.phase_residual_norm_squared_upper_bound == 4 * f0 < Q(1, 10000)
    assert report.pointwise_current_defect_upper_bound == j2 * 4 * f0 / 9
    assert report.harmonic_defect_upper_bound == j2 * 8 * f0 / 9
    assert report.attachment_work_upper_bound == Q(81, 125) + contact.hi * (
        rho_x**2 + j1 * rho_theta**2
    )
    assert report.contact_conductance_bounds.lo <= gamma.lo**2 / 1000
    assert report.contact_conductance_bounds.hi >= gamma.hi**2 / 1000
    assert report.horizon_bounds.lo <= 2 / contact.hi <= report.horizon_bounds.hi
    assert (
        report.fine_decay_rate_bounds.lo
        <= gamma.lo**2 / 32
        <= report.fine_decay_rate_bounds.hi
    )
    assert report.initial_form_contrast_bounds.lo > 36000
    assert report.initial_form_contrast_bounds.lo <= 12 / gamma.hi
    assert report.initial_form_contrast_bounds.hi >= 12 / gamma.lo
    assert report.form_contrast_bounds.lo <= Q(11, 3) / gamma.hi
    assert report.form_contrast_bounds.hi >= 12 / gamma.lo
    assert report.conditional_on_preparation
    assert (
        "no_autonomous_preparation_contact_occurrence_or_physical_admission"
        in report.scope
    )
    with pytest.raises(FrozenInstanceError):
        report.deformation = Q(0)


def test_exact_fine_budget_boundary_rejects_but_zero_error_is_admitted():
    gamma = 1 / (1023 * pi_interval())
    boundary_form = Q(1, 200) / gamma.hi
    assert (gamma.hi * boundary_form) ** 2 == Q(1, 40000)
    with pytest.raises(ValueError, match="F0_upper"):
        _derive_organized_free_phase_probe(
            deformation=1, form_error_bound=boundary_form, phase_error_bound=0
        )
    inside = _derive_organized_free_phase_probe(
        deformation=1,
        form_error_bound=boundary_form * Q(999, 1000),
        phase_error_bound=0,
    )
    assert inside.fine_energy_margin > 0
    zero = _derive_organized_free_phase_probe(
        deformation=0, form_error_bound=0, phase_error_bound=0
    )
    assert zero.fine_energy_upper_bound == zero.harmonic_defect_upper_bound == 0
    assert zero.attachment_work_upper_bound == Q(81, 125)
    assert zero.current_coefficients == ((1, Q(1)),)
    with pytest.raises(TypeError):
        _derive_organized_free_phase_probe(
            deformation=zero, form_error_bound=0, phase_error_bound=0
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("deformation", True),
        ("deformation", Q(-1, 10)),
        ("deformation", Q(11, 10)),
        ("deformation", float("nan")),
        ("form_error_bound", True),
        ("form_error_bound", -1),
        ("form_error_bound", float("inf")),
        ("phase_error_bound", Q(-1, 100)),
        ("phase_error_bound", False),
        ("phase_error_bound", float("nan")),
        ("phase_error_bound", Q(1, 100)),
    ),
)
def test_invalid_primitive_law_and_error_bounds_reject_before_certificate(field, value):
    arguments = dict(deformation=Q(1, 3), form_error_bound=Q(0), phase_error_bound=Q(0))
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        _derive_organized_free_phase_probe(**arguments)
