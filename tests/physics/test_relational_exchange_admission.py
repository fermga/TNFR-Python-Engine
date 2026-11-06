"""Conditional local exchange admission, not a selected runtime phase law.

The native pressure is retained. Dirichlet form storage, phase storage scale
and nodewise power cancellation are additional premises. Exact algebra and
production pressure/response observations are kept distinct; no trajectory,
physical interpretation or parameter fit is supplied.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.physics._internal_mode_fixture import _metric_differential
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    evaluate_relational_uniform_tangent,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_response import (
    derive_joint_nodal_response,
    derive_phase_response,
    observe_phase_source_geometry,
)
from tnfr.physics.support_transport import observe_support_transport


@pytest.fixture(scope="module")
def path_reference():
    s = pytest.importorskip("sympy")
    graph = nx.path_graph(3)
    rows = tuple(tuple(graph.neighbors(i)) for i in graph)
    angle = s.atan(s.Rational(4, 3))
    phases = (-angle, s.S.Zero, angle)
    metric, _, source, jacobian = _metric_differential(s, rows, phases)
    degree = s.diag(1, 2, 1)
    laplacian = s.Matrix(((1, -1, 0), (-1, 2, -1), (0, -1, 1)))
    form = s.Matrix((s.Rational(3, 4), s.Rational(1, 4), s.Rational(1, 2)))
    capacity = s.diag(1, 2, s.Rational(1, 2))
    graph.graph["DNFR_WEIGHTS"] = {"epi": 0.5, "phase": 0.5, "vf": 0, "topo": 0}
    for i in graph:
        graph.nodes[i].update(
            EPI=float(form[i]),
            nu_f=float(capacity[i, i]),
            theta=float(phases[i]),
            delta_nfr=0.0,
        )
    capture = capture_non_epi_forcing(graph)
    for node, pressure in zip(graph, capture.full_kernel_pressure, strict=True):
        graph.nodes[node]["delta_nfr"] = float(pressure)
    return s, graph, phases, metric, source, jacobian, degree, laplacian, form, capacity


def _nonlinear_phase_correction(s, q, degree, source, capacity, e, beta, eta):
    """A supplied comparison row, not an installed execution option."""
    normalized_form = degree.inv() * q / s.sqrt(beta)
    response = s.Matrix(
        [
            value**3 * pressure**2 / (1 + value**2)
            for value, pressure in zip(normalized_form, source, strict=True)
        ]
    )
    return (eta * e / s.pi) * capacity * response


def test_native_p2_and_cotangent_noether_momentum_exclude_relative_exchange():
    s = pytest.importorskip("sympy")
    mean, contrast, delta, omega = s.symbols("mean contrast delta omega", real=True)
    e, w = s.symbols("e w", positive=True)
    h = s.pi * s.sin(delta) / delta
    form = s.Matrix((mean + contrast, mean - contrast))
    pressure = -e * s.Matrix(((1, -1), (-1, 1))) * form
    pressure += w * s.Matrix((delta, -delta)) / s.pi
    mean_rate = sum(pressure) / 2
    momentum = 2 * h * mean
    momentum_rate = s.diff(momentum, mean) * mean_rate + s.diff(momentum, delta) * omega
    assert mean_rate == 0
    assert s.simplify(momentum_rate - 2 * mean * s.diff(h, delta) * omega) == 0
    # On 0<delta<pi this numerator increases from zero, hence h'<0.
    numerator = s.sin(delta) - delta * s.cos(delta)
    assert s.diff(numerator, delta) == delta * s.sin(delta)
    assert s.limit(numerator, delta, 0) == 0
    assert s.simplify(s.diff(h, delta) + s.pi * numerator / delta**2) == 0
    assert s.simplify(momentum_rate.subs(delta, s.pi / 2)) == -8 * mean * omega / s.pi
    # A continuous full-state field must therefore have omega=0 on the
    # dense set mean!=0, delta!=0, and by continuity on its regular closure.
    # Restricting only to mean=0 would remove an essential premise.


def test_arbitrary_rotation_invariant_hamiltonian_reduces_to_held_relative_phase():
    s = pytest.importorskip("sympy")
    delta, total, relative, w = s.symbols("delta P J w", real=True)
    h = s.pi * s.sin(delta) / delta
    # pi0*dtheta0+pi1*dtheta1 = P*dq+J*ddelta, before selecting storage.
    p0, p1, dq, dd = s.symbols("p0 p1 dq dd", real=True)
    one_form = p0 * (dq - dd / 2) + p1 * (dq + dd / 2)
    assert s.expand(one_form - (p0 + p1) * dq - (p1 - p0) * dd / 2) == 0
    arbitrary = s.Function("E")(delta, total, relative)
    mean = total / (2 * h)
    canonical_mean_rate = s.diff(mean, delta) * s.diff(arbitrary, relative)
    assert (
        s.simplify(
            canonical_mean_rate
            + total * s.diff(h, delta) * s.diff(arbitrary, relative) / (2 * h**2)
        )
        == 0
    )
    # Native mean conservation forces E_J=0 on the full open domain.
    # The remaining native source then fixes E_delta=w*sin(delta).
    energy = w * (1 - s.cos(delta)) + s.Function("F")(total)
    coordinates = (delta, total, relative)
    canonical_rate = s.Matrix((s.diff(energy, relative), 0, -s.diff(energy, delta)))
    form = s.Matrix(((total / 2 - relative) / h, (total / 2 + relative) / h))
    induced = form.jacobian(coordinates) * canonical_rate
    assert s.simplify(induced - w * s.Matrix((delta, -delta)) / s.pi) == s.zeros(2, 1)
    assert canonical_rate[0] == s.diff(energy, relative, 2) == 0


def test_nodewise_exchange_cancels_source_power_with_heterogeneous_capacity(
    path_reference,
):
    s, _, _, metric, source, _, degree, laplacian, x, _ = path_reference
    e, w, beta = s.symbols("e w beta", positive=True)
    capacities = s.symbols("nu0:3", nonnegative=True)
    capacity = s.diag(*capacities)
    q = laplacian * x
    phase_rate = (w / beta) * metric.inv() * capacity * q
    form_rate = capacity * (-e * degree.inv() * q + w * source)
    phase_covector = -metric * source
    local_exchange = s.Matrix(
        [
            w * q[i] * capacities[i] * source[i]
            + beta * phase_covector[i] * phase_rate[i]
            for i in range(3)
        ]
    )
    assert local_exchange.applyfunc(s.simplify) == s.zeros(3, 1)
    loss = e * (q.T * capacity * degree.inv() * q)[0]
    storage_rate = q.dot(form_rate) + beta * phase_covector.dot(phase_rate)
    assert s.simplify(storage_rate + loss) == 0
    assert loss == e * (capacities[0] / 4 + 9 * capacities[1] / 32 + capacities[2] / 16)
    assert loss.is_nonnegative
    common_rate = s.Symbol("common_rate", nonzero=True)
    local_rotation_work = beta * common_rate * phase_covector
    assert local_rotation_work != s.zeros(3, 1)
    assert s.simplify(sum(local_rotation_work)) == 0
    # The chosen nodewise law also pauses the phase row at zero capacity;
    # this is a declared completion, not implied by the nodal product alone.
    assert (
        phase_rate[1].subs(capacities[1], 0) == form_rate[1].subs(capacities[1], 0) == 0
    )


def test_form_origin_and_unit_conversion_preserve_the_complete_candidate(
    path_reference,
):
    s, _, _, metric, source, _, degree, laplacian, x, capacity = path_reference
    e, w, beta, amplitude, clock = s.symbols("e w beta A T", positive=True)
    shift = s.Symbol("shift", real=True)

    def field(form, diffusion, weight, scale, mobility=capacity):
        q = laplacian * form
        return (
            mobility * (-diffusion * degree.inv() * q + weight * source),
            (weight / scale) * metric.inv() * mobility * q,
        )

    direct = field(x, e, w, beta)
    shifted = field(x + shift * s.ones(3, 1), e, w, beta)
    converted = field(
        amplitude * x, e, amplitude * w, amplitude**2 * beta, capacity / clock
    )
    assert all(
        (a - b).applyfunc(s.simplify) == s.zeros(3, 1)
        for a, b in zip(direct, shifted, strict=True)
    )
    assert (converted[0] - amplitude * direct[0] / clock).applyfunc(
        s.simplify
    ) == s.zeros(3, 1)
    assert (converted[1] - direct[1] / clock).applyfunc(s.simplify) == s.zeros(3, 1)
    # Reentering the engine's normalized two-channel representation also
    # requires compensating capacity; raw DNFR_WEIGHTS normalize internally.
    normalization = e + amplitude * w
    normalized = field(
        amplitude * x,
        e / normalization,
        amplitude * w / normalization,
        amplitude**2 * beta,
        normalization * capacity / clock,
    )
    assert all(
        (a - b).applyfunc(s.simplify) == s.zeros(3, 1)
        for a, b in zip(normalized, converted, strict=True)
    )


def test_native_pressure_and_shared_joint_response_keep_full_mean_and_source_chain(
    path_reference,
):
    s, graph, phases, metric, source, jacobian, degree, laplacian, x, capacity = (
        path_reference
    )
    e = w = s.Rational(1, 2)
    beta = s.S.One
    pressure = -e * degree.inv() * laplacian * x + w * source
    phase_rate = (w / beta) * metric.inv() * capacity * laplacian * x
    form_rate = capacity * pressure
    capture = capture_non_epi_forcing(graph)
    assert tuple(map(float, capture.full_kernel_pressure)) == pytest.approx(
        tuple(map(float, pressure)), abs=2e-15
    )
    assert float(sum(form_rate)) != 0.0
    snapshot = observe_support_transport(graph)
    gram = tuple(
        tuple(Q(s.expand_trig(s.cos(a - b)).simplify()) for b in phases) for a in phases
    )
    reference = derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=snapshot.support_neighbors,
        receiver_sources=tuple((i,) for i in range(3)),
        phase_factor=0,
    )
    report = derive_joint_nodal_response(
        snapshot,
        reference,
        epi_weight=Q(1, 2),
        phase_weight=Q(1, 2),
        capacity_weight=0,
        phase_rate_over_pi=tuple(float(rate / s.pi) for rate in phase_rate),
        capacity_rate=(0, 0, 0),
    )
    expected = -e * degree.inv() * laplacian * form_rate + w * jacobian * phase_rate
    assert tuple(map(float, report.pressure_rate)) == pytest.approx(
        tuple(map(float, expected)), abs=2e-15
    )
    assert tuple(map(float, report.epi_acceleration)) == pytest.approx(
        tuple(map(float, capacity * expected)), abs=2e-15
    )
    # An independent directional pressure difference changes both form and
    # primitive phase. It is a local response check, not an executed trajectory.
    step = 1e-6
    endpoints = []
    for sign in (-1, 1):
        perturbed = deepcopy(graph)
        for node in perturbed:
            perturbed.nodes[node]["EPI"] += sign * step * float(form_rate[node])
            perturbed.nodes[node]["theta"] += sign * step * float(phase_rate[node])
        endpoints.append(capture_non_epi_forcing(perturbed).full_kernel_pressure)
    finite_difference = tuple(
        float(b - a) / (2 * step) for a, b in zip(*endpoints, strict=True)
    )
    assert finite_difference == pytest.approx(tuple(map(float, expected)), abs=2e-10)
    # Neither graph pressure nor the derivative has gained a cotangent Kx.
    assert report.capacity_acceleration == (0, 0, 0)


def test_consensus_modes_distinguish_dirichlet_exchange_from_cotangent_exchange():
    s = pytest.importorskip("sympy")
    graph = nx.cycle_graph(6)
    for node in graph:
        graph.nodes[node].update(EPI=0.0, nu_f=1.0, delta_nfr=0.0)
    snapshot = observe_support_transport(graph)
    reference = derive_phase_response(
        cosine_gram=((1,) * 6,) * 6,
        mean_neighbors=snapshot.support_neighbors,
        receiver_sources=tuple((i,) for i in range(6)),
        phase_factor=0,
    )
    laplacian = s.eye(6) - s.Matrix(reference.mean_response)
    e, w, beta, nu, rate = s.symbols("e w beta nu rate", positive=True)
    source_jacobian = (
        s.Matrix(observe_phase_source_geometry(reference).scaled_source_jacobian) / s.pi
    )
    degree = s.diag(*(graph.degree(node) for node in graph))
    form_laplacian = degree * laplacian
    metric = s.pi * degree
    zeros = s.zeros(6)
    full = (
        (-e * nu * laplacian)
        .row_join(w * nu * source_jacobian)
        .col_join(((w / beta) * nu * metric.inv() * form_laplacian).row_join(zeros))
    )
    eta = s.Symbol("eta", positive=True)
    # At zero form and phase consensus the cotangent Kx correction has
    # zero linearization; the remaining blocks come from its complete law.
    cotangent_full = (
        (-e * laplacian)
        .row_join(w * source_jacobian)
        .col_join((metric.inv() / eta).row_join(zeros))
    )
    determinants = []
    cotangent_determinants = []
    values = (s.Rational(1, 2), s.Rational(3, 2))
    modes = (s.Matrix((2, 1, -1, -2, -1, 1)), s.Matrix((2, -1, -1, 2, -1, -1)))
    for mode, value in zip(modes, values, strict=True):
        assert laplacian * mode == value * mode
        lift = mode.col_join(s.zeros(6, 1)).row_join(s.zeros(6, 1).col_join(mode))
        tangent = lift.T * full * lift / mode.dot(mode)
        assert full * lift == lift * tangent
        polynomial = (rate * s.eye(2) - tangent).det()
        expected = (
            rate**2 + e * nu * value * rate + w**2 * nu**2 * value**2 / (beta * s.pi**2)
        )
        assert s.expand(polynomial - expected) == 0
        determinants.append(tangent.det())
        cotangent = lift.T * cotangent_full * lift / mode.dot(mode)
        assert cotangent_full * lift == lift * cotangent
        cotangent_determinants.append(cotangent.det())
    # The determinant scales quadratically in the graph mode; the earlier
    # cotangent comparison had a linear mode factor under its own premises.
    assert s.simplify(determinants[1] / determinants[0]) == 9
    assert s.simplify(cotangent_determinants[1] / cotangent_determinants[0]) == 3
    # These are tangent generators; no finite modal manifold is asserted.


def test_equal_replica_reduction_preserves_this_law_without_exchange_rescaling():
    s = pytest.importorskip("sympy")
    a, b, angle = s.symbols("a b angle", real=True)
    w, beta, nu = s.symbols("w beta nu", positive=True)
    h = s.pi * s.sin(angle) / angle
    base_laplacian = s.Matrix(((1, -1), (-1, 1)))
    lift = s.Matrix(((1, 0), (1, 0), (0, 1), (0, 1)))
    adjacency = s.Matrix(((0, 0, 1, 1), (0, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 0)))
    fine_laplacian = 2 * s.eye(4) - adjacency
    form = s.Matrix((a, b))
    assert fine_laplacian * lift == 2 * lift * base_laplacian
    fine_phase_rate = (
        (w / beta) * (2 * h * s.eye(4)).inv() * nu * fine_laplacian * lift * form
    )
    base_phase_rate = (w / beta) * (h * s.eye(2)).inv() * nu * base_laplacian * form
    assert (fine_phase_rate - lift * base_phase_rate).applyfunc(s.simplify) == s.zeros(
        4, 1
    )
    assert (
        s.expand(
            (lift * form).dot(fine_laplacian * lift * form)
            - 4 * form.dot(base_laplacian * form)
        )
        == 0
    )
    # Both Dirichlet and phase edge costs acquire m²=4, so beta is unchanged.
    assert sum(
        1 - s.cos(angle) for i in range(4) for j in range(i + 1, 4) if adjacency[i, j]
    ) == 4 * (1 - s.cos(angle))
    rho = s.Symbol("rho", nonnegative=True)
    source = s.Matrix((angle, -angle)) / s.pi
    fine_source = lift * source
    comparison = base_phase_rate + rho * nu * source
    fine_comparison = fine_phase_rate + rho * nu * fine_source
    assert (fine_comparison - lift * comparison).applyfunc(s.simplify) == s.zeros(4, 1)
    extra_loss = beta * rho * source.dot(h * nu * source)
    fine_extra_loss = beta * rho * fine_source.dot(2 * h * nu * fine_source)
    assert s.simplify(fine_extra_loss - 4 * extra_loss) == 0
    # The supplied passive phase term also respects counted replication;
    # this property alone therefore cannot select zero extra loss.
    e, eta = s.symbols("e eta", positive=True)
    nonlinear = _nonlinear_phase_correction(
        s, base_laplacian * form, s.eye(2), source, nu * s.eye(2), e, beta, eta
    )
    fine_nonlinear = _nonlinear_phase_correction(
        s,
        fine_laplacian * lift * form,
        2 * s.eye(4),
        fine_source,
        nu * s.eye(4),
        e,
        beta,
        eta,
    )
    assert (fine_nonlinear - lift * nonlinear).applyfunc(s.simplify) == s.zeros(4, 1)
    assert (
        s.simplify(
            fine_source.dot(2 * h * fine_nonlinear) - 4 * source.dot(h * nonlinear)
        )
        == 0
    )


def test_global_work_identifies_capacity_separable_coefficients_away_from_consensus():
    s = pytest.importorskip("sympy")
    x = s.Matrix(s.symbols("x0:3", real=True))
    theta = s.Matrix(s.symbols("theta0:3", real=True))
    phases = (s.S.Zero, s.pi / 6, s.pi / 12)
    rows = ((1,), (0, 2), (1,))
    metric, _, source, _ = _metric_differential(s, rows, phases)
    capacities = s.symbols("nu0:3", nonnegative=True)
    capacity = s.diag(*capacities)
    e, w, beta = s.symbols("e w beta", positive=True)
    degree = s.diag(1, 2, 1)
    laplacian = s.Matrix(((1, -1, 0), (-1, 2, -1), (0, -1, 1)))
    energy_form = ((x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2) / 2
    energy_phase = 2 - s.cos(theta[0] - theta[1]) - s.cos(theta[1] - theta[2])
    q = s.Matrix([s.diff(energy_form, value) for value in x])
    phase_gradient = s.Matrix([s.diff(energy_phase, value) for value in theta]).subs(
        dict(zip(theta, phases, strict=True))
    )
    assert all(value != 0 for value in phase_gradient)
    # Independent own-capacity polynomials admit nonlinear alternatives before
    # global work is imposed. Zero constant terms implement local freezing.
    unknowns = s.symbols("c0:9", real=True)
    candidate = s.Matrix(
        [
            sum(unknowns[3 * i + k] * capacities[i] ** (k + 1) for k in range(3))
            for i in range(3)
        ]
    )
    form_rate = capacity * (-e * degree.inv() * laplacian * x + w * source)
    prescribed_loss = e * q.dot(capacity * degree.inv() * q)
    work = q.dot(form_rate) + beta * phase_gradient.dot(candidate) + prescribed_loss
    coefficients = s.Poly(s.expand(work), *capacities).coeffs()
    solutions = s.solve(coefficients, unknowns, dict=True)
    assert len(solutions) == 1
    for i in range(3):
        assert (
            s.simplify(solutions[0][unknowns[3 * i]] - w * q[i] / (beta * metric[i, i]))
            == 0
        )
        assert solutions[0][unknowns[3 * i + 1]] == 0
        assert solutions[0][unknowns[3 * i + 2]] == 0
    selected = candidate.subs(solutions[0])
    assert s.simplify(work.subs(solutions[0])) == 0
    assert selected.subs(dict.fromkeys(capacities, 0)) == s.zeros(3, 1)
    # This finite nonlinear candidate family tests the theorem's mechanism;
    # the proof for arbitrary continuous own-capacity dependence is in §11.


def test_additional_phase_loss_is_passive_but_violates_fixed_form_homogeneity(
    path_reference,
):
    s, _, _, metric, source, _, degree, laplacian, x, _ = path_reference
    e, w, beta = s.symbols("e w beta", positive=True)
    rho = s.Symbol("rho", nonnegative=True)
    capacities = s.symbols("nu0:3", nonnegative=True)
    capacity = s.diag(*capacities)
    q = laplacian * x
    form_rate = capacity * (-e * degree.inv() * q + w * source)
    reference = (w / beta) * metric.inv() * capacity * q
    additional = rho * capacity * source
    comparison = reference + additional
    phase_gradient = -metric * source
    loss = e * q.dot(capacity * degree.inv() * q)
    extra_loss = beta * rho * source.dot(metric * capacity * source)
    work = q.dot(form_rate) + beta * phase_gradient.dot(comparison)
    assert s.simplify(work + loss + extra_loss) == 0
    assert loss.is_nonnegative and extra_loss.is_nonnegative
    for index, nu in enumerate(capacities):
        assert comparison[index].subs(nu, 0) == form_rate[index].subs(nu, 0) == 0
    multiplier = s.Symbol("multiplier", positive=True)
    assert (
        comparison.subs({nu: multiplier * nu for nu in capacities}, simultaneous=True)
        - multiplier * comparison
    ).applyfunc(s.simplify) == s.zeros(3, 1)
    # The complete mean is retained: the new term need not be a zero-mean
    # relative correction when capacities differ.
    mean_change = s.simplify(sum(additional) / 3)
    assert (
        s.simplify(
            mean_change
            - rho
            * s.atan(s.Rational(4, 3))
            * (capacities[0] - capacities[2])
            / (3 * s.pi)
        )
        == 0
    )

    # At every active node the unchanged stationary form row supplies
    # g=e*q/(d*w). Substitution in the full compared phase row gives a strictly
    # positive multiple of q, so q=g=0 remains necessary and sufficient.
    degree_i, metric_i, nu_i = s.symbols("degree_i metric_i nu_i", positive=True)
    q_i = s.Symbol("q_i", real=True)
    equilibrium_coefficient = w / (beta * metric_i) + rho * e / (degree_i * w)
    assert equilibrium_coefficient.is_positive
    assert s.solve(nu_i * q_i * equilibrium_coefficient, q_i) == [0]

    changed_form = multiplier * reference + additional
    homogeneity_defect = (changed_form - multiplier * comparison).applyfunc(s.simplify)
    assert (homogeneity_defect - (1 - multiplier) * additional).applyfunc(
        s.simplify
    ) == s.zeros(3, 1)
    assert homogeneity_defect.subs(
        {multiplier: 2, rho: 1, **dict.fromkeys(capacities, 1)}
    ) != s.zeros(3, 1)
    # For a signed homogeneous phase response, exchange scales linearly in
    # form while L scales quadratically. Any nonzero signed residual then
    # violates passivity at a sufficiently small amplitude of the right sign.
    loss_value = s.Symbol("loss_value", nonnegative=True)
    residual_magnitude = s.Symbol("residual_magnitude", positive=True)
    amplitude = residual_magnitude / (2 * (loss_value + 1))
    positive_work = s.factor(
        -(amplitude**2) * loss_value + amplitude * residual_magnitude
    )
    assert positive_work == residual_magnitude**2 * (loss_value + 2) / (
        4 * (loss_value + 1) ** 2
    )
    assert positive_work.is_positive


def test_passive_phase_coefficient_transforms_with_normalized_capacity(path_reference):
    s, _, _, metric, source, _, _, laplacian, x, capacity = path_reference
    e, w, beta, amplitude, clock = s.symbols("e w beta amplitude clock", positive=True)
    rho = s.Symbol("rho", nonnegative=True)

    def phase_row(form, mobility, weight, scale, relaxation):
        return (
            weight / scale
        ) * metric.inv() * mobility * laplacian * form + relaxation * mobility * source

    original = phase_row(x, capacity, w, beta, rho)
    converted = phase_row(
        amplitude * x, capacity / clock, amplitude * w, amplitude**2 * beta, rho
    )
    assert (converted - original / clock).applyfunc(s.simplify) == s.zeros(3, 1)
    normalization = e + amplitude * w
    normalized = phase_row(
        amplitude * x,
        normalization * capacity / clock,
        amplitude * w / normalization,
        amplitude**2 * beta,
        rho / normalization,
    )
    assert (normalized - converted).applyfunc(s.simplify) == s.zeros(3, 1)
    wrong = phase_row(
        amplitude * x,
        normalization * capacity / clock,
        amplitude * w / normalization,
        amplitude**2 * beta,
        rho,
    )
    assert (
        wrong - converted - (normalization - 1) * rho * capacity * source / clock
    ).applyfunc(s.simplify) == s.zeros(3, 1)
    # This is chart/clock covariance with all parameters transformed, not the
    # fixed-model signed form-response homogeneity tested above.


def test_fixed_pair_jet_separates_passive_completion_without_a_trajectory():
    s = pytest.importorskip("sympy")
    phases = (s.pi / 3, s.S.Zero)
    rows = ((1,), (0,))
    metric, _, source, jacobian = _metric_differential(s, rows, phases)
    capacity = s.diag(1, 2)
    laplacian = s.Matrix(((1, -1), (-1, 1)))
    e = w = s.Rational(1, 2)
    rho = s.Symbol("rho", nonnegative=True)
    form = s.zeros(2, 1)
    q = laplacian * form
    original_phase_rate = w * metric.inv() * capacity * q
    extra_phase_rate = rho * capacity * source
    form_rate = capacity * (-e * q + w * source)
    assert source == s.Matrix((-s.Rational(1, 3), s.Rational(1, 3)))
    assert form_rate == s.Matrix((-s.Rational(1, 6), s.Rational(1, 3)))
    assert original_phase_rate == s.zeros(2, 1)
    assert s.simplify(extra_phase_rate[0] - extra_phase_rate[1]) == -rho
    acceleration_change = (w * capacity * jacobian * extra_phase_rate).applyfunc(
        s.simplify
    )
    assert acceleration_change == s.Matrix((rho / (2 * s.pi), -rho / s.pi))
    assert s.simplify(acceleration_change[0] - acceleration_change[1]) == 3 * rho / (
        2 * s.pi
    )
    assert (
        s.simplify(rho * source.dot(metric * capacity * source)) == s.sqrt(3) * rho / 2
    )

    graph = nx.path_graph(2)
    for node, phase in zip(graph, phases, strict=True):
        graph.nodes[node].update(
            EPI=0.0,
            theta=float(phase),
            nu_f=float(capacity[node, node]),
            delta_nfr=99.0,
        )
    saved = deepcopy(graph)
    model = RelationalExchangeModel(storage_scale=1.0)
    field = evaluate_relational_exchange(graph, model=model)
    tangent = evaluate_relational_uniform_tangent(graph, model=model)
    assert tangent.field == field
    assert field.phase_rate == (0.0, 0.0)
    assert field.form_rate == pytest.approx(tuple(map(float, form_rate)), abs=2e-15)
    for index in range(2):
        assert field.pressure_split_residual[index] == Q(field.pressure[index]) - Q(
            model.phase_weight
        ) * Q(field.phase_source[index])
        assert field.nodal_rate_rounding_defect[index] == Q(field.form_rate[index]) - Q(
            field.capacity[index]
        ) * Q(field.pressure[index])
    # Supplied comparison rates are computed beside the captured baseline,
    # never written into its model/report or installed as an execution mode.
    represented_extra = tuple(
        Q(nu) * Q(g) for nu, g in zip(field.capacity, field.phase_source, strict=True)
    )
    represented_change = tuple(
        Q(model.phase_weight)
        * Q(field.capacity[index])
        * sum(
            (
                Q(coefficient) * rate
                for coefficient, rate in zip(row, represented_extra, strict=True)
            ),
            Q(0),
        )
        for index, row in enumerate(tangent.phase_source_jacobian)
    )
    assert tuple(map(float, represented_change)) == pytest.approx(
        tuple(float(value.subs(rho, 1)) for value in acceleration_change), abs=2e-15
    )
    assert nx.utils.graphs_equal(graph, saved)
    # The ideal derivative evaluated with materialized coefficients is not
    # a derivative of binary64 execution or a certified trajectory enclosure.


def test_phase_loss_changes_consensus_poles_without_selecting_an_equilibrium():
    s = pytest.importorskip("sympy")
    e, w, beta, nu, mode = s.symbols("e w beta nu mode", positive=True)
    rho = s.Symbol("rho", nonnegative=True)
    rate = s.Symbol("rate")
    generator = (
        nu * mode * s.Matrix(((-e, -w / s.pi), (w / (beta * s.pi), -rho / s.pi)))
    )
    polynomial = (rate * s.eye(2) - generator).det()
    trace_coefficient = nu * mode * (e + rho / s.pi)
    determinant = (nu * mode) ** 2 * (e * rho / s.pi + w**2 / (beta * s.pi**2))
    assert s.expand(polynomial - rate**2 - trace_coefficient * rate - determinant) == 0
    assert trace_coefficient.is_positive and determinant.is_positive
    assert (
        s.simplify(
            s.discriminant(polynomial, rate)
            - (nu * mode) ** 2 * ((e - rho / s.pi) ** 2 - 4 * w**2 / (beta * s.pi**2))
        )
        == 0
    )
    storage_coordinates = s.diag(1, s.sqrt(beta))
    transformed = storage_coordinates * generator * storage_coordinates.inv()
    assert s.simplify(
        transformed + transformed.T + 2 * nu * mode * s.diag(e, rho / s.pi)
    ) == s.zeros(2)
    assert (
        s.simplify(
            polynomial.subs(rho, 0)
            - rate**2
            - e * nu * mode * rate
            - w**2 * nu**2 * mode**2 / (beta * s.pi**2)
        )
        == 0
    )
    normalized_discriminant = s.simplify(
        s.discriminant(polynomial, rate) / (nu * mode) ** 2
    ).subs({e: s.Rational(1, 2), w: s.Rational(1, 2), beta: 1})
    real_decay = s.simplify(normalized_discriminant.subs(rho, 0))
    oscillatory_decay = s.simplify(normalized_discriminant.subs(rho, 1))
    assert s.simplify(real_decay - (s.Rational(1, 4) - 1 / s.pi**2)) == 0
    assert s.simplify(oscillatory_decay - (s.Rational(1, 4) - 1 / s.pi)) == 0
    assert real_decay.is_positive and oscillatory_decay.is_negative
    # The same e=w=1/2 and beta=1 therefore admit real decay at rho=0 and
    # damped oscillation at rho=1, despite both having strictly stable poles.
    # Different losses preserve the positive-coefficient modal stability
    # criterion. Existing numerical recovery/capture bounds are not reused.


def test_odd_nonlinear_response_has_fifth_order_equilibrium_jet_not_degree_one():
    s = pytest.importorskip("sympy")
    value, amplitude, contrast, angle = s.symbols(
        "value amplitude contrast angle", real=True
    )
    h = value**3 / (1 + value**2)
    assert s.simplify(h.subs(value, -value) + h) == 0
    assert s.factor(h.subs(value, 2 * value) - 2 * h) == (
        6 * value**3 / ((1 + value**2) * (1 + 4 * value**2))
    )
    # |h(u)| <= u²/2 is a global rational-square certificate. It is not
    # inferred from a collection of finite sampled amplitudes.
    magnitude = s.Symbol("magnitude", nonnegative=True)
    gap = magnitude**2 / 2 - h.subs(value, magnitude)
    certificate = magnitude**2 * (magnitude - 1) ** 2 / (2 * (1 + magnitude**2))
    assert s.factor(gap - certificate) == 0
    assert certificate.is_nonnegative

    e, beta, eta, nu = s.symbols("e beta eta nu", positive=True)
    # On an exact P2 chart, q=amplitude*contrast and
    # g=-amplitude*angle/pi. Both deviations vanish at equilibrium.
    correction = (
        eta
        * e
        * nu
        / s.pi
        * h.subs(value, amplitude * contrast / s.sqrt(beta))
        * (amplitude * angle / s.pi) ** 2
    )
    assert all(
        s.diff(correction, amplitude, order).subs(amplitude, 0) == 0
        for order in range(5)
    )
    assert (
        s.simplify(
            s.diff(correction, amplitude, 5).subs(amplitude, 0) / s.factorial(5)
            - eta
            * e
            * nu
            * contrast**3
            * angle**2
            / (s.pi**3 * beta ** s.Rational(3, 2))
        )
        == 0
    )
    # The equilibrium tangent, and hence its modes, cannot distinguish this
    # law from the reference. Finite-amplitude responses can.


def test_nonlinear_signed_work_has_a_coercive_total_passivity_majorant():
    s = pytest.importorskip("sympy")
    e, beta, degree, metric = s.symbols("e beta degree metric", positive=True)
    nu = s.Symbol("nu", nonnegative=True)
    eta, q, source = s.symbols("eta q source", real=True)
    correction = _nonlinear_phase_correction(
        s, s.Matrix((q,)), s.diag(degree), s.Matrix((source,)), s.diag(nu), e, beta, eta
    )[0]
    loss = e * nu * q**2 / degree
    signed_work = -beta * metric * source * correction
    u, relative_metric = s.symbols("u relative_metric", real=True)
    substitutions = {
        q: degree * s.sqrt(beta) * u,
        metric: s.pi * degree * relative_metric,
    }
    assert (
        s.simplify(
            signed_work.subs(substitutions)
            + eta
            * relative_metric
            * source**3
            * u
            / (1 + u**2)
            * loss.subs(substitutions)
        )
        == 0
    )
    assert correction.subs(nu, 0) == correction.subs(q, 0) == 0
    scale = s.Symbol("scale", positive=True)
    assert s.simplify(correction.subs(nu, scale * nu) - scale * correction) == 0

    # On the regular domain 0<H/(pi*d)<=1 and |g|<1. Write their bounded
    # product H*|g|³/(pi*d) as 1/(1+slack); its zero limit is separate.
    magnitude, slack, loss_value, eta_value = s.symbols(
        "magnitude slack loss_value eta_value", nonnegative=True
    )
    bound_product = 1 / (1 + slack)
    absolute_work = (
        eta_value * loss_value * bound_product * magnitude / (1 + magnitude**2)
    )
    gap = eta_value * loss_value / 2 - absolute_work
    certificate = (
        eta_value
        * loss_value
        / (2 * (1 + slack))
        * (slack + (magnitude - 1) ** 2 / (1 + magnitude**2))
    )
    assert s.factor(gap - certificate) == 0
    assert certificate.is_nonnegative
    margin = s.Symbol("margin", positive=True)
    # eta=2/(1+margin) covers exactly 0<eta<2. A zero loss forces all
    # active q rows to vanish; the unchanged form row then forces active g=0.
    assert s.simplify(
        (1 - eta_value / 2).subs(eta_value, 2 / (1 + margin))
    ) == margin / (1 + margin)
    assert (margin / (1 + margin)).is_positive
    assert (-(1 - eta_value / 2) * loss_value).subs(loss_value, 0) == 0
    w = s.Symbol("w", positive=True)
    admitted_eta = 2 / (1 + margin)
    stationary_phase_rate = nu * w * q / (beta * metric) + correction.subs(
        {source: e * q / (degree * w), eta: admitted_eta}
    )
    coefficient = w / (beta * metric) + (
        admitted_eta
        * e**3
        * q**4
        / (s.pi * degree**3 * s.sqrt(beta) * w**2 * (degree**2 * beta + q**2))
    )
    assert s.simplify(stationary_phase_rate - nu * q * coefficient) == 0
    assert coefficient.is_positive
    # Thus at every active node joint stationarity requires q=0 and then
    # g=0. Inactive nodes impose neither condition on their stored geometry.


def test_nonlinear_response_preserves_whole_law_units_but_not_form_proportionality(
    path_reference,
):
    s, _, _, metric, source, _, degree, laplacian, x, capacity = path_reference
    e, w, beta, eta, amplitude, clock = s.symbols(
        "e w beta eta amplitude clock", positive=True
    )

    def phase_row(form, mobility, diffusion, weight, storage_scale):
        q = laplacian * form
        return (
            weight / storage_scale
        ) * metric.inv() * mobility * q + _nonlinear_phase_correction(
            s, q, degree, source, mobility, diffusion, storage_scale, eta
        )

    original = phase_row(x, capacity, e, w, beta)
    converted = phase_row(
        amplitude * x, capacity / clock, e, amplitude * w, amplitude**2 * beta
    )
    assert (converted - original / clock).applyfunc(s.simplify) == s.zeros(3, 1)
    normalization = e + amplitude * w
    normalized = phase_row(
        amplitude * x,
        normalization * capacity / clock,
        e / normalization,
        amplitude * w / normalization,
        amplitude**2 * beta,
    )
    assert (normalized - converted).applyfunc(s.simplify) == s.zeros(3, 1)
    doubled_form = phase_row(2 * x, capacity, e, w, beta)
    assert (doubled_form - 2 * original).applyfunc(s.simplify) != s.zeros(3, 1)
    # beta, pressure weights and capacity transform in a unit conversion.
    # Keeping those coefficients fixed tests a different constitutive claim.


def test_nonlinear_pair_response_changes_signed_exchange_and_relative_motion():
    s = pytest.importorskip("sympy")
    phases = (s.pi / 3, s.S.Zero)
    metric, _, source, jacobian = _metric_differential(s, ((1,), (0,)), phases)
    capacity, degree = s.diag(1, 2), s.eye(2)
    laplacian = s.Matrix(((1, -1), (-1, 1)))
    form = s.Matrix((s.Rational(1, 2), -s.Rational(1, 2)))
    q = laplacian * form
    e = w = s.Rational(1, 2)
    correction = _nonlinear_phase_correction(s, q, degree, source, capacity, e, 1, 1)
    assert correction == s.Matrix((1 / (36 * s.pi), -1 / (18 * s.pi)))
    signed_work = s.simplify(-source.dot(metric * correction))
    assert signed_work == s.sqrt(3) / (24 * s.pi)
    loss = e * q.dot(capacity * q)
    assert loss == s.Rational(3, 2)
    assert (loss - signed_work).is_positive
    reflected = _nonlinear_phase_correction(s, -q, degree, source, capacity, e, 1, 1)
    assert reflected == -correction
    assert s.simplify(-source.dot(metric * reflected)) == -signed_work
    assert s.simplify(correction[0] - correction[1]) == 1 / (12 * s.pi)
    acceleration_change = (w * capacity * jacobian * correction).applyfunc(s.simplify)
    assert acceleration_change == s.Matrix((-1 / (24 * s.pi**2), 1 / (12 * s.pi**2)))

    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(
            EPI=float(form[node]),
            theta=float(phases[node]),
            nu_f=float(capacity[node, node]),
            delta_nfr=99.0,
        )
    saved = deepcopy(graph)
    model = RelationalExchangeModel(storage_scale=1.0)
    baseline = evaluate_relational_exchange(graph, model=model)
    expected_form = capacity * (-e * q + w * source)
    expected_phase = w * metric.inv() * capacity * q
    assert baseline.form_rate == pytest.approx(
        tuple(map(float, expected_form)), abs=2e-15
    )
    assert baseline.phase_rate == pytest.approx(
        tuple(map(float, expected_phase)), abs=2e-15
    )
    for index in range(2):
        assert baseline.pressure_split_residual[index] == (
            Q(baseline.pressure[index])
            - Q(model.epi_weight) * Q(-q[index])
            - Q(model.phase_weight) * Q(baseline.phase_source[index])
        )
        assert baseline.nodal_rate_rounding_defect[index] == (
            Q(baseline.form_rate[index])
            - Q(baseline.capacity[index]) * Q(baseline.pressure[index])
        )
    assert nx.utils.graphs_equal(graph, saved)
    # Only the native baseline is evaluated. The comparison is exact analytic
    # algebra beside it; no nonuniform state is sent to a uniform tangent API,
    # and neither field supplies a numerical trajectory error enclosure.
