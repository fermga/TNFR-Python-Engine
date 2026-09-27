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
