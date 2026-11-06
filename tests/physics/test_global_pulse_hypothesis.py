"""Constructed controls separating global-pulse hypotheses and observations.

The time-dependent stiffness and extra canonical source below are supplied
alternative laws, not engine defaults or an emerged primordial source. These
tests use exact identities and instantaneous fields only; no trajectory,
frequency scan, pulse producer or new runtime report is evaluated.
"""

from fractions import Fraction as Q
from itertools import product

import networkx as nx
import sympy as sp

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DEPI, ALIAS_DNFR
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.metrics.common import compute_coherence
from tnfr.physics._sine_flow import _held_sine_neighbor_field
from tnfr.physics.relational_sine_resonance import _conservative_full_tangent


def _zero(matrix):
    return sp.Matrix(matrix).applyfunc(sp.simplify) == sp.zeros(*matrix.shape)


def _parts(edges, forms, phases):
    """Independent edge storage and gradients, retaining every fine row."""
    size = len(forms)
    laplacian, sine = sp.zeros(size), sp.zeros(size, 1)
    potential = sp.S(0)
    for i, j in edges:
        laplacian[i, i] += 1
        laplacian[j, j] += 1
        laplacian[i, j] -= 1
        laplacian[j, i] -= 1
        current = sp.sin(phases[j] - phases[i])
        sine[i] += current
        sine[j] -= current
        potential += 1 - sp.cos(phases[j] - phases[i])
    return laplacian, laplacian * sp.Matrix(forms), sine, potential


def _path():
    forms, phases = sp.symbols("x0:3", real=True), sp.symbols("theta0:3", real=True)
    laplacian, q, sine, potential = _parts(((0, 1), (1, 2)), forms, phases)
    return (
        forms,
        phases,
        laplacian,
        q,
        sine,
        potential,
        sp.diag(1, sp.Rational(1, 2), 2),
    )


def _cycle_tangent():
    edges = ((0, 1), (1, 2), (2, 3), (0, 3))
    laplacian, _, scaled, _ = _conservative_full_tangent(
        edges, (Q(1, 2),) * 4, (Q(1),) * 4
    )
    # The shared rational assembly represents pi*d/dt, not d/dt.
    return sp.Matrix(laplacian), sp.Matrix(scaled) / sp.pi


def test_common_clock_keeps_both_rows_and_the_second_order_clock_derivative():
    time = sp.symbols("t", real=True)
    damping, coupling, mobility = sp.symbols("d A B", positive=True)
    gamma, form, phase = (sp.Function(name)(time) for name in ("gamma", "X", "Theta"))
    phase_acceleration = sp.diff(gamma * mobility * form, time).subs(
        sp.diff(form, time), gamma * (-damping * form - coupling * phase)
    )
    eliminated = phase_acceleration.subs(
        form, sp.diff(phase, time) / (gamma * mobility)
    )
    expected = (sp.diff(gamma, time) / gamma - gamma * damping) * sp.diff(
        phase, time
    ) - gamma**2 * coupling * mobility * phase
    assert sp.simplify(eliminated - expected) == 0

    # An explicit reparameterized free solution detects the omitted derivative.
    rate, tau, omega = sp.exp(time), sp.exp(time) - 1, 2 / sp.pi
    x, theta = sp.cos(omega * tau), sp.sin(omega * tau)
    assert sp.simplify(sp.diff(x, time) + rate * omega * theta) == 0
    assert sp.simplify(sp.diff(theta, time) - rate * omega * x) == 0
    wrong_residual = sp.diff(theta, time, 2) + rate**2 * omega**2 * theta
    assert sp.simplify(wrong_residual - sp.diff(theta, time)) == 0
    assert wrong_residual.subs(time, 0) == omega

    _, _, full, _ = _conservative_full_tangent(((0, 1),), (Q(1),) * 2, (Q(1),))
    embedding = sp.Matrix(((1, 0), (-1, 0), (0, 1), (0, -1)))
    block = sp.Matrix(((0, -omega), (omega, 0)))
    assert sp.Matrix(full) * embedding / sp.pi == embedding * block
    assert block * sp.zeros(2, 1) == sp.zeros(2, 1)
    one_row_change = sp.diag(2, 1) * block
    assert one_row_change != 2 * block
    assert one_row_change.det() == 2 * block.det()


def test_common_origins_are_invisible_but_uniform_pressure_need_not_be():
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    neighbors = (((1, 1, Q(1)),), ((0, 1, Q(1)), (2, 1, Q(1))), ((1, 1, Q(1)),))
    capacity = (Q(1), Q(2), Q(3))
    field = _held_sine_neighbor_field(model, (1, 2, 1), capacity, neighbors)
    state = (Q(1), Q(-1), Q(2), Q(0), Q(1, 8), Q(-1, 4))
    # Dyadic offsets keep this interval equality free from admission rounding;
    # the symbolic substitutions below cover arbitrary common real offsets.
    translated = tuple(value + Q(7, 4) for value in state[:3]) + tuple(
        value - Q(11, 8) for value in state[3:]
    )
    assert field(state) == field(translated)

    forms, phases, laplacian, q, sine, _, _ = _path()
    assert laplacian * sp.ones(3, 1) == sp.zeros(3, 1)
    assert sum(sine) == 0
    # A common additive form RATE is a removable origin translation. Pressure
    # enters through nu_i instead: the same scalar pressure is not this rate.
    assert laplacian * sp.Matrix(capacity) == sp.Matrix((-1, 0, 1))
    assert laplacian * sp.Matrix((Q(2),) * 3) == sp.zeros(3, 1)
    offset = sp.symbols("offset", real=True)
    changed_q = q.subs(dict(zip(forms, (x + offset for x in forms))), simultaneous=True)
    changed_sine = sine.subs(
        dict(zip(phases, (theta + offset for theta in phases))), simultaneous=True
    )
    assert _zero(changed_q - q) and _zero(changed_sine - sine)


def test_additive_probe_needs_modal_overlap_and_retains_conjugate_work():
    laplacian, generator = _cycle_tangent()
    driven, dark = sp.Matrix((1, 0, -1, 0)), sp.Matrix((0, 1, 0, -1))
    assert laplacian * driven == 2 * driven
    assert laplacian * dark == 2 * dark
    assert dark.dot(driven) == 0
    column = driven.col_join(sp.zeros(4, 1))
    output = (laplacian * dark).col_join(sp.zeros(4, 1))
    active = (laplacian * driven).col_join(sp.zeros(4, 1))
    assert active.dot(column) == 4
    # Eight Markov rows exhaust the dimension; a dark port is not excited
    # merely because its free frequency matches the driven mode's frequency.
    for _ in range(8):
        assert sp.simplify(output.dot(column)) == 0
        column = generator * column

    forms, phases, laplacian, q, sine, potential, metric = _path()
    beta, b, loss, signal = sp.symbols("beta b e u", positive=True)
    port = sp.Matrix((1, -1, 0))
    energy = (sp.Matrix(forms).dot(q) / 2) + beta * potential
    form_rate = -loss * metric * q + beta * b * metric * sine + signal * port
    phase_rate = b * metric * q
    gradient = sp.Matrix([sp.diff(energy, v) for v in forms + phases])
    rate = sp.simplify(gradient.dot(form_rate.col_join(phase_rate)))
    expected = -loss * q.dot(metric * q) + signal * port.dot(q)
    assert sp.simplify(rate - expected) == 0
    assert port.dot(q) == (laplacian * port).dot(sp.Matrix(forms))
    assert (laplacian * sp.ones(3, 1)).dot(sp.Matrix(forms)) == 0


def test_intrinsic_susceptibility_and_peak_depend_on_the_measured_output():
    s = sp.symbols("s")
    d, k, b, weight = sp.symbols("d k b lambda", positive=True)
    block = sp.Matrix(((-d, -k / b), (b, 0)))
    transfer = (s * sp.eye(2) - block).inv() * sp.Matrix((1, 0))
    denominator = s**2 + d * s + k
    assert sp.simplify(transfer[0] - s / denominator) == 0
    assert sp.simplify(transfer[1] - b / denominator) == 0
    assert sp.simplify(weight * transfer[0] - weight * s / denominator) == 0

    z = sp.symbols("omega_squared", nonnegative=True)
    squared_denominator = (k - z) ** 2 + d**2 * z
    form_gain, phase_gain = z / squared_denominator, b**2 / squared_denominator
    assert (
        sp.simplify(sp.diff(form_gain, z) - (k**2 - z**2) / squared_denominator**2) == 0
    )
    assert (
        sp.simplify(
            sp.diff(phase_gain, z)
            + b**2 * (2 * z + d**2 - 2 * k) / squared_denominator**2
        )
        == 0
    )
    # Real free poles coexist with an interior form/work maximum and DC phase
    # maximum. A second declared damping value has a positive phase maximum.
    assert (d**2 - 4 * k).subs({d: 3, k: 1}) == 5
    assert sp.diff(form_gain, z).subs({d: 3, k: 1, z: 1}) == 0
    assert (
        sp.simplify(
            sp.diff(phase_gain, z).subs({d: 3, k: 1, b: 1})
            + (2 * z + 7) / (z**2 + 7 * z + 1) ** 2
        )
        == 0
    )
    assert sp.diff(phase_gain, z).subs({d: 1, k: 1, z: sp.Rational(1, 2)}) == 0
    assert block * sp.zeros(2, 1) == sp.zeros(2, 1)

    # The baseline C4 mode has r=1, e=w=1/2, beta=1, in original t.
    model = RelationalExchangeModel(1)
    e, w = map(Q, model.effective_weights)
    assert e == w == Q(1, 2)
    actual_d, actual_k = sp.Rational(e), sp.Rational(w) ** 2 / sp.pi**2
    assert (actual_d**2 - 4 * actual_k).is_positive is True
    assert (actual_d**2 - 2 * actual_k).is_positive is True


def test_genuine_stiffness_modulation_has_parameter_work_jumps_and_a_mode_equation():
    forms, phases, _, q, sine, potential, metric = _path()
    beta, b, loss, rho, rho_rate = sp.symbols("beta b e rho rho_dot", positive=True)
    energy = sp.Matrix(forms).dot(q) / 2 + beta * rho * potential
    form_rate = -loss * metric * q + beta * b * rho * metric * sine
    phase_rate = b * metric * q
    gradient = sp.Matrix([sp.diff(energy, v) for v in forms + phases])
    energy_rate = (
        gradient.dot(form_rate.col_join(phase_rate)) + sp.diff(energy, rho) * rho_rate
    )
    assert (
        sp.simplify(
            energy_rate + loss * q.dot(metric * q) - beta * rho_rate * potential
        )
        == 0
    )
    before, after = sp.symbols("rho_before rho_after", positive=True)
    assert (
        sp.simplify(
            energy.subs(rho, after)
            - energy.subs(rho, before)
            - beta * (after - before) * potential
        )
        == 0
    )

    # Keep b fixed: scaling only phase storage is not a common clock change.
    # This alternative is not passed through the normalized model constructor.
    embedding = sp.Matrix(((1, 0), (-1, 0), (0, 1), (0, -1)))
    for stiffness in (Q(1), Q(4)):
        _, _, full, _ = _conservative_full_tangent(((0, 1),), (Q(1),) * 2, (stiffness,))
        block = sp.Matrix(((0, -2 * sp.Rational(stiffness) / sp.pi), (2 / sp.pi, 0)))
        assert sp.Matrix(full) * embedding / sp.pi == embedding * block
        assert block[1, 0] * block[0, 1] == -4 * sp.Rational(stiffness) / sp.pi**2
    time = sp.symbols("t", real=True)
    x, theta, varying = (sp.Function(name)(time) for name in ("X", "Theta", "rho"))
    acceleration = sp.diff(2 * x / sp.pi, time).subs(
        sp.diff(x, time), -2 * varying * theta / sp.pi
    )
    assert sp.simplify(acceleration + 4 * varying * theta / sp.pi**2) == 0


def test_two_positive_stiffness_quarters_amplify_a_seed_and_account_for_pump_work():
    time, frequency = sp.symbols("t omega", positive=True)
    transition = sp.Matrix(
        (
            (sp.cos(frequency * time), sp.sin(frequency * time) / frequency),
            (-frequency * sp.sin(frequency * time), sp.cos(frequency * time)),
        )
    )
    generator = sp.Matrix(((0, 1), (-(frequency**2), 0)))
    assert _zero(sp.diff(transition, time) - generator * transition)
    assert transition.subs(time, 0) == sp.eye(2)
    quarter = transition.subs(time, sp.pi / (2 * frequency))
    # These are the actual original-time P2 frequencies from the preceding
    # supplied stiffness family, not frequencies after silently changing t.
    low, high = 2 / sp.pi, 4 / sp.pi
    first, second = quarter.subs(frequency, low), quarter.subs(frequency, high)
    monodromy = second * first
    assert monodromy == sp.diag(-sp.Rational(1, 2), -2)
    assert first.det() == second.det() == monodromy.det() == 1
    assert first * first == -sp.eye(2)
    assert monodromy * sp.zeros(2, 1) == sp.zeros(2, 1)
    seed = sp.Matrix((0, 1))
    midway, endpoint = first * seed, monodromy * seed

    def storage(state, omega):
        return (state[1] ** 2 + omega**2 * state[0] ** 2) / 2

    upward_work = (high**2 - low**2) * midway[0] ** 2 / 2
    downward_work = (low**2 - high**2) * endpoint[0] ** 2 / 2
    assert storage(midway, low) == storage(seed, low) == sp.Rational(1, 2)
    assert storage(endpoint, high) == storage(midway, high) == 2
    assert upward_work == sp.Rational(3, 2) and downward_work == 0
    assert storage(endpoint, low) - storage(seed, low) == upward_work + downward_work


def test_autonomous_positive_source_requires_reaction_for_total_storage_balance():
    forms, phases, _, q, sine, potential, metric = _path()
    beta, b, epsilon, frequency = sp.symbols("beta b epsilon Omega", positive=True)
    source, momentum = sp.symbols("Q P", real=True)
    modulation = sp.exp(epsilon * source)
    nodal = sp.Matrix(forms).dot(q) / 2 + beta * modulation * potential
    reservoir = (momentum**2 + frequency**2 * source**2) / 2
    form_rate, phase_rate = beta * b * modulation * metric * sine, b * metric * q
    reaction = -beta * epsilon * modulation * potential
    source_rate, momentum_rate = momentum, -(frequency**2) * source + reaction
    gradient = sp.Matrix([sp.diff(nodal, v) for v in forms + phases])
    nodal_rate = (
        gradient.dot(form_rate.col_join(phase_rate))
        + sp.diff(nodal, source) * source_rate
    )
    reservoir_rate = (
        sp.diff(reservoir, source) * source_rate
        + sp.diff(reservoir, momentum) * momentum_rate
    )
    exchange = beta * epsilon * modulation * momentum * potential
    assert sp.simplify(nodal_rate - exchange) == 0
    assert sp.simplify(reservoir_rate + exchange) == 0
    assert sp.simplify(nodal_rate + reservoir_rate) == 0
    missing_reaction_rate = (
        nodal_rate + frequency**2 * source * momentum - momentum * frequency**2 * source
    )
    assert sp.simplify(missing_reaction_rate - exchange) == 0
    witness = {beta: 1, epsilon: 1, source: 0, momentum: 1}
    witness.update(dict(zip(phases, (0, sp.pi / 2, 0))))
    assert sp.simplify(missing_reaction_rate.subs(witness)) == 2
    assert modulation.is_positive is True

    # This source is an added state and Hamiltonian. Removing its coupling
    # recovers the old field plus a decoupled oscillator, not an emerged pump.
    assert sp.simplify(reaction.subs(epsilon, 0)) == 0
    assert _zero(form_rate.subs(epsilon, 0) - beta * b * metric * sine)


def test_pulse_and_diagnostic_observations_do_not_establish_global_emergence():
    time = sp.symbols("t", real=True)
    amplitude, frequency = sp.symbols("A Omega", positive=True)
    source, momentum = amplitude * sp.cos(
        frequency * time
    ), -amplitude * frequency * sp.sin(frequency * time)
    assert sp.simplify(sp.diff(source, time) - momentum) == 0
    assert sp.simplify(sp.diff(momentum, time) + frequency**2 * source) == 0
    forms, phases, _, q, sine, potential, _ = _path()
    consensus = dict(zip(forms + phases, (sp.S(3),) * 3 + (sp.S(2),) * 3))
    assert q.subs(consensus) == sine.subs(consensus) == sp.zeros(3, 1)
    assert potential.subs(consensus) == 0
    # The auxiliary oscillator can pulse while every nodal coordinate stays
    # at exact consensus. Its existence does not generate a nodal perturbation.
    assert source.subs(time, 0) != source.subs(time, sp.pi / frequency)

    # Conversely, the existing doubled-C5 internal family hides its motion
    # from the global phase phasor, even before discarding internal coordinates.
    edges = tuple(
        sorted(
            {
                tuple(sorted((2 * i + p, 2 * ((i + 1) % 5) + r)))
                for i in range(5)
                for p, r in product((0, 1), repeat=2)
            }
        )
    )
    fine_forms = tuple(sp.Rational(1 if i % 2 == 0 else -1, 8) for i in range(10))
    fine_phases = tuple(2 * sp.pi * (i // 2) / 5 for i in range(10))
    _, fine_q, fine_sine, _ = _parts(edges, fine_forms, fine_phases)
    assert _zero(fine_sine)
    assert fine_q == 4 * sp.Matrix(fine_forms)
    phase_rates = fine_q / (4 * sp.pi)
    assert phase_rates[0] == 1 / (8 * sp.pi) and phase_rates[1] == -1 / (8 * sp.pi)
    assert sp.simplify(sum(sp.cos(value) for value in fine_phases)) == 0
    assert sp.simplify(sum(sp.sin(value) for value in fine_phases)) == 0
    delta = sp.symbols("delta", real=True)
    for trig in (sp.cos, sp.sin):
        phasor = sum(
            trig(2 * sp.pi * i / 5 + sign * delta) for i in range(5) for sign in (1, -1)
        )
        assert sp.simplify(sp.expand_trig(phasor)) == 0

    graph = nx.Graph()
    graph.add_edges_from(edges)
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_DNFR, 0)
        set_attr(graph.nodes[node], ALIAS_DEPI, 0)
    assert compute_coherence(graph) == 1.0
    # Pressures/form rates really vanish here; this is not stale-metric data.
    # C=1 nevertheless misses the nonzero primitive phase row.
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    neighbor_rows = tuple(
        tuple((j, 1, Q(1)) for j in sorted(graph.neighbors(i))) for i in range(10)
    )
    field = _held_sine_neighbor_field(model, (4,) * 10, (Q(1),) * 10, neighbor_rows)
    # Enclose the same exact twist, rather than substituting a different phase
    # state. Outward intervals need only contain the independently proved zero.
    pulse_point_box = tuple(
        Q(int(value.p), int(value.q)) for value in fine_forms
    ) + tuple(pi_interval() * Q(2 * (i // 2), 5) for i in range(10))
    rates = field(pulse_point_box)
    assert all(value.contains(0) and value.width < Q(1, 2**100) for value in rates[:10])
    assert (rates[10] * pi_interval()).contains(Q(1, 8))
    assert (rates[11] * pi_interval()).contains(Q(-1, 8))
