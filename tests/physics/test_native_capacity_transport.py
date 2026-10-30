"""Exact controls for capacity transport on a supplied joined-cycle support.

The alternatives retain the same ten-node state and structural clock. They
change the amount measure in M*nu_dot=-(kappa/2)*B*nu, not the native pressure
or phase law. No runtime stage, trajectory, support event or solver is run.
The final control compares fixed-state accounting before and after a supplied
link change; it does not admit that change as an autonomous event.
"""

import pytest
import sympy as sp


def _clean(expression):
    return sp.simplify(sp.expand(expression))


def _same(left, right):
    assert left.shape == right.shape
    assert all(_clean(a - b) == 0 for a, b in zip(left, right))


def _storage(edges, form, phase):
    return sum(
        (form[i] - form[j]) ** 2 / 2 + 1 - sp.cos(phase[j] - phase[i]) for i, j in edges
    )


@pytest.fixture(scope="module")
def transport():
    # Independently retained neighbor rows and oriented cycle/bridge edges.
    neighbors = (
        (1, 4, 5),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 3),
        (0, 6, 9),
        (5, 7),
        (6, 8),
        (7, 9),
        (5, 8),
    )
    cycle_edges = tuple(
        (offset + i, offset + (i + 1) % 5) for offset in (0, 5) for i in range(5)
    )
    edges = cycle_edges + ((0, 5),)
    incidence = sp.zeros(10, len(edges))
    for column, (left, right) in enumerate(edges):
        incidence[left, column] = 1
        incidence[right, column] = -1
    laplacian = incidence * incidence.T
    adjacency = sp.zeros(10)
    for i, row in enumerate(neighbors):
        for j in row:
            adjacency[i, j] = 1
    degree = sp.diag(*(len(row) for row in neighbors))
    kappa = sp.symbols("kappa", positive=True)
    nu = sp.Matrix(sp.symbols("nu0:10", positive=True))
    measures = sp.Matrix(sp.symbols("measure0:10", positive=True))
    measure = sp.diag(*measures)
    generator = -kappa * measure.inv() * laplacian / 2
    reciprocal = generator.subs(dict.fromkeys(measures, 1))
    normalized = generator.subs({measures[i]: degree[i, i] / 2 for i in range(10)})
    angle = 2 * sp.pi / 5
    locked_phase = sp.Matrix([angle * (i % 5) for i in range(10)])
    return {
        "neighbors": neighbors,
        "edges": edges,
        "cycle_edges": cycle_edges,
        "incidence": incidence,
        "B": laplacian,
        "A": adjacency,
        "D": degree,
        "kappa": kappa,
        "nu": nu,
        "measures": measures,
        "M": measure,
        "G": generator,
        "edge": reciprocal,
        "normalized": normalized,
        "locked_phase": locked_phase,
    }


def test_measure_family_and_cycle_equivalence(transport):
    law = transport
    laplacian, degree, nu, kappa = (law[key] for key in ("B", "D", "nu", "kappa"))
    _same(laplacian, degree - law["A"])
    assert tuple(degree.diagonal()) == (3, 2, 2, 2, 2, 3, 2, 2, 2, 2)
    assert len(law["edges"]) == 11
    neighbor_transfer = sp.Matrix(
        [sum(nu[j] - nu[i] for j in row) for i, row in enumerate(law["neighbors"])]
    )
    _same(law["M"] * law["G"] * nu, kappa * neighbor_transfer / 2)
    _same(law["edge"] * nu, kappa * neighbor_transfer / 2)
    _same(law["normalized"] * nu, kappa * degree.inv() * neighbor_transfer)
    _same(law["edge"], -kappa * laplacian / 2)
    _same(law["normalized"], -kappa * degree.inv() * laplacian)
    # Removing the supplied bridge restores degree two on both components.
    cycle_incidence = law["incidence"][:, :-1]
    cycle_laplacian = cycle_incidence * cycle_incidence.T
    _same(sp.diag(*cycle_laplacian.diagonal()), 2 * sp.eye(10))
    _same(
        -kappa * cycle_laplacian / 2, -kappa * (2 * sp.eye(10)).inv() * cycle_laplacian
    )
    # A common clock multiplier cannot fix a degree-dependent row change.
    assert _clean(law["edge"][0, 1] / law["normalized"][0, 1]) == sp.Rational(3, 2)
    assert _clean(law["edge"][1, 0] / law["normalized"][1, 0]) == 1


def test_weighted_balance_variance_and_positive_generator(transport):
    law = transport
    nu, measures, generator = (law[key] for key in ("nu", "measures", "G"))
    ones = sp.ones(10, 1)
    rates = generator * nu
    _same(generator * ones, sp.zeros(10, 1))
    _same(measures.T * generator, sp.zeros(1, 10))
    _same(law["M"] * generator, (law["M"] * generator).T)
    for i in range(10):
        for j in range(10):
            value = generator[i, j]
            if i != j:
                assert value.is_nonnegative is True
            else:
                assert value.is_negative is True
    # A positive principal cofactor independently checks connected rank nine.
    assert law["B"][:9, :9].det() == 25
    # The common minimum is nondecreasing: after subtracting it all off-node
    # increments are positive-coefficient multiples of nonnegative contrasts.
    gaps = sp.Matrix(sp.symbols("gap0:10", nonnegative=True))
    for i in range(10):
        at_minimum = (generator * gaps)[i].subs(gaps[i], 0)
        assert at_minimum.is_nonnegative is True
        assert (-at_minimum).is_nonpositive is True  # Corresponding maximum.
    center = sp.symbols("fixed_center", real=True)
    variance = sum(measures[i] * (nu[i] - center) ** 2 / 2 for i in range(10))
    actual_derivative = sum(sp.diff(variance, nu[i]) * rates[i] for i in range(10))
    edge_loss = law["kappa"] * sum((nu[i] - nu[j]) ** 2 for i, j in law["edges"]) / 2
    assert _clean(actual_derivative + edge_loss) == 0
    # The invariant amount fixes the consensus value for each chosen measure;
    # it does not identify that measure with physical mass or energy.
    amount = measures.dot(nu)
    assert _clean(sum(sp.diff(amount, nu[i]) * rates[i] for i in range(10))) == 0


def test_native_storage_cancellation_on_joined_support(transport):
    law = transport
    form = sp.Matrix(sp.symbols("x0:10", real=True))
    phase = sp.Matrix(sp.symbols("theta0:10", real=True))
    metric = sp.Matrix(sp.symbols("H0:10", positive=True))
    storage = _storage(law["edges"], form, phase)
    form_gradient = sp.Matrix([sp.diff(storage, value) for value in form])
    phase_gradient = sp.Matrix([sp.diff(storage, value) for value in phase])
    neighbor_form = sp.Matrix(
        [sum(form[i] - form[j] for j in row) for i, row in enumerate(law["neighbors"])]
    )
    neighbor_sine = sp.Matrix(
        [
            sum(sp.sin(phase[j] - phase[i]) for j in row)
            for i, row in enumerate(law["neighbors"])
        ]
    )
    _same(form_gradient, law["B"] * form)
    _same(form_gradient, neighbor_form)
    _same(phase_gradient, -neighbor_sine)
    # On the native regular chart H*g equals this neighbor-sine sum. H here
    # retains that state-dependent metric; it is not a constant sine-law gain.
    source = sp.diag(*metric).inv() * neighbor_sine
    pressure = -law["D"].inv() * neighbor_form / 2 + source / 2
    form_rate = sp.diag(*law["nu"]) * pressure
    phase_rate = sp.diag(*metric).inv() * sp.diag(*law["nu"]) * neighbor_form / 2
    actual_derivative = form_gradient.dot(form_rate) + phase_gradient.dot(phase_rate)
    expected_loss = sum(
        law["nu"][i] * neighbor_form[i] ** 2 / (2 * law["D"][i, i]) for i in range(10)
    )
    assert _clean(actual_derivative + expected_loss) == 0
    assert all(sp.diff(storage, value) == 0 for value in law["nu"])
    # Check the actual relative resultants at the supplied joined lock.
    locked = law["locked_phase"]
    for i, row in enumerate(law["neighbors"]):
        real = _clean(sum(sp.cos(locked[j] - locked[i]) for j in row))
        imaginary = _clean(sum(sp.sin(locked[j] - locked[i]) for j in row))
        assert imaginary == 0 and real.is_positive is True
        assert sp.atan(imaginary / real) == 0
        expected_real = (1 + sp.sqrt(5)) / 2 if i in (0, 5) else (sp.sqrt(5) - 1) / 2
        assert _clean(real - expected_real) == 0
        native_metric = sp.pi * sp.Abs(real) * sp.sinc(0)
        assert _clean(native_metric - sp.pi * expected_real) == 0


def test_port_contrast_discriminates_means_and_cut_balances(transport):
    law = transport
    base, eta = sp.symbols("base eta", positive=True)
    witness = base * sp.ones(10, 1) + eta * sp.eye(10)[:, 0]
    assert all(value.is_positive is True for value in witness)
    edge_rate, normalized_rate = law["edge"] * witness, law["normalized"] * witness
    kappa = law["kappa"]
    expected_edge = sp.zeros(10, 1)
    expected_edge[0] = -3 * kappa * eta / 2
    for i in (1, 4, 5):
        expected_edge[i] = kappa * eta / 2
    expected_normalized = sp.zeros(10, 1)
    expected_normalized[0] = -kappa * eta
    expected_normalized[1] = expected_normalized[4] = kappa * eta / 2
    expected_normalized[5] = kappa * eta / 3
    _same(edge_rate, expected_edge)
    _same(normalized_rate, expected_normalized)
    rate_minor = sp.Matrix(
        [[edge_rate[0], normalized_rate[0]], [edge_rate[1], normalized_rate[1]]]
    ).det()
    assert _clean(rate_minor + kappa**2 * eta**2 / 4) == 0
    # Even a scalar timing adjustment at this particular state cannot match
    # the two non-collinear capacity velocities.
    assert _clean(sum(edge_rate)) == 0
    assert _clean(sum(normalized_rate) - kappa * eta / 3) == 0
    half_degree = sp.ones(1, 10) * law["D"] / 2
    assert _clean((half_degree * normalized_rate)[0]) == 0
    assert _clean((half_degree * edge_rate)[0] + kappa * eta / 2) == 0
    assert _clean(sum(witness) / 10 - (base + eta / 10)) == 0
    assert _clean((half_degree * witness)[0] / 11 - (base + 3 * eta / 22)) == 0
    left, right = sp.Matrix([[1] * 5 + [0] * 5]), sp.Matrix([[0] * 5 + [1] * 5])
    outward = kappa * (law["nu"][0] - law["nu"][5]) / 2
    _same(left * law["M"] * law["G"] * law["nu"], sp.Matrix([-outward]))
    _same(right * law["M"] * law["G"] * law["nu"], sp.Matrix([outward]))
    assert _clean((left * edge_rate)[0] + kappa * eta / 2) == 0
    # Under normalized averaging raw regional sums have internal contributions:
    # their conservation law is for half-degree amounts instead.
    assert _clean((left * normalized_rate)[0]) == 0
    assert _clean((right * normalized_rate)[0] - kappa * eta / 3) == 0
    left_level, right_level = sp.symbols("left_level right_level", positive=True)
    regional_uniform = left_level * left.T + right_level * right.T
    for generator in (law["edge"], law["normalized"]):
        assert _clean(sum(generator * regional_uniform)) == 0
    # Global raw-total drift is blind to a constant profile in each region.
    # This does not make the two regional responses equal.
    # Ordinary regional capacity alone also misses the actual port value.
    uniform = base * sp.ones(10, 1)
    rearranged = uniform + base * (sp.eye(10)[:, 0] - sp.eye(10)[:, 1]) / 4
    assert all(value.is_positive is True for value in rearranged)
    _same(left * rearranged, left * uniform)
    assert _clean((left * law["edge"] * uniform)[0]) == 0
    assert _clean((left * law["edge"] * rearranged)[0] + kappa * base / 8) == 0


def test_matched_native_acceleration_and_blind_cases(transport):
    law = transport
    base, eta = sp.symbols("base eta", positive=True)
    amplitude = sp.symbols("amplitude", real=True, nonzero=True)
    form = sp.Matrix(sp.symbols("x0:10", real=True))
    nu = law["nu"]
    h = (1 + sp.sqrt(5)) / 2
    metric = sp.diag(*(sp.pi * (h if i in (0, 5) else h - 1) for i in range(10)))
    pressure = -law["D"].inv() * law["B"] * form / 2  # g=0 at both aligned twists.
    form_rate = sp.diag(*nu) * pressure
    phase_rate = metric.inv() * sp.diag(*nu) * law["B"] * form / 2
    capacity_difference = (law["edge"] - law["normalized"]) * nu
    # Both models have the same initial native rates. Their form/phase chain
    # rule contributions cancel; only the capacity columns differ initially.
    form_acceleration = form_rate.jacobian(nu) * capacity_difference
    phase_acceleration = phase_rate.jacobian(nu) * capacity_difference
    preparation = {
        **{nu[i]: base + (eta if i == 0 else 0) for i in range(10)},
        **{form[i]: amplitude if i == 0 else 0 for i in range(10)},
    }
    expected_form, expected_phase = sp.zeros(10, 1), sp.zeros(10, 1)
    product = law["kappa"] * eta * amplitude
    expected_form[0], expected_form[5] = product / 4, product / 36
    expected_phase[0], expected_phase[5] = -3 * product / (4 * sp.pi * h), -product / (
        12 * sp.pi * h
    )
    _same(form_acceleration.subs(preparation), expected_form)
    _same(phase_acceleration.subs(preparation), expected_phase)
    uniform_capacity = dict.fromkeys(nu, base)
    _same(capacity_difference.subs(uniform_capacity), sp.zeros(10, 1))
    _same(form_acceleration.subs(uniform_capacity), sp.zeros(10, 1))
    _same(phase_acceleration.subs(uniform_capacity), sp.zeros(10, 1))
    origin = sp.symbols("origin", real=True)
    locked_form = dict.fromkeys(form, origin)
    for response in (form_rate, phase_rate, form_acceleration, phase_acceleration):
        _same(response.subs(locked_form), sp.zeros(10, 1))
    # At exact lock these native rows vanish for every positive capacity
    # profile, so capacity redistribution can be invisible to native outputs.


def test_link_change_separates_measure_and_storage_jumps(transport):
    law = transport
    nu = law["nu"]
    ones = sp.ones(1, 10)
    prior_measure = sp.eye(10)  # Two degree-two C5 components.
    after_measure = law["D"] / 2
    amount_change = (ones * (after_measure - prior_measure) * nu)[0]
    assert _clean(amount_change - (nu[0] + nu[5]) / 2) == 0
    assert amount_change.is_positive is True
    form = sp.Matrix(sp.symbols("x0:10", real=True))
    phase = sp.Matrix(sp.symbols("theta0:10", real=True))
    old_storage = _storage(law["cycle_edges"], form, phase)
    # Assemble the new form storage by incidence; the old expression is an
    # independent edge sum. The selected beta remains one in both states.
    new_storage = (form.T * law["B"] * form)[0] / 2 + sum(
        1 - sp.cos(phase[j] - phase[i]) for i, j in law["edges"]
    )
    storage_change = _clean(new_storage - old_storage)
    expected = (form[0] - form[5]) ** 2 / 2 + 1 - sp.cos(phase[5] - phase[0])
    assert _clean(storage_change - expected) == 0
    origin = sp.symbols("origin", real=True)
    aligned_lock = {
        **dict.fromkeys(form, origin),
        **{phase[i]: law["locked_phase"][i] for i in range(10)},
    }
    assert _clean(storage_change.subs(aligned_lock)) == 0
    # The degree-weighted amount can change with zero storage change and no
    # nodal capacity write. These are distinct accounting questions, not an
    # event occurrence law or a reservoir that has funded the added bridge.
