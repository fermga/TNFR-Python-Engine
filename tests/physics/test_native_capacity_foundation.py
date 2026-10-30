"""Six symbolic controls for native exchange with released C5 capacities.

The supplied continuous capacity law is compared with held capacity in the
same structural clock. The native resultant and metric are reconstructed
independently from their defining local formulas, not from a normal form.
No runtime, adaptive policy, operator event, trajectory or solver is invoked.
"""

import pytest
import sympy as sp


def _zero(expression):
    return sp.simplify(sp.expand(expression)) == 0


def _same_matrix(left, right):
    assert left.shape == right.shape
    assert all(_zero(a - b) for a, b in zip(left, right))


@pytest.fixture(scope="module")
def native():
    """Build the full local field before any linearization or reduction."""
    e, w, beta, kappa, mean = sp.symbols("e w beta kappa nu_bar", positive=True)
    origin = sp.symbols("x_bar", real=True)
    x = sp.Matrix(sp.symbols("x0:5", real=True))
    delta = sp.Matrix(sp.symbols("delta0:5", real=True))
    nu = sp.Matrix(sp.symbols("nu0:5", positive=True))
    edges = tuple((index, (index + 1) % 5) for index in range(5))
    incidence = sp.zeros(5)
    for column, (left, right) in enumerate(edges):
        incidence[left, column], incidence[right, column] = 1, -1
    laplacian = incidence * incidence.T
    c, s = sp.cos(2 * sp.pi / 5), sp.sin(2 * sp.pi / 5)
    real, imaginary = [], []
    for index in range(5):
        right = delta[(index + 1) % 5] - delta[index]
        left = delta[(index - 1) % 5] - delta[index]
        # The closing edge has the same phasor as its continuous lifted gap.
        real.append(
            c * sp.cos(right) - s * sp.sin(right) + c * sp.cos(left) + s * sp.sin(left)
        )
        imaginary.append(
            s * sp.cos(right) + c * sp.sin(right) - s * sp.cos(left) + c * sp.sin(left)
        )
    angle = tuple(sp.atan(im / re) for re, im in zip(real, imaginary))
    source = sp.Matrix([value / sp.pi for value in angle])
    metric = tuple(
        sp.pi * sp.sqrt(re**2 + im**2) * sp.sinc(value)
        for re, im, value in zip(real, imaginary, angle)
    )
    gradient = laplacian * x
    pressure = -e * gradient / 2 + w * source
    form_rate = sp.diag(*nu) * pressure
    phase_rate = sp.Matrix(
        [w * nu[i] * gradient[i] / (beta * metric[i]) for i in range(5)]
    )
    capacity_rate = -kappa * laplacian * nu / 2
    field = form_rate.col_join(phase_rate).col_join(capacity_rate)
    state = x.col_join(delta).col_join(nu)
    uniform_form = dict.fromkeys(x, origin)
    winding = dict.fromkeys(delta, sp.S.Zero)
    uniform_capacity = dict.fromkeys(nu, mean)
    # q=0 kills phase-metric derivatives before evaluating the removable
    # sinc derivative at its zero argument. The field itself has sinc(0)=1.
    full_derivative = field.jacobian(state)
    jacobian = (
        full_derivative.subs(uniform_form)
        .subs(winding)
        .subs(uniform_capacity)
        .applyfunc(sp.simplify)
    )
    return {
        "e": e,
        "w": w,
        "beta": beta,
        "kappa": kappa,
        "mean": mean,
        "x": x,
        "delta": delta,
        "nu": nu,
        "edges": edges,
        "incidence": incidence,
        "B": laplacian,
        "c": c,
        "real": real,
        "imaginary": imaginary,
        "source": source,
        "metric": metric,
        "gradient": gradient,
        "pressure": pressure,
        "form_rate": form_rate,
        "phase_rate": phase_rate,
        "capacity_rate": capacity_rate,
        "field": field,
        "state": state,
        "uniform_form": uniform_form,
        "winding": winding,
        "uniform_capacity": uniform_capacity,
        "jacobian": jacobian,
        "full_derivative": full_derivative,
    }


def test_full_native_fifteen_state_jacobian_from_resultant_formula(native):
    b, c, mean = native["B"], native["c"], native["mean"]
    e, w, beta, kappa = (native[name] for name in ("e", "w", "beta", "kappa"))
    assert c.is_positive is True
    for re, im, source, metric in zip(
        native["real"], native["imaginary"], native["source"], native["metric"]
    ):
        assert _zero(re.subs(native["winding"]) - 2 * c)
        assert _zero(im.subs(native["winding"]))
        assert _zero(source.subs(native["winding"]))
        assert _zero(metric.subs(native["winding"]) - 2 * sp.pi * c)
    expected = sp.zeros(15)
    expected[:5, :5] = -mean * e * b / 2
    expected[:5, 5:10] = -mean * w * b / (2 * sp.pi)
    expected[5:10, :5] = mean * w * b / (2 * beta * sp.pi * c)
    expected[10:15, 10:15] = -kappa * b / 2
    _same_matrix(native["jacobian"], expected)
    # The equal-capacity family is an equilibrium, including all fifteen rows.
    equilibrium = (
        native["field"]
        .subs(native["uniform_form"])
        .subs(native["winding"])
        .subs(native["uniform_capacity"])
    )
    _same_matrix(equilibrium, sp.zeros(15, 1))
    _same_matrix(native["jacobian"][:10, 10:15], sp.zeros(10, 5))


def test_c5_modes_include_native_quadratics_and_capacity_decay(native):
    b, jacobian = native["B"], native["jacobian"]
    e, w, beta, kappa, mean, c = (
        native[name] for name in ("e", "w", "beta", "kappa", "mean", "c")
    )
    parameter = sp.symbols("lambda")
    mu_low, mu_high = (5 - sp.sqrt(5)) / 2, (5 + sp.sqrt(5)) / 2
    spatial = b.charpoly(parameter)
    expected_spatial = (
        parameter * (parameter - mu_low) ** 2 * (parameter - mu_high) ** 2
    )
    assert _zero(spatial.as_expr() - expected_spatial)
    # The odd vectors are sine vectors divided by their nonzero first entry.
    # This exact basis stays in Q(sqrt(5)), avoiding nested sine radicals.
    cos_low, cos_high = (sp.sqrt(5) - 1) / 4, -(sp.sqrt(5) + 1) / 4
    basis = [
        sp.ones(5, 1),
        sp.Matrix([1, cos_low, cos_high, cos_high, cos_low]),
        sp.Matrix([0, 1, 2 * cos_low, -2 * cos_low, -1]),
        sp.Matrix([1, cos_high, cos_low, cos_low, cos_high]),
        sp.Matrix([0, 1, 2 * cos_high, -2 * cos_high, -1]),
    ]
    eigenvalues = [sp.S.Zero, mu_low, mu_low, mu_high, mu_high]
    basis_matrix = sp.Matrix.hstack(*basis)
    norms = (
        sp.S(5),
        sp.Rational(5, 2),
        5 - sp.sqrt(5),
        sp.Rational(5, 2),
        5 + sp.sqrt(5),
    )
    _same_matrix(basis_matrix.T * basis_matrix, sp.diag(*norms))
    assert all(norm.is_positive is True for norm in norms)
    factors = []
    for vector, mu in zip(basis, eigenvalues):
        _same_matrix(b * vector, mu * vector)
        embedded = sp.zeros(15, 3)
        for row in range(3):
            embedded[5 * row : 5 * (row + 1), row] = vector
        # Projection of the independently differentiated fifteen-state field.
        modal = (embedded.T * jacobian * embedded / (vector.dot(vector))).applyfunc(
            sp.simplify
        )
        _same_matrix(jacobian * embedded, embedded * modal)
        actual = modal.charpoly(parameter).as_expr()
        if mu == 0:
            assert _zero(actual - parameter**3)
            continue
        quadratic = (
            parameter**2
            + mean * e * mu * parameter / 2
            + mean**2 * w**2 * mu**2 / (4 * beta * sp.pi**2 * c)
        )
        assert _zero(actual - (parameter + kappa * mu / 2) * quadratic)
        coefficients = sp.Poly(quadratic, parameter).all_coeffs()
        assert all(
            sp.simplify(coefficient).is_positive is True for coefficient in coefficients
        )
        assert sp.simplify(kappa * mu / 2).is_positive is True
        # Holding all capacities removes only their four nonuniform decay rates.
        assert _zero(actual.subs(kappa, 0) - parameter * quadratic)
        factors.append(actual)
    assert len(factors) == 4  # Twelve stable dimensions plus three neutral ones.


def test_native_storage_balance_survives_capacity_release(native):
    x, delta, nu = (native[name] for name in ("x", "delta", "nu"))
    e, w, beta, kappa = (native[name] for name in ("e", "w", "beta", "kappa"))
    phase_storage = sum(
        1 - sp.cos(2 * sp.pi / 5 + delta[right] - delta[left])
        for left, right in native["edges"]
    )
    storage = (x.T * native["B"] * x)[0] / 2 + beta * phase_storage
    _same_matrix(sp.Matrix([sp.diff(storage, item) for item in x]), native["gradient"])
    _same_matrix(
        sp.Matrix([sp.diff(storage, item) for item in delta]).applyfunc(sp.expand_trig),
        (-beta * sp.Matrix(native["imaginary"])).applyfunc(sp.expand_trig),
    )
    assert all(sp.diff(storage, item) == 0 for item in nu)
    # Independently verify H*g=imaginary resultant on the positive-real chart, also
    # at the removable zero of the phase displacement.
    re = sp.symbols("re", positive=True)
    im = sp.symbols("im", real=True)
    angle = sp.atan(im / re)
    g = angle / sp.pi
    h = sp.pi * sp.sqrt(re**2 + im**2) * sp.sinc(angle)
    nonzero_angle = sp.symbols("nonzero_angle", real=True, nonzero=True)
    assert _zero(
        (nonzero_angle * sp.sinc(nonzero_angle)).rewrite(sp.sin) - sp.sin(nonzero_angle)
    )
    assert _zero(sp.sqrt(re**2 + im**2) * sp.sin(angle) - im)
    assert _zero(h.subs(im, 0) - sp.pi * re)
    assert g.subs(im, 0) == 0
    gradients = native["gradient"]
    independent_metrics = sp.symbols("h0:5", positive=True)
    currents = sp.symbols("s0:5", real=True)
    total = sp.S.Zero
    for index in range(5):
        pressure = (
            -e * gradients[index] / 2 + w * currents[index] / independent_metrics[index]
        )
        rate_x = nu[index] * pressure
        rate_theta = (
            w * nu[index] * gradients[index] / (beta * independent_metrics[index])
        )
        total += gradients[index] * rate_x - beta * currents[index] * rate_theta
        total += sp.diff(storage, nu[index]) * native["capacity_rate"][index]
    expected = -e * sum(nu[i] * gradients[i] ** 2 for i in range(5)) / 2
    assert _zero(total - expected)
    assert sp.diff(sp.expand(total), kappa) == 0


def _acceleration_difference(native):
    """Differentiate the same two native rows along the two full vectors."""
    physical = native["field"][:10, 0]
    derivative = native["full_derivative"][:10, :]
    held = physical.col_join(sp.zeros(5, 1))
    released = native["field"]
    # Subtract directions before multiplication; shared x/theta chain-rule
    # terms cancel without expanding their large trigonometric expressions.
    return derivative * (released - held)


def test_matched_capacity_release_has_the_claimed_acceleration(native):
    difference = _acceleration_difference(native)
    loss = native["B"] * native["nu"] / 2
    expected_x = -native["kappa"] * sp.diag(*loss) * native["pressure"]
    expected_theta = sp.Matrix(
        [
            -native["kappa"]
            * native["w"]
            * native["gradient"][i]
            * loss[i]
            / (native["beta"] * native["metric"][i])
            for i in range(5)
        ]
    )
    _same_matrix(difference, expected_x.col_join(expected_theta))
    # Releasing nu does not change either initial velocity at a matched state.
    held = native["field"].subs(native["kappa"], 0)
    _same_matrix(native["field"][:10, 0], held[:10, 0])


def test_acceleration_discriminator_has_blind_cases_and_nonzero_witness(native):
    difference = _acceleration_difference(native)
    _same_matrix(difference.subs(native["uniform_capacity"]), sp.zeros(10, 1))
    _same_matrix(
        difference.subs(native["uniform_form"]).subs(native["winding"]),
        sp.zeros(10, 1),
    )
    # Positive, unequal capacities on the same regular winding source. A form
    # impulse supplies pressure independently; no response defines that law.
    witness = dict(zip(native["x"], (1, 0, 0, 0, 0)))
    witness.update(dict(zip(native["nu"], (2, 1, 1, 1, 1))))
    witness.update(native["winding"])
    actual = difference.subs(witness).applyfunc(sp.simplify)
    profile = sp.Matrix([1, sp.Rational(1, 4), 0, 0, sp.Rational(1, 4)])
    expected_x = native["kappa"] * native["e"] * profile
    expected_theta = (
        -native["kappa"]
        * native["w"]
        * profile
        / (native["beta"] * sp.pi * native["c"])
    )
    _same_matrix(actual, expected_x.col_join(expected_theta))
    assert actual[0].is_positive is True
    assert actual[5].is_negative is True
    # The blind equilibrium still has hidden capacity motion when nu is unequal.
    assert native["capacity_rate"].subs(witness)[0] == -native["kappa"]


def test_capacity_generator_preserves_mass_and_the_capacity_hull(native):
    kappa, nu = native["kappa"], native["nu"]
    generator = native["capacity_rate"].jacobian(nu)
    _same_matrix(generator * sp.ones(5, 1), sp.zeros(5, 1))
    _same_matrix(sp.ones(1, 5) * generator, sp.zeros(1, 5))
    for i in range(5):
        assert generator[i, i] == -kappa
        for j in range(5):
            if i != j:
                assert generator[i, j].is_nonnegative is True
    # Uniformization identifies a doubly stochastic one-neighbor transition;
    # its positive Poisson mixture establishes the continuous hull property.
    transition = sp.eye(5) + generator / kappa
    _same_matrix(transition * sp.ones(5, 1), sp.ones(5, 1))
    _same_matrix(sp.ones(1, 5) * transition, sp.ones(1, 5))
    assert all(entry.is_nonnegative is True for entry in transition)
    minimum, maximum = sp.symbols("minimum maximum", positive=True)
    gaps = sp.symbols("gap0:5", nonnegative=True)
    for target in range(5):
        at_minimum = {
            nu[i]: minimum + (gaps[i] if i != target else 0) for i in range(5)
        }
        at_maximum = {
            nu[i]: maximum - (gaps[i] if i != target else 0) for i in range(5)
        }
        assert (
            sp.expand(native["capacity_rate"][target].subs(at_minimum)).is_nonnegative
            is True
        )
        assert (
            sp.expand(native["capacity_rate"][target].subs(at_maximum)).is_nonpositive
            is True
        )
    variance_rate = (nu.T * native["capacity_rate"])[0]
    expected = -kappa * sum((nu[a] - nu[b]) ** 2 for a, b in native["edges"]) / 2
    assert _zero(variance_rate - expected)
