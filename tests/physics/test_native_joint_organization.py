"""Exact joint recovery and transmission controls on the supplied joined C5.

All thirty coordinates belong to the declared mathematical law. The production
relational executor still holds capacities. These controls differentiate the
native local field and its initial jets; they invoke no runtime, solver,
trajectory, event or numerical spectral fit. Local nonlinear continuation and
the sufficiently small preparation domain remain obligations of the proof.
"""

import pytest
import sympy as sp


def _clean(expression):
    return sp.simplify(sp.expand(expression))


def _same(left, right):
    assert left.shape == right.shape
    assert all(_clean(a - b) == 0 for a, b in zip(left, right))


def _build_native(*, joined):
    """Construct the complete local field, before taking any derivatives."""
    cycles = tuple(
        (offset + i, offset + (i + 1) % 5) for offset in (0, 5) for i in range(5)
    )
    edges = cycles + (((0, 5),) if joined else ())
    neighbors = [[] for _ in range(10)]
    incidence = sp.zeros(10, len(edges))
    c, s = sp.cos(2 * sp.pi / 5), sp.sin(2 * sp.pi / 5)
    phase_edges = []
    for column, (left, right) in enumerate(edges):
        incidence[left, column], incidence[right, column] = 1, -1
        cosine, sine = (1, 0) if (left, right) == (0, 5) else (c, s)
        neighbors[left].append((right, cosine, sine))
        neighbors[right].append((left, cosine, -sine))
        phase_edges.append((left, right, cosine, sine))
    laplacian = incidence * incidence.T
    degree = sp.diag(*(len(row) for row in neighbors))
    form = sp.Matrix(sp.symbols("x0:10", real=True))
    delta = sp.Matrix(sp.symbols("delta0:10", real=True))
    capacity = sp.Matrix(sp.symbols("nu0:10", positive=True))
    kappa, mean = sp.symbols("kappa nu_bar", positive=True)
    origin = sp.symbols("origin", real=True)
    real, imaginary = [], []
    for i, row in enumerate(neighbors):
        real.append(
            sum(
                cosine * sp.cos(delta[j] - delta[i])
                - sine * sp.sin(delta[j] - delta[i])
                for j, cosine, sine in row
            )
        )
        imaginary.append(
            sum(
                sine * sp.cos(delta[j] - delta[i])
                + cosine * sp.sin(delta[j] - delta[i])
                for j, cosine, sine in row
            )
        )
    angles = tuple(sp.atan(im / re) for re, im in zip(real, imaginary))
    source = sp.Matrix([angle / sp.pi for angle in angles])
    metric = sp.Matrix(
        [
            sp.pi * sp.sqrt(re**2 + im**2) * sp.sinc(angle)
            for re, im, angle in zip(real, imaginary, angles)
        ]
    )
    gradient = laplacian * form
    pressure = -degree.inv() * gradient / 2 + source / 2
    form_rate = sp.diag(*capacity) * pressure
    phase_rate = sp.Matrix(
        [capacity[i] * gradient[i] / (2 * metric[i]) for i in range(10)]
    )
    capacity_rate = -kappa * laplacian * capacity / 2
    phase_storage = sum(
        1
        - cosine * sp.cos(delta[right] - delta[left])
        + sine * sp.sin(delta[right] - delta[left])
        for left, right, cosine, sine in phase_edges
    )
    state = form.col_join(delta).col_join(capacity)
    return {
        "edges": edges,
        "incidence": incidence,
        "B": laplacian,
        "D": degree,
        "c": c,
        "x": form,
        "delta": delta,
        "nu": capacity,
        "kappa": kappa,
        "mean": mean,
        "real": real,
        "imaginary": imaginary,
        "source": source,
        "metric": metric,
        "pressure": pressure,
        "phase_storage": phase_storage,
        "field": form_rate.col_join(phase_rate).col_join(capacity_rate),
        "state": state,
        "uniform_form": dict.fromkeys(form, origin),
        "winding": dict.fromkeys(delta, 0),
        "uniform_capacity": dict.fromkeys(capacity, mean),
    }


@pytest.fixture(scope="module")
def native():
    law = _build_native(joined=True)
    law["derivative"] = law["field"].jacobian(law["state"])
    # Uniform form annihilates the q*D(H^-1) terms before the removable sinc
    # derivative at zero is substituted. No tangent block is supplied here.
    law["tangent"] = (
        law["derivative"]
        .subs(law["uniform_form"])
        .subs(law["winding"])
        .subs(law["uniform_capacity"])
        .applyfunc(sp.simplify)
    )
    law["H"] = sp.diag(*law["metric"].subs(law["winding"]).applyfunc(sp.simplify))
    law["K"] = sp.hessian(law["phase_storage"], law["delta"]).subs(law["winding"])
    return law


@pytest.fixture(scope="module")
def preparation(native):
    law = native
    amplitude, eta = sp.symbols("amplitude eta", positive=True)
    form_values = {law["x"][i]: amplitude if i == 1 else 0 for i in range(10)}
    capacity_values = {
        law["nu"][i]: law["mean"] + (eta if i == 0 else -eta if i == 1 else 0)
        for i in range(10)
    }

    def at_source(value):
        return (
            value.subs(form_values)
            .subs(law["winding"])
            .subs(capacity_values)
            .applyfunc(sp.simplify)
        )

    receiver_rows = tuple(range(5, 10)) + tuple(range(15, 20))
    native_field = law["field"][:20, 0]
    receiver_derivative = law["derivative"].extract(receiver_rows, range(30))
    receiver_y = receiver_derivative[:, :20]
    capacity_rate = law["field"][20:30, 0]
    mixed = sp.zeros(10, 20)
    for j in range(10):
        mixed += receiver_y.diff(law["nu"][j]) * capacity_rate[j]
    source_field = at_source(law["field"])
    source_capacity_columns = at_source(native_field.jacobian(law["nu"]))
    source_receiver_derivative = at_source(receiver_derivative)
    second = source_receiver_derivative * source_field
    held_direction = source_field[:20, 0].col_join(sp.zeros(10, 1))
    held_second = source_receiver_derivative * held_direction
    source_capacity_rate = source_field[20:30, 0]
    capacity_acceleration = capacity_rate.jacobian(law["nu"]) * source_capacity_rate
    term_one = (
        source_receiver_derivative[:, :20]
        * source_capacity_columns
        * source_capacity_rate
    )
    term_two = 2 * at_source(mixed) * source_field[:20, 0]
    term_three = (
        source_capacity_columns.extract(receiver_rows, range(10))
        * capacity_acceleration
    )
    return {
        "amplitude": amplitude,
        "eta": eta,
        "at_source": at_source,
        "field": source_field,
        "capacity_values": capacity_values,
        "receiver_rows": receiver_rows,
        "receiver_derivative": source_receiver_derivative,
        "second": second.applyfunc(_clean),
        "held_second": held_second.applyfunc(_clean),
        "third_terms": tuple(
            value.applyfunc(_clean) for value in (term_one, term_two, term_three)
        ),
        "third_difference": (term_one + term_two + term_three).applyfunc(_clean),
    }


def test_full_thirty_state_tangent_from_native_resultants(native):
    law = native
    laplacian, degree, metric, stiffness = (law[key] for key in ("B", "D", "H", "K"))
    mean, kappa = law["mean"], law["kappa"]
    assert law["tangent"].shape == (30, 30)
    h = (1 + sp.sqrt(5)) / 2
    for i in range(10):
        expected = h if i in (0, 5) else h - 1
        assert _clean(law["real"][i].subs(law["winding"]) - expected) == 0
        assert _clean(law["imaginary"][i].subs(law["winding"])) == 0
        assert expected.is_positive is True
        assert _clean(metric[i, i] - sp.pi * expected) == 0
    expected_tangent = sp.zeros(30)
    expected_tangent[:10, :10] = -mean * degree.inv() * laplacian / 2
    expected_tangent[:10, 10:20] = -mean * metric.inv() * stiffness / 2
    expected_tangent[10:20, :10] = mean * metric.inv() * laplacian / 2
    expected_tangent[20:30, 20:30] = -kappa * laplacian / 2
    _same(law["tangent"], expected_tangent)
    _same(law["tangent"][:20, 20:30], sp.zeros(20, 10))
    _same(law["tangent"][20:30, :20], sp.zeros(10, 20))
    equilibrium = (
        law["field"]
        .subs(law["uniform_form"])
        .subs(law["winding"])
        .subs(law["uniform_capacity"])
    )
    _same(equilibrium, sp.zeros(30, 1))
    _same(
        law["source"].jacobian(law["delta"]).subs(law["winding"]),
        -metric.inv() * stiffness,
    )


def test_quotient_stability_structure_and_origin_drift(native):
    law = native
    laplacian, stiffness, degree = (law[key] for key in ("B", "K", "D"))
    c, mean, kappa = law["c"], law["mean"], law["kappa"]
    assert c.is_positive is True
    _same(
        stiffness,
        law["incidence"] * sp.diag(*([c] * 10 + [1])) * law["incidence"].T,
    )
    assert laplacian[:9, :9].det() == 25
    assert _clean(stiffness[:9, :9].det(method="domain-ge") - 25 * c**8) == 0
    # Both positive edge-Gram forms have exactly one common-origin kernel.
    _same(laplacian * sp.ones(10, 1), sp.zeros(10, 1))
    _same(stiffness * sp.ones(10, 1), sp.zeros(10, 1))
    native_tangent = law["tangent"][:20, :20]
    native_energy = sp.diag(laplacian, stiffness)
    expected_loss = sp.zeros(20)
    expected_loss[:10, :10] = -mean * laplacian * degree.inv() * laplacian
    _same(
        native_tangent.T * native_energy + native_energy * native_tangent, expected_loss
    )
    output = laplacian.row_join(sp.zeros(10))
    observed_derivative = output * native_tangent
    # Normalize only nonzero known scalars; this minor stays in Q(sqrt(5)).
    witness = (observed_derivative[:, 10:20] * (-2 * sp.pi / mean)).applyfunc(_clean)
    minor = witness[:9, :9].det(method="domain-ge")
    h = 1 + 2 * c
    expected_minor = 625 * (1 + 10 * c) / (128 * h**2)
    assert _clean(minor - expected_minor) == 0
    assert (1 + 10 * c).is_positive is True and h.is_positive is True
    # [output; output*A] has an invertible 18-column triangular minor: B's
    # first nine columns and the nine phase columns just checked. Its kernel
    # is exactly the two common origins, the energy-observability criterion.
    origins = sp.diag(sp.ones(10, 1), sp.ones(10, 1))
    _same(output.col_join(observed_derivative) * origins, sp.zeros(20, 2))
    relative = sp.eye(10)[:, :9]
    relative[9, :] = -sp.ones(1, 9)
    projection = (relative.T * relative).inv() * relative.T
    embed = sp.diag(relative, relative, relative)
    project = sp.diag(projection, projection, projection)
    quotient = project * law["tangent"] * embed
    assert quotient.shape == (27, 27)
    full_origins = sp.diag(sp.ones(10, 1), sp.ones(10, 1), sp.ones(10, 1))
    _same(law["tangent"] * full_origins, sp.zeros(30, 3))
    _same(project * embed, sp.eye(27))
    drift = law["tangent"] * embed - embed * quotient
    _same(project * drift, sp.zeros(27))
    _same(
        law["tangent"][20:30, 20:30].T + law["tangent"][20:30, 20:30],
        -kappa * laplacian,
    )
    # Irregular degrees/metrics produce actual quotient-to-origin coupling.
    form_mean_row = sp.ones(1, 10) * law["tangent"][:10, :10] / 10
    phase_mean_row = sp.ones(1, 10) * law["tangent"][10:20, :10] / 10
    assert _clean(form_mean_row[0] - mean / 60) == 0
    assert _clean(phase_mean_row[0] + mean / (20 * sp.pi * c * h)) == 0
    _same(sp.ones(1, 10) * law["tangent"][20:30, :], sp.zeros(1, 30))
    # The proof uses these positive quotient forms and observability to infer
    # Hurwitz stability, then smoothness for nonlinear continuation. There is
    # no fitted decay rate, explicit basin or numerical eigenvalue test here.


def test_mean_preserving_capacity_source(native, preparation):
    law, source = native, preparation
    mean, eta, kappa = law["mean"], source["eta"], law["kappa"]
    initial_capacity = sp.Matrix(
        [source["capacity_values"][value] for value in law["nu"]]
    )
    assert _clean(sum(initial_capacity) - 10 * mean) == 0
    odds = sp.symbols("positive_capacity_odds", positive=True)
    admitted = initial_capacity.subs(eta, mean * odds / (1 + odds)).applyfunc(_clean)
    assert all(value.is_positive is True for value in admitted)
    expected = sp.zeros(10, 1)
    expected[0], expected[1] = -2 * kappa * eta, 3 * kappa * eta / 2
    expected[2], expected[4], expected[5] = (
        -kappa * eta / 2,
        kappa * eta / 2,
        kappa * eta / 2,
    )
    _same(source["field"][20:30, 0], expected)
    assert _clean(sum(expected)) == 0
    _same(source["field"][20:30, 0].subs(eta, 0), sp.zeros(10, 1))
    # The receiver gets capacity at first order, before its native deformation.
    assert expected[5].is_positive is True


def test_receiver_second_jet_under_joint_law(native, preparation):
    law, source = native, preparation
    amplitude, eta, mean = source["amplitude"], source["eta"], law["mean"]
    h = (1 + sp.sqrt(5)) / 2
    form_factor = sp.Rational(1, 36) - 1 / (4 * sp.pi**2 * h**2)
    phase_factor = 1 / (12 * sp.pi * h)
    expected = sp.zeros(10, 1)
    scale = amplitude * mean * (mean + eta)
    expected[0], expected[5] = scale * form_factor, -scale * phase_factor
    _same(source["field"].extract(source["receiver_rows"], [0]), sp.zeros(10, 1))
    _same(source["second"], expected)
    _same(source["held_second"], expected)
    assert sp.diff(source["second"][0], law["kappa"]) == 0
    assert sp.diff(source["second"][5], law["kappa"]) == 0
    # h>1 and pi>3 prove the form coefficient positive without a float cutoff.
    assert (h - 1).is_positive is True and (sp.pi - 3).is_positive is True
    assert _clean(form_factor - (sp.pi**2 * h**2 - 9) / (36 * sp.pi**2 * h**2)) == 0
    receiver_observation = sp.zeros(2, 10)
    receiver_observation[0, 0], receiver_observation[1, 5] = 1, 1
    for i in range(1, 5):
        receiver_observation[0, i] = -sp.Rational(1, 4)
        receiver_observation[1, i + 5] = -sp.Rational(1, 4)
    _same(
        receiver_observation * source["second"], sp.Matrix([expected[0], expected[5]])
    )
    # These are derivatives of deformation relative to the prepared shape.
    # The shared second jet proves transmission, not capacity-law separation.


def test_receiver_third_jet_distinguishes_capacity_release(native, preparation):
    law, source = native, preparation
    amplitude, eta, mean, kappa = (
        source["amplitude"],
        source["eta"],
        law["mean"],
        law["kappa"],
    )
    h = (1 + sp.sqrt(5)) / 2
    form_factor = sp.Rational(1, 36) - 1 / (4 * sp.pi**2 * h**2)
    phase_factor = 1 / (12 * sp.pi * h)
    expected = sp.zeros(10, 1)
    scale = kappa * eta * amplitude * (eta - mean)
    expected[0], expected[5] = scale * form_factor, -scale * phase_factor
    _same(source["third_difference"], expected)
    _same(source["third_terms"][2], sp.zeros(10, 1))
    # F is linear in capacities. Its receiver third-jet difference is exactly
    # F_y(F_nu*r)+2F_{y,nu}[F,r]+F_nu*(r_nu*r). The first two terms retain the
    # source's actual receiver mixed partials; no donor H derivative is
    # discarded globally. The last term vanishes because receiver p=q=0.
    expected_first, expected_second = sp.zeros(10, 1), sp.zeros(10, 1)
    expected_first[0], expected_first[5] = (
        -2 * kappa * eta * amplitude * mean * form_factor,
        2 * kappa * eta * amplitude * mean * phase_factor,
    )
    expected_second[0], expected_second[5] = (
        kappa * eta * amplitude * (mean + eta) * form_factor,
        -kappa * eta * amplitude * (mean + eta) * phase_factor,
    )
    _same(source["third_terms"][0], expected_first)
    _same(source["third_terms"][1], expected_second)
    odds = sp.symbols("positive_capacity_odds", positive=True)
    admitted_scale = _clean(scale.subs(eta, mean * odds / (1 + odds)))
    assert admitted_scale.is_negative is True
    assert _clean(source["third_difference"][5] / (-phase_factor) - scale) == 0
    # Smoothness makes the leading difference cubic (coefficient / 6).
    # These jets supply no numerical horizon or validated remainder band.


def test_blind_preparations_and_disconnected_receiver(native, preparation):
    source = preparation
    _same(source["second"].subs(source["amplitude"], 0), sp.zeros(10, 1))
    for parameter in (source["amplitude"], source["eta"], native["kappa"]):
        _same(source["third_difference"].subs(parameter, 0), sp.zeros(10, 1))
    # Uniform capacity suppresses release, but does not suppress the inherited
    # held-law transmission from a nonzero form perturbation.
    h = (1 + sp.sqrt(5)) / 2
    assert (
        _clean(
            source["second"][5].subs(source["eta"], 0)
            + source["amplitude"] * native["mean"] ** 2 / (12 * sp.pi * h)
        )
        == 0
    )
    separate = _build_native(joined=False)
    _same(separate["D"], 2 * sp.eye(10))
    receiver = separate["field"].extract(
        tuple(range(5, 10)) + tuple(range(15, 20)), [0]
    )
    donor_state = separate["state"].extract(
        tuple(range(5)) + tuple(range(10, 15)) + tuple(range(20, 25)), [0]
    )
    _same(receiver.jacobian(donor_state), sp.zeros(10, 15))
    receiver_lock = {
        **{separate["x"][i]: 0 for i in range(5, 10)},
        **{separate["delta"][i]: 0 for i in range(5, 10)},
    }
    _same(receiver.subs(receiver_lock), sp.zeros(10, 1))
    # The independently rebuilt disconnected receiver is stationary for any
    # admitted donor state. Removing a bridge must also recompute its metric.
    for i in range(5, 10):
        assert (
            _clean(
                separate["metric"][i].subs(separate["winding"])
                - 2 * sp.pi * separate["c"]
            )
            == 0
        )
