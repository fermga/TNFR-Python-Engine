"""All spatial modes of the conditional fresh-pressure phase comparison.

These exact derivatives bind the graph and phase owners. Local nonlinear
stability follows analytically from the smooth quotient law and its Hurwitz
Jacobian; tests do not estimate a basin or certify the independent-pressure
runtime. No new evolution law or coefficients are selected here.
"""

import pytest

from tests.physics._internal_mode_fixture import _exact_generator, _phase_support
from tnfr.physics.phase_response import derive_phase_response
from tnfr.physics.support_transport import observe_support_transport

s = pytest.importorskip("sympy")


@pytest.fixture(scope="module")
def prism_laplacian():
    graph, rows = _phase_support()
    laplacian = -s.Matrix(_exact_generator(observe_support_transport(graph)))
    reference = derive_phase_response(
        cosine_gram=((1,) * 6,) * 6,
        mean_neighbors=rows,
        receiver_sources=tuple((i,) for i in range(6)),
        phase_factor=1,
    )
    assert s.Matrix(reference.mean_response) - s.eye(6) == -laplacian
    return laplacian, rows


def test_all_six_actual_support_modes_bind_both_phase_linearizations(prism_laplacian):
    laplacian, rows = prism_laplacian
    assert laplacian == laplacian.T
    assert laplacian.eigenvals() == {
        0: 1,
        s.Rational(2, 3): 1,
        1: 2,
        s.Rational(5, 3): 2,
    }
    angles = s.Matrix(s.symbols("t0:6", real=True))
    current = s.Matrix(
        [
            sum(s.sin(angles[j] - angles[i]) for j in row) / len(row)
            for i, row in enumerate(rows)
        ]
    )
    at_consensus = dict.fromkeys(angles, 0)
    assert current.jacobian(angles).subs(at_consensus) == -laplacian
    # The phase-source derivative uses the argument of a neighbor sum; the
    # current uses a mean of pairwise sines. They agree only to this order.
    assert laplacian * s.ones(6, 1) == s.zeros(6, 1)
    assert laplacian.rank() == 5


def test_every_nonuniform_mode_has_two_real_strictly_negative_rates(prism_laplacian):
    laplacian, _ = prism_laplacian
    e, k, sigma, gain, lam, rate = s.symbols(
        "e k sigma gain lambda rate", positive=True
    )
    block = s.Matrix([[-e, -k], [-sigma * e, -sigma * k - gain]])
    total = e + sigma * k + gain
    discriminant = s.expand(total**2 - 4 * e * gain)
    positive_form = (e - gain) ** 2 + 2 * sigma * k * (e + gain) + (sigma * k) ** 2
    assert s.expand(discriminant - positive_form) == 0
    assert block.trace() == -total
    assert s.expand(block.det()) == e * gain
    assert s.expand((rate * s.eye(2) - lam * block).det()) == s.expand(
        rate**2 + lam * total * rate + lam**2 * e * gain
    )
    # Positive determinant, negative trace and positive discriminant give two
    # distinct real negative roots for every lambda>0, for ALL positive gains.
    full = s.kronecker_product(block, laplacian)
    for value, _, vectors in laplacian.eigenvects():
        for vector in vectors:
            embedding = s.diag(vector, vector)
            assert full * embedding == embedding * (value * block)
    assert full.rank() == 10
    assert full * s.ones(6, 1).col_join(s.zeros(6, 1)) == s.zeros(12, 1)
    assert full * s.zeros(6, 1).col_join(s.ones(6, 1)) == s.zeros(12, 1)
    # The two zero directions are the independent absolute EPI/phase means.
    # Cross-triangle modes lambda=2/3 and 5/3 introduce no oscillatory seed.


def test_complete_quotient_retains_cubic_mean_drift(prism_laplacian):
    laplacian, rows = prism_laplacian
    t = s.symbols("t", real=True)
    e, w, amplitude, linear, gain = s.symbols("e w A B gain", positive=True)
    sigma = s.pi * amplitude + linear
    # Deliberately nonrepeated centered preparation, outside the old lift.
    form = s.Matrix([1, -1, 0, 0, 0, 0])
    phase = s.Matrix([0, 0, 0, 1, 1, -2])
    mean = s.ones(1, 6) / 6
    center = s.eye(6) - s.ones(6, 6) / 6
    source_first = -laplacian * phase / s.pi
    source_third = []
    current_third = []
    for i, row in enumerate(rows):
        m1, m2, m3 = (
            sum(phase[j] ** power for j in row) / len(row) for power in (1, 2, 3)
        )
        # Imaginary log of the neighbor phasor sum, divided by pi.
        source_third.append((-m3 + 3 * m1 * m2 - 2 * m1**3) / (6 * s.pi))
        phasor_jet = 1 + s.I * m1 * t - m2 * t**2 / 2 - s.I * m3 * t**3 / 6
        argument_jet = s.im(s.series(s.log(phasor_jet), t, 0, 4).removeO().expand())
        assert s.simplify(argument_jet.coeff(t, 3) / s.pi - source_third[-1]) == 0
        current_third.append(-sum((phase[j] - phase[i]) ** 3 for j in row) / 18)
    pressure_first = -e * laplacian * form + w * source_first
    pressure_third = w * s.Matrix(source_third)
    phase_first = sigma * pressure_first - gain * laplacian * phase
    phase_third = (
        sigma * pressure_third
        - amplitude * s.pi**3 * pressure_first.applyfunc(lambda value: value**3) / 6
        + gain * s.Matrix(current_third)
    )
    assert s.simplify(mean * pressure_first) == s.zeros(1, 1)
    assert s.simplify(mean * phase_first) == s.zeros(1, 1)
    assert s.simplify(center * pressure_first - pressure_first) == s.zeros(6, 1)
    assert s.simplify(center * phase_first - phase_first) == s.zeros(6, 1)
    # A uniform shift of form or primitive phase changes neither first row.
    assert laplacian * s.ones(6, 1) == s.zeros(6, 1)
    # Nonlinear translation symmetry closes the quotient but does not freeze
    # its means. This exact cubic term is lost by projecting them away forever.
    assert s.simplify((mean * pressure_third)[0]) == w / (18 * s.pi)
    assert (mean * (t * pressure_first + t**3 * pressure_third))[0] != 0
    assert s.expand((mean * phase_third)[0]) != 0
    # Smooth autonomous quotient + Hurwitz linearization gives a sufficiently
    # small attracting neighborhood. Cubic mean rates are integrable there;
    # the limiting absolute means need not equal their initial values.
