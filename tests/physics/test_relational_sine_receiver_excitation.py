"""Full-field tangent gain and nonlinear mixing controls, without trajectories.

Formal coefficients below differentiate the actual eleven-node sine rows.
They are local algebra, not numerical forecasts or receiver-passage evidence.
"""

from dataclasses import replace
from fractions import Fraction as Q
from itertools import combinations

import pytest
from mpmath import mp

from tests.physics.test_relational_sine_formation import _explicit, _report, _support
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.sdk import export_to_json
from tnfr.utils.io import json_loads


@pytest.fixture(scope="module")
def algebra():
    s = pytest.importorskip("sympy")
    edges, neighbors = _support()
    incidence = s.zeros(12, 11)
    for row, (i, j) in enumerate(edges):
        incidence[row, i], incidence[row, j] = -1, 1
    capacities = [
        s.Rational(3, 2) if i == 6 else s.Rational(1, 2) if i == 9 else 1
        for i in range(11)
    ]
    k = s.diag(*(nu / s.Integer(len(row)) for nu, row in zip(capacities, neighbors)))
    cosine = (s.sqrt(5) - 1) / 4
    hessian = incidence.T * s.diag(*([cosine] * 5 + [1] * 7)) * incidence
    permutation = list(range(11))
    permutation[1], permutation[4] = 4, 1
    permutation[2], permutation[3] = 3, 2
    reflection = s.zeros(11)
    for i, j in enumerate(permutation):
        reflection[i, j] = 1
    return s, edges, neighbors, incidence, k, hessian, reflection


def test_tangent_storage_and_silent_projection_use_noncommuting_full_matrices(algebra):
    s, _, _, incidence, k, hessian, reflection = algebra
    laplacian = incidence.T * incidence
    x = s.Matrix(s.symbols("x0:11", real=True))
    eta = s.Matrix(s.symbols("eta0:11", real=True))
    a, e = s.symbols("a e", positive=True)
    q = laplacian * x
    xdot = -e * k * q - a * k * hessian * eta
    phasedot = a * k * q
    form_rate = q.dot(xdot)
    phase_rate = (hessian * eta).dot(phasedot)
    loss = e * q.dot(k * q)
    assert s.expand(form_rate + phase_rate + loss) == 0

    # Scalar simultaneous modes would lose actual mobility/target geometry.
    commutator = laplacian * k * hessian - hessian * k * laplacian
    assert any(s.simplify(entry) != 0 for entry in commutator)
    for operator in (laplacian, k, hessian):
        assert operator * reflection == reflection * operator
    odd = (s.eye(11) - reflection) / 2
    even = (s.eye(11) + reflection) / 2
    assert even * laplacian * odd == s.zeros(11)
    assert even * laplacian * k * laplacian * odd == s.zeros(11)
    for node in range(5, 11):
        assert odd[node, :] == s.zeros(1, 11)
    # Commutation propagates the odd invariant subspace for every tangent
    # time. Zero receiver rows are exact identities, not small eigenvectors.
    joint_reflection = s.diag(reflection, reflection)
    generator = (
        (-e * k * laplacian)
        .row_join(-a * k * hessian)
        .col_join((a * k * laplacian).row_join(s.zeros(11)))
    )
    assert generator * joint_reflection == joint_reflection * generator

    # The phase quadratic dominates the receiver-only quadratic, with the
    # actual positive donor cosine and both bridges retained.
    receiver_incidence = incidence[5:10, :]
    remainder = hessian - receiver_incidence.T * receiver_incidence
    donor = incidence[:5, :]
    bridges = incidence[10:, :]
    assert remainder == (s.sqrt(5) - 1) * donor.T * donor / 4 + bridges.T * bridges


def test_energy_angle_comparison_has_uniform_strict_gain_without_modal_decoupling():
    s = pytest.importorskip("sympy")
    phi = s.Symbol("phi", real=True)
    comparison = s.Rational(2, 9) * (phi + s.sin(phi) * s.cos(phi))
    assert s.simplify(s.diff(comparison, phi) - 4 * s.cos(phi) ** 2 / 9) == 0
    assert comparison.subs(phi, s.pi / 2) == s.pi / 9
    # lambda_min(B)>=1/44, lambda_max(C)<=lambda_max(B)<=3.
    # pi>3 and sqrt(132)<12 imply 2h/k<4 on the positive stratum.
    assert 3 * 44 < 12**2
    crossing = s.Symbol("crossing", nonnegative=True)
    log_derivative_bound = -1 + s.Rational(2, 9) * (4 + crossing)
    assert (
        s.expand(
            log_derivative_bound - s.Rational(2, 9) * (crossing - s.Rational(1, 2))
        )
        == 0
    )
    # crossing=sin(phi)cos(phi)<=1/2, independently of the exchange sign.
    f, y = s.symbols("f y", nonnegative=True)
    assert s.expand((f**2 + y**2) - 2 * f * y - (f - y) ** 2) == 0
    derivative = 2 * s.cos(phi) / s.sin(phi) - s.diff(comparison, phi)
    positive_factor = (
        2 * s.cos(phi) * (9 - 2 * s.sin(phi) * s.cos(phi)) / (9 * s.sin(phi))
    )
    assert s.trigsimp(s.expand(derivative - positive_factor)) == 0
    # exp(pi/9)>1+pi/9>4/3 gives exp(-pi/9)<3/4. The remaining
    # gap to the receiver barrier is finite even at the full F=4 budget.
    assert 1 + Q(3, 9) == Q(4, 3)
    assert Q(3, 4) * 4 == 3 < Q(7, 2)
    assert s.sqrt(7) - s.sqrt(6) > 0


def test_full_nonlinear_current_bound_is_an_actual_positive_square_identity(algebra):
    s, edges, neighbors, incidence, k, _, _ = algebra
    sine = s.Matrix(s.symbols("edge_sine0:12", real=True))
    cosine = s.symbols("edge_cosine0:12", real=True)
    current = -incidence.T * sine
    capacities = [k[node, node] * len(neighbors[node]) for node in range(11)]
    variance = 0
    for node in range(11):
        local = [
            -incidence[edge, node] * sine[edge]
            for edge in range(12)
            if incidence[edge, node] != 0
        ]
        variance += k[node, node] * sum(
            (first - second) ** 2 for first, second in combinations(local, 2)
        )
    deficits = [3 - capacities[i] - capacities[j] for i, j in edges]
    assert all(value >= 0 for value in deficits)
    potential = sum(1 - value for value in cosine)
    squares = variance + sum(
        value * sine[edge] ** 2 for edge, value in enumerate(deficits)
    )
    squares += 3 * sum((1 - value) ** 2 for value in cosine)
    unit_circle_residual = 3 * sum(
        1 - sine[edge] ** 2 - cosine[edge] ** 2 for edge in range(12)
    )
    assert (
        s.expand(
            6 * potential - current.dot(k * current) - squares - unit_circle_residual
        )
        == 0
    )
    # The unit-circle residual is exactly zero for actual phase differences;
    # all remaining terms are nonnegative, without an acute-sector premise.
    x = s.Matrix(s.symbols("nonlinear_x0:11", real=True))
    q = incidence.T * incidence * x
    a, e = s.symbols("a e", positive=True)
    xdot, phasedot = k * (-e * q + a * current), a * k * q
    assert s.expand(q.dot(xdot) - current.dot(phasedot) + e * q.dot(k * q)) == 0


def test_absolute_phase_ceiling_is_monotone_in_initial_form_budget_and_exactly_strict():
    s = pytest.importorskip("sympy")
    form, phase = s.symbols("form phase", positive=True)
    total = form + phase
    angle = s.atan(s.sqrt(phase / form))
    log_ceiling = (
        s.log(total)
        + s.Rational(2, 9) * (angle + s.sqrt(form * phase) / total)
        - s.pi / 9
    )
    derivative = 1 / total - s.Rational(2, 9) * s.sqrt(form * phase) / total**2
    assert s.simplify(s.diff(log_ceiling, form) - derivative) == 0
    assert s.simplify(s.limit(log_ceiling, form, 0) - s.log(phase)) == 0
    positive_numerator = 8 * total + (s.sqrt(form) - s.sqrt(phase)) ** 2
    assert s.simplify(9 * total**2 * derivative - positive_numerator) == 0

    # Independently check the elementary outward certificate at F=4.
    # No tangent or nonlinear trajectory is evaluated to choose this ceiling.
    assert Q(559, 250) ** 2 < 5 < Q(56, 25) ** 2
    phase_upper = (25 - 5 * Q(559, 250)) / 4
    phase_lower = (25 - 5 * Q(56, 25)) / 4
    assert phase_upper == Q(691, 200) and phase_lower == Q(69, 20)
    assert phase_upper / 4 < Q(93, 100) ** 2
    pi = pi_interval()
    assert pi.lo > Q(157, 50) and pi.hi < Q(22, 7)
    ratio = Q(7, 193)
    angle_upper = Q(11, 14) - ratio + ratio**3 / 3
    assert angle_upper == Q(226292477, 301940394) < Q(3, 4)
    comparison_upper = Q(2, 9) * (Q(3, 4) + Q(1, 2))
    exponent_lower = Q(157, 50) / 9 - comparison_upper
    assert exponent_lower == Q(16, 225)
    exp_lower = 1 + exponent_lower + exponent_lower**2 / 2
    ceiling_upper = (4 + phase_upper) / exp_lower
    assert ceiling_upper == Q(3019275, 434824) < Q(139, 20)
    assert Q(139, 20) - Q(7, 2) == Q(69, 20)
    assert Q(139, 20) < 7
    # At a receiver barrier crossing, nonnegative bridge potential leaves
    # donor potential strictly below V5. Staying inside the donor's original
    # well would require at least V5, so a donor exit must precede that time.
    assert phase_lower + Q(7, 2) == Q(139, 20)


def _formal_coefficients(algebra, sign):
    """Independent coefficient recurrence for x, theta, edge sin and cos.

    Coefficients are derivatives divided by factorial. Differentiating
    sin(gap) and cos(gap) avoids any numerical trajectory or target rounding.
    """
    s, edges, neighbors, _, k, _, _ = algebra
    a, e, sine, cosine = s.symbols("a e sine cosine", real=True)
    source = (0, sign, 0, 0, -sign, 0, 0, 0, 0, 0, 1)
    x = [[s.Integer(value)] for value in source]
    phase = [[s.Integer(0)] for _ in source]
    currents = [[sine if row < 5 else s.Integer(0)] for row in range(12)]
    cosines = [[cosine if row < 5 else s.Integer(1)] for row in range(12)]
    # _support orients each donor cycle edge forward, including 4->0.
    # All five exact target gaps are alpha modulo2pi.
    for order in range(5):
        q = [
            sum(x[i][order] - x[j][order] for j in row)
            for i, row in enumerate(neighbors)
        ]
        current = [s.Integer(0)] * 11
        for edge, (i, j) in enumerate(edges):
            current[i] += currents[edge][order]
            current[j] -= currents[edge][order]
        for node in range(11):
            x[node].append(
                s.expand(
                    k[node, node] * (-e * q[node] + a * current[node]) / (order + 1)
                )
            )
            phase[node].append(s.expand(a * k[node, node] * q[node] / (order + 1)))
        for edge, (i, j) in enumerate(edges):
            sine_next = cosine_next = s.Integer(0)
            for left in range(order + 1):
                derivative = (order - left + 1) * (
                    phase[j][order - left + 1] - phase[i][order - left + 1]
                )
                sine_next += cosines[edge][left] * derivative
                cosine_next -= currents[edge][left] * derivative
            currents[edge].append(s.expand(sine_next / (order + 1)))
            cosines[edge].append(s.expand(cosine_next / (order + 1)))
    return a, e, sine, cosine, x, phase


def _multiply_series(first, second, degree):
    return [
        sum(first[j] * second[n - j] for j in range(n + 1)) for n in range(degree + 1)
    ]


def test_equal_quadratic_sources_have_different_exact_receiver_sixth_derivative(
    algebra,
):
    s, edges, _, incidence, k, _, reflection = algebra
    laplacian = incidence.T * incidence
    preparations = [
        s.Matrix((0, sign, 0, 0, -sign, 0, 0, 0, 0, 0, 1)) for sign in (1, -1)
    ]
    first, second = preparations
    assert (s.eye(11) + reflection) * first == (s.eye(11) + reflection) * second
    for matrix in (
        laplacian / 2,
        laplacian * k * laplacian,
        laplacian * k * laplacian * k * laplacian,
        laplacian * k * laplacian * k * laplacian * k * laplacian,
    ):
        assert first.dot(matrix * first) == second.dot(matrix * second)
    assert first.dot(laplacian * first) / 2 == 3
    assert (laplacian * first).dot(k * laplacian * first) == s.Rational(23, 3)

    jets = [_formal_coefficients(algebra, sign) for sign in (1, -1)]
    a, e, sine, _, xp, tp = jets[0]
    xm, tm = jets[1][4:]
    n0 = s.expand(6 * (xp[0][3] - xm[0][3]))
    assert n0 == -8 * a**3 * sine / 9
    assert s.expand(120 * (xp[5][5] - xm[5][5]) - (e**2 - a**2) * n0 / 6) == 0
    assert s.expand(120 * (tp[5][5] - tm[5][5]) + a * e * n0 / 6) == 0
    for node in range(5, 10):
        assert xp[node][:5] == xm[node][:5]
        assert tp[node][:5] == tm[node][:5]

    coefficients = []
    for _, _, _, _, x, phase in jets:
        coefficient = 0
        for i, j in edges[5:10]:
            dx = [x[j][n] - x[i][n] for n in range(6)] + [0]
            gap = [phase[j][n] - phase[i][n] for n in range(6)] + [0]
            square = _multiply_series(gap, gap, 6)
            fourth = _multiply_series(square, square, 6)
            sixth = _multiply_series(fourth, square, 6)
            coefficient += _multiply_series(dx, dx, 6)[6] / 2
            coefficient += square[6] / 2 - fourth[6] / 24 + sixth[6] / 720
        coefficients.append(s.expand(coefficient))
    difference = s.expand(720 * (coefficients[0] - coefficients[1]))
    assert s.expand(difference - 2 * e**3 * n0 / 3) == 0
    assert difference.subs({e: s.Rational(1, 2), a: 1 / (2 * s.pi)}) == -sine / (
        108 * s.pi**3
    )

    loss_coefficients = []
    for _, _, _, _, x, _ in jets:
        q = [
            [
                sum(x[node][n] - x[neighbor][n] for neighbor in algebra[2][node])
                for n in range(5)
            ]
            for node in range(11)
        ]
        loss_coefficients.append(
            s.expand(
                sum(
                    e * k[node, node] * _multiply_series(q[node], q[node], 4)[4]
                    for node in range(5, 10)
                )
            )
        )
    # Four derivatives of the actual loss rate give five derivatives of
    # accumulated loss. Internal-only q would miss the port/intermediary term.
    loss_difference = 24 * (loss_coefficients[0] - loss_coefficients[1])
    assert s.expand(loss_difference - e**2 * n0 / 3) == 0


def test_reader_decomposes_actual_source_and_rederives_full_support_moments(algebra):
    s, _, _, incidence, k, _, reflection = algebra
    source = _explicit(
        donor=(Q(1, 5), Q(-3, 10), Q(1, 7), Q(2, 7), Q(4, 5)),
        hidden=Q(2, 5),
    )
    report = source.receiver_excitation()
    vector = s.Matrix(source.initial_epi)
    even = (vector + reflection * vector) / 2
    odd = (vector - reflection * vector) / 2
    laplacian = incidence.T * incidence
    assert report.source is source
    assert report.even_initial_epi == tuple(even)
    assert report.odd_initial_epi == tuple(odd)
    assert (
        tuple(e + o for e, o in zip(report.even_initial_epi, report.odd_initial_epi))
        == source.initial_epi
    )
    assert report.even_form_storage == even.dot(laplacian * even) / 2
    assert report.odd_form_storage == odd.dot(laplacian * odd) / 2
    assert (
        report.even_form_storage + report.odd_form_storage
        == source.initial_form_storage
    )
    assert report.even_dissipative_norm_squared == (laplacian * even).dot(
        k * laplacian * even
    )
    assert report.odd_dissipative_norm_squared == (laplacian * odd).dot(
        k * laplacian * odd
    )
    assert report.source_coordinates == (
        Q(2, 5),
        Q(-1, 5),
        Q(1, 20),
        Q(1, 70),
        Q(-11, 20),
        Q(-1, 14),
    )
    assert report.bridge_form_storage == Q(1, 10)
    assert (
        report.even_internal_form_storage + report.bridge_form_storage
        == report.even_form_storage
    )
    assert report.odd_internal_form_storage == report.odd_form_storage
    assert report.available and report.tangent_odd_receiver_silence_certified
    assert (
        report.tangent_receiver_phase_cost_upper_bound
        == Q(3, 4) * report.even_form_storage
    )
    assert report.tangent_phase_cost_bound_strict
    assert report.nonlinear_correction_status == "positive_bound"
    # Derived source gradients are not authoritative inputs to this reader.
    stale = replace(source, initial_form_gradient=(Q(0),) * 11).receiver_excitation()
    assert stale.even_dissipative_norm_squared == report.even_dissipative_norm_squared
    assert stale.odd_dissipative_norm_squared == report.odd_dissipative_norm_squared


def test_tangent_bound_retains_odd_mixing_and_exact_zero_source_boundaries():
    controls = [
        _explicit(donor=(0, sign, 0, 0, -sign), hidden=1).receiver_excitation()
        for sign in (1, -1)
    ]
    for report in controls:
        assert report.even_form_storage == 1 and report.odd_form_storage == 2
        assert report.even_dissipative_norm_squared == Q(8, 3)
        assert report.odd_dissipative_norm_squared == 5
        assert report.tangent_receiver_phase_cost_upper_bound == Q(3, 4)
    assert controls[0].odd_initial_epi == tuple(
        -value for value in controls[1].odd_initial_epi
    )
    assert (
        controls[0].necessary_nonlinear_correction_norm_bounds
        == controls[1].necessary_nonlinear_correction_norm_bounds
    )
    # The full nonlinear sixth-derivative difference above prevents this
    # equal tangent ceiling from becoming an equal nonlinear-response claim.
    for source in (_report(0), _explicit(donor=(0, 1, 0, 0, -1), hidden=0)):
        report = source.receiver_excitation()
        assert source.silent_donor_subspace
        assert report.even_form_storage == 0
        assert report.tangent_receiver_phase_cost_upper_bound == 0
        assert report.tangent_phase_cost_bound_strict is False
        assert report.nonlinear_correction_status == "positive_bound"
    tiny = _report(Q(1, 2**200)).receiver_excitation()
    assert tiny.even_form_storage == Q(1, 2**400)
    assert tiny.tangent_receiver_phase_cost_upper_bound > 0
    assert tiny.tangent_phase_cost_bound_strict is True


def test_correction_is_a_conditional_interval_threshold_not_measured_motion():
    context = mp.clone()
    context.dps = 100
    report = _report(2).receiver_excitation()
    expected = context.sqrt(7) - context.sqrt(6)
    bound = report.necessary_nonlinear_correction_norm_bounds

    def convert(q):
        return context.mpf(q.numerator) / q.denominator

    assert convert(bound.lo) <= expected <= convert(bound.hi)
    assert report.necessary_nonlinear_correction_norm_lower_bound == bound.lo > 0
    assert report.tangent_receiver_phase_cost_upper_bound == 3
    # This is an analytic threshold, not an upper bound on the actual full
    # phase motion or a substitute for the earlier independently evaluated H2.
    assert not hasattr(report, "receiver_only_targets_excluded")
    assert not hasattr(report, "donor_component_entry_certified")

    large = _report(3).receiver_excitation()
    assert large.available
    assert large.nonlinear_correction_status == "nonpositive_margin"
    assert large.necessary_nonlinear_correction_norm_bounds is None
    assert large.necessary_nonlinear_correction_norm_lower_bound is None

    # Exact positivity below the threshold is distinct from resolving its
    # difference of square roots on the128-bit interval grid.
    numerator = int(context.sqrt(context.mpf(14) / 3) * 2**240)
    near = _report(Q(numerator, 2**240)).receiver_excitation()
    assert 3 * near.even_form_storage / 2 < 7
    assert near.nonlinear_correction_status == "unresolved_bound"
    assert near.necessary_nonlinear_correction_norm_bounds.contains(0)
    assert near.necessary_nonlinear_correction_norm_lower_bound is None


@pytest.mark.parametrize(
    "source",
    (
        _report(1, model=RelationalExchangeModel(2, phase_domain="regular")),
        _report(1, model=RelationalExchangeModel(1, 3, 2, phase_domain="regular")),
        _report(1, contrast=Q(-1, 2)),
        replace(_report(1), law="other_pressure_law"),
    ),
)
def test_unsupported_tangent_law_retains_geometric_decomposition_only(source):
    report = source.receiver_excitation()
    assert report.status == "unavailable" and report.unavailable_reasons
    assert (
        report.even_form_storage + report.odd_form_storage
        == source.initial_form_storage
    )
    assert report.tangent_receiver_phase_cost_upper_bound is None
    assert report.tangent_phase_cost_bound_strict is None
    assert report.necessary_nonlinear_correction_norm_bounds is None
    assert not report.tangent_odd_receiver_silence_certified
    assert report.full_phase_budget_status == "unavailable_law"
    assert report.actual_full_phase_storage_upper_bound is None
    assert report.donor_potential_barrier_first_required is None
    assert report.necessary_donor_phase_release_lower_bound is None


def test_full_nonlinear_budget_uses_all_form_components_and_exact_boundary():
    context = mp.clone()
    context.dps = 90
    exact_release = (56 - 25 * context.sqrt(5)) / 20
    for source in (_report(0), _report(2), _explicit(donor=(0, 1, 0, 0, -1), hidden=1)):
        report = source.receiver_excitation()
        assert report.full_phase_budget_status == "available"
        assert report.actual_full_phase_storage_upper_bound == Q(139, 20)
        assert report.donor_potential_barrier_first_required is True
        lower = report.necessary_donor_phase_release_lower_bound
        assert 0 < context.mpf(lower.numerator) / lower.denominator < exact_release

    # The bound is on total F, even though the tangent output uses F_even.
    odd_large = _explicit(donor=(0, 2, 0, 0, -2), hidden=1).receiver_excitation()
    assert odd_large.even_form_storage == 1 and odd_large.odd_form_storage == 8
    assert odd_large.tangent_receiver_phase_cost_upper_bound == Q(3, 4)
    assert odd_large.nonlinear_correction_status == "positive_bound"
    just_outside = _report(2 + Q(1, 2**200)).receiver_excitation()
    assert just_outside.source.initial_form_storage > 4
    for report in (odd_large, just_outside):
        assert report.available
        assert report.full_phase_budget_status == "outside_original_form_budget"
        assert report.actual_full_phase_storage_upper_bound is None
        assert report.donor_potential_barrier_first_required is None
        assert report.necessary_donor_phase_release_lower_bound is None

    source = _report(2)
    pristine = source.receiver_excitation()
    stale = replace(source, twist_phase_storage_bounds=I(99)).receiver_excitation()
    assert (
        stale.necessary_donor_phase_release_lower_bound
        == pristine.necessary_donor_phase_release_lower_bound
    )


def test_reader_rejects_malformed_preparation_and_exports_exact_fields(tmp_path):
    source = _explicit(donor=(0, 1, 0, 0, -1), hidden=1)
    for change in (
        {"initial_epi": (Q(0),) * 10},
        {"initial_epi": (False,) + source.initial_epi[1:]},
        {"capacity": (True,) + source.capacity[1:]},
        {"initial_form_storage": 0},
        {"degrees": (1,) * 11},
        {"initial_phase_turns": (Q(0),) * 11},
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(source, **change).receiver_excitation()
    report = source.receiver_excitation()
    projected = report.to_dict()
    assert projected["schema"] == "tnfr.relational-sine-receiver-excitation.v1"
    assert projected["report"]["tangent_receiver_phase_cost_upper_bound"] == {
        "numerator": 3,
        "denominator": 4,
    }
    path = tmp_path / "receiver-excitation.json"
    export_to_json(report, path)
    exported = json_loads(path.read_bytes())
    assert exported["report"] == projected["report"]
