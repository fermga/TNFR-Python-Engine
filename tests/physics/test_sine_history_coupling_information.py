"""Independent whole-interval coupling controls for a common unknown law.

The oracle constructs full form/phase block matrices from the physical rows;
it does not factor them into graph moments and two-channel powers. Whole
interval bounds retain a single common coupling through both history stages.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from functools import lru_cache
from itertools import product, repeat
from math import factorial

import pytest

# Reuse only immutable independent oracle helpers; no imported fixtures or
# collected tests are executed by these controls. The full-block oracle below
# is new and does not call any production polynomial implementation.
from tests.physics.test_sine_history_information import _B
from tests.physics.test_sine_history_information import _arguments as _history_arguments
from tests.physics.test_sine_history_information import (
    _contains,
    _display,
    _heat_rows,
    _history_coefficients,
    _sine,
    _source,
    _tangent,
)
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_flow as flow
from tnfr.physics import _sine_history_coupling_information as owner


def _arguments(**changes):
    arguments = _history_arguments()
    arguments["gamma_bounds"] = (Q(49, 100), Q(51, 100))
    arguments.update(changes)
    return arguments


def _report(**changes):
    return owner._prepared_sine_history_coupling_information(**_arguments(**changes))


def _matrix_product(left, right):
    return tuple(
        tuple(
            sum((a * right[k][j] for k, a in enumerate(row) if a), Q(0))
            for j in range(len(right[0]))
        )
        for row in left
    )


def _zero_matrix(size=8):
    return tuple((Q(0),) * size for _ in range(size))


def _node_generator(on):
    rows = [[Q(0)] * 4 for _ in range(4)]
    for i, j in (((0, 1), (1, 2), (2, 3)) if on else ((1, 2), (2, 3))):
        rows[i][j] += Q(1, 4)
        rows[j][i] += Q(1, 4)
        rows[i][i] -= Q(1, 4)
        rows[j][j] -= Q(1, 4)
    return tuple(map(tuple, rows))


def _full_blocks(on):
    """J(gamma)=J0+e J1 from x'=Gx+gamma Gphi, phi'=-gamma Gx."""
    generator = _node_generator(on)
    constant, slope = [[Q(0)] * 8 for _ in range(8)], [[Q(0)] * 8 for _ in range(8)]
    for i in range(4):
        for j in range(4):
            constant[i][j] = generator[i][j]
            constant[i][j + 4] = generator[i][j] / 2
            constant[i + 4][j] = -generator[i][j] / 2
            slope[i][j + 4] = generator[i][j]
            slope[i + 4][j] = -generator[i][j]
    return tuple(map(tuple, constant)), tuple(map(tuple, slope))


@lru_cache(maxsize=None)
def _block_taylor_terms(on, duration):
    """Full eight-coordinate Taylor powers, each a polynomial in the SAME e."""
    j0, j1 = _full_blocks(on)
    identity = tuple(tuple(Q(i == j) for j in range(8)) for i in range(8))
    terms = [(identity,)]
    zero = _zero_matrix()
    for degree in range(1, 17):
        previous = terms[-1]
        coefficients = []
        for power in range(degree + 1):
            left = (
                _matrix_product(j0, previous[power]) if power < len(previous) else zero
            )
            right = _matrix_product(j1, previous[power - 1]) if power else zero
            coefficients.append(
                tuple(
                    tuple(duration * (a + b) / degree for a, b in zip(arow, brow))
                    for arow, brow in zip(left, right)
                )
            )
        terms.append(tuple(coefficients))
    return tuple(terms)


@lru_cache(maxsize=None)
def _full_history_polynomials(reflect_phase=False):
    """Ordered full-matrix product through TOTAL degree16, not a 2x2 oracle."""
    prefix = _block_taylor_terms(False, Q(1, 4))
    off, on = _block_taylor_terms(False, Q(1, 4)), _block_taylor_terms(True, Q(1, 4))
    result = [[Q(0)] * 17 for _ in range(8)]
    for response_order in range(1, 17):
        for prefix_order in range(17 - response_order):
            for response_power, (off_matrix, on_matrix) in enumerate(
                zip(off[response_order], on[response_order])
            ):
                row = tuple(
                    (
                        on_matrix[0][j]
                        + on_matrix[1][j]
                        - off_matrix[0][j]
                        - off_matrix[1][j]
                    )
                    / 2
                    for j in range(8)
                )
                if reflect_phase:
                    row = tuple(
                        value if j < 4 else -value for j, value in enumerate(row)
                    )
                if not any(row):
                    continue
                for prefix_power, prefix_matrix in enumerate(prefix[prefix_order]):
                    projected = _matrix_product((row,), prefix_matrix)[0]
                    for i, value in enumerate(projected):
                        result[i][response_power + prefix_power] += value
    return tuple(map(tuple, result))


def _source_polynomials(arguments):
    physical = _full_history_polynomials()
    ordered = tuple(
        physical[node + (4 if channel == "relative_phase" else 0)]
        for node, channel in arguments["coordinate_labels"]
    )
    center = tuple(
        sum(
            (
                row[k] * value
                for row, value in zip(ordered, arguments["initial_center"])
            ),
            Q(0),
        )
        for k in range(17)
    )
    columns = tuple(
        tuple(
            sum(
                (
                    row[k] * source[j]
                    for row, source in zip(ordered, arguments["initial_generators"])
                ),
                Q(0),
            )
            for k in range(17)
        )
        for j in range(len(arguments["uncertainty_labels"]))
    )
    return center, columns


def _evaluate(polynomial, value):
    return sum(
        (coefficient * value**power for power, coefficient in enumerate(polynomial)),
        Q(0),
    )


def _tail(exponent, degree=16):
    return (
        exponent ** (degree + 1) / factorial(degree + 1) / (1 - exponent / (degree + 2))
    )


def _general_tube(form, phase, degree, duration, gamma):
    z = gamma * degree * duration / 2
    growth = 1 / (1 - z**2 / 2)
    phase_ceiling = (phase + z * form) * growth
    error = duration * gamma * degree * phase_ceiling**3 * growth / 3
    return z, growth, phase_ceiling, form + z * phase_ceiling, error


@pytest.fixture(scope="module", autouse=True)
def no_nonlinear_trajectory_event_or_retained_source_evaluation():
    def forbidden(*args, **kwargs):
        pytest.fail("coupling controls cannot integrate or replay a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
            (external, "_derive_organized_external_interaction"),
            (external, "_assess_organized_external_interaction"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def report():
    return _report()


@pytest.mark.parametrize("gamma", [Q(49, 100), Q(1, 2), Q(51, 100)])
def test_both_physical_tangent_rows_and_cayley_hamilton_change_with_gamma(gamma):
    channel = ((Q(1), gamma), (-gamma, Q(0)))
    square = _matrix_product(channel, channel)
    assert square == tuple(
        tuple(channel[i][j] - (gamma**2 if i == j else 0) for j in range(2))
        for i in range(2)
    )
    centered = tuple(
        tuple(channel[i][j] - (Q(1, 2) if i == j else 0) for j in range(2))
        for i in range(2)
    )
    assert _matrix_product(centered, centered) == (
        (Q(1, 4) - gamma**2, Q(0)),
        (Q(0), Q(1, 4) - gamma**2),
    )
    if gamma != Q(1, 2):
        assert Q(1, 4) - gamma**2 != 0
    for on in (False, True):
        j0, j1 = _full_blocks(on)
        actual = tuple(
            tuple(a + (gamma - Q(1, 2)) * b for a, b in zip(arow, brow))
            for arow, brow in zip(j0, j1)
        )
        for column in range(8):
            form = tuple(_B(Q(i == column)) for i in range(4))
            phase = tuple(_B(Q(i + 4 == column)) for i in range(4))
            edges = ((0, 1), (1, 2), (2, 3)) if on else ((1, 2), (2, 3))
            form_heat, phase_heat = _heat_rows(form, edges), _heat_rows(phase, edges)
            expected = tuple(
                x + gamma * p for x, p in zip(form_heat, phase_heat)
            ) + tuple(-gamma * x for x in form_heat)
            assert tuple(actual[i][column] for i in range(8)) == tuple(
                x.lo for x in expected
            )


def test_full_eight_coordinate_polynomial_equals_compact_production_after_source_contraction(
    report,
):
    center, columns = _source_polynomials(_arguments())
    assert report.polynomial_degree == 16
    assert report.parameter_center_coefficients == center
    assert report.parameter_generator_coefficients == columns
    assert any(center[1:]) and any(any(column[1:]) for column in columns)
    args = _arguments()
    physical = _full_history_polynomials()
    for corner in product((-1, 1), repeat=6):
        state = _source(args, corner)
        direct = tuple(
            sum((row[k] * x for row, x in zip(physical, state)), Q(0))
            for k in range(17)
        )
        affine = tuple(
            center[k] + sum(column[k] * u for column, u in zip(columns, corner))
            for k in range(17)
        )
        assert direct == affine


def test_independent_critical_closed_modes_fit_uniform_taylor_tail(report):
    exact = _history_coefficients(Q(1, 4), Q(1, 4))
    polynomial = _full_history_polynomials()
    tail = 2 * _tail(report.series_norm_step)
    for series, oracle in zip(polynomial, exact):
        assert series[0] - tail <= oracle.lo <= oracle.hi <= series[0] + tail
    center, columns = _source_polynomials(_arguments())
    physical_center = sum(
        (x * c for x, c in zip(exact, _arguments()["initial_center"])), _B(0)
    )
    _contains(report.critical_reference.tangent_center_bounds, physical_center)
    assert (
        abs(report.predicted_response - center[0])
        <= report.computational_center_error_upper_bound
        + 2 * report.initial_state_upper_bound * report.series_tail_upper_bound
    )


def test_entire_unknown_interval_variation_is_bounded_without_endpoint_inference(
    report, record_property
):
    eta = Q(1, 100)
    center, columns = _source_polynomials(_arguments())
    polynomial_loss = sum(
        (
            eta**k * (abs(center[k]) + sum(abs(column[k]) for column in columns))
            for k in range(1, 17)
        ),
        Q(0),
    )
    assert report.gamma_bounds == (Q(49, 100), Q(51, 100))
    assert report.coupling_deviation_upper_bound == eta
    assert report.parameter_polynomial_error_upper_bound == polynomial_loss
    assert (
        report.parameter_tail_error_upper_bound
        == 4 * report.initial_state_upper_bound * report.series_tail_upper_bound
    )
    assert (
        report.parameter_response_error_upper_bound
        == polynomial_loss + report.parameter_tail_error_upper_bound
        <= Q(7, 10**10)
    )
    # For EVERY |e|<=eta and |u_j|<=1, each nonconstant coefficient of
    # c(e)+sum d_j(e)u_j is bounded by the following exact coefficient sum.
    # Evaluated endpoints below are additional controls, not the interval proof.
    for corner in product((-1, 1), repeat=6):
        coefficients = tuple(
            center[k] + sum(row[k] * u for row, u in zip(columns, corner))
            for k in range(17)
        )
        assert (
            sum(abs(coefficients[k]) * eta**k for k in range(1, 17)) <= polynomial_loss
        )
        for e in (-eta, Q(0), eta):
            assert abs(_evaluate(coefficients, e) - coefficients[0]) <= polynomial_loss
    center_loss = sum(abs(center[k]) * eta**k for k in range(1, 17))
    assert center_loss > 0
    assert polynomial_loss > sum(
        eta**k * sum(abs(row[k]) for row in columns) for k in range(1, 17)
    )
    record_property("uniform_coupling_polynomial_loss", _display(polynomial_loss))
    record_property(
        "uniform_coupling_tail_loss", _display(report.parameter_tail_error_upper_bound)
    )
    record_property(
        "uniform_coupling_total_loss",
        _display(report.parameter_response_error_upper_bound),
    )
    record_property(
        "policy",
        "one UNKNOWN held gamma common to both stages/branches; a single critical-center forecast",
    )


def test_global_tail_is_an_upper_bound_for_all_omitted_total_degrees(report):
    z = Q(151, 200)
    assert report.series_norm_step == z
    assert report.initial_state_upper_bound == Q(3, 10000)
    assert report.series_tail_upper_bound == _tail(z)
    partial = sum((z**k / factorial(k) for k in range(17, 61)), Q(0))
    remaining = z**61 / factorial(61) / (1 - z / 62)
    assert 0 < partial < partial + remaining <= report.series_tail_upper_bound
    assert report.series_tail_upper_bound < Q(101, 100) * partial
    # Binomial grouping of the two ordered Taylor series has total norm z;
    # it is not valid to bound only one stage's omitted terms.
    assert _tail(z) > 2 * _tail(z / 2)


def test_general_gamma_nonlinear_budget_covers_both_stages_and_keeps_reference_meaning(
    report,
):
    b0, p0, gamma = Q(3, 10000), Q(1, 10**8), Q(51, 100)
    z, c, phase, form, error = _general_tube(b0, p0, Q(2), Q(1, 2), gamma)
    assert z == Q(51, 200)
    assert report.off_phase_growth_bound == report.on_phase_growth_bound == c
    assert (
        report.off_phase_tube_upper_bound == report.on_phase_tube_upper_bound == phase
    )
    assert (
        report.off_nonlinear_error_upper_bound
        == report.on_nonlinear_error_upper_bound
        == error
    )
    assert report.nonlinear_response_error_upper_bound == 2 * error < Q(18, 10**14)
    zp, cp, ph, bh, ex = _general_tube(b0, p0, Q(2), Q(1, 4), gamma)
    assert report.preparation_phase_growth_bound == cp
    assert report.preparation_phase_tube_upper_bound == ph
    assert report.preparation_form_tube_upper_bound == bh
    assert report.preparation_form_error_upper_bound == ex < Q(5, 10**15)
    assert report.preparation_phase_error_upper_bound == zp * ex
    assert report.critical_reference.gamma == Q(1, 2)
    assert (
        report.preparation_phase_tube_upper_bound
        > report.critical_reference.preparation_phase_tube_upper_bound
    )
    assert (
        report.nonlinear_response_error_upper_bound
        > report.critical_reference.nonlinear_response_error_upper_bound
    )
    assert form > b0 and phase > p0


def test_fixed_unknown_law_forecast_passes_unchanged_useful_target(
    report, record_property
):
    reference = report.critical_reference
    assert report.predicted_response == reference.predicted_response
    assert (
        report.forecast_error_upper_bound
        == reference.linear_information_error_bounds[1]
        + reference.computational_center_error_upper_bound
        + report.parameter_response_error_upper_bound
        + report.nonlinear_response_error_upper_bound
    )
    assert report.forecast_error_upper_bound < reference.tolerance == Q(1, 200000000)
    assert report.forecast_response_bounds[0] > Q(18, 10**10)
    assert report.status == "certified" and report.response_sign == "positive"
    record_property("fixed_forecast", _display(report.predicted_response))
    record_property(
        "unknown_coupling_forecast_error", _display(report.forecast_error_upper_bound)
    )
    record_property(
        "uniform_actual_response_interval",
        tuple(_display(x) for x in report.forecast_response_bounds),
    )
    record_property(
        "uniform_nonlinear_error", _display(report.nonlinear_response_error_upper_bound)
    )


def test_whole_channel_hybrid_obstruction_is_uniform_for_each_gamma(
    report, record_property
):
    delta, epsilon, eta, gamma = Q(1, 10000), Q(1, 10**8), Q(1, 100), Q(51, 100)
    reflected = _full_history_polynomials(True)
    half_polynomial = tuple(delta * x for x in reflected[2])
    loss = (
        sum((abs(half_polynomial[k]) * eta**k for k in range(1, 17)), Q(0))
        + 4 * delta * report.series_tail_upper_bound
    )
    assert loss <= Q(7, 10**10)
    # At a fixed gamma, complete actual reached states from x2=2delta+-delta
    # yield (xplus,phiminus) and (xminus,phiplus). Both belong to the product
    # of THAT gamma's actual full-channel projections. There is no cross-gamma
    # hybrid, coordinate-box substitution, or claim of joint reachability.
    plus = _tangent(Q(1, 4), (0, 0, 3 * delta, 0), (0, 0, 0, 0), on=False)[0]
    minus = _tangent(Q(1, 4), (0, 0, delta, 0), (0, 0, 0, 0), on=False)[0]
    hybrid_plus, hybrid_minus = plus[:4] + minus[4:], minus[:4] + plus[4:]
    direct = _history_coefficients(Q(0), Q(1, 4))
    critical_half = sum(
        (c * (a - b) / 2 for c, a, b in zip(direct, hybrid_plus, hybrid_minus)), _B(0)
    )
    tail = 2 * delta * report.series_tail_upper_bound
    assert (
        half_polynomial[0] - tail
        <= critical_half.lo
        <= critical_half.hi
        <= half_polynomial[0] + tail
    )
    critical_lower = delta * (Q(54375, 1073741824) + Q(3045, 402653184) - Q(1, 2097152))
    assert critical_half.lo > critical_lower > Q(577, 10**11)
    assert reflected[2] != _full_history_polynomials()[2]
    zp, cp, ph, bh, ex = _general_tube(3 * delta, epsilon, Q(2), Q(1, 4), gamma)
    assert ph < Q(39, 100) * delta and bh < Q(305, 100) * delta
    ephase = zp * ex
    zr, cr, _, _, branch = _general_tube(
        Q(305, 100) * delta, Q(39, 100) * delta, Q(2), Q(1, 4), gamma
    )
    transfer = 2 * cr * (ex + zr * ephase)
    assert transfer < Q(11, 10**15)
    hybrid_nonlinear = 2 * branch
    assert hybrid_nonlinear < Q(84, 10**15)
    lower = critical_lower - loss - transfer - hybrid_nonlinear
    assert lower > Q(1, 200000000)
    for e in (-eta, Q(0), eta):
        assert abs(_evaluate(half_polynomial, e) - half_polynomial[0]) <= loss
    record_property("uniform_hybrid_coupling_loss", _display(loss))
    record_property("uniform_hybrid_initial_error_transfer", _display(transfer))
    record_property("uniform_hybrid_nonlinear_error", _display(hybrid_nonlinear))
    record_property("each_gamma_channel_product_unavoidable_lower", _display(lower))
    record_property(
        "comparator",
        "for EACH gamma: product of exact actual full-channel projections; unknown class is union of same-gamma products",
    )


@pytest.mark.parametrize("gamma", [Q(49, 100), Q(1, 2), Q(51, 100)])
def test_actual_full_field_and_jets_use_gamma_in_both_rows(gamma):
    delta = Q(1, 10000)
    field = flow._held_quarter_sine_field(
        capacity=(1,) * 4, gamma=gamma, weighted_edges=((1, 2, 1), (2, 3, 1))
    )
    initial = (Q(0), Q(0), 2 * delta, Q(0)) + (Q(0),) * 4
    first = field(initial)
    expected = tuple(
        _B(x)
        for x in (
            0,
            delta / 2,
            -delta,
            delta / 2,
            0,
            -gamma * delta / 2,
            gamma * delta,
            -gamma * delta / 2,
        )
    )
    for actual, oracle in zip(first, expected):
        _contains((actual.lo, actual.hi), oracle, Q(1, 10**35))
    jets = tuple(
        Jet((I(value), derivative)) for value, derivative in zip(initial, first)
    )
    second = field(jets)
    f = 1 - gamma**2
    expected_second = tuple(
        _B(x)
        for x in (
            0,
            -3 * delta * f / 8,
            3 * delta * f / 4,
            -3 * delta * f / 8,
            0,
            3 * gamma * delta / 8,
            -3 * gamma * delta / 4,
            3 * gamma * delta / 8,
        )
    )
    for actual, oracle in zip(second, expected_second):
        _contains((actual.coeffs[1].lo, actual.coeffs[1].hi), oracle, Q(1, 10**35))
    nonlinear_initial = initial[:4] + (0, 0, -2 * delta, 0)
    current = field(nonlinear_initial)
    sine = _sine(2 * delta)
    _contains(
        (current[1].lo, current[1].hi), delta / 2 - gamma * sine / 4, Q(1, 10**35)
    )
    _contains((current[5].lo, current[5].hi), _B(-gamma * delta / 2), Q(1, 10**35))
    assert sine.hi < 2 * delta


def test_singleton_coupling_recovers_the_complete_critical_reference():
    result = _report(gamma_bounds=(Q(1, 2), Q(1, 2)))
    assert result.coupling_deviation_upper_bound == 0
    assert result.series_tail_upper_bound > 0
    assert result.parameter_polynomial_error_upper_bound == 0
    assert result.parameter_tail_error_upper_bound == 0
    assert result.parameter_response_error_upper_bound == 0
    for name in (
        "preparation_form_error_upper_bound",
        "preparation_phase_error_upper_bound",
        "nonlinear_response_error_upper_bound",
        "linear_information_error_bounds",
        "forecast_error_upper_bound",
        "forecast_response_bounds",
        "status",
    ):
        assert getattr(result, name) == getattr(result.critical_reference, name)


@pytest.mark.parametrize("zero", ["elapsed_time", "contact_weight"])
def test_exact_matched_zero_preserves_nonzero_preparation(zero):
    result = _report(**{zero: Q(0)})
    assert result.critical_reference.preparation_kernel is not None
    assert result.preparation_form_error_upper_bound > 0
    assert result.off_nonlinear_error_upper_bound > 0
    assert result.parameter_center_coefficients == (Q(0),) * 17
    assert result.parameter_generator_coefficients == ((Q(0),) * 17,) * 6
    assert result.parameter_response_error_upper_bound == 0
    assert result.nonlinear_response_error_upper_bound == 0
    assert result.forecast_response_bounds == (Q(0), Q(0))
    assert result.status == "certified" and result.response_sign == "zero"


def test_both_windows_share_the_capacity_clock(report):
    scaled = _report(
        capacity=(Q(3),) * 4, preparation_time=Q(1, 12), elapsed_time=Q(1, 12)
    )
    for name in (
        "parameter_center_coefficients",
        "parameter_generator_coefficients",
        "series_norm_step",
        "series_tail_upper_bound",
        "parameter_response_error_upper_bound",
        "preparation_form_error_upper_bound",
        "preparation_phase_error_upper_bound",
        "nonlinear_response_error_upper_bound",
        "forecast_error_upper_bound",
        "forecast_response_bounds",
        "status",
    ):
        assert getattr(scaled, name) == getattr(report, name)
    assert scaled.preparation_mobility_degree == 3 * report.preparation_mobility_degree


def test_labels_and_signed_latent_columns_keep_their_source_association(report):
    args = _arguments()
    names = {0: "origin", 1: "retained", 2: "near", 3: "far"}
    rows = (7, 2, 4, 1, 6, 0, 3, 5)
    columns = (5, 1, 3, 0, 4, 2)
    signs = (-1, 1, -1, 1, 1, -1)
    changed = _report(
        nodes=("far", "retained", "origin", "near"),
        old_edges=(("near", "retained"), ("far", "near")),
        contact=("retained", "origin"),
        retained_nodes=("retained", "origin"),
        observation_nodes=("origin", "retained"),
        phase_anchor="origin",
        coordinate_labels=tuple(
            (names[args["coordinate_labels"][i][0]], args["coordinate_labels"][i][1])
            for i in rows
        ),
        initial_center=tuple(args["initial_center"][i] for i in rows),
        initial_generators=tuple(
            tuple(
                args["initial_generators"][i][j] * signs[k]
                for k, j in enumerate(columns)
            )
            for i in rows
        ),
        uncertainty_labels=tuple(args["uncertainty_labels"][j] for j in columns),
    )
    assert changed.parameter_center_coefficients == report.parameter_center_coefficients
    assert changed.parameter_generator_coefficients == tuple(
        tuple(signs[k] * x for x in report.parameter_generator_coefficients[j])
        for k, j in enumerate(columns)
    )
    for name in (
        "parameter_response_error_upper_bound",
        "nonlinear_response_error_upper_bound",
        "forecast_error_upper_bound",
        "forecast_response_bounds",
    ):
        assert getattr(changed, name) == getattr(report, name)


def test_zero_dimensional_source_readout_and_declared_cost_are_separate(report):
    point = _report(initial_generators=((),) * 8, uncertainty_labels=())
    assert point.parameter_generator_coefficients == ()
    assert point.linear_information_error_bounds == (Q(0), Q(0))
    assert (
        point.parameter_polynomial_error_upper_bound > 0
    )  # Unknown law moves the known center.
    zero = _report(
        initial_center=(Q(0),) * 8, initial_generators=((),) * 8, uncertainty_labels=()
    )
    assert (
        zero.parameter_response_error_upper_bound
        == zero.nonlinear_response_error_upper_bound
        == 0
    )
    assert zero.forecast_response_bounds == (Q(0), Q(0))
    noisy = _report(reading_error_bound=Q(1, 10**7), preparation_relation_cost=13)
    assert noisy.forecast_error_upper_bound == report.forecast_error_upper_bound + Q(
        1, 10**7
    )
    assert (
        noisy.parameter_response_error_upper_bound
        == report.parameter_response_error_upper_bound
    )
    assert noisy.information_error_lower_bound == report.information_error_lower_bound
    assert noisy.critical_reference.preparation_relation_cost == 13
    assert noisy.status == noisy.response_sign == "unresolved"


def test_one_shot_primitive_inputs_are_normalized_before_rebuilding_reference(report):
    args = _arguments()
    for name in (
        "nodes",
        "old_conductances",
        "capacity",
        "contact",
        "retained_nodes",
        "observation_nodes",
        "initial_center",
        "uncertainty_labels",
        "gamma_bounds",
    ):
        args[name] = iter(args[name])
    for name in ("old_edges", "coordinate_labels", "initial_generators"):
        args[name] = iter(tuple(iter(row) for row in args[name]))
    assert owner._prepared_sine_history_coupling_information(**args) == report
    with pytest.raises(FrozenInstanceError):
        report.gamma_bounds = (Q(1, 2), Q(1, 2))
    with pytest.raises(TypeError):
        _report(critical_reference=report.critical_reference)


@pytest.mark.parametrize(
    "gamma,window", [((Q(49, 100), Q(51, 100)), Q(50, 151)), ((Q(1, 2), Q(1)), Q(1, 4))]
)
def test_closed_series_and_general_tube_boundaries_are_admitted(gamma, window):
    result = _report(gamma_bounds=gamma, preparation_time=window, elapsed_time=window)
    assert result.series_norm_step == 1
    z = gamma[1] * 2 * window
    assert z <= Q(1, 2)
    assert result.on_phase_growth_bound == 1 / (1 - z**2 / 2)


def _invalid_cases():
    return (
        {"gamma_bounds": ()},
        {"gamma_bounds": (Q(1, 2),)},
        {"gamma_bounds": (Q(0), Q(1, 2), Q(1))},
        {"gamma_bounds": {Q(49, 100), Q(51, 100)}},
        {"gamma_bounds": repeat(Q(1, 2))},
        {"gamma_bounds": (False, Q(1, 2))},
        {"gamma_bounds": (Q(-1, 100), Q(1, 2))},
        {"gamma_bounds": (Q(51, 100), Q(49, 100))},
        {"gamma_bounds": (Q(3, 5), Q(4, 5))},
        {"gamma_bounds": (Q(0), Q(49, 100))},
        {"gamma_bounds": (Q(0), float("inf"))},
        {"gamma_bounds": (float("nan"), Q(1, 2))},
        {"preparation_time": Q(-1, 100)},
        {"preparation_time": True},
        {"preparation_relation_cost": True},
        {"capacity": (1, 1, 0, 1)},
        {"old_edges": ((1, 2), (2, 1))},
        {"observation_nodes": (2,)},
        {"coordinate_labels": ((0, "form"),) * 8},
        {"initial_center": (0, 0, 0, 0, Q(1, 100), 0, 0, 0)},
        {"initial_center": (0, 0, 0, 0, 0, 0, Q(1, 3), 0)},
        # norm=1.1325>1 while the nonlinear tube remains admitted.
        {"preparation_time": Q(1, 2)},
        # General tube z=.6>.5 while the series norm=.9 remains admitted.
        {
            "gamma_bounds": (Q(1, 2), Q(2)),
            "preparation_time": Q(3, 20),
            "elapsed_time": Q(3, 20),
        },
        # Exact matched-zero shortcuts must not bypass raw/domain admission.
        {"elapsed_time": Q(0), "gamma_bounds": (Q(3, 5), Q(4, 5))},
        {"contact_weight": Q(0), "preparation_time": Q(1)},
    )


@pytest.mark.parametrize("changes", _invalid_cases())
def test_all_primitive_and_domain_admission_precedes_reference_and_matrix_work(
    monkeypatch, changes
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid source reached scientific work")

    for name in (
        "_prepared_sine_history_information",
        "exact_matrix_product",
        "_prepared_exponential_series_tail_upper",
        "_prepared_matched_coupling_polynomials",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    with pytest.raises((TypeError, ValueError)):
        _report(**changes)


@pytest.mark.parametrize(
    "exponent,degree", [(Q(0), 0), (Q(1), 0), (Q(1, 2), 16), (Q(1), 32)]
)
def test_shared_positive_tail_uses_exact_first_omitted_term(exponent, degree):
    result = owner._prepared_exponential_series_tail_upper(exponent, degree)
    assert result == _tail(exponent, degree)
    assert (result == 0) == (exponent == 0)


@pytest.mark.parametrize(
    "exponent,degree",
    [
        (True, 16),
        (0.5, 16),
        (Q(-1), 16),
        (Q(1) + Q(1, 10**100), 16),
        (Q(1), True),
        (Q(1), -1),
        (Q(1), 33),
        (Q(1), Q(16)),
    ],
)
def test_shared_tail_rejects_malformed_or_out_of_domain_inputs(exponent, degree):
    with pytest.raises((TypeError, ValueError)):
        owner._prepared_exponential_series_tail_upper(exponent, degree)
