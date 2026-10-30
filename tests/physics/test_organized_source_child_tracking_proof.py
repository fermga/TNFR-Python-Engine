"""Independent static proof controls for accumulated child-core error.

Fine incidence, nonlinear secants and complete clock rows are rebuilt here.
No trajectory, reset or reserved response supplies a comparison state.
"""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics._organized_source_port import _observe_organized_source_capacity

CHILD_EDGES = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))
EDGES = tuple(
    (9 * child + slot, 9 * child + (slot + 1) % 9)
    for child in range(5)
    for slot in range(9)
) + tuple((9 * a + slot, 9 * b + slot) for a, b in CHILD_EDGES for slot in range(9))
SWAP = tuple(9 * (0, 2, 1, 3, 4)[i // 9] + i % 9 for i in range(45))
UNIT = (Q(1),) * 45
H = Q(1, 128)


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Tracking proof controls cannot execute events or trajectories")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(context, value):
    if isinstance(value, (Q, int)):
        return context.mpf(value.numerator) / int(value.denominator)
    return value


def _project(row):
    return tuple((row[i] + row[SWAP[i]]) / 2 for i in range(45))


def _center(row):
    mean = sum(row, Q(0)) / len(row)
    return tuple(value - mean for value in row)


def _divergence(currents):
    result = [0 * currents[0]] * 45
    for (a, b), current in zip(EDGES, currents):
        result[a] += current
        result[b] -= current
    return tuple(result)


def _laplacian(row, weights=None):
    weights = (Q(1),) * 90 if weights is None else weights
    return _divergence(
        tuple(weight * (row[a] - row[b]) for (a, b), weight in zip(EDGES, weights))
    )


def _edge_angles(context, offsets):
    alpha = 2 * context.pi / 9
    return tuple(
        (alpha if a // 9 == b // 9 else 0) + _mp(context, offsets[b] - offsets[a])
        for a, b in EDGES
    )


def _fields(context, form, offsets):
    gamma = 1 / (1023 * context.pi)
    gradient = _laplacian(form)
    currents = tuple(context.sin(angle) for angle in _edge_angles(context, offsets))
    sine = _divergence(currents)
    return (
        tuple((-_mp(context, q) + gamma * s) / 4 for q, s in zip(gradient, sine)),
        tuple(gamma * _mp(context, q) / 4 for q in gradient),
    )


def _energy(context, form, offsets):
    return context.fsum(
        _mp(context, form[a] - form[b]) ** 2 / 2 + 1 - context.cos(angle)
        for (a, b), angle in zip(EDGES, _edge_angles(context, offsets))
    ) - 45 * (1 - context.cos(2 * context.pi / 9))


def _hidden_lift(a, b, form=None, offsets=None):
    x = [Q(0)] * 45 if form is None else list(form)
    delta = [Q(0)] * 45 if offsets is None else list(offsets)
    for j in range(9):
        x[9 + j] += a[j]
        x[18 + j] -= a[j]
        delta[9 + j] += b[j]
        delta[18 + j] -= b[j]
    return tuple(x), tuple(delta)


def _edge_forcing(context, offsets):
    actual = _edge_angles(context, offsets)
    exchanged = _edge_angles(context, tuple(offsets[i] for i in SWAP))
    projected = _edge_angles(context, _project(offsets))
    return tuple(
        (context.sin(left) + context.sin(right)) / 2 - context.sin(middle)
        for left, right, middle in zip(actual, exchanged, projected)
    )


@pytest.mark.parametrize("profile", ("frozen", "alternating", "mixed"))
def test_edge_forcing_retains_two_cycles_four_matching_edges_and_constant_five(
    mp, profile
):
    b = {
        "frozen": (Q(0), H) + (Q(0),) * 7,
        "alternating": tuple((-1) ** j * H for j in range(9)),
        "mixed": tuple(Q((5 * j) % 11 - 5, 4096) for j in range(9)),
    }[profile]
    projected = _project(tuple(Q((7 * i) % 13 - 6, 4096) for i in range(45)))
    _, offsets = _hidden_lift((Q(0),) * 9, b, offsets=projected)
    forcing = _edge_forcing(mp, offsets)
    projected_angles = _edge_angles(mp, projected)
    cycle_count = matching_count = 0
    for (left, right), angle, current in zip(EDGES, projected_angles, forcing):
        cycle = left // 9 == right // 9 and left // 9 in (1, 2)
        matching = left // 9 != right // 9 and {left // 9, right // 9} != {0, 4}
        cycle_count += cycle
        matching_count += matching
        odd_gap = b[right % 9] - b[left % 9] if cycle else b[left % 9]
        expected = mp.sin(angle) * (mp.cos(_mp(mp, odd_gap)) - 1)
        if not (cycle or matching):
            expected = mp.mpf(0)
        assert mp.almosteq(current, expected, abs_eps=mp.mpf("1e-95"))
        if cycle or matching:
            assert abs(current) <= _mp(mp, odd_gap**2 / 2)
    assert cycle_count == 18 and matching_count == 36
    fourth_cycle = sum(((b[(j + 1) % 9] - b[j]) ** 4 for j in range(9)), Q(0))
    fourth_matching = sum((value**4 for value in b), Q(0))
    squared_hidden = sum((value**2 for value in b), Q(0))
    assert fourth_cycle <= 8 * squared_hidden**2
    edge_bound = fourth_cycle / 2 + fourth_matching
    assert edge_bound <= 5 * squared_hidden**2
    assert mp.fsum(value**2 for value in forcing) <= _mp(mp, edge_bound)
    fine_sine = _divergence(tuple(mp.sin(angle) for angle in _edge_angles(mp, offsets)))
    projected_sine = _divergence(tuple(mp.sin(angle) for angle in projected_angles))
    defect = tuple(a - b for a, b in zip(_project(fine_sine), projected_sine))
    for actual, expected in zip(defect, _divergence(forcing)):
        assert mp.almosteq(actual, expected, abs_eps=mp.mpf("1e-95"))
    assert abs(mp.fsum(defect)) < mp.mpf("1e-95")


def test_frozen_matched_initial_state_has_nonzero_quadratic_form_defect(mp):
    state = _hidden_lift((H,) + (Q(0),) * 8, (Q(0), H) + (Q(0),) * 7)
    core = tuple(_project(row) for row in state)
    assert core == ((Q(0),) * 45,) * 2
    fine_rows, core_rows = _fields(mp, *state), _fields(mp, *core)
    defect = _project(fine_rows[0])[9] - core_rows[0][9]
    gamma, alpha = 1 / (1023 * mp.pi), 2 * mp.pi / 9
    expected = gamma * mp.sin(alpha) * (mp.cos(_mp(mp, H)) - 1) / 4
    assert mp.almosteq(defect, expected, abs_eps=mp.mpf("1e-95"))
    assert defect < 0
    assert all(abs(value) < mp.mpf("1e-95") for row in core_rows for value in row)
    assert _energy(mp, *state) < mp.mpf(".01")


def _matrix(rows):
    return tuple(tuple(Q(value) for value in row) for row in rows)


def _scale(matrix, factor):
    return tuple(tuple(factor * value for value in row) for row in matrix)


def _sum(left, right):
    return tuple(tuple(x + y for x, y in zip(a, b)) for a, b in zip(left, right))


def _blocks(rows):
    return tuple(
        tuple(value for block_row in row for value in block_row)
        for blocks in rows
        for row in zip(*blocks)
    )


@pytest.mark.parametrize("gamma", (Q(1, 4096), Q(1, 8), Q(1, 2)))
@pytest.mark.parametrize(
    "weights", ((Q(1, 2),) * 3, (Q(1, 2), Q(3, 4), Q(1)), (Q(1),) * 3)
)
def test_forced_metric_identity_and_young_bound_hold_as_exact_matrix_inequalities(
    gamma, weights
):
    # A grounded incidence tests the complete quadratic in p, q and the
    # arbitrary edge forcing j. No selected state vectors stand in for PSD.
    incidence = _matrix(((1, 0, 1), (-1, 1, 0)))
    transpose = tuple(zip(*incidence))
    multiply = exact_matrix_product
    laplacian = multiply(incidence, transpose)
    weighted = tuple(
        tuple(value * weight for value, weight in zip(row, weights))
        for row in incidence
    )
    stiffness = multiply(weighted, transpose)
    identity = _matrix(((1, 0), (0, 1)))
    zero22, zero23 = _scale(identity, Q(0)), _scale(incidence, Q(0))
    zero32 = tuple(zip(*zero23))
    identity3 = _matrix(tuple(tuple(int(i == j) for j in range(3)) for i in range(3)))
    zero33 = _scale(identity3, Q(0))
    assert exact_symmetric_semidefinite(_sum(stiffness, _scale(laplacian, Q(-1, 2))))
    assert exact_symmetric_semidefinite(_sum(laplacian, _scale(stiffness, Q(-1))))
    assert exact_symmetric_semidefinite(_sum(laplacian, _scale(identity, Q(-4, 15))))
    if len(set(weights)) > 1:
        assert multiply(laplacian, stiffness) != multiply(stiffness, laplacian)
    generator = _scale(
        _blocks(
            (
                (
                    _scale(laplacian, Q(-1)),
                    _scale(stiffness, -gamma),
                    _scale(incidence, gamma),
                ),
                (_scale(laplacian, gamma), zero22, zero23),
                (zero32, zero32, zero33),
            )
        ),
        Q(1, 4),
    )
    metric = _blocks(
        (
            (identity, _scale(identity, gamma), zero23),
            (_scale(identity, gamma), identity, zero23),
            (zero32, zero32, zero33),
        )
    )
    derivative = _sum(
        multiply(tuple(zip(*generator)), metric), multiply(metric, generator)
    )
    expected = _blocks(
        (
            (
                _scale(laplacian, -(1 - gamma**2) / 2),
                _scale(stiffness, -gamma / 4),
                _scale(incidence, gamma / 4),
            ),
            (
                _scale(stiffness, -gamma / 4),
                _scale(stiffness, -(gamma**2) / 2),
                _scale(incidence, gamma**2 / 4),
            ),
            (_scale(transpose, gamma / 4), _scale(transpose, gamma**2 / 4), zero33),
        )
    )
    assert derivative == expected
    allowed = _blocks(
        (
            (_scale(laplacian, Q(-1, 16)), zero22, zero23),
            (zero22, _scale(laplacian, -(gamma**2) / 16), zero23),
            (zero32, zero32, _scale(identity3, 2 * gamma**2)),
        )
    )
    assert exact_symmetric_semidefinite(_sum(allowed, _scale(derivative, Q(-1))))
    rate = gamma**2 / (60 * (1 + gamma))
    scalar_allowed = _sum(
        _scale(metric, -rate),
        _blocks(
            (
                (zero22, zero22, zero23),
                (zero22, zero22, zero23),
                (zero32, zero32, _scale(identity3, 2 * gamma**2)),
            )
        ),
    )
    assert exact_symmetric_semidefinite(_sum(scalar_allowed, _scale(derivative, Q(-1))))
    assert gamma**2 / (4 * (1 + gamma)) == 15 * rate


def test_full_fine_gap_and_projection_mass_are_exact():
    basis = tuple(tuple(Q(i == j) for i in range(45)) for j in range(45))
    laplacian = tuple(zip(*(_laplacian(row) for row in basis)))
    centered_identity = tuple(
        tuple(Q(i == j) - Q(1, 45) for j in range(45)) for i in range(45)
    )
    assert exact_symmetric_semidefinite(
        _sum(laplacian, _scale(centered_identity, Q(-4, 15)))
    )
    assert _laplacian((Q(1),) * 45) == (Q(0),) * 45
    row = _project(tuple(Q((7 * i) % 13 - 6, 4096) for i in range(45)))
    representatives = tuple(row[i] for i in (*range(18), *range(27, 45)))
    masses = (1,) * 9 + (2,) * 9 + (1,) * 18
    assert sum(value**2 for value in row) == sum(
        mass * value**2 for mass, value in zip(masses, representatives)
    )
    assert sum(value**2 for value in row) != sum(value**2 for value in representatives)


def test_actual_nonlinear_secant_keeps_both_error_rows_and_mean_leaf(mp):
    base_x = _project(tuple(Q((5 * i) % 17 - 8, 4096) for i in range(45)))
    base_phase = _project(tuple(Q((7 * i) % 11 - 5, 4096) for i in range(45)))
    p = _center(_project(tuple(Q((3 * i) % 7 - 3, 8192) for i in range(45))))
    q = _center(_project(tuple(Q((11 * i) % 19 - 9, 8192) for i in range(45))))
    a = tuple(Q((3 * j) % 7 - 3, 8192) for j in range(9))
    b = tuple(Q((5 * j) % 11 - 5, 8192) for j in range(9))
    fine = _hidden_lift(a, b, base_x, base_phase)
    core = tuple(
        tuple(value - difference for value, difference in zip(row, delta))
        for row, delta in zip((base_x, base_phase), (p, q))
    )
    assert sum(p) == sum(q) == 0
    assert sum(fine[0]) == sum(core[0]) and sum(fine[1]) == sum(core[1])
    assert _energy(mp, *fine) < mp.mpf(".01") and _energy(mp, *core) < mp.mpf(".01")
    left_angles, right_angles = _edge_angles(mp, base_phase), _edge_angles(mp, core[1])
    secant = tuple(
        (
            (mp.sin(left) - mp.sin(right)) / (left - right)
            if left != right
            else mp.cos(left)
        )
        for left, right in zip(left_angles, right_angles)
    )
    assert all(mp.mpf(".5") < value <= 1 for value in secant)
    # This is an actual sine secant, and it does not commute with L.
    commutator = tuple(
        x - y
        for x, y in zip(
            _laplacian(_laplacian(q, secant)), _laplacian(_laplacian(q), secant)
        )
    )
    assert max(map(abs, commutator)) > mp.mpf("1e-10")
    gamma = 1 / (1023 * mp.pi)
    lp = tuple(_mp(mp, value) for value in _laplacian(p))
    kq = _laplacian(q, secant)
    forcing = _divergence(_edge_forcing(mp, fine[1]))
    expected = (
        tuple(
            (-x - gamma * y + gamma * force) / 4 for x, y, force in zip(lp, kq, forcing)
        ),
        tuple(gamma * value / 4 for value in lp),
    )
    actual = tuple(
        tuple(x - y for x, y in zip(_project(left), right))
        for left, right in zip(_fields(mp, *fine), _fields(mp, *core))
    )
    for actual_row, expected_row in zip(actual, expected):
        assert abs(mp.fsum(actual_row)) < mp.mpf("1e-95")
        for actual_value, expected_value in zip(actual_row, expected_row):
            assert mp.almosteq(actual_value, expected_value, abs_eps=mp.mpf("1e-95"))
    derivative = 2 * mp.fsum(
        (_mp(mp, x) + gamma * _mp(mp, y)) * dx + (_mp(mp, y) + gamma * _mp(mp, x)) * dy
        for x, y, dx, dy in zip(p, q, *actual)
    )
    metric = mp.fsum(
        _mp(mp, x) ** 2 + _mp(mp, y) ** 2 + 2 * gamma * _mp(mp, x * y)
        for x, y in zip(p, q)
    )
    rate = gamma**2 / (60 * (1 + gamma))
    hidden_square = _mp(mp, sum(value**2 for value in b))
    assert derivative <= -rate * metric + 10 * gamma**2 * hidden_square**2


def test_independent_complete_rows_match_every_declared_clock(mp):
    state = _hidden_lift((H,) + (Q(0),) * 8, (Q(0), H) + (Q(0),) * 7)
    report = _observe_organized_source_capacity(
        form=state[0],
        phase_offsets=state[1],
        capacities=UNIT,
        phase_mobility="shared_capacity",
    )
    rows = _fields(mp, *state)
    gamma = 1 / (1023 * mp.pi)
    for clock, factor in (
        ("tau", mp.mpf(1)),
        ("slow", 2 / gamma**2),
        ("original", mp.mpf(1023) / 1024),
    ):
        for name, row in zip(("form", "phase"), rows):
            intervals = getattr(report, f"{name}_{clock}_rate_bounds")
            for interval, value in zip(intervals, row):
                assert _mp(mp, interval.lo) <= factor * value <= _mp(mp, interval.hi)
    assert any(value != 0 for value in rows[1])


@pytest.mark.parametrize(
    "form_shift,phase_shift", ((Q(1, 8), Q(0)), (Q(0), Q(1, 7)), (Q(1, 8), Q(-1, 7)))
)
def test_unmatched_constant_means_exclude_a_decaying_unforced_error_bound(
    mp, form_shift, phase_shift
):
    # Both supplied states are exact equilibria with identical edge data.
    # Their nonzero common-mode error therefore persists without forcing.
    fine = ((Q(0),) * 45, (Q(0),) * 45)
    core = ((form_shift,) * 45, (phase_shift,) * 45)
    assert _laplacian(core[0]) == (Q(0),) * 45
    assert _edge_angles(mp, fine[1]) == _edge_angles(mp, core[1])
    for state in (fine, core):
        assert all(
            abs(value) < mp.mpf("1e-95") for row in _fields(mp, *state) for value in row
        )
    gamma = 1 / (1023 * mp.pi)
    metric = 45 * (
        _mp(mp, form_shift**2 + phase_shift**2)
        + 2 * gamma * _mp(mp, form_shift * phase_shift)
    )
    assert metric > 0
    assert -(gamma**2 / (60 * (1 + gamma))) * metric < 0
    assert sum(core[0]) != sum(fine[0]) or sum(core[1]) != sum(fine[1])
