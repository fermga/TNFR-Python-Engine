"""Independent present-state quotient, observation and initialization controls.

The four boundary readings require both hidden relative coordinates. Exact
symmetry reduction, a smooth local obstruction and a same-member error bound
are separate claims; none supplies current readings from a history packet.
Only immutable pure full-law/oracle helpers are imported from prior controls.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import product, repeat
from math import factorial

import numpy as np
import pytest

from tests.physics.test_sine_history_information import _B, _display, _sine
from tests.physics.test_sine_history_interface import _physical_generator
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics.linear_observation import derive_coordinate_memory
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_flow as full_flow
from tnfr.physics import _sine_present_state as owner

CONTACTS = (None, (0, 1), (0, 3))
GAMMAS = (Q(49, 100), Q(1, 2), Q(51, 100))
VISIBLE = (0, 2, 3, 5)
RHO = Q(1, 10**9)
HORIZON = Q(1, 4)


def _matrix_product(left, right):
    return tuple(
        tuple(sum((a * b for a, b in zip(row, column)), Q(0)) for column in zip(*right))
        for row in left
    )


def _mv(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), Q(0)) for row in matrix)


def _basis(size):
    return tuple(tuple(Q(i == j) for i in range(size)) for j in range(size))


def _projection():
    return tuple(
        tuple(Q(j == i) - Q(j == anchor) for j in range(8))
        for anchor, nodes in ((0, (1, 2, 3)), (4, (5, 6, 7)))
        for i in nodes
    )


def _lift_matrix():
    return tuple(
        (Q(0),) * 6 if i in (0, 4) else _basis(6)[i - 1 if i < 4 else i - 2]
        for i in range(8)
    )


def _project(values):
    return tuple(values[i] - values[0] for i in (1, 2, 3)) + tuple(
        values[i] - values[4] for i in (5, 6, 7)
    )


def _lift(values):
    return (Q(0), *values[:3], Q(0), *values[3:])


def _generator(contact, gamma):
    return _physical_generator(
        (0, 1) if contact is None else contact, gamma, contact is not None
    )


def _relative_generator(contact, gamma):
    return _matrix_product(
        _matrix_product(_projection(), _generator(contact, gamma)), _lift_matrix()
    )


def _oracle_rates(values, contact, gamma):
    """Both physical rows from separate exact edge sums and a rational sine oracle."""
    forms, phases = values[:4], values[4:]
    form, phase = [_B(0) for _ in range(4)], [_B(0) for _ in range(4)]
    for i, j in ((1, 2), (2, 3)) + (() if contact is None else (contact,)):
        difference = forms[j] - forms[i]
        current = _sine(phases[j] - phases[i])
        form_edge = (difference + gamma * current) / 4
        phase_edge = _B(-gamma * difference / 4)
        form[i], form[j] = form[i] + form_edge, form[j] - form_edge
        phase[i], phase[j] = phase[i] + phase_edge, phase[j] - phase_edge
    return tuple(form + phase)


def _contains(actual, expected, width=Q(1, 10**30)):
    expected = expected if isinstance(expected, _B) else _B(expected)
    assert actual.lo <= expected.lo <= expected.hi <= actual.hi
    assert actual.hi - actual.lo < width


def _cosine(value):
    value = abs(Q(value))
    assert value <= 1
    upper = sum(
        ((-1) ** k * value ** (2 * k) / factorial(2 * k) for k in range(51)), Q(0)
    )
    return _B(upper - value**102 / factorial(102), upper)


def _exponential(value):
    assert 0 <= value <= 1
    lower = sum((value**k / factorial(k) for k in range(51)), Q(0))
    return _B(lower, lower + value**51 / factorial(51) / (1 - value / 52))


@pytest.fixture(scope="module", autouse=True)
def no_nonlinear_trajectory_event_or_retained_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("present-state controls cannot integrate or replay a source")

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
def observation():
    return owner._prepared_sine_present_observation()


def test_boundary_observation_has_six_dimensional_exact_tangent_closure(
    observation, record_property
):
    relative, lift = _projection(), _lift_matrix()
    output = tuple(relative[i] for i in VISIBLE)
    anchor = observation.anchor_observation
    assert observation.relative_projection == relative
    assert observation.relative_lift == lift
    assert observation.boundary_observation == output
    assert observation.output_from_relative == tuple(_basis(6)[i] for i in VISIBLE)
    assert anchor.generator == _generator(None, Q(1, 2))
    assert anchor.output_rows == output
    assert anchor.output_rank == 4 and anchor.dimension == 6
    assert anchor.rank_progression == (4, 6, 6)
    assert _matrix_product(relative, lift) == _basis(6)
    assert observation.relative_to_anchor == _matrix_product(anchor.observation, lift)
    assert observation.anchor_to_relative == _matrix_product(
        relative, anchor.right_inverse
    )
    assert _matrix_product(
        observation.relative_to_anchor, observation.anchor_to_relative
    ) == _basis(6)
    assert _matrix_product(
        observation.anchor_to_relative, observation.relative_to_anchor
    ) == _basis(6)
    # Exactly the two common channel shifts are discarded by these six rows.
    assert _mv(relative, (1, 1, 1, 1, 0, 0, 0, 0)) == (0,) * 6
    assert _mv(relative, (0, 0, 0, 0, 1, 1, 1, 1)) == (0,) * 6
    record_property("present_output_dimension", "4")
    record_property("exact_relative_state_dimension", "6")
    record_property("rank_progression", "4,6,6")
    with pytest.raises(FrozenInstanceError):
        observation.dimension = 4


def test_first_boundary_derivatives_reconstruct_both_hidden_relative_coordinates():
    for gamma in GAMMAS:
        matrix = _relative_generator(None, gamma)
        for state in (
            *_basis(6),
            (Q(1, 3), Q(-2, 7), Q(3, 5), Q(-1, 11), Q(4, 13), Q(2, 9)),
        ):
            rate = _mv(matrix, state)
            a2 = state[0] - 4 * rate[3] / gamma
            b2 = state[3] + 4 * rate[0] / gamma + 4 * rate[3] / gamma**2
            assert (a2, b2) == (state[1], state[4])


def test_common_closure_is_an_affine_polynomial_identity_not_endpoint_sampling(
    observation,
):
    relative, lift = _projection(), _lift_matrix()
    c, t = (
        observation.anchor_observation.observation,
        observation.anchor_observation.right_inverse,
    )
    labels = tuple(
        (contact, gamma) for contact in CONTACTS for gamma in (GAMMAS[0], GAMMAS[-1])
    )
    assert observation.family_labels == labels
    for contact in CONTACTS:
        constant = _generator(contact, Q(0))
        at_one = _generator(contact, Q(1))
        linear = tuple(
            tuple(b - a for a, b in zip(left, right))
            for left, right in zip(constant, at_one)
        )
        # Coefficientwise equality proves the identity for every real gamma.
        for coefficient in (constant, linear):
            reduced = _matrix_product(_matrix_product(relative, coefficient), lift)
            anchor_reduced = _matrix_product(_matrix_product(c, coefficient), t)
            assert _matrix_product(relative, coefficient) == _matrix_product(
                reduced, relative
            )
            assert _matrix_product(c, coefficient) == _matrix_product(anchor_reduced, c)
    for label, full, reduced, anchor_reduced in zip(
        labels,
        observation.full_generators,
        observation.relative_generators,
        observation.anchor_reduced_generators,
    ):
        contact, gamma = label
        assert full == _generator(contact, gamma)
        assert reduced == _relative_generator(contact, gamma)
        assert anchor_reduced == _matrix_product(_matrix_product(c, full), t)


def test_visible_memory_retains_both_hidden_channels_and_signed_kernel():
    gamma = Q(1, 2)
    result = derive_coordinate_memory(_relative_generator(None, gamma), VISIBLE)
    b = ((Q(1, 4), gamma / 4),) * 2 + ((-gamma / 4, Q(0)),) * 2
    c = ((Q(1, 4), Q(1, 4), gamma / 4, gamma / 4), (-gamma / 4, -gamma / 4, Q(0), Q(0)))
    d = ((Q(-1, 2), -gamma / 2), (gamma / 2, Q(0)))
    kernel = (
        ((1 - gamma**2) / 16, (1 - gamma**2) / 16, gamma / 16, gamma / 16),
    ) * 2 + (
        (-gamma / 16, -gamma / 16, -(gamma**2) / 16, -(gamma**2) / 16),
    ) * 2
    assert result.hidden_indices == (1, 4)
    assert result.hidden_to_visible == b
    assert result.visible_to_hidden == c
    assert result.hidden_generator == d
    assert result.kernel_at_zero == kernel
    hidden_initial = (Q(2, 7), Q(-3, 11))
    full = (0, hidden_initial[0], 0, 0, hidden_initial[1], 0)
    assert tuple(_mv(result.generator, full)[i] for i in VISIBLE) == _mv(
        b, hidden_initial
    )
    assert kernel[2][0] < 0 < kernel[0][0]


@pytest.mark.parametrize(
    "hidden_coordinate", (1, 4), ids=("hidden-form", "hidden-phase")
)
def test_equal_four_readings_have_different_actual_instantaneous_rates(
    hidden_coordinate,
):
    # These are full-law points, not evaluated witnesses from the earlier
    # source. The chapter separately proves an open acute reached chart.
    gamma, delta = Q(1, 2), Q(1, 100)
    field = owner._held_sine_present_field(contact=None, gamma=gamma)
    first, second = (Q(0),) * 6, [Q(0)] * 6
    second[hidden_coordinate] = delta
    assert tuple(first[i] for i in VISIBLE) == tuple(second[i] for i in VISIBLE)
    difference = tuple(b - a for a, b in zip(field(first), field(second)))
    if hidden_coordinate == 1:
        _contains(difference[0], delta / 4)
        _contains(difference[3], -gamma * delta / 4)
        assert difference[0].lo > 0 > difference[3].hi
    else:
        _contains(difference[0], gamma * _sine(delta) / 4)
        _contains(difference[3], 0)
        assert difference[0].lo > 0


def test_actual_derivative_map_has_nonzero_hidden_minor_in_an_acute_chart(
    record_property,
):
    gamma = Q(1, 2)
    state = (Q(1, 8), Q(-1, 16), Q(3, 32), Q(1, 16), Q(1, 8), Q(-1, 32))
    field = owner._held_sine_present_field(contact=None, gamma=gamma)
    columns = []
    for hidden in (1, 4):
        jets = tuple(Jet((value, Q(i == hidden))) for i, value in enumerate(state))
        derivative = field(jets)
        columns.append((derivative[3].coeffs[1], derivative[0].coeffs[1]))
    _contains(columns[0][0], -gamma / 4)
    _contains(columns[0][1], Q(1, 4))
    _contains(columns[1][0], 0)
    cosine = _cosine(state[4] - state[3])
    _contains(columns[1][1], gamma * cosine / 4)
    determinant = columns[0][0] * columns[1][1] - columns[1][0] * columns[0][1]
    expected = -(gamma**2) * cosine / 16
    _contains(determinant, expected)
    assert determinant.hi < 0 and cosine.lo > 0
    # This generic acute point is not asserted to be a previous source reach.
    # Identity on the four visible rows and this 2x2 hidden minor gives a
    # rank-six derivative map. This is a local smooth-realization argument,
    # distinct from tangent rank or an arbitrary finite-precision encoding.
    record_property("actual_hidden_minor_upper", _display(determinant.hi))


@pytest.mark.parametrize("contact", CONTACTS, ids=("old", "port01", "port03"))
def test_exact_quotient_matches_independent_full_rows_and_moving_reference(contact):
    gamma = Q(51, 100)
    state = (
        Q(1, 8),
        Q(-1, 16),
        Q(3, 16),
        Q(1, 32),
        Q(1, 16),
        Q(1, 8),
        Q(-1, 8),
        Q(1, 32),
    )
    relative = _project(state)
    actual = owner._held_sine_present_field(contact=contact, gamma=gamma)(relative)
    full = _oracle_rates(state, contact, gamma)
    expected = _project(full)
    for value, oracle in zip(actual, expected):
        _contains(value, oracle)
    if contact is not None:
        assert full[0].hi < 0 or full[0].lo > 0
        assert full[4].hi < 0 or full[4].lo > 0
        # Taking only non-anchor rows would silently freeze a moving reference.
        assert expected[0].hi < full[1].lo or full[1].hi < expected[0].lo
        assert expected[3].hi < full[5].lo or full[5].hi < expected[3].lo


def test_uniform_gauges_and_state_continuous_switches_preserve_current_coordinates():
    state = (
        Q(1, 8),
        Q(-1, 16),
        Q(3, 16),
        Q(1, 32),
        Q(1, 16),
        Q(1, 8),
        Q(-1, 8),
        Q(1, 32),
    )
    shifted = tuple(
        value + (Q(7, 4) if i < 4 else Q(-5, 8)) for i, value in enumerate(state)
    )
    relative = _project(state)
    assert relative == _project(shifted)
    for contact in CONTACTS:
        assert _project(_oracle_rates(state, contact, Q(1, 2))) == _project(
            _oracle_rates(shifted, contact, Q(1, 2))
        )
        # Changing a prescribed support changes the field, not its input state.
        field = owner._held_sine_present_field(contact=contact, gamma=Q(1, 2))
        before = tuple(relative)
        field(relative)
        assert relative == before == _project(state)
    # Equal common form means reconstruct receiver excess from relative forms.
    on, off, mean = (
        (Q(1, 9), Q(-1, 7), Q(2, 11)),
        (Q(-1, 5), Q(1, 13), Q(3, 17)),
        Q(7, 11),
    )
    on0, off0 = mean - sum(on) / 4, mean - sum(off) / 4
    assert on0 - off0 == -sum(a - b for a, b in zip(on, off)) / 4
    assert sum((on0, *(on0 + value for value in on))) == 4 * mean
    assert sum((off0, *(off0 + value for value in off))) == 4 * mean


def test_full_joint_tangent_is_the_jet_derivative_of_the_actual_quotient():
    for contact in CONTACTS:
        field = owner._held_sine_present_field(contact=contact, gamma=Q(1, 2))
        matrix = _relative_generator(contact, Q(1, 2))
        for direction in _basis(6):
            result = field(tuple(Jet((0, value)) for value in direction))
            for value, exact in zip(result, _mv(matrix, direction)):
                _contains(value.coeffs[0], 0)
                _contains(value.coeffs[1], exact)


def test_unknown_common_gamma_remains_a_law_parameter_not_a_point_forecast():
    state = (Q(0), Q(1, 10), Q(0), Q(0), Q(0), Q(0))
    lower = owner._held_sine_present_field(contact=None, gamma=GAMMAS[0])(state)
    upper = owner._held_sine_present_field(contact=None, gamma=GAMMAS[-1])(state)
    difference = upper[3] - lower[3]
    _contains(difference, -(GAMMAS[-1] - GAMMAS[0]) * state[1] / 4)
    assert difference.hi < 0


def test_global_lipschitz_bound_and_fixed_same_member_initialization_lifetime(
    record_property,
):
    gamma = GAMMAS[-1]
    # The nonlinear Jacobian has cos(edge gap) in [-1,1]. Its row norm is
    # convex in each cosine, so independent cube corners bound the whole
    # phase domain, including mutually incompatible edge cosines.
    for contact in CONTACTS:
        edges = ((1, 2), (2, 3)) + (() if contact is None else (contact,))
        for cosines in product((-1, 1), repeat=len(edges)):
            jacobian = [[Q(0)] * 8 for _ in range(8)]
            for (i, j), cosine in zip(edges, cosines):
                for a, b in ((i, j), (j, i)):
                    jacobian[a][a] -= Q(1, 4)
                    jacobian[a][b] += Q(1, 4)
                    jacobian[a][a + 4] -= gamma * cosine / 4
                    jacobian[a][b + 4] += gamma * cosine / 4
                    jacobian[a + 4][a] += gamma / 4
                    jacobian[a + 4][b] -= gamma / 4
            assert max(sum(map(abs, row)) for row in jacobian) <= 1 + gamma
            relative = _matrix_product(
                _matrix_product(_projection(), jacobian), _lift_matrix()
            )
            assert max(sum(map(abs, row)) for row in relative) <= 2 * (1 + gamma)
    result = owner._bound_sine_present_initialization(
        initial_error_bound=RHO, elapsed_time=HORIZON
    )
    assert result.lipschitz_bound == Q(151, 50)
    exact = RHO * _exponential(Q(151, 200))
    assert exact.hi <= result.state_error_upper_bound
    assert result.state_error_upper_bound - exact.lo < Q(1, 10**30)
    assert (
        result.matched_receiver_error_upper_bound
        == Q(3, 2) * result.state_error_upper_bound
    )
    assert result.matched_receiver_error_upper_bound < Q(9, 2) * RHO < Q(5, 10**9)
    record_property("initial_radius", str(RHO))
    record_property(
        "same_member_state_error_upper", _display(result.state_error_upper_bound)
    )
    record_property(
        "same_member_matched_receiver_error_upper",
        _display(result.matched_receiver_error_upper_bound),
    )
    record_property("fixed_error_target", "5e-9")


def test_zero_elapsed_time_retains_initial_uncertainty_and_exact_zero_radius():
    initial = owner._bound_sine_present_initialization(
        initial_error_bound=RHO, elapsed_time=0
    )
    assert initial.state_error_upper_bound == RHO
    assert initial.matched_receiver_error_upper_bound == Q(3, 2) * RHO
    zero = owner._bound_sine_present_initialization(
        initial_error_bound=0, elapsed_time=HORIZON
    )
    assert zero.state_error_upper_bound == zero.matched_receiver_error_upper_bound == 0


def test_invalid_field_primitives_reject_before_full_field_compilation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid present-field primitives reached full-field work")

    monkeypatch.setattr(owner, "_held_quarter_sine_field", forbidden)
    invalid = (
        {"contact": (0, True)},
        {"contact": (1, 0)},
        {"contact": (0, 2)},
        {"contact": (0, 1, 3)},
        {"contact": repeat(0)},
        {"contact": {0, 1}},
        {"gamma": True},
        {"gamma": float("nan")},
        {"gamma": float("inf")},
        {"gamma": 1j},
        {"gamma": Q(49, 100) - Q(1, 10**100)},
        {"gamma": Q(51, 100) + Q(1, 10**100)},
    )
    for changes in invalid:
        arguments = {"contact": None, "gamma": Q(1, 2)}
        arguments.update(changes)
        with pytest.raises((TypeError, ValueError)):
            owner._held_sine_present_field(**arguments)


def test_six_coordinate_state_admission_precedes_full_field_and_never_decodes_history(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid present coordinates reached full-field arithmetic")

    invalid_shape = (
        (),
        (5000, 10000),
        (5000, 10000, 1),
        (0,) * 8,
        repeat(0),
        {0, 1},
        (Jet.constant(0, 1), 0, 0, 0, 0, 0),
        (Jet.constant(0, 1),) * 5 + (Jet.constant(0, 2),),
    )
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_held_quarter_sine_field", lambda **kwargs: forbidden)
        field = owner._held_sine_present_field(contact=(0, 1), gamma=Q(1, 2))
        for state in invalid_shape:
            with pytest.raises((TypeError, ValueError)):
                field(state)
    # Scalar admission belongs to the existing full-field owner. Exercise it
    # through the new quotient before any trigonometric numerical work.
    field = owner._held_sine_present_field(contact=(0, 1), gamma=Q(1, 2))
    monkeypatch.setattr(full_flow, "sin", forbidden)
    for bad in (True, float("nan"), float("inf"), 1j):
        with pytest.raises((TypeError, ValueError)):
            field((bad, 0, 0, 0, 0, 0))


def test_six_coordinate_states_accept_intervals_and_one_shot_ordered_inputs():
    field = owner._held_sine_present_field(contact=(0, 3), gamma=Q(1, 2))
    lower, upper = Q(-1, 100), Q(1, 100)
    result = field(I(lower, upper) for _ in range(6))
    for corner in product((lower, upper), repeat=6):
        oracle = _project(_oracle_rates(_lift(corner), (0, 3), Q(1, 2)))
        for actual, expected in zip(result, oracle):
            assert actual.lo <= expected.lo <= expected.hi <= actual.hi


def test_fixed_width_integer_fraction_components_normalize_before_arithmetic():
    def fixed_fraction(numerator, denominator):
        return Q(np.int64(numerator), np.int64(denominator))

    gamma = fixed_fraction(51, 100)
    state = (Q(0), Q(1, 32), Q(-1, 64), Q(1, 16), Q(-1, 32), Q(0))
    fixed_state = tuple(
        fixed_fraction(value.numerator, value.denominator) for value in state
    )
    ordinary = owner._held_sine_present_field(contact=(0, 3), gamma=Q(51, 100))(state)
    fixed = owner._held_sine_present_field(contact=(0, 3), gamma=gamma)(fixed_state)
    assert fixed == ordinary
    result = owner._bound_sine_present_initialization(
        initial_error_bound=fixed_fraction(1, 10**9),
        elapsed_time=fixed_fraction(1, 4),
    )
    ordinary_bound = owner._bound_sine_present_initialization(
        initial_error_bound=RHO, elapsed_time=HORIZON
    )
    assert result == ordinary_bound
    for value in (result.initial_error_bound, result.elapsed_time):
        assert type(value.numerator) is type(value.denominator) is int


def test_invalid_initialization_primitives_reject_before_scalar_bound(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid initialization data reached exponential work")

    monkeypatch.setattr(owner, "scalar_affine_upper", forbidden)
    for arguments in (
        {"initial_error_bound": True, "elapsed_time": 0},
        {"initial_error_bound": -Q(1, 10**100), "elapsed_time": 0},
        {"initial_error_bound": float("nan"), "elapsed_time": 0},
        {"initial_error_bound": 1j, "elapsed_time": 0},
        {"initial_error_bound": 0, "elapsed_time": True},
        {"initial_error_bound": 0, "elapsed_time": -Q(1, 10**100)},
        {"initial_error_bound": 0, "elapsed_time": HORIZON + Q(1, 10**100)},
        {"initial_error_bound": 0, "elapsed_time": float("inf")},
    ):
        with pytest.raises((TypeError, ValueError)):
            owner._bound_sine_present_initialization(**arguments)


def test_invalid_observation_budget_rejects_before_matrix_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid observation budget reached matrix work")

    monkeypatch.setattr(owner, "exact_matrix_product", forbidden)
    monkeypatch.setattr(owner, "_prepared_present_support_matrices", forbidden)
    for budget in (True, False, 0, -1, Q(10), "10", None):
        with pytest.raises((TypeError, ValueError)):
            owner._prepared_sine_present_observation(max_rank_calls=budget)
