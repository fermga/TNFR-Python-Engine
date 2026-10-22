"""Exact positive-sign model controls, independent of nodal diffusion admission."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import pytest

from tnfr.mathematics._exact_linear_algebra import exact_matrix_product
from tnfr.mathematics.linear_observation import (
    derive_coordinate_memory,
    derive_linear_observation,
)


def test_damped_exchange_needs_hidden_phase_and_retains_positive_generator_sign():
    # x'=-x-theta, theta'=x; observing x alone loses its initial derivative.
    result = derive_linear_observation(((-1, -1), (1, 0)), ((1, 0),))
    assert result.output_rank == 1 and result.dimension == 2
    assert result.rank_progression == (1, 2, 2)
    assert result.observation == ((1, 0), (-1, -1))
    assert result.output_map == ((1, 0),)
    # The minimal state is (x, x'), with x''=-x-x'.
    assert result.reduced_generator == ((0, 1), (-1, -1))
    assert result.right_inverse == ((1, 0), (-1, -1))
    assert "C J=G C" in result.exact_identity_checks
    with pytest.raises(FrozenInstanceError):
        result.dimension = 1


def test_nilpotent_chain_requires_complete_last_level_without_stability_premise():
    result = derive_linear_observation(((0, 1, 0), (0, 0, 1), (0, 0, 0)), ((1, 0, 0),))
    assert result.observation == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert result.reduced_generator == result.generator
    assert result.rank_progression == (1, 2, 3, 3)
    assert result.selected_row_labels == ((0, 0), (1, 0), (2, 0))
    assert result.stabilization_power == 3
    assert result.rank_calls == 8
    assert result.matrix_product_calls == 9


@pytest.mark.parametrize(
    "third_rows",
    (
        ((0,) * 6,) * 2,
        ((1, -2, Q(3, 7), 0, Q(1, 5), -1), (0, Q(5, 11), -2, Q(1, 3), 0, 2)),
    ),
    ids=("held-third-block", "coupled-third-block"),
)
def test_first_two_source_rows_reveal_all_three_state_blocks(third_rows):
    """Check the first jets, not a global linearization of a nonlinear source.

    For y'=-a*u+M*y and u'=-d*u-e*y-f*w, y, y', y'' do not
    consume w'. Changing that third row must preserve their information.
    """
    a, d, e, f = Q(3, 2), Q(2, 3), Q(5, 6), Q(7, 4)
    coupling = ((Q(-9, 35), Q(2, 5)), (Q(2, 5), Q(-9, 35)))
    generator = (
        (*coupling[0], -a, 0, 0, 0),
        (*coupling[1], 0, -a, 0, 0),
        (-e, 0, -d, 0, -f, 0),
        (0, -e, 0, -d, 0, -f),
        *third_rows,
    )
    result = derive_linear_observation(
        generator, ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0))
    )
    assert result.output_rank == 2 and result.dimension == 6
    assert result.rank_progression == (2, 4, 6, 6)
    assert result.selected_row_labels == tuple(
        (power, node) for power in range(3) for node in range(2)
    )

    def mv(matrix, vector):
        return tuple(
            sum(entry * value for entry, value in zip(row, vector)) for row in matrix
        )

    # A basis checks the whole linear observation map; a mixed state also
    # exercises reconstruction with both spatial coupling terms present.
    mixed = (Q(2, 3), Q(-5, 7), Q(11, 13), Q(-3, 4), Q(-4, 9), Q(7, 5))
    basis = tuple(tuple(Q(i == j) for i in range(6)) for j in range(6))
    for state in (*basis, mixed):
        y, u, w = state[:2], state[2:4], state[4:]
        my = mv(coupling, y)
        dy = tuple(my[i] - a * u[i] for i in range(2))
        du = tuple(-d * u[i] - e * y[i] - f * w[i] for i in range(2))
        mdy = mv(coupling, dy)
        ddy = tuple(mdy[i] - a * du[i] for i in range(2))
        jet = mv(result.observation, state)
        assert jet == (*y, *dy, *ddy)

        read_y, read_dy, read_ddy = jet[:2], jet[2:4], jet[4:]
        read_my, read_mdy = mv(coupling, read_y), mv(coupling, read_dy)
        recovered_u = tuple((read_my[i] - read_dy[i]) / a for i in range(2))
        recovered_w = tuple(
            (read_ddy[i] - read_mdy[i] - a * d * recovered_u[i] - a * e * read_y[i])
            / (a * f)
            for i in range(2)
        )
        assert (*read_y, *recovered_u, *recovered_w) == state
        assert mv(result.right_inverse, jet) == state

    # Equal readings can hide u; equal readings and first rates can hide w.
    base_jet = mv(result.observation, mixed)
    for block, multiplier in ((1, -a), (2, a * f)):
        change = (Q(4, 5), Q(-3, 2))
        altered = list(mixed)
        for node in range(2):
            altered[2 * block + node] += change[node]
        altered_jet = mv(result.observation, altered)
        assert altered_jet[: 2 * block] == base_jet[: 2 * block]
        assert tuple(
            altered_jet[2 * block + node] - base_jet[2 * block + node]
            for node in range(2)
        ) == tuple(multiplier * value for value in change)


def test_unobservable_unstable_direction_does_not_enter_the_minimal_output_state():
    result = derive_linear_observation(((-2, 0), (0, 7)), ((1, 0), (2, 0)))
    assert result.dimension == result.output_rank == 1
    assert result.reduced_generator == ((-2,),)
    assert result.output_map == ((1,), (2,))
    assert result.right_inverse == ((1,), (0,))


def test_all_zero_output_has_empty_state_and_no_inverse():
    result = derive_linear_observation(((1, 2), (3, 4)), ((0, 0),))
    assert result.dimension == result.output_rank == 0
    assert result.observation == result.reduced_generator == ()
    assert result.right_inverse == ((), ()) and result.output_map == ((),)
    assert result.rank_progression == (0, 0)
    assert result.stabilization_power == 1


def test_tiny_rational_coupling_is_retained_without_binary64_materialization():
    tiny = Q(1, 10**400)
    result = derive_linear_observation(((0, tiny), (0, 0)), ((1, 0),))
    assert result.generator[0][1] == tiny
    assert result.observation == ((1, 0), (0, tiny))
    assert result.dimension == 2
    assert result.right_inverse == ((1, 0), (0, 1 / tiny))
    assert result.max_coefficient_bits >= tiny.denominator.bit_length()


def test_rank_budget_is_exhaustive_and_exact_budget_is_sufficient():
    j, o = ((-1, -1), (1, 0)), ((1, 0),)
    complete = derive_linear_observation(j, o)
    with pytest.raises(ValueError, match="budget"):
        derive_linear_observation(j, o, max_rank_calls=complete.rank_calls - 1)
    exact = derive_linear_observation(j, o, max_rank_calls=complete.rank_calls)
    assert exact.observation == complete.observation
    assert exact.rank_calls == complete.rank_calls


@pytest.mark.parametrize("limit", (True, False, 0, -1, 1.5, "10", None, Q(10)))
def test_invalid_rank_budget_rejects_before_algebra(limit):
    with pytest.raises((TypeError, ValueError)):
        derive_linear_observation(((1,),), ((1,),), max_rank_calls=limit)


@pytest.mark.parametrize(
    "generator, output",
    (
        ((), ((1,),)),
        (((1, 2),), ((1,),)),
        (((1,),), ()),
        (((1,),), ((1, 2),)),
        (((1,),), ((),)),
        (((True,),), ((1,),)),
        (((1,),), ((False,),)),
        (((float("inf"),),), ((1,),)),
        (((1,),), ((float("nan"),),)),
        (((1 + 1j,),), ((1,),)),
        ({(1,)}, ((1,),)),
        (((1,),), {(1,)}),
        (((1,),), {"output": (1,)}),
        (((1,),), ("1",)),
    ),
)
def test_invalid_matrix_admission(generator, output):
    with pytest.raises((TypeError, ValueError)):
        derive_linear_observation(generator, output)


def test_real_input_is_its_represented_rational_not_an_inferred_exact_constant():
    result = derive_linear_observation(((0.1,),), ((1.0,),))
    assert result.generator == ((Q(0.1),),)
    assert result.generator != ((Q(1, 10),),)


def test_coordinate_memory_of_normalized_p3_retains_mediator_and_initial_source():
    nu = Q(3, 2)
    generator = ((-nu, nu, 0), (nu / 2, -nu, nu / 2), (0, nu, -nu))
    result = derive_coordinate_memory(generator, (0, 2))
    assert result.visible_indices == (0, 2) and result.hidden_indices == (1,)
    assert result.visible_generator == ((-nu, 0), (0, -nu))
    assert result.hidden_to_visible == ((nu,), (nu,))
    assert result.visible_to_hidden == ((nu / 2, nu / 2),)
    assert result.hidden_generator == ((-nu,),)
    assert result.kernel_at_zero == ((nu**2 / 2,) * 2,) * 2

    # A mediator-only initial difference changes both endpoint rates even
    # with y0=0; dropping the initial hidden term would erase this response.
    hidden_initial = Q(2, 3)
    full_rate = tuple(row[1] * hidden_initial for row in result.generator)
    assert (full_rate[0], full_rate[2]) == (nu * hidden_initial,) * 2
    assert (
        tuple(row[0] * hidden_initial for row in result.hidden_to_visible)
        == (nu * hidden_initial,) * 2
    )
    # A donor-only difference has no direct recipient rate, but its second
    # derivative sees the two-edge path. No exponential or trajectory needed.
    squared = exact_matrix_product(result.generator, result.generator)
    assert result.generator[2][0] == 0
    assert squared[2][0] == nu**2 / 2 == result.kernel_at_zero[1][0]


def test_damped_joint_coordinate_memory_has_signed_kernel_and_hidden_source():
    # x'=-x-theta, theta'=x-2theta gives K(t)=-exp(-2t), not a
    # nonnegative diffusion kernel. The initial source is -exp(-2t)*theta0.
    result = derive_coordinate_memory(((-1, -1), (1, -2)), (0,))
    assert result.visible_generator == ((-1,),)
    assert result.hidden_to_visible == ((-1,),)
    assert result.visible_to_hidden == ((1,),)
    assert result.hidden_generator == ((-2,),)
    assert result.kernel_at_zero == ((-1,),)
    assert result.hidden_to_visible[0][0] * Q(3, 2) == Q(-3, 2)


def test_zero_instantaneous_kernel_does_not_remove_later_memory():
    # y'=h1, h1'=h2, h2'=y. D^2=0 gives exp(Dt)=I+tD, so
    # K(0)=BC=0 while K(t)=t*BDC=t is nonzero for positive time.
    result = derive_coordinate_memory(((0, 1, 0), (0, 0, 1), (1, 0, 0)), (0,))
    assert result.hidden_indices == (1, 2)
    assert result.kernel_at_zero == ((0,),)
    assert exact_matrix_product(result.hidden_generator, result.hidden_generator) == (
        (0, 0),
        (0, 0),
    )
    first_kernel_derivative = exact_matrix_product(
        exact_matrix_product(result.hidden_to_visible, result.hidden_generator),
        result.visible_to_hidden,
    )
    assert first_kernel_derivative == ((1,),)


def test_hidden_instability_and_initial_source_are_not_silently_excluded():
    result = derive_coordinate_memory(((-1, 2), (0, 7)), (0,))
    assert result.hidden_generator == ((7,),)
    assert result.kernel_at_zero == ((0,),)
    # C=0 makes the kernel vanish, but the source 2*exp(7t)*h0 remains.
    assert result.visible_to_hidden == ((0,),)
    assert result.hidden_to_visible == ((2,),)


def test_coordinate_memory_reorders_coordinates_and_detaches_one_shot_inputs():
    generator = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    visible = [2, 0]
    result = derive_coordinate_memory((iter(row) for row in generator), iter(visible))
    assert generator == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert visible == [2, 0]
    assert result.visible_indices == (2, 0) and result.hidden_indices == (1,)
    assert result.visible_generator == ((9, 7), (3, 1))
    assert result.hidden_to_visible == ((8,), (2,))
    assert result.visible_to_hidden == ((6, 4),)
    assert result.hidden_generator == ((5,),)
    assert result.kernel_at_zero == ((48, 32), (12, 8))
    generator[2][0] = 999
    visible.reverse()
    assert result.generator == ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    assert result.visible_indices == (2, 0)
    with pytest.raises(FrozenInstanceError):
        result.visible_indices = (0, 2)


def test_coordinate_memory_keeps_tiny_rational_and_represented_coefficients():
    tiny = Q(1, 10**400)
    result = derive_coordinate_memory(((0, tiny), (0.1, 0)), (0,))
    assert result.hidden_to_visible == ((tiny,),)
    assert result.visible_to_hidden == ((Q(0.1),),)
    assert result.kernel_at_zero == ((tiny * Q(0.1),),)
    assert result.kernel_at_zero[0][0] != 0
    assert result.kernel_at_zero[0][0] != tiny / 10


@pytest.mark.parametrize(
    "visible",
    (
        (),
        (0, 1, 2),
        (0, 0),
        (-1,),
        (3,),
        (True,),
        (0.0,),
        (Q(0),),
        "0",
        {0},
        {0: 1},
        None,
    ),
)
def test_coordinate_memory_rejects_ambiguous_or_invalid_index_selection(visible):
    with pytest.raises((TypeError, ValueError)):
        derive_coordinate_memory(((0, 1, 0), (0, 0, 1), (1, 0, 0)), visible)


@pytest.mark.parametrize(
    "generator",
    (
        (),
        ((1,),),
        ((1, 2), (3,)),
        ((1, 2), (3, True)),
        ((1, 2), (3, float("nan"))),
        ((1, 2), (3, float("inf"))),
        ((1, 2), (3, 1j)),
        ((1, 2), (3, "4")),
        {(1, 2), (3, 4)},
        {"rows": ((1, 2), (3, 4))},
        ((1, 2), {3, 4}),
    ),
)
def test_coordinate_memory_validates_the_full_raw_generator(generator):
    with pytest.raises((TypeError, ValueError)):
        derive_coordinate_memory(generator, (0,))


def test_coordinate_memory_accepts_numpy_integer_indices_but_not_boolean_indices():
    np = pytest.importorskip("numpy")
    result = derive_coordinate_memory(((0, 1), (1, 0)), (np.int64(0),))
    assert result.visible_indices == (0,)
    assert type(result.visible_indices[0]) is int
    with pytest.raises(TypeError, match="non-boolean integers"):
        derive_coordinate_memory(((0, 1), (1, 0)), (np.bool_(False),))


def test_coordinate_memory_rejects_real_underflow_before_zero_substitution():
    class UnderflowingReal(float):
        def __float__(self):
            return 0.0

    with pytest.raises(ValueError, match="underflows"):
        derive_coordinate_memory(((0, 1), (UnderflowingReal(1.0), 0)), (0,))
