"""Exact positive-sign model controls, independent of nodal diffusion admission."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import pytest

from tnfr.mathematics.linear_observation import derive_linear_observation


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
