"""Independent ideal-output controls for the supplied chemical comparator."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import pytest
import sympy as s

from tnfr.research.chemical_observation import (
    ChemicalObservationModel,
    ChemicalOutputJet,
    ChemicalState,
    chemical_output_jet,
    reconstruct_chemical_state,
)


def model(matrix=((0,),), **kwargs):
    parameters = dict(
        epsilon_u=Q(3, 2), epsilon_w=Q(5, 7), q=Q(2, 5), f=Q(4, 3), delta=Q(7, 6)
    )
    parameters.update(kwargs)
    return ChemicalObservationModel(matrix, **parameters)


@pytest.fixture(scope="module")
def symbolic():
    """Differentiate the complete source, independently of the jet kernel."""
    u = s.Matrix(s.symbols("u0:2"))
    v = s.Matrix(s.symbols("v0:2"))
    w = s.Matrix(s.symbols("w0:2"))
    phi = s.Matrix(s.symbols("phi0:2"))
    eu, ew, q, f, delta = s.symbols("eu ew q f delta", positive=True)
    laplacian = s.Matrix([[-1, 1], [1, -1]])
    state = s.Matrix.vstack(v, u, w)
    du = (
        s.Matrix([(u[i] - u[i] ** 2 + w[i] * (q - u[i])) / eu for i in range(2)])
        + laplacian * u
    )
    dw = (
        s.Matrix([(phi[i] + f * v[i] - w[i] * (q + u[i])) / ew for i in range(2)])
        + delta * laplacian * w
    )
    field = s.Matrix.vstack(u - v, du, dw)
    jets = [v]
    for _ in range(3):
        jets.append(jets[-1].jacobian(state) * field)
    return state, phi, (eu, ew, q, f, delta), jets


@pytest.mark.parametrize("u", [(Q(1, 5), Q(3, 5)), (Q(2, 5), Q(2, 5)), (0, 0)])
def test_full_source_symbolic_jets_and_inverse(symbolic, u):
    state, phi, parameters, expressions = symbolic
    source = model(((-1, 1), (1, -1)))
    v, w, light = (Q(1, 7), Q(2, 7)), (Q(3, 8), Q(5, 8)), (Q(1, 9), Q(2, 9))
    substitutions = dict(zip(state, (*v, *u, *w)))
    substitutions.update(zip(phi, light))
    substitutions.update(
        zip(
            parameters,
            (source.epsilon_u, source.epsilon_w, source.q, source.f, source.delta),
        )
    )
    expected = tuple(
        tuple(Q(value.subs(substitutions)) for value in row) for row in expressions
    )
    jet = chemical_output_jet(source, u, v, w, light)
    assert (jet.y, jet.y1, jet.y2, jet.y3) == expected
    recovered = reconstruct_chemical_state(source, jet.y, jet.y1, jet.y2, y3=jet.y3)
    assert recovered == ChemicalState(u, v, w)


def test_symbolic_regular_rank_is_not_a_sampled_linearization(symbolic):
    state, _, parameters, jets = symbolic
    eu, _, q, _, _ = parameters
    observation = s.Matrix.vstack(*jets[:3])
    determinant = observation.jacobian(state).det()
    assert s.factor(determinant - (q - state[2]) * (q - state[3]) / eu**2) == 0


def test_snapshot_and_first_jet_hide_distinct_positive_states():
    source = model()
    first = chemical_output_jet(source, [Q(1, 5)], [Q(1, 7)], [Q(1, 3)], [0])
    changed_u = chemical_output_jet(source, [Q(1, 4)], [Q(1, 7)], [Q(1, 3)], [0])
    changed_w = chemical_output_jet(source, [Q(1, 5)], [Q(1, 7)], [Q(2, 3)], [0])
    assert first.y == changed_u.y and first.y1 != changed_u.y1
    assert (first.y, first.y1) == (changed_w.y, changed_w.y1)
    assert (
        changed_w.y2[0] - first.y2[0]
        == (source.q - Q(1, 5)) * Q(1, 3) / source.epsilon_u
    )


def test_reconstructed_state_still_needs_the_independent_input():
    source = model()
    base = chemical_output_jet(source, [Q(1, 5)], [Q(1, 7)], [Q(1, 3)], [0])
    driven = chemical_output_jet(source, [Q(1, 5)], [Q(1, 7)], [Q(1, 3)], [Q(1, 9)])
    assert (base.y, base.y1, base.y2) == (driven.y, driven.y1, driven.y2)
    assert driven.y3[0] - base.y3[0] == (source.q - Q(1, 5)) * Q(1, 9) / (
        source.epsilon_u * source.epsilon_w
    )
    assert base.y3 != driven.y3


def test_transverse_singular_state_needs_third_jet_and_consistent_second_jet():
    source = model(q=Q(1, 2))
    jet = chemical_output_jet(source, [source.q], [Q(1, 7)], [Q(1, 3)], [0])
    with pytest.raises(ValueError, match="unsupported singular"):
        reconstruct_chemical_state(source, jet.y, jet.y1, jet.y2)
    assert reconstruct_chemical_state(source, jet.y, jet.y1, jet.y2, y3=jet.y3).w == (
        Q(1, 3),
    )
    with pytest.raises(ValueError, match="inconsistent singular"):
        reconstruct_chemical_state(source, jet.y, jet.y1, [jet.y2[0] + 1], y3=jet.y3)


def test_diffusion_can_recover_a_tangential_singular_site(symbolic):
    state, phi, parameters, jets = symbolic
    source = model(((-1, 1), (1, -1)), epsilon_u=1, q=Q(1, 2), delta=1)
    u, v, w, light = (Q(1, 2), Q(1, 4)), (Q(1, 3), Q(1, 5)), (Q(2, 3), Q(3, 5)), (0, 0)
    substitutions = dict(zip(state, (*v, *u, *w)))
    substitutions.update(zip(phi, light))
    substitutions.update(zip(parameters, (1, source.epsilon_w, source.q, source.f, 1)))
    second = s.Matrix.vstack(*jets[:3]).jacobian(state).subs(substitutions)
    third = s.Matrix.vstack(*jets).jacobian(state).subs(substitutions)
    assert second.rank() == 5 and third.rank() == 6
    no_diffusion = dict(substitutions)
    no_diffusion[parameters[-1]] = 0
    assert s.Matrix.vstack(*jets).jacobian(state).subs(no_diffusion).rank() == 5
    jet = chemical_output_jet(source, u, v, w, light)
    assert jet.y1[0] + jet.y2[0] == 0
    # The pointwise implementation abstains; this is not global unobservability.
    with pytest.raises(ValueError, match="unsupported singular"):
        reconstruct_chemical_state(source, jet.y, jet.y1, jet.y2, y3=jet.y3)


def test_restricted_q_one_family_retains_hidden_bromide():
    source = model(q=1)
    for initial_v in (Q(1, 3), Q(4, 3)):
        jets = [
            chemical_output_jet(source, [1], [initial_v], [hidden], [Q(2, 7)])
            for hidden in (Q(1, 4), Q(3, 4))
        ]
        assert jets[0] == jets[1]
        assert jets[0].y1 == (1 - initial_v,)
        assert jets[0].y2 == (initial_v - 1,)
        assert jets[0].y3 == (1 - initial_v,)
        with pytest.raises(ValueError, match="unsupported singular"):
            reconstruct_chemical_state(
                source, jets[0].y, jets[0].y1, jets[0].y2, y3=jets[0].y3
            )


def test_exact_reconstruction_can_be_arbitrarily_poorly_conditioned():
    source = model()
    for separation in (Q(1, 10), Q(1, 10**20)):
        u = (source.q + separation,)
        first = chemical_output_jet(source, u, [Q(1, 7)], [1], [0])
        second = chemical_output_jet(source, u, [Q(1, 7)], [2], [0])
        assert (first.y, first.y1) == (second.y, second.y1)
        assert abs(first.y2[0] - second.y2[0]) == separation / source.epsilon_u
        assert reconstruct_chemical_state(source, second.y, second.y1, second.y2).w == (
            2,
        )


def test_regular_error_bound_with_independent_spatial_jet_perturbations():
    source = model(((-1, 1), (1, -1)))
    u, v, w = (Q(1, 5), Q(3, 5)), (Q(1, 7), Q(2, 7)), (Q(1, 3), Q(2, 3))
    jet = chemical_output_jet(source, u, v, w, [0, 0])
    errors = (Q(1, 10000), Q(1, 20000), Q(1, 30000))
    alpha, b = errors[0] + errors[1], errors[1] + errors[2]
    separation, maximum_u, maximum_w = Q(1, 5), Q(3, 5), Q(2, 3)
    numerator_bound = (
        source.epsilon_u * (b + 2 * alpha) + (1 + 2 * maximum_u + alpha) * alpha
    )
    bound = (numerator_bound + maximum_w * alpha) / (separation - alpha)
    for signs in ((1, 1, 1), (-1, 1, -1), (1, -1, 1), (-1, -1, -1)):
        perturbed = [
            tuple(
                value + sign * error * (1 if i == 0 else -1)
                for i, value in enumerate(values)
            )
            for values, sign, error in zip((jet.y, jet.y1, jet.y2), signs, errors)
        ]
        reconstructed = reconstruct_chemical_state(source, *perturbed)
        assert (
            max(abs(actual - expected) for actual, expected in zip(reconstructed.u, u))
            <= alpha
        )
        assert (
            max(abs(actual - expected) for actual, expected in zip(reconstructed.w, w))
            <= bound
        )


@pytest.mark.parametrize("parameter", ["epsilon_u", "epsilon_w", "q", "f", "delta"])
@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), "1", 1j])
def test_model_rejects_nonphysical_scalars_before_arithmetic(parameter, invalid):
    with pytest.raises((TypeError, ValueError)):
        model(**{parameter: invalid})


@pytest.mark.parametrize(
    "invalid", [(), ((1,),), ((0, 1), (0,)), ((1, -1), (0, 0)), ((True,),), {0: (0,)}]
)
def test_diffusion_admission(invalid):
    with pytest.raises((TypeError, ValueError)):
        model(invalid)


@pytest.mark.parametrize("parameter", ["epsilon_u", "epsilon_w", "q", "f", "delta"])
def test_sign_domains(parameter):
    with pytest.raises(ValueError):
        model(**{parameter: -1})
    if parameter in ("epsilon_u", "epsilon_w", "q"):
        with pytest.raises(ValueError):
            model(**{parameter: 0})


def test_primitive_readmission_detachment_and_exact_small_values():
    source = model(q=Q(1, 10**400))
    assert source.q != 0 and source.q == Q(1, 10**400)
    with pytest.raises(FrozenInstanceError):
        source.q = 1
    object.__setattr__(source, "q", True)
    with pytest.raises(TypeError):
        chemical_output_jet(source, [1], [1], [1], [0])
    with pytest.raises(TypeError):
        reconstruct_chemical_state(source, [1], [0], [0])
    raw = [[-1, 1], [1, -1]]
    detached = model(raw)
    raw[0][0] = 9
    assert detached.diffusion_matrix[0] == (-1, 1)
    assert isinstance(model(epsilon_u=1.5).epsilon_u, Q)


def test_vector_and_inverse_domains_and_unused_optional_jet_admission():
    source = model()
    for u, v, w, light in [
        ([True], [1], [1], [0]),
        ([1], [-1], [1], [0]),
        ([1], [1], [1], [-1]),
        ([1, 1], [1, 1], [1, 1], [0, 0]),
        ({1}, [1], [1], [0]),
    ]:
        with pytest.raises((ValueError, TypeError)):
            chemical_output_jet(source, u, v, w, light)
    with pytest.raises(ValueError, match="u must be nonnegative"):
        reconstruct_chemical_state(source, [0], [-1], [0])
    with pytest.raises(ValueError, match="w must be nonnegative"):
        reconstruct_chemical_state(source, [0], [0], [-1])
    jet = chemical_output_jet(source, [Q(1, 5)], [Q(1, 7)], [Q(1, 3)], [0])
    with pytest.raises(TypeError):
        reconstruct_chemical_state(source, jet.y, jet.y1, jet.y2, y3=[True])
    # A regular inverse makes no claim about an arbitrary third derivative.
    assert reconstruct_chemical_state(
        source, jet.y, jet.y1, jet.y2, y3=[123]
    ) == ChemicalState([Q(1, 5)], jet.y, [Q(1, 3)])
    with pytest.raises(ValueError):
        ChemicalOutputJet([1], [0, 0], [0], [0])
    with pytest.raises(ValueError):
        model(tuple((0,) * 33 for _ in range(33)))
    forged = replace(source)
    object.__setattr__(forged, "diffusion_matrix", ((1,),))
    with pytest.raises(ValueError, match="row sums"):
        chemical_output_jet(forged, [1], [1], [1], [0])
