"""Original-input admission and independent GKSL generator checks."""

from fractions import Fraction

import numpy as np
import pytest

from tnfr.mathematics import build_delta_nfr, build_lindblad_delta_nfr


@pytest.mark.parametrize("role", ["hamiltonian", "collapse"])
@pytest.mark.parametrize("as_object_array", [False, True])
@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(True, id="boolean"),
        pytest.param(np.bool_(False), id="numpy-boolean"),
        pytest.param("1", id="text"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(complex(0, float("inf")), id="infinite-imaginary"),
        pytest.param(Fraction(1, 10**400), id="underflowing-rational"),
        pytest.param(10**400, id="overflowing-integer"),
    ],
)
def test_lindblad_rejects_invalid_original_components(
    role, as_object_array, entry
) -> None:
    matrix = [[entry, 0], [0, 0]]
    if as_object_array:
        matrix = np.array(matrix, dtype=object)
    kwargs = (
        {"hamiltonian": matrix}
        if role == "hamiltonian"
        else {"collapse_operators": [matrix]}
    )

    # Turning off residual checks does not turn off primitive admission.
    with pytest.raises(ValueError, match="finite representable"):
        build_lindblad_delta_nfr(
            **kwargs, ensure_trace_preserving=False, ensure_contractive=False
        )


@pytest.mark.parametrize("entry", [float("inf"), complex(float("nan"), 0)])
def test_lindblad_rejects_nonfinite_numeric_arrays(entry) -> None:
    matrix = np.array([[0, entry], [0, 0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="finite complex"):
        build_lindblad_delta_nfr(
            collapse_operators=[matrix],
            ensure_trace_preserving=False,
            ensure_contractive=False,
        )


def test_lindblad_column_vectorization_matches_direct_gksl_action() -> None:
    hamiltonian = np.array([[0.25, 0.5j], [-0.5j, -0.25]])
    rational_lowering = [[0, Fraction(1, 2)], [0, 0]]
    collapse = [
        np.array([[0, 0.5], [0, 0]], dtype=np.complex128),
        np.array([[0.25j, 0], [0, -0.25j]], dtype=np.complex128),
    ]
    generator = build_lindblad_delta_nfr(
        hamiltonian=hamiltonian,
        collapse_operators=[rational_lowering, collapse[1]],
        nu_f=2,
        scale=0.75,
    )

    # Matrix units span the full operator space, including off-diagonal
    # non-Hermitian inputs; this checks the complete superoperator action.
    for row, column in np.ndindex(2, 2):
        matrix_unit = np.zeros((2, 2), dtype=np.complex128)
        matrix_unit[row, column] = 1
        expected = -1j * (hamiltonian @ matrix_unit - matrix_unit @ hamiltonian)
        for operator in collapse:
            adjoint = operator.conj().T
            expected += operator @ matrix_unit @ adjoint
            expected -= 0.5 * (
                adjoint @ operator @ matrix_unit + matrix_unit @ adjoint @ operator
            )
        actual = (generator @ matrix_unit.reshape(-1, order="F")).reshape(
            (2, 2), order="F"
        )
        np.testing.assert_allclose(actual, 1.5 * expected, atol=1e-14)


def test_lindblad_trace_preservation_does_not_require_unitality() -> None:
    gamma = 0.25
    # A stacked array is an iterable of collapse matrices; it must not undergo
    # an ambiguous array truth-value test during admission.
    collapse = np.array([[[0, 0.5], [0, 0]]], dtype=np.complex128)
    generator = build_lindblad_delta_nfr(collapse_operators=collapse)
    identity_vector = np.eye(2).reshape(-1, order="F")

    np.testing.assert_allclose(identity_vector @ generator, np.zeros(4), atol=1e-14)
    identity_rate = (generator @ identity_vector).reshape((2, 2), order="F")
    np.testing.assert_allclose(identity_rate, np.diag([gamma, -gamma]), atol=1e-14)


def test_lindblad_checks_every_collapse_dimension() -> None:
    with pytest.raises(ValueError, match=r"collapse operator\[1\] dimension mismatch"):
        build_lindblad_delta_nfr(collapse_operators=[np.eye(2), np.eye(3)])


@pytest.mark.parametrize("builder", [build_delta_nfr, build_lindblad_delta_nfr])
@pytest.mark.parametrize("bad", [True, np.bool_(True), 2.0, "2", 0, -1])
def test_generator_dimensions_use_strict_integer_admission(builder, bad):
    with pytest.raises(ValueError, match="positive integer"):
        builder(dim=bad)


@pytest.mark.parametrize("builder", [build_delta_nfr, build_lindblad_delta_nfr])
@pytest.mark.parametrize("parameter", ["nu_f", "scale"])
@pytest.mark.parametrize("bad", [True, "2", 1j, np.inf, np.nan, Fraction(1, 10**400)])
def test_generator_scalars_are_admitted_before_arithmetic(builder, parameter, bad):
    with pytest.raises(ValueError, match="finite representable real"):
        builder(dim=2, **{parameter: bad})


@pytest.mark.parametrize("builder", [build_delta_nfr, build_lindblad_delta_nfr])
@pytest.mark.parametrize("factors", [(1e200, 1e200), (1e-200, 1e-200)])
def test_generator_rejects_an_unrepresentable_scale_product(builder, factors):
    with pytest.raises(ValueError, match="scaling"):
        builder(dim=2, nu_f=factors[0], scale=factors[1])


@pytest.mark.parametrize("bad", [True, "1", -1, np.inf, np.nan, Fraction(1, 10**400)])
def test_lindblad_tolerance_cannot_bypass_hermiticity(bad):
    with pytest.raises(ValueError, match="finite nonnegative"):
        build_lindblad_delta_nfr(hamiltonian=[[0, 1], [0, 0]], atol=bad)


def test_large_finite_adjacency_needs_no_redundant_hermitian_average():
    generator = build_delta_nfr(2, topology="adjacency", scale=1e308)
    np.testing.assert_array_equal(generator, [[0, 1e308], [1e308, 0]])


def test_representable_signed_scaling_preserves_unitary_generator():
    positive = build_delta_nfr(2, scale=Fraction(1, 2))
    negative = build_delta_nfr(2, scale=Fraction(-1, 2))
    np.testing.assert_array_equal(negative, -positive)


def test_generator_arithmetic_overflow_rejects_even_with_checks_disabled():
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="finite"),
    ):
        build_lindblad_delta_nfr(
            collapse_operators=[[[0, 1e200], [0, 0]]],
            ensure_trace_preserving=False,
            ensure_contractive=False,
        )


def test_false_string_disables_only_the_lindblad_spectral_gate():
    generator = build_lindblad_delta_nfr(
        collapse_operators=[[[0, 1], [0, 0]]],
        scale=-1,
        ensure_contractive="false",
    )
    # This signed auxiliary matrix exists, but is not a forward GKSL law.
    assert np.max(np.linalg.eigvals(generator).real) == pytest.approx(1)


@pytest.mark.parametrize("flag", ["ensure_trace_preserving", "ensure_contractive"])
def test_lindblad_rejects_unrecognized_boolean_controls(flag):
    with pytest.raises(ValueError):
        build_lindblad_delta_nfr(dim=2, **{flag: "not-a-boolean"})
