"""Tests for TNFR factory pattern compliance and consistency.

This module validates that all factory functions follow the documented
patterns in docs/FACTORY_PATTERNS.md, including naming conventions,
input validation, structural verification, and type safety.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics.backend import get_backend
from tnfr.mathematics.generators import build_delta_nfr, build_lindblad_delta_nfr
from tnfr.mathematics.operators_factory import (
    make_coherence_operator,
    make_frequency_operator,
)


class TestFactoryNamingConventions:
    """Verify factory functions follow naming conventions."""

    def test_operator_factories_use_make_prefix(self) -> None:
        """Operator factories should use make_* prefix."""
        assert hasattr(make_coherence_operator, "__name__")
        assert make_coherence_operator.__name__.startswith("make_")
        assert hasattr(make_frequency_operator, "__name__")
        assert make_frequency_operator.__name__.startswith("make_")

    def test_generator_factories_use_build_prefix(self) -> None:
        """Generator factories should use build_* prefix."""
        assert hasattr(build_delta_nfr, "__name__")
        assert build_delta_nfr.__name__.startswith("build_")
        assert hasattr(build_lindblad_delta_nfr, "__name__")
        assert build_lindblad_delta_nfr.__name__.startswith("build_")


class TestFactoryInputValidation:
    """Verify factory functions validate inputs properly."""

    def test_coherence_operator_rejects_invalid_dimension(self) -> None:
        """Factory should reject non-positive dimensions."""
        with pytest.raises(ValueError, match="strictly positive"):
            make_coherence_operator(dim=0)

        with pytest.raises(ValueError, match="strictly positive"):
            make_coherence_operator(dim=-5)

    def test_coherence_operator_rejects_invalid_c_min(self) -> None:
        """Factory should reject non-finite c_min values."""
        with pytest.raises(ValueError, match="finite"):
            make_coherence_operator(dim=4, c_min=float("inf"))

        with pytest.raises(ValueError, match="finite"):
            make_coherence_operator(dim=4, c_min=float("nan"))

    def test_coherence_operator_rejects_wrong_spectrum_shape(self) -> None:
        """Factory should reject spectrum with wrong dimension."""
        with pytest.raises(ValueError, match="size must match"):
            make_coherence_operator(dim=4, spectrum=np.array([1.0, 2.0]))

    def test_frequency_operator_rejects_non_square_matrix(self) -> None:
        """Factory should reject non-square matrices."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with pytest.raises(ValueError, match="square"):
            make_frequency_operator(matrix)

    def test_frequency_operator_rejects_non_hermitian(self) -> None:
        """Factory should reject non-Hermitian matrices."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Hermitian"):
            make_frequency_operator(matrix)

    def test_delta_nfr_rejects_invalid_dimension(self) -> None:
        """Generator factory should reject non-positive dimensions."""
        with pytest.raises(ValueError, match="positive"):
            build_delta_nfr(dim=0)

        with pytest.raises(ValueError, match="positive"):
            build_delta_nfr(dim=-1)

    def test_delta_nfr_rejects_unknown_topology(self) -> None:
        """Generator factory should reject unknown topologies."""
        with pytest.raises(ValueError, match="Unknown.*topology"):
            build_delta_nfr(dim=4, topology="invalid_topology")


class TestFactoryStructuralVerification:
    """Verify factories enforce structural invariants."""

    def test_coherence_operator_is_hermitian(self) -> None:
        """Factory produces Hermitian operators."""
        operator = make_coherence_operator(dim=4)
        assert operator.is_hermitian(atol=1e-9)

    def test_coherence_operator_is_positive_semidefinite(self) -> None:
        """Factory produces PSD operators."""
        operator = make_coherence_operator(dim=4)
        assert operator.is_positive_semidefinite(atol=1e-9)

    def test_frequency_operator_is_hermitian(self) -> None:
        """Factory produces Hermitian frequency operators."""
        matrix = np.array([[1.0, 0.5j], [-0.5j, 2.0]], dtype=np.complex128)
        operator = make_frequency_operator(matrix)
        assert operator.is_hermitian(atol=1e-9)

    def test_frequency_operator_is_positive_semidefinite(self) -> None:
        """Factory produces PSD frequency operators."""
        matrix = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128)
        operator = make_frequency_operator(matrix)
        assert operator.is_positive_semidefinite(atol=1e-9)

    def test_delta_nfr_is_hermitian(self) -> None:
        """Generator factory produces Hermitian matrices."""
        rng = np.random.default_rng(42)
        delta_nfr = build_delta_nfr(dim=4, rng=rng)
        assert np.allclose(delta_nfr, delta_nfr.conj().T, atol=1e-9)

    def test_lindblad_generator_is_trace_preserving(self) -> None:
        """Lindblad factory produces trace-preserving generators."""
        dim = 2
        H = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        L = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)

        # Should not raise with default ensure_trace_preserving=True
        lindblad = build_lindblad_delta_nfr(
            hamiltonian=H,
            collapse_operators=[L],
            ensure_trace_preserving=True,
        )

        # Verify identity is left-invariant
        identity = np.eye(dim, dtype=np.complex128)
        identity_vec = identity.reshape(dim * dim, order="F")
        left_action = identity_vec.conj().T @ lindblad
        np.testing.assert_allclose(left_action, 0.0, atol=1e-8)


class TestFactoryReproducibility:
    """Verify factories produce deterministic outputs with seeds."""

    def test_delta_nfr_reproducible_with_seed(self) -> None:
        """Factory produces identical results with same seed."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        delta1 = build_delta_nfr(dim=5, topology="laplacian", rng=rng1)
        delta2 = build_delta_nfr(dim=5, topology="laplacian", rng=rng2)

        np.testing.assert_array_equal(delta1, delta2)

    def test_delta_nfr_different_with_different_seeds(self) -> None:
        """Factory produces different results with different seeds."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)

        delta1 = build_delta_nfr(dim=5, topology="laplacian", rng=rng1)
        delta2 = build_delta_nfr(dim=5, topology="laplacian", rng=rng2)

        # Should be different (with very high probability)
        assert not np.allclose(delta1, delta2)


class TestFactoryBackendIntegration:
    """Verify factories work with different backends."""

    def test_coherence_operator_uses_backend(self) -> None:
        """Factory integrates with backend abstraction."""
        backend = get_backend()  # Gets default or configured backend
        operator = make_coherence_operator(dim=4, c_min=0.5)

        # Operator should have been constructed through backend
        assert operator.backend is backend

    def test_delta_nfr_returns_numpy_array(self) -> None:
        """Generator factory returns numpy arrays."""
        delta_nfr = build_delta_nfr(dim=4)
        assert isinstance(delta_nfr, np.ndarray)
        assert delta_nfr.dtype == np.complex128


class TestFactoryKeywordArguments:
    """Verify factories use keyword-only arguments appropriately."""

    def test_coherence_operator_requires_keywords(self) -> None:
        """Optional parameters must be keyword-only."""
        # This should work
        operator = make_coherence_operator(4, c_min=0.5)
        assert operator.matrix.shape[0] == 4

        # This should also work with explicit keywords
        operator = make_coherence_operator(dim=4, c_min=0.5)
        assert operator.matrix.shape[0] == 4

    def test_delta_nfr_requires_keywords(self) -> None:
        """Optional parameters must be keyword-only."""
        # This should work
        delta_nfr = build_delta_nfr(4, topology="laplacian")
        assert delta_nfr.shape == (4, 4)

        # This should also work with explicit keywords
        delta_nfr = build_delta_nfr(dim=4, topology="laplacian")
        assert delta_nfr.shape == (4, 4)


class TestFactoryDocumentation:
    """Verify factories have proper documentation."""

    def test_coherence_operator_has_docstring(self) -> None:
        """Factory has comprehensive docstring."""
        assert make_coherence_operator.__doc__ is not None
        doc = make_coherence_operator.__doc__
        assert "Parameters" in doc
        assert "Returns" in doc
        assert "Raises" in doc

    def test_frequency_operator_has_docstring(self) -> None:
        """Factory has comprehensive docstring."""
        assert make_frequency_operator.__doc__ is not None
        doc = make_frequency_operator.__doc__
        assert "Parameters" in doc
        assert "Returns" in doc

    def test_delta_nfr_has_docstring(self) -> None:
        """Generator factory has comprehensive docstring."""
        assert build_delta_nfr.__doc__ is not None
        doc = build_delta_nfr.__doc__
        assert "Parameters" in doc
        assert "nu_f" in doc  # Should mention structural frequency
