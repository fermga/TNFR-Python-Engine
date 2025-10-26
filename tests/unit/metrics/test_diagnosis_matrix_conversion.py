"""Unit tests for coherence matrix conversion helpers."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from tnfr.metrics.diagnosis import _coherence_matrix_to_numpy


class TestCoherenceMatrixToNumpy:
    """Behavioural tests for ``_coherence_matrix_to_numpy``."""

    def test_numpy_array_input_returns_copied_matrix_with_zero_diagonal(self) -> None:
        size = 3
        original = np.ones((size, size), dtype=float)
        result = _coherence_matrix_to_numpy(original, size=size, np_mod=np)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (size, size)
        assert result is not original
        assert np.allclose(np.diag(result), np.zeros(size))
        # Non-diagonal values preserved from the original input
        off_diag_indices = np.ones_like(result, dtype=bool)
        np.fill_diagonal(off_diag_indices, False)
        assert np.allclose(result[off_diag_indices], 1.0)
        # Original matrix remains unchanged (copy semantics)
        assert np.allclose(np.diag(original), np.ones(size))

    def test_dense_iterable_converts_to_float_array(self) -> None:
        size = 2
        weights = [[1, 2], [3, 4]]

        result = _coherence_matrix_to_numpy(weights, size=size, np_mod=np)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (size, size)
        assert np.issubdtype(result.dtype, np.floating)
        expected = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=float)
        assert np.array_equal(result, expected)

    def test_sparse_triplets_populate_matrix_and_clear_diagonal(self) -> None:
        size = 4
        weights = [
            (0, 1, 0.5),
            (1, 2, 1.5),
            (2, 0, 2.5),
            (2, 2, 9.0),  # Diagonal contribution must be cleared
        ]

        result = _coherence_matrix_to_numpy(weights, size=size, np_mod=np)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (size, size)
        expected = np.zeros((size, size), dtype=float)
        expected[0, 1] = 0.5
        expected[1, 2] = 1.5
        expected[2, 0] = 2.5
        # Diagonal should be forced to zero regardless of inputs
        assert np.array_equal(result, expected)
        assert np.allclose(np.diag(result), np.zeros(size))

    @pytest.mark.parametrize(
        "payload,size",
        [
            ([[1], [2, 3]], 2),  # Ragged rows
            ([[1, 2, 3], [4, 5, 6]], 3),  # Wrong shape (2x3 vs 3x3)
            (object(), 2),  # Non-iterable or unsupported type
        ],
    )
    def test_invalid_inputs_return_none(self, payload: object, size: int) -> None:
        result = _coherence_matrix_to_numpy(payload, size=size, np_mod=np)
        assert result is None
