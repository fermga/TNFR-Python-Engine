"""Compatibility imports for the shared exact rational algebra owner."""

from ..mathematics._exact_linear_algebra import ExactSquareMatrix
from ..mathematics._exact_linear_algebra import (  # noqa: F401 (private compatibility exports)
    _exact_square_matrix_product_unchecked as _exact_square_matrix_product_unchecked,
)
from ..mathematics._exact_linear_algebra import (  # noqa: F401 (private compatibility exports)
    _require_exact_square_matrix as _require_exact_square_matrix,
)
from ..mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_square_matrix_power,
    exact_square_matrix_product,
    exact_symmetric_semidefinite,
)

__all__ = [
    "ExactSquareMatrix",
    "exact_matrix_inverse",
    "exact_square_matrix_power",
    "exact_square_matrix_product",
    "exact_symmetric_semidefinite",
]
