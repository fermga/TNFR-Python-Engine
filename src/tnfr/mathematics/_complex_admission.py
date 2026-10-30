"""Represented complex scalar admission without losing either source channel."""

from __future__ import annotations

from numbers import Complex, Real
from typing import Any

from .._exact_time import finite_represented_real


def finite_represented_complex(value: Any, label: str) -> complex:
    """Admit finite binary64 components before constructing a complex scalar.

    Real inputs use the shared real admission directly. Complex inputs retain
    both channels; neither Boolean/text coercion nor nonzero underflow is
    allowed. This is numerical admission, not signed-scalar EPI projection.
    """
    if isinstance(value, Real):
        return complex(finite_represented_real(value, label)[0], 0.0)
    if not isinstance(value, Complex):
        raise TypeError(f"{label} must be a finite real or complex scalar")
    real = finite_represented_real(value.real, f"{label}.real")[0]
    imaginary = finite_represented_real(value.imag, f"{label}.imag")[0]
    return complex(real, imaginary)
