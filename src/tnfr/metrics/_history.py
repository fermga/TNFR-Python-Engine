"""Shared admission of retained observation sequences and optional samples."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from .._exact_time import finite_represented_real


def stored_sequence(value: Any, label: str) -> tuple[Any, ...]:
    """Detach ordered samples without consuming one-shot iterators.

    Lists, tuples, deques, other non-text sequences and one-dimensional NumPy
    arrays retain their order. Multidimensional arrays are not flattened into
    an invented observation stream.
    """
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise TypeError(f"{label} must be a one-dimensional sample sequence")
    elif isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, (Sequence, deque)
    ):
        raise TypeError(f"{label} must be an ordered sample sequence")
    return tuple(value)


def stored_value(value: Any, label: str) -> float | None:
    """Retain explicit absence and admit every present scalar before conversion."""
    return None if value is None else finite_represented_real(value, label)[0]
