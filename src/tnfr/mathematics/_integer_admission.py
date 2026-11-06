"""Strict integer admission shared by arithmetic models and exact operators."""

from __future__ import annotations

import operator

import numpy as np


def _integer_argument(value: int, name: str) -> int:
    """Read an integer argument without truncating or accepting logical values."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not boolean")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, not a truncated value") from exc
