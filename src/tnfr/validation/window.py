"""Primitive window validation helpers without heavy dependencies."""

from __future__ import annotations

import numbers

__all__ = ["validate_window"]


def validate_window(window: int, *, positive: bool = False) -> int:
    """Validate ``window`` as an integer and return it.

    Parameters
    ----------
    window:
        Value to coerce into an integer window size.
    positive:
        When ``True`` the window must be strictly greater than zero; otherwise
        zero is accepted. Negative values are never permitted.

    Returns
    -------
    int
        The validated integer window size.

    Raises
    ------
    TypeError
        If ``window`` is not an integer value.
    ValueError
        If ``window`` violates the positivity constraint.
    """

    if isinstance(window, bool) or not isinstance(window, numbers.Integral):
        raise TypeError("'window' must be an integer")
    if window < 0 or (positive and window == 0):
        kind = "positive" if positive else "non-negative"
        raise ValueError(f"'window'={window} must be {kind}")
    return int(window)
