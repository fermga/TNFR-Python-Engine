"""Deprecation utilities for TNFR validation migration.

This module provides deprecation warnings to guide users from scattered
validation functions to the unified TNFRValidator API.
"""

import functools
import warnings
from typing import Any, Callable, TypeVar

__all__ = [
    "deprecated_in_favor_of_unified_validator",
]

F = TypeVar("F", bound=Callable[..., Any])


def deprecated_in_favor_of_unified_validator(
    replacement: str,
    version: str = "0.6.0",
) -> Callable[[F], F]:
    """Decorator to mark validation functions as deprecated.

    This decorator adds a deprecation warning to functions that should be
    replaced by the unified TNFRValidator API.

    Parameters
    ----------
    replacement : str
        The recommended replacement (e.g., "TNFRValidator.validate_inputs()").
    version : str, optional
        Version when the function will be removed (default: "0.6.0").

    Returns
    -------
    Callable
        Decorated function with deprecation warning.

    Examples
    --------
    >>> @deprecated_in_favor_of_unified_validator("TNFRValidator.validate_inputs()")
    ... def old_validate_epi(value):
    ...     return value
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{func.__name__}() is deprecated and will be removed in version {version}. "
                f"Use {replacement} instead for a unified validation experience.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
