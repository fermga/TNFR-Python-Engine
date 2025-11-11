"""Dataclass compatibility for Python 3.9+.

Provides a wrapper for @dataclass that conditionally applies ``slots=True``
only on Python 3.10+ where it is supported, maintaining compatibility with
Python 3.9.
"""

import sys
from dataclasses import dataclass as _dataclass
from typing import Type, TypeVar, overload

_T = TypeVar("_T")

# Check if slots parameter is supported (Python 3.10+)
_SLOTS_SUPPORTED = sys.version_info >= (3, 10)


@overload
def dataclass(
    cls: Type[_T],
    /,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
) -> Type[_T]: ...


@overload
def dataclass(
    cls: None = None,
    /,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
) -> Type[_T]: ...


def dataclass(
    cls=None,
    /,
    *,
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
    match_args=True,
    kw_only=False,
    slots=False,
):
    """Compatibility wrapper for @dataclass supporting Python 3.9+.

    On Python 3.10+, passes all parameters including ``slots`` to dataclass.
    On Python 3.9, silently ignores ``slots`` parameter since it's not supported.

    This allows code to use ``slots=True`` for performance benefits on newer
    Python versions while maintaining backward compatibility with 3.9.

    Parameters
    ----------
    cls : type, optional
        Class to decorate (when used without parentheses).
    init : bool, default True
        Generate __init__ method.
    repr : bool, default True
        Generate __repr__ method.
    eq : bool, default True
        Generate __eq__ method.
    order : bool, default False
        Generate ordering methods.
    unsafe_hash : bool, default False
        Force generation of __hash__ method.
    frozen : bool, default False
        Make instances immutable.
    match_args : bool, default True
        Generate __match_args__ for pattern matching (Python 3.10+).
    kw_only : bool, default False
        Make all fields keyword-only.
    slots : bool, default False
        Generate __slots__ (Python 3.10+ only, ignored on 3.9).

    Returns
    -------
    type
        Decorated dataclass.

    Examples
    --------
    >>> from tnfr.compat.dataclass import dataclass
    >>> @dataclass(slots=True)  # Works on both 3.9 and 3.10+
    ... class Point:
    ...     x: float
    ...     y: float
    """

    # Build kwargs based on Python version
    kwargs = {
        "init": init,
        "repr": repr,
        "eq": eq,
        "order": order,
        "unsafe_hash": unsafe_hash,
        "frozen": frozen,
    }

    # Add parameters supported in Python 3.10+
    if sys.version_info >= (3, 10):
        kwargs["match_args"] = match_args
        kwargs["kw_only"] = kw_only
        kwargs["slots"] = slots

    # Handle decorator with and without parentheses
    def wrap(c):
        return _dataclass(c, **kwargs)

    if cls is None:
        # Called with parentheses: @dataclass(...)
        return wrap
    else:
        # Called without parentheses: @dataclass
        return wrap(cls)
