"""Lightweight stub for numpy when it's not installed.

This stub provides minimal type compatibility for numpy when it's not installed,
allowing type checking to succeed. At runtime, actual numpy operations will fail
with informative errors if called without the real numpy package.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

__all__ = [
    "ndarray",
    "float64",
    "float_",
    "complex128",
    "complexfloating",
    "dtype",
    "asarray",
    "array",
    "eye",
    "zeros",
    "ones",
    "isfinite",
    "all",
    "allclose",
    "diff",
    "any",
    "pi",
]


class _NotInstalledError(RuntimeError):
    """Raised when trying to use numpy operations without numpy installed."""

    def __init__(self, operation: str = "numpy operation") -> None:
        super().__init__(
            f"Cannot perform {operation}: numpy is not installed. "
            "Install it with: pip install numpy"
        )


class ndarray:
    """Stub for numpy.ndarray type."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise _NotInstalledError("array creation")

    @property
    def shape(self) -> tuple[int, ...]:
        raise _NotInstalledError("array.shape")

    @property
    def ndim(self) -> int:
        raise _NotInstalledError("array.ndim")

    @property
    def size(self) -> int:
        raise _NotInstalledError("array.size")


class dtype:
    """Stub for numpy.dtype type."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise _NotInstalledError("dtype")


class float64:
    """Stub for numpy.float64 type."""


class float_:
    """Stub for numpy.float_ type."""


class complex128:
    """Stub for numpy.complex128 type."""


class complexfloating:
    """Stub for numpy.complexfloating type."""


def asarray(*args: Any, **kwargs: Any) -> ndarray:
    """Stub for numpy.asarray."""
    raise _NotInstalledError("numpy.asarray")


def array(*args: Any, **kwargs: Any) -> ndarray:
    """Stub for numpy.array."""
    raise _NotInstalledError("numpy.array")


def eye(*args: Any, **kwargs: Any) -> ndarray:
    """Stub for numpy.eye."""
    raise _NotInstalledError("numpy.eye")


def zeros(*args: Any, **kwargs: Any) -> ndarray:
    """Stub for numpy.zeros."""
    raise _NotInstalledError("numpy.zeros")


def ones(*args: Any, **kwargs: Any) -> ndarray:
    """Stub for numpy.ones."""
    raise _NotInstalledError("numpy.ones")


def isfinite(*args: Any, **kwargs: Any) -> Any:
    """Stub for numpy.isfinite."""
    raise _NotInstalledError("numpy.isfinite")


def all(*args: Any, **kwargs: Any) -> Any:
    """Stub for numpy.all."""
    raise _NotInstalledError("numpy.all")


def allclose(*args: Any, **kwargs: Any) -> Any:
    """Stub for numpy.allclose."""
    raise _NotInstalledError("numpy.allclose")


def diff(*args: Any, **kwargs: Any) -> ndarray:
    """Stub for numpy.diff."""
    raise _NotInstalledError("numpy.diff")


def any(*args: Any, **kwargs: Any) -> Any:
    """Stub for numpy.any."""
    raise _NotInstalledError("numpy.any")


# Constants
pi: float = 3.141592653589793


if TYPE_CHECKING:
    # Provide typing namespace for numpy.typing when used in TYPE_CHECKING blocks
    class typing:
        """Stub for numpy.typing module."""

        class NDArray:
            """Stub for numpy.typing.NDArray."""
