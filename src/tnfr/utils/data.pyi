from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from typing import Any, Callable, TypeVar

T = TypeVar("T")

STRING_TYPES: tuple[type[str] | type[bytes] | type[bytearray], ...]
MAX_MATERIALIZE_DEFAULT: int
NEGATIVE_WEIGHTS_MSG: str

__all__: tuple[str, ...]


def convert_value(
    value: Any,
    conv: Callable[[Any], T],
    *,
    strict: bool = ...,
    key: str | None = ...,
    log_level: int | None = ...,
) -> tuple[bool, T | None]: ...


def negative_weights_warn_once(
    *,
    maxsize: int = ...,
) -> Callable[[Mapping[str, float]], None]: ...


def is_non_string_sequence(obj: Any) -> bool: ...


def flatten_structure(
    obj: Any,
    *,
    expand: Callable[[Any], Iterable[Any] | None] | None = ...,
) -> Iterator[Any]: ...


def normalize_materialize_limit(max_materialize: int | None) -> int | None: ...


def ensure_collection(
    it: Iterable[T],
    *,
    max_materialize: int | None = ...,
    error_msg: str | None = ...,
) -> Collection[T]: ...


def normalize_weights(
    dict_like: Mapping[str, Any],
    keys: Iterable[str] | Sequence[str],
    default: float = ...,
    *,
    error_on_negative: bool = ...,
    warn_once: bool | Callable[[Mapping[str, float]], None] = ...,
    error_on_conversion: bool = ...,
) -> dict[str, float]: ...


def normalize_counter(
    counts: Mapping[str, float | int],
) -> tuple[dict[str, float], float]: ...


def mix_groups(
    dist: Mapping[str, float],
    groups: Mapping[str, Iterable[str]],
    *,
    prefix: str = ...,
) -> dict[str, float]: ...
