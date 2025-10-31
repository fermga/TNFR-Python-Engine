from collections.abc import Iterable

from . import ValidationOutcome

__all__ = ("validate_sequence",)


def validate_sequence(
    names: Iterable[str] | object = ..., **kwargs: object
) -> ValidationOutcome[tuple[str, ...]]: ...
