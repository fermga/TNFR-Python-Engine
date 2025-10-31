from typing import Any, Iterable

from . import ValidationOutcome

__all__: Any

def __getattr__(name: str) -> Any: ...

def validate_sequence(
    names: Iterable[str] | object = ..., **kwargs: object
) -> ValidationOutcome[tuple[str, ...]]: ...
