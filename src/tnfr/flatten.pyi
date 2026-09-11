from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from .tokens import THOL, Token

__all__: list[str]

def __getattr__(name: str) -> Any: ...

class THOLEvaluator(Iterator[Token | object]):
    def __init__(self, item: THOL, *, max_materialize: int | None = ...) -> None: ...
    def __iter__(self) -> THOLEvaluator: ...
    def __next__(self) -> Token | object: ...

def parse_program_tokens(
    obj: Iterable[Any] | Sequence[Any] | Any,
    *,
    max_materialize: int | None = ...,
) -> list[Token]: ...
