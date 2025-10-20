from typing import Any

from .definitions import Operator

__all__: Any

def __getattr__(name: str) -> Any: ...

OPERATORS: dict[str, type[Operator]]

def discover_operators() -> None: ...

def register_operator(cls: type[Operator]) -> type[Operator]: ...

def get_operator_class(name: str) -> type[Operator]: ...
