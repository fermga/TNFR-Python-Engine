from typing import Any, Literal

from .definitions import Operador

__all__: Any

def __getattr__(name: Literal["OPERADORES"]) -> dict[str, type[Operador]]: ...
def __getattr__(name: str) -> Any: ...

OPERATORS: dict[str, type[Operador]]
discover_operators: Any
register_operator: Any
get_operator_class: Any
