from .definitions import Operator

__all__ = ["OPERATORS", "register_operator", "get_operator_class"]

OPERATORS: dict[str, type[Operator]]

def register_operator(cls: type[Operator]) -> type[Operator]: ...
def get_operator_class(name: str) -> type[Operator]: ...
