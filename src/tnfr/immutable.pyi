from __future__ import annotations

from typing import Any, Callable, Iterator, Mapping, Union

from ._compat import TypeAlias

FrozenPrimitive: TypeAlias = Union[int, float, complex, str, bool, bytes, None]
FrozenCollectionItems: TypeAlias = tuple["FrozenSnapshot", ...]
FrozenMappingItems: TypeAlias = tuple[tuple[Any, "FrozenSnapshot"], ...]
FrozenTaggedCollection: TypeAlias = tuple[str, FrozenCollectionItems]
FrozenTaggedMapping: TypeAlias = tuple[str, FrozenMappingItems]
FrozenSnapshot: TypeAlias = Union[
    FrozenPrimitive,
    FrozenCollectionItems,
    FrozenTaggedCollection,
    FrozenTaggedMapping,
]
ImmutableTagHandler: TypeAlias = Callable[[tuple[Any, ...]], bool]

__all__: tuple[str, ...]

def __getattr__(name: str) -> Any: ...
def _cycle_guard(value: Any, seen: set[int] | None = ...) -> Iterator[set[int]]: ...
def _check_cycle(
    func: Callable[[Any, set[int] | None], FrozenSnapshot],
) -> Callable[[Any, set[int] | None], FrozenSnapshot]: ...
def _freeze(value: Any, seen: set[int] | None = ...) -> FrozenSnapshot: ...
def _freeze_mapping(
    value: Mapping[Any, Any],
    seen: set[int] | None = ...,
) -> FrozenTaggedMapping: ...
def _is_immutable(value: Any) -> bool: ...
def _is_immutable_inner(value: Any) -> bool: ...

_IMMUTABLE_CACHE: Any
_IMMUTABLE_TAG_DISPATCH: Mapping[str, ImmutableTagHandler]
