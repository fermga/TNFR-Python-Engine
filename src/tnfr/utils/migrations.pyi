from collections.abc import Mapping, MutableMapping
from typing import Any, Hashable

from ..types import GraphLike

__all__ = ("migrate_legacy_phase_attributes",)


def migrate_legacy_phase_attributes(
    obj: GraphLike | Mapping[Hashable, MutableMapping[str, Any]]
) -> int: ...
