from collections.abc import Mapping, MutableMapping
from typing import Any, Hashable

from ..types import GraphLike

__all__ = ("migrate_legacy_phase_attributes", "migrate_legacy_remesh_cooldown")


def migrate_legacy_phase_attributes(
    obj: GraphLike | Mapping[Hashable, MutableMapping[str, Any]]
) -> int: ...


def migrate_legacy_remesh_cooldown(
    obj: GraphLike | MutableMapping[str, Any]
) -> int: ...
