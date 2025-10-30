from __future__ import annotations

from collections.abc import Iterable
from typing import Any

def spec(opt: str, /, **kwargs: Any) -> tuple[str, dict[str, Any]]: ...

def _coerce_optional_int(value: Any) -> int | None: ...

def _parse_cli_variants(values: Iterable[Any] | None) -> list[int | None]: ...
