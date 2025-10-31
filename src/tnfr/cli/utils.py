"""Utilities for CLI modules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..utils import normalize_optional_int


def spec(opt: str, /, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    """Create an argument specification pair.

    Parameters
    ----------
    opt:
        Option string to register, e.g. ``"--foo"``.
    **kwargs:
        Keyword arguments forwarded to
        :meth:`argparse.ArgumentParser.add_argument`.

    Returns
    -------
    tuple[str, dict[str, Any]]
        A pair suitable for collecting into argument specification sequences.
        If ``dest`` is not provided it is
        derived from ``opt`` by stripping leading dashes and replacing dots and
        hyphens with underscores. ``default`` defaults to ``None`` so missing
        options can be filtered easily.
    """

    kwargs = dict(kwargs)
    kwargs.setdefault("dest", opt.lstrip("-").replace("-", "_").replace(".", "_"))
    kwargs.setdefault("default", None)
    return opt, kwargs


def _parse_cli_variants(values: Iterable[Any] | None) -> list[int | None]:
    """Return a stable list of integer/``None`` variants for the CLI options."""

    if values is None:
        return [None]
    parsed: list[int | None] = []
    seen: set[int | None] = set()
    for raw in values:
        coerced = normalize_optional_int(raw, strict=True)
        if coerced in seen:
            continue
        seen.add(coerced)
        parsed.append(coerced)
    return parsed or [None]
