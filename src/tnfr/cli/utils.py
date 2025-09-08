"""Utilities for CLI modules."""

from typing import Any


def specs(*pairs: tuple[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Build a list of argument specifications.

    Each pair contains an option string and a mapping of keyword arguments for
    :meth:`argparse.ArgumentParser.add_argument`. The helper simply converts the
    variadic sequence into a list, which makes the specs reusable across
    parsers.
    """

    return list(pairs)
