"""Utilities for CLI modules."""

from __future__ import annotations

from argparse import ArgumentTypeError
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from .._exact_time import finite_represented_real


def finite_real_argument(token: str) -> float:
    """Admit a decimal CLI scalar without losing a nonzero source to zero.

    Decimal retains the textual source while the shared numerical owner admits
    its binary64 materialization. Range and sign remain the consumer's contract.
    """
    if not isinstance(token, str):
        raise ArgumentTypeError("argument must be decimal text")
    try:
        source = Decimal(token)
        value = finite_represented_real(float(source), "argument")[0]
        if value == 0.0 and source != 0:
            raise ValueError("argument is nonzero but underflows to represented zero")
    except (InvalidOperation, OverflowError, TypeError, ValueError) as exc:
        raise ArgumentTypeError(str(exc)) from exc
    return value


def probability_argument(token: str) -> float:
    """Retain the original probability for the SDK's closed-interval check."""
    from ..sdk._topology import probability

    represented = finite_real_argument(token)
    # Zero is already admitted; avoid constructing an enormous denominator for
    # a zero literal with a large negative exponent.
    source = Fraction(token) if represented != 0.0 else 0
    try:
        return probability(source)
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


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
