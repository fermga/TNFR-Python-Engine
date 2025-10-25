"""Tests for :mod:`tnfr.cli.utils`."""

from __future__ import annotations

from tnfr.cli import utils


def test_spec_normalizes_dotted_option_to_dest_and_default_none() -> None:
    """A dotted or dashed option should normalize dest and default."""

    opt, params = utils.spec("--grammar.enabled")

    assert opt == "--grammar.enabled"
    assert params["dest"] == "grammar_enabled"
    assert params["default"] is None


def test_spec_respects_explicit_dest_and_default() -> None:
    """Explicit kwargs should remain unchanged when provided."""

    opt, params = utils.spec("--grammar-enabled", dest="custom_dest", default=False)

    assert opt == "--grammar-enabled"
    assert params["dest"] == "custom_dest"
    assert params["default"] is False
