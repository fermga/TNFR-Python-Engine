"""Tests for :mod:`tnfr.cli.utils`."""

from __future__ import annotations

import pytest

from tnfr.cli import utils
from tnfr.utils import normalize_optional_int


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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (0, 0),
        ("0", 0),
        (" 12 ", 12),
        ("auto", None),
        ("NONE", None),
        ("Null", None),
    ],
)
def test_normalize_optional_int_accepts_known_sentinels(
    raw: object, expected: int | None
) -> None:
    """The shared helper should normalise integers and reserved sentinel strings."""

    assert normalize_optional_int(raw, strict=True) == expected


@pytest.mark.parametrize("raw", ["", " ", "abc", "1.5"])
def test_normalize_optional_int_rejects_invalid_values(raw: object) -> None:
    """Non-integer inputs besides the sentinels must raise ``ValueError``."""

    with pytest.raises(ValueError):
        normalize_optional_int(raw, strict=True)


def test_parse_cli_variants_defaults_to_auto_when_missing() -> None:
    """Parsing ``None`` or empty iterables should default to the ``auto`` sentinel."""

    assert utils._parse_cli_variants(None) == [None]
    assert utils._parse_cli_variants([]) == [None]


def test_parse_cli_variants_deduplicates_preserving_order() -> None:
    """Order should be preserved while duplicates collapse to a single entry."""

    variants = utils._parse_cli_variants(["auto", "1", "auto", "2", "1", -3])

    assert variants == [None, 1, 2, -3]


def test_parse_cli_variants_propagates_coercion_errors() -> None:
    """Invalid entries should bubble coercion errors to the caller."""

    with pytest.raises(ValueError):
        utils._parse_cli_variants(["auto", "bad-value", "2"])
