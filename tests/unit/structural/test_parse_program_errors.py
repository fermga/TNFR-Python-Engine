"""Focused tests for error handling in ``parse_program_tokens``."""

import pytest

from tnfr.flatten import parse_program_tokens


def test_parse_program_tokens_rejects_multiple_keys_mapping():
    payload = {"WAIT": 1, "THOL": {"body": []}}
    with pytest.raises(ValueError, match="Invalid token mapping"):
        parse_program_tokens([payload])


def test_parse_program_tokens_rejects_unrecognized_token_key():
    payload = {"BAD": 1}
    with pytest.raises(ValueError, match="Unrecognized token"):
        parse_program_tokens([payload])


def test_parse_program_tokens_rejects_unknown_thol_closing_glyph():
    payload = {"THOL": {"body": [], "close": "NOPE"}}
    with pytest.raises(ValueError, match="Unknown closing glyph"):
        parse_program_tokens([payload])


def test_parse_program_tokens_requires_thol_close_glyph_type():
    payload = {"THOL": {"body": [], "close": 123}}
    with pytest.raises(TypeError, match="THOL close glyph must be"):
        parse_program_tokens([payload])


def test_parse_program_tokens_rejects_non_iterable_source():
    with pytest.raises(TypeError, match="123 is not iterable"):
        parse_program_tokens(123)


def test_parse_program_tokens_requires_thol_mapping():
    with pytest.raises(TypeError, match="THOL specification must be a mapping"):
        parse_program_tokens([{"THOL": "bad"}])
