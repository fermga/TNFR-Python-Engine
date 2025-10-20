"""Tests ensuring operator name constants stay aligned with registry."""

import pytest

from tnfr.config import operator_names as names
from tnfr.operators.registry import OPERATORS, discover_operators, get_operator_class


def test_registry_matches_operator_constants() -> None:
    discover_operators()
    assert set(OPERATORS.keys()) == names.ALL_OPERATOR_NAMES


def test_validation_sets_are_subsets() -> None:
    assert names.INICIO_VALIDOS <= names.ALL_OPERATOR_NAMES
    assert names.TRAMO_INTERMEDIO <= names.ALL_OPERATOR_NAMES
    assert names.CIERRE_VALIDO <= names.ALL_OPERATOR_NAMES
    assert names.SELF_ORGANIZATION in names.ALL_OPERATOR_NAMES
    assert names.AUTOORGANIZACION_CIERRES <= names.ALL_OPERATOR_NAMES


def test_canonical_lookup_is_passthrough_for_english_tokens() -> None:
    for token in names.CANONICAL_OPERATOR_NAMES:
        assert names.canonical_operator_name(token) == token


def test_operator_display_name_returns_canonical_token() -> None:
    assert names.operator_display_name(names.EMISSION) == names.EMISSION
    assert names.operator_display_name("unknown") == "unknown"


def test_get_operator_class_rejects_spanish_tokens() -> None:
    discover_operators()
    with pytest.raises(KeyError):
        get_operator_class("emision")
