"""Tests ensuring operator name constants stay aligned with registry."""

import pytest

from tnfr.config import operator_names as names
from tnfr.operators import registry as registry_module
from tnfr.operators.registry import OPERATORS, discover_operators, get_operator_class


def test_registry_matches_operator_constants() -> None:
    discover_operators()
    assert set(OPERATORS.keys()) == names.ALL_OPERATOR_NAMES


def test_validation_sets_are_subsets() -> None:
    assert names.VALID_START_OPERATORS <= names.ALL_OPERATOR_NAMES
    assert names.INTERMEDIATE_OPERATORS <= names.ALL_OPERATOR_NAMES
    assert names.VALID_END_OPERATORS <= names.ALL_OPERATOR_NAMES
    assert names.SELF_ORGANIZATION in names.ALL_OPERATOR_NAMES
    assert names.SELF_ORGANIZATION_CLOSURES <= names.ALL_OPERATOR_NAMES


@pytest.mark.parametrize(
    ("legacy_name", "preferred_name"),
    (
        ("INICIO_VALIDOS", "VALID_START_OPERATORS"),
        ("TRAMO_INTERMEDIO", "INTERMEDIATE_OPERATORS"),
        ("CIERRE_VALIDO", "VALID_END_OPERATORS"),
        ("AUTOORGANIZACION_CIERRES", "SELF_ORGANIZATION_CLOSURES"),
    ),
)
def test_spanish_aliases_raise_attribute_error(legacy_name: str, preferred_name: str) -> None:
    with pytest.raises(AttributeError) as exc_info:
        getattr(names, legacy_name)
    message = str(exc_info.value)
    assert legacy_name in message
    assert preferred_name in message


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


def test_registry_exposes_only_english_collection_name() -> None:
    with pytest.raises(AttributeError) as exc_info:
        getattr(registry_module, "OPERADORES")
    assert "OPERADORES" in str(exc_info.value)
