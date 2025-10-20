"""Tests ensuring operator name constants stay aligned with registry."""

import pytest

from tnfr.config import operator_names as names
from tnfr.operators.registry import OPERADORES, discover_operators


def test_registry_matches_operator_constants() -> None:
    discover_operators()
    assert set(OPERADORES.keys()) == names.ALL_OPERATOR_NAMES


def test_validation_sets_are_subsets() -> None:
    assert names.INICIO_VALIDOS <= names.ALL_OPERATOR_NAMES
    assert names.TRAMO_INTERMEDIO <= names.ALL_OPERATOR_NAMES
    assert names.CIERRE_VALIDO <= names.ALL_OPERATOR_NAMES
    assert names.AUTOORGANIZACION in names.ALL_OPERATOR_NAMES
    assert names.AUTOORGANIZACION_CIERRES <= names.ALL_OPERATOR_NAMES


def test_alias_tables_cover_both_languages() -> None:
    assert names.SPANISH_OPERATOR_NAMES <= names.ALL_OPERATOR_NAMES
    assert names.ENGLISH_OPERATOR_NAMES <= names.ALL_OPERATOR_NAMES
    for canonical, aliases in names.ALIASES_BY_CANONICAL.items():
        assert canonical in aliases
        for alias in aliases:
            assert names.canonical_operator_name(alias) == canonical


def test_operator_display_names_include_aliases() -> None:
    label = names.operator_display_name(names.EMISION)
    assert names.EMISION in label and names.EMISSION in label


def test_spanish_tokens_raise_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning):
        assert names.canonical_operator_name(names.EMISION) == names.EMISSION


def test_registry_accepts_spanish_aliases_via_canonical_lookup() -> None:
    discover_operators()
    cls = OPERADORES[names.canonical_operator_name(names.RECEPCION)]
    assert cls.__name__ == "Reception"
