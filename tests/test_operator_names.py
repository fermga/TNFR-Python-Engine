"""Tests ensuring operator name constants stay aligned with registry."""

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
