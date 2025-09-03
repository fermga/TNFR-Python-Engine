"""Pruebas de defaults integrity."""

from collections import ChainMap

from tnfr.constants import (
    DEFAULTS,
    CORE_DEFAULTS,
    INIT_DEFAULTS,
    REMESH_DEFAULTS,
    METRIC_DEFAULTS,
)
from tnfr.constants import core, init, metric


def test_defaults_is_union_of_parts():
    expected = dict(
        ChainMap(METRIC_DEFAULTS, REMESH_DEFAULTS, INIT_DEFAULTS, CORE_DEFAULTS)
    )
    assert DEFAULTS == expected


def test_defaults_contains_submodule_parts():
    for part in (core.DEFAULTS_PART, init.DEFAULTS_PART, metric.DEFAULTS_PART):
        for k, v in part.items():
            assert DEFAULTS[k] == v
