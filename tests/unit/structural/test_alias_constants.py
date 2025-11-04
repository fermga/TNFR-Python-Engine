"""Ensure canonical alias constants remain synchronized with registry."""

from __future__ import annotations

import pytest

from tnfr.constants import get_aliases
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_D2VF,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_DSI,
    ALIAS_DVF,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)

_ALIAS_MAP = {
    "D2EPI": ALIAS_D2EPI,
    "D2VF": ALIAS_D2VF,
    "DEPI": ALIAS_DEPI,
    "DNFR": ALIAS_DNFR,
    "DSI": ALIAS_DSI,
    "DVF": ALIAS_DVF,
    "EPI": ALIAS_EPI,
    "EPI_KIND": ALIAS_EPI_KIND,
    "SI": ALIAS_SI,
    "THETA": ALIAS_THETA,
    "VF": ALIAS_VF,
}

@pytest.mark.parametrize("key, alias", sorted(_ALIAS_MAP.items()))
def test_alias_constants_match_registry(key: str, alias: tuple[str, ...]) -> None:
    """Alias constants must match the canonical registry entries."""

    assert alias == get_aliases(key)
