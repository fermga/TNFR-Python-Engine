"""Tests for add_edge callback validation."""

import pytest
from tnfr.node import add_edge


def test_add_edge_requires_callback_pair():
    with pytest.raises(ValueError):
        add_edge({}, 1, 2, 1.0, False, exists_cb=lambda *_: False)
    with pytest.raises(ValueError):
        add_edge({}, 1, 2, 1.0, False, set_cb=lambda *_: None)
