"""Tests for add_edge callback validation."""

import pytest
from tnfr.node import add_edge, _validate_callbacks


def test_validate_callbacks_requires_callback_pair():
    with pytest.raises(ValueError):
        _validate_callbacks(lambda *_: False, None)
    with pytest.raises(ValueError):
        _validate_callbacks(None, lambda *_: None)


def test_validate_callbacks_requires_callables():
    with pytest.raises(TypeError):
        _validate_callbacks(object(), lambda *_: None)
    with pytest.raises(TypeError):
        _validate_callbacks(lambda *_: False, object())


def test_add_edge_validates_callbacks():
    with pytest.raises(ValueError):
        add_edge({}, 1, 2, 1.0, False, exists_cb=lambda *_: False)
    with pytest.raises(ValueError):
        add_edge({}, 1, 2, 1.0, False, set_cb=lambda *_: None)
    with pytest.raises(TypeError):
        add_edge({}, 1, 2, 1.0, False, exists_cb=object(), set_cb=lambda *_: None)
    with pytest.raises(TypeError):
        add_edge({}, 1, 2, 1.0, False, exists_cb=lambda *_: False, set_cb=object())
