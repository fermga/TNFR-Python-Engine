"""Tests for ``value_utils`` module."""

import logging

from tnfr.utils import convert_value


def test_convert_value_logs_debug_by_default(caplog):
    def conv(_):
        raise ValueError("bad")

    with caplog.at_level(logging.DEBUG, logger="tnfr.utils.data"):
        ok, result = convert_value("x", conv, key="foo")

    assert not ok and result is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG


def test_convert_value_logs_custom_level(caplog):
    def conv(_):
        raise ValueError("bad")

    with caplog.at_level(logging.INFO, logger="tnfr.utils.data"):
        ok, result = convert_value(
            "x", conv, key="foo", log_level=logging.INFO
        )

    assert not ok and result is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
