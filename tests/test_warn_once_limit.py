import logging

from tnfr.logging_utils import warn_once


def test_warn_once_lru_limit(caplog):
    logger = logging.getLogger("warn_once_limit")
    warn = warn_once(logger, "%s", maxsize=2)
    with caplog.at_level(logging.WARNING):
        warn({"a": 1})
        warn({"a": 2})  # repeated, ignored
        warn({"b": 3})
        warn({"c": 4})  # evicts 'a'
        warn({"a": 5})  # 'a' warned again after eviction
    assert [r.message for r in caplog.records] == [
        "{'a': 1}",
        "{'b': 3}",
        "{'c': 4}",
        "{'a': 5}",
    ]


def test_warn_once_clear(caplog):
    logger = logging.getLogger("warn_once_clear")
    warn = warn_once(logger, "%s")
    with caplog.at_level(logging.WARNING):
        warn({"a": 1})
        warn({"a": 2})  # ignored
    assert [r.message for r in caplog.records] == ["{'a': 1}"]

    warn.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        warn({"a": 3})
    assert [r.message for r in caplog.records] == ["{'a': 3}"]
