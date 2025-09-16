import importlib
import logging

import tnfr.logging_utils as logging_utils


def reload_logging_utils():
    global logging_utils
    logging_utils = importlib.reload(logging_utils)
    return logging_utils


def test_get_logger_configures_root_once():
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    reload_logging_utils()
    logging_utils.get_logger("test")
    assert len(root.handlers) == 1
    assert root.level == logging.INFO
    assert root.handlers[0].formatter is not None
    logging_utils.get_logger("again")
    assert len(root.handlers) == 1


def test_warn_once_with_mapping(caplog):
    logger = logging.getLogger("tnfr.test.warn_once.mapping")
    warn_once = logging_utils.warn_once(logger, "values: %s")

    with caplog.at_level(logging.WARNING):
        warn_once({"a": 1, "b": 2})
        warn_once({"a": 3, "c": 4})

    messages = [record.message for record in caplog.records]
    assert messages == ["values: {'a': 1, 'b': 2}", "values: {'c': 4}"]


def test_warn_once_with_key_payload(caplog):
    logger = logging.getLogger("tnfr.test.warn_once.key")
    warn_once = logging_utils.warn_once(logger, "value: %s")

    with caplog.at_level(logging.WARNING):
        warn_once("alpha", "alpha warning")
        warn_once("alpha", "alpha warning repeat")
        warn_once("beta", "beta warning")

    messages = [record.message for record in caplog.records]
    assert messages == ["value: alpha warning", "value: beta warning"]


def test_warn_once_clear_and_unbounded(caplog):
    logger = logging.getLogger("tnfr.test.warn_once.clear")
    warn_once = logging_utils.warn_once(logger, "value: %s")

    with caplog.at_level(logging.WARNING):
        warn_once("gamma", "first")
        warn_once.clear()
        warn_once("gamma", "second")
        warn_once("gamma", "third")

    messages = [record.message for record in caplog.records]
    assert messages == ["value: first", "value: second"]

    caplog.clear()
    unbounded = logging_utils.warn_once(logger, "value: %s", maxsize=0)
    with caplog.at_level(logging.WARNING):
        unbounded("repeat", "first")
        unbounded("repeat", "second")

    messages = [record.message for record in caplog.records]
    assert messages == ["value: first", "value: second"]
