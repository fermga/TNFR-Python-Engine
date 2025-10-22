import importlib
import logging
from concurrent.futures import ThreadPoolExecutor

import tnfr.utils.init as logging_utils


def reload_logging_utils():
    """Reload logging_utils after clearing root handlers."""

    global logging_utils
    logging_utils = importlib.reload(logging_utils)
    return logging_utils


def _worker():
    logging_utils.get_logger("test_logger")


def test_get_logger_threadsafe():
    root = logging.getLogger()
    root.handlers.clear()
    reload_logging_utils()
    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(lambda _: _worker(), range(64)))
    assert len(root.handlers) == 1


def test_get_logger_preserves_existing_level():
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.ERROR)
    reload_logging_utils()
    logging_utils.get_logger("test_logger")
    assert root.level == logging.ERROR
    root.setLevel(logging.WARNING)


def test_get_logger_sets_level_when_notset():
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    reload_logging_utils()
    logging_utils.get_logger("test_logger")
    assert root.level == logging.INFO
    root.setLevel(logging.WARNING)


def test_get_logger_multiple_calls_do_not_reconfigure_root():
    root = logging.getLogger()
    root.handlers.clear()
    reload_logging_utils()
    handlers_before = root.handlers
    level_before = root.level
    logging_utils.get_logger("first")
    logging_utils.get_logger("second")
    assert root.handlers is handlers_before
    assert root.level == level_before
