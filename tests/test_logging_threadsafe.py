from concurrent.futures import ThreadPoolExecutor
import logging

import tnfr.logging_utils as logging_utils


def _worker():
    logging_utils.get_logger("test_logger")


def test_get_logger_threadsafe():
    root = logging.getLogger()
    root.handlers.clear()
    logging_utils._LOGGING_CONFIGURED = False
    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(lambda _: _worker(), range(64)))
    assert len(root.handlers) == 1


def test_get_logger_preserves_existing_level():
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.ERROR)
    logging_utils._LOGGING_CONFIGURED = False
    logging_utils.get_logger("test_logger")
    assert root.level == logging.ERROR
    root.setLevel(logging.WARNING)


def test_get_logger_sets_level_when_notset():
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    logging_utils._LOGGING_CONFIGURED = False
    logging_utils.get_logger("test_logger")
    assert root.level == logging.INFO
    root.setLevel(logging.WARNING)
