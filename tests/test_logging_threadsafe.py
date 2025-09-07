from concurrent.futures import ThreadPoolExecutor
import logging

from tnfr.logging_utils import get_logger


def _worker():
    get_logger("test_logger")


def test_get_logger_threadsafe():
    root = logging.getLogger()
    root.handlers.clear()
    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(lambda _: _worker(), range(64)))
    assert len(root.handlers) == 1
