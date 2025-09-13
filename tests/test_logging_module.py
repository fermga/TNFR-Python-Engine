import importlib
import logging

import tnfr.logging as tnfr_logging


def reload_logging():
    global tnfr_logging
    tnfr_logging = importlib.reload(tnfr_logging)
    return tnfr_logging


def test_get_module_logger_configures_root_once():
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    reload_logging()
    tnfr_logging.get_module_logger("test")
    assert len(root.handlers) == 1
    assert root.level == logging.INFO
    assert root.handlers[0].formatter is not None
    tnfr_logging.get_module_logger("again")
    assert len(root.handlers) == 1
