import importlib
import logging

import tnfr.logging_utils as logging_utils
import tnfr.logging_base as logging_base


def reload_logging():
    global logging_utils
    importlib.reload(logging_base)
    logging_utils = importlib.reload(logging_utils)
    return logging_utils


def test_get_module_logger_configures_root_once():
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    reload_logging()
    logging_utils.get_module_logger("test")
    assert len(root.handlers) == 1
    assert root.level == logging.INFO
    assert root.handlers[0].formatter is not None
    logging_utils.get_module_logger("again")
    assert len(root.handlers) == 1
