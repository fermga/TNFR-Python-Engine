import importlib
import logging


def reload_logging_modules():
    import tnfr.logging_utils as logging_proxy
    import tnfr.utils.init as logging_core

    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.NOTSET)

    logging_core = importlib.reload(logging_core)
    logging_proxy = importlib.reload(logging_proxy)
    return logging_proxy, logging_core


def test_logging_utils_reflects_logging_configured_flag():
    proxy, core = reload_logging_modules()

    assert core._LOGGING_CONFIGURED is False
    assert proxy._LOGGING_CONFIGURED is False

    proxy.get_logger("tnfr.test.logging_proxy_state")

    assert core._LOGGING_CONFIGURED is True
    assert proxy._LOGGING_CONFIGURED is True
