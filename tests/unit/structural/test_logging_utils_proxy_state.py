import logging


def test_utils_module_reflects_logging_configured_flag():
    import tnfr.utils as utils_pkg
    import tnfr.utils.init as logging_core

    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.NOTSET)
    logging_core._reset_logging_state()

    assert logging_core._LOGGING_CONFIGURED is False
    assert utils_pkg._LOGGING_CONFIGURED is False

    utils_pkg.get_logger("tnfr.test.logging_proxy_state")

    assert logging_core._LOGGING_CONFIGURED is True
    assert utils_pkg._LOGGING_CONFIGURED is True
