from tnfr.config import (
    get_precision_mode,
    set_precision_mode,
    get_telemetry_density,
    set_telemetry_density,
    get_diagnostics_level,
    set_diagnostics_level,
)


def test_precision_mode_default_and_setter():
    # Default should be "standard"
    assert get_precision_mode() == "standard"

    # Changing the mode should be reflected by the getter
    set_precision_mode("high")
    assert get_precision_mode() == "high"

    # Restore default to avoid leaking state to other tests
    set_precision_mode("standard")


def test_telemetry_density_default_and_setter():
    # Default should be "low"
    assert get_telemetry_density() == "low"

    set_telemetry_density("high")
    assert get_telemetry_density() == "high"

    # Restore default
    set_telemetry_density("low")


def test_diagnostics_level_default_and_setter():
    # Default should be "off"
    assert get_diagnostics_level() == "off"

    set_diagnostics_level("basic")
    assert get_diagnostics_level() == "basic"

    # Restore default
    set_diagnostics_level("off")
