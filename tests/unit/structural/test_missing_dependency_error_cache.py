from tnfr.io import _MISSING_TOML_ERROR, _MISSING_YAML_ERROR


def test_missing_dependency_error_cached() -> None:
    assert issubclass(_MISSING_TOML_ERROR, Exception)
    assert (
        _MISSING_TOML_ERROR.__doc__
        == "Fallback error used when tomllib/tomli is missing."
    )
    assert issubclass(_MISSING_YAML_ERROR, Exception)
    assert _MISSING_YAML_ERROR.__doc__ == (
        "Fallback error used when pyyaml is missing."
    )
