from tnfr.io import _missing_dependency_error


def test_missing_dependency_error_cached() -> None:
    cls1 = _missing_dependency_error("fake_dep")
    cls2 = _missing_dependency_error("fake_dep")
    assert cls1 is cls2
    assert issubclass(cls1, Exception)
    assert cls1.__doc__ == "Fallback error used when fake_dep is missing."
    cls3 = _missing_dependency_error("other_dep")
    assert cls3 is not cls1
