import pytest
from tnfr.helpers import alias_get


def test_alias_get_logs_on_error(caplog):
    d = {"x": "abc"}
    with caplog.at_level("WARNING"):
        result = alias_get(d, ["x"], int)
    assert result is None
    assert any("No se pudo convertir" in m for m in caplog.messages)


def test_alias_get_strict_raises():
    d = {"x": "abc"}
    with pytest.raises(ValueError):
        alias_get(d, ["x"], int, strict=True)
