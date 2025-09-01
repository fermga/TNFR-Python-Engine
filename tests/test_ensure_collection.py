import pytest
from tnfr.helpers import ensure_collection


def test_wraps_string():
    assert ensure_collection("node") == ["node"]


def test_wraps_bytes():
    data = b"node"
    assert ensure_collection(data) == [data]


def test_wraps_bytearray():
    arr = bytearray(b"node")
    assert ensure_collection(arr) == [arr]


def test_non_iterable_error():
    with pytest.raises(TypeError):
        ensure_collection(42)  # type: ignore[arg-type]

