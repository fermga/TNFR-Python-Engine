import pytest
from tnfr.helpers import ensure_collection


def test_wraps_string():
    assert ensure_collection("node") == ("node",)


def test_wraps_bytes():
    data = b"node"
    assert ensure_collection(data) == (data,)


def test_wraps_bytearray():
    arr = bytearray(b"node")
    assert ensure_collection(arr) == (arr,)


def test_max_materialize_limit():
    gen = (i for i in range(5))
    with pytest.raises(ValueError):
        ensure_collection(gen, max_materialize=3)


def test_default_limit_enforced():
    gen = (i for i in range(1001))
    with pytest.raises(ValueError):
        ensure_collection(gen)


def test_none_uses_default_limit():
    gen = (i for i in range(1001))
    with pytest.raises(ValueError):
        ensure_collection(gen, max_materialize=None)


def test_non_iterable_error():
    with pytest.raises(TypeError):
        ensure_collection(42)  # type: ignore[arg-type]

