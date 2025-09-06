import pytest
from tnfr.collections_utils import ensure_collection


def test_wraps_string():
    assert ensure_collection("node") == ("node",)


def test_wraps_bytes():
    data = b"node"
    assert ensure_collection(data) == (data,)


def test_wraps_bytearray():
    arr = bytearray(b"node")
    assert ensure_collection(arr) == (arr,)


def test_iterable_not_iterator_materialized():
    class CustomIterable:
        def __iter__(self):
            return (i for i in range(3))

    it = CustomIterable()
    assert ensure_collection(it, max_materialize=3) == (0, 1, 2)


def test_max_materialize_limit():
    gen = (i for i in range(5))
    with pytest.raises(ValueError) as exc:
        ensure_collection(gen, max_materialize=3)
    assert str(exc.value) == "Iterable con más de 3 elementos"
    assert list(gen) == [4]


def test_materialization_at_limit_allowed():
    gen = (i for i in range(3))
    assert ensure_collection(gen, max_materialize=3) == (0, 1, 2)


def test_negative_max_materialize_error():
    gen = (i for i in range(5))
    with pytest.raises(ValueError):
        ensure_collection(gen, max_materialize=-1)


def test_default_limit_enforced():
    gen = (i for i in range(1001))
    with pytest.raises(ValueError):
        ensure_collection(gen)


def test_none_disables_limit():
    gen = (i for i in range(1001))
    data = ensure_collection(gen, max_materialize=None)
    assert len(data) == 1001


def test_non_iterable_error():
    with pytest.raises(TypeError):
        ensure_collection(42)  # type: ignore[arg-type]
