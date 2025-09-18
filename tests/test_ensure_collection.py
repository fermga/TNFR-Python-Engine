import pytest
from tnfr.collections_utils import ensure_collection


def test_existing_collection_returned_as_is():
    items = [0, 1, 2]
    assert ensure_collection(items) is items


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
    assert (
        str(exc.value)
        == "Iterable produced 4 items, exceeds limit 3; first items: [0, 1, 2]"
    )
    assert list(gen) == [4]


def test_materialization_at_limit_allowed():
    gen = (i for i in range(3))
    assert ensure_collection(gen, max_materialize=3) == (0, 1, 2)


def test_negative_max_materialize_error():
    gen = (i for i in range(5))
    with pytest.raises(ValueError):
        ensure_collection(gen, max_materialize=-1)


def test_zero_limit_returns_empty():
    gen = (i for i in range(5))
    assert ensure_collection(gen, max_materialize=0) == ()


def test_zero_limit_does_not_consume_iterator():
    gen = (i for i in range(5))
    result = ensure_collection(gen, max_materialize=0)
    assert result == ()
    # The generator should remain untouched
    assert list(gen) == [0, 1, 2, 3, 4]


def test_default_limit_enforced():
    gen = (i for i in range(1001))
    with pytest.raises(ValueError):
        ensure_collection(gen)


def test_none_disables_limit():
    gen = (i for i in range(1001))
    data = ensure_collection(gen, max_materialize=None)
    assert len(data) == 1001


def test_custom_error_msg():
    gen = (i for i in range(5))
    with pytest.raises(ValueError, match="custom message"):
        ensure_collection(gen, max_materialize=3, error_msg="custom message")


def test_non_iterable_error():
    with pytest.raises(TypeError):
        ensure_collection(42)  # type: ignore[arg-type]


def test_max_materialize_accepts_float():
    gen = (i for i in range(3))
    assert ensure_collection(gen, max_materialize=3.0) == (0, 1, 2)


def test_map_iterable_materialized():
    data = map(lambda x: x * 2, range(3))
    assert ensure_collection(data, max_materialize=3) == (0, 2, 4)
    # Iterator should be exhausted
    assert list(data) == []
