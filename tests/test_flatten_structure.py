import pytest

from tnfr.collections_utils import flatten_structure


def _make_set() -> set[int]:
    return {1, 2, 3}


def _make_generator():
    def _simple_gen():
        for i in range(3):
            yield i

    return _simple_gen()


def _make_nested_tuple() -> tuple[object, ...]:
    return ("a", ("b", ["c", ("d",)]))


def _make_deep_nesting() -> object:
    data: object = "z"
    for _ in range(1000):
        data = [data]
    return data


@pytest.mark.parametrize(
    ("factory", "expected", "coerce"),
    (
        pytest.param(_make_set, {1, 2, 3}, set, id="set"),
        pytest.param(_make_generator, [0, 1, 2], list, id="generator"),
        pytest.param(_make_nested_tuple, ["a", "b", "c", "d"], list, id="tuple"),
        pytest.param(_make_deep_nesting, ["z"], list, id="deep-nesting"),
    ),
)
def test_flatten_structure(factory, expected, coerce):
    assert coerce(flatten_structure(factory())) == expected
