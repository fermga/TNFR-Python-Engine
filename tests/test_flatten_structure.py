from tnfr.collections_utils import flatten_structure


def test_flatten_structure_set():
    items = {1, 2, 3}
    assert set(flatten_structure(items)) == items


def _simple_gen():
    for i in range(3):
        yield i


def test_flatten_structure_generator():
    assert list(flatten_structure(_simple_gen())) == [0, 1, 2]
