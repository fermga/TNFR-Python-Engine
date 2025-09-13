from tnfr.collections_utils import flatten_structure


def test_flatten_structure_accepts_tuples():
    data = ("a", ("b", ["c", ("d",)]))
    assert list(flatten_structure(data)) == ["a", "b", "c", "d"]


def test_flatten_structure_handles_deep_nesting():
    data = "z"
    for _ in range(1000):
        data = [data]
    assert list(flatten_structure(data)) == ["z"]
