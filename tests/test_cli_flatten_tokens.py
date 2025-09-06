from tnfr.cli import _flatten_tokens


def test_flatten_tokens_accepts_tuples():
    data = ("a", ("b", ["c", ("d",)]))
    assert list(_flatten_tokens(data)) == ["a", "b", "c", "d"]
