"""Tests for caching effective alias keys to reduce iterations."""

from tnfr.alias import AliasAccessor


class CountingDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contains_calls = 0

    def __contains__(self, item):
        self.contains_calls += 1
        return super().__contains__(item)


def test_get_reuses_cached_key():
    d = CountingDict({"c": "1"})
    acc = AliasAccessor(int)
    aliases = ("a", "b", "c")

    assert acc.get(d, aliases) == 1
    first = d.contains_calls

    assert acc.get(d, aliases) == 1
    second = d.contains_calls - first

    assert first == 3
    assert second == 1


def test_manual_key_change_invalidates_cache():
    d = CountingDict({"c": "1"})
    acc = AliasAccessor(int)
    aliases = ("a", "b", "c")

    acc.get(d, aliases)
    d.contains_calls = 0
    d["b"] = "2"  # manual addition of earlier alias

    assert acc.get(d, aliases) == 2
    assert d.contains_calls == 2


def test_set_reuses_cached_key():
    d = CountingDict()
    acc = AliasAccessor(int)
    aliases = ("x", "y", "z")

    acc.set(d, aliases, "1")
    first = d.contains_calls

    acc.set(d, aliases, "2")
    second = d.contains_calls - first

    assert first == 3
    assert second == 1
