"""Tests for AliasAccessor key cache behavior and thread safety."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from tnfr.alias import AliasAccessor


class CountingDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contains_calls = 0

    def __contains__(self, item):
        self.contains_calls += 1
        return super().__contains__(item)


@pytest.fixture
def accessor():
    return AliasAccessor(int)


@pytest.mark.parametrize(
    "operation, modify, expected_second",
    [
        ("get", False, 1),
        ("set", False, 1),
        ("get", True, 2),
    ],
)
def test_key_cache_iteration_counts(accessor, operation, modify, expected_second):
    d = CountingDict({"c": "1"}) if operation != "set" else CountingDict()
    aliases = ("a", "b", "c") if operation != "set" else ("x", "y", "z")

    if operation == "get":
        assert accessor.get(d, aliases) == 1
    else:
        accessor.set(d, aliases, "1")

    assert d.contains_calls == 3
    d.contains_calls = 0

    if modify:
        d["b" if operation == "get" else "y"] = "2"

    if operation == "get":
        expected_value = 2 if modify else 1
        assert accessor.get(d, aliases) == expected_value
    else:
        accessor.set(d, aliases, "2")

    assert d.contains_calls == expected_second


@pytest.mark.parametrize("max_workers", [1, 16])
def test_key_cache_threadsafe(accessor, max_workers):
    d: dict[str, int] = {}
    aliases = ("k", "a")

    def worker(i: int) -> None:
        for _ in range(100):
            accessor.set(d, aliases, i)
            val = accessor.get(d, aliases)
            assert isinstance(val, int)

    if max_workers == 1:
        list(map(worker, range(16)))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(worker, range(16)))

    assert len(accessor._key_cache) == 1
