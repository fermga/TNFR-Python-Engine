"""Unified tests covering ``AliasAccessor`` caching, defaults, and concurrency."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from tnfr.alias import AliasAccessor, _alias_cache


class CountingDict(dict):
    """Dictionary that counts membership checks for cache verification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contains_calls = 0

    def __contains__(self, item):  # type: ignore[override]
        self.contains_calls += 1
        return super().__contains__(item)


@pytest.fixture(autouse=True)
def clear_alias_cache():
    """Reset the global alias tuple cache around each test for determinism."""

    _alias_cache.cache_clear()
    yield
    _alias_cache.cache_clear()


@pytest.fixture
def accessor() -> AliasAccessor[int]:
    """Provide a fresh ``AliasAccessor`` instance for each test."""

    return AliasAccessor(int)


@pytest.mark.parametrize("use_instance", [False, True])
def test_accessor_get_and_set_work_with_functions_and_object(use_instance: bool) -> None:
    if use_instance:
        accessor = AliasAccessor(int)

        def getter(d, aliases, *, default=None):
            return accessor.get(d, aliases, default=default)

        def setter(d, aliases, value):
            return accessor.set(d, aliases, value)

    else:
        getter = lambda d, aliases, *, default=None: AliasAccessor(int).get(  # noqa: E731
            d, aliases, default=default
        )
        setter = lambda d, aliases, value: AliasAccessor(int).set(d, aliases, value)  # noqa: E731

    data = {"a": "1"}
    assert getter(data, ("a", "b"), default=None) == 1
    setter(data, ("b", "c"), "2")
    assert getter(data, ("b", "c"), default=None) == 2


def test_accessor_get_uses_default_value() -> None:
    acc = AliasAccessor(int, default=7)
    assert acc.get({}, ("x", "y")) == 7


@pytest.mark.parametrize("operation", ["get", "set"])
def test_alias_tuple_cache_reuses_validated_aliases(accessor: AliasAccessor[int], operation: str) -> None:
    if operation == "get":
        data = {"b": "1"}
        aliases = ("a", "b")

        def invoke() -> None:
            assert accessor.get(data, aliases) == 1

    else:
        data: dict[str, int | str] = {}
        aliases = ("x", "y")

        def invoke() -> None:
            accessor.set(data, aliases, "5")

    invoke()
    info1 = _alias_cache.cache_info()
    invoke()
    info2 = _alias_cache.cache_info()

    assert info2.hits == info1.hits + 1
    assert info2.misses == info1.misses


@pytest.mark.parametrize(
    "operation, mutate, expected_second",
    [
        ("get", False, 1),
        ("set", False, 1),
        ("get", True, 2),
    ],
)
def test_key_cache_limits_membership_checks(
    accessor: AliasAccessor[int], operation: str, mutate: bool, expected_second: int
) -> None:
    data = CountingDict({"c": "1"}) if operation == "get" else CountingDict()
    aliases = ("a", "b", "c") if operation == "get" else ("x", "y", "z")

    if operation == "get":
        assert accessor.get(data, aliases) == 1
    else:
        accessor.set(data, aliases, "1")

    assert data.contains_calls == 3
    data.contains_calls = 0

    if mutate:
        data["b" if operation == "get" else "y"] = "2"

    if operation == "get":
        expected_value = 2 if mutate else 1
        assert accessor.get(data, aliases) == expected_value
    else:
        accessor.set(data, aliases, "2")

    assert data.contains_calls == expected_second


@pytest.mark.parametrize("max_workers", [1, 16])
def test_key_cache_threadsafe(accessor: AliasAccessor[int], max_workers: int) -> None:
    shared: dict[str, int] = {}
    aliases = ("k", "a")

    def worker(i: int) -> None:
        for _ in range(100):
            accessor.set(shared, aliases, i)
            value = accessor.get(shared, aliases)
            assert isinstance(value, int)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(worker, range(16)))

    assert len(shared) == 1
    assert set(shared.keys()) == {"k"}
    final_value = accessor.get(shared, aliases)
    assert isinstance(final_value, int)
    assert shared["k"] == final_value
