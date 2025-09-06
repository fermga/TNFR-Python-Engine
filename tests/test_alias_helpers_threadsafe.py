"""Pruebas de alias helpers threadsafe."""

from concurrent.futures import ThreadPoolExecutor

from tnfr.alias import alias_get, alias_set, _validate_aliases


def _worker(i):
    d = {}
    aliases = _validate_aliases((f"k{i}", f"a{i}"))
    alias_set(d, aliases, int, i)
    return alias_get(d, aliases, int)


def test_alias_helpers_thread_safety():
    with ThreadPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(_worker, range(32)))
    assert results == list(range(32))
