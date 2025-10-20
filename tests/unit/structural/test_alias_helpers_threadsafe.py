"""Unit tests verifying thread-safe behavior of alias helpers."""



from concurrent.futures import ThreadPoolExecutor

from tnfr.alias import AliasAccessor


def _worker(i):
    d = {}
    aliases = (f"k{i}", f"a{i}")
    acc = AliasAccessor(int)
    acc.set(d, aliases, i)
    return acc.get(d, aliases)


def test_alias_helpers_thread_safety():
    with ThreadPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(_worker, range(32)))
    assert results == list(range(32))
