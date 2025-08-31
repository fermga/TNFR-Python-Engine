from concurrent.futures import ThreadPoolExecutor

from tnfr.helpers import alias_lookup


def _worker(i):
    d = {}
    aliases = [f"k{i}", f"a{i}"]
    alias_lookup(d, aliases, int, value=i)
    return alias_lookup(d, aliases, int)


def test_alias_lookup_thread_safety():
    with ThreadPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(_worker, range(32)))
    assert results == list(range(32))
