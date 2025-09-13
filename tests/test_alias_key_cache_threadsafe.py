"""Pruebas concurrentes de AliasAccessor compartiendo instancia."""

from concurrent.futures import ThreadPoolExecutor

from tnfr.alias import AliasAccessor


def test_alias_accessor_key_cache_threadsafe():
    acc = AliasAccessor(int)
    d: dict[str, int] = {}
    aliases = ("k", "a")

    def worker(i: int) -> None:
        for _ in range(100):
            acc.set(d, aliases, i)
            val = acc.get(d, aliases)
            assert isinstance(val, int)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(worker, range(16)))

    assert len(acc._key_cache) == 1
