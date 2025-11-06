import hashlib
import random
import struct

import networkx as nx

from tnfr import rng as rng_mod
from tnfr.rng import cache_enabled, clear_rng_cache, make_rng


def _derive_seed(seed: int, key: int) -> int:
    seed_bytes = struct.pack(
        ">QQ",
        int(seed) & 0xFFFFFFFFFFFFFFFF,
        int(key) & 0xFFFFFFFFFFFFFFFF,
    )
    return int.from_bytes(hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big")


def test_make_rng_reproducible_sequence():
    clear_rng_cache()
    seed = 123
    key = 456
    rng1 = make_rng(seed, key)
    seq1 = [rng1.random() for _ in range(3)]
    rng2 = make_rng(seed, key)
    seq2 = [rng2.random() for _ in range(3)]

    seed_int = _derive_seed(seed, key)
    rng_ref = random.Random(seed_int)
    exp = [rng_ref.random() for _ in range(3)]

    assert seq1 == exp
    assert seq2 == exp


def test_cache_size_updates_from_graph():
    G = nx.Graph()

    # Initial state uses default size
    cache_enabled(G)
    default_size = rng_mod.DEFAULTS["JITTER_CACHE_SIZE"]
    assert rng_mod._CACHE_MAXSIZE == default_size

    G.graph["JITTER_CACHE_SIZE"] = 0
    cache_enabled(G)
    assert rng_mod._CACHE_MAXSIZE == 0

    G.graph["JITTER_CACHE_SIZE"] = 3
    make_rng(1, 2, G)
    assert rng_mod._CACHE_MAXSIZE == 3
