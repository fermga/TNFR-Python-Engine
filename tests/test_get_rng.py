import random
import hashlib
import struct
from tnfr.rng import get_rng, clear_rng_cache


def _derive_seed(seed: int, key: int) -> int:
    seed_bytes = struct.pack(
        ">QQ",
        int(seed) & 0xFFFFFFFFFFFFFFFF,
        int(key) & 0xFFFFFFFFFFFFFFFF,
    )
    return int.from_bytes(
        hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big"
    )


def test_get_rng_reproducible_sequence():
    clear_rng_cache()
    seed = 123
    key = 456
    rng1 = get_rng(seed, key)
    seq1 = [rng1.random() for _ in range(3)]
    rng2 = get_rng(seed, key)
    seq2 = [rng2.random() for _ in range(3)]

    seed_int = _derive_seed(seed, key)
    rng_ref = random.Random(seed_int)
    exp = [rng_ref.random() for _ in range(3)]

    assert seq1 == exp
    assert seq2 == exp
