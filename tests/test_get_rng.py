import random
import hashlib
from tnfr.helpers import get_rng

def _derive_seed(seed: int, key: int) -> int:
    seed_input = (int(seed), int(key))
    return int.from_bytes(
        hashlib.blake2b(repr(seed_input).encode()).digest()[:8], "big"
    )

def test_get_rng_reproducible_sequence():
    get_rng.cache_clear()
    seed = 123
    key = 456
    rng1 = get_rng(seed, key)
    seq1 = [rng1.random() for _ in range(3)]
    rng2 = get_rng(seed, key)
    seq2 = [rng2.random() for _ in range(3)]

    seed_int = _derive_seed(seed, key)
    rng_ref = random.Random(seed_int)
    exp1 = [rng_ref.random() for _ in range(3)]
    exp2 = [rng_ref.random() for _ in range(3)]

    assert seq1 == exp1
    assert seq2 == exp2
