import threading

from tnfr.rng import _rng_for_step, get_rng


def test_rng_for_step_reproducible_sequence():
    get_rng.cache_clear()
    rng1 = _rng_for_step(123, 5)
    seq1 = [rng1.random() for _ in range(3)]
    get_rng.cache_clear()
    rng2 = _rng_for_step(123, 5)
    seq2 = [rng2.random() for _ in range(3)]
    assert seq1 == seq2


def test_rng_for_step_changes_with_step():
    get_rng.cache_clear()
    rng1 = _rng_for_step(123, 4)
    rng2 = _rng_for_step(123, 5)
    assert [rng1.random() for _ in range(3)] != [
        rng2.random() for _ in range(3)
    ]


def test_rng_for_step_thread_independence():
    get_rng.cache_clear()

    results = []
    errors = []
    lock = threading.Lock()

    def worker():
        try:
            rng = _rng_for_step(123, 5)
            seq = [rng.random() for _ in range(3)]
            with lock:
                results.append(seq)
        except Exception as e:  # pragma: no cover - should not happen
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert all(seq == results[0] for seq in results)
