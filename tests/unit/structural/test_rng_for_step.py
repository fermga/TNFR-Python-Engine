import threading

from tnfr.rng import _rng_for_step, clear_rng_cache


def test_rng_for_step_reproducible_sequence():
    clear_rng_cache()
    rng1 = _rng_for_step(123, 5)
    seq1 = [rng1.random() for _ in range(3)]
    clear_rng_cache()
    rng2 = _rng_for_step(123, 5)
    seq2 = [rng2.random() for _ in range(3)]
    assert seq1 == seq2


def test_rng_for_step_changes_with_step():
    clear_rng_cache()
    rng1 = _rng_for_step(123, 4)
    rng2 = _rng_for_step(123, 5)
    assert [rng1.random() for _ in range(3)] != [
        rng2.random() for _ in range(3)
    ]


def test_rng_for_step_thread_independence():
    clear_rng_cache()

    results = []
    lock = threading.Lock()

    def worker():
        rng = _rng_for_step(123, 5)
        seq = [rng.random() for _ in range(3)]
        with lock:
            results.append(seq)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 20
    assert all(seq == results[0] for seq in results)
