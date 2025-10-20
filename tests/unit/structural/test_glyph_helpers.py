"""Unit tests for glyph count normalization and grouping helpers."""



from collections import Counter

from tnfr.utils import normalize_counter, mix_groups


def test_normalize_counter():
    counts = Counter({"A": 2, "B": 1})
    dist, total = normalize_counter(counts)
    assert total == 3
    assert dist == {"A": 2 / 3, "B": 1 / 3}

    empty_dist, empty_total = normalize_counter(Counter())
    assert empty_total == 0
    assert empty_dist == {}


def test_mix_groups():
    dist = {"A": 0.5, "B": 0.5}
    groups = {"ab": ("A", "B")}
    mixed = mix_groups(dist, groups)
    assert mixed["_ab"] == 1.0
    # se conserva la distribución original
    assert mixed["A"] == 0.5
