import math
import pytest

from tnfr.constants import get_aliases
from tnfr.metrics_utils import get_trig_cache, compute_Si
from tnfr.helpers.numeric import neighbor_phase_mean
from tnfr.alias import set_attr

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def test_trig_cache_reuse_between_modules(monkeypatch, graph_canon):
    cos_calls = 0
    sin_calls = 0
    orig_cos = math.cos
    orig_sin = math.sin

    def cos_wrapper(x):
        nonlocal cos_calls
        cos_calls += 1
        return orig_cos(x)

    def sin_wrapper(x):
        nonlocal sin_calls
        sin_calls += 1
        return orig_sin(x)

    monkeypatch.setattr(math, "cos", cos_wrapper)
    monkeypatch.setattr(math, "sin", sin_wrapper)

    G = graph_canon()
    G.add_edge(1, 2)
    set_attr(G.nodes[1], ALIAS_THETA, 0.0)
    set_attr(G.nodes[2], ALIAS_THETA, math.pi / 2)
    set_attr(G.nodes[1], ALIAS_VF, 0.0)
    set_attr(G.nodes[2], ALIAS_VF, 0.0)
    set_attr(G.nodes[1], ALIAS_DNFR, 0.0)
    set_attr(G.nodes[2], ALIAS_DNFR, 0.0)

    trig1 = get_trig_cache(G)
    assert cos_calls == 2
    assert sin_calls == 2

    assert neighbor_phase_mean(G, 1) == pytest.approx(math.pi / 2)
    assert cos_calls == 2
    assert sin_calls == 2

    compute_Si(G, inplace=False)
    assert cos_calls == 2
    assert sin_calls == 2

    trig2 = get_trig_cache(G)
    assert trig1 is trig2
