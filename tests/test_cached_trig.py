import gc
import math
from weakref import WeakKeyDictionary

from tnfr.helpers import get_cached_trig


class Obj:
    def __init__(self, th: float):
        self.theta = th


def test_get_cached_trig_uses_weakref_cache():
    cache = WeakKeyDictionary()
    o = Obj(math.pi / 2)
    cs = get_cached_trig(o, cache)
    assert cs == (math.cos(math.pi / 2), math.sin(math.pi / 2))
    assert len(cache) == 1
    del o
    gc.collect()
    assert len(cache) == 0

