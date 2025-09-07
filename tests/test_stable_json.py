from tnfr.helpers import _stable_json


def test_stable_json_set_order_deterministic():
    class Obj:
        def __init__(self, v):
            self.v = v
    s = {Obj(1), Obj(2), 3, "a"}
    res1 = _stable_json(s)
    res2 = _stable_json(s)
    assert res1 == res2
    assert res1 == sorted(res1, key=str)


def test_stable_json_respects_max_depth_dict():
    obj = {"a": {"b": {"c": 1}}}
    assert _stable_json(obj, max_depth=1) == {"a": "<max-depth>"}


def test_stable_json_respects_max_depth_list():
    obj = [1, [2, [3]]]
    assert _stable_json(obj, max_depth=1) == [1, "<max-depth>"]
