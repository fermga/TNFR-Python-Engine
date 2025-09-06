from typing import Any
from tnfr.helpers import _stable_json


def test_stable_json_includes_module():
    Foo1 = type("Foo", (), {"__module__": "mod1", "__slots__": ()})
    Foo2 = type("Foo", (), {"__module__": "mod2", "__slots__": ()})
    assert _stable_json(Foo1()) != _stable_json(Foo2())


def test_stable_json_handles_cycles():
    obj: list[Any] = []
    obj.append(obj)
    assert _stable_json(obj) == ["<recursion>"]
