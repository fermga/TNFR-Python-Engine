import importlib
import json

import pytest

from tnfr.utils.cache import stable_json
from .utils import clear_orjson_cache


def test_stable_json_dict_order_deterministic():
    clear_orjson_cache()
    obj = {"b": 1, "a": 2}
    res1 = stable_json(obj)
    res2 = stable_json(obj)
    assert isinstance(res1, str)
    assert isinstance(res2, str)
    assert res1 == res2
    assert json.loads(res1) == {"a": 2, "b": 1}


def test_stable_json_warns_with_orjson():
    if importlib.util.find_spec("orjson") is None:
        pytest.skip("orjson not installed")
    clear_orjson_cache()
    with pytest.warns(UserWarning, match="ignored when using orjson"):
        stable_json({"a": 1})
