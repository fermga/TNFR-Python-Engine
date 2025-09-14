import json

import pytest

from tnfr.helpers.node_cache import stable_json
from tnfr import json_utils


def test_stable_json_dict_order_deterministic():
    json_utils._clear_orjson_cache()
    json_utils._orjson = None
    obj = {"b": 1, "a": 2}
    res1 = stable_json(obj)
    res2 = stable_json(obj)
    assert res1 == res2
    assert json.loads(res1) == {"a": 2, "b": 1}


def test_stable_json_warns_with_orjson():
    if json_utils._orjson is None:
        pytest.skip("orjson not installed")
    with pytest.warns(UserWarning, match="ignored when using orjson"):
        stable_json({"a": 1})
