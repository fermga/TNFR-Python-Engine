import warnings

import tnfr.json_utils as json_utils


class DummyOrjson:
    OPT_SORT_KEYS = 1

    @staticmethod
    def dumps(obj, option=0, default=None):
        return b"{}"


def _reset_json_utils(monkeypatch, module):
    monkeypatch.setattr(json_utils, "optional_import", lambda name: module)
    json_utils._load_orjson.cache_clear()
    monkeypatch.setattr(json_utils, "_orjson", None)
    monkeypatch.setattr(json_utils, "_ignored_param_warned", False)


def test_json_dumps_without_orjson(monkeypatch):
    _reset_json_utils(monkeypatch, None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = json_utils.json_dumps({"a": 1}, ensure_ascii=False)
    assert result == b'{"a":1}'
    assert w == []


def test_json_dumps_with_orjson_warns(monkeypatch):
    _reset_json_utils(monkeypatch, DummyOrjson())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        json_utils.json_dumps({"a": 1}, ensure_ascii=False)
    assert len(w) == 1
    assert "ignored" in str(w[0].message)
