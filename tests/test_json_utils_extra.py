import warnings
from dataclasses import is_dataclass

import tnfr.json_utils as json_utils


class DummyOrjson:
    OPT_SORT_KEYS = 1

    @staticmethod
    def dumps(obj, option=0, default=None):
        return b"{}"


def _reset_json_utils(monkeypatch, module):
    monkeypatch.setattr(json_utils, "optional_import", lambda name: module)
    json_utils.clear_orjson_cache()


def test_json_dumps_without_orjson(monkeypatch, caplog):
    _reset_json_utils(monkeypatch, None)

    with warnings.catch_warnings(record=True):

        result = json_utils.json_dumps({"a": 1}, ensure_ascii=False, to_bytes=True)
    assert result == b'{"a":1}'
    assert caplog.records == []


def test_json_dumps_with_orjson_warns(monkeypatch):
    _reset_json_utils(monkeypatch, DummyOrjson())

    with warnings.catch_warnings(record=True) as w:

        json_utils.json_dumps({"a": 1}, ensure_ascii=False)
        json_utils.json_dumps({"a": 1}, ensure_ascii=False)
    assert sum("ignored" in str(r.message) for r in w) == 1


def test_params_passed_to_std(monkeypatch):
    _reset_json_utils(monkeypatch, None)

    captured = {}

    def fake_std(obj, params, **kwargs):
        captured["params"] = params
        return b"{}"

    monkeypatch.setattr(json_utils, "_json_dumps_std", fake_std)
    json_utils.json_dumps({"a": 1})
    assert is_dataclass(captured["params"])


def test_params_passed_to_orjson(monkeypatch):
    _reset_json_utils(monkeypatch, DummyOrjson())

    captured = {}

    def fake_orjson(orjson_mod, obj, params, **kwargs):
        captured["params"] = params
        return b"{}"

    monkeypatch.setattr(json_utils, "_json_dumps_orjson", fake_orjson)
    json_utils.json_dumps({"a": 1})
    assert is_dataclass(captured["params"])
