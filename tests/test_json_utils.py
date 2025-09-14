import logging

import tnfr.json_utils as json_utils


class FakeOrjson:
    OPT_SORT_KEYS = 1

    @staticmethod
    def dumps(obj, option=0, default=None):
        return b"{}"


def test_lazy_orjson_import(monkeypatch):
    calls = {"n": 0}

    def fake_cached_import(module, attr=None, **kwargs):
        calls["n"] += 1
        return FakeOrjson()

    monkeypatch.setattr(json_utils, "cached_import", fake_cached_import)
    json_utils._clear_orjson_cache()

    assert calls["n"] == 0
    json_utils.json_dumps({})
    assert calls["n"] == 1
    json_utils.json_dumps({})
    assert calls["n"] == 2


def test_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(
        json_utils, "cached_import", lambda *a, **k: FakeOrjson()
    )
    json_utils._clear_orjson_cache()

    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            json_utils.json_dumps({}, ensure_ascii=False)

    assert sum("ignored" in r.message for r in caplog.records) == 1


def test_warns_once_per_unique_combo(monkeypatch, caplog):
    monkeypatch.setattr(
        json_utils, "cached_import", lambda *a, **k: FakeOrjson()
    )
    json_utils._clear_orjson_cache()

    with caplog.at_level(logging.WARNING):
        json_utils.json_dumps({}, ensure_ascii=False)
        json_utils.json_dumps({}, ensure_ascii=False, separators=(";", ":"))
        json_utils.json_dumps({}, ensure_ascii=False, separators=(";", ":"))

    assert sum("ignored" in r.message for r in caplog.records) == 2


def test_json_dumps_returns_str_by_default():
    data = {"a": 1, "b": [1, 2, 3]}
    result = json_utils.json_dumps(data)
    assert isinstance(result, str)
    assert result == json_utils.json_dumps(data, to_bytes=False)
