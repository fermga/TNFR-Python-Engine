import logging

import tnfr.json_utils as json_utils
import tnfr.logging_utils as logging_utils


class FakeOrjson:
    OPT_SORT_KEYS = 1

    @staticmethod
    def dumps(obj, option=0, default=None):
        return b"{}"


def test_lazy_orjson_import(monkeypatch):
    calls = {"n": 0}

    def fake_optional_import(name):
        calls["n"] += 1
        return FakeOrjson()

    monkeypatch.setattr(json_utils, "optional_import", fake_optional_import)
    json_utils._load_orjson.cache_clear()
    logging_utils._WARNED_KEYS.clear()

    assert calls["n"] == 0
    json_utils.json_dumps({})
    assert calls["n"] == 1
    json_utils.json_dumps({})
    assert calls["n"] == 1


def test_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(json_utils, "optional_import", lambda name: FakeOrjson())
    json_utils._load_orjson.cache_clear()
    logging_utils._WARNED_KEYS.clear()

    with caplog.at_level(logging.WARNING):
        json_utils.json_dumps({}, ensure_ascii=False)
        json_utils.json_dumps({}, ensure_ascii=False)
    assert sum("ignored" in r.message for r in caplog.records) == 1
