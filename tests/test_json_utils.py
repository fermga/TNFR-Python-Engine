import warnings

import tnfr.json_utils as json_utils


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
    monkeypatch.setattr(json_utils, "_orjson", None)
    monkeypatch.setattr(json_utils, "_ignored_param_warned", False)

    assert calls["n"] == 0
    json_utils.json_dumps({})
    assert calls["n"] == 1
    json_utils.json_dumps({})
    assert calls["n"] == 1


def test_warns_once(monkeypatch):
    monkeypatch.setattr(json_utils, "optional_import", lambda name: FakeOrjson())
    json_utils._load_orjson.cache_clear()
    monkeypatch.setattr(json_utils, "_orjson", None)
    monkeypatch.setattr(json_utils, "_ignored_param_warned", False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        json_utils.json_dumps({}, ensure_ascii=False)
        json_utils.json_dumps({}, ensure_ascii=False)
    assert len(w) == 1
