import tnfr.json_utils as json_utils
import warnings


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
    json_utils.clear_orjson_cache()

    assert calls["n"] == 0
    json_utils.json_dumps({})
    assert calls["n"] == 1
    json_utils.json_dumps({})
    assert calls["n"] == 1


def test_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(json_utils, "optional_import", lambda name: FakeOrjson())
    json_utils.clear_orjson_cache()

    with warnings.catch_warnings(record=True) as w:
        for _ in range(2):
            json_utils.json_dumps({}, ensure_ascii=False)

    assert len(w) == 1


def test_json_dumps_returns_str_by_default():
    data = {"a": 1, "b": [1, 2, 3]}
    result = json_utils.json_dumps(data)
    assert isinstance(result, str)
    assert result == json_utils.json_dumps(data, to_bytes=False)
