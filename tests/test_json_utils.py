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
    json_utils._load_orjson.cache_clear()

    assert calls["n"] == 0
    json_utils.json_dumps({})
    assert calls["n"] == 1
    json_utils.json_dumps({})
    assert calls["n"] == 1


def test_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(json_utils, "optional_import", lambda name: FakeOrjson())
    json_utils._load_orjson.cache_clear()

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings("once", message=".*ignored when using orjson")

        json_utils.json_dumps({}, ensure_ascii=False)
        json_utils.json_dumps({}, ensure_ascii=False)

    assert len(w) == 1


def test_json_dumps_str_matches_json_dumps():
    data = {"a": 1, "b": [1, 2, 3]}
    assert json_utils.json_dumps_str(data) == json_utils.json_dumps(data, to_bytes=False)

