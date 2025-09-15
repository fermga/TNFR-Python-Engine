import logging
from typing import Mapping

import tnfr.import_utils as import_utils
import tnfr.json_utils as json_utils
from .utils import clear_orjson_cache


class DummyOrjson:
    OPT_SORT_KEYS = 1

    @staticmethod
    def dumps(obj, option=0, default=None):
        return b"{}"


def _reset_json_utils(monkeypatch, module):
    monkeypatch.setattr(
        json_utils, "cached_import", lambda name, attr=None, **kwargs: module
    )
    clear_orjson_cache()
    import_utils.prune_failed_imports()


def test_lazy_orjson_import(monkeypatch):
    calls = {"n": 0}

    def fake_cached_import(module, attr=None, **kwargs):
        calls["n"] += 1
        return DummyOrjson()

    monkeypatch.setattr(json_utils, "cached_import", fake_cached_import)
    clear_orjson_cache()

    assert calls["n"] == 0
    json_utils.json_dumps({})
    assert calls["n"] == 1
    json_utils.json_dumps({})
    assert calls["n"] == 2


def test_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(
        json_utils, "cached_import", lambda *a, **k: DummyOrjson()
    )
    clear_orjson_cache()

    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            json_utils.json_dumps({}, ensure_ascii=False)

    assert sum("ignored" in r.message for r in caplog.records) == 1


def test_warns_once_per_unique_combo(monkeypatch, caplog):
    monkeypatch.setattr(
        json_utils, "cached_import", lambda *a, **k: DummyOrjson()
    )
    clear_orjson_cache()

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


def test_json_dumps_without_orjson(monkeypatch, caplog):
    clear_orjson_cache()
    import_utils.prune_failed_imports()

    original = import_utils.importlib.import_module

    def fake_import(name, package=None):  # pragma: no cover - monkeypatch helper
        if name == "orjson":
            raise ImportError("missing")
        return original(name, package)

    monkeypatch.setattr(import_utils.importlib, "import_module", fake_import)

    with caplog.at_level(logging.WARNING, logger="tnfr.import_utils"):
        result = json_utils.json_dumps({"a": 1}, ensure_ascii=False, to_bytes=True)

    assert result == b'{"a":1}'
    assert any(
        "Failed to import module 'orjson'" in r.message for r in caplog.records
    )


def test_json_dumps_with_orjson_warns(monkeypatch, caplog):
    _reset_json_utils(monkeypatch, DummyOrjson())

    with caplog.at_level(logging.WARNING):
        json_utils.json_dumps({"a": 1}, ensure_ascii=False)
        json_utils.json_dumps({"a": 1}, ensure_ascii=False)
    assert sum("ignored" in r.message for r in caplog.records) == 1


def test_params_passed_to_std(monkeypatch):
    _reset_json_utils(monkeypatch, None)

    captured = {}

    def fake_std(obj, params, **kwargs):
        captured["params"] = params
        return b"{}"

    monkeypatch.setattr(json_utils, "_json_dumps_std", fake_std)
    json_utils.json_dumps({"a": 1})
    assert isinstance(captured["params"], Mapping)
    assert captured["params"]["sort_keys"] is False
    assert captured["params"]["ensure_ascii"] is True


def test_params_passed_to_orjson(monkeypatch):
    _reset_json_utils(monkeypatch, DummyOrjson())

    captured = {}

    def fake_orjson(_orjson_mod, obj, params, **kwargs):
        captured["params"] = params
        return b"{}"

    monkeypatch.setattr(json_utils, "_json_dumps_orjson", fake_orjson)
    json_utils.json_dumps({"a": 1})
    assert isinstance(captured["params"], Mapping)
    assert captured["params"]["sort_keys"] is False
    assert captured["params"]["ensure_ascii"] is True


def test_default_params_reused(monkeypatch):
    _reset_json_utils(monkeypatch, None)

    calls: list[Mapping[str, object]] = []

    def fake_std(obj, params, **kwargs):
        calls.append(params)
        return b"{}"

    monkeypatch.setattr(json_utils, "_json_dumps_std", fake_std)
    json_utils.json_dumps({"a": 1})
    json_utils.json_dumps(
        {"a": 1},
        sort_keys=False,
        default=None,
        ensure_ascii=True,
        separators=(",", ":"),
        cls=None,
        to_bytes=False,
    )
    json_utils.json_dumps({"a": 1}, sort_keys=True)
    assert calls[0] is json_utils.DEFAULT_PARAMS
    assert calls[1] is json_utils.DEFAULT_PARAMS
    assert calls[2] is not json_utils.DEFAULT_PARAMS
