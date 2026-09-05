"""JSON options and data must survive optional serializer selection."""

import json

import pytest

from tnfr.utils import io


class OptionalEncoder:
    """Minimal stand-in for orjson's documented differing defaults."""

    OPT_SORT_KEYS = 1

    @staticmethod
    def dumps(obj, *, option=0, default=None):
        return json.dumps(
            obj, sort_keys=bool(option), default=default,
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("options", [
    {}, {"ensure_ascii": True}, {"ensure_ascii": False},
    {"indent": 2}, {"separators": (", ", ": ")}, {"sort_keys": True},
])
def test_requested_format_does_not_depend_on_optional_backend(monkeypatch, available, options):
    monkeypatch.setattr(io, "cached_import", lambda *args, **kwargs: OptionalEncoder if available else None)
    payload = {"z": "\u03c0", "a": [1, 2]}
    expected_options = {"separators": (",", ":"), **options}
    assert io.json_dumps(payload, **options) == json.dumps(payload, **expected_options)


@pytest.mark.parametrize("available", [False, True])
def test_custom_encoder_is_not_ignored(monkeypatch, available):
    monkeypatch.setattr(io, "cached_import", lambda *args, **kwargs: OptionalEncoder if available else None)

    class Encoder(json.JSONEncoder):
        def default(self, obj):
            return "encoded"

    assert io.json_dumps({"value": object()}, cls=Encoder) == '{"value":"encoded"}'


def test_allow_nan_false_is_not_ignored(monkeypatch):
    monkeypatch.setattr(io, "cached_import", lambda *args, **kwargs: OptionalEncoder)
    with pytest.raises(ValueError):
        io.json_dumps({"value": float("nan")}, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2**80, {1: "one"}])
def test_real_optional_backend_matches_standard_data_semantics(monkeypatch, value):
    optional = pytest.importorskip("orjson")
    monkeypatch.setattr(io, "cached_import", lambda *args, **kwargs: optional)
    assert io.json_dumps(value, ensure_ascii=False) == json.dumps(
        value, ensure_ascii=False, separators=(",", ":"),
    )


@pytest.mark.parametrize("to_bytes", [False, True])
def test_default_converter_receives_same_unsupported_type(monkeypatch, to_bytes):
    optional = pytest.importorskip("orjson")
    monkeypatch.setattr(io, "cached_import", lambda *args, **kwargs: optional)
    result = io.json_dumps({"x": {1, 2}}, default=sorted, to_bytes=to_bytes)
    assert result == (b'{"x":[1,2]}' if to_bytes else '{"x":[1,2]}')


@pytest.mark.parametrize("mode", ["a", "ab", "x", "r+"])
def test_atomic_writer_rejects_nonreplacement_modes_without_data_loss(tmp_path, mode):
    target = tmp_path / "existing.txt"
    target.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="replacement"):
        io.safe_write(target, lambda stream: stream.write(b"added" if "b" in mode else "added"), mode=mode)
    assert target.read_text(encoding="utf-8") == "existing"
    assert list(tmp_path.iterdir()) == [target]


def test_explicit_nonatomic_append_retains_file_contents(tmp_path):
    target = tmp_path / "existing.txt"
    target.write_text("existing", encoding="utf-8")
    io.safe_write(target, lambda stream: stream.write(" added"), mode="a", atomic=False)
    assert target.read_text(encoding="utf-8") == "existing added"
