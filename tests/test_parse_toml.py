"""Tests for TOML parser integration with ``read_structured_file``."""

from pathlib import Path

import pytest
import tnfr.io as io_mod
from tnfr.io import read_structured_file


def test_read_structured_file_invokes_toml_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "data.toml"
    content = "a = 1"
    path.write_text(content, encoding="utf-8")

    called = {}

    def fake_parse(text: str) -> dict[str, int]:
        called["text"] = text
        return {"a": 1}

    monkeypatch.setattr(io_mod, "_parse_toml", fake_parse)
    monkeypatch.setitem(io_mod.PARSERS, ".toml", fake_parse)

    assert read_structured_file(path) == {"a": 1}
    assert called["text"] == content
