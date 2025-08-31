import pytest
from pathlib import Path
from tnfr.helpers import read_structured_file


def test_read_structured_file_corrupt_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        read_structured_file(path)
    assert str(path) in str(excinfo.value)


def test_read_structured_file_corrupt_yaml(tmp_path: Path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "bad.yaml"
    path.write_text("a: [1, 2", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        read_structured_file(path)
    assert str(path) in str(excinfo.value)
