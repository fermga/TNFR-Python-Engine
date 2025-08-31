import json

from tnfr.metrics import export_history


def test_export_history_creates_directory_csv(tmp_path, graph_canon):
    base = tmp_path / "non" / "existing" / "run"
    dir_path = base.parent
    assert not dir_path.exists()
    G = graph_canon()
    export_history(G, str(base), fmt="csv")
    assert dir_path.exists()
    assert (dir_path / (base.name + "_glifogram.csv")).is_file()
    assert (dir_path / (base.name + "_sigma.csv")).is_file()


def test_export_history_creates_directory_json(tmp_path, graph_canon):
    base = tmp_path / "other" / "path" / "history"
    dir_path = base.parent
    assert not dir_path.exists()
    G = graph_canon()
    export_history(G, str(base), fmt="json")
    assert dir_path.exists()
    assert (base.with_suffix(".json")).is_file()


def test_export_history_writes_optional_files(tmp_path, graph_canon):
    base = tmp_path / "extras" / "run"
    G = graph_canon()
    hist = G.graph.setdefault("history", {})
    hist["morph"] = [{"t": 0, "ID": 1, "CM": 2, "NE": 3, "PP": 4}]
    hist["EPI_support"] = [{"t": 0, "size": 1, "epi_norm": 0.5}]
    export_history(G, str(base), fmt="csv")
    dir_path = base.parent
    assert (dir_path / (base.name + "_morph.csv")).is_file()
    assert (dir_path / (base.name + "_epi_support.csv")).is_file()


def test_export_history_json_contains_optional(tmp_path, graph_canon):
    base = tmp_path / "extras" / "jsonrun"
    G = graph_canon()
    hist = G.graph.setdefault("history", {})
    hist["morph"] = [{"t": 0, "ID": 1, "CM": 2, "NE": 3, "PP": 4}]
    hist["EPI_support"] = [{"t": 0, "size": 1, "epi_norm": 0.5}]
    export_history(G, str(base), fmt="json")
    data = json.loads((base.with_suffix(".json")).read_text())
    assert data["morph"]
    assert data["epi_support"]
