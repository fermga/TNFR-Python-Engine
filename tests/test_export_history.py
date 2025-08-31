import networkx as nx

from tnfr.metrics import export_history


def test_export_history_creates_directory_csv(tmp_path):
    base = tmp_path / "non" / "existing" / "run"
    dir_path = base.parent
    assert not dir_path.exists()
    G = nx.Graph()
    export_history(G, str(base), fmt="csv")
    assert dir_path.exists()
    assert (dir_path / (base.name + "_glifogram.csv")).is_file()
    assert (dir_path / (base.name + "_sigma.csv")).is_file()


def test_export_history_creates_directory_json(tmp_path):
    base = tmp_path / "other" / "path" / "history"
    dir_path = base.parent
    assert not dir_path.exists()
    G = nx.Graph()
    export_history(G, str(base), fmt="json")
    assert dir_path.exists()
    assert (base.with_suffix(".json")).is_file()
