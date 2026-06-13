"""Tests for the structural-interface benchmark runner (Milestones 3 and 4).

WDBC is bundled with scikit-learn and runs offline.  The online Wine Quality
dataset is exercised only through the graceful-skip path so the suite does not
require network access.  Milestone 4 adds the non-circular held-out
model-error target, validated here against the circular localization sanity
check.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn")


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / "structural_interface_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "structural_interface_benchmark", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_benchmark_module()

# The full classical baseline suite that every benchmark must report.
EXPECTED_BASELINES = {
    "local_disagreement",
    "graph_total_variation",
    "local_class_entropy",
    "label_propagation_residual",
    "graph_cut_contribution",
    "mean_neighbour_distance",
    "degree",
    "constant",
    "random",
    "feature_deviation",
}


# ---------------------------------------------------------------------------
# WDBC offline benchmark
# ---------------------------------------------------------------------------


def test_wdbc_dataset_bundle_shape() -> None:
    bundle = MODULE.load_wdbc_dataset()
    assert bundle.sector == "biomedicine"
    assert len(bundle.records) == 569
    assert len(bundle.feature_keys) == 30
    assert bundle.state_key == "diagnosis"
    assert bundle.positive_value == "benign"


def test_prepare_interface_graph_preserves_sample_count() -> None:
    bundle = MODULE.load_wdbc_dataset()
    G = MODULE.prepare_interface_graph(bundle, k=8)
    assert G.number_of_nodes() == 569
    assert G.number_of_edges() > G.number_of_nodes()
    # Phase encoding + conflict pressure are present on every node.
    for node in G.nodes():
        assert "phase" in G.nodes[node]
        assert "delta_nfr" in G.nodes[node]
        assert "incident_state_conflicts" in G.nodes[node]


def test_run_dataset_benchmark_wdbc_structure(tmp_path: Path) -> None:
    bundle = MODULE.load_wdbc_dataset()
    result = MODULE.run_dataset_benchmark(
        bundle, k=8, seed=0, top_n=10, output_dir=tmp_path
    )

    # Dataset / graph blocks.
    assert result["dataset"]["name"] == "WDBC breast cancer"
    assert result["dataset"]["samples"] == 569
    assert result["graph"]["nodes"] == 569
    assert result["graph"]["k"] == 8

    # Circular target is explicitly declared.
    assert result["task"]["is_circular_target"] is True
    assert result["task"]["target_kind"] == "circular_local_disagreement"

    # Evaluation contains TNFR + every classical baseline.
    rows = {row["score"]: row for row in result["evaluation"]["score_comparison"]}
    assert MODULE.TNFR_SCORE_LABEL in rows
    assert EXPECTED_BASELINES.issubset(set(rows))
    for row in rows.values():
        assert math.isfinite(row["auc"])
        assert math.isfinite(row["precision_at_review_count"])

    # Closest classical baseline must be present for fair comparison.
    assert "local_disagreement" in rows

    # Hotspots and report artifacts.
    assert 0 < len(result["hotspots"]) <= 10
    for hotspot in result["hotspots"]:
        assert hotspot["prescription"]  # non-empty grammar-valid prescription
    for kind in ("json", "markdown", "html"):
        assert Path(result["report_paths"][kind]).exists()


def test_wdbc_circular_target_is_localization_sanity_check(tmp_path: Path) -> None:
    bundle = MODULE.load_wdbc_dataset()
    result = MODULE.run_dataset_benchmark(bundle, k=8, output_dir=None)
    rows = {row["score"]: row for row in result["evaluation"]["score_comparison"]}
    # On a circular target the TNFR score does NOT beat the closest classical
    # baseline; it ties it (both consume the same conflicts).  Documenting this
    # is the whole point of the fair-comparison design.
    assert rows[MODULE.TNFR_SCORE_LABEL]["auc"] == pytest.approx(
        rows["local_disagreement"]["auc"], abs=1e-9
    )
    # Topology-only / constant baselines carry no localization signal.
    assert rows["constant"]["auc"] == pytest.approx(0.5, abs=1e-9)


def test_baseline_formulas_documented_in_result() -> None:
    bundle = MODULE.load_wdbc_dataset()
    result = MODULE.run_dataset_benchmark(bundle, k=8, output_dir=None)
    names = set(result["baselines"]["names"])
    assert EXPECTED_BASELINES.issubset(names)
    for name in result["baselines"]["formulas"]:
        assert result["baselines"]["formulas"][name]


# ---------------------------------------------------------------------------
# Suite runner + graceful skipping
# ---------------------------------------------------------------------------


def test_run_benchmark_suite_wdbc(tmp_path: Path) -> None:
    summary = MODULE.run_benchmark_suite(
        ["wdbc"], k=8, seed=0, output_dir=tmp_path
    )
    assert summary["suite"] == "structural_interface_benchmark"
    assert len(summary["proof_of_concept_targets"]) == 1
    entry = summary["proof_of_concept_targets"][0]
    assert entry["dataset"] == "WDBC breast cancer"
    assert entry["review_node_count"] > 0
    assert entry["tnfr_auc"] is not None
    assert entry["local_disagreement_auc"] is not None
    # Independent targets are reserved for Milestone 4.
    assert summary["independent_targets"] == []
    assert summary["independent_targets_note"]
    assert Path(summary["summary_path"]).exists()


def test_run_benchmark_suite_skips_unknown_dataset(tmp_path: Path) -> None:
    summary = MODULE.run_benchmark_suite(
        ["does_not_exist"], output_dir=tmp_path
    )
    assert summary["proof_of_concept_targets"] == []
    assert summary["skipped"] == [
        {"dataset": "does_not_exist", "reason": "unknown dataset"}
    ]


def test_wine_dataset_skips_gracefully_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args, **kwargs):
        raise OSError("network disabled in test")

    monkeypatch.setattr(MODULE, "download_wine_quality_csv", _boom)
    summary = MODULE.run_benchmark_suite(["wine"], output_dir=tmp_path)
    assert summary["proof_of_concept_targets"] == []
    assert len(summary["skipped"]) == 1
    assert summary["skipped"][0]["dataset"] == "wine"


# ---------------------------------------------------------------------------
# Milestone 4: non-circular held-out model-error target
# ---------------------------------------------------------------------------


def test_held_out_model_error_labels_are_out_of_fold() -> None:
    bundle = MODULE.load_wdbc_dataset()
    target = MODULE.held_out_model_error_labels(bundle, seed=0, n_splits=5)
    # One label per record; aligned with build_knn_graph node ids 0..n-1.
    assert len(target.error_labels) == len(bundle.records)
    assert set(target.error_labels) == set(range(len(bundle.records)))
    assert all(isinstance(v, bool) for v in target.error_labels.values())
    # A real classifier has some errors but is far better than chance.
    assert 0 < target.error_count < len(bundle.records)
    assert 0.5 < target.cv_accuracy <= 1.0
    assert target.cv_folds == 5
    assert "LogisticRegression" in target.classifier_name


def test_held_out_model_error_labels_are_deterministic() -> None:
    bundle = MODULE.load_wdbc_dataset()
    a = MODULE.held_out_model_error_labels(bundle, seed=0)
    b = MODULE.held_out_model_error_labels(bundle, seed=0)
    assert a.error_labels == b.error_labels
    assert a.error_count == b.error_count


def test_model_error_target_differs_from_circular_target() -> None:
    bundle = MODULE.load_wdbc_dataset()
    G = MODULE.prepare_interface_graph(bundle, k=8)
    incident = {
        node: int(G.nodes[node].get("incident_state_conflicts", 0))
        for node in G.nodes()
    }
    threshold = MODULE._auto_conflict_threshold(incident)
    circular = MODULE.circular_review_labels(G, conflict_threshold=threshold)
    model_error = MODULE.held_out_model_error_labels(bundle, seed=0).error_labels
    # Non-circularity: the two targets must not be the same set of review nodes.
    assert circular != model_error


def test_run_dataset_model_error_benchmark_structure(tmp_path: Path) -> None:
    bundle = MODULE.load_wdbc_dataset()
    result = MODULE.run_dataset_model_error_benchmark(
        bundle, k=8, seed=0, top_n=10, output_dir=tmp_path
    )

    # Non-circular target is explicitly declared.
    assert result["task"]["target_kind"] == "held_out_model_error"
    assert result["task"]["is_circular_target"] is False

    # Model metadata is reported.
    model = result["model"]
    assert "LogisticRegression" in model["classifier"]
    assert model["cv_folds"] == 5
    assert 0.5 < model["cv_accuracy"] <= 1.0
    assert model["error_count"] > 0

    rows = {row["score"]: row for row in result["evaluation"]["score_comparison"]}
    assert MODULE.TNFR_SCORE_LABEL in rows
    # The reference baselines required by Milestone 4 acceptance are present.
    for name in MODULE.REFERENCE_BASELINES:
        assert name in rows
        assert math.isfinite(rows[name]["auc"])

    # The model-error report is written under a distinct stem.
    for kind in ("json", "markdown", "html"):
        path = Path(result["report_paths"][kind])
        assert path.exists()
        assert "_model_error" in path.name


def test_model_error_not_reported_as_circular_superiority(tmp_path: Path) -> None:
    bundle = MODULE.load_wdbc_dataset()
    result = MODULE.run_dataset_model_error_benchmark(bundle, k=8, output_dir=None)
    rows = {row["score"]: row for row in result["evaluation"]["score_comparison"]}
    # On the independent target, TNFR and local disagreement are NOT forced to
    # tie (the circular coupling is gone).  We only assert TNFR is a competitive
    # localizer, never that it trivially equals the closest baseline.
    tnfr_auc = rows[MODULE.TNFR_SCORE_LABEL]["auc"]
    local_auc = rows["local_disagreement"]["auc"]
    assert math.isfinite(tnfr_auc)
    assert tnfr_auc > 0.5  # better than chance on a real held-out target
    # Control baselines remain uninformative on the independent target too.
    assert rows["constant"]["auc"] == pytest.approx(0.5, abs=1e-9)
    # Honest interpretation must flag the non-circular framing.
    assert "non-circular" in result["honest_interpretation"].lower()
    assert local_auc > 0.5


def test_run_benchmark_suite_model_error_target(tmp_path: Path) -> None:
    summary = MODULE.run_benchmark_suite(
        ["wdbc"], targets=("model_error",), output_dir=tmp_path
    )
    # No circular proof-of-concept entry; one independent (non-circular) entry.
    assert summary["proof_of_concept_targets"] == []
    assert len(summary["independent_targets"]) == 1
    entry = summary["independent_targets"][0]
    assert entry["dataset"] == "WDBC breast cancer"
    assert entry["target_kind"] == "held_out_model_error"
    assert entry["review_node_count"] > 0
    assert entry["tnfr_auc"] is not None
    # Each reference baseline AUC is surfaced in the summary entry.
    for name in MODULE.REFERENCE_BASELINES:
        assert entry[f"{name}_auc"] is not None
    assert summary["config"]["targets"] == ["model_error"]
    assert Path(summary["summary_path"]).exists()


def test_run_benchmark_suite_all_targets(tmp_path: Path) -> None:
    summary = MODULE.run_benchmark_suite(
        ["wdbc"], targets=("circular", "model_error"), output_dir=tmp_path
    )
    assert len(summary["proof_of_concept_targets"]) == 1
    assert len(summary["independent_targets"]) == 1
    # Circular and non-circular reports coexist under distinct stems.
    files = {p.name for p in tmp_path.glob("structural_interface_wdbc*")}
    assert "structural_interface_wdbc_breast_cancer.json" in files
    assert "structural_interface_wdbc_breast_cancer_model_error.json" in files


# ---------------------------------------------------------------------------
# Additional offline datasets (cross-domain validation)
# ---------------------------------------------------------------------------


def test_dataset_loaders_registry() -> None:
    assert set(MODULE.DATASET_LOADERS) == {
        "wdbc",
        "wine",
        "wine_white",
        "iris",
        "digits",
    }


def test_iris_dataset_bundle_shape() -> None:
    bundle = MODULE.load_iris_dataset()
    assert bundle.sector == "botany"
    assert len(bundle.records) == 150
    assert len(bundle.feature_keys) == 4
    # Binary band derived from the ternary species label, mirroring the
    # quality_band / diagnosis pattern.
    assert bundle.state_key == "species_band"
    assert bundle.positive_value == "virginica"
    bands = {rec["species_band"] for rec in bundle.records}
    assert bands == {"virginica", "other"}
    assert sum(rec["species_band"] == "virginica" for rec in bundle.records) == 50


def test_digits_dataset_bundle_shape() -> None:
    bundle = MODULE.load_digits_dataset()
    assert bundle.sector == "computer vision"
    assert len(bundle.records) == 1797
    assert len(bundle.feature_keys) == 64
    assert bundle.state_key == "parity"
    assert bundle.positive_value == "even"
    bands = {rec["parity"] for rec in bundle.records}
    assert bands == {"even", "odd"}
    # Representative scalar feature for the deviation control baseline.
    assert bundle.feature_label == "mean pixel intensity"
    assert all("feature_value" in rec for rec in bundle.records)


def test_iris_model_error_benchmark_is_non_circular(tmp_path: Path) -> None:
    bundle = MODULE.load_iris_dataset()
    result = MODULE.run_dataset_model_error_benchmark(
        bundle, k=8, seed=0, output_dir=tmp_path
    )
    assert result["task"]["is_circular_target"] is False
    assert result["task"]["target_kind"] == "held_out_model_error"
    assert result["model"]["error_count"] > 0
    rows = {row["score"]: row for row in result["evaluation"]["score_comparison"]}
    assert MODULE.TNFR_SCORE_LABEL in rows
    assert rows[MODULE.TNFR_SCORE_LABEL]["auc"] > 0.5
    for name in MODULE.REFERENCE_BASELINES:
        assert name in rows


def test_run_benchmark_suite_offline_datasets(tmp_path: Path) -> None:
    # The offline scikit-learn datasets must run end to end without network.
    summary = MODULE.run_benchmark_suite(
        ["iris", "digits"], targets=("model_error",), output_dir=tmp_path
    )
    assert summary["skipped"] == []
    assert len(summary["independent_targets"]) == 2
    sectors = {entry["sector"] for entry in summary["independent_targets"]}
    assert sectors == {"botany", "computer vision"}
    for entry in summary["independent_targets"]:
        assert entry["tnfr_auc"] is not None
        for name in MODULE.REFERENCE_BASELINES:
            assert entry[f"{name}_auc"] is not None


def test_wine_white_skips_gracefully_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args, **kwargs):
        raise OSError("network disabled in test")

    monkeypatch.setattr(MODULE, "download_wine_quality_csv", _boom)
    summary = MODULE.run_benchmark_suite(["wine_white"], output_dir=tmp_path)
    assert summary["proof_of_concept_targets"] == []
    assert len(summary["skipped"]) == 1
    assert summary["skipped"][0]["dataset"] == "wine_white"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def test_parse_datasets_all_and_csv() -> None:
    assert MODULE._parse_datasets("all") == [
        "wdbc",
        "wine",
        "wine_white",
        "iris",
        "digits",
    ]
    assert MODULE._parse_datasets("offline") == ["wdbc", "iris", "digits"]
    assert MODULE._parse_datasets("wdbc,wine") == ["wdbc", "wine"]
    assert MODULE._parse_datasets("wdbc") == ["wdbc"]


def test_default_stem_slug() -> None:
    assert MODULE._default_stem("WDBC breast cancer") == (
        "structural_interface_wdbc_breast_cancer"
    )
    assert MODULE._default_stem("UCI Wine Quality (red)") == (
        "structural_interface_uci_wine_quality_red"
    )


def test_arg_parser_defaults() -> None:
    parser = MODULE.build_arg_parser()
    args = parser.parse_args([])
    assert args.dataset == "all"
    assert args.k == 8
    assert args.seed == 0
    assert args.target == "circular"
    assert args.cv_folds == 5
    assert args.output == "results/reports"


def test_arg_parser_target_choices() -> None:
    parser = MODULE.build_arg_parser()
    for choice in ("circular", "model_error", "all"):
        args = parser.parse_args(["--target", choice])
        assert args.target == choice
    with pytest.raises(SystemExit):
        parser.parse_args(["--target", "nonsense"])


def test_target_choices_map() -> None:
    assert MODULE._TARGET_CHOICES["circular"] == ("circular",)
    assert MODULE._TARGET_CHOICES["model_error"] == ("model_error",)
    assert MODULE._TARGET_CHOICES["all"] == ("circular", "model_error")
