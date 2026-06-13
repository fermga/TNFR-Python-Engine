#!/usr/bin/env python3
"""TNFR Structural Interface Benchmark (Milestone 3).

This benchmark turns the structural-interface validation layer into a
reproducible, cross-domain runner.  It builds a k-nearest-neighbour graph from a
real dataset, encodes a binary state band as a TNFR phase, scores graph-local
structural interfaces with the phase-gate tetrad, and compares the TNFR score
against the full classical baseline suite from
``tnfr.validation.interface_baselines``.

Honest scope
------------
The review target used here is the *circular* local-disagreement target: a node
is a review case when enough of its neighbours carry the opposite state band.
TNFR phase stress also uses those conflicts, so a high AUC against this target is
a localization sanity check, **not** a fair claim of superiority over classical
graph metrics.  The benchmark summary separates these proof-of-concept circular
targets from independent targets (the latter are added in Milestone 4).

Datasets
--------
- ``wdbc``: Wisconsin Diagnostic Breast Cancer (bundled with scikit-learn,
  offline).
- ``iris``: Iris, binary virginica band (bundled with scikit-learn, offline).
- ``digits``: handwritten digits, even/odd parity (bundled, offline,
  high-dimensional hard target).
- ``wine``: UCI Red Wine Quality (downloaded on first run, cached under
  ``results/data``).
- ``wine_white``: UCI White Wine Quality (downloaded on first run, cached).

The ``offline`` dataset alias selects only the bundled scikit-learn datasets
(``wdbc``, ``iris``, ``digits``) so the suite runs without network access.

Usage (PowerShell)::

    $env:PYTHONPATH=(Resolve-Path -Path ./src).Path
    python benchmarks/structural_interface_benchmark.py --dataset all \
        --k 8 --seed 0 --output results/reports

Run a single dataset::

    python benchmarks/structural_interface_benchmark.py --dataset wdbc
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.request import urlopen

# Ensure local src is importable ------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.validation.interface_baselines import (  # noqa: E402
    BASELINE_FORMULAS,
)
from tnfr.validation.structural_interface import (  # noqa: E402
    StructuralInterfaceProblem,
    build_knn_graph,
    encode_phase_from_binary_state,
    evaluate_interface_scores,
    export_structural_interface_report,
    full_baseline_score_maps,
    interface_score_maps,
    score_structural_interfaces,
)

WINE_QUALITY_RED_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

WINE_QUALITY_WHITE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-white.csv"
)

#: TNFR score label used in the comparison table.
TNFR_SCORE_LABEL = "tnfr_phase_stress"


# ---------------------------------------------------------------------------
# Dataset bundles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetBundle:
    """A prepared dataset ready for structural-interface graph construction."""

    name: str
    sector: str
    records: list[dict[str, Any]]
    feature_keys: list[str]
    state_key: str
    positive_value: Any
    feature_label: str
    node_attributes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_wdbc_dataset() -> DatasetBundle:
    """Load the bundled Wisconsin Diagnostic Breast Cancer dataset (offline)."""
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    feature_names = [str(name) for name in data.feature_names]
    # Representative morphology feature surfaced as a clean baseline attribute.
    feature_label = "mean concavity"
    feature_index = feature_names.index(feature_label)

    records: list[dict[str, Any]] = []
    for row, target in zip(data.data, data.target):
        diagnosis = str(data.target_names[int(target)])
        record: dict[str, Any] = {
            name: float(value) for name, value in zip(feature_names, row)
        }
        record["diagnosis"] = diagnosis
        record["feature_value"] = float(row[feature_index])
        records.append(record)

    return DatasetBundle(
        name="WDBC breast cancer",
        sector="biomedicine",
        records=records,
        feature_keys=feature_names,
        state_key="diagnosis",
        positive_value="benign",
        feature_label=feature_label,
        node_attributes=["diagnosis", "feature_value"],
        metadata={"source": "scikit-learn bundled", "online": False},
    )


def download_wine_quality_csv(
    *,
    cache_path: Path | None = None,
    url: str = WINE_QUALITY_RED_URL,
    timeout: float = 30.0,
) -> Path:
    """Download the UCI red wine quality CSV, using a local cache if present."""
    path = cache_path or _ROOT / "results" / "data" / "winequality-red.csv"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = urlopen(url, timeout=timeout).read()  # noqa: S310 - fixed UCI URL
    path.write_bytes(payload)
    return path


def _build_wine_bundle(
    csv_path: Path,
    *,
    name: str,
    quality_threshold: int,
) -> DatasetBundle:
    """Build a wine-quality bundle from a semicolon-delimited UCI CSV."""
    rows = list(
        csv.DictReader(
            csv_path.read_text(encoding="utf-8").splitlines(), delimiter=";"
        )
    )
    feature_keys = [key for key in rows[0] if key != "quality"]
    feature_label = "alcohol"

    records: list[dict[str, Any]] = []
    for row in rows:
        quality = int(row["quality"])
        band = "high" if quality >= int(quality_threshold) else "low"
        record: dict[str, Any] = {key: float(row[key]) for key in feature_keys}
        record["quality"] = quality
        record["quality_band"] = band
        record["feature_value"] = float(row[feature_label])
        records.append(record)

    return DatasetBundle(
        name=name,
        sector="food chemistry",
        records=records,
        feature_keys=feature_keys,
        state_key="quality_band",
        positive_value="high",
        feature_label=feature_label,
        node_attributes=["quality", "quality_band", "feature_value"],
        metadata={
            "source": "UCI online",
            "online": True,
            "quality_threshold": int(quality_threshold),
        },
    )


def load_wine_dataset(
    *,
    cache_path: Path | None = None,
    quality_threshold: int = 6,
    download_timeout: float = 30.0,
) -> DatasetBundle:
    """Load the UCI Red Wine Quality dataset (downloads + caches on first run)."""
    csv_path = download_wine_quality_csv(
        cache_path=cache_path, timeout=download_timeout
    )
    return _build_wine_bundle(
        csv_path,
        name="UCI Wine Quality (red)",
        quality_threshold=quality_threshold,
    )


def load_wine_white_dataset(
    *,
    cache_path: Path | None = None,
    quality_threshold: int = 6,
    download_timeout: float = 30.0,
) -> DatasetBundle:
    """Load the UCI White Wine Quality dataset (downloads + caches on first run)."""
    path = cache_path or _ROOT / "results" / "data" / "winequality-white.csv"
    csv_path = download_wine_quality_csv(
        cache_path=path,
        url=WINE_QUALITY_WHITE_URL,
        timeout=download_timeout,
    )
    return _build_wine_bundle(
        csv_path,
        name="UCI Wine Quality (white)",
        quality_threshold=quality_threshold,
    )


def load_iris_dataset() -> DatasetBundle:
    """Load the bundled Iris dataset (offline) as a binary virginica band.

    Setosa is linearly separable from the other two species, so a held-out
    classifier errs almost exclusively on the genuine versicolor/virginica
    overlap.  This makes Iris a clean test of whether TNFR phase stress
    localizes the *single* real interface in an otherwise easy dataset.
    """
    from sklearn.datasets import load_iris

    data = load_iris()
    feature_names = [str(name) for name in data.feature_names]
    feature_label = "petal length (cm)"
    feature_index = feature_names.index(feature_label)

    records: list[dict[str, Any]] = []
    for row, target in zip(data.data, data.target):
        species = str(data.target_names[int(target)])
        record: dict[str, Any] = {
            name: float(value) for name, value in zip(feature_names, row)
        }
        record["species"] = species
        record["species_band"] = (
            "virginica" if species == "virginica" else "other"
        )
        record["feature_value"] = float(row[feature_index])
        records.append(record)

    return DatasetBundle(
        name="Iris (virginica boundary)",
        sector="botany",
        records=records,
        feature_keys=feature_names,
        state_key="species_band",
        positive_value="virginica",
        feature_label=feature_label,
        node_attributes=["species", "species_band", "feature_value"],
        metadata={"source": "scikit-learn bundled", "online": False},
    )


def load_digits_dataset() -> DatasetBundle:
    """Load the bundled handwritten digits dataset (offline) as even/odd parity.

    Even-vs-odd parity from raw 8x8 pixels is a genuinely hard, high-dimensional
    binary task whose held-out errors are spread across the manifold rather than
    concentrated on a single clean boundary.  It is the hardest stress test in
    the suite and is included precisely so the comparison is not cherry-picked.
    """
    from sklearn.datasets import load_digits

    data = load_digits()
    feature_names = [f"pixel_{index // 8}_{index % 8}" for index in range(64)]
    feature_label = "mean pixel intensity"

    records: list[dict[str, Any]] = []
    for row, target in zip(data.data, data.target):
        digit = int(target)
        record: dict[str, Any] = {
            name: float(value) for name, value in zip(feature_names, row)
        }
        record["digit"] = digit
        record["parity"] = "even" if digit % 2 == 0 else "odd"
        record["feature_value"] = float(sum(row) / len(row))
        records.append(record)

    return DatasetBundle(
        name="Digits (even/odd parity)",
        sector="computer vision",
        records=records,
        feature_keys=feature_names,
        state_key="parity",
        positive_value="even",
        feature_label=feature_label,
        node_attributes=["digit", "parity", "feature_value"],
        metadata={"source": "scikit-learn bundled", "online": False},
    )


DATASET_LOADERS: Mapping[str, Callable[..., DatasetBundle]] = {
    "wdbc": load_wdbc_dataset,
    "wine": load_wine_dataset,
    "wine_white": load_wine_white_dataset,
    "iris": load_iris_dataset,
    "digits": load_digits_dataset,
}


# ---------------------------------------------------------------------------
# Graph preparation and circular target
# ---------------------------------------------------------------------------


def attach_incident_conflict_pressure(G: Any, state_key: str) -> None:
    """Set ΔNFR/coherence from local state-disagreement rate (construction step).

    This mirrors examples 91/92: structural potential telemetry (Φ_s) only has
    meaning if each node carries a ``delta_nfr`` derived from its graph-local
    conflict.  It is part of graph construction, not a validation-time mutation,
    and no operator is applied.
    """
    for node in G.nodes():
        degree = G.degree[node]
        conflicts = sum(
            1
            for neighbour in G.neighbors(node)
            if G.nodes[neighbour].get(state_key) != G.nodes[node].get(state_key)
        )
        conflict_rate = conflicts / degree if degree else 0.0
        G.nodes[node]["incident_state_conflicts"] = int(conflicts)
        G.nodes[node]["delta_nfr"] = float(conflict_rate)
        G.nodes[node]["dnfr"] = float(conflict_rate)
        G.nodes[node]["coherence"] = 1.0 / (1.0 + conflict_rate)


def circular_review_labels(
    G: Any, *, conflict_threshold: int
) -> dict[Any, bool]:
    """Circular review target: nodes with enough opposite-state neighbours.

    Declared circular because the TNFR phase stress also consumes these
    conflicts.  This is a localization sanity check, not an external target.
    """
    return {
        node: int(G.nodes[node].get("incident_state_conflicts", 0))
        >= int(conflict_threshold)
        for node in G.nodes()
    }


def prepare_interface_graph(
    bundle: DatasetBundle,
    *,
    k: int,
) -> Any:
    """Build the kNN graph, encode phase, and attach conflict pressure."""
    G = build_knn_graph(
        bundle.records,
        bundle.feature_keys,
        k=k,
        node_attributes=bundle.node_attributes,
    )
    encode_phase_from_binary_state(
        G, bundle.state_key, positive_value=bundle.positive_value
    )
    attach_incident_conflict_pressure(G, bundle.state_key)
    return G


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _auto_conflict_threshold(labels_source: Mapping[Any, int]) -> int:
    """Pick a conflict threshold that yields a non-empty, non-trivial target.

    Uses the 90th percentile of incident-conflict counts (at least 1), so the
    review set stays small relative to the graph for any dataset/k.
    """
    counts = sorted(int(v) for v in labels_source.values())
    if not counts:
        return 1
    index = min(len(counts) - 1, int(math.ceil(0.9 * len(counts))) - 1)
    return max(1, counts[index])


def run_dataset_benchmark(
    bundle: DatasetBundle,
    *,
    k: int = 8,
    seed: int = 0,
    conflict_threshold: int | None = None,
    top_n: int = 10,
    output_dir: Path | None = None,
    stem: str | None = None,
) -> dict[str, Any]:
    """Run the structural-interface benchmark for one dataset.

    Returns a result mapping compatible with the structural-interface renderers
    and containing the full classical baseline comparison.
    """
    G = prepare_interface_graph(bundle, k=k)

    incident = {
        node: int(G.nodes[node].get("incident_state_conflicts", 0))
        for node in G.nodes()
    }
    if conflict_threshold is None:
        conflict_threshold = _auto_conflict_threshold(incident)
    labels = circular_review_labels(G, conflict_threshold=conflict_threshold)

    problem = StructuralInterfaceProblem(
        graph=G,
        state_key=bundle.state_key,
        domain=bundle.sector,
        metadata=dict(bundle.metadata),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scores = score_structural_interfaces(problem, state_key=bundle.state_key)
        baselines = full_baseline_score_maps(
            G,
            state_key=bundle.state_key,
            feature_key="feature_value",
            seed=seed,
        )

    tnfr_scores = interface_score_maps(scores)
    score_maps: dict[str, Mapping[Any, float]] = {TNFR_SCORE_LABEL: tnfr_scores}
    score_maps.update(baselines)

    evaluation = evaluate_interface_scores(labels, score_maps)
    hotspots = [score.as_dict() for score in scores[: int(top_n)]]

    result: dict[str, Any] = {
        "dataset": {
            "name": bundle.name,
            "sector": bundle.sector,
            "samples": len(bundle.records),
            "feature_count": len(bundle.feature_keys),
            "feature_baseline": bundle.feature_label,
            **bundle.metadata,
        },
        "graph": {
            "construction": f"standardized {k}-NN graph",
            "k": int(k),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
        "task": {
            "target_definition": (
                "circular local disagreement: incident opposite-state "
                f"neighbours >= {conflict_threshold}"
            ),
            "target_kind": "circular_local_disagreement",
            "is_circular_target": True,
            "conflict_threshold": int(conflict_threshold),
        },
        "evaluation": evaluation,
        "baselines": {
            "names": sorted(baselines),
            "formulas": {
                name: BASELINE_FORMULAS[name]
                for name in baselines
                if name in BASELINE_FORMULAS
            },
        },
        "hotspots": hotspots,
        "seed": int(seed),
        "honest_interpretation": (
            "The review target is the circular local-disagreement target, which "
            "the TNFR phase stress also consumes. High AUC here is a localization "
            "sanity check, not external superiority over classical graph metrics. "
            "TNFR is compared against the closest classical baseline "
            "(local_disagreement) and stronger references (graph total variation, "
            "local class entropy, label-propagation residual, graph cut). "
            "Non-circular held-out targets are introduced in Milestone 4."
        ),
    }

    if output_dir is not None:
        paths = export_structural_interface_report(
            result,
            Path(output_dir),
            stem=stem or _default_stem(bundle.name),
        )
        result["report_paths"] = {key: str(path) for key, path in paths.items()}

    return result


def _default_stem(dataset_name: str) -> str:
    slug = "".join(
        char.lower() if char.isalnum() else "_" for char in dataset_name
    ).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"structural_interface_{slug}"


def _tnfr_vs_closest_baseline(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract TNFR vs local_disagreement AUC/precision for the summary."""
    rows = {
        row["score"]: row
        for row in result.get("evaluation", {}).get("score_comparison", [])
    }
    tnfr = rows.get(TNFR_SCORE_LABEL, {})
    closest = rows.get("local_disagreement", {})
    return {
        "tnfr_auc": tnfr.get("auc"),
        "tnfr_precision_at_review_count": tnfr.get("precision_at_review_count"),
        "local_disagreement_auc": closest.get("auc"),
        "local_disagreement_precision_at_review_count": closest.get(
            "precision_at_review_count"
        ),
    }


# ---------------------------------------------------------------------------
# Milestone 4: non-circular held-out model-error target
# ---------------------------------------------------------------------------

#: Classical baselines that the non-circular validation must report TNFR against
#: (Milestone 4 acceptance criteria).
REFERENCE_BASELINES: tuple[str, ...] = (
    "local_disagreement",
    "local_class_entropy",
    "graph_total_variation",
    "label_propagation_residual",
)


@dataclass(frozen=True)
class ModelErrorTarget:
    """Held-out (out-of-fold) model-error review target.

    Every record receives a prediction from a classifier that did **not** see it
    during training (stratified k-fold out-of-fold prediction).  A node is a
    review case when the held-out prediction disagrees with the true label.
    This target is independent of the graph-local disagreement that the TNFR
    phase stress consumes, so it is a *non-circular* validation target.
    """

    error_labels: dict[Any, bool]
    classifier_name: str
    cv_folds: int
    cv_accuracy: float
    error_count: int


def held_out_model_error_labels(
    bundle: DatasetBundle,
    *,
    seed: int = 0,
    n_splits: int = 5,
) -> ModelErrorTarget:
    """Compute out-of-fold misclassification labels for every record.

    Uses a deterministic ``StandardScaler + LogisticRegression`` pipeline with
    stratified k-fold cross-validation.  Node ids match
    :func:`build_knn_graph` (integer indices ``0..n-1`` over ``bundle.records``),
    so the returned labels align with the structural-interface graph.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(
        [[float(rec[key]) for key in bundle.feature_keys] for rec in bundle.records],
        dtype=float,
    )
    targets = np.asarray(
        [
            1 if rec.get(bundle.state_key) == bundle.positive_value else 0
            for rec in bundle.records
        ],
        dtype=int,
    )

    smallest_class = int(np.bincount(targets).min()) if targets.size else 0
    folds = max(2, min(int(n_splits), smallest_class if smallest_class >= 2 else 2))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    oof_pred = cross_val_predict(pipeline, features, targets, cv=cv)

    error_labels = {
        index: bool(int(oof_pred[index]) != int(targets[index]))
        for index in range(len(targets))
    }
    accuracy = float((oof_pred == targets).mean()) if targets.size else 0.0
    return ModelErrorTarget(
        error_labels=error_labels,
        classifier_name="StandardScaler+LogisticRegression",
        cv_folds=folds,
        cv_accuracy=accuracy,
        error_count=int(sum(error_labels.values())),
    )


def run_dataset_model_error_benchmark(
    bundle: DatasetBundle,
    *,
    k: int = 8,
    seed: int = 0,
    n_splits: int = 5,
    top_n: int = 10,
    output_dir: Path | None = None,
    stem: str | None = None,
) -> dict[str, Any]:
    """Run the structural-interface benchmark against a non-circular target.

    The graph, the TNFR phase stress, and the classical baselines are identical
    to :func:`run_dataset_benchmark`.  Only the review target changes: instead of
    local disagreement it is the held-out (out-of-fold) model error, which is
    independent of the conflicts that TNFR phase stress consumes.
    """
    G = prepare_interface_graph(bundle, k=k)
    target = held_out_model_error_labels(bundle, seed=seed, n_splits=n_splits)
    labels = target.error_labels

    problem = StructuralInterfaceProblem(
        graph=G,
        state_key=bundle.state_key,
        domain=bundle.sector,
        metadata=dict(bundle.metadata),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scores = score_structural_interfaces(problem, state_key=bundle.state_key)
        baselines = full_baseline_score_maps(
            G,
            state_key=bundle.state_key,
            feature_key="feature_value",
            seed=seed,
        )

    tnfr_scores = interface_score_maps(scores)
    score_maps: dict[str, Mapping[Any, float]] = {TNFR_SCORE_LABEL: tnfr_scores}
    score_maps.update(baselines)

    evaluation = evaluate_interface_scores(labels, score_maps)
    hotspots = [score.as_dict() for score in scores[: int(top_n)]]

    result: dict[str, Any] = {
        "dataset": {
            "name": bundle.name,
            "sector": bundle.sector,
            "samples": len(bundle.records),
            "feature_count": len(bundle.feature_keys),
            "feature_baseline": bundle.feature_label,
            **bundle.metadata,
        },
        "graph": {
            "construction": f"standardized {k}-NN graph",
            "k": int(k),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
        "task": {
            "target_definition": (
                "held-out model error: out-of-fold misclassification by a "
                f"{target.classifier_name} pipeline ({target.cv_folds}-fold "
                "stratified CV)"
            ),
            "target_kind": "held_out_model_error",
            "is_circular_target": False,
        },
        "model": {
            "classifier": target.classifier_name,
            "cv_folds": target.cv_folds,
            "cv_accuracy": target.cv_accuracy,
            "error_count": target.error_count,
        },
        "evaluation": evaluation,
        "baselines": {
            "names": sorted(baselines),
            "formulas": {
                name: BASELINE_FORMULAS[name]
                for name in baselines
                if name in BASELINE_FORMULAS
            },
        },
        "hotspots": hotspots,
        "seed": int(seed),
        "honest_interpretation": (
            "Non-circular validation. The review target is held-out model error "
            "(out-of-fold misclassification by a StandardScaler+LogisticRegression "
            "pipeline), which is independent of the graph-local disagreement that "
            "TNFR phase stress consumes. TNFR is compared against the closest "
            "classical baselines (local disagreement, local class entropy, graph "
            "total variation, label-propagation residual). A node can be "
            "misclassified without local disagreement and vice versa, so neither "
            "TNFR nor local disagreement is guaranteed to win. A TNFR claim is "
            "accepted only if it matches or beats these baselines, or adds a "
            "stable grammar-valid operator prescription beyond them."
        ),
    }

    if output_dir is not None:
        paths = export_structural_interface_report(
            result,
            Path(output_dir),
            stem=stem or f"{_default_stem(bundle.name)}_model_error",
        )
        result["report_paths"] = {key: str(path) for key, path in paths.items()}

    return result


def _auc_by_score(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        row["score"]: row.get("auc")
        for row in result.get("evaluation", {}).get("score_comparison", [])
    }


def _tnfr_vs_reference_baselines(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract TNFR + reference-baseline AUCs for the non-circular summary."""
    rows = {
        row["score"]: row
        for row in result.get("evaluation", {}).get("score_comparison", [])
    }
    tnfr = rows.get(TNFR_SCORE_LABEL, {})
    summary: dict[str, Any] = {
        "tnfr_auc": tnfr.get("auc"),
        "tnfr_precision_at_review_count": tnfr.get("precision_at_review_count"),
    }
    for name in REFERENCE_BASELINES:
        summary[f"{name}_auc"] = rows.get(name, {}).get("auc")
    return summary


def run_benchmark_suite(
    datasets: Sequence[str],
    *,
    targets: Sequence[str] = ("circular",),
    k: int = 8,
    seed: int = 0,
    top_n: int = 10,
    output_dir: Path | None = None,
    quality_threshold: int = 6,
    download_timeout: float = 30.0,
    cv_folds: int = 5,
    wdbc_conflict_threshold: int | None = None,
    wine_conflict_threshold: int | None = None,
) -> dict[str, Any]:
    """Run the benchmark over one or more datasets and write a consolidated summary.

    ``targets`` selects the review target(s):

    - ``"circular"`` — local-disagreement proof-of-concept target (Milestone 3);
    - ``"model_error"`` — held-out out-of-fold misclassification, the
      non-circular validation target (Milestone 4).

    Online datasets that cannot be reached are skipped gracefully (recorded in
    the ``skipped`` section) so offline runs still produce a report.  A
    requested target that fails for one dataset (for example ``model_error``
    without scikit-learn) is also skipped without aborting the suite.
    """
    requested = tuple(targets)
    run_circular = "circular" in requested
    run_model_error = "model_error" in requested

    circular_results: list[dict[str, Any]] = []
    model_error_results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for name in datasets:
        loader = DATASET_LOADERS.get(name)
        if loader is None:
            skipped.append({"dataset": name, "reason": "unknown dataset"})
            continue

        try:
            if name in {"wine", "wine_white"}:
                bundle = loader(
                    quality_threshold=quality_threshold,
                    download_timeout=download_timeout,
                )
                conflict_threshold = wine_conflict_threshold
            elif name == "wdbc":
                bundle = loader()
                conflict_threshold = wdbc_conflict_threshold
            else:
                bundle = loader()
                conflict_threshold = None
        except Exception as exc:  # noqa: BLE001 - graceful skip on data failure
            skipped.append({"dataset": name, "reason": str(exc)})
            continue

        if run_circular:
            circular_results.append(
                run_dataset_benchmark(
                    bundle,
                    k=k,
                    seed=seed,
                    conflict_threshold=conflict_threshold,
                    top_n=top_n,
                    output_dir=output_dir,
                    stem=_default_stem(bundle.name),
                )
            )

        if run_model_error:
            try:
                model_error_results.append(
                    run_dataset_model_error_benchmark(
                        bundle,
                        k=k,
                        seed=seed,
                        n_splits=cv_folds,
                        top_n=top_n,
                        output_dir=output_dir,
                        stem=f"{_default_stem(bundle.name)}_model_error",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - graceful skip (e.g. no sklearn)
                skipped.append(
                    {
                        "dataset": name,
                        "target": "model_error",
                        "reason": str(exc),
                    }
                )

    summary = {
        "suite": "structural_interface_benchmark",
        "config": {
            "datasets": list(datasets),
            "targets": list(requested),
            "k": int(k),
            "seed": int(seed),
            "top_n": int(top_n),
            "cv_folds": int(cv_folds),
        },
        "proof_of_concept_targets": [
            {
                "dataset": result["dataset"]["name"],
                "sector": result["dataset"]["sector"],
                "target_kind": result["task"]["target_kind"],
                "review_node_count": result["evaluation"]["review_node_count"],
                "total_nodes": result["evaluation"]["total_nodes"],
                **_tnfr_vs_closest_baseline(result),
            }
            for result in circular_results
        ],
        "independent_targets": [
            {
                "dataset": result["dataset"]["name"],
                "sector": result["dataset"]["sector"],
                "target_kind": result["task"]["target_kind"],
                "review_node_count": result["evaluation"]["review_node_count"],
                "total_nodes": result["evaluation"]["total_nodes"],
                "cv_accuracy": result["model"]["cv_accuracy"],
                "error_count": result["model"]["error_count"],
                **_tnfr_vs_reference_baselines(result),
            }
            for result in model_error_results
        ],
        "independent_targets_note": (
            "Non-circular validation uses held-out out-of-fold model error as the "
            "review target, compared against the closest classical baselines "
            "(local disagreement, local class entropy, graph total variation, "
            "label-propagation residual)."
            if run_model_error
            else (
                "Non-circular held-out targets (downstream model error, temporal "
                "transition, perturbation sensitivity) are available via the "
                "'model_error' target."
            )
        ),
        "skipped": skipped,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "structural_interface_benchmark_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        summary["summary_path"] = str(summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_datasets(value: str) -> list[str]:
    if value == "all":
        return ["wdbc", "wine", "wine_white", "iris", "digits"]
    if value == "offline":
        return ["wdbc", "iris", "digits"]
    return [item.strip() for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TNFR structural-interface cross-domain benchmark"
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset(s): 'wdbc', 'wine', 'wine_white', 'iris', 'digits', a "
            "comma-separated list, 'offline' (no-network subset), or 'all' "
            "(default)."
        ),
    )
    parser.add_argument(
        "--k", type=int, default=8, help="Neighbours per node (default 8)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for random baseline (default 0)."
    )
    parser.add_argument(
        "--target",
        default="circular",
        choices=["circular", "model_error", "all"],
        help=(
            "Review target: 'circular' (local-disagreement proof of concept), "
            "'model_error' (held-out out-of-fold misclassification, non-circular), "
            "or 'all'. Default 'circular'."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Stratified CV folds for the model-error target (default 5).",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Top hotspots to report (default 10)."
    )
    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=6,
        help="Wine quality high/low split (default 6).",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=30.0,
        help="Online download timeout in seconds (default 30).",
    )
    parser.add_argument(
        "--output",
        default="results/reports",
        help="Output directory for reports (default results/reports).",
    )
    return parser


_TARGET_CHOICES: Mapping[str, tuple[str, ...]] = {
    "circular": ("circular",),
    "model_error": ("model_error",),
    "all": ("circular", "model_error"),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    datasets = _parse_datasets(args.dataset)
    targets = _TARGET_CHOICES[args.target]
    output_dir = Path(args.output) if args.output else None

    summary = run_benchmark_suite(
        datasets,
        targets=targets,
        k=args.k,
        seed=args.seed,
        top_n=args.top_n,
        output_dir=output_dir,
        quality_threshold=args.quality_threshold,
        download_timeout=args.download_timeout,
        cv_folds=args.cv_folds,
    )

    print("TNFR Structural Interface Benchmark")
    print(f"  datasets requested: {', '.join(datasets)}")
    print(f"  targets: {', '.join(targets)}")
    for entry in summary["proof_of_concept_targets"]:
        print(
            f"  [{entry['dataset']}] circular target — "
            f"TNFR AUC={entry['tnfr_auc']:.3f} "
            f"(local_disagreement AUC={entry['local_disagreement_auc']:.3f}); "
            f"review nodes={entry['review_node_count']}/{entry['total_nodes']}"
        )
    for entry in summary["independent_targets"]:
        print(
            f"  [{entry['dataset']}] held-out model error — "
            f"TNFR AUC={entry['tnfr_auc']:.3f} "
            f"(local_disagreement AUC={entry['local_disagreement_auc']:.3f}, "
            f"entropy AUC={entry['local_class_entropy_auc']:.3f}); "
            f"errors={entry['review_node_count']}/{entry['total_nodes']} "
            f"(CV acc={entry['cv_accuracy']:.3f})"
        )
    for entry in summary["skipped"]:
        target_note = f" [{entry['target']}]" if entry.get("target") else ""
        print(f"  [skip] {entry['dataset']}{target_note}: {entry['reason']}")
    if output_dir is not None:
        print(f"  summary -> {summary.get('summary_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
