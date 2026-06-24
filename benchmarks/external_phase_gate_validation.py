#!/usr/bin/env python3
"""External phase-gate validation battery.

Question
--------
Does TNFR local phase telemetry add predictive information for edge-local
coupling compatibility that topology-only and global phase-order baselines
miss?

The task is intentionally narrow and falsifiable.  Every sample uses the same
graph topology, but different node phase assignments.  The external target is
the fraction of graph edges whose wrapped phase difference falls inside a fixed
coupling window.  This creates paired states with identical topology and nearly
identical/global-identical phase histograms:

* smooth travelling waves: locally compatible on a cycle graph;
* scrambled waves: same phases, randomly assigned to nodes, locally broken.

Global order metrics and topology-only metrics cannot distinguish those paired
states.  TNFR's graph-local phase gradient should distinguish them because it
measures phase differences along actual coupling edges.

This benchmark is not a proof of broad physical validity.  It is a small,
reproducible example showing where TNFR-style telemetry is operationally useful.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

loaded_tnfr = sys.modules.get("tnfr")
loaded_path = str(getattr(loaded_tnfr, "__file__", "")) if loaded_tnfr else ""
if loaded_tnfr is not None and not loaded_path.startswith(str(SRC_DIR)):
    for name in list(sys.modules):
        if name == "tnfr" or name.startswith("tnfr."):
            del sys.modules[name]

import networkx as nx  # noqa: E402

from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.validation.phase_gate import (  # noqa: E402
    compare_against_global_baselines,
    edge_phase_differences,
    wrap_angle,
)

TAU = 2.0 * math.pi
DEFAULT_PATTERNS: tuple[str, ...] = (
    "coherent",
    "smooth_wave_q1",
    "smooth_wave_q2",
    "smooth_wave_q4",
    "two_domain",
    "scrambled_wave_q1",
    "scrambled_wave_q2",
    "scrambled_wave_q4",
    "random_uniform",
    "alternating_antiphase",
)


@dataclass(frozen=True)
class ScalarRule:
    """A one-feature threshold classifier fitted on training records."""

    model: str
    feature: str
    threshold: float
    positive_when: str
    train_balanced_accuracy: float


def build_cycle_graph(nodes: int) -> nx.Graph:
    """Build the fixed topology used by the validation battery."""
    if nodes < 8:
        raise ValueError("nodes must be >= 8 for the phase-gate battery")
    return nx.cycle_graph(nodes)


def travelling_wave_phases(
    nodes: int,
    q: int,
    rng: np.random.Generator,
    *,
    noise: float = 0.01,
) -> np.ndarray:
    """Create a smooth q-winding phase wave on the node order."""
    offset = float(rng.uniform(0.0, TAU))
    base = offset + TAU * q * np.arange(nodes, dtype=float) / float(nodes)
    phases = base + rng.normal(0.0, noise, size=nodes)
    return np.mod(phases, TAU)


def phases_for_pattern(
    pattern: str,
    nodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate phases for a named validation pattern."""
    if pattern == "coherent":
        center = float(rng.uniform(0.0, TAU))
        return np.mod(center + rng.normal(0.0, 0.04, size=nodes), TAU)

    if pattern.startswith("smooth_wave_q"):
        q = int(pattern.rsplit("q", 1)[1])
        return travelling_wave_phases(nodes, q, rng)

    if pattern.startswith("scrambled_wave_q"):
        q = int(pattern.rsplit("q", 1)[1])
        phases = travelling_wave_phases(nodes, q, rng)
        return rng.permutation(phases)

    if pattern == "two_domain":
        offset = float(rng.uniform(0.0, TAU))
        phases = np.empty(nodes, dtype=float)
        phases[: nodes // 2] = offset
        phases[nodes // 2 :] = offset + math.pi
        phases += rng.normal(0.0, 0.02, size=nodes)
        return np.mod(phases, TAU)

    if pattern == "random_uniform":
        return rng.uniform(0.0, TAU, size=nodes)

    if pattern == "alternating_antiphase":
        offset = float(rng.uniform(0.0, TAU))
        phases = np.asarray(
            [offset + (math.pi if i % 2 else 0.0) for i in range(nodes)],
            dtype=float,
        )
        phases += rng.normal(0.0, 0.02, size=nodes)
        return np.mod(phases, TAU)

    raise ValueError(f"unknown validation pattern: {pattern}")


def assign_phase_state(G: nx.Graph, phases: Sequence[float]) -> None:
    """Attach phase and local structural-pressure attributes to graph nodes."""
    nodes = sorted(G.nodes())
    if len(nodes) != len(phases):
        raise ValueError("phase count must match graph node count")

    for node, phase in zip(nodes, phases):
        value = float(phase)
        G.nodes[node]["phase"] = value
        G.nodes[node]["theta"] = value

    for node in nodes:
        diffs = [
            abs(wrap_angle(G.nodes[neighbor]["phase"] - G.nodes[node]["phase"]))
            for neighbor in G.neighbors(node)
        ]
        local_pressure = float(np.mean(diffs) / math.pi) if diffs else 0.0
        G.nodes[node]["delta_nfr"] = local_pressure
        G.nodes[node]["dnfr"] = local_pressure
        G.nodes[node]["coherence"] = 1.0 / (1.0 + local_pressure)


def feature_row(
    *,
    seed: int,
    pattern: str,
    nodes: int,
    gate: float,
    min_compliance: float,
) -> dict[str, Any]:
    """Generate one benchmark record."""
    rng = np.random.default_rng(seed)
    phases = phases_for_pattern(pattern, nodes, rng)
    return feature_row_from_phases(
        seed=seed,
        pattern=pattern,
        nodes=nodes,
        phases=phases,
        gate=gate,
        min_compliance=min_compliance,
    )


def feature_row_from_phases(
    *,
    seed: int,
    pattern: str,
    nodes: int,
    phases: Sequence[float],
    gate: float,
    min_compliance: float,
) -> dict[str, Any]:
    """Generate one benchmark record from a fixed phase assignment."""
    G = build_cycle_graph(nodes)
    assign_phase_state(G, phases)

    diffs = edge_phase_differences(G)
    features = compare_against_global_baselines(
        G,
        gate,
        min_compliance=min_compliance,
    )

    return {
        "seed": seed,
        "pattern": pattern,
        "nodes": nodes,
        "edges": G.number_of_edges(),
        "label": bool(features["label"]),
        "edge_gate_compliance": float(features["edge_gate_compliance"]),
        "edge_diff_mean": float(np.mean(diffs)),
        "edge_diff_max": float(np.max(diffs)),
        "tnfr_mean_phase_gradient": float(features["tnfr_mean_phase_gradient"]),
        "tnfr_max_phase_gradient": float(features["tnfr_max_phase_gradient"]),
        "tnfr_mean_abs_curvature": float(features["tnfr_mean_abs_curvature"]),
        "tnfr_phi_s_abs_mean": float(features["tnfr_phi_s_abs_mean"]),
        "global_order_r": float(features["global_order_r"]),
        "circular_variance": float(features["circular_variance"]),
        "phase_histogram_entropy": float(features["phase_histogram_entropy"]),
        "topology_avg_degree": float(features["topology_avg_degree"]),
        "topology_clustering": float(features["topology_clustering"]),
        "topology_diameter": float(features["topology_diameter"]),
    }


def generate_records(
    *,
    nodes: int,
    runs: int,
    gate: float,
    min_compliance: float,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
) -> list[dict[str, Any]]:
    """Generate the full validation dataset."""
    records: list[dict[str, Any]] = []
    pattern_indices = {pattern: index for index, pattern in enumerate(patterns)}
    for run in range(runs):
        base_seed = 10_000 + run * 101
        paired_scrambled: set[str] = set()
        for pattern_offset, pattern in enumerate(patterns):
            seed = base_seed + pattern_offset
            if pattern in paired_scrambled:
                continue
            if pattern.startswith("smooth_wave_q"):
                q = int(pattern.rsplit("q", 1)[1])
                rng = np.random.default_rng(seed)
                phases = travelling_wave_phases(nodes, q, rng)
                records.append(
                    feature_row_from_phases(
                        seed=seed,
                        pattern=pattern,
                        nodes=nodes,
                        phases=phases,
                        gate=gate,
                        min_compliance=min_compliance,
                    )
                )
                scrambled = f"scrambled_wave_q{q}"
                if scrambled in pattern_indices:
                    scrambled_seed = base_seed + pattern_indices[scrambled]
                    scrambled_rng = np.random.default_rng(scrambled_seed)
                    records.append(
                        feature_row_from_phases(
                            seed=scrambled_seed,
                            pattern=scrambled,
                            nodes=nodes,
                            phases=scrambled_rng.permutation(phases),
                            gate=gate,
                            min_compliance=min_compliance,
                        )
                    )
                    paired_scrambled.add(scrambled)
                continue
            records.append(
                feature_row(
                    seed=seed,
                    pattern=pattern,
                    nodes=nodes,
                    gate=gate,
                    min_compliance=min_compliance,
                )
            )
    return records


def split_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split records by run seed parity."""
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for record in records:
        run_index = (int(record["seed"]) - 10_000) // 101
        if run_index % 2 == 0:
            train.append(record)
        else:
            test.append(record)
    return train, test


def confusion_counts(
    labels: Sequence[bool],
    predictions: Sequence[bool],
) -> dict[str, int]:
    """Compute binary confusion counts."""
    tp = sum(bool(y) and bool(p) for y, p in zip(labels, predictions))
    tn = sum((not bool(y)) and (not bool(p)) for y, p in zip(labels, predictions))
    fp = sum((not bool(y)) and bool(p) for y, p in zip(labels, predictions))
    fn = sum(bool(y) and (not bool(p)) for y, p in zip(labels, predictions))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_metrics(
    labels: Sequence[bool],
    predictions: Sequence[bool],
) -> dict[str, float]:
    """Compute robust binary-classification metrics without external ML deps."""
    counts = confusion_counts(labels, predictions)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {
        **counts,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
    }


def threshold_candidates(values: Sequence[float]) -> list[float]:
    """Return stable threshold candidates from observed scalar values."""
    unique = sorted(set(float(v) for v in values if math.isfinite(float(v))))
    if not unique:
        return [0.0]
    candidates = [unique[0] - 1.0]
    candidates.extend((a + b) / 2.0 for a, b in zip(unique, unique[1:]))
    candidates.append(unique[-1] + 1.0)
    candidates.extend(unique)
    return sorted(set(candidates))


def predict_with_rule(
    rule: ScalarRule, records: Sequence[dict[str, Any]]
) -> list[bool]:
    """Apply a fitted scalar threshold rule."""
    predictions: list[bool] = []
    for record in records:
        value = float(record[rule.feature])
        if rule.positive_when == "<=":
            predictions.append(value <= rule.threshold)
        else:
            predictions.append(value >= rule.threshold)
    return predictions


def fit_scalar_rule(
    records: Sequence[dict[str, Any]],
    *,
    model: str,
    feature: str,
) -> ScalarRule:
    """Fit a one-dimensional threshold rule by balanced accuracy."""
    labels = [bool(record["label"]) for record in records]
    values = [float(record[feature]) for record in records]
    best: ScalarRule | None = None
    for threshold in threshold_candidates(values):
        for positive_when in ("<=", ">="):
            if positive_when == "<=":
                predictions = [value <= threshold for value in values]
            else:
                predictions = [value >= threshold for value in values]
            score = classification_metrics(labels, predictions)["balanced_accuracy"]
            candidate = ScalarRule(model, feature, threshold, positive_when, score)
            if best is None or score > best.train_balanced_accuracy:
                best = candidate
    if best is None:
        raise RuntimeError(f"could not fit scalar rule for {feature}")
    return best


def majority_baseline(records: Sequence[dict[str, Any]]) -> bool:
    """Return the majority class from training records."""
    positives = sum(bool(record["label"]) for record in records)
    return positives >= (len(records) - positives)


def evaluate_rule(
    rule: ScalarRule,
    train: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one scalar rule on train and test splits."""
    train_labels = [bool(record["label"]) for record in train]
    test_labels = [bool(record["label"]) for record in test]
    train_predictions = predict_with_rule(rule, train)
    test_predictions = predict_with_rule(rule, test)
    return {
        "model": rule.model,
        "feature": rule.feature,
        "threshold": rule.threshold,
        "positive_when": rule.positive_when,
        "train": classification_metrics(train_labels, train_predictions),
        "test": classification_metrics(test_labels, test_predictions),
    }


def evaluate_models(
    train: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fit and evaluate all benchmark baselines."""
    feature_models = [
        ("TNFR mean grad_phi", "tnfr_mean_phase_gradient"),
        ("TNFR max grad_phi", "tnfr_max_phase_gradient"),
        ("TNFR mean abs K_phi", "tnfr_mean_abs_curvature"),
        ("TNFR abs Phi_s stress", "tnfr_phi_s_abs_mean"),
        ("Global order parameter R", "global_order_r"),
        ("Circular variance", "circular_variance"),
        ("Phase histogram entropy", "phase_histogram_entropy"),
        ("Topology average degree", "topology_avg_degree"),
        ("Topology clustering", "topology_clustering"),
        ("Topology diameter", "topology_diameter"),
    ]
    results = [
        evaluate_rule(fit_scalar_rule(train, model=model, feature=feature), train, test)
        for model, feature in feature_models
    ]

    majority = majority_baseline(train)
    train_labels = [bool(record["label"]) for record in train]
    test_labels = [bool(record["label"]) for record in test]
    results.append(
        {
            "model": "Majority class",
            "feature": "label frequency",
            "threshold": None,
            "positive_when": "constant",
            "train": classification_metrics(train_labels, [majority] * len(train)),
            "test": classification_metrics(test_labels, [majority] * len(test)),
        }
    )
    return sorted(
        results,
        key=lambda row: (row["test"]["balanced_accuracy"], row["test"]["mcc"]),
        reverse=True,
    )


def paired_wave_checks(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare smooth and scrambled wave pairs with matched phase histograms."""
    by_key = {(record["seed"], record["pattern"]): record for record in records}
    deltas: dict[str, list[float]] = {
        "global_order_r": [],
        "phase_histogram_entropy": [],
        "tnfr_mean_phase_gradient": [],
        "tnfr_mean_abs_curvature": [],
    }
    label_flips = 0
    pair_count = 0

    # Seeds differ by pattern index in DEFAULT_PATTERNS, so reconstruct by run.
    pattern_index = {pattern: index for index, pattern in enumerate(DEFAULT_PATTERNS)}
    for record in records:
        pattern = str(record["pattern"])
        if not pattern.startswith("smooth_wave_q"):
            continue
        q = pattern.rsplit("q", 1)[1]
        scrambled = f"scrambled_wave_q{q}"
        run_index = (int(record["seed"]) - 10_000) // 101
        smooth_seed = 10_000 + run_index * 101 + pattern_index[pattern]
        scrambled_seed = 10_000 + run_index * 101 + pattern_index[scrambled]
        smooth_record = by_key.get((smooth_seed, pattern))
        scrambled_record = by_key.get((scrambled_seed, scrambled))
        if smooth_record is None or scrambled_record is None:
            continue
        pair_count += 1
        if bool(smooth_record["label"]) != bool(scrambled_record["label"]):
            label_flips += 1
        for feature in deltas:
            deltas[feature].append(
                abs(float(smooth_record[feature]) - float(scrambled_record[feature]))
            )

    return {
        "pair_count": pair_count,
        "label_flips": label_flips,
        "median_abs_delta_global_order_r": median(deltas["global_order_r"]),
        "median_abs_delta_phase_histogram_entropy": median(
            deltas["phase_histogram_entropy"]
        ),
        "median_abs_delta_tnfr_mean_phase_gradient": median(
            deltas["tnfr_mean_phase_gradient"]
        ),
        "median_abs_delta_tnfr_mean_abs_curvature": median(
            deltas["tnfr_mean_abs_curvature"]
        ),
    }


def median(values: Sequence[float]) -> float:
    """Return a float median with a safe empty fallback."""
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=float)))


def pattern_summary(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate label and feature statistics by pattern."""
    rows: list[dict[str, Any]] = []
    for pattern in DEFAULT_PATTERNS:
        subset = [record for record in records if record["pattern"] == pattern]
        if not subset:
            continue
        rows.append(
            {
                "pattern": pattern,
                "n": len(subset),
                "positive_rate": float(np.mean([record["label"] for record in subset])),
                "edge_gate_compliance_mean": float(
                    np.mean([record["edge_gate_compliance"] for record in subset])
                ),
                "tnfr_mean_phase_gradient_mean": float(
                    np.mean([record["tnfr_mean_phase_gradient"] for record in subset])
                ),
                "global_order_r_mean": float(
                    np.mean([record["global_order_r"] for record in subset])
                ),
            }
        )
    return rows


def run_validation(
    *,
    nodes: int = 64,
    runs: int = 40,
    gate: float = math.pi / 4.0,
    min_compliance: float = 0.90,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
    output_html: Path | None = None,
) -> dict[str, Any]:
    """Run the full validation battery and optionally write reports."""
    records = generate_records(
        nodes=nodes,
        runs=runs,
        gate=gate,
        min_compliance=min_compliance,
    )
    train, test = split_records(records)
    model_results = evaluate_models(train, test)
    summary = {
        "metadata": {
            "nodes": nodes,
            "runs": runs,
            "gate_radians": gate,
            "min_edge_compliance": min_compliance,
            "patterns": list(DEFAULT_PATTERNS),
            "train_records": len(train),
            "test_records": len(test),
            "target": "edge-local phase-gate compatibility",
            "scope_note": (
                "Narrow synthetic validation; evidence of operational value, "
                "not a broad physical proof."
            ),
        },
        "model_results": model_results,
        "paired_wave_checks": paired_wave_checks(records),
        "pattern_summary": pattern_summary(records),
        "records": records,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    if output_html is not None:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(render_html(summary), encoding="utf-8")
    return summary


def format_float(value: Any, digits: int = 4) -> str:
    """Format report floats compactly."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a Markdown table."""
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the validation report as Markdown."""
    meta = summary["metadata"]
    model_rows = [
        [
            row["model"],
            row["feature"],
            row["positive_when"],
            format_float(row["threshold"]),
            format_float(row["test"]["balanced_accuracy"]),
            format_float(row["test"]["accuracy"]),
            format_float(row["test"]["mcc"]),
        ]
        for row in summary["model_results"]
    ]
    pattern_rows = [
        [
            row["pattern"],
            row["n"],
            format_float(row["positive_rate"]),
            format_float(row["edge_gate_compliance_mean"]),
            format_float(row["tnfr_mean_phase_gradient_mean"]),
            format_float(row["global_order_r_mean"]),
        ]
        for row in summary["pattern_summary"]
    ]
    paired = summary["paired_wave_checks"]
    return (
        "\n\n".join(
            [
                "# TNFR External Phase-Gate Validation",
                (
                    "This report tests whether graph-local TNFR phase telemetry "
                    "adds predictive information for edge-local coupling "
                    "compatibility beyond topology-only and global phase-order "
                    "baselines."
                ),
                "## Scope",
                (
                    f"Nodes: {meta['nodes']}  \n"
                    f"Runs: {meta['runs']}  \n"
                    f"Gate: {meta['gate_radians']:.6f} rad  \n"
                    f"Minimum edge compliance for a positive label: "
                    f"{meta['min_edge_compliance']:.2f}  \n"
                    f"Train records: {meta['train_records']}  \n"
                    f"Test records: {meta['test_records']}"
                ),
                "## Model comparison",
                markdown_table(
                    [
                        "Model",
                        "Feature",
                        "Positive when",
                        "Threshold",
                        "Test balanced acc.",
                        "Test acc.",
                        "Test MCC",
                    ],
                    model_rows,
                ),
                "## Matched smooth/scrambled wave check",
                (
                    f"Pairs: {paired['pair_count']}  \n"
                    f"Label flips: {paired['label_flips']}  \n"
                    f"Median |Δ global R|: "
                    f"{paired['median_abs_delta_global_order_r']:.6e}  \n"
                    f"Median |Δ phase histogram entropy|: "
                    f"{paired['median_abs_delta_phase_histogram_entropy']:.6e}  \n"
                    f"Median |Δ TNFR mean |∇φ||: "
                    f"{paired['median_abs_delta_tnfr_mean_phase_gradient']:.6f}  \n"
                    f"Median |Δ TNFR mean |Kφ||: "
                    f"{paired['median_abs_delta_tnfr_mean_abs_curvature']:.6f}"
                ),
                "## Pattern summary",
                markdown_table(
                    [
                        "Pattern",
                        "N",
                        "Positive rate",
                        "Mean compliance",
                        "Mean TNFR grad_phi",
                        "Mean global R",
                    ],
                    pattern_rows,
                ),
                "## Honest interpretation",
                (
                    "A strong TNFR result here means that graph-local phase "
                    "telemetry distinguishes locally compatible phase states that "
                    "global phase-order and topology-only baselines cannot separate. "
                    "It does not establish broad TNFR validity; it demonstrates a "
                    "specific operational advantage on a controlled coupling task."
                ),
            ]
        )
        + "\n"
    )


def render_html(summary: dict[str, Any]) -> str:
    """Render the validation report as simple standalone HTML."""
    markdown = render_markdown(summary)
    lines = markdown.splitlines()
    body: list[str] = []
    in_table = False
    table_rows: list[str] = []

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        body.append("<table>")
        for index, raw in enumerate(table_rows):
            if index == 1:
                continue
            cells = [cell.strip() for cell in raw.strip("|").split("|")]
            tag = "th" if index == 0 else "td"
            body.append(
                "<tr>"
                + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells)
                + "</tr>"
            )
        body.append("</table>")
        in_table = False
        table_rows = []

    for line in lines:
        if line.startswith("| "):
            in_table = True
            table_rows.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
    flush_table()
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>TNFR External Phase-Gate Validation</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
th {{ background: #f3f5f7; }}
</style>
</head>
<body>
{body}
</body>
</html>
""".format(
        body="\n".join(body)
    )


def print_console_summary(summary: dict[str, Any]) -> None:
    """Print a compact terminal summary."""
    print("External phase-gate validation")
    print("=" * 36)
    for row in summary["model_results"][:6]:
        test = row["test"]
        print(
            f"{row['model']:<28} "
            f"balanced_acc={test['balanced_accuracy']:.3f} "
            f"acc={test['accuracy']:.3f} mcc={test['mcc']:.3f}"
        )
    paired = summary["paired_wave_checks"]
    print()
    print(
        "Matched wave check: "
        f"pairs={paired['pair_count']}, label_flips={paired['label_flips']}, "
        f"median ΔR={paired['median_abs_delta_global_order_r']:.3e}, "
        f"median ΔTNFR-grad={paired['median_abs_delta_tnfr_mean_phase_gradient']:.3f}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--gate", type=float, default=math.pi / 4.0)
    parser.add_argument("--min-compliance", type=float, default=0.90)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "external_validation"
        / "phase_gate_validation.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=REPO_ROOT / "results" / "reports" / "external_phase_gate_validation.md",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "reports"
        / "external_phase_gate_validation.html",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    summary = run_validation(
        nodes=args.nodes,
        runs=args.runs,
        gate=args.gate,
        min_compliance=args.min_compliance,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        output_html=args.output_html,
    )
    if not args.quiet:
        print_console_summary(summary)
        print(f"\nJSON: {args.output_json}")
        print(f"Markdown: {args.output_markdown}")
        print(f"HTML: {args.output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
