#!/usr/bin/env python3
"""Example 92 — Online Wine Quality Phase-Gate Audit.

This example downloads the real UCI Red Wine Quality dataset and applies the
TNFR phase-gate monitor to a food/chemical quality-control problem:

    Which wine samples are chemically close to samples in the opposite quality
    band and should be prioritized for review?

The dataset is downloaded from UCI on first run and cached under
``results/data`` for reproducibility.  This is not a wine-scoring model; it is a
local graph audit that prioritizes chemically ambiguous samples for review.

Run:
    python examples/10_applications/92_wine_quality_phase_gate_demo.py
"""
from __future__ import annotations

import csv
import html
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

loaded_tnfr = sys.modules.get("tnfr")
loaded_path = str(getattr(loaded_tnfr, "__file__", "")) if loaded_tnfr else ""
if loaded_tnfr is not None and not loaded_path.startswith(str(SRC)):
    for name in list(sys.modules):
        if name == "tnfr" or name.startswith("tnfr."):
            del sys.modules[name]

import networkx as nx  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from tnfr.validation.phase_gate import (  # noqa: E402
    DEFAULT_PHASE_GATE,
    analyze_phase_gate,
    rank_phase_stress_hotspots,
)

WINE_QUALITY_RED_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)


def download_wine_quality_csv(
    *,
    cache_path: Path | None = None,
    url: str = WINE_QUALITY_RED_URL,
    timeout: float = 30.0,
) -> Path:
    """Download the UCI red wine quality CSV, using a local cache if present."""
    path = cache_path or ROOT / "results" / "data" / "winequality-red.csv"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = urlopen(url, timeout=timeout).read()
    path.write_bytes(payload)
    return path


def load_wine_rows(path: Path) -> list[dict[str, str]]:
    """Load semicolon-delimited UCI wine quality rows."""
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines(), delimiter=";"))


def build_wine_quality_graph(
    rows: list[dict[str, str]],
    *,
    k: int = 10,
    quality_threshold: int = 6,
) -> tuple[nx.Graph, list[str]]:
    """Build a chemistry kNN graph with quality band encoded as phase."""
    feature_names = [name for name in rows[0] if name != "quality"]
    X = [[float(row[name]) for name in feature_names] for row in rows]
    scaled = StandardScaler().fit_transform(X)
    qualities = [int(row["quality"]) for row in rows]
    quality_band = ["high" if q >= int(quality_threshold) else "low" for q in qualities]

    neighbours = NearestNeighbors(n_neighbors=int(k) + 1)
    neighbours.fit(scaled)
    distances, indices = neighbours.kneighbors(scaled)

    G = nx.Graph()
    for node, row in enumerate(rows):
        band = quality_band[node]
        phase = 0.0 if band == "high" else math.pi
        G.add_node(
            node,
            phase=phase,
            theta=phase,
            EPI=1.0,
            quality=qualities[node],
            quality_band=band,
            alcohol=float(row["alcohol"]),
            volatile_acidity=float(row["volatile acidity"]),
            sulphates=float(row["sulphates"]),
            density=float(row["density"]),
            glyph_history=[],
        )

    for node in range(len(rows)):
        for distance, neighbour in zip(distances[node][1:], indices[node][1:]):
            G.add_edge(node, int(neighbour), chemistry_distance=float(distance))

    for node in G.nodes():
        degree = G.degree[node]
        conflicts = sum(
            1
            for neighbour in G.neighbors(node)
            if G.nodes[neighbour]["quality_band"] != G.nodes[node]["quality_band"]
        )
        conflict_rate = conflicts / degree if degree else 0.0
        G.nodes[node]["incident_quality_conflicts"] = conflicts
        G.nodes[node]["delta_nfr"] = conflict_rate
        G.nodes[node]["dnfr"] = conflict_rate
        G.nodes[node]["coherence"] = 1.0 / (1.0 + conflict_rate)

    return G, feature_names


def _binary_auc(labels: Mapping[int, bool], scores: Mapping[int, float]) -> float:
    positives = [scores[node] for node, label in labels.items() if label]
    negatives = [scores[node] for node, label in labels.items() if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    ties = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def _precision_at_review_count(
    labels: Mapping[int, bool],
    scores: Mapping[int, float],
) -> float:
    review_count = sum(labels.values())
    if review_count <= 0:
        return 0.0
    ranked = sorted(scores, key=scores.get, reverse=True)[:review_count]
    return sum(1 for node in ranked if labels[node]) / review_count


def _mean_neighbour_distance(G: nx.Graph, node: int) -> float:
    distances = [
        float(G.edges[node, neighbour].get("chemistry_distance", 0.0))
        for neighbour in G.neighbors(node)
    ]
    return sum(distances) / len(distances) if distances else 0.0


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def run_demo(
    *,
    cache_path: Path | None = None,
    k: int = 10,
    quality_threshold: int = 6,
    review_conflict_threshold: int = 9,
    top_n: int = 10,
    output_dir: Path | None = None,
    download_timeout: float = 30.0,
) -> dict[str, Any]:
    """Run the online wine-quality phase-gate audit."""
    csv_path = download_wine_quality_csv(
        cache_path=cache_path,
        timeout=download_timeout,
    )
    rows = load_wine_rows(csv_path)
    G, feature_names = build_wine_quality_graph(
        rows,
        k=k,
        quality_threshold=quality_threshold,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        report = analyze_phase_gate(G, gate=DEFAULT_PHASE_GATE, top_n=top_n)
        all_hotspots = rank_phase_stress_hotspots(
            G,
            gate=DEFAULT_PHASE_GATE,
            top_n=None,
        )

    hotspot_scores = {int(item.node): float(item.stress_score) for item in all_hotspots}
    distance_scores = {
        int(node): _mean_neighbour_distance(G, int(node)) for node in G.nodes()
    }
    degree_scores = {int(node): float(G.degree[node]) for node in G.nodes()}
    alcohol_scores = {int(node): float(G.nodes[node]["alcohol"]) for node in G.nodes()}
    volatile_scores = {
        int(node): float(G.nodes[node]["volatile_acidity"]) for node in G.nodes()
    }
    constant_scores = {int(node): 1.0 for node in G.nodes()}
    review_labels = {
        int(node): int(G.nodes[node]["incident_quality_conflicts"])
        >= int(review_conflict_threshold)
        for node in G.nodes()
    }

    score_rows = [
        (
            "TNFR phase-stress hotspot",
            hotspot_scores,
            "phase gate + |∇φ| + |Kφ| + incident excess",
        ),
        (
            "Mean chemistry-neighbour distance",
            distance_scores,
            "feature-space distance only",
        ),
        ("Topology degree", degree_scores, "topology only"),
        ("Alcohol value", alcohol_scores, "single chemical feature"),
        ("Volatile acidity", volatile_scores, "single chemical feature"),
        ("Global constant baseline", constant_scores, "no localization signal"),
    ]
    comparison = [
        {
            "score": name,
            "basis": basis,
            "auc": _binary_auc(review_labels, scores),
            "precision_at_review_count": _precision_at_review_count(
                review_labels,
                scores,
            ),
        }
        for name, scores, basis in score_rows
    ]

    prescriptions_by_target = {
        prescription.target: prescription.sequence
        for prescription in report.operator_prescriptions
        if prescription.scope == "node"
    }
    top_hotspots = []
    for hotspot in report.hotspots:
        node = int(hotspot.node)
        top_hotspots.append(
            {
                "sample_id": node,
                "quality": int(G.nodes[node]["quality"]),
                "quality_band": G.nodes[node]["quality_band"],
                "incident_quality_conflicts": int(
                    G.nodes[node]["incident_quality_conflicts"]
                ),
                "stress_score": float(hotspot.stress_score),
                "alcohol": G.nodes[node]["alcohol"],
                "volatile_acidity": G.nodes[node]["volatile_acidity"],
                "sulphates": G.nodes[node]["sulphates"],
                "density": G.nodes[node]["density"],
                "prescription": list(
                    prescriptions_by_target.get(node, ("IL", "OZ", "THOL", "SHA"))
                ),
            }
        )

    qualities = [int(row["quality"]) for row in rows]
    quality_counts = {value: qualities.count(value) for value in sorted(set(qualities))}
    band_counts = {
        "high": sum(1 for value in qualities if value >= quality_threshold),
        "low": sum(1 for value in qualities if value < quality_threshold),
    }

    summary = {
        "dataset": {
            "name": "UCI Red Wine Quality",
            "source_url": WINE_QUALITY_RED_URL,
            "cached_csv": str(csv_path),
            "sector": "food chemistry / quality control",
            "samples": len(rows),
            "features": len(feature_names),
            "quality_counts": quality_counts,
            "quality_band_threshold": quality_threshold,
            "quality_band_counts": band_counts,
        },
        "graph": {
            "construction": f"Standardized chemistry {k}-NN graph",
            "nodes": int(G.number_of_nodes()),
            "edges": int(G.number_of_edges()),
            "k": int(k),
        },
        "phase_gate": {
            "gate": DEFAULT_PHASE_GATE,
            "edge_compliance": report.compliance.compliance_ratio,
            "violations": report.compliance.violation_count,
            "global_order_r": report.baseline_summary["global_order_r"],
            "phase_histogram_entropy": report.baseline_summary[
                "phase_histogram_entropy"
            ],
            "recommendation": report.recommendation,
        },
        "review_definition": {
            "meaning": "sample has at least N chemistry-neighbours in the opposite quality band",
            "threshold": int(review_conflict_threshold),
            "review_node_count": int(sum(review_labels.values())),
        },
        "score_comparison": comparison,
        "top_hotspots": top_hotspots,
        "honest_interpretation": (
            "This example does not claim a better wine-quality predictor. It shows "
            "that TNFR phase-gated local telemetry can prioritize real samples whose "
            "chemical neighbourhood conflicts with their coarse quality band. That "
            "is useful as a review/audit queue for food-chemistry quality control."
        ),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "wine_quality_phase_gate_demo.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        markdown = render_markdown(summary)
        (output_dir / "wine_quality_phase_gate_demo.md").write_text(
            markdown,
            encoding="utf-8",
        )
        (output_dir / "wine_quality_phase_gate_demo.html").write_text(
            render_html(markdown),
            encoding="utf-8",
        )

    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render a Markdown report for the wine-quality audit."""
    comparison_rows = [
        [
            row["score"],
            row["basis"],
            f"{row['auc']:.3f}",
            f"{row['precision_at_review_count']:.3f}",
        ]
        for row in summary["score_comparison"]
    ]
    hotspot_rows = [
        [
            item["sample_id"],
            item["quality"],
            item["quality_band"],
            item["incident_quality_conflicts"],
            f"{item['stress_score']:.3f}",
            f"{item['alcohol']:.2f}",
            f"{item['volatile_acidity']:.3f}",
            f"{item['sulphates']:.3f}",
            " → ".join(item["prescription"]),
        ]
        for item in summary["top_hotspots"]
    ]
    return "\n\n".join(
        [
            "# TNFR UCI Wine Quality Phase-Gate Audit",
            "## Dataset",
            (
                f"Name: {summary['dataset']['name']}  \n"
                f"Sector: {summary['dataset']['sector']}  \n"
                f"Source: {summary['dataset']['source_url']}  \n"
                f"Samples: {summary['dataset']['samples']}  \n"
                f"Features: {summary['dataset']['features']}  \n"
                f"Quality counts: {summary['dataset']['quality_counts']}  \n"
                f"Quality band counts: {summary['dataset']['quality_band_counts']}"
            ),
            "## Graph and phase-gate state",
            (
                f"Graph: {summary['graph']['construction']}  \n"
                f"Nodes: {summary['graph']['nodes']}  \n"
                f"Edges: {summary['graph']['edges']}  \n"
                f"Edge compliance: {summary['phase_gate']['edge_compliance']:.4f}  \n"
                f"Violations: {summary['phase_gate']['violations']}  \n"
                f"Global order R: {summary['phase_gate']['global_order_r']:.4f}  \n"
                f"Recommendation: {summary['phase_gate']['recommendation']}"
            ),
            "## Review task",
            (
                f"Definition: {summary['review_definition']['meaning']}  \n"
                f"Threshold: {summary['review_definition']['threshold']}  \n"
                f"Review nodes: {summary['review_definition']['review_node_count']}"
            ),
            "## Score comparison",
            _markdown_table(
                ["Score", "Basis", "AUC", "Precision@review_count"],
                comparison_rows,
            ),
            "## Top TNFR hotspots",
            _markdown_table(
                [
                    "Sample",
                    "Quality",
                    "Band",
                    "Conflicts",
                    "Stress",
                    "Alcohol",
                    "Volatile acidity",
                    "Sulphates",
                    "TNFR prescription",
                ],
                hotspot_rows,
            ),
            "## Honest interpretation",
            str(summary["honest_interpretation"]),
        ]
    ) + "\n"


def render_html(markdown: str) -> str:
    """Render a standalone HTML report from the Markdown summary."""
    body: list[str] = []
    table_rows: list[str] = []
    in_table = False

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
        table_rows = []
        in_table = False

    for line in markdown.splitlines():
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
<title>TNFR UCI Wine Quality Phase-Gate Audit</title>
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
""".format(body="\n".join(body))


def main() -> None:
    output_dir = ROOT / "results" / "reports"
    summary = run_demo(output_dir=output_dir)

    print("TNFR UCI Wine Quality Phase-Gate Audit")
    print("Downloaded dataset: UCI Red Wine Quality")
    print(f"Samples: {summary['dataset']['samples']}")
    print(f"Graph edges: {summary['graph']['edges']}")
    print(f"Phase-gate compliance: {summary['phase_gate']['edge_compliance']:.4f}")
    print(f"Gate violations: {summary['phase_gate']['violations']}")
    print(
        "Review nodes (>= "
        f"{summary['review_definition']['threshold']} opposite-band neighbours): "
        f"{summary['review_definition']['review_node_count']}"
    )
    print("\nScore comparison:")
    for row in summary["score_comparison"]:
        print(
            f"  {row['score']:<38} "
            f"AUC={row['auc']:.3f} "
            f"P@N={row['precision_at_review_count']:.3f}"
        )
    print("\nTop TNFR hotspots:")
    for item in summary["top_hotspots"][:5]:
        print(
            f"  sample={item['sample_id']:>4} "
            f"quality={item['quality']} "
            f"band={item['quality_band']:<4} "
            f"conflicts={item['incident_quality_conflicts']:<2} "
            f"stress={item['stress_score']:.3f} "
            f"sequence={' -> '.join(item['prescription'])}"
        )
    print(f"\nReports written to: {output_dir}")


if __name__ == "__main__":
    main()
