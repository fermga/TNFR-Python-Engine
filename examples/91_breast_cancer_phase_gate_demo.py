#!/usr/bin/env python3
"""Example 91 — Biomedical Phase-Gate Audit on WDBC data.

This example uses the real Wisconsin Diagnostic Breast Cancer dataset bundled
with scikit-learn.  It builds a k-nearest-neighbour graph from tumor morphology
measurements, encodes the diagnosis as a binary phase signal, and asks a very
concrete non-financial question:

    Which cases sit in a local morphology neighbourhood that disagrees with
    their diagnostic phase?

Scope
-----
This is a structural audit / model-review example, not clinical decision
support.  It demonstrates that TNFR phase-gate telemetry localizes graph-local
diagnostic boundary cases that global class balance and topology-only baselines
cannot localize.

Run:
    python examples/91_breast_cancer_phase_gate_demo.py
"""
from __future__ import annotations

import html
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping

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
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from tnfr.validation.phase_gate import (  # noqa: E402
    DEFAULT_PHASE_GATE,
    analyze_phase_gate,
    rank_phase_stress_hotspots,
)


def build_wdbc_knn_graph(k: int = 8) -> tuple[nx.Graph, Any]:
    """Build a morphology kNN graph for the real WDBC dataset."""
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    y = data.target

    neighbours = NearestNeighbors(n_neighbors=int(k) + 1)
    neighbours.fit(X)
    distances, indices = neighbours.kneighbors(X)

    G = nx.Graph()
    for node, target in enumerate(y):
        diagnosis = str(data.target_names[int(target)])
        # scikit-learn encoding: 0=malignant, 1=benign.
        phase = 0.0 if diagnosis == "benign" else math.pi
        G.add_node(
            node,
            phase=phase,
            theta=phase,
            EPI=1.0,
            diagnosis=diagnosis,
            mean_radius=float(data.data[node][0]),
            mean_texture=float(data.data[node][1]),
            mean_concavity=float(data.data[node][6]),
            glyph_history=[],
        )

    for node in range(len(y)):
        for distance, neighbour in zip(distances[node][1:], indices[node][1:]):
            G.add_edge(node, int(neighbour), morphology_distance=float(distance))

    # Attach local structural pressure from diagnostic disagreement with
    # morphology neighbours.  This keeps Φ_s tied to graph-local conflict.
    for node in G.nodes():
        degree = G.degree[node]
        conflicts = sum(
            1
            for neighbour in G.neighbors(node)
            if G.nodes[neighbour]["diagnosis"] != G.nodes[node]["diagnosis"]
        )
        conflict_rate = conflicts / degree if degree else 0.0
        G.nodes[node]["incident_diagnostic_conflicts"] = conflicts
        G.nodes[node]["delta_nfr"] = conflict_rate
        G.nodes[node]["dnfr"] = conflict_rate
        G.nodes[node]["coherence"] = 1.0 / (1.0 + conflict_rate)

    return G, data


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
        float(G.edges[node, neighbour].get("morphology_distance", 0.0))
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
    k: int = 8,
    review_conflict_threshold: int = 3,
    top_n: int = 10,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the WDBC phase-gate audit and optionally export reports."""
    G, data = build_wdbc_knn_graph(k=k)

    # Structural potential may emit harmless divide warnings on degenerate
    # graph-distance internals; suppress them for report readability.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        report = analyze_phase_gate(G, gate=DEFAULT_PHASE_GATE, top_n=top_n)
        all_hotspots = rank_phase_stress_hotspots(
            G,
            gate=DEFAULT_PHASE_GATE,
            top_n=None,
        )

    hotspot_scores = {int(item.node): float(item.stress_score) for item in all_hotspots}
    degree_scores = {int(node): float(G.degree[node]) for node in G.nodes()}
    distance_scores = {
        int(node): _mean_neighbour_distance(G, int(node)) for node in G.nodes()
    }
    constant_scores = {int(node): 1.0 for node in G.nodes()}
    review_labels = {
        int(node): int(G.nodes[node]["incident_diagnostic_conflicts"])
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
            "Mean morphology-neighbour distance",
            distance_scores,
            "feature-space distance only",
        ),
        ("Topology degree", degree_scores, "topology only"),
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
                "diagnosis": G.nodes[node]["diagnosis"],
                "incident_diagnostic_conflicts": int(
                    G.nodes[node]["incident_diagnostic_conflicts"]
                ),
                "stress_score": float(hotspot.stress_score),
                "mean_radius": G.nodes[node]["mean_radius"],
                "mean_texture": G.nodes[node]["mean_texture"],
                "mean_concavity": G.nodes[node]["mean_concavity"],
                "prescription": list(
                    prescriptions_by_target.get(node, ("IL", "OZ", "THOL", "SHA"))
                ),
            }
        )

    summary = {
        "dataset": {
            "name": "Wisconsin Diagnostic Breast Cancer (scikit-learn bundled)",
            "sector": "biomedical / diagnostic morphology",
            "samples": int(G.number_of_nodes()),
            "features": int(len(data.feature_names)),
            "target_counts": {
                str(data.target_names[value]): int((data.target == value).sum())
                for value in sorted(set(data.target))
            },
        },
        "graph": {
            "construction": f"Standardized morphology {k}-NN graph",
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
            "meaning": "case has at least N morphology-neighbours with opposite diagnosis",
            "threshold": int(review_conflict_threshold),
            "review_node_count": int(sum(review_labels.values())),
        },
        "score_comparison": comparison,
        "top_hotspots": top_hotspots,
        "honest_interpretation": (
            "This example does not claim clinical validity. It shows that TNFR "
            "phase-gated local telemetry can prioritize real biomedical samples "
            "whose morphology-neighbourhood conflicts with their diagnostic phase. "
            "Global class balance and topology-only baselines do not provide that "
            "node-level localization."
        ),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "wdbc_phase_gate_demo.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        markdown = render_markdown(summary)
        (output_dir / "wdbc_phase_gate_demo.md").write_text(
            markdown,
            encoding="utf-8",
        )
        (output_dir / "wdbc_phase_gate_demo.html").write_text(
            render_html(markdown),
            encoding="utf-8",
        )

    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the WDBC audit summary as Markdown."""
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
            item["diagnosis"],
            item["incident_diagnostic_conflicts"],
            f"{item['stress_score']:.3f}",
            f"{item['mean_radius']:.2f}",
            f"{item['mean_texture']:.2f}",
            f"{item['mean_concavity']:.5f}",
            " → ".join(item["prescription"]),
        ]
        for item in summary["top_hotspots"]
    ]
    return "\n\n".join(
        [
            "# TNFR WDBC Biomedical Phase-Gate Audit",
            "## Dataset",
            (
                f"Name: {summary['dataset']['name']}  \n"
                f"Sector: {summary['dataset']['sector']}  \n"
                f"Samples: {summary['dataset']['samples']}  \n"
                f"Features: {summary['dataset']['features']}  \n"
                f"Target counts: {summary['dataset']['target_counts']}"
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
            "## Node-level review task",
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
                    "Diagnosis",
                    "Conflicts",
                    "Stress",
                    "Mean radius",
                    "Mean texture",
                    "Mean concavity",
                    "TNFR prescription",
                ],
                hotspot_rows,
            ),
            "## Honest interpretation",
            str(summary["honest_interpretation"]),
        ]
    ) + "\n"


def render_html(markdown: str) -> str:
    """Render a small standalone HTML report from the Markdown summary."""
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
<title>TNFR WDBC Biomedical Phase-Gate Audit</title>
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

    print("TNFR WDBC Biomedical Phase-Gate Audit")
    print("Real dataset: Wisconsin Diagnostic Breast Cancer")
    print(f"Samples: {summary['dataset']['samples']}")
    print(f"Graph edges: {summary['graph']['edges']}")
    print(f"Phase-gate compliance: {summary['phase_gate']['edge_compliance']:.4f}")
    print(f"Gate violations: {summary['phase_gate']['violations']}")
    print(
        "Review nodes (>= "
        f"{summary['review_definition']['threshold']} opposite-diagnosis neighbours): "
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
            f"  sample={item['sample_id']:>3} "
            f"diagnosis={item['diagnosis']:<9} "
            f"conflicts={item['incident_diagnostic_conflicts']:<2} "
            f"stress={item['stress_score']:.3f} "
            f"sequence={' -> '.join(item['prescription'])}"
        )
    print(f"\nReports written to: {output_dir}")


if __name__ == "__main__":
    main()
