"""TNFR Structural Interface analysis on real graph data.

This module is the orchestration layer for TNFR Structural Interface Theory.
It builds graphs from records, encodes a node state/label as a TNFR phase, and
reuses the low-level phase-gate primitives in :mod:`tnfr.validation.phase_gate`
to score graph-local *structural interfaces*: regions where neighbouring nodes
are close under the graph relation but differ in phase/state.

Scope and honesty
------------------
The TNFR interface score here is computed from phase telemetry only
(phase gradient, phase curvature, and incident phase-gate violation excess).
When the phase encodes a class/label, this score is, by construction, a
graph-local label-disagreement detector.  It is therefore mathematically
related to the classical k-NN disagreement baseline, which is included
explicitly so the comparison is fair and the relationship is visible.

To make a *non-circular* claim, callers must supply an independent target to
:func:`evaluate_interface_scores` (for example held-out model error, a temporal
transition, or an expert review flag) rather than the local disagreement that
the phase encoding already represents.

All functions are read-only telemetry: they do not mutate EPI, phases, ``dnfr``,
or graph topology, except for graph builders/encoders that create new graphs or
explicitly set requested node attributes.

References
----------
- ``docs/STRUCTURAL_INTERFACE_THEORY_PLAN.md`` — roadmap and acceptance criteria
- ``src/tnfr/validation/phase_gate.py`` — low-level U3/tetrad diagnostics
- AGENTS.md §"Telemetry & Structural Field Tetrad"
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a core dependency
    np = None  # type: ignore[assignment]

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency guard
    nx = None  # type: ignore[assignment]

from ..physics.fields import compute_structural_potential
from .interface_baselines import (
    compute_all_baselines,
    constant_baseline,
    degree_score,
    local_disagreement,
    mean_neighbour_distance,
)
from .phase_gate import (
    DEFAULT_MIN_COMPLIANCE,
    DEFAULT_PHASE_GATE,
    prescribe_phase_gate_operators,
    rank_phase_stress_hotspots,
)

__all__ = [
    "StructuralInterfaceProblem",
    "StructuralInterfaceScore",
    "build_knn_graph",
    "encode_phase_from_binary_state",
    "local_state_disagreement",
    "score_structural_interfaces",
    "interface_score_maps",
    "baseline_score_maps",
    "full_baseline_score_maps",
    "evaluate_interface_scores",
    "render_structural_interface_markdown",
    "render_structural_interface_html",
    "export_structural_interface_report",
]

_DEFAULT_DISTANCE_KEY = "distance"


def _require_networkx() -> None:
    if nx is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("networkx is required for structural-interface analysis")


def _require_numpy() -> None:
    if np is None:  # pragma: no cover - core dependency guard
        raise RuntimeError("numpy is required for structural-interface analysis")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralInterfaceProblem:
    """A structural-interface problem definition over a graph.

    Parameters
    ----------
    graph:
        Graph whose nodes carry a phase and (optionally) a state/label.
    state_key:
        Node attribute holding the discrete state/label (e.g. class band).
    phase_key:
        Node attribute holding the TNFR phase.  Defaults to ``"phase"``.
    domain:
        Free-text sector tag (e.g. ``"biomedical"``) for reporting only.
    distance_key:
        Edge attribute holding the feature-space distance, when available.
    metadata:
        Arbitrary report metadata.
    """

    graph: Any
    state_key: str
    phase_key: str = "phase"
    domain: str = "generic"
    distance_key: str = _DEFAULT_DISTANCE_KEY
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuralInterfaceScore:
    """Per-node structural-interface telemetry and canonical prescription."""

    node: Any
    tnfr_stress: float
    phase_gradient: float
    abs_curvature: float
    structural_potential: float
    incident_violation_count: int
    incident_gate_pressure: float
    prescription: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        node = self.node
        if not isinstance(node, (str, int, float, bool)) and node is not None:
            node = repr(node)
        return {
            "node": node,
            "tnfr_stress": self.tnfr_stress,
            "phase_gradient": self.phase_gradient,
            "abs_curvature": self.abs_curvature,
            "structural_potential": self.structural_potential,
            "incident_violation_count": self.incident_violation_count,
            "incident_gate_pressure": self.incident_gate_pressure,
            "prescription": list(self.prescription),
        }


# ---------------------------------------------------------------------------
# Graph construction and phase encoding
# ---------------------------------------------------------------------------


def _standardize(matrix: Sequence[Sequence[float]]) -> "np.ndarray":
    _require_numpy()
    arr = np.asarray(matrix, dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (arr - mean) / std


def _knn_indices_distances(
    scaled: "np.ndarray", k: int
) -> tuple["np.ndarray", "np.ndarray"]:
    """Return (distances, indices) for the ``k+1`` nearest neighbours.

    Uses scikit-learn when available; otherwise falls back to a memory-light
    numpy brute-force search (O(n^2) time, O(n) memory per row).
    """
    n_neighbors = int(k) + 1
    try:
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=n_neighbors)
        model.fit(scaled)
        return model.kneighbors(scaled)
    except ImportError:
        _require_numpy()
        count = scaled.shape[0]
        n_neighbors = min(n_neighbors, count)
        distances = np.empty((count, n_neighbors), dtype=float)
        indices = np.empty((count, n_neighbors), dtype=int)
        for i in range(count):
            row = np.sqrt(((scaled - scaled[i]) ** 2).sum(axis=1))
            order = np.argsort(row, kind="stable")[:n_neighbors]
            indices[i] = order
            distances[i] = row[order]
        return distances, indices


def build_knn_graph(
    records: Sequence[Mapping[str, Any]],
    feature_keys: Sequence[str],
    *,
    k: int = 10,
    node_attributes: Sequence[str] | None = None,
    distance_key: str = _DEFAULT_DISTANCE_KEY,
    standardize: bool = True,
) -> Any:
    """Build a standardized k-nearest-neighbour graph from records.

    Parameters
    ----------
    records:
        Sequence of mappings (one per sample).
    feature_keys:
        Numeric feature names used to build the feature space.
    k:
        Number of neighbours per node (excluding self).
    node_attributes:
        Optional extra attributes copied verbatim onto each node.
    distance_key:
        Edge attribute name used to store the feature-space distance.
    standardize:
        Whether to z-score features before computing distances.

    Returns
    -------
    networkx.Graph
        Undirected graph; nodes are integer record indices.
    """
    _require_networkx()
    _require_numpy()
    if not records:
        return nx.Graph()
    if k < 1:
        raise ValueError("k must be >= 1")

    matrix = [[float(record[key]) for key in feature_keys] for record in records]
    scaled = _standardize(matrix) if standardize else np.asarray(matrix, dtype=float)
    distances, indices = _knn_indices_distances(scaled, k)

    G = nx.Graph()
    extra = tuple(node_attributes or ())
    for node, record in enumerate(records):
        attrs: dict[str, Any] = {}
        for key in extra:
            if key in record:
                attrs[key] = record[key]
        G.add_node(node, **attrs)

    for node in range(len(records)):
        for distance, neighbour in zip(distances[node][1:], indices[node][1:]):
            G.add_edge(node, int(neighbour), **{distance_key: float(distance)})

    return G


def encode_phase_from_binary_state(
    G: Any,
    state_key: str,
    *,
    positive_value: Any,
    phase_key: str = "phase",
    positive_phase: float = 0.0,
    negative_phase: float = math.pi,
    also_set_theta: bool = True,
) -> Any:
    """Encode a binary node state as a TNFR phase in-place.

    Nodes whose ``state_key`` equals ``positive_value`` receive
    ``positive_phase``; all others receive ``negative_phase``.  The graph is
    returned for chaining.
    """
    _require_networkx()
    for node in G.nodes():
        value = G.nodes[node].get(state_key)
        phase = positive_phase if value == positive_value else negative_phase
        G.nodes[node][phase_key] = float(phase)
        if also_set_theta:
            G.nodes[node]["theta"] = float(phase)
    return G


# ---------------------------------------------------------------------------
# Baselines and TNFR scoring
# ---------------------------------------------------------------------------


def local_state_disagreement(G: Any, state_key: str) -> dict[Any, int]:
    """Return, per node, the count of neighbours with a different state.

    This is the classical graph-local label-disagreement signal.  It doubles as
    a baseline score and as a candidate (circular) review target; it must NOT be
    used as the only ground truth for an external TNFR superiority claim.

    Thin wrapper over :func:`tnfr.validation.interface_baselines.local_disagreement`
    (single source of truth), returning integer counts for backward compatibility.
    """
    _require_networkx()
    return {
        node: int(value)
        for node, value in local_disagreement(G, state_key=state_key).items()
    }


def score_structural_interfaces(
    problem: StructuralInterfaceProblem | Any,
    *,
    gate: float = DEFAULT_PHASE_GATE,
    state_key: str | None = None,
    phase_key: str = "phase",
    distance_key: str = _DEFAULT_DISTANCE_KEY,
    min_compliance: float = DEFAULT_MIN_COMPLIANCE,
) -> list[StructuralInterfaceScore]:
    """Compute per-node TNFR structural-interface scores.

    Accepts either a :class:`StructuralInterfaceProblem` or a raw graph.  The
    TNFR stress reuses :func:`rank_phase_stress_hotspots`; structural potential
    is read from existing ``dnfr``/``delta_nfr`` attributes (0 if absent) and is
    reported as telemetry, not folded into the ranking score.
    """
    if isinstance(problem, StructuralInterfaceProblem):
        G = problem.graph
        phase_key = problem.phase_key
    else:
        G = problem
    _require_networkx()

    phase_keys = (phase_key, "theta")
    hotspots = rank_phase_stress_hotspots(G, gate, top_n=None, phase_keys=phase_keys)
    potential = compute_structural_potential(G)
    prescriptions = prescribe_phase_gate_operators(
        G,
        gate,
        min_compliance=min_compliance,
        top_n=max(1, G.number_of_nodes()),
        phase_keys=phase_keys,
    )
    node_prescription = {
        prescription.target: prescription.sequence
        for prescription in prescriptions
        if prescription.scope == "node"
    }
    default_prescription = next(
        (
            prescription.sequence
            for prescription in prescriptions
            if prescription.scope == "network"
        ),
        ("IL", "SHA"),
    )

    scores: list[StructuralInterfaceScore] = []
    for hotspot in hotspots:
        node = hotspot.node
        scores.append(
            StructuralInterfaceScore(
                node=node,
                tnfr_stress=hotspot.stress_score,
                phase_gradient=hotspot.phase_gradient,
                abs_curvature=hotspot.abs_curvature,
                structural_potential=float(potential.get(node, 0.0)),
                incident_violation_count=hotspot.incident_violation_count,
                incident_gate_pressure=hotspot.incident_excess,
                prescription=tuple(node_prescription.get(node, default_prescription)),
            )
        )
    return scores


def interface_score_maps(
    scores: Sequence[StructuralInterfaceScore],
) -> dict[Any, float]:
    """Return a node -> TNFR stress map from structural-interface scores."""
    return {score.node: float(score.tnfr_stress) for score in scores}


def baseline_score_maps(
    G: Any,
    *,
    state_key: str,
    distance_key: str = _DEFAULT_DISTANCE_KEY,
) -> dict[str, dict[Any, float]]:
    """Return a compact classical baseline set for quick comparison.

    Includes the closest classical analogue (local state disagreement), plus
    feature-distance, topology, and a constant baseline.  For the full fair-
    comparison suite (graph total variation, entropy, label-propagation
    residual, graph cut, random control) use :func:`full_baseline_score_maps`.
    """
    _require_networkx()
    return {
        "local_state_disagreement": local_disagreement(G, state_key=state_key),
        "mean_neighbour_distance": mean_neighbour_distance(
            G, distance_key=distance_key
        ),
        "topology_degree": degree_score(G),
        "constant_baseline": constant_baseline(G),
    }


def full_baseline_score_maps(
    G: Any,
    *,
    state_key: str,
    distance_key: str = _DEFAULT_DISTANCE_KEY,
    phase_key: str = "phase",
    feature_key: str | None = None,
    seed: int = 0,
) -> dict[str, dict[Any, float]]:
    """Return the full classical baseline suite for fair benchmarking.

    Delegates to
    :func:`tnfr.validation.interface_baselines.compute_all_baselines`.
    """
    _require_networkx()
    return compute_all_baselines(
        G,
        state_key=state_key,
        distance_key=distance_key,
        phase_key=phase_key,
        feature_key=feature_key,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _binary_auc(labels: Mapping[Any, bool], scores: Mapping[Any, float]) -> float:
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
    labels: Mapping[Any, bool], scores: Mapping[Any, float]
) -> float:
    review_count = sum(1 for value in labels.values() if value)
    if review_count <= 0:
        return 0.0
    ranked = sorted(scores, key=lambda node: scores[node], reverse=True)
    ranked = ranked[:review_count]
    return sum(1 for node in ranked if labels.get(node)) / review_count


def evaluate_interface_scores(
    labels: Mapping[Any, bool],
    score_maps: Mapping[str, Mapping[Any, float]],
) -> dict[str, Any]:
    """Evaluate ranking scores against a binary review/interface target.

    The ``labels`` mapping defines the ground-truth interface/review nodes.
    Each entry of ``score_maps`` is ranked with ROC-AUC and precision@review.
    """
    review_count = sum(1 for value in labels.values() if value)
    rows = []
    for name, scores in score_maps.items():
        rows.append(
            {
                "score": name,
                "auc": _binary_auc(labels, scores),
                "precision_at_review_count": _precision_at_review_count(labels, scores),
            }
        )
    return {
        "review_node_count": int(review_count),
        "total_nodes": int(len(labels)),
        "score_comparison": rows,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "| " + " | ".join(str(cell) for cell in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def render_structural_interface_markdown(result: Mapping[str, Any]) -> str:
    """Render a structural-interface benchmark result as Markdown."""
    dataset = result.get("dataset", {})
    graph = result.get("graph", {})
    task = result.get("task", {})
    evaluation = result.get("evaluation", {})
    comparison = evaluation.get("score_comparison", [])
    hotspots = result.get("hotspots", [])

    comparison_rows = [
        [
            row.get("score"),
            f"{row.get('auc', 0.0):.3f}",
            f"{row.get('precision_at_review_count', 0.0):.3f}",
        ]
        for row in comparison
    ]
    hotspot_rows = [
        [
            item.get("node"),
            f"{item.get('tnfr_stress', 0.0):.3f}",
            f"{item.get('phase_gradient', 0.0):.3f}",
            f"{item.get('abs_curvature', 0.0):.3f}",
            item.get("incident_violation_count", 0),
            " → ".join(item.get("prescription", [])),
        ]
        for item in hotspots
    ]
    return (
        "\n\n".join(
            [
                "# TNFR Structural Interface Audit",
                "## Dataset",
                (
                    f"Name: {dataset.get('name', 'unknown')}  \n"
                    f"Sector: {dataset.get('sector', 'generic')}  \n"
                    f"Samples: {dataset.get('samples', graph.get('nodes', 0))}"
                ),
                "## Graph",
                (
                    f"Construction: {graph.get('construction', 'k-NN graph')}  \n"
                    f"Nodes: {graph.get('nodes', 0)}  \n"
                    f"Edges: {graph.get('edges', 0)}"
                ),
                "## Review target",
                (
                    f"Definition: {task.get('target_definition', 'unspecified')}  \n"
                    f"Circular with phase encoding: {task.get('is_circular_target', 'unknown')}  \n"
                    f"Review nodes: {evaluation.get('review_node_count', 0)}"
                ),
                "## Score comparison",
                _markdown_table(
                    ["Score", "AUC", "Precision@review_count"], comparison_rows
                ),
                "## Top TNFR hotspots",
                _markdown_table(
                    [
                        "Node",
                        "Stress",
                        "grad φ",
                        "|Kφ|",
                        "Violations",
                        "TNFR prescription",
                    ],
                    hotspot_rows,
                ),
                "## Honest interpretation",
                str(
                    result.get(
                        "honest_interpretation",
                        "TNFR phase-stress localizes graph-local interfaces. When the "
                        "review target equals local disagreement, high AUC is a "
                        "localization sanity check, not external superiority.",
                    )
                ),
            ]
        )
        + "\n"
    )


def render_structural_interface_html(markdown: str) -> str:
    """Render a standalone HTML report from a Markdown summary."""
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
<html lang="en">
<head>
<meta charset="utf-8">
<title>TNFR Structural Interface Audit</title>
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


def export_structural_interface_report(
    result: Mapping[str, Any],
    output_dir: Path,
    *,
    stem: str = "structural_interface_report",
) -> dict[str, Path]:
    """Write JSON, Markdown, and HTML reports for a benchmark result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    html_path = output_dir / f"{stem}.html"

    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown = render_structural_interface_markdown(result)
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(render_structural_interface_html(markdown), encoding="utf-8")

    return {"json": json_path, "markdown": md_path, "html": html_path}
