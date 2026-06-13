"""Phase-gated coupling diagnostics for graph signals.

This module turns the U3 phase-compatibility idea into a reusable validation
surface.  It answers a practical question: given a graph and a phase value per
node, which edges are locally compatible enough to couple?

The functions are read-only telemetry.  They do not mutate EPI, phases, DNFR,
or graph topology.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency guard
    nx = None  # type: ignore[assignment]

from ..physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)

TAU = 2.0 * math.pi
DEFAULT_PHASE_GATE = math.pi / 4.0
DEFAULT_MIN_COMPLIANCE = 0.90


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    data = [float(v) for v in values]
    return sum(data) / len(data) if data else default


def _json_safe_node(node: Any) -> Any:
    if node is None or isinstance(node, (str, int, float, bool)):
        return node
    return repr(node)


def _json_safe_target(target: Any) -> Any:
    if isinstance(target, tuple):
        return [_json_safe_target(item) for item in target]
    if isinstance(target, list):
        return [_json_safe_target(item) for item in target]
    return _json_safe_node(target)


def _require_networkx() -> None:
    if nx is None:
        raise RuntimeError("networkx is required for phase-gate diagnostics")


def wrap_angle(angle: float) -> float:
    """Wrap an angular difference to ``(-pi, pi]``."""
    return (float(angle) + math.pi) % TAU - math.pi


def get_node_phase(
    G: Any,
    node: Any,
    *,
    phase_keys: Sequence[str] = ("phase", "theta"),
    default: float = 0.0,
) -> float:
    """Read a node phase using TNFR-compatible phase aliases."""
    data = G.nodes[node]
    for key in phase_keys:
        if key in data:
            return float(data[key])
    return float(default)


def node_phases(
    G: Any,
    *,
    phase_keys: Sequence[str] = ("phase", "theta"),
) -> list[float]:
    """Return node phases in graph iteration order."""
    return [get_node_phase(G, node, phase_keys=phase_keys) for node in G.nodes()]


def circular_order_parameter(phases: Sequence[float]) -> float:
    """Return the Kuramoto global order parameter ``R`` in ``[0, 1]``."""
    if not phases:
        return 1.0
    cos_mean = _mean(math.cos(float(p)) for p in phases)
    sin_mean = _mean(math.sin(float(p)) for p in phases)
    return math.hypot(cos_mean, sin_mean)


def phase_histogram_entropy(phases: Sequence[float], bins: int = 16) -> float:
    """Return normalized phase-histogram entropy in ``[0, 1]``."""
    if not phases or bins <= 1:
        return 0.0
    counts = [0] * bins
    for phase in phases:
        wrapped = float(phase) % TAU
        index = min(bins - 1, int((wrapped / TAU) * bins))
        counts[index] += 1
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(bins)


@dataclass(frozen=True)
class PhaseGateViolation:
    """One edge whose phase difference exceeds the coupling gate."""

    u: Any
    v: Any
    phase_u: float
    phase_v: float
    phase_difference: float
    excess: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "u": _json_safe_node(self.u),
            "v": _json_safe_node(self.v),
            "phase_u": self.phase_u,
            "phase_v": self.phase_v,
            "phase_difference": self.phase_difference,
            "excess": self.excess,
        }


@dataclass(frozen=True)
class PhaseGateCompliance:
    """Aggregate edge-local phase-gate compliance."""

    gate: float
    edge_count: int
    gated_edges: int
    violation_count: int
    compliance_ratio: float
    min_difference: float
    mean_difference: float
    max_difference: float
    violations: tuple[PhaseGateViolation, ...]

    def passed(self, min_compliance: float = DEFAULT_MIN_COMPLIANCE) -> bool:
        """Return True if enough edges are inside the gate."""
        return self.compliance_ratio >= float(min_compliance)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "edge_count": self.edge_count,
            "gated_edges": self.gated_edges,
            "violation_count": self.violation_count,
            "compliance_ratio": self.compliance_ratio,
            "min_difference": self.min_difference,
            "mean_difference": self.mean_difference,
            "max_difference": self.max_difference,
            "violations": [violation.as_dict() for violation in self.violations],
        }


@dataclass(frozen=True)
class PhaseStressHotspot:
    """Node ranked by local phase stress and incident gate violations."""

    node: Any
    phase: float
    phase_gradient: float
    abs_curvature: float
    incident_violation_count: int
    incident_excess: float
    stress_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": _json_safe_node(self.node),
            "phase": self.phase,
            "phase_gradient": self.phase_gradient,
            "abs_curvature": self.abs_curvature,
            "incident_violation_count": self.incident_violation_count,
            "incident_excess": self.incident_excess,
            "stress_score": self.stress_score,
        }


@dataclass(frozen=True)
class PhaseGateOperatorPrescription:
    """TNFR canonical operator sequence suggested by the phase-gate state.

    Prescriptions are read-only guidance.  They do not apply operators or mutate
    the graph; they map U3 phase-gate telemetry to TNFR's canonical operator
    vocabulary so downstream systems can decide how to act.
    """

    scope: str
    target: Any
    sequence: tuple[str, ...]
    priority: float
    grammar_basis: tuple[str, ...]
    rationale: str
    expected_effect: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "target": _json_safe_target(self.target),
            "sequence": list(self.sequence),
            "priority": self.priority,
            "grammar_basis": list(self.grammar_basis),
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
        }


@dataclass(frozen=True)
class PhaseGateReport:
    """Full phase-gate diagnostic report."""

    compliance: PhaseGateCompliance
    hotspots: tuple[PhaseStressHotspot, ...]
    baseline_summary: Mapping[str, Any]
    min_compliance: float
    recommendation: str
    operator_prescriptions: tuple[PhaseGateOperatorPrescription, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "compliance": self.compliance.as_dict(),
            "hotspots": [hotspot.as_dict() for hotspot in self.hotspots],
            "baseline_summary": dict(self.baseline_summary),
            "min_compliance": self.min_compliance,
            "recommendation": self.recommendation,
            "operator_prescriptions": [
                prescription.as_dict()
                for prescription in self.operator_prescriptions
            ],
        }


def edge_phase_difference(
    G: Any,
    u: Any,
    v: Any,
    *,
    phase_keys: Sequence[str] = ("phase", "theta"),
) -> float:
    """Return absolute wrapped phase difference for one edge or node pair."""
    phase_u = get_node_phase(G, u, phase_keys=phase_keys)
    phase_v = get_node_phase(G, v, phase_keys=phase_keys)
    return abs(wrap_angle(phase_u - phase_v))


def edge_phase_differences(
    G: Any,
    *,
    phase_keys: Sequence[str] = ("phase", "theta"),
) -> list[float]:
    """Return absolute wrapped phase differences over graph edges."""
    return [
        edge_phase_difference(G, u, v, phase_keys=phase_keys)
        for u, v in G.edges()
    ]


def detect_phase_gate_violations(
    G: Any,
    gate: float = DEFAULT_PHASE_GATE,
    *,
    phase_keys: Sequence[str] = ("phase", "theta"),
) -> list[PhaseGateViolation]:
    """Return edges whose wrapped phase difference exceeds ``gate``."""
    violations: list[PhaseGateViolation] = []
    gate_value = float(gate)
    for u, v in G.edges():
        phase_u = get_node_phase(G, u, phase_keys=phase_keys)
        phase_v = get_node_phase(G, v, phase_keys=phase_keys)
        difference = abs(wrap_angle(phase_u - phase_v))
        if difference > gate_value:
            violations.append(
                PhaseGateViolation(
                    u=u,
                    v=v,
                    phase_u=phase_u,
                    phase_v=phase_v,
                    phase_difference=difference,
                    excess=difference - gate_value,
                )
            )
    return violations


def compute_edge_gate_compliance(
    G: Any,
    gate: float = DEFAULT_PHASE_GATE,
    *,
    phase_keys: Sequence[str] = ("phase", "theta"),
) -> PhaseGateCompliance:
    """Compute edge-local phase-gate compliance for a graph state."""
    differences = edge_phase_differences(G, phase_keys=phase_keys)
    edge_count = len(differences)
    if edge_count == 0:
        return PhaseGateCompliance(
            gate=float(gate),
            edge_count=0,
            gated_edges=0,
            violation_count=0,
            compliance_ratio=1.0,
            min_difference=0.0,
            mean_difference=0.0,
            max_difference=0.0,
            violations=(),
        )
    gate_value = float(gate)
    gated_edges = sum(diff <= gate_value for diff in differences)
    violations = tuple(
        detect_phase_gate_violations(G, gate_value, phase_keys=phase_keys)
    )
    return PhaseGateCompliance(
        gate=gate_value,
        edge_count=edge_count,
        gated_edges=gated_edges,
        violation_count=len(violations),
        compliance_ratio=gated_edges / edge_count,
        min_difference=min(differences),
        mean_difference=_mean(differences),
        max_difference=max(differences),
        violations=violations,
    )


def rank_phase_stress_hotspots(
    G: Any,
    gate: float | None = DEFAULT_PHASE_GATE,
    *,
    top_n: int | None = None,
    phase_keys: Sequence[str] = ("phase", "theta"),
) -> list[PhaseStressHotspot]:
    """Rank nodes by local phase stress and incident gate violation excess."""
    grad = compute_phase_gradient(G)
    curvature = compute_phase_curvature(G)
    incident_counts = {node: 0 for node in G.nodes()}
    incident_excess = {node: 0.0 for node in G.nodes()}

    if gate is not None:
        for violation in detect_phase_gate_violations(
            G, float(gate), phase_keys=phase_keys
        ):
            incident_counts[violation.u] = incident_counts.get(violation.u, 0) + 1
            incident_counts[violation.v] = incident_counts.get(violation.v, 0) + 1
            incident_excess[violation.u] = incident_excess.get(violation.u, 0.0) + violation.excess
            incident_excess[violation.v] = incident_excess.get(violation.v, 0.0) + violation.excess

    hotspots: list[PhaseStressHotspot] = []
    for node in G.nodes():
        phase_gradient = float(grad.get(node, 0.0))
        abs_curvature = abs(float(curvature.get(node, 0.0)))
        excess = float(incident_excess.get(node, 0.0))
        count = int(incident_counts.get(node, 0))
        stress_score = phase_gradient + abs_curvature + excess
        hotspots.append(
            PhaseStressHotspot(
                node=node,
                phase=get_node_phase(G, node, phase_keys=phase_keys),
                phase_gradient=phase_gradient,
                abs_curvature=abs_curvature,
                incident_violation_count=count,
                incident_excess=excess,
                stress_score=stress_score,
            )
        )

    hotspots.sort(key=lambda item: item.stress_score, reverse=True)
    if top_n is not None:
        return hotspots[: max(0, int(top_n))]
    return hotspots


def prescribe_phase_gate_operators(
    G: Any,
    gate: float = DEFAULT_PHASE_GATE,
    *,
    min_compliance: float = DEFAULT_MIN_COMPLIANCE,
    top_n: int = 5,
    phase_keys: Sequence[str] = ("phase", "theta"),
) -> list[PhaseGateOperatorPrescription]:
    """Map phase-gate telemetry to TNFR canonical operator guidance.

    This is the TNFR-specific layer of the diagnostic: global baselines can say
    whether a signal looks ordered, and U3 telemetry says which edges are locally
    phase-compatible.  This function translates that structural state into a
    conservative canonical-operator prescription for an already-active graph
    state.  It is read-only guidance and must be followed by a fresh U3 check
    before any guarded ``UM``/``RA`` coupling attempt.
    """
    compliance = compute_edge_gate_compliance(G, gate, phase_keys=phase_keys)
    hotspots = rank_phase_stress_hotspots(
        G,
        gate,
        top_n=top_n,
        phase_keys=phase_keys,
    )
    prescriptions: list[PhaseGateOperatorPrescription] = []

    if compliance.edge_count == 0:
        return [
            PhaseGateOperatorPrescription(
                scope="network",
                target="edgeless_graph",
                sequence=("SHA",),
                priority=0.0,
                grammar_basis=("U1b", "structural metrology"),
                rationale="No graph edges are available for U3 phase-gated coupling.",
                expected_effect="Preserve the current EPI while topology is inspected.",
            )
        ]

    violation_pressure = 1.0 - compliance.compliance_ratio
    if compliance.passed(min_compliance) and compliance.violation_count == 0:
        prescriptions.append(
            PhaseGateOperatorPrescription(
                scope="network",
                target="all_edges",
                sequence=("UM", "RA", "SHA"),
                priority=0.0,
                grammar_basis=("U3", "U1b"),
                rationale="All edges satisfy the U3 phase gate.",
                expected_effect="Allow guarded coupling and resonance propagation, then close observation.",
            )
        )
        return prescriptions

    if compliance.passed(min_compliance):
        prescriptions.append(
            PhaseGateOperatorPrescription(
                scope="network",
                target="mostly_compatible_graph",
                sequence=("IL", "UM", "SHA"),
                priority=violation_pressure,
                grammar_basis=("U3", "U1b", "operator postconditions"),
                rationale=(
                    "Minimum U3 compliance passes, but some local edges still "
                    "exceed the phase gate."
                ),
                expected_effect="Stabilize local phase stress before guarded coupling.",
            )
        )
    else:
        prescriptions.append(
            PhaseGateOperatorPrescription(
                scope="network",
                target="phase_gate_failed",
                sequence=("IL", "OZ", "THOL", "SHA"),
                priority=violation_pressure,
                grammar_basis=("U3", "U2", "U4", "U1b", "U5"),
                rationale=(
                    "U3 compliance is below the configured minimum; new coupling "
                    "should be held while stress is stabilized through controlled "
                    "dissonance and self-organization."
                ),
                expected_effect="Stabilize base state, open controlled reorganization, self-organize, then close before remeasurement.",
            )
        )

    for hotspot in hotspots:
        if hotspot.incident_violation_count <= 0 and hotspot.stress_score <= float(gate):
            continue
        sequence = ("IL", "OZ", "THOL", "SHA") if hotspot.incident_violation_count else ("IL", "SHA")
        basis = ("U2", "U4", "U5", "U1b") if hotspot.incident_violation_count else ("U1b", "structural metrology")
        prescriptions.append(
            PhaseGateOperatorPrescription(
                scope="node",
                target=hotspot.node,
                sequence=sequence,
                priority=hotspot.stress_score,
                grammar_basis=basis,
                rationale=(
                    f"Hotspot has grad_phi={hotspot.phase_gradient:.6f}, "
                    f"abs K_phi={hotspot.abs_curvature:.6f}, and "
                    f"{hotspot.incident_violation_count} incident gate violation(s)."
                ),
                expected_effect="Lower local phase-gradient/curvature stress before any UM/RA retry.",
            )
        )

    prescriptions.sort(key=lambda item: item.priority, reverse=True)
    return prescriptions[: max(1, int(top_n) + 1)]


def _topology_baselines(G: Any) -> dict[str, float]:
    _require_networkx()
    node_count = int(G.number_of_nodes())
    edge_count = int(G.number_of_edges())
    avg_degree = (
        sum(dict(G.degree()).values()) / node_count if node_count else 0.0
    )
    clustering = float(nx.average_clustering(G)) if node_count else 0.0
    if node_count <= 1:
        diameter = 0.0
    elif nx.is_connected(G):
        diameter = float(nx.diameter(G))
    else:
        component_diameters = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if subgraph.number_of_nodes() > 1:
                component_diameters.append(nx.diameter(subgraph))
        diameter = float(max(component_diameters, default=0.0))
    return {
        "topology_node_count": float(node_count),
        "topology_edge_count": float(edge_count),
        "topology_avg_degree": float(avg_degree),
        "topology_clustering": clustering,
        "topology_diameter": diameter,
    }


def compare_against_global_baselines(
    G: Any,
    gate: float = DEFAULT_PHASE_GATE,
    *,
    min_compliance: float = DEFAULT_MIN_COMPLIANCE,
    histogram_bins: int = 16,
    phase_keys: Sequence[str] = ("phase", "theta"),
) -> dict[str, Any]:
    """Return TNFR local telemetry and global/topological baselines.

    The returned dictionary is intentionally flat so it can be used as a row in
    benchmarks, CSV exports, dashboards, or ML feature pipelines.
    """
    compliance = compute_edge_gate_compliance(G, gate, phase_keys=phase_keys)
    grad = compute_phase_gradient(G)
    curvature = compute_phase_curvature(G)
    phi_s = compute_structural_potential(G)
    phases = node_phases(G, phase_keys=phase_keys)
    order_r = circular_order_parameter(phases)
    topology = _topology_baselines(G)

    grad_values = list(grad.values())
    curvature_values = [abs(v) for v in curvature.values()]
    phi_values = [abs(v) for v in phi_s.values()]

    return {
        "label": compliance.passed(min_compliance),
        "edge_gate_compliance": compliance.compliance_ratio,
        "edge_diff_mean": compliance.mean_difference,
        "edge_diff_max": compliance.max_difference,
        "tnfr_mean_phase_gradient": _mean(grad_values),
        "tnfr_max_phase_gradient": max(grad_values, default=0.0),
        "tnfr_mean_abs_curvature": _mean(curvature_values),
        "tnfr_phi_s_abs_mean": _mean(phi_values),
        "global_order_r": order_r,
        "circular_variance": 1.0 - order_r,
        "phase_histogram_entropy": phase_histogram_entropy(
            phases, bins=histogram_bins
        ),
        **topology,
    }


def analyze_phase_gate(
    G: Any,
    gate: float = DEFAULT_PHASE_GATE,
    *,
    min_compliance: float = DEFAULT_MIN_COMPLIANCE,
    top_n: int = 10,
    histogram_bins: int = 16,
    phase_keys: Sequence[str] = ("phase", "theta"),
) -> PhaseGateReport:
    """Build a full phase-gate diagnostic report for a graph state."""
    compliance = compute_edge_gate_compliance(G, gate, phase_keys=phase_keys)
    hotspots = tuple(
        rank_phase_stress_hotspots(
            G, gate, top_n=top_n, phase_keys=phase_keys
        )
    )
    baseline_summary = compare_against_global_baselines(
        G,
        gate,
        min_compliance=min_compliance,
        histogram_bins=histogram_bins,
        phase_keys=phase_keys,
    )
    operator_prescriptions = tuple(
        prescribe_phase_gate_operators(
            G,
            gate,
            min_compliance=min_compliance,
            top_n=top_n,
            phase_keys=phase_keys,
        )
    )

    if compliance.passed(min_compliance) and compliance.violation_count == 0:
        recommendation = "couple"
    elif compliance.passed(min_compliance):
        recommendation = "couple_with_hotspot_monitoring"
    else:
        recommendation = "hold_coupling_and_stabilize_hotspots"

    return PhaseGateReport(
        compliance=compliance,
        hotspots=hotspots,
        baseline_summary=baseline_summary,
        min_compliance=float(min_compliance),
        recommendation=recommendation,
        operator_prescriptions=operator_prescriptions,
    )


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    header = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def render_phase_gate_markdown(
    report: PhaseGateReport,
    *,
    title: str = "TNFR Phase-Gate Coupling Report",
) -> str:
    """Render a phase-gate report as Markdown."""
    comp = report.compliance
    baselines = report.baseline_summary
    hotspot_rows = [
        [
            item.node,
            f"{item.phase_gradient:.6f}",
            f"{item.abs_curvature:.6f}",
            item.incident_violation_count,
            f"{item.incident_excess:.6f}",
            f"{item.stress_score:.6f}",
        ]
        for item in report.hotspots
    ]
    baseline_rows = [
        ["TNFR mean grad_phi", f"{baselines['tnfr_mean_phase_gradient']:.6f}"],
        ["TNFR max grad_phi", f"{baselines['tnfr_max_phase_gradient']:.6f}"],
        ["TNFR mean abs K_phi", f"{baselines['tnfr_mean_abs_curvature']:.6f}"],
        ["TNFR abs Phi_s stress", f"{baselines['tnfr_phi_s_abs_mean']:.6f}"],
        ["Global order R", f"{baselines['global_order_r']:.6f}"],
        ["Phase histogram entropy", f"{baselines['phase_histogram_entropy']:.6f}"],
        ["Topology avg degree", f"{baselines['topology_avg_degree']:.6f}"],
    ]
    prescription_rows = [
        [
            item.scope,
            _json_safe_target(item.target),
            " → ".join(item.sequence),
            f"{item.priority:.6f}",
            ", ".join(item.grammar_basis),
            item.expected_effect,
        ]
        for item in report.operator_prescriptions
    ]
    return "\n\n".join(
        [
            f"# {title}",
            "## Coupling decision",
            (
                f"Recommendation: **{report.recommendation}**  \n"
                f"Gate: {comp.gate:.6f} rad  \n"
                f"Minimum compliance: {report.min_compliance:.2f}  \n"
                f"Compliance: {comp.compliance_ratio:.4f} "
                f"({comp.gated_edges}/{comp.edge_count} edges)  \n"
                f"Violations: {comp.violation_count}  \n"
                f"Mean edge Δφ: {comp.mean_difference:.6f} rad  \n"
                f"Max edge Δφ: {comp.max_difference:.6f} rad"
            ),
            "## Baseline comparison",
            _markdown_table(["Metric", "Value"], baseline_rows),
            "## Phase-stress hotspots",
            _markdown_table(
                [
                    "Node",
                    "grad_phi",
                    "abs K_phi",
                    "Violations",
                    "Incident excess",
                    "Stress score",
                ],
                hotspot_rows,
            ),
            "## TNFR canonical operator prescription",
            _markdown_table(
                [
                    "Scope",
                    "Target",
                    "Sequence",
                    "Priority",
                    "Grammar basis",
                    "Expected effect",
                ],
                prescription_rows,
            ),
            "## Interpretation",
            (
                "Use this report as a local coupling diagnostic.  High global "
                "order does not guarantee edge-local compatibility; TNFR phase "
                "gradient and curvature identify where the graph signal is "
                "locally misaligned with the coupling topology.  The operator "
                "prescription is TNFR-specific read-only guidance: it maps the "
                "observed U3 state to canonical stabilization/coupling sequences."
            ),
        ]
    ) + "\n"


def render_phase_gate_html(
    report: PhaseGateReport,
    *,
    title: str = "TNFR Phase-Gate Coupling Report",
) -> str:
    """Render a phase-gate report as standalone HTML."""
    markdown = render_phase_gate_markdown(report, title=title)
    body: list[str] = []
    table_rows: list[str] = []
    in_table = False

    def flush_table() -> None:
        nonlocal table_rows, in_table
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
<title>{title}</title>
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
""".format(title=html.escape(title), body="\n".join(body))


def export_phase_gate_report(
    G: Any,
    output_path: str | Path,
    *,
    gate: float = DEFAULT_PHASE_GATE,
    min_compliance: float = DEFAULT_MIN_COMPLIANCE,
    fmt: str | None = None,
    top_n: int = 10,
    title: str = "TNFR Phase-Gate Coupling Report",
) -> Path:
    """Analyze a graph and export a phase-gate report.

    Supported formats are ``json``, ``md``/``markdown``, and ``html``.  If
    ``fmt`` is omitted, the output file suffix is used.
    """
    path = Path(output_path)
    report = analyze_phase_gate(
        G,
        gate=gate,
        min_compliance=min_compliance,
        top_n=top_n,
    )
    inferred = path.suffix.lower().lstrip(".")
    format_name = (fmt or inferred or "json").lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if format_name == "json":
        path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    elif format_name in {"md", "markdown"}:
        path.write_text(render_phase_gate_markdown(report, title=title), encoding="utf-8")
    elif format_name == "html":
        path.write_text(render_phase_gate_html(report, title=title), encoding="utf-8")
    else:
        raise ValueError("fmt must be one of: json, md, markdown, html")
    return path


__all__ = [
    "DEFAULT_MIN_COMPLIANCE",
    "DEFAULT_PHASE_GATE",
    "PhaseGateCompliance",
    "PhaseGateOperatorPrescription",
    "PhaseGateReport",
    "PhaseGateViolation",
    "PhaseStressHotspot",
    "analyze_phase_gate",
    "circular_order_parameter",
    "compare_against_global_baselines",
    "compute_edge_gate_compliance",
    "detect_phase_gate_violations",
    "edge_phase_difference",
    "edge_phase_differences",
    "export_phase_gate_report",
    "get_node_phase",
    "node_phases",
    "phase_histogram_entropy",
    "prescribe_phase_gate_operators",
    "rank_phase_stress_hotspots",
    "render_phase_gate_html",
    "render_phase_gate_markdown",
    "wrap_angle",
]
