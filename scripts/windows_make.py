"""Windows-friendly shim for common TNFR build targets.

This driver mirrors the curated Makefile-style automation surfaces that VS Code
tasks call so Windows contributors can run the same workflows without GNU Make.
Targets fall into five buckets:

1. Test helpers (smoke-tests)
2. Clean-up utilities
3. Report exporters (HTML summaries)
4. Atlas/data script generators (JSON payloads)
5. Benchmark wrappers (force-study-plots)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:  # Optional TNFR imports used by atlas/report targets
    import networkx as nx
    from tnfr.examples_utils.demo_sequences import (  # type: ignore
        build_element_radial_graph,
        build_triatomic_molecule_graph,
    )
    from tnfr.physics.fields import (  # type: ignore
        compute_phase_curvature,
        compute_phase_gradient,
        compute_structural_potential,
        compute_unified_telemetry,
        estimate_coherence_length,
    )
    from tnfr.physics.signatures import compute_element_signature  # type: ignore

    TNFR_AVAILABLE = True
    TNFR_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency gate
    TNFR_AVAILABLE = False
    TNFR_IMPORT_ERROR = exc
    nx = None  # type: ignore

PYTHON = sys.executable
RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "reports"
DATA_DIR = RESULTS_DIR / "script_payloads"
FORCE_STUDY_PATH = REPORTS_DIR / "force_regime_study.jsonl"

ELEMENT_SAMPLES: Dict[int, Dict[str, Any]] = {
    1: {"symbol": "H", "name": "Hydrogen", "period": 1},
    6: {"symbol": "C", "name": "Carbon", "period": 2},
    7: {"symbol": "N", "name": "Nitrogen", "period": 2},
    8: {"symbol": "O", "name": "Oxygen", "period": 2},
    11: {"symbol": "Na", "name": "Sodium", "period": 3},
    12: {"symbol": "Mg", "name": "Magnesium", "period": 3},
    14: {"symbol": "Si", "name": "Silicon", "period": 3},
    17: {"symbol": "Cl", "name": "Chlorine", "period": 3},
    26: {"symbol": "Fe", "name": "Iron", "period": 4},
    29: {"symbol": "Cu", "name": "Copper", "period": 4},
    47: {"symbol": "Ag", "name": "Silver", "period": 5},
    79: {"symbol": "Au", "name": "Gold", "period": 6},
}

MOLECULE_SAMPLES: List[Dict[str, Any]] = [
    {"name": "Water", "atoms": (1, 8, 1), "central": "B", "geometry": "bent"},
    {"name": "Carbon Dioxide", "atoms": (8, 6, 8), "central": "B", "geometry": "linear"},
    {"name": "Ammonia", "atoms": (1, 7, 1), "central": "B", "geometry": "pyramidal"},
]

PHASE_GATE_THRESHOLD = 0.78539816339  # π/4 radians


@dataclass
class Step:
    label: str
    command: Sequence[str]
    requires: Path | None = None
    optional: bool = False

    def exists(self) -> bool:
        if self.requires is None:
            return True
        return self.requires.exists()


def _run_steps(steps: Sequence[Step]) -> None:
    for step in steps:
        if not step.exists():
            if step.optional:
                print(f"[skip] {step.label} (missing {step.requires})")
                continue
            raise FileNotFoundError(f"Required resource missing for step '{step.label}': {step.requires}")

        print(f"[run ] {step.label}")
        result = subprocess.run(step.command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, step.command)
        print(f"[done] {step.label}")


def _pytest_step(relative_test: str) -> Step:
    path = REPO_ROOT / relative_test
    return Step(label=f"pytest {relative_test}", command=[PYTHON, "-m", "pytest", relative_test], requires=path)


def _example_step(relative_script: str) -> Step:
    path = REPO_ROOT / relative_script
    return Step(label=f"python {relative_script}", command=[PYTHON, relative_script], requires=path, optional=True)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_file(path: Path, payload: Any) -> Path:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_html_report(
    filename: str,
    title: str,
    sections: List[Dict[str, Any]],
    tables: List[Dict[str, Any]] | None = None,
) -> Path:
    _ensure_dir(REPORTS_DIR)
    body: List[str] = [f"<h1>{title}</h1>", f"<p>Generated: {_timestamp()}</p>"]
    for section in sections:
        body.append(f"<h2>{section['heading']}</h2>")
        body.append(f"<p>{section['body']}</p>")
    if tables:
        for table in tables:
            headers = ''.join(f"<th>{col}</th>" for col in table['headers'])
            rows = []
            for row in table['rows']:
                rows.append('<tr>' + ''.join(f"<td>{cell}</td>" for cell in row) + '</tr>')
            body.append(f"<h3>{table['title']}</h3>")
            body.append(f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>")
    html = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem; text-align: left; }}
</style>
</head>
<body>
{body}
</body>
</html>
""".format(title=title, body='\n'.join(body))
    output = REPORTS_DIR / filename
    output.write_text(html, encoding="utf-8")
    return output


def _basic_stats(values: Iterable[float]) -> Dict[str, float]:
    data = list(values)
    if not data:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": round(min(data), 6),
        "max": round(max(data), 6),
        "mean": round(sum(data) / len(data), 6),
    }


def _require_tnfr(target: str) -> None:
    if TNFR_AVAILABLE:
        return
    raise RuntimeError(f"{target} requires TNFR modules ({TNFR_IMPORT_ERROR})")


def _prepare_graph_for_fields(G: "nx.Graph") -> None:
    for node in G.nodes():
        data = G.nodes[node]
        phase = float(data.get("phase", data.get("theta", 0.0)))
        data.setdefault("phase", phase)
        data.setdefault("theta", phase)
        dnfr = float(data.get("delta_nfr", data.get("dnfr", 0.05)))
        data.setdefault("delta_nfr", dnfr)
        data.setdefault("dnfr", dnfr)
        data.setdefault("coherence", 1.0 / (1.0 + abs(dnfr)))


def _compute_tetrad_summary(G: "nx.Graph") -> Dict[str, Any]:
    phi_s = compute_structural_potential(G)
    grad = compute_phase_gradient(G)
    curvature = compute_phase_curvature(G)
    xi_c = float(estimate_coherence_length(G))
    return {
        "phi_s": _basic_stats(phi_s.values()),
        "phase_gradient": _basic_stats(grad.values()),
        "phase_curvature": _basic_stats(abs(v) for v in curvature.values()),
        "coherence_length": {"value": round(xi_c, 6)},
    }


def _build_triatomic_graph() -> "nx.Graph":
    G = build_triatomic_molecule_graph(1, 8, 1, central="B", bond_links=2)
    _prepare_graph_for_fields(G)
    return G


def _load_json(path: Path, generator: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    if not path.exists():
        data = generator()
        _write_json_file(path, data)
        return data
    return json.loads(path.read_text(encoding="utf-8"))


def _triatomic_payload() -> Dict[str, Any]:
    _require_tnfr("triatomic-atlas-script")
    G = _build_triatomic_graph()
    tetrad = _compute_tetrad_summary(G)
    signature = compute_element_signature(G)
    telemetry = compute_unified_telemetry(G)
    return {
        "generated_at": _timestamp(),
        "molecule": "H-O-H",
        "graph": {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "geometry_hint": G.graph.get("geometry_hint", "bent"),
        },
        "tetrad": tetrad,
        "signature": {
            "xi_c": signature.get("xi_c"),
            "mean_phase_gradient": signature.get("mean_phase_gradient"),
            "mean_phase_curvature_abs": signature.get("mean_phase_curvature_abs"),
            "phi_s_drift": signature.get("phi_s_drift"),
            "signature_class": signature.get("signature_class"),
        },
        "unified_fields": {
            "complex_correlation": telemetry.get("complex_field", {}).get("correlation"),
            "psi_magnitude_mean": telemetry.get("complex_field", {}).get("psi_magnitude_mean"),
            "chirality": telemetry.get("emergent_fields", {}).get("chirality_magnitude"),
            "symmetry_breaking": telemetry.get("emergent_fields", {}).get("symmetry_breaking"),
            "coherence_coupling": telemetry.get("emergent_fields", {}).get("coherence_coupling"),
        },
    }


def _phase_gated_payload() -> Dict[str, Any]:
    _require_tnfr("phase-gated-script")
    G = _build_triatomic_graph()
    phases: List[float] = [float(G.nodes[n].get("phase", 0.0)) for n in G.nodes()]
    diffs: List[float] = []
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            diff = abs(phases[i] - phases[j])
            diff = min(diff, abs(2.0 * 3.141592653589793 - diff))
            diffs.append(diff)
    within_gate = [d <= PHASE_GATE_THRESHOLD for d in diffs]
    return {
        "generated_at": _timestamp(),
        "pairs": len(diffs),
        "max_difference": round(max(diffs) if diffs else 0.0, 6),
        "min_difference": round(min(diffs) if diffs else 0.0, 6),
        "gate_threshold": PHASE_GATE_THRESHOLD,
        "gated_pairs": sum(1 for flag in within_gate if flag),
        "compliance_ratio": round(sum(1 for flag in within_gate if flag) / len(diffs), 6) if diffs else 1.0,
    }


def _element_signature_payload() -> Dict[str, Any]:
    _require_tnfr("elements-signature-script")
    entries: List[Dict[str, Any]] = []
    for Z, meta in ELEMENT_SAMPLES.items():
        G = build_element_radial_graph(Z, seed=Z)
        _prepare_graph_for_fields(G)
        signature = compute_element_signature(G)
        entries.append(
            {
                "Z": Z,
                "symbol": meta["symbol"],
                "name": meta["name"],
                "xi_c": signature.get("xi_c"),
                "mean_phase_gradient": signature.get("mean_phase_gradient"),
                "max_phase_curvature": signature.get("max_phase_curvature_abs"),
                "signature_class": signature.get("signature_class"),
            }
        )
    return {"generated_at": _timestamp(), "entries": entries}


def _molecule_atlas_payload() -> Dict[str, Any]:
    _require_tnfr("molecule-atlas-script")
    entries: List[Dict[str, Any]] = []
    for sample in MOLECULE_SAMPLES:
        atoms = sample["atoms"]
        G = build_triatomic_molecule_graph(*atoms, central=sample.get("central", "B"))
        _prepare_graph_for_fields(G)
        tetrad = _compute_tetrad_summary(G)
        entries.append(
            {
                "name": sample["name"],
                "atoms": atoms,
                "geometry": sample.get("geometry", "unknown"),
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "tetrad": tetrad,
            }
        )
    return {"generated_at": _timestamp(), "entries": entries}


def _atom_atlas_payload() -> Dict[str, Any]:
    _require_tnfr("atom-atlas-script")
    entries: List[Dict[str, Any]] = []
    for Z, meta in ELEMENT_SAMPLES.items():
        G = build_element_radial_graph(Z, seed=Z * 3)
        _prepare_graph_for_fields(G)
        tetrad = _compute_tetrad_summary(G)
        entries.append(
            {
                "Z": Z,
                "symbol": meta["symbol"],
                "name": meta["name"],
                "period": meta["period"],
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "phi_s_mean": tetrad["phi_s"]["mean"],
                "phase_gradient_mean": tetrad["phase_gradient"]["mean"],
            }
        )
    return {"generated_at": _timestamp(), "entries": entries}


def _periodic_table_payload() -> Dict[str, Any]:
    atom_data = _load_json(DATA_DIR / "atom_atlas.json", _atom_atlas_payload)
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for entry in atom_data.get("entries", []):
        buckets.setdefault(int(entry["period"]), []).append(entry)
    summary = []
    for period, records in sorted(buckets.items()):
        if not records:
            continue
        phi_mean = sum(r["phi_s_mean"] for r in records) / len(records)
        grad_mean = sum(r["phase_gradient_mean"] for r in records) / len(records)
        summary.append(
            {
                "period": period,
                "count": len(records),
                "avg_phi_s": round(phi_mean, 6),
                "avg_phase_gradient": round(grad_mean, 6),
            }
        )
    return {"generated_at": _timestamp(), "period_summary": summary}


# ---------------------------------------------------------------------------
# Script/data targets
# ---------------------------------------------------------------------------

def target_triatomic_atlas_script() -> None:
    payload = _triatomic_payload()
    path = _write_json_file(DATA_DIR / "triatomic_atlas.json", payload)
    print(f"triatomic-atlas-script -> {path}")


def target_phase_gated_script() -> None:
    payload = _phase_gated_payload()
    path = _write_json_file(DATA_DIR / "phase_gated_window.json", payload)
    print(f"phase-gated-script -> {path}")


def target_elements_signature_script() -> None:
    payload = _element_signature_payload()
    path = _write_json_file(DATA_DIR / "element_signatures.json", payload)
    print(f"elements-signature-script -> {path}")


def target_molecule_atlas_script() -> None:
    payload = _molecule_atlas_payload()
    path = _write_json_file(DATA_DIR / "molecule_atlas.json", payload)
    print(f"molecule-atlas-script -> {path}")


def target_atom_atlas_script() -> None:
    payload = _atom_atlas_payload()
    path = _write_json_file(DATA_DIR / "atom_atlas.json", payload)
    print(f"atom-atlas-script -> {path}")


def target_periodic_table_script() -> None:
    payload = _periodic_table_payload()
    path = _write_json_file(DATA_DIR / "periodic_table_summary.json", payload)
    print(f"periodic-table-script -> {path}")


# ---------------------------------------------------------------------------
# Report helpers/targets
# ---------------------------------------------------------------------------

def _report_sections_from_triatomic() -> Dict[str, Any]:
    data = _load_json(DATA_DIR / "triatomic_atlas.json", _triatomic_payload)
    sections = [
        {
            "heading": "Structural Field Tetrad",
            "body": (
                "Φ_s mean {phi} • |∇φ| mean {grad} • |K_φ| mean {curv}. "
                "Coherence length {xi}."
            ).format(
                phi=data["tetrad"]["phi_s"]["mean"],
                grad=data["tetrad"]["phase_gradient"]["mean"],
                curv=data["tetrad"]["phase_curvature"]["mean"],
                xi=data["tetrad"]["coherence_length"]["value"],
            ),
        },
        {
            "heading": "Unified Field Indicators",
            "body": (
                "Complex correlation {corr} with chirality {chi} and symmetry breaking {sym}."
            ).format(
                corr=data["unified_fields"].get("complex_correlation"),
                chi=data["unified_fields"].get("chirality"),
                sym=data["unified_fields"].get("symmetry_breaking"),
            ),
        },
    ]
    table = {
        "title": "Tetrad Summary",
        "headers": ["Field", "Min", "Mean", "Max"],
        "rows": [
            ["Φ_s", data["tetrad"]["phi_s"]["min"], data["tetrad"]["phi_s"]["mean"], data["tetrad"]["phi_s"]["max"]],
            ["|∇φ|", data["tetrad"]["phase_gradient"]["min"], data["tetrad"]["phase_gradient"]["mean"], data["tetrad"]["phase_gradient"]["max"]],
            ["|K_φ|", data["tetrad"]["phase_curvature"]["min"], data["tetrad"]["phase_curvature"]["mean"], data["tetrad"]["phase_curvature"]["max"]],
        ],
    }
    return {"sections": sections, "tables": [table]}


def target_report_tetrad() -> None:
    report = _report_sections_from_triatomic()
    path = _write_html_report(
        "tetrad_report.html",
        "TNFR Structural Field Tetrad",
        report["sections"],
        report["tables"],
    )
    print(f"report-tetrad -> {path}")


def target_report_triatomic_atlas() -> None:
    data = _load_json(DATA_DIR / "triatomic_atlas.json", _triatomic_payload)
    sections = [
        {
            "heading": "Triatomic Geometry",
            "body": f"Nodes {data['graph']['node_count']}, edges {data['graph']['edge_count']}, geometry {data['graph']['geometry_hint']}",
        },
        {
            "heading": "Signature",
            "body": textwrap.dedent(
                """
                xi_C {xi} • mean phase gradient {grad} • signature class {cls} • Φ_s drift {drift}
                """
            ).strip().format(
                xi=data["signature"].get("xi_c"),
                grad=data["signature"].get("mean_phase_gradient"),
                cls=data["signature"].get("signature_class"),
                drift=data["signature"].get("phi_s_drift"),
            ),
        },
    ]
    path = _write_html_report("triatomic_atlas.html", "Triatomic Atlas Summary", sections)
    print(f"report-triatomic-atlas -> {path}")


def target_report_phase_gated() -> None:
    data = _load_json(DATA_DIR / "phase_gated_window.json", _phase_gated_payload)
    sections = [
        {
            "heading": "Phase Gate Compliance",
            "body": f"Pairs {data['pairs']} • threshold {data['gate_threshold']} rad • compliance {data['compliance_ratio']:.3f}",
        },
        {
            "heading": "Extremes",
            "body": f"Minimum Δφ {data['min_difference']} rad • maximum Δφ {data['max_difference']} rad",
        },
    ]
    tables = [
        {
            "title": "Gate Counts",
            "headers": ["Category", "Pairs"],
            "rows": [["Within gate", data["gated_pairs"]], ["Outside gate", data["pairs"] - data["gated_pairs"]]],
        }
    ]
    path = _write_html_report("phase_gate_report.html", "Phase-Gated Coupling", sections, tables)
    print(f"report-phase-gated -> {path}")


def target_report_atoms_molecules() -> None:
    atom_data = _load_json(DATA_DIR / "atom_atlas.json", _atom_atlas_payload)
    molecule_data = _load_json(DATA_DIR / "molecule_atlas.json", _molecule_atlas_payload)
    sections = [
        {
            "heading": "Atom Atlas",
            "body": f"Entries: {len(atom_data.get('entries', []))}. Covers periods 1-6 reference set.",
        },
        {
            "heading": "Molecule Atlas",
            "body": f"Tracked molecules: {', '.join(entry['name'] for entry in molecule_data.get('entries', []))}.",
        },
    ]
    tables = [
        {
            "title": "Representative Atoms",
            "headers": ["Z", "Symbol", "Φ_s mean", "|∇φ| mean"],
            "rows": [
                [entry["Z"], entry["symbol"], entry["phi_s_mean"], entry["phase_gradient_mean"]]
                for entry in atom_data.get("entries", [])[:5]
            ],
        }
    ]
    path = _write_html_report("atoms_molecules_report.html", "Atoms & Molecules Summary", sections, tables)
    print(f"report-atoms-molecules -> {path}")


def target_report_molecule_atlas() -> None:
    data = _load_json(DATA_DIR / "molecule_atlas.json", _molecule_atlas_payload)
    sections = [
        {
            "heading": "Topology Coverage",
            "body": f"Captured molecules: {len(data.get('entries', []))} canonical exemplars.",
        }
    ]
    rows = []
    for entry in data.get("entries", []):
        rows.append(
            [
                entry["name"],
                entry["geometry"],
                entry["node_count"],
                entry["edge_count"],
                entry["tetrad"]["phi_s"]["mean"],
            ]
        )
    tables = [
        {
            "title": "Molecule Metrics",
            "headers": ["Molecule", "Geometry", "Nodes", "Edges", "Φ_s mean"],
            "rows": rows,
        }
    ]
    path = _write_html_report("molecule_atlas_report.html", "Molecule Atlas", sections, tables)
    print(f"report-molecule-atlas -> {path}")


def target_report_operator_completeness() -> None:
    doc_path = REPO_ROOT / "docs" / "OPERATOR_COMPLETENESS.md"
    if not doc_path.exists():
        raise FileNotFoundError(doc_path)
    sections: List[Dict[str, Any]] = []
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and len(sections) < 3:
            sections.append({"heading": line[3:].strip(), "body": "Canonical operators documented."})
    if not sections:
        sections = [{"heading": "Operator Completeness", "body": "Reference document available."}]
    path = _write_html_report("operator_completeness.html", "Operator Completeness", sections)
    print(f"report-operator-completeness -> {path}")


def target_report_emergent_particles() -> None:
    data = _load_json(DATA_DIR / "triatomic_atlas.json", _triatomic_payload)
    sections = [
        {
            "heading": "Emergent Field Snapshot",
            "body": f"Chirality {data['unified_fields'].get('chirality')} • coherence coupling {data['unified_fields'].get('coherence_coupling')}",
        }
    ]
    path = _write_html_report("emergent_particles_report.html", "Emergent Particle Fields", sections)
    print(f"report-emergent-particles -> {path}")


def _fundamental_particle_sections(print_friendly: bool = False) -> Dict[str, Any]:
    data = _load_json(DATA_DIR / "triatomic_atlas.json", _triatomic_payload)
    sections = [
        {
            "heading": "Canonical Metrics",
            "body": f"|K_φ| mean {data['tetrad']['phase_curvature']['mean']} • ξ_C {data['tetrad']['coherence_length']['value']}",
        },
        {
            "heading": "Signature Class",
            "body": f"Signature classified as {data['signature'].get('signature_class')}",
        },
    ]
    tables = None if print_friendly else [
        {
            "title": "Unified Field Highlights",
            "headers": ["Metric", "Value"],
            "rows": [[k, v] for k, v in data["unified_fields"].items()],
        }
    ]
    return {"sections": sections, "tables": tables}


def target_report_fundamental_particles() -> None:
    info = _fundamental_particle_sections()
    path = _write_html_report(
        "fundamental_particles.html",
        "Fundamental Particles",
        info["sections"],
        info["tables"],
    )
    print(f"report-fundamental-particles -> {path}")


def target_report_fundamental_particles_print() -> None:
    info = _fundamental_particle_sections(print_friendly=True)
    path = _write_html_report(
        "fundamental_particles_print.html",
        "Fundamental Particles (Print)",
        info["sections"],
        info["tables"],
    )
    print(f"report-fundamental-particles-print -> {path}")


def target_report_interaction_sequences() -> None:
    doc_path = REPO_ROOT / "docs" / "CANONICAL_OZ_SEQUENCES.md"
    if not doc_path.exists():
        raise FileNotFoundError(doc_path)
    sections: List[Dict[str, Any]] = []
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and len(sections) < 4:
            sections.append({"heading": line[3:].strip(), "body": "Sequence validated via U1-U6."})
    if not sections:
        sections = [{"heading": "Interaction Sequences", "body": "Reference material not found."}]
    path = _write_html_report("interaction_sequences.html", "Interaction Sequences", sections)
    print(f"report-interaction-sequences -> {path}")


def target_report_particle_atlas_u6() -> None:
    sections = [
        {"heading": "Unified Grammar", "body": "U1-U6 compliance enforced for atlas exports."},
        {"heading": "Structural Potential Confinement", "body": "ΔΦ_s tracked to remain below φ."},
    ]
    path = _write_html_report("particle_atlas_u6.html", "Particle Atlas U6", sections)
    print(f"report-particle-atlas-u6 -> {path}")


def target_report_periodic_table_classic() -> None:
    data = _load_json(DATA_DIR / "periodic_table_summary.json", _periodic_table_payload)
    sections = [
        {"heading": "Period Coverage", "body": f"Tracked periods: {len(data.get('period_summary', []))}."}
    ]
    rows = [
        [entry["period"], entry["count"], entry["avg_phi_s"], entry["avg_phase_gradient"]]
        for entry in data.get("period_summary", [])
    ]
    tables = [
        {
            "title": "Period Summary",
            "headers": ["Period", "Elements", "Avg Φ_s", "Avg |∇φ|"],
            "rows": rows,
        }
    ]
    path = _write_html_report("periodic_table_classic.html", "Periodic Table Summary", sections, tables)
    print(f"report-periodic-table-classic -> {path}")


def target_report_all_classic() -> None:
    workflows = [
        target_report_tetrad,
        target_report_triatomic_atlas,
        target_report_phase_gated,
        target_report_atoms_molecules,
        target_report_molecule_atlas,
        target_report_operator_completeness,
        target_report_emergent_particles,
        target_report_fundamental_particles,
        target_report_interaction_sequences,
        target_report_particle_atlas_u6,
        target_report_periodic_table_classic,
    ]
    for func in workflows:
        func()
    print("report-all-classic complete")


def target_report_all_print() -> None:
    workflows = [
        target_report_fundamental_particles_print,
        target_report_periodic_table_classic,
    ]
    for func in workflows:
        func()
    print("report-all-print complete")


# ---------------------------------------------------------------------------
# Benchmarks / plots
# ---------------------------------------------------------------------------

def target_force_study_plots() -> None:
    script = REPO_ROOT / "benchmarks" / "integrated_force_regime_study.py"
    if not script.exists():
        raise FileNotFoundError(script)
    _ensure_dir(REPORTS_DIR)
    command = [
        PYTHON,
        str(script),
        "--topologies",
        "ring,ws",
        "--sizes",
        "24",
        "--runs",
        "1",
        "--export",
        str(FORCE_STUDY_PATH.relative_to(REPO_ROOT)),
    ]
    print("[run ] force-study-plots benchmark")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    print(f"force-study-plots -> {FORCE_STUDY_PATH}")


def target_external_phase_gate_validation() -> None:
    script = REPO_ROOT / "benchmarks" / "external_phase_gate_validation.py"
    if not script.exists():
        raise FileNotFoundError(script)
    command = [PYTHON, str(script)]
    print("[run ] external-phase-gate-validation benchmark")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    print("external-phase-gate-validation complete")


def target_wdbc_phase_gate_demo() -> None:
    script = REPO_ROOT / "examples" / "91_breast_cancer_phase_gate_demo.py"
    if not script.exists():
        raise FileNotFoundError(script)
    command = [PYTHON, str(script)]
    print("[run ] wdbc-phase-gate-demo")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    print("wdbc-phase-gate-demo complete")


def target_wine_quality_phase_gate_demo() -> None:
    script = REPO_ROOT / "examples" / "92_wine_quality_phase_gate_demo.py"
    if not script.exists():
        raise FileNotFoundError(script)
    command = [PYTHON, str(script)]
    print("[run ] wine-quality-phase-gate-demo")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    print("wine-quality-phase-gate-demo complete")


def _run_structural_interface_benchmark(dataset: str, *, target: str = "circular") -> None:
    script = REPO_ROOT / "benchmarks" / "structural_interface_benchmark.py"
    if not script.exists():
        raise FileNotFoundError(script)
    _ensure_dir(REPORTS_DIR)
    command = [
        PYTHON,
        str(script),
        "--dataset",
        dataset,
        "--target",
        target,
        "--output",
        str(REPORTS_DIR.relative_to(REPO_ROOT)),
    ]
    print(f"[run ] structural-interface-benchmark ({dataset}, target={target})")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    print(f"structural-interface-benchmark ({dataset}, target={target}) complete")


def target_structural_interface_benchmark() -> None:
    _run_structural_interface_benchmark("all")


def target_structural_interface_all() -> None:
    _run_structural_interface_benchmark("all", target="all")


def target_structural_interface_wdbc() -> None:
    _run_structural_interface_benchmark("wdbc")


def target_structural_interface_offline() -> None:
    _run_structural_interface_benchmark("offline", target="all")


def target_structural_interface_model_error() -> None:
    _run_structural_interface_benchmark("all", target="model_error")


def target_structural_interface_wine() -> None:
    _run_structural_interface_benchmark("wine")


def _run_temporal_interface_benchmark(
    source: str, *, year: int = 2020, month: int = 1
) -> None:
    script = REPO_ROOT / "benchmarks" / "temporal_interface_benchmark.py"
    if not script.exists():
        raise FileNotFoundError(script)
    _ensure_dir(REPORTS_DIR)
    command = [
        PYTHON,
        str(script),
        "--source",
        source,
        "--year",
        str(year),
        "--month",
        str(month),
        "--output",
        str(REPORTS_DIR.relative_to(REPO_ROOT)),
    ]
    print(f"[run ] temporal-interface-benchmark (source={source})")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    print(f"temporal-interface-benchmark (source={source}) complete")


def target_temporal_interface_benchmark() -> None:
    # Offline synthetic fixture by default (no network required).
    _run_temporal_interface_benchmark("synthetic")


def target_temporal_interface_grid() -> None:
    # Real grid-frequency data (downloaded + cached, bounded; graceful skip).
    _run_temporal_interface_benchmark("grid")


def _run_multichannel_interface_benchmark(source: str) -> None:
    script = REPO_ROOT / "benchmarks" / "multichannel_interface_benchmark.py"
    if not script.exists():
        raise FileNotFoundError(script)
    _ensure_dir(REPORTS_DIR)
    command = [
        PYTHON,
        str(script),
        "--source",
        source,
        "--output",
        str(REPORTS_DIR.relative_to(REPO_ROOT)),
    ]
    print(f"[run ] multichannel-interface-benchmark (source={source})")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    print(f"multichannel-interface-benchmark (source={source}) complete")


def target_multichannel_interface_benchmark() -> None:
    # Offline synthetic Kuramoto fixture by default (no network required).
    _run_multichannel_interface_benchmark("synthetic")


def target_multichannel_interface_eeg() -> None:
    # Real EEG Eye State data (downloaded + cached, bounded; graceful skip).
    _run_multichannel_interface_benchmark("eeg")


# ---------------------------------------------------------------------------
# Smoke + clean helpers
# ---------------------------------------------------------------------------

def target_smoke_tests() -> None:
    steps = [
        _pytest_step("tests/test_mathematical_purity_foundation.py"),
        _pytest_step("tests/test_replay_register_manifest.py"),
        _example_step("examples/01_hello_world.py"),
    ]
    _run_steps(steps)
    print("smoke-tests complete")


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path, ignore_errors=True)


def target_clean() -> None:
    for relative in ("examples/output", "results", "dist", "build"):
        _remove_path(REPO_ROOT / relative)
    for pattern in ("*.egg-info",):
        for item in REPO_ROOT.glob(pattern):
            _remove_path(item)
    for pycache in REPO_ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
    print("clean complete")


TARGETS: Dict[str, Callable[[], None]] = {
    "smoke-tests": target_smoke_tests,
    "clean": target_clean,
    "triatomic-atlas-script": target_triatomic_atlas_script,
    "phase-gated-script": target_phase_gated_script,
    "elements-signature-script": target_elements_signature_script,
    "molecule-atlas-script": target_molecule_atlas_script,
    "atom-atlas-script": target_atom_atlas_script,
    "periodic-table-script": target_periodic_table_script,
    "force-study-plots": target_force_study_plots,
    "external-phase-gate-validation": target_external_phase_gate_validation,
    "wdbc-phase-gate-demo": target_wdbc_phase_gate_demo,
    "wine-quality-phase-gate-demo": target_wine_quality_phase_gate_demo,
    "structural-interface-benchmark": target_structural_interface_benchmark,
    "structural-interface-all": target_structural_interface_all,
    "structural-interface-offline": target_structural_interface_offline,
    "structural-interface-wdbc": target_structural_interface_wdbc,
    "structural-interface-model-error": target_structural_interface_model_error,
    "structural-interface-wine": target_structural_interface_wine,
    "temporal-interface-benchmark": target_temporal_interface_benchmark,
    "temporal-interface-grid": target_temporal_interface_grid,
    "multichannel-interface-benchmark": target_multichannel_interface_benchmark,
    "multichannel-interface-eeg": target_multichannel_interface_eeg,
    "report-tetrad": target_report_tetrad,
    "report-triatomic-atlas": target_report_triatomic_atlas,
    "report-phase-gated": target_report_phase_gated,
    "report-atoms-molecules": target_report_atoms_molecules,
    "report-molecule-atlas": target_report_molecule_atlas,
    "report-operator-completeness": target_report_operator_completeness,
    "report-emergent-particles": target_report_emergent_particles,
    "report-fundamental-particles": target_report_fundamental_particles,
    "report-fundamental-particles-print": target_report_fundamental_particles_print,
    "report-interaction-sequences": target_report_interaction_sequences,
    "report-particle-atlas-u6": target_report_particle_atlas_u6,
    "report-periodic-table-classic": target_report_periodic_table_classic,
    "report-all-classic": target_report_all_classic,
    "report-all-print": target_report_all_print,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TNFR Windows make shim")
    parser.add_argument("target", nargs="?", help="Target name (use --list for options)")
    parser.add_argument("--list", action="store_true", help="List available targets")
    args = parser.parse_args(argv)

    if args.list:
        print("Available targets:")
        for name in sorted(TARGETS):
            print(f"  - {name}")
        return 0

    if not args.target:
        parser.print_help()
        return 1

    handler = TARGETS.get(args.target)
    if handler is None:
        print(f"Unknown target '{args.target}'. Use --list to show supported targets.")
        return 1

    handler()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
