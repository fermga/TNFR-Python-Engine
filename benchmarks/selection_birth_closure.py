"""Bounded selection, dispatch and pressure-consumption audit on causal C8.

Selectors keep their built-in policies. Explicit THOL interventions are
separate controls, never relabeled as selector decisions or autonomous birth.
Each branch executes a fresh copy of the shared preparation procedure: live
graphs contain locks and must not be shallow-copied or reconstructed by history
assignment. The exact symmetry examples are detached declared models.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.thol_birth_transport import (  # noqa: E402
    CASES,
    prepare_birth_selection_source,
    run_birth_transport_case,
)
from benchmarks.thol_pressure_feedback import (  # noqa: E402
    NEXT_STEP,
    _RecordingIntegrator,
    _integration_record,
    _payload,
    _state,
)
from tnfr.alias import get_attr  # noqa: E402
from tnfr.constants.aliases import ALIAS_DSI, ALIAS_SI  # noqa: E402
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.dynamics.integrators import update_epi_via_nodal_equation  # noqa: E402
from tnfr.dynamics.runtime import step  # noqa: E402
from tnfr.dynamics.selectors import (  # noqa: E402
    AbstractSelector,
    DefaultGlyphSelector,
    ParametricGlyphSelector,
)
from tnfr.operators import apply_glyph  # noqa: E402
from tnfr.operators.definitions import SelfOrganization  # noqa: E402
from tnfr.operators.factor_contracts import (
    resolve_runtime_operator_factors,
)  # noqa: E402
from tnfr.operators.grammar_dynamics import validate_candidate  # noqa: E402
from tnfr.physics.selector_symmetry import derive_selector_symmetry  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.selector import _selector_thresholds  # noqa: E402
from tnfr.types import Glyph  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

SELECTORS = {"default": DefaultGlyphSelector, "parametric": ParametricGlyphSelector}
DISPATCHES = ("none", "primitive", "public")


def _policy(graph, *, preparation=None):
    prefixes = ("SELECTOR_", "GLYPH_", "GRAMMAR_", "THOL_", "UM_", "PHASE_", "VF_")
    names = (
        "RANDOM_SEED",
        "EPI_MIN",
        "EPI_MAX",
        "CLIP_MODE",
        "SORT_NODES",
        "GRAMMAR",
        "AL_MAX_LAG",
        "EN_MAX_LAG",
        "SI_WEIGHTS",
        "VALIDATE_OPERATOR_PRECONDITIONS",
        "GAMMA",
        "_gamma_spec",
        "use_extended_dynamics",
        "INTEGRATOR_METHOD",
        "DT",
        "DT_MIN",
        "DT_MAX",
        "EPS_DNFR_STABLE",
    )
    targets = (0,) if preparation is None else preparation["prep_targets"]
    prefix = (
        ("IL", "OZ")
        if preparation is None
        else tuple(row["glyph"] for row in preparation["actual_prefix"])
    )
    return {
        "declared_graph_settings": {
            key: deepcopy(value)
            for key, value in graph.graph.items()
            if key.startswith(prefixes) or key in names
        },
        "resolved_selector_thresholds": dict(_selector_thresholds(graph)),
        "normalized_pressure_weights": dict(graph.graph["_dnfr_weights"]),
        "resolved_thol_factors": dict(
            resolve_runtime_operator_factors(
                graph.graph.get("GLYPH_FACTORS"),
                Glyph.THOL,
                graph.graph,
            )
        ),
        "external_target": targets[0] if len(targets) == 1 else None,
        "external_targets": targets,
        "external_prefix": prefix,
        "scope": "Configured choices and preparation, not constants derived here",
    }


def _decision_history(graph):
    history = graph.graph.get("history", {})
    return {
        "since_AL": dict(history.get("since_AL", {})),
        "since_EN": dict(history.get("since_EN", {})),
        "sense_sigma_mag": tuple(history.get("sense_sigma_mag", ())),
        "mutation_abstentions": deepcopy(history.get("mutation_abstentions", ())),
    }


class _RecordingSelector(AbstractSelector):
    """Delegate the actual lifecycle without changing its returned decisions."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.preparations = []
        self.decisions = {}

    def prepare(self, graph, nodes):
        self.delegate.prepare(graph, nodes)
        preselection = self.delegate._preselection
        self.preparations.append(
            {
                "state": _state(graph),
                "nodes": tuple(nodes),
                "decision_history_before_lag_increment": _decision_history(graph),
                "sense_index": tuple(
                    float(get_attr(graph.nodes[n], ALIAS_SI, 0.5)) for n in nodes
                ),
                "sense_delta": tuple(
                    float(get_attr(graph.nodes[n], ALIAS_DSI, 0.0)) for n in nodes
                ),
                "preselection": asdict(preselection),
                "normalization": dict(graph.graph["_sel_norms"]),
                "selector_weights": dict(graph.graph.get("_selector_weights", {})),
            }
        )

    def select(self, graph, node):
        result = self.delegate.select(graph, node)
        self.decisions[node] = str(result)
        return result


def run_selector_case(name):
    """One ordinary all-node runtime step with refreshed pressure and Si."""
    if name not in SELECTORS:
        raise ValueError(f"selector must be one of {tuple(SELECTORS)}")
    graph, preparation = prepare_birth_selection_source()
    return _run_prepared_selector(graph, preparation, name)


def _run_prepared_selector(graph, preparation, name):
    """Shared instrumentation of the built-in lifecycle on a prepared graph."""
    if name not in SELECTORS:
        raise ValueError(f"selector must be one of {tuple(SELECTORS)}")
    graph.graph["GLYPH_SELECTOR_N_JOBS"] = 1
    selector = _RecordingSelector(SELECTORS[name]())
    integrator = _RecordingIntegrator()
    graph.graph.update(
        glyph_selector=selector,
        integrator=integrator,
        INTEGRATOR_METHOD="euler",
    )
    policy = _policy(graph, preparation=preparation)
    step(graph, dt=NEXT_STEP, use_Si=True, apply_glyphs=True)
    return {
        "selector": name,
        "preparation": preparation,
        "policy": policy,
        "selection_contexts": selector.preparations,
        "actual_selector_proposals": selector.decisions,
        "integration": integrator.records,
        "endpoint": _state(graph),
        "decision_history_after_step": _decision_history(graph),
        "scope": (
            "Built-in proposals followed by ordinary lag/grammar/primitive "
            "dispatch and integration; committed histories and integrator "
            "input are recorded separately. Subsequent runtime phase/capacity "
            "updates are included only in the whole-step endpoint"
        ),
    }


def run_dispatch_case(dispatch, *, refresh):
    """An explicitly requested parent THOL, separate from selector selection."""
    if dispatch not in DISPATCHES:
        raise ValueError(f"dispatch must be one of {DISPATCHES}")
    graph, preparation = prepare_birth_selection_source()
    policy = _policy(graph)
    before = _state(graph)
    admission = validate_candidate(graph, 0, "THOL")
    if not admission.allowed:
        raise RuntimeError("the causal source did not admit the THOL control")
    if dispatch == "primitive":
        apply_glyph(graph, 0, "THOL")
    elif dispatch == "public":
        SelfOrganization()(graph, 0, collect_metrics=True)
    raw = _state(graph)
    if refresh:
        default_compute_delta_nfr(graph)
    consumed = _state(graph)
    update_epi_via_nodal_equation(graph, dt=NEXT_STEP, method="euler")
    return {
        "dispatch": dispatch,
        "refresh_before_flow": refresh,
        "preparation": preparation,
        "policy": policy,
        "admission_allowed": admission.allowed,
        "before": before,
        "raw_after_dispatch": raw,
        "integration": _integration_record(consumed, _state(graph), graph),
        "scope": (
            "Explicit parent intervention and one held-input shared Euler "
            "step, not a selector-selected birth or a complete operator word"
        ),
    }


def symmetry_examples():
    """Exact necessary-condition examples, not symmetry of the runtime C8."""
    count = 8
    group = tuple(
        tuple((offset + sign * i) % count for i in range(count))
        for sign in (1, -1)
        for offset in range(count)
    )
    support = tuple(
        tuple((i - j) % count in (1, count - 1) for j in range(count))
        for i in range(count)
    )
    # Full state of this abstract example: uniform triad, empty history, no marks.
    # It has no runtime sampling order, hidden phase gauge or selector adapter.
    uniform = ((Fraction(1), Fraction(1), Fraction(0), (), None),) * count
    marked = tuple(
        (*label[:-1], "prepared-parent" if i == 0 else None)
        for i, label in enumerate(uniform)
    )
    results = {}
    for name, labels in (("uniform_unmarked", uniform), ("marked_parent", marked)):
        result = derive_selector_symmetry(
            state_labels=labels,
            relation_labels=support,
            permutations=group,
            candidates=tuple(range(count)),
        )
        results[name] = {
            **asdict(result),
            "unique_equivariant_selection_obstructed": (
                result.unique_equivariant_selection_obstructed
            ),
            "scope": result.scope,
        }
    return results


def run_study():
    return {
        "selectors": [run_selector_case(name) for name in SELECTORS],
        "dispatch_controls": [
            run_dispatch_case(dispatch, refresh=refresh)
            for dispatch in DISPATCHES
            for refresh in (False, True)
        ],
        "transport_controls": [run_birth_transport_case(case) for case in CASES],
        "detached_symmetry_examples": symmetry_examples(),
        "limitations": (
            "State readouts are declared projections, not complete runtime seals",
            "Prepared IL/OZ target already breaks parent equivalence",
            "Positive public birth/UM controls have externally supplied targets and words",
            "No autonomous selector-to-public-birth adapter is derived",
            "No sustained pattern, physical particle or physical correspondence is certified",
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "artifacts/research/selection_birth_closure_2026_09_18.json"),
    )
    args = parser.parse_args()
    scope = (
        "src/tnfr",
        "benchmarks/selection_birth_closure.py",
        "benchmarks/thol_birth_transport.py",
        "benchmarks/thol_pressure_feedback.py",
        "benchmarks/capacity_localization.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O1.b-O3.a-selection-birth-closure-audit",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        graph_construction="Existing marked causal C8 checkerboard preparation",
        capacity_specification="Unit preparation; unchanged operator factors",
        solver="Shared Euler; recorded held or refreshed pressure",
        timestep=NEXT_STEP,
        seed=17,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=("IL", "OZ", "branch-specific recorded continuation"),
        telemetry=(
            "actual proposals and commits",
            "birth/link support",
            "consumed pressure",
        ),
        controls=(
            "no THOL",
            "primitive versus public",
            "held versus refreshed",
            "disabled links",
            "stale sample",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **run_study()}
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed while executing the study")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
    safe_write(args.output, lambda stream: stream.write(encoded))
    print(f"Wrote bounded selection/birth audit to {args.output}")


if __name__ == "__main__":
    main()
