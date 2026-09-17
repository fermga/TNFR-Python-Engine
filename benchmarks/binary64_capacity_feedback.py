"""Bind a small fixed set of P2 lattice boundaries to actual UM and nodal flow.

Each independent preparation executes one UM, one refreshed h=0.25 held
Euler interval, then SHA. The 512-update bound belongs to the conditional
numeric capacity theorem; no long runtime word or horizon search is run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.capacity_feedback import (  # noqa: E402
    SOURCE_SCOPE, STEP, _artifact_payload, _family, prepare_p2,
)
from benchmarks.child_coupling_feedback import _advance_forced_support_interval  # noqa: E402
from benchmarks.structural_perturbation_response import (  # noqa: E402
    _diagnostics, _event, _materialized, _reference, _word,
)
from tnfr.operators.definitions import Coupling, Silence  # noqa: E402
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors  # noqa: E402
from tnfr.physics.capacity_feedback import (  # noqa: E402
    derive_p2_binary64_coupling_lattice, observe_p2_binary64_coupling_lattice,
)
from tnfr.physics.forced_support import (  # noqa: E402
    observe_forced_support_pattern, observe_forced_support_step,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.types import Glyph  # noqa: E402

INDICES = (0, 1, 6, 7, 2**52)


def run_binary64_capacity_case(index):
    if type(index) is not int or index not in INDICES:
        raise ValueError(f"index must be one of the declared boundary controls {INDICES}")
    graph = prepare_p2(capacity=(1.0 + index * math.ulp(1.0), 1.0))
    current = capture_non_epi_forcing(graph)
    # Independent comparison data, not an invented ancestor of this graph.
    baseline = capture_non_epi_forcing(prepare_p2())
    target = _reference(baseline)
    factors = resolve_runtime_operator_factors(graph.graph["GLYPH_FACTORS"], Glyph.UM, graph.graph)
    lattice = derive_p2_binary64_coupling_lattice(coupling_factor=factors["UM_vf_sync"])
    ops = (Coupling(), Silence())
    word, admission = _word(ops, initialized=True)
    if word is None:
        raise RuntimeError(f"finite UM/SHA word refused: {admission}")
    before = current
    record = {
        "case": index, "word": admission, "initial": _materialized(graph),
        "initial_capture": asdict(before), "initial_diagnostics": _diagnostics(graph),
        "uniform_comparison_reference": asdict(target), "lattice_reference": asdict(lattice),
        "comparison_scope": "Independent uniform-capacity P2 reference, not an observed ancestor",
    }
    current, event = _event(graph, 0, ops[0], word.step(0), target, current, refresh=True)
    if event["status"] != "executed":
        raise RuntimeError(f"declared UM refused: {event['reason']}")
    observation = observe_p2_binary64_coupling_lattice(
        lattice, capacity_before=float(before.snapshot.capacity[0]),
        capacity_after=float(current.snapshot.capacity[0]),
    )
    family_before = _family(before, baseline, base_capacity=1)
    family_event = _family(current, baseline, base_capacity=1)
    held = _reference(current)
    following, flow = _advance_forced_support_interval(
        graph, target, held, current, current, duration=STEP,
    )
    family_after = _family(following, baseline, base_capacity=1)
    budget = observe_forced_support_step(held, current.snapshot, following.snapshot, STEP)
    h = Fraction(STEP)
    modeled_change = tuple(h * nu * p for nu, p in zip(
        current.snapshot.capacity, budget.before.modeled_pressure, strict=True,
    ))
    pressure_effect = tuple(h * nu * p for nu, p in zip(
        current.snapshot.capacity, budget.before.pressure_defect, strict=True,
    ))
    execution_effect = budget.support_budget.state_defect
    actual_change = tuple(b - a for a, b in zip(current.snapshot.epi, following.snapshot.epi))
    residual = tuple(actual - modeled - dp - dx for actual, modeled, dp, dx in zip(
        actual_change, modeled_change, pressure_effect, execution_effect, strict=True,
    ))
    if any(residual):
        raise RuntimeError("represented EPI change lost its exact nodal defect budget")
    record.update(
        status="measured", event=event, flow=flow,
        lattice_observation=asdict(observation),
        family_checks=(family_before, family_event, family_after),
        post_um_capture=asdict(current), final_capture=asdict(following),
        final_before_closure=_materialized(graph), final_diagnostics=_diagnostics(graph),
        represented_epi_change=actual_change, exact_model_epi_change=modeled_change,
        pressure_realization_epi_effect=pressure_effect, execution_epi_effect=execution_effect,
        epi_identity_residual=residual,
        current_exact_frozen_profile=held.relative_profile,
        current_conditional_profile_mismatch=asdict(observe_forced_support_pattern(
            target, nodes=held.source.nodes, epi=held.relative_profile,
        )),
        actual_uniform_target_error=asdict(observe_forced_support_pattern(
            target, nodes=following.snapshot.nodes, epi=following.snapshot.epi,
        )),
    )
    _, closure = _event(graph, 0, ops[1], word.step(1), target, following, refresh=True)
    if closure["status"] != "executed":
        raise RuntimeError(f"declared closure refused: {closure['reason']}")
    record["closure_after_measurement"] = closure
    record["scope"] = (
        "One actual UM and executor-owned held-input interval per independently "
        "prepared boundary. A capacity-kernel terminal index is not an observed "
        "future runtime state. A nonzero exact frozen profile is not a lower "
        "bound on represented EPI error; histories and derivatives are not "
        "claimed fixed even when primary coordinates are unchanged."
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/research/binary64_capacity_feedback.json",
    )
    args = parser.parse_args()
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-P2-binary64-capacity-boundaries-and-EPI-stall",
        git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "networkx": nx.__version__,
                  "numpy": np.__version__},
        graph_construction="Five independent unit P2 graphs, phase zero and EPI=(0.5,0.5)",
        capacity_specification="(1+j*2^-52,1), declared j in (0,1,6,7,2^52); default target UM",
        solver="Shared default Euler with canonical refresh; four held-input internal substeps",
        timestep=STEP, seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=("one UM", "one held Euler interval", "SHA after measurement"),
        telemetry=("capacity lattice and production endpoint", "tetrad/coherence and histories",
                   "exact-model/represented EPI distinction", "signed pressure/execution defects"),
        controls=("zero gap", "six/seven index boundary", "upper interval endpoint"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": SOURCE_SCOPE,
              "cases": [run_binary64_capacity_case(index) for index in INDICES],
              "experimental_status": "No empirical correspondence tested"}
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance:
        raise RuntimeError("source changed while executing the boundary artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_artifact_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Wrote finite represented-capacity boundary controls to {args.output}")


if __name__ == "__main__":
    main()
