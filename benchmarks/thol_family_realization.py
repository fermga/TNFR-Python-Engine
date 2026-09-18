"""Derive exact finite state realizations from authenticated family evidence.

The complete preceding family analysis is replayed before its models are used.
All algebra is delegated to the shared observer; no graph, trajectory,
exponential, partition search or fitted temporal law is constructed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from benchmarks.thol_family_closure import (  # noqa: E402
    CONTROL_PATH, CONTROL_SHA256, LINEAGE_PATH, LINEAGE_SHA256, MODEL_ORDER,
    _check_digest, _digest, _equal, _json, _reference, _snapshot,
    run_study as replay_family_study,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.utils.io import safe_write  # noqa: E402

FAMILY_PATH = ROOT / "artifacts/research/thol_family_closure_2026_09_18.json"
FAMILY_SHA256 = "13642d936260e849ac5e588f32a6ea6716b74d733ab8232e4376f5f48d621ae1"
DEFAULT_MAX_RANK_CALLS = 4096
_ENVELOPE_FIELDS = ("manifest", "source_scope", "experimental_status")


def load_family_evidence(family_path=FAMILY_PATH, control_path=CONTROL_PATH,
                         lineage_path=LINEAGE_PATH, *,
                         expected_family_sha256=FAMILY_SHA256,
                         expected_control_sha256=CONTROL_SHA256,
                         expected_lineage_sha256=LINEAGE_SHA256):
    """Admit bytes, retain the old producer, and compare every scientific field.

    Actual file locations may differ from the producer's recorded locators.
    Only those two locator strings are normalized for the replay comparison;
    complete byte identities, producer metadata and calculations must agree.
    """
    _check_digest(expected_family_sha256)
    raw = Path(family_path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_family_sha256:
        raise ValueError("retained family bytes differ from their expected digest")
    retained = json.loads(raw)
    manifest = CoreExperimentManifest(**retained["manifest"])
    manifest.validate_for_admission()
    if manifest.claim_id != "O1.b-generated-family-affine-state-sufficiency":
        raise ValueError("family evidence must preserve the original family-analysis producer")
    if (not isinstance(retained["source_scope"], list) or not retained["source_scope"]
            or any(type(item) is not str or not item for item in retained["source_scope"])):
        raise ValueError("retained family source scope must be a nonempty path list")
    if retained["experimental_status"] != "No empirical correspondence tested":
        raise ValueError("family evidence must preserve its non-empirical scope")
    replayed = replay_family_study(
        control_path, lineage_path, expected_control_sha256=expected_control_sha256,
        expected_lineage_sha256=expected_lineage_sha256,
    )
    scientific = {key: value for key, value in retained.items() if key not in _ENVELOPE_FIELDS}
    normalized = _json(replayed)
    for key in ("retained_controls", "retained_lineage"):
        if type(scientific[key]["path"]) is not str or not scientific[key]["path"]:
            raise ValueError("historical input locator must remain nonempty text")
        normalized[key]["path"] = scientific[key]["path"]
    _equal(normalized, scientific, "complete family scientific payload")
    if tuple(row["model"] for row in replayed["models"]) != MODEL_ORDER:
        raise ValueError("family replay lost the complete four-model order")
    if hashlib.sha256(Path(family_path).read_bytes()).hexdigest() != actual:
        raise RuntimeError("family evidence changed during admission")
    binding = {
        "path": str(family_path), "sha256": actual, "byte_count": len(raw),
        "historical_manifest": retained["manifest"],
        "historical_source_scope": retained["source_scope"],
        "historical_experimental_status": retained["experimental_status"],
        "historical_producer_preserved": True,
    }
    comparison = {
        "whole_scientific_payload_equal": True,
        "compared_fields": tuple(scientific),
        "metadata_retained_without_reassignment": _ENVELOPE_FIELDS,
        "locator_fields_normalized": ("retained_controls.path", "retained_lineage.path"),
        "scope": (
            "All preceding scientific fields are replayed, including every cached model, "
            "closure matrix, witness, numerical diagnostic, reset and input identity. "
            "File locators may move; hashes and original producer metadata may not."
        ),
    }
    return replayed, binding, comparison


def run_study(family_path=FAMILY_PATH, control_path=CONTROL_PATH,
              lineage_path=LINEAGE_PATH, *, expected_family_sha256=FAMILY_SHA256,
              expected_control_sha256=CONTROL_SHA256,
              expected_lineage_sha256=LINEAGE_SHA256,
              max_rank_calls=DEFAULT_MAX_RANK_CALLS):
    """Complete four bounded exact realizations or raise without partial output."""
    from tnfr.physics.epi_memory import observe_forced_support_realization

    if type(max_rank_calls) is not int or max_rank_calls <= 0:
        raise ValueError("max_rank_calls must be a positive integer")
    family, binding, comparison = load_family_evidence(
        family_path, control_path, lineage_path,
        expected_family_sha256=expected_family_sha256,
        expected_control_sha256=expected_control_sha256,
        expected_lineage_sha256=expected_lineage_sha256,
    )
    models = []
    blocks = family["partition"]["blocks"]
    for old in family["models"]:
        # Reuse the existing strict deserializers. They check every cached field.
        row = _json(old)
        reference = _reference(row["exact_observation"]["reference"])
        snapshot = _snapshot(row["actual_snapshot"])
        realization = observe_forced_support_realization(
            reference, blocks, epi=snapshot.epi, max_rank_calls=max_rank_calls,
        )
        _equal(asdict(realization.closure), row["exact_observation"],
               "realization closure against authenticated family replay")
        models.append({
            "model": old["model"], "observation_time": old["observation_time"],
            "reference_sha256": old["reference_sha256"],
            "actual_snapshot_sha256": _digest(asdict(snapshot)),
            "realization": asdict(realization),
            "old_target_readout": deepcopy(old["old_target_readout"]),
            "observer_reset": deepcopy(old["observer_reset"]),
            "scope": (
                "Realization uses the raw affine state of this fixed model. "
                "The original target and event observer reset remain separate retained readouts."
            ),
        })
    bindings = {
        "retained_family": binding,
        "retained_controls": family["retained_controls"],
        "retained_lineage": family["retained_lineage"],
    }
    for path, key in ((family_path, "retained_family"),
                      (control_path, "retained_controls"), (lineage_path, "retained_lineage")):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != bindings[key]["sha256"]:
            raise RuntimeError("retained input bytes changed during realization")
    return {
        **bindings, "family_replay": comparison,
        "partition": deepcopy(family["partition"]), "models": tuple(models),
        "max_rank_calls_per_model": max_rank_calls,
        "new_trajectories_executed": False, "matrix_exponentials_evaluated": False,
        "partition_search_performed": False,
        "scope": (
            "Exact finite all-state affine realizations of the declared family outputs for "
            "four retained fixed models. A full fine-state dimension is an obstruction to "
            "reducing this observation, not to TNFR or to other physical models. No graph "
            "execution, fitted memory, empirical macro-entity or autonomous maintenance is claimed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=Path, default=FAMILY_PATH)
    parser.add_argument("--controls", type=Path, default=CONTROL_PATH)
    parser.add_argument("--lineage", type=Path, default=LINEAGE_PATH)
    parser.add_argument("--expected-family-sha256", default=FAMILY_SHA256)
    parser.add_argument("--expected-control-sha256", default=CONTROL_SHA256)
    parser.add_argument("--expected-lineage-sha256", default=LINEAGE_SHA256)
    parser.add_argument("--max-rank-calls", type=int, default=DEFAULT_MAX_RANK_CALLS)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "artifacts/research/thol_family_realization.json"))
    args = parser.parse_args()
    inputs = (args.family, args.controls, args.lineage)
    if args.output.resolve() in tuple(path.resolve() for path in inputs):
        raise ValueError("new realization must not overwrite retained input evidence")
    scope = ("src/tnfr", "benchmarks/thol_family_realization.py",
             "benchmarks/thol_family_closure.py", "benchmarks/thol_lineage_coordination.py",
             "benchmarks/thol_distributed_target.py", "benchmarks/thol_distributed_transport.py",
             "benchmarks/thol_birth_transport.py", "benchmarks/thol_pressure_feedback.py",
             "benchmarks/thol_eligibility_dispatch.py", "benchmarks/capacity_localization.py",
             "benchmarks/structural_target_compatibility.py", "benchmarks/thol_preparation_policy.py",
             "benchmarks/selection_birth_closure.py")
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O1.b-generated-family-exact-state-realization", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="No graph construction; fully replayed retained family evidence",
        capacity_specification="Four authenticated held forced models; no coefficient fitting",
        solver="Bounded exact row-space realization; no time integration or exponential",
        timestep=None, seed=None, result_status=ClaimStatus.DERIVED,
        operator_sequence=(), telemetry=("exact rank progression", "C/T/G/D/Cb",
                                         "projected-state/rate identities", "resource counters"),
        controls=("whole preceding family analysis replay", "exact affine identity checks"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    result = run_study(
        args.family, args.controls, args.lineage,
        expected_family_sha256=args.expected_family_sha256,
        expected_control_sha256=args.expected_control_sha256,
        expected_lineage_sha256=args.expected_lineage_sha256,
        max_rank_calls=args.max_rank_calls,
    )
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("analysis source changed during execution")
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **result,
              "experimental_status": "No empirical correspondence tested"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
    safe_write(args.output, lambda stream: stream.write(encoded))
    print(f"Wrote detached family state-realization analysis to {args.output}")


if __name__ == "__main__":
    main()
