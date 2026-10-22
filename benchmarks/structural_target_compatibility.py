"""Exact Q3 target/channel analysis of retained U/P/F data; no new trajectories.

The input file is content-addressed and its producer manifest stays historical.
This observer validates detached arithmetic, not the provenance of a live graph.
Generate a missing input with structural_perturbation_response.py first.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.physics._cycle_algebra import dot  # noqa: E402
from tnfr.physics.forced_support import (  # noqa: E402
    derive_forced_support_balance,
    observe_forced_support_target,
)
from tnfr.physics.forcing_realization import (  # noqa: E402
    NonEpiForcingObservation,
    decompose_non_epi_forcing,
)
from tnfr.physics.support_transport import SupportTransportSnapshot  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)

SOURCE_SCOPE = (
    "src/tnfr",
    "benchmarks/capacity_localization.py",
    "benchmarks/thol_pressure_feedback.py",
    "benchmarks/structural_target_compatibility.py",
)


def _snapshot(data):
    """Decode numeric snapshot fields while preserving literal ordered node IDs."""
    values = {}
    for name, value in data.items():
        if name == "nodes":
            values[name] = tuple(value)
        elif name == "conductance":
            values[name] = tuple((i, j, Fraction(w)) for i, j, w in value)
        elif name == "support_neighbors":
            values[name] = tuple(tuple(row) for row in value)
        else:
            values[name] = (
                tuple(map(Fraction, value))
                if isinstance(value, list)
                else Fraction(value)
            )
    return SupportTransportSnapshot(**values)


def _reference(data):
    return derive_forced_support_balance(
        _snapshot(data["source"]),
        epi_weight=Fraction(data["epi_weight"]),
        forcing=tuple(map(Fraction, data["forcing"])),
    )


def _capture(data):
    values = {}
    for name, value in data.items():
        if name == "snapshot":
            values[name] = _snapshot(value)
        elif name == "normalized_weights":
            values[name] = tuple((key, Fraction(weight)) for key, weight in value)
        else:
            values[name] = (
                tuple(map(Fraction, value))
                if isinstance(value, list)
                else Fraction(value)
            )
    return NonEpiForcingObservation(**values)


def analyze_target_case(case):
    """Use actual post-event EPI and held channel coefficients with distinct roles."""
    if case["status"] != "measured":
        raise ValueError("target analysis requires a completed measured branch")
    target, current = _reference(case["original_target"]), _reference(
        case["postevent_reference"]
    )
    if case["events"]:
        event = case["events"][-1]
        capture = _capture(event["after_forcing_capture"])
        actual = event["after_refresh"]["state"]
        channel_time = actual["time"]
    else:
        # U has no extra event. Its first captured interval has the same held
        # phase/capacity/support inputs; its different EPI is not substituted.
        segment = case["segments"][0]
        capture = _capture(segment["forcing_capture"])
        actual = case["initial"]["state"]
        channel_time = segment["after_refresh"]["time"]
    phase = tuple(map(Fraction, actual["phase"]))
    if tuple(actual["nodes"]) != current.source.nodes or any(
        tuple(map(Fraction, actual[name])) != getattr(current.source, field)
        for name, field in (
            ("epi", "epi"),
            ("capacity", "capacity"),
            ("pressure", "stored_pressure"),
        )
    ):
        raise ValueError("reference source does not match the recorded actual endpoint")
    if any(
        getattr(capture.snapshot, name) != getattr(current.source, name)
        for name in ("nodes", "conductance", "support_neighbors", "capacity")
    ):
        raise ValueError("channel capture differs from the held post-event model")
    if (
        capture.phase != phase
        or capture.forcing != current.forcing
        or capture.epi_weight != current.epi_weight
    ):
        raise ValueError("channel capture does not match the held phase/forcing inputs")
    observation = observe_forced_support_target(
        target,
        current,
        current.source,
        forcing_components=decompose_non_epi_forcing(capture),
    )
    recorded_limit = Fraction(
        case["conditional_fixed_model_limit"]["pattern"]["error_variance"]
    )
    if observation.limiting_pattern.error_variance != recorded_limit:
        raise ValueError("retained conditional limit disagrees with the rebuilt model")
    return observation, {
        "case": case["case"],
        "observation": asdict(observation),
        "state_time": actual["time"],
        "channel_capture_time": channel_time,
        "channel_capture": asdict(capture),
        "scope": (
            "Actual post-event EPI and stored pressure with matched held coefficients. "
            "The channel capture's kernel defect belongs only to its own EPI; it "
            "is not transferred to target EPI or to the post-event state"
        ),
    }


def compare_target_channels(before, after):
    """Signed midpoint allocation of a residual-energy change, not causal isolation."""
    if before.target_reference != after.target_reference:
        raise ValueError("channel comparison requires one identical original target")
    old, new = dict(before.projected_rate_channels), dict(after.projected_rate_channels)
    if tuple(old) != tuple(new):
        raise ValueError("channel comparisons require the same channel order")
    metric = before.target_reference.metric_weights
    midpoint = tuple(
        (a + b) / 2
        for a, b in zip(
            before.compatibility_residual,
            after.compatibility_residual,
            strict=True,
        )
    )
    contributions = tuple(
        (
            name,
            dot(
                metric,
                tuple(
                    mid * (b - a)
                    for mid, a, b in zip(midpoint, old[name], new[name], strict=True)
                ),
            ),
        )
        for name in old
    )
    change = after.compatibility_energy - before.compatibility_energy
    residual = change - sum((value for _, value in contributions), Fraction(0))
    if residual:
        raise RuntimeError("signed target-channel allocation lost its identity")
    return {
        "channel_contributions": contributions,
        "compatibility_energy_change": change,
        "identity_residual": residual,
        "scope": (
            "Exact symmetric algebraic allocation of the observed residual-energy "
            "change. Channels interact; these are not separately executed ablations"
        ),
    }


def analyze_target_report(parent):
    if tuple(case["case"] for case in parent["cases"]) != ("U", "P", "F"):
        raise ValueError("target analysis requires ordered U/P/F branches")
    observations, cases = zip(*(analyze_target_case(case) for case in parent["cases"]))
    return {
        "cases": cases,
        "perturbation_channel_change": compare_target_channels(
            observations[0], observations[1]
        ),
        "feedback_channel_change": compare_target_channels(
            observations[1], observations[2]
        ),
        "runtime_executed": False,
        "scope": "Exact held-model analysis of supplied historical captures; no live causal seal",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts/research/structural_perturbation_response.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/structural_target_compatibility.json",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("the derived report must not overwrite its input evidence")
    source_bytes = args.input.read_bytes()
    parent = json.loads(source_bytes)
    parent_manifest = CoreExperimentManifest(**parent["manifest"])
    parent_manifest.validate_for_admission()
    # Producer metadata cannot choose or omit this consumer's implementation.
    scope = SOURCE_SCOPE
    provenance = current_git_source_provenance(ROOT, scope)
    report = analyze_target_report(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-fixed-target-compatibility-channels",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version()},
        graph_construction="Detached retained U/P/F post-event snapshots; no graph evolution",
        capacity_specification="Exact represented capacities from input; no new choices",
        solver="Offline exact rational identities; no time integration",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "projected target rate",
            "signed channel Gram terms",
            "old-target energy derivative",
            "fixed-model profile mismatch",
        ),
        controls=(
            "independent P2/P3 rational tests",
            "distinct input and analysis provenance",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("analysis source changed during the observation")
    if args.input.read_bytes() != source_bytes:
        raise RuntimeError("input evidence changed during the observation")
    report.update(
        manifest=manifest.to_dict(),
        source_scope=scope,
        input_evidence={
            "path": str(args.input),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "producer_manifest": parent["manifest"],
            "producer_source_scope": parent["source_scope"],
            "scope": "Retained producer declaration; not relabeled as a current execution",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for case in report["cases"]:
        result = case["observation"]
        print(
            case["case"],
            "target compatible:",
            result["target_compatible"],
            "residual energy:",
            float(result["compatibility_energy"]),
            "current Rdot:",
            float(result["model_energy_rate"]),
        )
    print(
        "Feedback channel allocation:",
        {
            name: float(value)
            for name, value in report["feedback_channel_change"][
                "channel_contributions"
            ]
        },
    )
    print(f"Wrote detached target analysis to {args.output}")


if __name__ == "__main__":
    main()
