"""Five coupled mathematical audits of one retained C6 carried prefix.

The B31 history is revalidated against its complete retained source chain.
This report reuses its eighteen transitions; it creates no new trajectory.
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

from benchmarks.c6_winding_carry_itinerary import SOURCE_SCOPE, _state  # noqa: E402
from benchmarks.c6_winding_defect_budget import _number  # noqa: E402
from benchmarks.c6_winding_pressure_sign import CAPACITY, STEP, _canonical  # noqa: E402
from benchmarks.c6_winding_pressure_stencil import analyze_c6_winding_pressure_stencil  # noqa: E402
from benchmarks.c6_winding_rounding_cells import _represented_scalar  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.dynamics._euler_kernel import NodalRemainderStep  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics._cycle_algebra import laplacian_action  # noqa: E402
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile, observe_c6_carried_profile_step  # noqa: E402
from tnfr.physics.c6_carried_tube import (  # noqa: E402
    derive_c6_carried_band_horizon, derive_c6_carried_contraction, derive_c6_carried_tube,
    observe_c6_carried_cut_exclusion,
)
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice, observe_c6_pressure_lattice  # noqa: E402
from tnfr.physics.forced_support import observe_forced_support_pattern  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402

INPUT_NAMES = (
    "c6_winding_pressure_stencil.json", "c6_winding_pressure_levels.json",
    "c6_winding_pressure_repayment.json", "c6_winding_pressure_sign.json",
    "c6_winding_carry_itinerary.json", "c6_winding_pressure_lattice.json",
)
INPUT_KEYS = ("input_evidence", "parent_input_evidence", "previous_input_evidence",
              "earlier_input_evidence", "ancestor_input_evidence", "source_input_evidence")
LINEAGE_KEYS = (
    ("input_evidence", "parent_input_evidence", "previous_input_evidence", "earlier_input_evidence", "source_input_evidence"),
    ("input_evidence", "parent_input_evidence", "previous_input_evidence", "source_input_evidence"),
    ("input_evidence", "parent_input_evidence", "source_input_evidence"),
    ("input_evidence", "source_input_evidence"), ("input_evidence",),
)


def _check_lineage(producers, hashes=None):
    """Validate the declared historical DAG without inferring scientific results."""
    if len(producers) != 6 or hashes is not None and len(hashes) != 6:
        raise ValueError("the coupled audit requires the six B31 through B26 reports")
    for i, keys in enumerate(LINEAGE_KEYS):
        for key, target in zip(keys, range(i + 1, 6), strict=True):
            evidence, producer = producers[i][key], producers[target]
            if (_canonical(evidence["producer_manifest"]) != _canonical(producer["manifest"])
                    or tuple(evidence["producer_source_scope"]) != tuple(producer["source_scope"])):
                raise ValueError("the retained lineage metadata differs from the supplied historical producer")
            if hashes is not None and evidence["sha256"] != hashes[target]:
                raise ValueError("the retained lineage hashes do not identify the supplied historical bytes")


def _verified_inputs(producers):
    _check_lineage(producers)
    parent = producers[0]
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if manifest.claim_id != "O3.a-C6-carried-pressure-stencil" or tuple(parent["source_scope"]) != SOURCE_SCOPE:
        raise ValueError("input must be the retained B31 pressure-stencil report")
    rebuilt = analyze_c6_winding_pressure_stencil(*producers[1:])
    if any(key not in parent or _canonical(parent[key]) != _canonical(value) for key, value in rebuilt.items()):
        raise ValueError("the retained B31 report differs from its complete shared-owner replay")
    initial = _state(parent["source"]["inherited_state"])
    endpoint = _state(parent["continuation"]["endpoint"])
    source = producers[-1]["lattice_reference"]["source"]
    reference = derive_c6_pressure_lattice(
        phase=tuple(rebuilt["source"]["phase"]), epi_weight=_represented_scalar(source["epi_weight"]),
        phase_weight=_represented_scalar(source["phase_weight"]), epi_lower=.375, epi_upper=.625,
    )
    return rebuilt, reference, initial, endpoint


def _retained_rows(parent, initial, endpoint):
    rows = parent["continuation"]["steps"]
    if type(parent["continuation"]["step_count"]) is not int or parent["continuation"]["step_count"] != len(rows) or len(rows) != 18:
        raise ValueError("the selected B31 prefix must contain its eighteen retained transitions")
    previous = initial
    result = []
    for ordinal, row in enumerate(rows, 1):
        before, after = _state(row["before"]), _state(row["after"])
        timestep = _represented_scalar(row["timestep"])
        capacity = tuple(_represented_scalar(value) for value in row["capacity"])
        pressure = tuple(_represented_scalar(value) for value in row["pressure"])
        if before != previous or timestep != STEP or capacity != CAPACITY:
            raise ValueError("retained states, step duration or capacity break the inherited canonical prefix")
        step = NodalRemainderStep(
            before, after, timestep, capacity, pressure,
            *(tuple(_number(value) for value in row[key]) for key in (
                "exact_increment", "visible_increment", "carry_transfer", "nodal_balance_residual",
            )),
        )
        result.append((ordinal, step))
        previous = after
    if previous != endpoint:
        raise ValueError("the retained prefix does not end at its declared carried endpoint")
    return tuple(result)


def _step_record(observed):
    data = asdict(observed)
    data.pop("profile")
    data["pressure_observation"].pop("reference")
    # The shared observer validates the complete readout; retain the identities
    # relevant to this decomposition without eighteen repeated graph matrices.
    readout = observed.readout
    data["readout"] = {
        "readout_shift": readout.readout_shift,
        "mean_nodal_readout_shift": readout.mean_nodal_readout_shift,
        "pressure_identity_residual": readout.pressure_identity_residual,
        "stored_minus_reconstructed_reference": readout.stored_minus_reconstructed_reference,
    }
    data["graph_provenance_certified"] = observed.graph_provenance_certified
    return data


def _energy(values):
    return sum((value * value for value in values), Fraction(0))


def analyze_c6_winding_coupled_budget(parent, previous, ancestor, earlier, origin, source_parent):
    """Complete five exact audits using only the revalidated eighteen-step prefix."""
    producers = (parent, previous, ancestor, earlier, origin, source_parent)
    rebuilt, lattice, initial, endpoint = _verified_inputs(producers)
    profile = derive_c6_carried_profile(lattice)
    balance = profile.forced_balance
    current_visible = observe_forced_support_pattern(balance, nodes=balance.source.nodes, epi=tuple(map(Fraction, endpoint.epi)))
    current_exact = observe_forced_support_pattern(balance, nodes=balance.source.nodes, epi=endpoint.exact_epi)
    current_pressure = observe_c6_pressure_lattice(lattice, epi=endpoint.epi)
    if _canonical(current_pressure.pressure) != _canonical(rebuilt["continuation"]["refreshed_pressure"]["pressure"]):
        raise ValueError("the retained current pressure differs from its fresh shared CPU realization")
    node, cut = 4, lattice.rows[4].nonpositive_max_index
    cut_threshold = -lattice.gradient_quantum * cut - laplacian_action(balance.relative_profile)[node]
    current_laplacian_error = laplacian_action(current_visible.relative_error)[node]
    observations = tuple(observe_c6_carried_profile_step(profile, step=step)
                         for _, step in _retained_rows(parent, initial, endpoint))
    contraction = derive_c6_carried_contraction(profile, timestep=STEP)
    tube = derive_c6_carried_tube(profile, state=initial, timestep=STEP)
    band = derive_c6_carried_band_horizon(tube)
    cut_assessment = observe_c6_carried_cut_exclusion(tube, node=node)
    prefixes = []
    mean_source = mean_rounding = mean_carry = mean_change = Fraction(0)
    energy_envelope = tube.initial_energy
    for ordinal, observed in enumerate(observations, 1):
        pressure = observed.pressure_observation
        homogeneous = tuple(e - contraction.step_factor * gradient
                            for e, gradient in zip(observed.error_before, laplacian_action(observed.error_before), strict=True))
        energy_before, energy_after = _energy(observed.error_before), _energy(observed.error_after)
        homogeneous_energy = _energy(homogeneous)
        energy_envelope = contraction.norm_factor * energy_envelope + (1 - contraction.norm_factor) * tube.energy_floor
        norm_checks = {
            "carry": max(map(abs, observed.step.before.remainder)) <= tube.carry_bound,
            "product_rounding": max(map(abs, pressure.epi_reduction_error)) <= tube.product_error_bound,
            "assembly_rounding": max(map(abs, pressure.assembly_error)) <= tube.assembly_error_bound,
            "combined_rounding": max(map(abs, observed.rounding_defect)) <= tube.rounding_bound,
            "forcing_component": max(map(abs, observed.forcing_defect)) <= tube.forcing_component_bound,
            "centered_forcing": _energy(observed.centered_forcing_defect) <= tube.centered_forcing_norm_squared_bound,
            "homogeneous_contraction": homogeneous_energy <= contraction.energy_factor * energy_before,
            "energy_before": energy_before <= tube.energy_bound, "energy_after": energy_after <= tube.energy_bound,
            "finite_energy_envelope": energy_after <= energy_envelope,
            "mean_increment": abs(observed.mean_change) <= tube.mean_increment_bound,
        }
        if not all(norm_checks.values()):
            raise RuntimeError("a retained canonical step violates a derived uniform coupled bound")
        mean_source += observed.mean_source_contribution
        mean_rounding += observed.mean_rounding_contribution
        mean_carry += observed.mean_carry_contribution
        mean_change += observed.mean_change
        actual_mean_change = sum((b - a for a, b in zip(initial.exact_epi, observed.step.after.exact_epi, strict=True)), Fraction(0)) / 6
        residual = actual_mean_change - mean_source - mean_rounding - mean_carry
        if residual or actual_mean_change != mean_change or mean_carry:
            raise RuntimeError("the retained mean prefix lost its exact signed decomposition")
        prefixes.append({
            "ordinal": ordinal, "energy_before": energy_before, "energy_after": energy_after,
            "homogeneous_energy": homogeneous_energy,
            "homogeneous_energy_bound": contraction.energy_factor * energy_before,
            "finite_energy_envelope": energy_envelope,
            "uniform_checks": norm_checks, "mean_source_prefix": mean_source,
            "mean_rounding_prefix": mean_rounding, "mean_carry_prefix": mean_carry,
            "mean_change": mean_change, "mean_identity_residual": residual,
            "absolute_mean_prefix_bound": ordinal * tube.mean_increment_bound,
        })
    contraction_record = asdict(contraction)
    contraction_record.pop("profile")
    tube_record = asdict(tube)
    tube_record.pop("contraction")
    tube_record.update(infinite_mean_control_certified=tube.infinite_mean_control_certified,
                       future_runtime_certified=tube.future_runtime_certified)
    band_record = asdict(band)
    band_record.pop("tube")
    band_record.update(actual_band_exit_certified=band.actual_band_exit_certified,
                       future_runtime_certified=band.future_runtime_certified)
    cut_record = asdict(cut_assessment)
    cut_record.pop("tube")
    cut_record.update(cut_reachability_certified=cut_assessment.cut_reachability_certified,
                      conclusion="cut_excluded_under_tube_premises" if cut_assessment.cut_excluded else "inconclusive")
    pressure_record = asdict(current_pressure)
    pressure_record.pop("reference")
    return {
        "source": {
            "retained_B31_report_replayed": True, "retained_step_count": len(observations),
            "envelope_initial_location": "B31.source.inherited_state (B30 endpoint)",
            "current_location": "B31.continuation.endpoint", "initial_state": asdict(initial),
            "endpoint_state": asdict(endpoint), "phase": lattice.source.phase,
        },
        "B32_profile": {
            "reference": asdict(profile), "current_visible_pattern": asdict(current_visible),
            "current_reconstructed_pattern": asdict(current_exact), "current_pressure": pressure_record,
            "node4_nonpositive_cut": {
                "current_gradient_index": current_pressure.gradient_indices[node], "nonpositive_max_index": cut,
                "necessary_gradient_change_upper_bound": cut - current_pressure.gradient_indices[node],
                "current_visible_laplacian_error": current_laplacian_error,
                "required_visible_laplacian_error": cut_threshold,
                "required_laplacian_error_increase": cut_threshold - current_laplacian_error,
            },
            "scope": "A centered relative profile with separate mean drift; no zero-pressure equilibrium is inferred",
        },
        "B33_retained_steps": {
            "observations": tuple(_step_record(observed) for observed in observations),
            "all_recurrence_residuals_zero": all(not any(observed.recurrence_residual) for observed in observations),
            "all_mean_residuals_zero": all(observed.mean_identity_residual == 0 for observed in observations),
            "all_carry_means_zero": all(observed.mean_carry_contribution == 0 for observed in observations),
            "scope": "Fresh CPU pressure and shared carried-step replay on retained states; no new trajectory or graph seal",
        },
        "B34_contraction": contraction_record,
        "B35_uniform_tube": {
            "reference": tube_record,
            "node4_cut_assessment": cut_record,
            "scope": "Uniform arithmetic/carry bounds over the declared numerical slab, not measured-maxima extrapolation",
        },
        "B36_prefix_and_mean": {
            "prefixes": tuple(prefixes), "all_uniform_checks_pass": all(all(row["uniform_checks"].values()) for row in prefixes),
            "mean_source_sum": mean_source, "mean_rounding_sum": mean_rounding, "mean_carry_sum": mean_carry,
            "mean_change": mean_change, "mean_identity_residual": mean_change - mean_source - mean_rounding - mean_carry,
            "finite_band_horizon": band_record,
            "retained_prefix_within_band_horizon": band.unbounded_conditional_prefix
            or band.maximum_steps is not None and len(observations) <= band.maximum_steps,
            "scope": (
                "The observed finite signed mean budget is separate from the analytically bounded conditional "
                "future numerical prefix. Its derived horizon is not executed, does not certify live grammar "
                "or fixed future phases, and a failed sufficient bound does not establish actual band exit."
            ),
        },
        "runtime_executed": False, "new_graph_trajectories": 0, "new_trajectory_steps": 0,
        "live_provenance_certified": False, "original_tail_reachability_certified": False,
        "infinite_band_invariance_certified": False, "future_pressure_sign_exit_certified": False,
        "full_runtime_stability_certified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = ("input", "parent-input", "previous-input", "earlier-input", "ancestor-input", "source-input")
    for option, name in zip(options, INPUT_NAMES, strict=True):
        parser.add_argument("--" + option, type=Path, default=ROOT / "artifacts/research" / name)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/c6_winding_coupled_budget.json")
    args = parser.parse_args()
    paths = tuple(getattr(args, name.replace("-", "_")) for name in options)
    if args.output.resolve() in tuple(path.resolve() for path in paths):
        raise ValueError("derived output must not overwrite a historical input")
    raw = tuple(path.read_bytes() for path in paths)
    producers = tuple(json.loads(value) for value in raw)
    hashes = tuple(hashlib.sha256(value).hexdigest() for value in raw)
    _check_lineage(producers, hashes)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_coupled_budget(*producers)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-coupled-carried-budget", git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Revalidated B31 eighteen-transition carried C6 prefix and its complete retained source chain",
        capacity_specification="Unit C6 capacity/support, inherited fixed phases/default weights and slab [3/8,5/8]",
        solver="Exact centered profile, retained shared-step identities, contraction, uniform defect tube and finite band budget",
        result_status=ClaimStatus.DERIVED,
        telemetry=("centered Poisson profile", "canonical-pressure and carried-step decomposition",
                   "exact centered contraction", "uniform rounding/carry bounds", "finite prefix and band bootstrap"),
        controls=("eighteen retained transitions only", "no new trajectory", "no parameter changes",
                  "conditional numerical domain bounds distinct from full-runtime admission"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance or tuple(path.read_bytes() for path in paths) != raw:
        raise RuntimeError("analysis source or historical input changed during the coupled-budget audit")
    report.update(manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE)
    for key, path, digest, producer in zip(INPUT_KEYS, paths, hashes, producers, strict=True):
        report[key] = {"path": str(path), "sha256": digest, "producer_manifest": producer["manifest"],
                       "producer_source_scope": producer["source_scope"]}
    for key in LINEAGE_KEYS[0]:
        report["input_evidence"]["producer_" + key] = producers[0][key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote coupled C6 carried-budget audit to {args.output}")


if __name__ == "__main__":
    main()
