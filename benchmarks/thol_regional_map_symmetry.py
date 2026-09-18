"""Exact map symmetries of authenticated retained regional TNFR coefficients.

This observer reuses the prior regional geometry and symmetry owners. It reads
saved evidence only: no trajectory, old admission replay, phase/pressure or
operator kernel is evaluated. Topological symmetry, map equivariance and
symmetry of captured fields are reported separately.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks import thol_regional_input_geometry as geometry  # noqa: E402
from benchmarks import thol_retained_reset_audit as reset  # noqa: E402
from benchmarks.thol_family_closure import _snapshot  # noqa: E402
from benchmarks.thol_full_state_response import _payload  # noqa: E402
from benchmarks.thol_regional_response_criterion import _dataclass_payload  # noqa: E402
from tnfr.mathematics.krylov import exact_rank  # noqa: E402
from tnfr.physics._cycle_algebra import dot  # noqa: E402
from tnfr.physics.equivariance import observe_exact_map_symmetries  # noqa: E402
from tnfr.physics.regional_response import observe_regional_input_geometry  # noqa: E402
from tnfr.operators._reception_kernel import reception_represented_affine_row  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

INPUT_PATH, INPUT_SHA256, INPUT_CLAIM = geometry.INPUT_PATH, geometry.INPUT_SHA256, geometry.INPUT_CLAIM
CAP = 2000


def _admit_common_fields(retained, previous):
    """Bind every displayed field to the same admitted support and coefficients.

    Exact snapshot checks reconstruct cached linear quantities only. The
    captured nonlinear forcing remains measured evidence; no phase kernel or
    historical producer is called and no pressure is inferred from a derivative.
    """
    admission = retained["admission"]
    raw_reference = admission["original_reference"]
    snapshot = _snapshot(raw_reference["source"])
    common = retained["common_coefficients"]
    nodes, n = snapshot.nodes, len(snapshot.nodes)
    metric = previous["metric_weights"]
    equal, vector = geometry._equal, geometry._vector
    matrices = {key: geometry._matrix(common[key], n, n, key) for key in ("A", "S", "T")}
    epi_weight = vector((raw_reference["epi_weight"],), "EPI coefficient")[0]
    dt = vector((common["dt"],), "timestep")[0]
    if epi_weight <= 0 or any(v <= 0 for v in snapshot.capacity):
        raise ValueError("positive EPI coefficient and capacities are required")
    strengths = tuple(sum((w for i, _j, w in snapshot.conductance if i == k), F(0))
                      for k in range(n))
    if any(d <= 0 for d in strengths):
        raise ValueError("positive conductance strengths are required")
    equal(metric, tuple(d/v for d, v in zip(strengths, snapshot.capacity, strict=True)),
          "full H=d/nu metric")
    equal(vector(raw_reference["strengths"], "reference strengths"), strengths, "reference strengths")
    weights = {(i, j): w for i, j, w in snapshot.conductance}
    expected_a = tuple(tuple(v*epi_weight*(F(i == j)-weights.get((i, j), F(0))/strengths[i])
                             for j in range(n)) for i, v in enumerate(snapshot.capacity))
    equal(matrices["A"], expected_a, "nodal diffusion A=diag(nu)*epi_weight*L_rw")
    c = vector(common["c"], "common Reception offset")
    if len(c) != n:
        raise ValueError("Reception offset must match the full node space")
    fields = {"reception_offset_c": c}
    sources = {}
    local_rows = None
    if set(admission["witnesses"]) != set(geometry.WITNESSES):
        raise ValueError("both admitted source witnesses are required")
    for name in geometry.WITNESSES:
        witness = admission["witnesses"][name]
        pair_sources = []
        for side in ("control", "perturbed"):
            row = witness[side]
            label = f"{name}_{side}"
            current = _snapshot(row["snapshot"])
            for key in ("nodes", "conductance", "support_neighbors", "capacity"):
                equal(getattr(current, key), getattr(snapshot, key), f"{label} common {key}")
            observation = row["observation"]
            equal(_snapshot(observation["snapshot"]), current, f"{label} observation snapshot")
            equal(vector((observation["epi_weight"],), "captured EPI coefficient")[0],
                  epi_weight, f"{label} EPI coefficient")
            for key in ("A", "S"):
                equal(geometry._matrix(row[key], n, n, key), matrices[key], f"{label} common {key}")
            equal(vector(row["c"], "captured offset"), c, f"{label} common c")
            saved_rows = row["rows"]
            if len(saved_rows) != n or tuple(item["node"] for item in saved_rows) != nodes:
                raise ValueError("local Reception rows must follow the complete retained node order")
            current_rows = tuple((vector(item["row"], "local Reception row"),
                                  vector((item["offset"],), "local Reception offset")[0])
                                 for item in saved_rows)
            if any(len(coefficients) != n for coefficients, _offset in current_rows):
                raise ValueError("local Reception rows must match the full node space")
            if local_rows is None:
                local_rows = current_rows
            else:
                equal(current_rows, local_rows, f"{label} common local Reception rows")
            forcing = vector(observation["forcing"], "captured non-EPI forcing")
            phase = vector(observation["phase"], "captured phase coordinates")
            source = vector(row["b"], "captured source rate")
            if any(len(values) != n for values in (forcing, phase, source)):
                raise ValueError("captured fields must match the full node space")
            equal(source, tuple(v*f for v, f in zip(current.capacity, forcing, strict=True)),
                  f"{label} source b=nu*forcing")
            equal(vector(row["vectors"]["x0"], "generation EPI"), current.epi,
                  f"{label} generation EPI")
            fields[f"{label}_generation_epi"] = current.epi
            fields[f"{label}_generation_phase_coordinates"] = phase
            pair_sources.append(source)
        equal(pair_sources[0], pair_sources[1], f"{name} paired source cancellation")
        difference = tuple(a-b for a, b in zip(pair_sources[1], pair_sources[0], strict=True))
        equal(vector(witness["paired_source_difference"], "admitted source difference"),
              difference, f"{name} admitted source difference")
        equal(vector(retained["witnesses"][name]["source_difference"], "criterion source difference"),
              difference, f"{name} criterion source difference")
        fields[f"{name}_source_rate_b"] = pair_sources[0]
        fields[f"{name}_held_affine_offset_c_plus_hb"] = tuple(
            offset+dt*b for offset, b in zip(c, pair_sources[0], strict=True))
        sources[name] = pair_sources[0]
    if common["paired_b_cancels"] is not True:
        raise ValueError("retained paired source assertion differs")
    cross_equal = sources["cohort"] == sources["localized"]
    if type(common["cross_experiment_b_equal"]) is not bool or common["cross_experiment_b_equal"] != cross_equal:
        raise ValueError("retained cross-experiment source assertion differs")
    # Recompose only the already recorded chronological single-target rows.
    # This is an exact coefficient identity, not execution of another order.
    composed = [tuple(F(i == j) for j in range(n)) for i in range(n)]
    offset = [F(0)]*n
    for i, (coefficients, addition) in enumerate(local_rows):
        next_row = tuple(sum((coefficient*composed[k][j] for k, coefficient in enumerate(coefficients)), F(0))
                         for j in range(n))
        next_offset = addition+dot(coefficients, tuple(offset))
        composed[i], offset[i] = next_row, next_offset
    equal(tuple(composed), matrices["S"], "ordered local row product S")
    equal(tuple(offset), c, "ordered local row offset c")
    return snapshot, matrices, fields, cross_equal, local_rows


def _local_row_covariance(local_rows, permutations):
    """Relabel the saved local row family without executing another product."""
    checks, preserved = [], []
    size = len(local_rows)
    for ordinal, permutation in enumerate(permutations):
        p = permutation.destinations
        witness = None
        maximum = F(0)
        for i in range(size):
            for j in range(size):
                defect = local_rows[p[i]][0][p[j]]-local_rows[i][0][j]
                if abs(defect) > maximum:
                    maximum, witness = abs(defect), ("coefficient", i, j, defect)
            defect = local_rows[p[i]][1]-local_rows[i][1]
            if abs(defect) > maximum:
                maximum, witness = abs(defect), ("offset", i, defect)
        if maximum == 0:
            preserved.append(ordinal)
        checks.append({"permutation_index": ordinal, "preserved": maximum == 0,
                       "max_abs_defect": maximum, "witness": witness})
    return {"rows": local_rows, "checks": tuple(checks), "group_indices": tuple(preserved),
            "ordered_product_reconstructed": True,
            "scope": "Covariance of saved single-target affine rows, separate from their retained ordered product; "
                     "no simultaneous stage, alternative product execution or complete-runtime equivariance."}


def _fixed_environment(matrices, symmetry, geometries, stages):
    """Apply one group's fixed input basis to already derived regional maps."""
    fixed = {}
    for stage, key in stages:
        old = geometries[stage]
        images = tuple(tuple(dot(row, tuple(dot(mrow, value) for mrow in matrices[key]))
                             for row in old.centering) for value in symmetry.fixed_input_basis)
        image_map = tuple(tuple(column[i] for column in images) for i in range(len(old.region_indices)))
        rank = exact_rank(image_map)
        fixed[stage] = {
            "transition_key": key,
            "centering": old.centering,
            "fixed_input_labels": symmetry.fixed_input_labels,
            "fixed_input_basis": symmetry.fixed_input_basis,
            "fixed_input_map": image_map,
            "fixed_input_dimension": len(symmetry.fixed_input_basis),
            "fixed_input_rank": rank,
            "fixed_input_protected_dimension": old.shape_dimension-rank,
            "unrestricted_input_labels": old.input_labels,
            "unrestricted_input_basis": old.input_basis,
            "unrestricted_input_map": old.input_map,
            "unrestricted_input_rank": old.rank,
            "unrestricted_protected_dimension": old.protected_dimension,
            "same_input_basis_as_unrestricted": symmetry.fixed_input_basis == old.input_basis,
        }
    return fixed


def analyze_retained(retained, *, cap=CAP):
    """Classify complete support and exact-map groups on prior evidence only."""
    previous = geometry.analyze_retained(retained)
    snapshot, matrices, fields, cross_equal, local_rows = _admit_common_fields(retained, previous)
    metric, children = previous["metric_weights"], previous["children"]
    symmetry = observe_exact_map_symmetries(
        snapshot, metric_weights=metric, region=children,
        operators=matrices, fields=fields, cap=cap)
    fixed = _fixed_environment(matrices, symmetry, previous["geometries"], geometry.STAGES)
    return {
        "nodes": snapshot.nodes, "children": children, "region_indices": previous["region_indices"],
        "metric_weights": metric, "snapshot": snapshot,
        "common_coefficients": retained["common_coefficients"],
        "symmetry": symmetry, "common_group_environment": fixed,
        "local_reception_family": _local_row_covariance(local_rows, symmetry.permutations),
        "cross_experiment_source_equal": cross_equal,
        "verified_witnesses": geometry.WITNESSES,
        "native_calls": 0, "kernel_calls": 0, "forcing_capture_calls": 0,
        "new_trajectories": 0, "historical_admission_calls": 0,
        "geometry_evaluations": previous["geometry_evaluations"],
        "input_reachability_certified": False,
        "complete_runtime_equivariance_certified": False,
        "repeated_invariance_certified": False,
        "autonomous_maintenance_certified": False,
        "scope": "Complete finite weighted-support permutations and exact declared-map commutators; "
                 "captured field stabilizers are separate and phase uses the exact saved coordinate chart. "
                 "Fixed-environment inputs refer only to the common operator subgroup and are algebraic, "
                 "not certified reachable interventions. No scalar kernel, trajectory, symmetry restoration, "
                 "emergent shape selection, repeated runtime invariance or physical-particle identification.",
    }


def _admit_snapshot_rows(retained, baseline):
    """Rebuild represented coefficients, without evaluating an EPI update.

    The scalar blend's coefficients depend on the retained mix and explicit
    input support, not on the preceding sequential EPI writes. Clipping and
    binary64 evaluation defects do not belong to this declared linear map.
    """
    snapshot = baseline["snapshot"]
    nodes, size = snapshot.nodes, len(snapshot.nodes)
    local = baseline["local_reception_family"]["rows"]
    admitted = []
    common_interval = None
    for name in geometry.WITNESSES:
        for side in ("control", "perturbed"):
            for i, item in enumerate(retained["admission"]["witnesses"][name][side]["rows"]):
                control, config = item["control"], item["configuration"]
                mix = geometry._vector((control["EN_mix"],), "captured EN mix")[0]
                represented = float(mix)
                if not 0 <= mix <= 1 or F.from_float(represented) != mix:
                    raise ValueError("snapshot comparison requires a represented mix in [0,1]")
                configured = geometry._vector((config["GLYPH_FACTORS"]["EN_mix"],), "configured mix")[0]
                geometry._equal(mix, configured, "captured/configured EN mix")
                neighbors = tuple(control["neighbor_indices"])
                if (any(type(j) is not int or not 0 <= j < size for j in neighbors)
                        or len(set(neighbors)) != len(neighbors)
                        or not neighbors or set(neighbors) != set(snapshot.support_neighbors[i])):
                    raise ValueError("snapshot Reception input support differs")
                lower, upper = geometry._vector((config["EPI_MIN"], config["EPI_MAX"]), "EPI interval")
                if config["CLIP_MODE"] != "hard" or lower >= upper:
                    raise ValueError("snapshot comparison requires the retained hard interval")
                if common_interval is None:
                    common_interval = lower, upper
                geometry._equal((lower, upper), common_interval, "common hard interval")
                row = tuple(F.from_float(value) for value in reception_represented_affine_row(
                    size, i, neighbors, represented))
                geometry._equal(row, local[i][0], "same-snapshot represented Reception row")
                if local[i][1] != 0:
                    raise ValueError("unclipped Reception row must have zero offset")
                admitted.append({"witness": name, "side": side, "node": nodes[i],
                                 "mix": mix, "neighbor_indices": neighbors,
                                 "hard_interval": (lower, upper), "represented_row_matches": True})
    return tuple(admitted)


def analyze_snapshot_comparison(retained, *, cap=CAP):
    """Compare declared sequential and simultaneous EN maps on identical data.

    This is an offline coefficient comparison. It neither reconstructs a live
    graph nor claims that captured sequential defects apply to the new map.
    """
    result = analyze_retained(retained, cap=cap)
    admission = _admit_snapshot_rows(retained, result)
    snapshot, metric = result["snapshot"], result["metric_weights"]
    local = result["local_reception_family"]["rows"]
    j = tuple(row for row, _offset in local)
    a = dict(result["symmetry"].operators)["A"]
    dt = geometry._vector((retained["common_coefficients"]["dt"],), "timestep")[0]
    u = tuple(tuple(x-dt*y for x, y in zip(jr, ar, strict=True)) for jr, ar in zip(j, a, strict=True))
    matrices = {"A": a, "J": j, "U": u}
    fields = dict(result["symmetry"].fields)
    region = result["region_indices"]
    mass = sum(metric[i] for i in region)
    for name in geometry.WITNESSES:
        difference = tuple(y-x for x, y in zip(
            fields[f"{name}_control_generation_epi"],
            fields[f"{name}_perturbed_generation_epi"], strict=True))
        regional_mean = sum(metric[i]*difference[i] for i in region)/mass
        fields[f"{name}_paired_environment"] = tuple(
            regional_mean if i in region else value for i, value in enumerate(difference))
    symmetry = observe_exact_map_symmetries(
        snapshot, metric_weights=metric, region=result["children"],
        operators=matrices, fields=fields, cap=cap)
    stages = (("reception", "J"), ("held_pressure_interval", "U"))
    geometries = {stage: observe_regional_input_geometry(matrices[key], metric, result["region_indices"])
                  for stage, key in stages}
    fixed = _fixed_environment(matrices, symmetry, geometries, stages)
    contrasts = {}
    for stage, _key in stages:
        contrasts[stage] = {
            "sequential_mean_contrast_image": tuple(row[0] for row in result["common_group_environment"][stage]["unrestricted_input_map"]),
            "snapshot_mean_contrast_image": tuple(row[0] for row in geometries[stage].input_map),
            "unrestricted_rank_change": geometries[stage].rank-result["common_group_environment"][stage]["unrestricted_input_rank"],
        }
    result["snapshot_comparison"] = {
        "symmetry": symmetry, "row_admission": admission,
        "unrestricted_geometries": geometries, "common_group_environment": fixed,
        "contrasts": contrasts,
        "paired_environment_fixed": {
            name: set(symmetry.common_group_indices).issubset(
                dict(symmetry.field_group_indices)[f"{name}_paired_environment"])
            for name in geometry.WITNESSES},
        "runtime_admission": {
            "status": "not_certified", "stage_calls": 0,
            "reason": "Coefficient evidence is not a complete graph-owned stage checkpoint; no runtime state/history/grammar restoration is performed.",
            "sequential_defects_transferred": False,
            "required_evidence": ("node EPI kinds and complete node attributes", "graph configuration and callbacks",
                                  "operator/grammar and source histories", "monitors, RNG and cache ownership",
                                  "same-snapshot scalar rounding/clipping and auxiliary writes"),
        },
        "scope": "Declared unclipped same-snapshot coefficients only. Fixed-input protection assumes orbit-invariant environmental inputs; captured fields have their separately reported stabilizers. No source/state symmetrization or complete-runtime equivariance.",
    }
    result.update(coefficient_builder_calls=len(admission), kernel_calls=len(admission),
                  scalar_evolution_kernel_calls=0, geometry_evaluations=4)
    result["scope"] = "Sequential/snapshot declared-map comparison; kernel_calls counts only coefficient builders. No graph evolution, source replacement, native trajectory or full-state symmetry claim."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--expected-sha256", default=INPUT_SHA256)
    parser.add_argument("--snapshot-comparison", action="store_true",
                        help="Compare the existing immutable-snapshot EN coefficient semantics")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mode = "snapshot_reception_comparison" if args.snapshot_comparison else "regional_map_symmetry"
    if args.output is None:
        args.output = ROOT / f"artifacts/research/thol_{mode}_2026_09_18.json"
    if args.output.resolve() == args.input.resolve():
        raise ValueError("output must not overwrite retained input")
    scope = ("src/tnfr", "benchmarks")
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id=("O3.a-snapshot-reception-comparison" if args.snapshot_comparison else "O3.a-regional-map-symmetry"), git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version()},
        graph_construction="Detached weighted-support graph from authenticated original regional coefficients",
        capacity_specification="Retained positive capacities and original full H=d/nu",
        solver="Exact finite permutation, commutator and fixed-input arithmetic; no evolution",
        timestep=None, seed=None, result_status=ClaimStatus.DERIVED,
        operator_sequence=("reception",),
        telemetry=("complete support group", "map subgroup", "captured field stabilizers", "fixed-environment image"),
        controls=("authenticated prior claim", "common sources and coefficients checked",
                  "no graph evolution; coefficient builders only" if args.snapshot_comparison else "no runtime or kernel calls"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    retained, binding = reset._load(args.input, args.expected_sha256, INPUT_CLAIM)
    result = analyze_snapshot_comparison(retained) if args.snapshot_comparison else analyze_retained(retained)
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != args.expected_sha256:
        raise RuntimeError("retained input changed during symmetry analysis")
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during retained symmetry analysis")
    report = {"manifest": manifest.to_dict(), "source_scope": scope,
              "historical_inputs": {"criterion": binding}, **result}
    rendered = json.dumps(_payload(report), default=_dataclass_payload, indent=2, allow_nan=False)+"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(args.output, lambda stream: stream.write(rendered))
    print(f"Wrote {mode} to {args.output}")


if __name__ == "__main__":
    main()
