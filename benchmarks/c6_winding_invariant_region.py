"""Audit indefinite confinement candidates for the unchanged carried C6 map.

The starting state is the fully replayed B47 endpoint. Candidate coordinates
come from adjacent binary64 values around its exact forced profile. Universal
set inclusion, finite exclusion and unfinished search have different outcomes.
No trajectory is advanced to search for recurrence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.c6_winding_carry_itinerary import SOURCE_SCOPE  # noqa: E402
from benchmarks.c6_winding_relay_handoff import (  # noqa: E402
    HISTORICAL_NAME, INPUT_NAME as RELAY_NAME, replay_c6_relay_handoff_evidence,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics._exact_linear_algebra import exact_matrix_inverse  # noqa: E402
from tnfr.physics.c6_carried_viability import derive_c6_carried_viability  # noqa: E402
from tnfr.physics.c6_pressure_lattice import _observe_rebuilt_c6_pressure_lattice  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402

INPUT_NAME = "c6_winding_relay_handoff.json"


def _profile_cube(profile, state):
    """Adjacent represented brackets are a candidate, not a trapping premise."""
    mean = sum(state.exact_epi, F(0)) / 6
    targets = tuple(mean + z for z in profile.forced_balance.relative_profile)
    pairs = []
    for value in targets:
        rounded = float(value)
        if F(rounded) == value:
            raise ValueError("the selected profile lies on a represented breakpoint")
        lower = math.nextafter(rounded, -math.inf) if F(rounded) > value else rounded
        upper = math.nextafter(rounded, math.inf) if F(rounded) < value else rounded
        if not state.epi_lower <= lower < upper <= state.epi_upper:
            raise ValueError("the profile candidate must remain in the retained band")
        pairs.append((lower, upper))
    rows = tuple(tuple(pair[int(bool(mask & (1 << i)))] for i, pair in enumerate(pairs))
                 for mask in range(64))
    return targets, tuple(pairs), rows


def _quadratic_control(reference, rows, pairs, timestep):
    """Verify exact obstructions to two specific common-quadratic strategies.

    The finite dual support is a stored candidate certificate. Rational
    inversion and equality checks validate it against every freshly produced
    pressure coefficient; LP tolerance is never a premise of the result.
    Neither obstruction is an instability theorem for the carried orbit.
    """
    areas = tuple(tuple(F(timestep) * F(p) for p in
                        _observe_rebuilt_c6_pressure_lattice(reference, row).pressure) for row in rows)
    unit = F(1, math.lcm(*(value.denominator for row in areas for value in row)))
    vectors = tuple(tuple(value / unit for value in row) for row in areas)
    base = vectors[0]
    matrix = tuple(tuple(base[i] - vectors[1 << j][i] for j in range(6)) for i in range(6))
    inverse = exact_matrix_inverse(matrix)
    alpha = tuple(sum((a * b for a, b in zip(row, base)), F(0)) for row in inverse)
    metric = [F(1)]
    for i in range(5):
        metric.append(metric[-1] * matrix[i][i + 1] / matrix[i + 1][i])
    if (any(value <= 0 for value in metric)
            or not all(metric[i] * matrix[i][j] == metric[j] * matrix[j][i]
                       for i in range(6) for j in range(6))):
        raise ValueError("this exact affine control requires its derived diagonal symmetrizer")
    widths = tuple(F(pair[1]) - F(pair[0]) for pair in pairs)
    direction = tuple(min(widths) / value for value in widths)
    image = tuple(sum((a * b for a, b in zip(row, direction)), F(0)) for row in matrix)
    quadratic = sum((v * d * a for v, d, a in zip(direction, metric, image)), F(0))
    if quadratic >= 0:
        raise ValueError("the retained negative quadratic witness does not verify")
    errors = tuple(tuple(row[i] - base[i] + sum(
        (matrix[i][j] for j in range(6) if mask >> j & 1), F(0),
    ) for i in range(6)) for mask, row in enumerate(vectors))

    # A basis for every symmetric H: Q*S*Q^T + Q*a*1^T + 1*a^T*Q^T
    # + K*beta*1*1^T, where Q's columns are e_i-e_5. Congruence shows
    # H>0 implies S>0. The positive dual forces an outward component.
    directions = tuple(tuple(F(int(i == j) - int(i == 5)) for i in range(6)) for j in range(5))
    matrices, trace = [], []
    for i in range(5):
        for j in range(i, 5):
            matrices.append(tuple(tuple(directions[i][a] * directions[j][b]
                                        + (directions[j][a] * directions[i][b] if i != j else F(0))
                                        for b in range(6)) for a in range(6)))
            trace.append(F(i == j))
    for direction in directions:
        matrices.append(tuple(tuple(direction[a] + direction[b] for b in range(6)) for a in range(6)))
        trace.append(F(0))
    trace.append(F(0))
    scale = abs(matrix[1][0])
    support = (0, 2, 14, 62, 186, 187, 246, 255, 258, 259, 261, 263,
               282, 311, 352, 358, 359, 379, 381, 382, 383)
    constraints = []
    for index in support:
        mask, coordinate = divmod(index, 6)
        sign = F(1 if mask >> coordinate & 1 else -1)
        row = vectors[mask]
        constraints.append(tuple(sign * sum((value * p for value, p in zip(mat[coordinate], row)), F(0))
                                 / scale for mat in matrices) + (sign * sum(row, F(0)),))
    system = tuple(tuple(row[column] for row in constraints) + (-trace[column],)
                   for column in range(21)) + ((F(1),) * 21 + (F(0),),)
    solution = tuple(row[-1] for row in exact_matrix_inverse(system))
    weights, multiplier = solution[:-1], solution[-1]
    coefficients = tuple(sum((weight * row[column] for weight, row in zip(weights, constraints)), F(0))
                         for column in range(21))
    if (not all(value > 0 for value in weights) or multiplier <= 0 or sum(weights) != 1
            or coefficients != tuple(multiplier * value for value in trace)):
        raise ValueError("the proposed common-quadratic dual certificate does not verify")
    return {
        "exact_nodal_areas": areas, "increment_unit": unit, "affine_matrix": matrix,
        "affine_equilibrium_switches": alpha, "symmetrizing_diagonal": tuple(metric),
        "negative_direction": tuple(min(widths) / value for value in widths),
        "negative_direction_image": image, "negative_quadratic_value": quadratic,
        "mixed_remainders": errors,
        "mixed_bounds": tuple((min(row[i] for row in errors), max(row[i] for row in errors)) for i in range(6)),
        "positive_dual_support": tuple(divmod(index, 6) for index in support),
        "positive_dual_weights": weights, "positive_trace_multiplier": multiplier,
        "dual_identity_residual": tuple(a - multiplier * b for a, b in zip(coefficients, trace)),
        "all_orthant_inward_common_quadratic_criterion_excluded": True,
        "all_quadratic_invariants_excluded": False, "piecewise_potentials_excluded": False,
        "actual_trajectory_instability_certified": False,
    }


def analyze_c6_winding_invariant_region(
    parent, *, relay_bytes, historical_bytes, max_work_items=500_000, max_boxes=30_000,
):
    """Recheck B47, derive the candidate, then require universal inclusion."""
    profile, state = replay_c6_relay_handoff_evidence(
        parent, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
    )
    targets, pairs, rows = _profile_cube(profile, state)
    h = parent["source"]["timestep"]
    control = _quadratic_control(profile.lattice, rows, pairs, h)
    result = derive_c6_carried_viability(
        profile.lattice, state=state, epi_states=rows, timestep=h,
        max_work_items=max_work_items, max_boxes=max_boxes,
    )
    proof = asdict(result)
    proof.pop("reference")
    proof.update(
        conditional_invariance_certified=result.conditional_invariance_certified,
        conditional_boundedness_certified=result.conditional_boundedness_certified,
        origin_exit_certified=result.origin_exit_certified,
        exclusion_step_bound=result.exclusion_step_bound,
        future_runtime_certified=result.future_runtime_certified,
        asymptotic_convergence_certified=result.asymptotic_convergence_certified,
    )
    return {
        "contract": {
            "question": "Does an exact forward-invariant region contain the unchanged B47 C6 endpoint?",
            "candidate": "Adjacent binary64 brackets of the exact forced profile at the retained mean",
            "proof_rule": "K_next = K intersect F^-1(K), with exact RN ties and the full incoming carry",
            "success_rule": "A complete finite fixed point containing the supplied origin",
            "computational_max_work_items": max_work_items, "computational_max_boxes": max_boxes,
            "resource_limits_are_physical_parameters": False, "pressure_projected_or_carry_reset": False,
        },
        "source": {
            "retained_B47_endpoint": asdict(state), "phase": profile.lattice.source.phase,
            "capacity": (1.,) * 6, "timestep": h,
            "historical_nodal_steps_replayed": 356, "previous_conditional_steps_replayed": 204,
            "serialized_numerical_chain_verified": True, "live_execution_seal_recreated": False,
        },
        "B48_profile_candidate": {
            "exact_targets": targets, "coordinate_pairs": pairs, "visible_rows": rows,
            "actual_endpoint_mask": rows.index(state.epi),
        },
        "B48_common_quadratic_controls": control,
        "B48_exact_correlated_viability": proof,
        "new_conditional_trajectory_steps": 0, "new_live_graph_steps": 0,
        "indefinite_trapping_certified": result.conditional_boundedness_certified,
        "candidate_exit_certified": result.origin_exit_certified,
        "whole_band_exit_certified": False, "future_runtime_certified": False,
        "all_correlated_candidates_excluded": False,
    }


def analyze_c6_winding_forward_envelope(
    parent, *, relay_bytes, historical_bytes, max_intersections=250_000,
):
    """Reuse the B47 source and admit only a universally closed past envelope."""
    from tnfr.physics.c6_carried_viability import derive_c6_carried_forward_envelope

    profile, state = replay_c6_relay_handoff_evidence(
        parent, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
    )
    targets, pairs, rows = _profile_cube(profile, state)
    result = derive_c6_carried_forward_envelope(
        profile.lattice, state=state, epi_states=rows, timestep=parent["source"]["timestep"],
        max_intersections=max_intersections,
    )
    proof = asdict(result)
    proof.pop("reference")
    proof.update(
        conditional_invariance_certified=result.conditional_invariance_certified,
        conditional_boundedness_certified=result.conditional_boundedness_certified,
        future_runtime_certified=result.future_runtime_certified,
        asymptotic_convergence_certified=result.asymptotic_convergence_certified,
    )
    return {
        "contract": {
            "question": "Can a relational envelope of compatible past states close around the retained C6 branch?",
            "candidate": "The unchanged profile cube with pressure-derived pair strips and one zone per RN cell",
            "proof_rule": "R_next is the least per-cell difference-bound hull of F(R) intersected with the protected domain",
            "success_rule": "A nonempty retained envelope has its complete image inside the protected domain, with actual entry verified",
            "computational_max_intersections": max_intersections,
            "resource_limit_is_physical_parameter": False, "pressure_projected_or_carry_reset": False,
        },
        "source": {
            "retained_B47_endpoint": asdict(state), "phase": profile.lattice.source.phase,
            "capacity": (1.,) * 6, "timestep": parent["source"]["timestep"],
            "historical_nodal_steps_replayed": 356, "previous_conditional_steps_replayed": 204,
            "serialized_numerical_chain_verified": True, "live_execution_seal_recreated": False,
        },
        "B49_profile_candidate": {
            "exact_targets": targets, "coordinate_pairs": pairs, "visible_rows": rows,
            "actual_endpoint_mask": rows.index(state.epi),
        },
        "B49_relational_forward_envelope": proof,
        "new_conditional_trajectory_steps": len(result.entry_steps), "new_live_graph_steps": 0,
        "indefinite_trapping_certified": result.conditional_boundedness_certified,
        "whole_band_exit_certified": False, "future_runtime_certified": False,
        "all_correlated_candidates_excluded": False,
    }


def _rebuild_c6_forward_domain(parent, *, relay_bytes, historical_bytes, envelope_bytes):
    """Reconstruct B49 and its clipped inclusion for point and region proofs."""
    from tnfr.physics.c6_carried_viability import (
        C6CarriedForwardZone, _dbm_intersection, _dbm_subset,
    )

    retained = json.loads(envelope_bytes)
    if (retained.get("source_scope") != list(SOURCE_SCOPE)
            or retained.get("manifest", {}).get("claim_id") != "O3.a-C6-carried-forward-envelope"):
        raise ValueError("the predecessor audit requires the retained B49 forward-envelope report")
    fresh = analyze_c6_winding_forward_envelope(
        parent, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
    )
    if any(retained.get(key) != value for key, value in _payload(fresh).items()):
        raise ValueError("the retained B49 envelope differs from its complete canonical reconstruction")
    profile, state = replay_c6_relay_handoff_evidence(
        parent, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
    )
    envelope = fresh["B49_relational_forward_envelope"]
    rows = envelope["epi_states"]
    grid = envelope["grid_quantum"]
    zones = tuple(C6CarriedForwardZone(item["epi"], item["bounds"]) for item in envelope["retained_zones"])
    source_by_row = {zone.epi: zone.bounds for zone in zones}
    protected_by_row = {item["epi"]: item["bounds"] for item in envelope["initial_zones"]}
    # This is clipped inclusion, not a claim that no point can leave the cube.
    # Along any prefix that stays in the cube, protected pairs keep it in D
    # and this identity keeps it in the retained R259 domain containing B47.
    for row, pressure in zip(rows, envelope["pressures"], strict=True):
        added = tuple(F(envelope["timestep"]) * F(p) / grid for p in pressure) + (F(0),)
        if any(value.denominator != 1 for value in added):
            raise RuntimeError("the replayed envelope lost its common increment grid")
        zone = source_by_row[row]
        translated = tuple(tuple(zone[i][j] + int(added[i] - added[j]) for j in range(7)) for i in range(7))
        for target_row, cell in protected_by_row.items():
            if not _dbm_subset(_dbm_intersection(translated, cell), source_by_row[target_row]):
                raise RuntimeError("the retained envelope lost its exact clipped forward inclusion")
    return fresh, profile, state, zones


def analyze_c6_winding_temporal_predecessors(
    parent, *, relay_bytes, historical_bytes, envelope_bytes, witness_bytes,
    max_depth=128, max_row_checks=32_768,
):
    """Audit exact point pasts, separating later absence and initial visits."""
    from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
    from tnfr.physics.c6_carried_viability import derive_c6_carried_predecessors

    fresh, profile, state, zones = _rebuild_c6_forward_domain(
        parent, relay_bytes=relay_bytes, historical_bytes=historical_bytes, envelope_bytes=envelope_bytes,
    )
    validation = json.loads(witness_bytes)
    if validation.get("input_sha256") != hashlib.sha256(envelope_bytes).hexdigest():
        raise ValueError("the selected witnesses do not bind the retained B49 bytes")
    envelope = fresh["B49_relational_forward_envelope"]
    rows, grid, origin = envelope["epi_states"], envelope["grid_quantum"], state.exact_epi
    selected = tuple(item for item in validation["outgoing_slab_sum_projections"]
                     if item.get("hypothetical_original_mean_affine_coset_witness") is not None)
    if not selected:
        raise ValueError("the predecessor audit requires nonempty exact target witnesses")
    records = []
    identities = set()
    for item in selected:
        mask, node, direction = item["mask"], item["node"], item["direction"]
        indices = item["hypothetical_original_mean_affine_coset_witness"]
        if (type(mask) is not int or not 0 <= mask < len(rows)
                or type(node) is not int or not 0 <= node < 6 or direction not in ("lower", "upper")
                or type(indices) is not list or len(indices) != 6 or any(type(n) is not int for n in indices)):
            raise ValueError("each witness requires a valid row, facet and six exact integer indices")
        identity = mask, node, direction, tuple(indices)
        if identity in identities:
            raise ValueError("selected outgoing witnesses must be distinct")
        identities.add(identity)
        exact = tuple(x + grid * n for x, n in zip(origin, indices, strict=True))
        if sum(exact) != sum(origin) or tuple(map(float, exact)) != rows[mask]:
            raise ValueError("the selected point must preserve the exact original mean and its RN row")
        target = NodalRemainderState(
            rows[mask], tuple(x - F(visible) for x, visible in zip(exact, rows[mask], strict=True)),
            state.epi_lower, state.epi_upper,
        )
        outgoing = advance_nodal_remainder(
            target, timestep=envelope["timestep"], capacity=(1.,) * 6,
            pressure=_observe_rebuilt_c6_pressure_lattice(profile.lattice, target.epi).pressure,
        )
        edge = (min if direction == "lower" else max)(row[node] for row in rows)
        leaves_facet = (outgoing.after.epi[node] < edge if direction == "lower"
                        else outgoing.after.epi[node] > edge)
        if not leaves_facet:
            raise ValueError("the selected point does not leave its declared cube facet")
        domains = {}
        for label, domain in (("retained_R259", zones), ("original_cube", None)):
            proof = derive_c6_carried_predecessors(
                profile.lattice, state=state, target=target, epi_states=rows,
                timestep=envelope["timestep"], domain_zones=domain,
                max_depth=max_depth, max_row_checks=max_row_checks,
            )
            payload = asdict(proof)
            payload.pop("reference")
            payload.update(
                completed_depth=proof.completed_depth, frontier_counts=proof.frontier_counts,
                past_exclusion_depth=proof.past_exclusion_depth,
                maximum_compatible_past_depth=proof.maximum_compatible_past_depth,
                origin_reachability_depths=proof.origin_reachability_depths,
                finite_origin_reachability_certified=proof.finite_origin_reachability_certified,
                origin_path_within_domain_excluded=proof.origin_path_within_domain_excluded,
                whole_domain_exclusion_certified=proof.whole_domain_exclusion_certified,
                conditional_boundedness_certified=proof.conditional_boundedness_certified,
                future_runtime_certified=proof.future_runtime_certified,
            )
            domains[label] = payload
        records.append(dict(mask=mask, node=node, direction=direction, grid_indices=tuple(indices),
                            target=asdict(target), outgoing_step=asdict(outgoing), domains=domains,
                            selected_point_cannot_cause_first_cube_exit=(
                                domains["retained_R259"]["origin_path_within_domain_excluded"])))
    return {
        "contract": {
            "question": "Which selected outward points have a compatible exact past in the declared domain?",
            "proof_rule": "P_next consists of all admitted X-h*p(row); every accepted edge is shared-kernel checked",
            "exclusion_scope": "A complete empty depth-n frontier excludes n-step pasts, not earlier visits or whole facets",
            "computational_max_depth": max_depth, "computational_max_row_checks": max_row_checks,
            "resource_limits_are_physical_parameters": False,
        },
        "source": fresh["source"],
        "B50_B49_reconstruction": dict(
            complete_envelope_rebuilt=True, completed_layers=259, outgoing_facets=56,
            clipped_forward_inclusion_verified=True,
            envelope_sha256=hashlib.sha256(envelope_bytes).hexdigest(),
            witness_sha256=hashlib.sha256(witness_bytes).hexdigest(),
        ),
        "B50_selected_point_pasts": tuple(records),
        "selected_points_are_actual_reached_states": False,
        "all_outgoing_facets_excluded": False,
        "new_conditional_trajectory_steps": 0, "new_live_graph_steps": 0,
        "indefinite_trapping_certified": False, "whole_band_exit_certified": False,
        "future_runtime_certified": False,
    }


def _c6_unsafe_slab(bounds, node, direction, lower, upper, shift):
    """Intersect one source zone with a complete original-cube exit condition."""
    from tnfr.physics.c6_carried_viability import _dbm_close

    values = [list(row) for row in bounds]
    if direction == "lower":
        values[node][6] = min(values[node][6], lower[node] - int(shift[node]) - 1)
    elif direction == "upper":
        values[6][node] = min(values[6][node], -(upper[node] - int(shift[node]) + 1))
    else:
        raise ValueError("a first-exit direction must be lower or upper")
    return _dbm_close(values)


def _rebuild_c6_outgoing_regions(parent, *, relay_bytes, historical_bytes, envelope_bytes):
    """Rebuild the fixed domain and its complete original-cube exit cover."""
    from tnfr.physics.c6_carried_viability import (
        C6CarriedForwardZone, _prepare_carried_family, _predecessor_domain,
    )

    fresh, profile, state, zones = _rebuild_c6_forward_domain(
        parent, relay_bytes=relay_bytes, historical_bytes=historical_bytes, envelope_bytes=envelope_bytes,
    )
    envelope = fresh["B49_relational_forward_envelope"]
    grid = envelope["grid_quantum"]
    _ref, origin, h, rows, pressures, areas, cells = _prepare_carried_family(
        profile.lattice, state=state, epi_states=envelope["epi_states"], timestep=envelope["timestep"],
        row_limit=4096, row_limit_label="max_cells",
    )
    # Pair strips can tighten axis projections. Only the complete RN cells
    # identify a first exit from the original cube, so reconstruct them here.
    rn_cells = _predecessor_domain(rows, cells, origin, grid, None)
    lower = tuple(min(-cell.bounds[6][i] for cell in rn_cells) for i in range(6))
    upper = tuple(max(cell.bounds[i][6] for cell in rn_cells) for i in range(6))
    by_row = {zone.epi: zone.bounds for zone in zones}
    facets, targets = [], []
    for mask, (row, area) in enumerate(zip(rows, areas, strict=True)):
        shift = tuple(added / grid for added in area)
        if any(value.denominator != 1 for value in shift):
            raise RuntimeError("a canonical region increment escaped the common grid")
        for node in range(6):
            for direction in ("lower", "upper"):
                slab = _c6_unsafe_slab(by_row[row], node, direction, lower, upper, shift)
                if slab is not None:
                    facets.append(dict(mask=mask, node=node, direction=direction))
                    targets.append((C6CarriedForwardZone(row, slab),))
    geometry = dict(lower_grid=lower, upper_grid=upper, grid_quantum=grid,
                    affine_origin=origin, complete_RN_cells=tuple(asdict(cell) for cell in rn_cells))
    return fresh, profile, state, zones, geometry, tuple(facets), tuple(targets)


def analyze_c6_winding_region_exclusions(
    parent, *, relay_bytes, historical_bytes, envelope_bytes, max_intersections=250_000,
):
    """Cover every outgoing slab and test complete regional predecessor hulls."""
    from tnfr.physics.c6_carried_viability import derive_c6_carried_region_exclusions

    fresh, profile, state, zones, geometry, facets, targets = _rebuild_c6_outgoing_regions(
        parent, relay_bytes=relay_bytes, historical_bytes=historical_bytes, envelope_bytes=envelope_bytes,
    )
    envelope = fresh["B49_relational_forward_envelope"]
    result = derive_c6_carried_region_exclusions(
        profile.lattice, state=state, epi_states=envelope["epi_states"], timestep=envelope["timestep"],
        target_regions=tuple(targets), domain_zones=zones, max_intersections=max_intersections,
    )
    proof = asdict(result)
    proof.pop("reference")
    for payload, query in zip(proof["queries"], result.queries, strict=True):
        payload.update(completed_depth=query.completed_depth, past_exclusion_depth=query.past_exclusion_depth,
                       origin_path_within_domain_excluded=query.origin_path_within_domain_excluded,
                       actual_origin_reachability_certified=query.actual_origin_reachability_certified)
    proof.update(common_grid_relaxes_coordinate_cosets=result.common_grid_relaxes_coordinate_cosets,
                 conditional_invariance_certified=result.conditional_invariance_certified,
                 conditional_boundedness_certified=result.conditional_boundedness_certified,
                 future_runtime_certified=result.future_runtime_certified,
                 asymptotic_convergence_certified=result.asymptotic_convergence_certified)
    groups = {}
    for facet, query in zip(facets, result.queries, strict=True):
        key = facet["node"], facet["direction"]
        group = groups.setdefault(key, dict(node=key[0], direction=key[1], total=0, excluded=0))
        group["total"] += 1
        group["excluded"] += int(query.origin_path_within_domain_excluded)
    complete = bool(facets) and all(query.origin_path_within_domain_excluded for query in result.queries)
    return {
        "contract": {
            "question": "Which complete outward slabs cannot be reached before the first cube exit?",
            "proof_rule": "Per-cell hulls cover complete predecessor regions; empty or exactly stationary layers exclude all origin paths only if every layer excludes origin",
            "scope": "The common grid overapproximates coordinate cosets; origin in a hull is not a reached trajectory",
            "computational_max_intersections": max_intersections, "resource_limit_is_physical_parameter": False,
        },
        "source": fresh["source"],
        "B51_B49_reconstruction": dict(
            complete_envelope_rebuilt=True, clipped_forward_inclusion_verified=True,
            envelope_sha256=hashlib.sha256(envelope_bytes).hexdigest(),
        ),
        "B51_cube_geometry": geometry,
        "B51_outgoing_facets": tuple(facets),
        "B51_region_predecessors": proof,
        "B51_exit_groups": tuple(groups.values()),
        "excluded_first_exit_slabs": sum(query.origin_path_within_domain_excluded for query in result.queries),
        "all_first_exit_slabs_excluded": complete,
        "indefinite_trapping_certified": complete,
        "new_conditional_trajectory_steps": 0, "new_live_graph_steps": 0,
        "whole_band_exit_certified": False, "future_runtime_certified": False,
    }


def _c6_poisson_contrast_weights():
    """Primitive integer zero-mean solution of L_rw*w parallel to e2-e0."""
    laplacian = tuple(tuple(F(int(i == j)) - F(int(j in ((i - 1) % 6, (i + 1) % 6)), 2)
                            for j in range(6)) for i in range(6))
    matrix = laplacian[:5] + ((F(1),) * 6,)
    rhs = tuple(F(int(i == 2) - int(i == 0)) for i in range(5)) + (F(0),)
    inverse = exact_matrix_inverse(matrix)
    rational = tuple(sum(a * b for a, b in zip(row, rhs, strict=True)) for row in inverse)
    denominator = math.lcm(*(value.denominator for value in rational))
    integers = tuple(int(value * denominator) for value in rational)
    divisor = math.gcd(*integers)
    weights = tuple(value // divisor for value in integers)
    image = tuple(sum(a * b for a, b in zip(row, weights, strict=True)) for row in laplacian)
    if sum(weights) or image != tuple(F(3, 2) * (int(i == 2) - int(i == 0)) for i in range(6)):
        raise RuntimeError("the C6 nodal contrast lost its exact Poisson identity")
    return weights


def analyze_c6_winding_excursion_exclusion(
    parent, *, relay_bytes, historical_bytes, envelope_bytes, region_bytes, max_prefix_steps=4096,
):
    """Combine prior regional exclusions with a derived nodal excursion gate."""
    context = _rebuild_c6_outgoing_regions(
        parent, relay_bytes=relay_bytes, historical_bytes=historical_bytes, envelope_bytes=envelope_bytes,
    )
    report, _proved = _analyze_c6_excursion_context(
        context, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
        envelope_bytes=envelope_bytes, region_bytes=region_bytes, max_prefix_steps=max_prefix_steps,
    )
    return report


def _analyze_c6_excursion_context(
    context, *, relay_bytes, historical_bytes, envelope_bytes, region_bytes, max_prefix_steps,
):
    """Reuse one canonical reconstruction while rechecking all B51/B52 premises."""
    from tnfr.physics.c6_carried_excursion import derive_c6_carried_excursion_exclusion
    from tnfr.physics.c6_carried_viability import derive_c6_carried_region_exclusions

    fresh, profile, state, zones, geometry, facets, targets = context
    prior = json.loads(region_bytes)
    if prior.get("manifest", {}).get("claim_id") != "O3.a-C6-carried-region-exclusions":
        raise ValueError("the regional evidence has a different claim contract")
    expected_hashes = tuple(hashlib.sha256(blob).hexdigest() for blob in (
        relay_bytes, historical_bytes, envelope_bytes,
    ))
    prior_inputs = prior.get("input_evidence", ())
    if len(prior_inputs) != 4 or tuple(item["sha256"] for item in prior_inputs[1:]) != expected_hashes:
        raise ValueError("the regional evidence has different historical inputs")
    if (prior.get("source") != _payload(fresh["source"])
            or prior.get("B51_cube_geometry") != _payload(geometry)
            or prior.get("B51_outgoing_facets") != _payload(facets)):
        raise ValueError("the regional evidence does not identify the rebuilt origin and whole exit cover")
    old_queries = prior["B51_region_predecessors"]["queries"]
    if len(old_queries) != len(facets):
        raise ValueError("the regional evidence lost a labeled target query")
    selected = tuple(index for index, query in enumerate(old_queries) if query["origin_path_within_domain_excluded"])
    if not selected or prior["excluded_first_exit_slabs"] != len(selected):
        raise ValueError("the regional evidence has inconsistent positive coverage")
    envelope = fresh["B49_relational_forward_envelope"]
    rechecked = derive_c6_carried_region_exclusions(
        profile.lattice, state=state, epi_states=envelope["epi_states"], timestep=envelope["timestep"],
        target_regions=tuple(targets[index] for index in selected), domain_zones=zones,
    )
    for index, query in zip(selected, rechecked.queries, strict=True):
        public = asdict(query)
        public.update(completed_depth=query.completed_depth, past_exclusion_depth=query.past_exclusion_depth,
                      origin_path_within_domain_excluded=query.origin_path_within_domain_excluded,
                      actual_origin_reachability_certified=query.actual_origin_reachability_certified)
        if not query.origin_path_within_domain_excluded or _payload(public) != old_queries[index]:
            raise ValueError("a claimed prior regional exclusion failed exact reconstruction")
    rows = envelope["epi_states"]
    active_value = min(row[2] for row in rows)
    active = tuple(row for row in rows if row[2] == active_value)
    additional = tuple(index for index, facet in enumerate(facets)
                       if facet["node"] == 2 and facet["direction"] == "lower")
    result = derive_c6_carried_excursion_exclusion(
        profile.lattice, state=state, epi_states=rows, timestep=envelope["timestep"],
        active_epi_states=active, weights=_c6_poisson_contrast_weights(),
        target_regions=tuple(zone for index in additional for zone in targets[index]),
        domain_zones=zones, max_prefix_steps=max_prefix_steps,
    )
    proof = asdict(result)
    proof.pop("reference")
    proof.update(origin_path_within_domain_excluded=result.origin_path_within_domain_excluded,
                 actual_target_reached=result.actual_target_reached,
                 observed_prefix_steps=result.observed_prefix_steps,
                 common_grid_relaxes_coordinate_cosets=result.common_grid_relaxes_coordinate_cosets,
                 conditional_invariance_certified=result.conditional_invariance_certified,
                 conditional_boundedness_certified=result.conditional_boundedness_certified,
                 asymptotic_convergence_certified=result.asymptotic_convergence_certified,
                 future_runtime_certified=result.future_runtime_certified)
    proved = set(selected)
    if result.origin_path_within_domain_excluded:
        proved.update(additional)
    groups = {}
    for index, facet in enumerate(facets):
        key = facet["node"], facet["direction"]
        group = groups.setdefault(key, dict(node=key[0], direction=key[1], total=0, excluded=0))
        group["total"] += 1
        group["excluded"] += int(index in proved)
    return {
        "contract": {
            "question": "Can a derived monotone excursion functional exclude the remaining node-2 lower slabs?",
            "proof_rule": "Strict nodal drift and separated ingress exclude every later visit; only the derived initial prefix is executed",
            "weights_are_proof_coordinates_not_physical_parameters": True,
            "max_prefix_steps": max_prefix_steps,
        },
        "source": fresh["source"],
        "B52_reconstruction": dict(complete_envelope_rebuilt=True, clipped_forward_inclusion_verified=True,
                                   prior_region_sha256=hashlib.sha256(region_bytes).hexdigest(),
                                   prior_positive_queries_reverified=len(selected),
                                   prior_query_verification_intersections=rechecked.intersections),
        "B52_cube_geometry": geometry,
        "B52_proof_coordinate_derivation": dict(
            weights=_c6_poisson_contrast_weights(), zero_mean=True,
            primitive_integer_solution=True, laplacian="unit-C6 random-walk Laplacian",
            exact_laplacian_image=(F(-3, 2), F(0), F(3, 2), F(0), F(0), F(0)),
            binary64_pressure_drift_verified_separately=True,
        ),
        "B52_outgoing_facets": facets,
        "B52_excursion_exclusion": proof,
        "B52_exit_groups": tuple(groups.values()),
        "excluded_first_exit_slabs": len(proved),
        "additional_first_exit_slabs_excluded": len(proved - set(selected)),
        "all_first_exit_slabs_excluded": len(proved) == len(facets),
        "indefinite_trapping_certified": len(proved) == len(facets),
        "primary_proof_origin_remains_B47": True,
        "new_conditional_trajectory_steps": result.observed_prefix_steps,
        "new_live_graph_steps": 0, "whole_band_exit_certified": False, "future_runtime_certified": False,
    }, proved


def _cut_c6_proven_facets(zones, rows, shifts, geometry, facets):
    """Remove only independently excluded whole original-cube exit slabs."""
    from tnfr.physics.c6_carried_viability import C6CarriedForwardZone, _dbm_close

    domain = {zone.epi: zone.bounds for zone in zones}
    for facet in facets:
        mask, node, direction = facet["mask"], facet["node"], facet["direction"]
        row = rows[mask]
        if row not in domain:
            continue
        bounds = [list(values) for values in domain[row]]
        if direction == "lower":
            bound = -(geometry["lower_grid"][node] - shifts[mask][node])
            bounds[6][node] = min(bounds[6][node], bound)
        else:
            bound = geometry["upper_grid"][node] - shifts[mask][node]
            bounds[node][6] = min(bounds[node][6], bound)
        closed = _dbm_close(bounds)
        if closed is None:
            del domain[row]
        else:
            domain[row] = closed
    return tuple(C6CarriedForwardZone(row, domain[row]) for row in rows if row in domain)


def _c6_public_exact_proof(proof):
    public = asdict(proof)
    public.pop("reference")
    for key in (
        "domain_confined_paths_covered", "clipped_forward_inclusion_certified",
        "origin_path_within_domain_excluded", "actual_target_reached", "observed_prefix_steps",
        "common_grid_relaxes_coordinate_cosets", "conditional_invariance_certified",
        "conditional_boundedness_certified", "asymptotic_convergence_certified", "future_runtime_certified",
    ):
        if hasattr(proof, key):
            public[key] = getattr(proof, key)
    return public


def _c6_mode_candidates(raw, facets, rows):
    """Normalize untrusted rational proposals; no stored success flag is read."""
    data = json.loads(raw)
    if type(data) is not dict or data.get("schema") != "c6-mode-excursion-candidates-v1":
        raise ValueError("the mode candidates have an unsupported schema")
    candidates = data.get("candidates")
    if type(candidates) is not list or not 0 < len(candidates) <= len(facets):
        raise ValueError("mode candidates must be a nonempty bounded list")
    output, seen = [], set()
    for item in candidates:
        if type(item) is not dict:
            raise ValueError("each mode candidate must be an object")
        target = item.get("target")
        if (type(target) is not dict or set(target) != {"mask", "node", "direction"}
                or type(target["mask"]) is not int or type(target["node"]) is not int
                or target not in facets):
            raise ValueError("each mode candidate must identify a complete original unsafe slab")
        index = facets.index(target)
        active = item.get("active_masks")
        if (index in seen or type(active) is not list or not active
                or any(type(mask) is not int or not 0 <= mask < len(rows) for mask in active)
                or len(set(active)) != len(active) or target["mask"] not in active):
            raise ValueError("mode candidates require distinct targets and valid active masks")
        seen.add(index)
        weights, offsets = item.get("weights"), item.get("offsets")
        if (type(weights) is not list or len(weights) != 6 or type(offsets) is not list
                or len(offsets) != len(active) or any(type(value) is not str for value in weights + offsets)):
            raise ValueError("mode weights and offsets must be complete exact rational string arrays")
        try:
            values = tuple(F(value) for value in weights + offsets)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("mode proof coordinates must be exact finite rationals") from exc
        scale = math.lcm(*(value.denominator for value in values))
        integers = tuple(int(value * scale) for value in values)
        divisor = math.gcd(*integers) or 1
        integers = tuple(value // divisor for value in integers)
        output.append((index, tuple(active), integers[:6], integers[6:], F(scale, divisor)))
    return tuple(output)


def analyze_c6_winding_mode_excursions(
    parent, *, parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes,
    excursion_bytes, candidates_bytes, max_intersections=250_000, max_prefix_steps=4096,
):
    """Rebuild prior proofs, refine their domain and verify each mode proposal.

    The origin-injected hull covers all original-cube-confined prefixes because
    only reverified exit sources are removed. Clipped closure alone never
    establishes trapping. All candidates use the same independently rebuilt
    pre-candidate domain, so no candidate can assume its own exclusion.
    """
    from tnfr.physics.c6_carried_excursion import derive_c6_carried_mode_excursion_exclusion
    from tnfr.physics.c6_carried_viability import (
        C6CarriedForwardZone, _dbm_intersection, derive_c6_carried_reachable_envelope,
    )

    if parent != json.loads(parent_bytes):
        raise ValueError("the mode audit requires the unchanged retained B47 input bytes")
    blobs = (parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes, excursion_bytes)
    for position in (3, 4, 5):
        retained = json.loads(blobs[position])
        expected = tuple(hashlib.sha256(blob).hexdigest() for blob in blobs[:position])
        evidence = retained.get("input_evidence", ())
        if tuple(item.get("sha256") for item in evidence) != expected:
            raise ValueError("the mode audit lineage does not bind all retained historical input bytes")
    prior = json.loads(excursion_bytes)
    if (prior.get("manifest", {}).get("claim_id") != "O3.a-C6-carried-excursion-exclusion"
            or prior.get("source_scope") != list(SOURCE_SCOPE)):
        raise ValueError("the mode audit requires the retained B52 excursion contract")
    context = _rebuild_c6_outgoing_regions(
        parent, relay_bytes=relay_bytes, historical_bytes=historical_bytes, envelope_bytes=envelope_bytes,
    )
    fresh, profile, state, zones, geometry, facets, targets = context
    old, prior_proved = _analyze_c6_excursion_context(
        context, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
        envelope_bytes=envelope_bytes, region_bytes=region_bytes,
        max_prefix_steps=prior["contract"]["max_prefix_steps"],
    )
    if any(prior.get(key) != value for key, value in _payload(old).items()):
        raise ValueError("the retained B52 excursion differs from its complete canonical reconstruction")
    envelope = fresh["B49_relational_forward_envelope"]
    rows, grid, h = envelope["epi_states"], envelope["grid_quantum"], envelope["timestep"]
    shifts = tuple(tuple(F(h) * F(p) / grid for p in pressure) for pressure in envelope["pressures"])
    if any(value.denominator != 1 for shift in shifts for value in shift):
        raise RuntimeError("the mode audit lost its canonical increment lattice")
    shifts = tuple(tuple(map(int, shift)) for shift in shifts)
    candidates = _c6_mode_candidates(candidates_bytes, facets, rows)
    cut_domain = _cut_c6_proven_facets(
        zones, rows, shifts, geometry, tuple(facets[index] for index in sorted(prior_proved)),
    )
    reachable = derive_c6_carried_reachable_envelope(
        profile.lattice, state=state, epi_states=rows, timestep=h,
        domain_zones=cut_domain, max_intersections=max_intersections,
    )
    by_row = {zone.epi: zone.bounds for zone in reachable.retained_zones}
    proved, records, new_steps = set(prior_proved), [], 0
    for index, active_masks, weights, offsets, scale in candidates:
        pieces = tuple(C6CarriedForwardZone(zone.epi, intersection) for zone in targets[index]
                       if zone.epi in by_row
                       and (intersection := _dbm_intersection(zone.bounds, by_row[zone.epi])) is not None)
        record = dict(target=facets[index], normalization_scale=scale, target_regions=tuple(asdict(z) for z in pieces))
        if not pieces:
            record.update(status="empty_reachable_target", origin_path_within_domain_excluded=True, proof=None)
            proved.add(index)
        else:
            active = tuple(rows[mask] for mask in active_masks if rows[mask] in by_row)
            offset_pairs = tuple((rows[mask], offset) for mask, offset in zip(active_masks, offsets, strict=True)
                                 if rows[mask] in by_row)
            proof = derive_c6_carried_mode_excursion_exclusion(
                profile.lattice, state=state, epi_states=rows, timestep=h,
                active_epi_states=active, weights=weights, cell_offsets=offset_pairs,
                target_regions=pieces, domain_zones=reachable.retained_zones, max_prefix_steps=max_prefix_steps,
            )
            record.update(status=proof.status, origin_path_within_domain_excluded=proof.origin_path_within_domain_excluded,
                          proof=_c6_public_exact_proof(proof))
            new_steps += proof.observed_prefix_steps
            if proof.origin_path_within_domain_excluded:
                proved.add(index)
        records.append(record)
    additional = proved - prior_proved
    post_domain = _cut_c6_proven_facets(
        reachable.retained_zones, rows, shifts, geometry, tuple(facets[index] for index in sorted(additional)),
    )
    post = derive_c6_carried_reachable_envelope(
        profile.lattice, state=state, epi_states=rows, timestep=h,
        domain_zones=post_domain, max_intersections=max_intersections,
    )
    final = {zone.epi: zone.bounds for zone in post.retained_zones}
    absent = {index for index, regions in enumerate(targets)
              if all(zone.epi not in final or _dbm_intersection(zone.bounds, final[zone.epi]) is None for zone in regions)}
    proved.update(absent)
    groups = {}
    for index, facet in enumerate(facets):
        key = facet["node"], facet["direction"]
        group = groups.setdefault(key, dict(node=key[0], direction=key[1], total=0, excluded=0))
        group["total"] += 1
        group["excluded"] += int(index in proved)
    return {
        "contract": {
            "question": "Can exact cell-offset excursion proofs exclude further whole first-exit regions?",
            "proof_rule": "Reverified prior exclusions, origin-injected domain hull, exact per-target drift and ingress separation",
            "candidate_coefficients_are_untrusted": True,
            "weights_and_offsets_are_proof_coordinates_not_physical_parameters": True,
            "max_intersections_per_closure": max_intersections, "max_prefix_steps_per_candidate": max_prefix_steps,
            "resource_limits_are_physical_parameters": False, "pressure_projected_or_carry_reset": False,
        },
        "source": fresh["source"],
        "B53_reconstruction": dict(
            complete_envelope_rebuilt=True, prior_B52_completely_rebuilt=True,
            prior_excursion_sha256=hashlib.sha256(excursion_bytes).hexdigest(),
            candidate_bytes_sha256=hashlib.sha256(candidates_bytes).hexdigest(),
            prior_first_exit_slabs_reverified=len(prior_proved),
            prior_excursion_prefix_steps_replayed=old["new_conditional_trajectory_steps"],
            prior_regional_verification=old["B52_reconstruction"],
        ),
        "B53_cube_geometry": geometry, "B53_outgoing_facets": facets,
        "B53_prior_proven_facets": tuple(facets[index] for index in sorted(prior_proved)),
        "B53_origin_injected_reachable_envelope": _c6_public_exact_proof(reachable),
        "B53_mode_excursion_proofs": tuple(records),
        "B53_post_exclusion_reachable_envelope": _c6_public_exact_proof(post),
        "B53_geometrically_absent_facets": tuple(facets[index] for index in sorted(absent)),
        "B53_exit_groups": tuple(groups.values()),
        "B53_remaining_first_exit_facets": tuple(facet for index, facet in enumerate(facets) if index not in proved),
        "excluded_first_exit_slabs": len(proved), "additional_first_exit_slabs_excluded": len(proved - prior_proved),
        "all_first_exit_slabs_excluded": len(proved) == len(facets),
        "primary_proof_origin_remains_B47": True,
        "new_conditional_trajectory_steps": new_steps, "new_live_graph_steps": 0,
        "indefinite_trapping_certified": False, "whole_band_exit_certified": False,
        "future_runtime_certified": False, "asymptotic_convergence_certified": False,
    }


def analyze_c6_winding_return_regions(
    parent, *, parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes,
    excursion_bytes, candidates_bytes, mode_bytes, max_intersections=250_000,
    query_max_intersections=500_000,
):
    """Reverify B53 and exclude full unsafe regions through exact short returns."""
    from benchmarks.c6_winding_carry_itinerary import _state
    from tnfr.physics.c6_carried_return import derive_c6_carried_return_region_exclusions
    from tnfr.physics.c6_carried_viability import C6CarriedForwardZone
    from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice

    prior = json.loads(mode_bytes)
    inputs = (parent_bytes, relay_bytes, historical_bytes, envelope_bytes,
              region_bytes, excursion_bytes, candidates_bytes)
    if (prior.get("manifest", {}).get("claim_id") != "O3.a-C6-carried-mode-excursions"
            or prior.get("source_scope") != list(SOURCE_SCOPE)
            or tuple(item.get("sha256") for item in prior.get("input_evidence", ()))
            != tuple(hashlib.sha256(raw).hexdigest() for raw in inputs)):
        raise ValueError("the return audit requires the complete retained B53 input lineage")
    fresh = analyze_c6_winding_mode_excursions(
        parent, parent_bytes=parent_bytes, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
        envelope_bytes=envelope_bytes, region_bytes=region_bytes, excursion_bytes=excursion_bytes,
        candidates_bytes=candidates_bytes,
        max_intersections=prior["contract"]["max_intersections_per_closure"],
        max_prefix_steps=prior["contract"]["max_prefix_steps_per_candidate"],
    )
    if any(prior.get(key) != value for key, value in _payload(fresh).items()):
        raise ValueError("the retained B53 result differs from its complete canonical reconstruction")
    previous = fresh["B53_post_exclusion_reachable_envelope"]
    rows = tuple(map(tuple, previous["epi_states"]))
    state = _state(_payload(previous["state"]))
    channels = dict(json.loads(historical_bytes)["B43_original_phase_tail_runtime"]["source"]["normalized_weights"])
    reference = derive_c6_pressure_lattice(
        phase=tuple(fresh["source"]["phase"]), epi_weight=float(F(channels["epi"])),
        phase_weight=float(F(channels["phase"])), epi_lower=state.epi_lower, epi_upper=state.epi_upper,
    )
    domain = tuple(C6CarriedForwardZone(tuple(z["epi"]), tuple(map(tuple, z["bounds"])))
                   for z in previous["retained_zones"])
    by_row = {z.epi: z.bounds for z in domain}
    geometry = fresh["B53_cube_geometry"]
    shifts = tuple(tuple(F(previous["timestep"]) * F(p) / previous["grid_quantum"] for p in row)
                   for row in previous["pressures"])
    if any(v.denominator != 1 for row in shifts for v in row):
        raise RuntimeError("the return audit lost its canonical integer increments")
    labels = fresh["B53_remaining_first_exit_facets"]
    targets = []
    for item in labels:
        mask, node, direction = item["mask"], item["node"], item["direction"]
        slab = _c6_unsafe_slab(by_row[rows[mask]], node, direction,
                               geometry["lower_grid"], geometry["upper_grid"], shifts[mask])
        if slab is None:
            raise RuntimeError("a retained B53 pending slab unexpectedly became empty")
        targets.append((C6CarriedForwardZone(rows[mask], slab),))
    upper2 = max(row[2] for row in rows)
    result = derive_c6_carried_return_region_exclusions(
        reference, state=state, epi_states=rows, timestep=previous["timestep"],
        transient_epi_states=tuple(row for row in rows if row[2] == upper2),
        domain_zones=domain, target_region_groups=tuple(targets),
        max_intersections=max_intersections, query_max_intersections=query_max_intersections,
    )
    records, additional = [], []
    for label, query in zip(labels, result.queries, strict=True):
        public = asdict(query)
        for key in ("completed_depth", "past_exclusion_depth", "origin_path_within_domain_excluded",
                    "actual_origin_reachability_certified"):
            public[key] = getattr(query, key)
        records.append(dict(target=label, query=public))
        if query.origin_path_within_domain_excluded:
            additional.append(label)
    remaining = tuple(label for label in labels if label not in additional)
    groups = [dict(group) for group in fresh["B53_exit_groups"]]
    for group in groups:
        group["excluded"] += sum(item["node"] == group["node"] and item["direction"] == group["direction"]
                                 for item in additional)
    envelope = _c6_public_exact_proof(result.return_envelope)
    for key in ("domain_confined_origin_paths_covered", "clipped_return_inclusion_certified", "intersections"):
        envelope[key] = getattr(result.return_envelope, key)
    return dict(
        contract=dict(question="Can exact short-return predecessors exclude complete pending first-exit regions?",
                      transient_selection="Upper represented node-2 value, with no internal transition",
                      max_intersections=max_intersections, query_max_intersections=query_max_intersections,
                      intermediate_exit_gates_preserved=True, physical_dynamics_changed=False),
        source=fresh["source"],
        B54_reconstruction=dict(prior_B53_completely_rebuilt=True, prior_mode_sha256=hashlib.sha256(mode_bytes).hexdigest(),
                                prior_first_exit_slabs_reverified=fresh["excluded_first_exit_slabs"],
                                prior_excursion_prefix_steps_replayed=fresh["B53_reconstruction"]["prior_excursion_prefix_steps_replayed"]),
        B54_cube_geometry=geometry, B54_outgoing_facets=fresh["B53_outgoing_facets"],
        B54_return_envelope=envelope, B54_region_queries=tuple(records),
        B54_query_intersections=result.intersections, B54_exit_groups=tuple(groups),
        B54_additional_first_exit_facets=tuple(additional), B54_remaining_first_exit_facets=remaining,
        excluded_first_exit_slabs=fresh["excluded_first_exit_slabs"] + len(additional),
        additional_first_exit_slabs_excluded=len(additional), all_first_exit_slabs_excluded=not remaining,
        primary_proof_origin_remains_B47=True, new_conditional_trajectory_steps=0, new_live_graph_steps=0,
        indefinite_trapping_certified=False, whole_band_exit_certified=False,
        future_runtime_certified=False, asymptotic_convergence_certified=False,
    )


def analyze_c6_winding_return_unions(
    parent, *, parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes,
    excursion_bytes, candidates_bytes, mode_bytes, return_bytes, max_intersections=250_000,
    query_max_intersections=500_000, query_max_subsumptions=1_000_000, query_max_zones=5000,
):
    """Rebuild B54 and retain exact unions of backward return regions.

    Separating predecessor pieces avoids the spurious paths introduced by a
    per-cell convex hull. The complete original unsafe slabs remain the targets;
    this analysis still concerns paths before their first original-cube exit.
    """
    from benchmarks.c6_winding_carry_itinerary import _state
    from tnfr.physics.c6_carried_return import derive_c6_carried_return_union_exclusions
    from tnfr.physics.c6_carried_viability import C6CarriedForwardZone
    from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice

    prior = json.loads(return_bytes)
    inputs = (parent_bytes, relay_bytes, historical_bytes, envelope_bytes,
              region_bytes, excursion_bytes, candidates_bytes, mode_bytes)
    if (prior.get("manifest", {}).get("claim_id") != "O3.a-C6-carried-return-regions"
            or prior.get("source_scope") != list(SOURCE_SCOPE)
            or tuple(item.get("sha256") for item in prior.get("input_evidence", ()))
            != tuple(hashlib.sha256(raw).hexdigest() for raw in inputs)):
        raise ValueError("the union audit requires the complete retained B54 input lineage")
    fresh = analyze_c6_winding_return_regions(
        parent, parent_bytes=parent_bytes, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
        envelope_bytes=envelope_bytes, region_bytes=region_bytes, excursion_bytes=excursion_bytes,
        candidates_bytes=candidates_bytes, mode_bytes=mode_bytes,
        max_intersections=prior["contract"]["max_intersections"],
        query_max_intersections=prior["contract"]["query_max_intersections"],
    )
    if any(prior.get(key) != value for key, value in _payload(fresh).items()):
        raise ValueError("the retained B54 result differs from its complete canonical reconstruction")
    previous = fresh["B54_return_envelope"]
    state = _state(_payload(previous["state"]))
    channels = dict(json.loads(historical_bytes)["B43_original_phase_tail_runtime"]["source"]["normalized_weights"])
    reference = derive_c6_pressure_lattice(
        phase=tuple(fresh["source"]["phase"]), epi_weight=float(F(channels["epi"])),
        phase_weight=float(F(channels["phase"])), epi_lower=state.epi_lower, epi_upper=state.epi_upper,
    )

    def zones(values):
        return tuple(C6CarriedForwardZone(tuple(z["epi"]), tuple(map(tuple, z["bounds"]))) for z in values)

    labels = fresh["B54_remaining_first_exit_facets"]
    targets = tuple(zones(item["query"]["target_regions"]) for item in fresh["B54_region_queries"]
                    if item["target"] in labels)
    result = derive_c6_carried_return_union_exclusions(
        reference, state=state, epi_states=tuple(map(tuple, previous["epi_states"])),
        timestep=previous["timestep"], transient_epi_states=tuple(map(tuple, previous["transient_epi_states"])),
        domain_zones=zones(previous["domain_zones"]), target_region_groups=targets,
        max_intersections=max_intersections, query_max_intersections=query_max_intersections,
        query_max_subsumptions=query_max_subsumptions, query_max_zones=query_max_zones,
    )
    records, additional = [], []
    for label, query in zip(labels, result.queries, strict=True):
        public = asdict(query)
        for key in ("completed_depth", "past_exclusion_depth", "origin_path_within_domain_excluded",
                    "actual_origin_reachability_certified"):
            public[key] = getattr(query, key)
        records.append(dict(target=label, query=public))
        if query.origin_path_within_domain_excluded:
            additional.append(label)
    remaining = tuple(label for label in labels if label not in additional)
    groups = [dict(group) for group in fresh["B54_exit_groups"]]
    for group in groups:
        group["excluded"] += sum(item["node"] == group["node"] and item["direction"] == group["direction"]
                                 for item in additional)
    envelope = _c6_public_exact_proof(result.return_envelope)
    for key in ("domain_confined_origin_paths_covered", "clipped_return_inclusion_certified", "intersections"):
        envelope[key] = getattr(result.return_envelope, key)
    return dict(
        contract=dict(question="Can unjoined return predecessors exclude complete pending first-exit regions?",
                      max_intersections=max_intersections, query_max_intersections=query_max_intersections,
                      query_max_subsumptions=query_max_subsumptions, query_max_zones=query_max_zones,
                      no_convex_hull_approximation_in_queries=True, physical_dynamics_changed=False),
        source=fresh["source"],
        B55_reconstruction=dict(prior_B54_completely_rebuilt=True,
                                prior_return_sha256=hashlib.sha256(return_bytes).hexdigest(),
                                prior_first_exit_slabs_reverified=fresh["excluded_first_exit_slabs"],
                                prior_excursion_prefix_steps_replayed=fresh["B54_reconstruction"]["prior_excursion_prefix_steps_replayed"]),
        B55_cube_geometry=fresh["B54_cube_geometry"], B55_outgoing_facets=fresh["B54_outgoing_facets"],
        B55_return_envelope=envelope, B55_union_queries=tuple(records),
        B55_query_intersections=result.intersections, B55_query_subsumptions=result.subsumptions,
        B55_exit_groups=tuple(groups), B55_additional_first_exit_facets=tuple(additional),
        B55_remaining_first_exit_facets=remaining,
        excluded_first_exit_slabs=fresh["excluded_first_exit_slabs"] + len(additional),
        additional_first_exit_slabs_excluded=len(additional), all_first_exit_slabs_excluded=not remaining,
        primary_proof_origin_remains_B47=True, new_conditional_trajectory_steps=0, new_live_graph_steps=0,
        indefinite_trapping_certified=False, whole_band_exit_certified=False,
        future_runtime_certified=False, asymptotic_convergence_certified=False,
    )


def analyze_c6_winding_return_histories(
    parent, *, parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes,
    excursion_bytes, candidates_bytes, mode_bytes, return_bytes, union_bytes, history_candidates_bytes,
    max_intersections=250_000, max_word_work_items=10_000,
):
    """Separate exact path-count feasibility from guarded return chronology."""
    from benchmarks.c6_winding_carry_itinerary import _state
    from tnfr.physics.c6_carried_return import (
        derive_c6_carried_return_count_relaxation, derive_c6_carried_return_word_budget,
    )
    from tnfr.physics.c6_carried_viability import C6CarriedForwardZone
    from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice

    prior = json.loads(union_bytes)
    inputs = (parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes,
              excursion_bytes, candidates_bytes, mode_bytes, return_bytes)
    if (prior.get("manifest", {}).get("claim_id") != "O3.a-C6-carried-return-unions"
            or prior.get("source_scope") != list(SOURCE_SCOPE)
            or tuple(item.get("sha256") for item in prior.get("input_evidence", ()))
            != tuple(hashlib.sha256(raw).hexdigest() for raw in inputs)):
        raise ValueError("the history audit requires the complete retained B55 input lineage")
    fresh = analyze_c6_winding_return_unions(
        parent, parent_bytes=parent_bytes, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
        envelope_bytes=envelope_bytes, region_bytes=region_bytes, excursion_bytes=excursion_bytes,
        candidates_bytes=candidates_bytes, mode_bytes=mode_bytes, return_bytes=return_bytes,
        **{key: prior["contract"][key] for key in (
            "max_intersections", "query_max_intersections", "query_max_subsumptions", "query_max_zones")},
    )
    if any(prior.get(key) != value for key, value in _payload(fresh).items()):
        raise ValueError("the retained B55 result differs from its complete canonical reconstruction")
    previous = fresh["B55_return_envelope"]
    candidates = json.loads(history_candidates_bytes)
    relation_hash = hashlib.sha256(json.dumps(
        _payload(previous["return_relation"]), sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    if (type(candidates.get("schema_version")) is not int or candidates["schema_version"] != 1
            or candidates.get("return_relation_sha256") != relation_hash):
        raise ValueError("history candidates must bind the complete canonical return relation")
    size = len(previous["return_relation"])
    generators = []
    for sparse in candidates["coordinate_generators"]:
        dense, seen = [0] * size, set()
        for pair in sparse:
            if (type(pair) is not list or len(pair) != 2 or any(type(v) is not int for v in pair)
                    or not 0 <= pair[0] < size or pair[0] in seen):
                raise ValueError("each sparse coordinate generator needs distinct valid integer edge entries")
            seen.add(pair[0])
            dense[pair[0]] = pair[1]
        generators.append(tuple(dense))
    rows = tuple(map(tuple, previous["epi_states"]))
    state = _state(_payload(previous["state"]))
    channels = dict(json.loads(historical_bytes)["B43_original_phase_tail_runtime"]["source"]["normalized_weights"])
    reference = derive_c6_pressure_lattice(
        phase=tuple(fresh["source"]["phase"]), epi_weight=float(F(channels["epi"])),
        phase_weight=float(F(channels["phase"])), epi_lower=state.epi_lower, epi_upper=state.epi_upper,
    )
    common = dict(state=state, epi_states=rows, timestep=previous["timestep"],
                  transient_epi_states=tuple(map(tuple, previous["transient_epi_states"])),
                  domain_zones=tuple(C6CarriedForwardZone(tuple(z["epi"]), tuple(map(tuple, z["bounds"])))
                                     for z in previous["domain_zones"]), max_intersections=max_intersections)
    counts = derive_c6_carried_return_count_relaxation(
        reference, **common, positive_circulation=tuple(candidates["positive_circulation"]),
        coordinate_generators=tuple(generators),
    )
    retained = {z.epi: z.bounds for z in counts.return_envelope.retained_zones}

    def contains(bounds, coordinates):
        x = (*coordinates, 0)
        return all(x[i] - x[j] <= bounds[i][j] for i in range(7) for j in range(7))

    labels = fresh["B55_remaining_first_exit_facets"]
    proposed = candidates["remaining_target_proposals"]
    if [p["target"] for p in proposed] != list(labels):
        raise ValueError("count endpoint proposals must cover every original remaining slab exactly once")
    target_queries = [item for item in fresh["B55_union_queries"] if item["target"] in labels]
    witnesses = []
    for proposal, item in zip(proposed, target_queries, strict=True):
        point = tuple(proposal["endpoint_grid_coordinates"])
        if len(point) != 6 or any(type(value) is not int for value in point):
            raise ValueError("count endpoint coordinates must be six exact integers")
        target = rows[item["target"]["mask"]]
        original = item["query"]["target_regions"][0]["bounds"]
        if not contains(original, point) or not contains(retained[target], point):
            raise ValueError("a count endpoint must belong to its complete original unsafe slab and retained cell")
        index = proposal["terminal_transient_edge_index"]
        shift = (0,) * 6
        base = target
        if index is not None:
            if type(index) is not int or not 0 <= index < len(counts.return_envelope.intermediate_transitions):
                raise ValueError("terminal transient edge index is invalid")
            edge = counts.return_envelope.intermediate_transitions[index]
            if edge.target_epi != target or not contains(edge.target_guard, point):
                raise ValueError("a count endpoint must satisfy its selected terminal transient guard")
            base, shift = edge.source_epi, edge.shift
        displacement = tuple(x - a for x, a in zip(point, shift, strict=True))
        if base not in retained or not contains(retained[base], displacement):
            raise ValueError("the terminal base endpoint must lie in its retained source region")
        if index is not None and not contains(edge.source_guard, displacement):
            raise ValueError("the terminal base endpoint misses its transient source guard")
        witness = counts.construct_counts(target_epi=base, displacement=displacement)
        public = asdict(witness)
        edge_counts = public.pop("edge_counts")
        public.update(edge_counts_sha256=hashlib.sha256(json.dumps(edge_counts).encode()).hexdigest(),
                      minimum_edge_count=min(edge_counts), total_edge_count=sum(edge_counts),
                      joint_guard_satisfaction_certified=False, actual_origin_reachability_certified=False)
        witnesses.append(dict(target=item["target"], endpoint_grid_coordinates=point,
                              terminal_transient_edge_index=index, terminal_transient_shift=shift, count_witness=public))
    words = []
    proposals = candidates.get("closed_return_words")
    if type(proposals) is not list or not proposals:
        raise ValueError("the history campaign requires nonempty closed return-word proposals")
    for proposal in proposals:
        proof = derive_c6_carried_return_word_budget(
            reference, **common, word_edge_indices=tuple(proposal["edge_indices"]),
            max_work_items=max_word_work_items,
        )
        public = asdict(proof)
        public.pop("return_envelope")
        for key in ("word_budget_certified", "conditional_word_identity_certified",
                    "common_grid_relaxes_coordinate_cosets", "actual_origin_reachability_certified",
                    "conditional_boundedness_certified", "future_runtime_certified"):
            public[key] = getattr(proof, key)
        words.append(dict(name=proposal["name"], proof=public))
    envelope = _c6_public_exact_proof(counts.return_envelope)
    count_public = asdict(counts)
    count_public.pop("return_envelope")
    for key in ("exact_cycle_displacement_lattice_certified", "nonnegative_counts_cover_coordinate_cosets",
                "abstract_mode_walk_exists_for_each_coordinate_coset_point", "joint_guard_satisfaction_certified",
                "actual_origin_reachability_certified", "conditional_boundedness_certified"):
        count_public[key] = getattr(counts, key)
    return dict(
        contract=dict(question="Which joint temporal constraints are lost by global counts and recovered by exact word guards?",
                      max_intersections=max_intersections, max_word_work_items=max_word_work_items,
                      physical_dynamics_changed=False), source=fresh["source"],
        B56_reconstruction=dict(prior_B55_completely_rebuilt=True, prior_union_sha256=hashlib.sha256(union_bytes).hexdigest(),
                                prior_first_exit_slabs_reverified=fresh["excluded_first_exit_slabs"]),
        B56_return_envelope=envelope, B56_relation_sha256=relation_hash,
        B56_count_relaxation=count_public, B56_target_count_witnesses=tuple(witnesses), B56_word_budgets=tuple(words),
        B56_outgoing_facets=fresh["B55_outgoing_facets"], B56_exit_groups=fresh["B55_exit_groups"],
        B56_remaining_first_exit_facets=labels, excluded_first_exit_slabs=fresh["excluded_first_exit_slabs"],
        additional_first_exit_slabs_excluded=0, all_first_exit_slabs_excluded=not labels,
        primary_proof_origin_remains_B47=True, new_conditional_trajectory_steps=0, new_live_graph_steps=0,
        indefinite_trapping_certified=False, whole_band_exit_certified=False,
        future_runtime_certified=False, asymptotic_convergence_certified=False,
    )


def _c6_memory_configuration(fresh, previous_prefix, historical_bytes, domain_zones):
    """Recover the unchanged canonical nodal source for a declared proof domain."""
    from benchmarks.c6_winding_carry_itinerary import _state
    from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice

    previous = fresh[f"{previous_prefix}_return_envelope"]
    state = _state(_payload(previous["state"]))
    rows = tuple(map(tuple, previous["epi_states"]))
    channels = dict(json.loads(historical_bytes)["B43_original_phase_tail_runtime"]["source"]["normalized_weights"])
    reference = derive_c6_pressure_lattice(
        phase=tuple(fresh["source"]["phase"]), epi_weight=float(F(channels["epi"])),
        phase_weight=float(F(channels["phase"])), epi_lower=state.epi_lower, epi_upper=state.epi_upper,
    )
    return reference, dict(state=state, epi_states=rows, timestep=previous["timestep"],
                           transient_epi_states=tuple(map(tuple, previous["transient_epi_states"])),
                           domain_zones=domain_zones)


def _analyze_c6_memory_domain(
    fresh, previous_prefix, queries, *, historical_bytes, domain_zones,
    max_intersections, max_memory_work, max_memory_arcs, query_max_intersections,
):
    """One shared canonical memory and whole-target readout for both campaigns."""
    from tnfr.physics.c6_carried_return import derive_c6_carried_return_memory_region_exclusions, _move
    from tnfr.physics.c6_carried_viability import _dbm_intersection

    reference, arguments = _c6_memory_configuration(fresh, previous_prefix, historical_bytes, domain_zones)
    state, rows = arguments["state"], arguments["epi_states"]
    labels = fresh[f"{previous_prefix}_remaining_first_exit_facets"]
    targets = tuple(_c6_read_zones(item["query"]["target_regions"]) for item in queries)
    result = derive_c6_carried_return_memory_region_exclusions(
        reference, **arguments, max_intersections=max_intersections,
        max_memory_work=max_memory_work, max_memory_arcs=max_memory_arcs,
        target_region_groups=targets, query_max_intersections=query_max_intersections,
    )
    proof = result.memory_envelope
    envelope = proof.return_envelope
    origin_zone = ((0,) * 7,) * 7
    target_observations, additional = [], []
    for item in queries:
        label = item["target"]
        target = tuple(tuple(value for value in line) for line in item["query"]["target_regions"][0]["bounds"])
        target_row = rows[label["mask"]]
        direct, terminal = [], []
        origin_hit = target_row == state.epi and _dbm_intersection(origin_zone, target) is not None
        if proof.initialization_complete:
            memory_states = [(-1, state.epi, origin_zone)] + [
                (i, edge.target_epi, zone)
                for i, (edge, zone) in enumerate(zip(envelope.return_relation, proof.retained_endpoint_zones, strict=True))
                if zone is not None]
            for index, row, zone in memory_states:
                if index != -1 and row == target_row and _dbm_intersection(zone, target) is not None:
                    direct.append(index)
                for j, edge in enumerate(envelope.intermediate_transitions):
                    if edge.source_epi != row or edge.target_epi != target_row:
                        continue
                    piece = _dbm_intersection(zone, edge.source_guard)
                    if piece is not None and _dbm_intersection(_move(piece, edge.shift), target) is not None:
                        terminal.append((index, j))
        excluded = proof.initialization_complete and not origin_hit and not direct and not terminal
        if excluded:
            additional.append(label)
        target_observations.append(dict(
            target=label, original_target_bounds=target, origin_in_target=origin_hit,
            direct_memory_indices=tuple(direct), terminal_transient_memory_edges=tuple(terminal),
            observation_complete=proof.initialization_complete,
            origin_path_before_original_cube_exit_excluded=excluded,
            actual_origin_reachability_certified=False,
        ))
    memory_queries = []
    for label, query in zip(labels, result.queries, strict=True):
        public_query = asdict(query)
        public_query.update(origin_path_within_domain_excluded=query.origin_path_within_domain_excluded,
                            actual_origin_reachability_certified=False)
        memory_queries.append(dict(target=label, query=public_query))
        if query.origin_path_within_domain_excluded and label not in additional:
            additional.append(label)
    remaining = tuple(label for label in labels if label not in additional)
    groups = [dict(group) for group in fresh[f"{previous_prefix}_exit_groups"]]
    for group in groups:
        group["excluded"] += sum(label["node"] == group["node"] and label["direction"] == group["direction"]
                                 for label in additional)
    public = asdict(proof)
    public.pop("return_envelope")
    for key in ("memory_work", "domain_confined_origin_histories_covered", "conditional_invariance_certified",
                "conditional_boundedness_certified", "future_runtime_certified", "asymptotic_convergence_certified"):
        public[key] = getattr(proof, key)
    return dict(
        return_envelope=_c6_public_exact_proof(envelope), memory_envelope=public,
        memory_queries=tuple(memory_queries), query_relation_complete=result.query_relation_complete,
        query_relation_intersections=result.query_relation_intersections,
        target_observations=tuple(target_observations), exit_groups=tuple(groups),
        additional_first_exit_facets=tuple(additional), remaining_first_exit_facets=remaining,
    )


def analyze_c6_winding_return_memory(
    parent, *, parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes,
    excursion_bytes, candidates_bytes, mode_bytes, return_bytes, union_bytes, history_candidates_bytes,
    history_bytes, max_intersections=250_000, max_memory_work=500_000, max_memory_arcs=100_000,
    query_max_intersections=250_000,
):
    """Refine whole-target coverage by the exact last completed return record."""
    from tnfr.physics.c6_carried_viability import C6CarriedForwardZone

    prior = json.loads(history_bytes)
    inputs = (parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes,
              excursion_bytes, candidates_bytes, mode_bytes, return_bytes, union_bytes, history_candidates_bytes)
    if (prior.get("manifest", {}).get("claim_id") != "O3.a-C6-carried-return-histories"
            or prior.get("source_scope") != list(SOURCE_SCOPE)
            or tuple(item.get("sha256") for item in prior.get("input_evidence", ()))
            != tuple(hashlib.sha256(raw).hexdigest() for raw in inputs)):
        raise ValueError("the memory audit requires the complete retained B56 input lineage")
    fresh = analyze_c6_winding_return_histories(
        parent, parent_bytes=parent_bytes, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
        envelope_bytes=envelope_bytes, region_bytes=region_bytes, excursion_bytes=excursion_bytes,
        candidates_bytes=candidates_bytes, mode_bytes=mode_bytes, return_bytes=return_bytes,
        union_bytes=union_bytes, history_candidates_bytes=history_candidates_bytes,
        **{key: prior["contract"][key] for key in ("max_intersections", "max_word_work_items")},
    )
    if any(prior.get(key) != value for key, value in _payload(fresh).items()):
        raise ValueError("the retained B56 result differs from its complete canonical reconstruction")
    labels = fresh["B56_remaining_first_exit_facets"]
    queries = [item for item in json.loads(union_bytes)["B55_union_queries"] if item["target"] in labels]
    domain = tuple(C6CarriedForwardZone(tuple(z["epi"]), tuple(map(tuple, z["bounds"])))
                   for z in fresh["B56_return_envelope"]["domain_zones"])
    analyzed = _analyze_c6_memory_domain(
        fresh, "B56", queries, historical_bytes=historical_bytes, domain_zones=domain,
        max_intersections=max_intersections, max_memory_work=max_memory_work,
        max_memory_arcs=max_memory_arcs, query_max_intersections=query_max_intersections,
    )
    additional, remaining = analyzed["additional_first_exit_facets"], analyzed["remaining_first_exit_facets"]
    return dict(
        contract=dict(question="Does joint carried-state memory of the last return exclude further whole first-exit slabs?",
                      max_intersections=max_intersections, max_memory_work=max_memory_work,
                      max_memory_arcs=max_memory_arcs, query_max_intersections=query_max_intersections,
                      query_budget_scope="One shared graph and one separate budget per target",
                      physical_dynamics_changed=False), source=fresh["source"],
        B57_reconstruction=dict(prior_B56_completely_rebuilt=True,
                                prior_history_sha256=hashlib.sha256(history_bytes).hexdigest(),
                                prior_first_exit_slabs_reverified=fresh["excluded_first_exit_slabs"]),
        **{f"B57_{key}": value for key, value in analyzed.items()},
        B57_outgoing_facets=fresh["B56_outgoing_facets"],
        excluded_first_exit_slabs=fresh["excluded_first_exit_slabs"] + len(additional),
        additional_first_exit_slabs_excluded=len(additional), all_first_exit_slabs_excluded=not remaining,
        primary_proof_origin_remains_B47=True, new_conditional_trajectory_steps=0, new_live_graph_steps=0,
        indefinite_trapping_certified=False, whole_band_exit_certified=False,
        future_runtime_certified=False, asymptotic_convergence_certified=False,
    )


def _c6_read_zones(values):
    from tnfr.physics.c6_carried_viability import C6CarriedForwardZone

    return tuple(C6CarriedForwardZone(tuple(z["epi"]), tuple(map(tuple, z["bounds"]))) for z in values)


def _compare_c6_memory_query_geometry(before, after, groups, maximum):
    """Compare actual complete operators and initial arrays, never only counts."""
    from tnfr.physics.c6_carried_return import (
        _return_memory_query_graph, _return_memory_query_initialization,
    )

    def digest(value):
        return hashlib.sha256(json.dumps(_payload(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    first, second = before.return_envelope, after.return_envelope
    source_equal = all(getattr(first, key) == getattr(second, key) for key in (
        "reference", "state", "epi_states", "pressures", "timestep", "grid_quantum", "transient_epi_states",
    ))

    def identities(envelope):
        return tuple((e.source_epi, e.intermediate_epi, e.target_epi, e.shift) for e in envelope.return_relation)

    records_equal = identities(first) == identities(second)
    graph_before, complete_before, work_before = _return_memory_query_graph(before, maximum)
    graph_after, complete_after, work_after = _return_memory_query_graph(after, maximum)
    graph_equal = complete_before and complete_after and graph_before == graph_after
    initial = []
    for targets in groups:
        left, left_complete, left_work = _return_memory_query_initialization(before, targets, maximum)
        right, right_complete, right_work = _return_memory_query_initialization(after, targets, maximum)
        initial.append(dict(
            complete=left_complete and right_complete, equal=left_complete and right_complete and left == right,
            before_sha256=digest(left), after_sha256=digest(right),
            before_intersections=left_work, after_intersections=right_work,
        ))
    endpoints_equal = before.retained_endpoint_zones == after.retained_endpoint_zones
    roots_equal = before.root_endpoint_zones == after.root_endpoint_zones
    return dict(
        canonical_source_equal=source_equal, return_record_identities_equal=records_equal,
        retained_endpoint_arrays_equal=endpoints_equal, origin_images_equal=roots_equal,
        complete_query_graph_equal=graph_equal, before_graph_complete=complete_before,
        after_graph_complete=complete_after, before_graph_intersections=work_before,
        after_graph_intersections=work_after, before_arc_count=len(graph_before), after_arc_count=len(graph_after),
        before_graph_sha256=digest(graph_before), after_graph_sha256=digest(graph_after),
        target_initializations=tuple(initial),
        all_target_operators_and_initial_conditions_identical=(
            source_equal and records_equal and endpoints_equal and roots_equal and graph_equal
            and all(item["equal"] for item in initial)),
        actual_origin_reachability_certified=False, conditional_boundedness_certified=False,
    )


def analyze_c6_winding_memory_cuts(
    parent, *, parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes,
    excursion_bytes, candidates_bytes, mode_bytes, return_bytes, union_bytes, history_candidates_bytes,
    history_bytes, memory_bytes, max_intersections=250_000, max_memory_work=500_000,
    max_memory_arcs=100_000, query_max_intersections=250_000,
):
    """Audit whether previously proved slab cuts change the residual query operator."""
    from tnfr.physics.c6_carried_return import derive_c6_carried_return_memory_envelope, _return_target_groups

    prior = json.loads(memory_bytes)
    inputs = (parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes, excursion_bytes,
              candidates_bytes, mode_bytes, return_bytes, union_bytes, history_candidates_bytes, history_bytes)
    if (prior.get("manifest", {}).get("claim_id") != "O3.a-C6-carried-return-memory"
            or prior.get("source_scope") != list(SOURCE_SCOPE)
            or tuple(item.get("sha256") for item in prior.get("input_evidence", ()))
            != tuple(hashlib.sha256(raw).hexdigest() for raw in inputs)):
        raise ValueError("the cut audit requires the complete retained B57 input lineage")
    fresh = analyze_c6_winding_return_memory(
        parent, parent_bytes=parent_bytes, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
        envelope_bytes=envelope_bytes, region_bytes=region_bytes, excursion_bytes=excursion_bytes,
        candidates_bytes=candidates_bytes, mode_bytes=mode_bytes, return_bytes=return_bytes, union_bytes=union_bytes,
        history_candidates_bytes=history_candidates_bytes, history_bytes=history_bytes,
        **{key: prior["contract"][key] for key in (
            "max_intersections", "max_memory_work", "max_memory_arcs", "query_max_intersections")},
    )
    if any(prior.get(key) != value for key, value in _payload(fresh).items()):
        raise ValueError("the retained B57 result differs from its complete canonical reconstruction")
    previous = fresh["B57_return_envelope"]
    domain = _c6_read_zones(previous["domain_zones"])
    reference, arguments = _c6_memory_configuration(fresh, "B57", historical_bytes, domain)
    original = derive_c6_carried_return_memory_envelope(
        reference, **arguments, **{key: prior["contract"][key] for key in (
            "max_intersections", "max_memory_work", "max_memory_arcs")},
    )
    b54, b55 = json.loads(return_bytes), json.loads(union_bytes)
    facets = (*b54["B54_additional_first_exit_facets"], *b55["B55_additional_first_exit_facets"],
              *fresh["B57_additional_first_exit_facets"])
    envelope = original.return_envelope
    shifts = tuple(tuple(F(envelope.timestep) * F(p) / envelope.grid_quantum for p in pressure)
                   for pressure in envelope.pressures)
    if any(v.denominator != 1 for row in shifts for v in row):
        raise RuntimeError("the cut audit requires integral canonical nodal shifts")
    shifts = tuple(tuple(map(int, row)) for row in shifts)
    cut_domain = _cut_c6_proven_facets(domain, envelope.epi_states, shifts, b55["B55_cube_geometry"], facets)
    changed = derive_c6_carried_return_memory_envelope(
        reference, **dict(arguments, domain_zones=cut_domain), max_intersections=max_intersections,
        max_memory_work=max_memory_work, max_memory_arcs=max_memory_arcs,
    )
    labels = fresh["B57_remaining_first_exit_facets"]
    selected = [item for item in b55["B55_union_queries"] if item["target"] in labels]
    if [item["target"] for item in selected] != list(labels):
        raise ValueError("the cut audit must preserve every original residual target")
    groups = tuple(_c6_read_zones(item["query"]["target_regions"]) for item in selected)
    # A cut must not silently shrink the target whose whole-slab status is reused.
    _return_target_groups(groups, changed.return_envelope.domain_zones, 4096)
    comparison = _compare_c6_memory_query_geometry(original, changed, groups, query_max_intersections)
    for label, item in zip(labels, comparison["target_initializations"], strict=True):
        item["target"] = label
    public = asdict(changed)
    public.pop("return_envelope")
    return dict(
        contract=dict(question="Do certified slab complements change any remaining last-memory query?",
                      max_intersections=max_intersections, max_memory_work=max_memory_work,
                      max_memory_arcs=max_memory_arcs, query_max_intersections=query_max_intersections,
                      physical_dynamics_changed=False), source=fresh["source"],
        B58_reconstruction=dict(prior_B57_completely_rebuilt=True,
                                prior_memory_sha256=hashlib.sha256(memory_bytes).hexdigest(),
                                prior_first_exit_slabs_reverified=fresh["excluded_first_exit_slabs"]),
        B58_proved_cuts=facets, B58_original_cube_geometry=b55["B55_cube_geometry"],
        B58_original_target_groups=tuple(tuple(asdict(zone) for zone in group) for group in groups),
        B58_cut_domain=tuple(asdict(zone) for zone in cut_domain),
        B58_return_envelope=_c6_public_exact_proof(changed.return_envelope), B58_memory_envelope=public,
        B58_query_equivalence=comparison, B58_outgoing_facets=fresh["B57_outgoing_facets"],
        B58_exit_groups=fresh["B57_exit_groups"], B58_remaining_first_exit_facets=labels,
        same_query_geometry_repetition_avoided=comparison["all_target_operators_and_initial_conditions_identical"],
        excluded_first_exit_slabs=fresh["excluded_first_exit_slabs"], additional_first_exit_slabs_excluded=0,
        all_first_exit_slabs_excluded=not labels, primary_proof_origin_remains_B47=True,
        new_conditional_trajectory_steps=0, new_live_graph_steps=0, indefinite_trapping_certified=False,
        whole_band_exit_certified=False, future_runtime_certified=False, asymptotic_convergence_certified=False,
    )


def _c6_priority_hints(raw, target, inputs, maximum):
    """Read bounded advisory regions; their contents never authorize a cut."""
    from tnfr.physics.c6_carried_viability import _positive_integer

    maximum = _positive_integer(maximum, "max_priority_regions")
    if (type(target) is not tuple or len(target) != 3 or type(target[0]) is not int
            or type(target[1]) is not int or not 0 <= target[0] < 64
            or not 0 <= target[1] < 6 or target[2] not in ("lower", "upper")):
        raise ValueError("priority_target must be an exact (mask, node, direction) tuple")
    prior = json.loads(raw)
    if (prior.get("manifest", {}).get("claim_id") != "O3.a-C6-carried-safe-partition"
            or prior.get("source_scope") != list(SOURCE_SCOPE)
            or tuple(item.get("sha256") for item in prior.get("input_evidence", ()))
            != tuple(hashlib.sha256(value).hexdigest() for value in inputs)):
        raise ValueError("priority hints require the same fourteen-input B59 lineage")
    label = dict(mask=target[0], node=target[1], direction=target[2])
    selected = [item for item in prior["B59_partition_queries"] if item["target"] == label]
    if len(selected) != 1:
        raise ValueError("the priority target must identify exactly one B59 query")
    vertices = prior["B59_partition"]["vertices"]
    zones = selected[0]["query"]["retained_endpoint_zones"]
    if not vertices or len(vertices) > maximum or len(zones) != len(vertices) + 1:
        raise ValueError("priority hints require bounded complete indexed histories and a sentinel")
    if any(type(vertex) is not list or len(vertex) != 2
           or any(type(value) is not int or value < 0 for value in vertex) for vertex in vertices):
        raise ValueError("priority vertices require exact nonnegative memory and piece indices")
    hints = []
    for vertex, zone in zip(vertices, zones[:-1], strict=True):
        if zone is not None:
            if type(zone) is not list or len(zone) != 7 or any(type(row) is not list or len(row) != 7 for row in zone):
                raise ValueError("priority regions must be complete seven-dimensional DBMs")
            zone = tuple(tuple(row) for row in zone)
        hints.append((vertex[0], zone))
    return tuple(hints), label


def analyze_c6_winding_safe_partition(
    parent, *, parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes,
    excursion_bytes, candidates_bytes, mode_bytes, return_bytes, union_bytes, history_candidates_bytes,
    history_bytes, memory_bytes, cuts_bytes, max_intersections=250_000, max_memory_work=500_000,
    max_memory_arcs=100_000, exclusion_query_max_intersections=250_000,
    max_partition_pieces=5000, max_partition_construction_work=100_000,
    max_partition_arcs=100_000, max_partition_work=500_000, query_max_intersections=250_000,
    excluded_predecessor_depth=0, priority_bytes=None, priority_target=None,
    max_priority_regions=5000, max_cover_work=100_000,
):
    """Keep certified nonconvex safe pieces separate while querying full exit slabs."""
    from tnfr.physics.c6_carried_return import (
        _return_predecessor_depth, derive_c6_carried_return_safe_partition_region_exclusions,
    )

    depth = _return_predecessor_depth(excluded_predecessor_depth)
    prior = json.loads(cuts_bytes)
    inputs = (parent_bytes, relay_bytes, historical_bytes, envelope_bytes, region_bytes, excursion_bytes,
              candidates_bytes, mode_bytes, return_bytes, union_bytes, history_candidates_bytes,
              history_bytes, memory_bytes)
    if priority_bytes is None and priority_target is not None:
        raise ValueError("priority_target requires advisory priority_bytes")
    hints, priority_label = (), None
    if priority_bytes is not None:
        from tnfr.physics.c6_carried_viability import _positive_integer

        _positive_integer(max_cover_work, "max_cover_work")
        hints, priority_label = _c6_priority_hints(
            priority_bytes, priority_target, (*inputs, cuts_bytes), max_priority_regions,
        )
    if (prior.get("manifest", {}).get("claim_id") != "O3.a-C6-carried-memory-cuts"
            or prior.get("source_scope") != list(SOURCE_SCOPE)
            or tuple(item.get("sha256") for item in prior.get("input_evidence", ()))
            != tuple(hashlib.sha256(raw).hexdigest() for raw in inputs)):
        raise ValueError("the safe partition requires the complete retained B58 input lineage")
    fresh = analyze_c6_winding_memory_cuts(
        parent, parent_bytes=parent_bytes, relay_bytes=relay_bytes, historical_bytes=historical_bytes,
        envelope_bytes=envelope_bytes, region_bytes=region_bytes, excursion_bytes=excursion_bytes,
        candidates_bytes=candidates_bytes, mode_bytes=mode_bytes, return_bytes=return_bytes, union_bytes=union_bytes,
        history_candidates_bytes=history_candidates_bytes, history_bytes=history_bytes, memory_bytes=memory_bytes,
        **{key: prior["contract"][key] for key in (
            "max_intersections", "max_memory_work", "max_memory_arcs", "query_max_intersections")},
    )
    if any(prior.get(key) != value for key, value in _payload(fresh).items()):
        raise ValueError("the retained B58 result differs from its complete canonical reconstruction")
    # B58 has reverified the complete B57 report. Its uncut domain retains the
    # excluded slab, so the owner can prove it again before deriving preimages.
    baseline = json.loads(memory_bytes)
    previous = baseline["B57_return_envelope"]
    domain = _c6_read_zones(previous["domain_zones"])
    reference, arguments = _c6_memory_configuration(baseline, "B57", historical_bytes, domain)
    labels = fresh["B58_remaining_first_exit_facets"]
    selected = [item for item in baseline["B57_memory_queries"] if item["target"] in labels]
    excluded_labels = baseline["B57_additional_first_exit_facets"]
    proposals = [item for item in baseline["B57_memory_queries"] if item["target"] in excluded_labels]
    if ([item["target"] for item in selected] != list(labels)
            or [item["target"] for item in proposals] != list(excluded_labels)):
        raise ValueError("the safe partition must retain every original target and declared exclusion proposal")
    groups = tuple(_c6_read_zones(item["query"]["target_regions"]) for item in selected)
    excluded_groups = tuple(_c6_read_zones(item["query"]["target_regions"]) for item in proposals)
    derive = derive_c6_carried_return_safe_partition_region_exclusions
    extra = {}
    if priority_bytes is not None:
        from tnfr.physics.c6_carried_return import derive_c6_carried_return_safe_cover_region_exclusions

        derive = derive_c6_carried_return_safe_cover_region_exclusions
        extra = dict(priority_memory_regions=hints, max_priority_regions=max_priority_regions,
                     max_cover_work=max_cover_work)
        if priority_label not in labels:
            raise ValueError("the priority target must remain an original complete pending label")
    result = derive(
        reference, **arguments, target_region_groups=groups, excluded_target_region_groups=excluded_groups,
        max_intersections=max_intersections, max_memory_work=max_memory_work, max_memory_arcs=max_memory_arcs,
        exclusion_query_max_intersections=exclusion_query_max_intersections,
        max_partition_pieces=max_partition_pieces, max_partition_construction_work=max_partition_construction_work,
        max_partition_arcs=max_partition_arcs, max_partition_work=max_partition_work,
        query_max_intersections=query_max_intersections,
        excluded_predecessor_depth=depth, **extra,
    )
    partition = result.partition if priority_bytes is None else result.cover.partition
    memory = partition.memory_envelope
    public = asdict(partition)
    public.pop("memory_envelope")
    base_public = asdict(memory)
    base_public.pop("return_envelope")
    additional, queries = [], []
    for label, query in zip(labels, result.queries, strict=True):
        item = asdict(query)
        item.update(origin_path_within_domain_excluded=query.origin_path_within_domain_excluded,
                    actual_origin_reachability_certified=False)
        queries.append(dict(target=label, query=item))
        if query.origin_path_within_domain_excluded:
            additional.append(label)
    remaining = tuple(label for label in labels if label not in additional)
    exit_groups = [dict(group) for group in fresh["B58_exit_groups"]]
    for group in exit_groups:
        group["excluded"] += sum(label["node"] == group["node"] and label["direction"] == group["direction"]
                                 for label in additional)
    report = dict(
        contract=dict(question="Do fixed exact safe-memory pieces exclude further whole first-exit slabs?",
                      max_intersections=max_intersections, max_memory_work=max_memory_work,
                      max_memory_arcs=max_memory_arcs, exclusion_query_max_intersections=exclusion_query_max_intersections,
                      max_partition_pieces=max_partition_pieces,
                      max_partition_construction_work=max_partition_construction_work,
                      max_partition_arcs=max_partition_arcs, max_partition_work=max_partition_work,
                      query_max_intersections=query_max_intersections, physical_dynamics_changed=False),
        source=fresh["source"], B59_reconstruction=dict(
            prior_B58_completely_rebuilt=True, prior_cuts_sha256=hashlib.sha256(cuts_bytes).hexdigest(),
            prior_first_exit_slabs_reverified=fresh["excluded_first_exit_slabs"]),
        B59_original_cube_geometry=fresh["B58_original_cube_geometry"],
        B59_exclusion_proposals=tuple(excluded_labels),
        B59_return_envelope=_c6_public_exact_proof(memory.return_envelope),
        B59_memory_envelope=base_public, B59_partition=public,
        B59_partition_queries=tuple(queries), B59_query_relation_complete=result.query_relation_complete,
        B59_query_relation_intersections=result.query_relation_intersections,
        B59_outgoing_facets=fresh["B58_outgoing_facets"], B59_exit_groups=tuple(exit_groups),
        B59_additional_first_exit_facets=tuple(additional), B59_remaining_first_exit_facets=remaining,
        excluded_first_exit_slabs=fresh["excluded_first_exit_slabs"] + len(additional),
        additional_first_exit_slabs_excluded=len(additional), all_first_exit_slabs_excluded=not remaining,
        primary_proof_origin_remains_B47=True, new_conditional_trajectory_steps=0, new_live_graph_steps=0,
        indefinite_trapping_certified=False, whole_band_exit_certified=False,
        future_runtime_certified=False, asymptotic_convergence_certified=False,
    )
    if depth:
        report = {key.replace("B59_", "B60_", 1) if key.startswith("B59_") else key: value
                  for key, value in report.items()}
        report["contract"].update(
            question="Do exact excluded-target predecessor paths refine whole first-exit checks?",
            excluded_predecessor_depth=depth,
        )
    if priority_bytes is not None:
        report = {key.replace("B60_" if depth else "B59_", "B63_", 1)
                  if key.startswith("B60_" if depth else "B59_") else key: value
                  for key, value in report.items()}
        cover = asdict(result.cover)
        cover.pop("partition")
        report["B63_cover"] = cover
        report["B63_cover_verified"] = result.cover.check.certified
        report["B63_priority_evidence"] = dict(
            sha256=hashlib.sha256(priority_bytes).hexdigest(), target=priority_label,
            advisory_only=True, authorizes_cuts=False,
        )
        report["contract"].update(
            question="Does a verified retained cover with advisory priority exclude whole original targets?",
            excluded_predecessor_depth=depth, priority_target=priority_label,
            max_priority_regions=max_priority_regions, max_cover_work=max_cover_work,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/research" / INPUT_NAME)
    parser.add_argument("--relay-input", type=Path, default=ROOT / "artifacts/research" / RELAY_NAME)
    parser.add_argument("--historical-input", type=Path, default=ROOT / "artifacts/research" / HISTORICAL_NAME)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=("backward", "forward", "predecessors", "regions", "excursion", "modes", "returns", "unions", "histories", "memory", "memory-cuts", "safe-partition"), default="backward")
    parser.add_argument("--envelope-input", type=Path, default=ROOT / "artifacts/research/c6_winding_forward_envelope.json")
    parser.add_argument("--witness-input", type=Path, default=ROOT / "artifacts/research/c6_winding_forward_envelope.validation.json")
    parser.add_argument("--max-depth", type=int, default=128)
    parser.add_argument("--max-row-checks", type=int, default=32_768)
    parser.add_argument("--max-work-items", type=int, default=500_000)
    parser.add_argument("--max-boxes", type=int, default=30_000)
    parser.add_argument("--max-intersections", type=int, default=250_000)
    parser.add_argument("--region-input", type=Path, default=ROOT / "artifacts/research/c6_winding_region_exclusions.json")
    parser.add_argument("--max-prefix-steps", type=int, default=4096)
    parser.add_argument("--excursion-input", type=Path, default=ROOT / "artifacts/research/c6_winding_excursion_exclusion.json")
    parser.add_argument("--mode-candidates-input", type=Path, default=ROOT / "benchmarks/c6_winding_mode_excursion_candidates.json")
    parser.add_argument("--mode-input", type=Path, default=ROOT / "artifacts/research/c6_winding_mode_excursions.json")
    parser.add_argument("--max-return-query-intersections", type=int, default=500_000)
    parser.add_argument("--return-input", type=Path, default=ROOT / "artifacts/research/c6_winding_return_regions.json")
    parser.add_argument("--max-return-query-subsumptions", type=int, default=1_000_000)
    parser.add_argument("--max-return-query-zones", type=int, default=5000)
    parser.add_argument("--union-input", type=Path, default=ROOT / "artifacts/research/c6_winding_return_unions.json")
    parser.add_argument("--history-candidates-input", type=Path, default=ROOT / "benchmarks/c6_winding_return_count_candidates.json")
    parser.add_argument("--max-word-work-items", type=int, default=10_000)
    parser.add_argument("--history-input", type=Path, default=ROOT / "artifacts/research/c6_winding_return_histories.json")
    parser.add_argument("--memory-input", type=Path, default=ROOT / "artifacts/research/c6_winding_return_memory.json")
    parser.add_argument("--cuts-input", type=Path, default=ROOT / "artifacts/research/c6_winding_memory_cuts.json")
    parser.add_argument("--max-exclusion-query-intersections", type=int, default=250_000)
    parser.add_argument("--max-partition-pieces", type=int, default=5000)
    parser.add_argument("--max-partition-construction-work", type=int, default=100_000)
    parser.add_argument("--max-partition-arcs", type=int, default=100_000)
    parser.add_argument("--max-partition-work", type=int, default=500_000)
    parser.add_argument("--excluded-predecessor-depth", type=int, default=0,
                        help="Exact return layers before reverified excluded targets; safe-partition only")
    parser.add_argument("--priority-input", type=Path,
                        help="Advisory B59 query report for the verified safe-cover path")
    parser.add_argument("--priority-target", nargs=3, metavar=("MASK", "NODE", "DIRECTION"),
                        help="Original query whose retained B59 regions prioritize complete visits")
    parser.add_argument("--max-priority-regions", type=int, default=5000)
    parser.add_argument("--max-cover-work", type=int, default=100_000)
    parser.add_argument("--max-memory-work", type=int, default=500_000)
    parser.add_argument("--max-memory-arcs", type=int, default=100_000)
    parser.add_argument("--max-memory-query-intersections", type=int, default=250_000)
    args = parser.parse_args()
    from tnfr.physics.c6_carried_return import _return_predecessor_depth

    depth = _return_predecessor_depth(args.excluded_predecessor_depth)
    if depth and args.method != "safe-partition":
        raise ValueError("excluded predecessor depth requires the safe-partition method")
    if (args.priority_input is None) != (args.priority_target is None):
        raise ValueError("priority input and target must be supplied together")
    if args.priority_input is not None and args.method != "safe-partition":
        raise ValueError("priority input requires the safe-partition method")
    if args.output is None:
        name = {"backward": "c6_winding_invariant_region.json", "forward": "c6_winding_forward_envelope.json",
                "predecessors": "c6_winding_temporal_predecessors.json",
                "regions": "c6_winding_region_exclusions.json",
                "excursion": "c6_winding_excursion_exclusion.json",
                "modes": "c6_winding_mode_excursions.json", "returns": "c6_winding_return_regions.json",
                "unions": "c6_winding_return_unions.json", "histories": "c6_winding_return_histories.json",
                "memory": "c6_winding_return_memory.json", "memory-cuts": "c6_winding_memory_cuts.json",
                "safe-partition": "c6_winding_safe_partition.json"}[args.method]
        if depth:
            name = "c6_winding_safe_predecessors.json"
        if args.priority_input is not None:
            name = "c6_winding_safe_cover.json"
        args.output = ROOT / "artifacts/research" / name
    inputs = (args.input, args.relay_input, args.historical_input)
    if args.method == "predecessors":
        inputs += (args.envelope_input, args.witness_input)
    elif args.method == "regions":
        inputs += (args.envelope_input,)
    elif args.method == "excursion":
        inputs += (args.envelope_input, args.region_input)
    elif args.method in ("modes", "returns", "unions", "histories", "memory", "memory-cuts", "safe-partition"):
        inputs += (args.envelope_input, args.region_input, args.excursion_input, args.mode_candidates_input)
        if args.method in ("returns", "unions", "histories", "memory", "memory-cuts", "safe-partition"):
            inputs += (args.mode_input,)
        if args.method in ("unions", "histories", "memory", "memory-cuts", "safe-partition"):
            inputs += (args.return_input,)
        if args.method in ("histories", "memory", "memory-cuts", "safe-partition"):
            inputs += (args.union_input, args.history_candidates_input)
        if args.method in ("memory", "memory-cuts", "safe-partition"):
            inputs += (args.history_input,)
        if args.method in ("memory-cuts", "safe-partition"):
            inputs += (args.memory_input,)
        if args.method == "safe-partition":
            inputs += (args.cuts_input,)
            if args.priority_input is not None:
                inputs += (args.priority_input,)
    if args.output.resolve() in tuple(path.resolve() for path in inputs):
        raise ValueError("the derived report must not overwrite any retained input")
    input_bytes = tuple(path.read_bytes() for path in inputs)
    raw, relay_raw, historical_raw = input_bytes[:3]
    parent = json.loads(raw)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    if args.method == "backward":
        report = analyze_c6_winding_invariant_region(
            parent, relay_bytes=relay_raw, historical_bytes=historical_raw,
            max_work_items=args.max_work_items, max_boxes=args.max_boxes,
        )
    elif args.method == "forward":
        report = analyze_c6_winding_forward_envelope(
            parent, relay_bytes=relay_raw, historical_bytes=historical_raw,
            max_intersections=args.max_intersections,
        )
    elif args.method == "predecessors":
        report = analyze_c6_winding_temporal_predecessors(
            parent, relay_bytes=relay_raw, historical_bytes=historical_raw,
            envelope_bytes=input_bytes[3], witness_bytes=input_bytes[4],
            max_depth=args.max_depth, max_row_checks=args.max_row_checks,
        )
    elif args.method == "regions":
        report = analyze_c6_winding_region_exclusions(
            parent, relay_bytes=relay_raw, historical_bytes=historical_raw,
            envelope_bytes=input_bytes[3], max_intersections=args.max_intersections,
        )
    elif args.method == "excursion":
        prior = json.loads(input_bytes[4])
        if prior["input_evidence"][0]["sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("the regional evidence has a different retained B47 input")
        report = analyze_c6_winding_excursion_exclusion(
            parent, relay_bytes=relay_raw, historical_bytes=historical_raw,
            envelope_bytes=input_bytes[3], region_bytes=input_bytes[4], max_prefix_steps=args.max_prefix_steps,
        )
    elif args.method == "modes":
        report = analyze_c6_winding_mode_excursions(
            parent, parent_bytes=raw, relay_bytes=relay_raw, historical_bytes=historical_raw,
            envelope_bytes=input_bytes[3], region_bytes=input_bytes[4], excursion_bytes=input_bytes[5],
            candidates_bytes=input_bytes[6], max_intersections=args.max_intersections,
            max_prefix_steps=args.max_prefix_steps,
        )
    elif args.method == "returns":
        report = analyze_c6_winding_return_regions(
            parent, parent_bytes=raw, relay_bytes=relay_raw, historical_bytes=historical_raw,
            envelope_bytes=input_bytes[3], region_bytes=input_bytes[4], excursion_bytes=input_bytes[5],
            candidates_bytes=input_bytes[6], mode_bytes=input_bytes[7], max_intersections=args.max_intersections,
            query_max_intersections=args.max_return_query_intersections,
        )
    elif args.method == "unions":
        report = analyze_c6_winding_return_unions(
            parent, parent_bytes=raw, relay_bytes=relay_raw, historical_bytes=historical_raw,
            envelope_bytes=input_bytes[3], region_bytes=input_bytes[4], excursion_bytes=input_bytes[5],
            candidates_bytes=input_bytes[6], mode_bytes=input_bytes[7], return_bytes=input_bytes[8],
            max_intersections=args.max_intersections, query_max_intersections=args.max_return_query_intersections,
            query_max_subsumptions=args.max_return_query_subsumptions, query_max_zones=args.max_return_query_zones,
        )
    elif args.method == "histories":
        report = analyze_c6_winding_return_histories(
            parent, parent_bytes=raw, relay_bytes=relay_raw, historical_bytes=historical_raw,
            envelope_bytes=input_bytes[3], region_bytes=input_bytes[4], excursion_bytes=input_bytes[5],
            candidates_bytes=input_bytes[6], mode_bytes=input_bytes[7], return_bytes=input_bytes[8],
            union_bytes=input_bytes[9], history_candidates_bytes=input_bytes[10],
            max_intersections=args.max_intersections, max_word_work_items=args.max_word_work_items,
        )
    else:
        analyzer = {"memory": analyze_c6_winding_return_memory, "memory-cuts": analyze_c6_winding_memory_cuts,
                    "safe-partition": analyze_c6_winding_safe_partition}[args.method]
        extra = dict(memory_bytes=input_bytes[12]) if args.method in ("memory-cuts", "safe-partition") else {}
        if args.method == "safe-partition":
            extra.update(cuts_bytes=input_bytes[13],
                         exclusion_query_max_intersections=args.max_exclusion_query_intersections,
                         max_partition_pieces=args.max_partition_pieces,
                         max_partition_construction_work=args.max_partition_construction_work,
                         max_partition_arcs=args.max_partition_arcs, max_partition_work=args.max_partition_work)
            if depth:
                extra["excluded_predecessor_depth"] = depth
            if args.priority_input is not None:
                extra.update(priority_bytes=input_bytes[14],
                             priority_target=(int(args.priority_target[0]), int(args.priority_target[1]),
                                              args.priority_target[2]),
                             max_priority_regions=args.max_priority_regions, max_cover_work=args.max_cover_work)
        report = analyzer(
            parent, parent_bytes=raw, relay_bytes=relay_raw, historical_bytes=historical_raw,
            envelope_bytes=input_bytes[3], region_bytes=input_bytes[4], excursion_bytes=input_bytes[5],
            candidates_bytes=input_bytes[6], mode_bytes=input_bytes[7], return_bytes=input_bytes[8],
            union_bytes=input_bytes[9], history_candidates_bytes=input_bytes[10], history_bytes=input_bytes[11],
            max_intersections=args.max_intersections, max_memory_work=args.max_memory_work,
            max_memory_arcs=args.max_memory_arcs,
            query_max_intersections=args.max_memory_query_intersections,
            **extra,
        )
    sha, dirty, digest = provenance
    telemetry = ("exact pressure table", "negative quadratic direction", "positive rational dual certificate",
                 "complete descending sets", "origin membership", "resource outcome")
    if args.method == "forward":
        telemetry = ("exact pressure table", "protected pair strips", "complete descending relational layers",
                     "outgoing facets", "original point membership", "actual entry", "resource outcome")
    if args.method == "predecessors":
        telemetry = ("complete B49 reconstruction", "exact selected outgoing points", "complete predecessor frontiers",
                     "shared-kernel edge identities", "finite-past exclusion versus initial transient")
    if args.method == "regions":
        telemetry = ("complete B49 reconstruction", "full RN cube geometry", "complete unsafe slab cover",
                     "complete regional predecessor layers", "origin exclusion", "first-exit group coverage")
    if args.method == "excursion":
        telemetry = ("reverified prior region exclusions", "exact drift", "dual and primal ingress bounds",
                     "proof-derived initial prefix", "full nodal carry", "whole first-exit group coverage")
    if args.method == "modes":
        telemetry = ("complete B52 reconstruction", "origin-injected reachable hulls", "exact mode-dependent drift",
                     "per-target primal and dual bounds", "whole first-exit coverage", "remaining complete regions")
    if args.method == "returns":
        telemetry = ("complete B53 reconstruction", "exact short-return guards", "intermediate exit preservation",
                     "complete backward return layers", "unchanged origin exclusion", "whole first-exit coverage")
    if args.method == "unions":
        telemetry = ("complete B54 reconstruction", "unjoined transient target preimages", "exact region unions",
                     "complete backward layers", "unchanged origin exclusion", "bounded subsumption work")
    if args.method == "histories":
        telemetry = ("complete B55 reconstruction", "exact cycle displacement generators", "positive balanced circulation",
                     "count witnesses without RN chronology", "joint word guards", "sharp consecutive-repeat budgets")
    if args.method == "memory":
        telemetry = ("complete B56 reconstruction", "exact completed-return endpoint guards", "complete memory relation",
                     "atomic descending worklist updates", "every direct and terminal transient target", "resource scope")
    if args.method == "memory-cuts":
        telemetry = ("complete B57 reconstruction", "proved whole-slab complements", "complete clipped query graph",
                     "all original target initializations", "exact operator equality versus resource abstention")
    if args.method == "safe-partition":
        telemetry = ("complete B58 reconstruction", "reverified whole-target exclusions", "unjoined exact bad pieces",
                     "disjoint safe seeds", "persistent seed clipping", "complete original target queries")
        if depth:
            telemetry += ("exact guarded predecessor words", "complete predecessor layers", "bounded unjoined cuts")
        if args.priority_input is not None:
            telemetry += ("advisory-only priority with charged classification", "complete retained-cover image check",
                          "canonical premise reconstruction", "separate validation and query costs")
    claim = {"backward": "O3.a-C6-carried-invariant-region", "forward": "O3.a-C6-carried-forward-envelope",
             "predecessors": "O3.a-C6-carried-temporal-predecessors",
             "regions": "O3.a-C6-carried-region-exclusions",
             "excursion": "O3.a-C6-carried-excursion-exclusion",
             "modes": "O3.a-C6-carried-mode-excursions", "returns": "O3.a-C6-carried-return-regions",
             "unions": "O3.a-C6-carried-return-unions", "histories": "O3.a-C6-carried-return-histories",
             "memory": "O3.a-C6-carried-return-memory", "memory-cuts": "O3.a-C6-carried-memory-cuts",
             "safe-partition": "O3.a-C6-carried-safe-partition"}[args.method]
    solver = {"backward": "Canonical refreshed pressure with exact carried-map preimages and complete set inclusion",
              "forward": "Canonical carried translations and exact relational past-image envelopes",
              "predecessors": "Canonical inverse carried steps and complete exact target-point predecessor frontiers",
              "regions": "Canonical inverse translations with complete regional predecessor outer envelopes",
              "excursion": "Canonical carried nodal excursion with exact ingress separation and a derived finite prefix",
              "modes": "Canonical carried translations, origin-injected reachable hulls and exact cell-offset excursion certificates",
              "returns": "Canonical complete short-return guards and whole-region backward exclusion",
              "unions": "Canonical guarded returns and exact unions of whole-region predecessors",
              "histories": "Canonical return-count identities and exact jointly guarded word repetition",
              "memory": "Canonical joint carried-state endpoint covers indexed by the last completed return",
              "memory-cuts": "Canonical proved-slab cuts and exact whole-target query-operator comparison",
              "safe-partition": "Canonical reverified bad preimages, disjoint safe-memory seeds and bounded whole-target queries"}[args.method]
    if depth:
        claim = "O3.a-C6-carried-safe-predecessors"
        solver = "Canonical exact excluded-target predecessor paths with persistent disjoint safe-memory seeds"
    if args.priority_input is not None:
        claim = "O3.a-C6-carried-safe-cover"
        solver = "Canonical rebuilt safe seeds, advisory priority and verified complete retained-cover image inclusion"
    manifest = CoreExperimentManifest(
        claim_id=claim, git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Fixed unit C6; retained B43/B46/B47 numerical lineage fully replayed",
        capacity_specification="Conditional unit capacity; historical terminal SHA is not undone",
        solver=solver,
        result_status=ClaimStatus.DERIVED,
        telemetry=telemetry,
        controls=("full incoming carry preserved", "no sampled recurrence; proof-derived entry only when required",
                  "unfinished descents never certify invariance", "numerical proof differs from live graph provenance"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance
            or tuple(path.read_bytes() for path in inputs) != input_bytes):
        raise RuntimeError("source or retained input changed during the invariant-region audit")
    report.update(manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE, input_evidence=tuple(
        {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in zip(inputs, input_bytes, strict=True)
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote C6 invariant-region report to {args.output}")
    if args.method == "predecessors":
        print("Audited exact point pasts:", len(report["B50_selected_point_pasts"]))
    elif args.method in ("regions", "excursion", "modes", "returns", "unions", "histories", "memory", "memory-cuts", "safe-partition"):
        key = {"regions": "B51_outgoing_facets", "excursion": "B52_outgoing_facets", "modes": "B53_outgoing_facets",
               "returns": "B54_outgoing_facets", "unions": "B55_outgoing_facets",
               "histories": "B56_outgoing_facets", "memory": "B57_outgoing_facets", "memory-cuts": "B58_outgoing_facets",
               "safe-partition": "B59_outgoing_facets"}[args.method]
        if depth:
            key = "B60_outgoing_facets"
        if args.priority_input is not None:
            key = "B63_outgoing_facets"
        print("Excluded first-exit slabs:", report["excluded_first_exit_slabs"], "/", len(report[key]))
    else:
        key = "B48_exact_correlated_viability" if args.method == "backward" else "B49_relational_forward_envelope"
        print("Outcome:", report[key]["status"])


if __name__ == "__main__":
    main()
