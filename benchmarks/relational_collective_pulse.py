"""Static causal modes of one supplied opposite-winding return geometry.

This finite eigensystem study uses the native uniform-form tangent, not an
auxiliary oscillator or a trajectory. Numerical poles are not ideal root
certificates; the ideal common-offset quotient retains its represented defect.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
from dataclasses import asdict
from fractions import Fraction as Q
from pathlib import Path

import networkx as nx
import numpy as np

from benchmarks import relational_return_geometry as geometry_owner
from tnfr.dynamics import relational as tangent_owner
from tnfr.dynamics.relational import evaluate_relational_uniform_tangent
from tnfr.mathematics._exact_linear_algebra import exact_matrix_product as product
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.utils.io import json_dumps, safe_write

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "benchmarks/relational_collective_pulse.py",
    "benchmarks/relational_return_geometry.py",
    "src/tnfr/dynamics/relational.py",
    "src/tnfr/mathematics/_exact_linear_algebra.py",
    "src/tnfr/mathematics/_rational_interval.py",
    "src/tnfr/mathematics/_phase_midpoint.py",
    "src/tnfr/mathematics/_phase_resultant_chamber.py",
    "src/tnfr/physics/phase_cycle_geometry.py",
    "src/tnfr/utils/io.py",
)
OBSERVATIONS = (
    "receiver_port_minus_ring_mean_form",
    "receiver_port_minus_ring_mean_phase",
    "right_minus_left_ring_mean_form",
    "right_minus_left_ring_mean_phase",
    "mediator_form_rate",
    "mediator_phase_rate",
)
RESOLVENT_POINT = 1 + 2j


def _pairs(values):
    return [[float(value.real), float(value.imag)] for value in values]


def _encoded(value):
    def exact(item):
        if isinstance(item, Q):
            return {"numerator": item.numerator, "denominator": item.denominator}
        raise TypeError(f"unsupported report value {type(item).__name__}")

    return (
        json_dumps(value, default=exact, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _coordinates(generator):
    """Exact declared difference/lift maps; do not repair generator row sums."""
    visible = tuple(range(10)) + tuple(range(11, 21))
    d = tuple(
        tuple(Q(int(j == i) - int(j == (10 if i < 11 else 21))) for j in range(22))
        for i in visible
    )
    lift = tuple(tuple(Q(int(i == j)) for j in visible) for i in range(22))
    observation = [[Q(0)] * 22 for _ in OBSERVATIONS]
    for channel in range(2):
        offset = 11 * channel
        observation[channel][offset + 5] = Q(1)
        for i in range(5):
            observation[channel][offset + 5 + i] -= Q(1, 5)
            observation[2 + channel][offset + 5 + i] = Q(1, 5)
            observation[2 + channel][offset + i] = -Q(1, 5)
    observation[4], observation[5] = list(generator[10]), list(generator[21])
    return d, lift, tuple(map(tuple, observation))


def _modal_data(generator, initial, observations):
    """Non-Hermitian left/right residues, with every numerical pole retained."""
    matrix, initial, observations = map(
        lambda a: np.asarray(a, dtype=float), (generator, initial, observations)
    )
    values, vectors = np.linalg.eig(matrix)
    inverse = np.linalg.inv(vectors)
    condition = float(np.linalg.cond(vectors))
    if not math.isfinite(condition) or condition * np.finfo(float).eps >= 1:
        raise ArithmeticError("eigenbasis is numerically unresolved; no modal verdict")
    residues = (observations @ vectors) * (inverse @ initial)[None, :]
    pole_tolerance = 128 * np.finfo(float).eps * max(1.0, np.linalg.norm(matrix, 2))
    residue_tolerance = (
        128
        * np.finfo(float).eps
        * max(1.0, np.linalg.norm(observations, 2))
        * condition
    )
    rows = []
    for k in np.lexsort((values.imag, values.real)):
        value = values[k]
        separation = min(abs(value - other) for j, other in enumerate(values) if j != k)
        oscillatory = abs(value.imag) > pole_tolerance
        decay = -value.real if value.real < -pole_tolerance else None
        rows.append(
            {
                "pole": _pairs((value,))[0],
                "residues": _pairs(residues[:, k]),
                "nearest_pole_separation": float(separation),
                "individually_separated": bool(separation > pole_tolerance),
                "eigen_residual_l2": float(
                    np.linalg.norm(matrix @ vectors[:, k] - value * vectors[:, k])
                ),
                "numerically_oscillatory": bool(oscillatory),
                "decay_time": None if decay is None else float(1 / decay),
                "period": (
                    None if not oscillatory else float(2 * math.pi / abs(value.imag))
                ),
                "cycles_per_decay_time": (
                    None
                    if decay is None or not oscillatory
                    else float(abs(value.imag) / (2 * math.pi * decay))
                ),
            }
        )
    modal = np.sum(residues / (RESOLVENT_POINT - values)[None, :], axis=1)
    direct = observations @ np.linalg.solve(
        RESOLVENT_POINT * np.eye(len(matrix)) - matrix, initial
    )
    return {
        "poles": rows,
        "eigenbasis_condition": condition,
        "pole_zero_tolerance": float(pole_tolerance),
        "residue_zero_tolerance": float(residue_tolerance),
        "numerical_policy": "128*binary64_epsilon_scaled_by_matrix_or_observation_norm; residue_scale_includes_eigenbasis_condition; not_physical_threshold",
        "resolvent_point": _pairs((RESOLVENT_POINT,))[0],
        "direct_resolvent": _pairs(direct),
        "modal_resolvent": _pairs(modal),
        "resolvent_reassembly_error_linf": float(max(abs(direct - modal))),
        "degeneracy_scope": "individual_residues_of_unseparated_poles_are_basis_dependent; their_sum_enters_the_transfer",
    }


def _causal_controls(generator, initial, observation):
    groups = tuple(0 if i % 11 < 5 else 1 if i % 11 < 10 else 2 for i in range(22))
    null = tuple(
        tuple(value if groups[i] == groups[j] else Q(0) for j, value in enumerate(row))
        for i, row in enumerate(generator)
    )
    frozen = tuple(
        (Q(0),) * 22 if i in (10, 21) else row for i, row in enumerate(generator)
    )
    seed = tuple((value,) for value in initial)
    second = product(observation[:2], product(frozen, product(frozen, seed)))
    numeric_null = np.asarray(null, dtype=float)
    response = np.asarray(observation[:2], dtype=float) @ np.linalg.solve(
        RESOLVENT_POINT * np.eye(22) - numeric_null, np.asarray(initial, dtype=float)
    )
    return {
        "block_diagonal_generator": null,
        "groups": groups,
        "cross_blocks_exactly_zero": all(
            null[i][j] == 0
            for i in range(22)
            for j in range(22)
            if groups[i] != groups[j]
        ),
        "null_receiver_resolvent": _pairs(response),
        "frozen_mediator_receiver_second_moment": tuple(row[0] for row in second),
        "scope": "full_state_linear_ablation_clamps_cross_region_deviations; not_native_disconnected_geometry; donor_subspace_invariant_implies_zero_receiver_transfer_at_all_orders; frozen_mediator_keeps_return_route",
    }


def _fingerprints():
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _trace_cube_certificate(root):
    """Enclose the proved opposite-sector trace identity at the ideal root.

    On this triangle-free unit graph tr((D^-1 B)^3)=25. The edge-class
    sum below is tr(D^-1 B S^-1 K S^-1 B), with S_i=sum_j cos(delta_ij).
    Combined with the existing Hurwitz quotient theorem, positive tr(J^3)
    excludes an entirely real spectrum. It does not certify modal residues.
    """
    pi = pi_interval()
    angle = 2 * pi * I(root.lower, root.upper)
    z, r, b = sin(angle / 4), cos(angle), cos(2 * angle / 3)
    s = z + r + b
    mixed_trace = 29 / s + 38 / (3 * z) + 4 / (3 * b) + 4 * (2 * r + b) / s**2
    trace = -Q(25, 8) + 3 * mixed_trace / (8 * pi**2)
    return {
        "mixed_trace": (mixed_trace.lo, mixed_trace.hi),
        "trace_cube": (trace.lo, trace.hi),
        "positive_lower_bound": trace.lo > 0,
        "scope": "ideal_root_interval_and_default_positive_unit_capacity_law; Hurwitz_quotient_plus_positive_cube_trace_proves_at_least_one_damped_complex_pair; no_causal_residue_or_specific_pole_enclosure",
    }


def analyze_collective_pulse():
    """Read one static native tangent and its causal modal contribution."""
    for module, path in (
        (geometry_owner, SOURCE_PATHS[1]),
        (tangent_owner, SOURCE_PATHS[2]),
    ):
        if Path(module.__file__).resolve() != (ROOT / path).resolve():
            raise ValueError(
                "imported owner differs from fingerprinted workspace source"
            )
    fingerprints = _fingerprints()
    geometry = geometry_owner.analyze_return_geometry()
    field = geometry.opposite.native_field
    graph = nx.Graph()
    graph.add_nodes_from(field.nodes)
    graph.add_edges_from(field.edges, weight=1.0)
    for i, node in enumerate(field.nodes):
        graph.nodes[node].update(
            EPI=field.epi[i], theta=field.phase[i], nu_f=field.capacity[i]
        )
    tangent = evaluate_relational_uniform_tangent(graph, model=field.model)
    generator = tuple(tuple(map(Q, row)) for row in tangent.generator)
    d, lift, observation = _coordinates(generator)
    reduced = product(d, product(generator, lift))
    lhs, rhs = product(d, generator), product(reduced, d)
    quotient_defect = tuple(
        tuple(a - b for a, b in zip(left, right, strict=True))
        for left, right in zip(lhs, rhs, strict=True)
    )
    initial = (Q(1),) + (Q(0),) * 21
    reduced_initial = tuple(row[0] for row in product(d, tuple((v,) for v in initial)))
    reduced_observation = product(observation, lift)
    modes = _modal_data(reduced, reduced_initial, reduced_observation)
    full_response = np.asarray(observation, dtype=float) @ np.linalg.solve(
        RESOLVENT_POINT * np.eye(22) - np.asarray(generator, dtype=float),
        np.asarray(initial, dtype=float),
    )
    quotient_response = np.array(
        [complex(*value) for value in modes["direct_resolvent"]]
    )
    if fingerprints != _fingerprints():
        raise ValueError("source changed during static analysis")
    return {
        "protocol": "opposite_return_collective_pulse_static_v1",
        "snapshot": {
            "nodes": field.nodes,
            "edges": field.edges,
            "epi": field.epi,
            "phase": field.phase,
            "capacity": field.capacity,
            "model": asdict(field.model),
            "pressure_path": field.pressure_path,
            "form_rate": field.form_rate,
            "phase_rate": field.phase_rate,
            "geometry_status": geometry.opposite.status,
            "root_bracket": asdict(geometry.opposite_root),
        },
        "generator": generator,
        "input": initial,
        "observation_names": OBSERVATIONS,
        "observations": observation,
        "difference_map": d,
        "lift": lift,
        "quotient_generator": reduced,
        "quotient_observations": reduced_observation,
        "quotient_input": reduced_initial,
        "common_offset_residuals": tangent.common_offset_residuals,
        "quotient_identity_residual": quotient_defect,
        "full_resolvent": _pairs(full_response),
        "full_vs_quotient_resolvent_error_linf": float(
            max(abs(full_response - quotient_response))
        ),
        "modal": modes,
        "controls": _causal_controls(generator, initial, observation),
        "ideal_trace_cube_certificate": _trace_cube_certificate(geometry.opposite_root),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "backend": "numpy.linalg_eig_inv_solve_on_CPU; native_fused_pressure",
            "precision": "binary64_nonnormal_eigensystem; exact_represented_maps_and_defects",
            "randomness": "none",
        },
        "source_sha256": fingerprints,
        "source_scope": "partial_listed_owner_fingerprints_not_full_source_archive_or_provenance_authentication",
        "scope": "unit_tangent_initial_direction_not_finite_nonlinear_preparation; materialized_uniform_form_derivative_near_ideal_equilibrium; quotient_rounding_defect_retained; all_poles_not_cherry_picked; ideal_complex_pair_existence_only_not_certified_pole_locations_or_causal_residues; no_trajectory_sustained_pulse_support_birth_or_physical_identification",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError("refusing to replace retained static evidence")
    report = analyze_collective_pulse()
    encoded = _encoded(report)
    if args.output is None:
        print(encoded, end="")
    else:
        safe_write(
            args.output,
            lambda stream: stream.write(encoded),
            mode="x",
            atomic=False,
            sync=True,
            newline="\n",
        )
        print(f"Static modal evidence written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
