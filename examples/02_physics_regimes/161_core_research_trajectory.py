"""Measure restricted S16 persistence along a pure-EPI trajectory.

The experiment integrates the fixed linear nodal channel

    dEPI/dt = nu_f * DeltaNFR,    DeltaNFR = -L_rw EPI,

on one connected symmetric weighted path.  Every frozen snapshot stores the
pressure implied by its current EPI and the corresponding nodal derivative.
The same initial state is evolved with a coarse explicit-Euler mesh and its
exact half-step refinement.  Comparisons at common times use persistent node
ids directly; quotient structural distance is retained only as telemetry.
Both meshes are also compared with the fixed linear semigroup evaluated through
the reversible generator's symmetric similarity transform.

This is a finite two-mesh measurement.  Agreement does not establish a limit,
an order of convergence, or persistence under phase dynamics, operators,
changing support, nonlinear pressure, or REMESH.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx

import tnfr
from tnfr.mathematics.unified_numerical import np
from tnfr.physics.core_research_trajectory import (
    compare_core_research_trajectory_refinement,
)
from tnfr.physics.structural_diffusion import (
    diagnose_euler_relaxation_window,
    relaxation_spectrum,
    structural_diffusion_operator,
)
from tnfr.physics.structural_state_distance import StructuralChannelScales
from tnfr.research import (
    ClaimStatus,
    CoreExperimentManifest,
    current_git_source_provenance,
)


NODES = (0, 1, 2, 3)
EDGE_WEIGHTS = (2.0, 0.75, 2.0)
INITIAL_EPI = (2.0, -1.0, 3.0, 0.5)
FREQUENCY = (0.5, 1.5, 1.5, 0.5)
PHASE = (0.0, 0.2, 0.4, 0.6)
PARTITION = ((0, 3), (1, 2))
NODE_LABEL_ATTRIBUTES = ("persistent_identity",)
REFINEMENT_RATIO = 2
EULER_LIMIT_FRACTION = 0.25
NUMERICAL_TOLERANCE = 1e-10
SPECTRAL_RELATIVE_TOLERANCE = 1e-12
SOURCE_SNAPSHOT_PATHS = (
    "src/tnfr",
    "examples/02_physics_regimes/161_core_research_trajectory.py",
)
SCALES = StructuralChannelScales(
    epi=1.0,
    frequency=1.0,
    phase=math.pi,
    pressure=1.0,
    epi_rate=1.0,
    edge_conductance=1.0,
    edge_length=1.0,
)


def _snapshot_from_epi(template: nx.Graph, epi: Sequence[float]) -> nx.Graph:
    """Freeze one graph state with internally consistent pure-EPI telemetry."""
    snapshot = template.copy()
    nodes = tuple(snapshot)
    field = np.asarray(tuple(epi), dtype=float)
    if field.shape != (len(nodes),) or not np.all(np.isfinite(field)):
        raise ValueError("epi must contain one finite scalar per persistent node")

    for node, value in zip(nodes, field):
        snapshot.nodes[node]["EPI"] = float(value)

    operator_nodes, laplacian = structural_diffusion_operator(snapshot)
    if tuple(operator_nodes) != nodes:
        raise RuntimeError("the diffusion operator changed persistent node order")
    frequency = np.asarray(
        [snapshot.nodes[node]["nu_f"] for node in nodes], dtype=float
    )
    with np.errstate(over="raise", invalid="raise"):
        pressure = -(laplacian @ field)
        epi_rate = frequency * pressure
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(epi_rate)):
        raise ValueError("pure-EPI telemetry exceeded floating-point range")

    for node, delta_nfr, derivative in zip(nodes, pressure, epi_rate):
        snapshot.nodes[node]["delta_nfr"] = float(delta_nfr)
        snapshot.nodes[node]["dEPI_dt"] = float(derivative)
    return snapshot


def reference_state() -> nx.Graph:
    """Construct the fixed weighted path used by both timestep meshes."""
    graph = nx.path_graph(NODES)
    for edge, weight in zip(graph.edges, EDGE_WEIGHTS):
        graph.edges[edge]["weight"] = weight
    for node, frequency, phase in zip(NODES, FREQUENCY, PHASE):
        graph.nodes[node].update(
            nu_f=frequency,
            theta=phase,
            persistent_identity=f"n{node}",
        )
    return _snapshot_from_epi(graph, INITIAL_EPI)


def protocol_parameters(graph: nx.Graph) -> dict[str, float | int]:
    """Derive stable nested meshes and a slow-mode observation horizon."""
    rates = np.asarray(relaxation_spectrum(graph), dtype=float)
    positive_rates = rates[rates > NUMERICAL_TOLERANCE]
    if len(positive_rates) != len(graph) - 1:
        raise ValueError("the protocol requires exactly one consensus mode")

    slowest_rate = float(positive_rates[0])
    fastest_rate = float(positive_rates[-1])
    euler_stability_limit = 2.0 / fastest_rate
    coarse_dt = EULER_LIMIT_FRACTION * euler_stability_limit
    fine_dt = coarse_dt / REFINEMENT_RATIO
    slow_mode_time = 1.0 / slowest_rate
    coarse_steps = int(math.ceil(slow_mode_time / coarse_dt))
    fine_steps = REFINEMENT_RATIO * coarse_steps
    horizon = coarse_steps * coarse_dt

    # The comparison threshold is one coarse-step change at the declared EPI
    # rate scale.  It is dimensionless after division by the EPI scale.
    refinement_tolerance = coarse_dt * SCALES.epi_rate / SCALES.epi
    return {
        "slowest_decay_rate": slowest_rate,
        "fastest_decay_rate": fastest_rate,
        "euler_stability_limit": euler_stability_limit,
        "coarse_dt": coarse_dt,
        "fine_dt": fine_dt,
        "coarse_steps": coarse_steps,
        "fine_steps": fine_steps,
        "slow_mode_time": slow_mode_time,
        "horizon": horizon,
        "refinement_tolerance": refinement_tolerance,
    }


def integrate_forward_euler(
    initial: nx.Graph,
    *,
    dt: float,
    step_count: int,
) -> tuple[tuple[nx.Graph, ...], tuple[float, ...]]:
    """Integrate the nodal equation and freeze every endpoint snapshot."""
    if isinstance(step_count, bool) or not isinstance(step_count, int):
        raise ValueError("step_count must be a positive integer")
    if step_count <= 0:
        raise ValueError("step_count must be a positive integer")
    if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    nodes = tuple(initial)
    state = np.asarray(
        [initial.nodes[node]["EPI"] for node in nodes], dtype=float
    )
    snapshots: list[nx.Graph] = []
    times: list[float] = []
    for step in range(step_count + 1):
        snapshot = _snapshot_from_epi(initial, state)
        snapshots.append(snapshot)
        times.append(float(step * dt))
        if step == step_count:
            break
        epi_rate = np.asarray(
            [snapshot.nodes[node]["dEPI_dt"] for node in nodes], dtype=float
        )
        with np.errstate(over="raise", invalid="raise"):
            state = state + dt * epi_rate
        if not np.all(np.isfinite(state)):
            raise ValueError("explicit-Euler state exceeded floating-point range")
    return tuple(snapshots), tuple(times)


def snapshot_consistency(snapshot: nx.Graph) -> dict[str, float]:
    """Measure stored pressure and nodal-rate residuals for one snapshot."""
    nodes, laplacian = structural_diffusion_operator(snapshot)
    epi = np.asarray([snapshot.nodes[node]["EPI"] for node in nodes], dtype=float)
    frequency = np.asarray(
        [snapshot.nodes[node]["nu_f"] for node in nodes], dtype=float
    )
    stored_pressure = np.asarray(
        [snapshot.nodes[node]["delta_nfr"] for node in nodes], dtype=float
    )
    stored_rate = np.asarray(
        [snapshot.nodes[node]["dEPI_dt"] for node in nodes], dtype=float
    )
    expected_pressure = -(laplacian @ epi)
    expected_rate = frequency * stored_pressure
    return {
        "pure_epi_pressure_residual_linf": float(
            np.max(np.abs(stored_pressure - expected_pressure), initial=0.0)
        ),
        "nodal_equation_residual_linf": float(
            np.max(np.abs(stored_rate - expected_rate), initial=0.0)
        ),
    }


def exact_pure_epi_state(initial: nx.Graph, time: float) -> tuple[float, ...]:
    r"""Evaluate ``exp(-A*time) EPI(0)`` through a symmetric similarity.

    For ``A=H^-1 B`` and ``H=diag(d_i/nu_f_i)``, the symmetric generator is
    ``S=H^-1/2 B H^-1/2``.  Therefore

    ``exp(-A t) = H^-1/2 exp(-S t) H^1/2``.

    The formula is exact for the declared fixed linear model; its NumPy
    eigendecomposition is still a floating-point evaluation.
    """
    if isinstance(time, bool) or not math.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and nonnegative")
    nodes = tuple(initial)
    adjacency = nx.to_numpy_array(
        initial, nodelist=nodes, weight="weight", dtype=float
    )
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("the exact reference requires symmetric conductance")
    strength = adjacency.sum(axis=1)
    frequency = np.asarray(
        [initial.nodes[node]["nu_f"] for node in nodes], dtype=float
    )
    if np.any(strength <= 0.0) or np.any(frequency <= 0.0):
        raise ValueError("the exact reference requires positive degree and nu_f")

    mobility_root = np.sqrt(frequency / strength)
    combinatorial_laplacian = np.diag(strength) - adjacency
    symmetric_generator = (
        mobility_root[:, None]
        * combinatorial_laplacian
        * mobility_root[None, :]
    )
    rates, modes = np.linalg.eigh(symmetric_generator)
    rates = np.maximum(rates, 0.0)
    initial_epi = np.asarray(
        [initial.nodes[node]["EPI"] for node in nodes], dtype=float
    )
    transformed_initial = initial_epi / mobility_root
    transformed_state = modes @ (
        np.exp(-rates * time) * (modes.T @ transformed_initial)
    )
    state = mobility_root * transformed_state
    if not np.all(np.isfinite(state)):
        raise ValueError("the exact pure-EPI reference is not finite")
    return tuple(float(value) for value in state)


def exact_reference_diagnostics(
    initial: nx.Graph,
    coarse_snapshots: Sequence[nx.Graph],
    fine_snapshots: Sequence[nx.Graph],
    comparison: Any,
) -> dict[str, Any]:
    """Compare both Euler meshes with the fixed-generator semigroup."""
    nodes = tuple(initial)
    samples: list[dict[str, Any]] = []
    for match in comparison.samples:
        reference = np.asarray(
            exact_pure_epi_state(initial, float(match.coarse_time)), dtype=float
        )
        coarse_epi = np.asarray(
            [
                coarse_snapshots[match.coarse_index].nodes[node]["EPI"]
                for node in nodes
            ],
            dtype=float,
        )
        fine_epi = np.asarray(
            [
                fine_snapshots[match.fine_index].nodes[node]["EPI"]
                for node in nodes
            ],
            dtype=float,
        )
        coarse_error = float(
            np.max(np.abs(coarse_epi - reference), initial=0.0)
        )
        fine_error = float(np.max(np.abs(fine_epi - reference), initial=0.0))
        is_initial = match.coarse_index == 0
        samples.append(
            {
                "time": float(match.coarse_time),
                "coarse_error_to_exact_linf": coarse_error,
                "fine_error_to_exact_linf": fine_error,
                "scaled_coarse_error_to_exact_linf": coarse_error / SCALES.epi,
                "scaled_fine_error_to_exact_linf": fine_error / SCALES.epi,
                "fine_strictly_closer": None
                if is_initial
                else fine_error < coarse_error,
            }
        )

    noninitial = [sample for sample in samples if sample["time"] > 0.0]
    initial_range = max(INITIAL_EPI) - min(INITIAL_EPI)
    maximum_coarse_error = max(
        sample["coarse_error_to_exact_linf"] for sample in samples
    )
    maximum_fine_error = max(
        sample["fine_error_to_exact_linf"] for sample in samples
    )
    representative_indices = sorted({0, len(samples) // 2, len(samples) - 1})
    return {
        "method": (
            "fixed-generator semigroup via the symmetric similarity "
            "H^-1/2 B H^-1/2 and numpy.linalg.eigh"
        ),
        "maximum_coarse_error_to_exact_linf": maximum_coarse_error,
        "maximum_fine_error_to_exact_linf": maximum_fine_error,
        "maximum_coarse_error_fraction_of_initial_range": (
            maximum_coarse_error / initial_range
        ),
        "maximum_fine_error_fraction_of_initial_range": (
            maximum_fine_error / initial_range
        ),
        "fine_strictly_closer_at_every_noninitial_common_time": all(
            bool(sample["fine_strictly_closer"]) for sample in noninitial
        ),
        "common_time_sample_count": len(samples),
        "full_error_series_digest": _content_digest(samples),
        "representative_samples": [
            samples[index] for index in representative_indices
        ],
        "scope": (
            "floating-point evaluation of the exact semigroup formula for this "
            "fixed symmetric linear pure-EPI generator"
        ),
    }


def _snapshot_record(snapshot: nx.Graph, time: float) -> dict[str, Any]:
    """Return compact identity-preserving telemetry for one frozen state."""
    nodes = tuple(snapshot)
    return {
        "time": float(time),
        "node_order": list(nodes),
        "persistent_identity": [
            snapshot.nodes[node]["persistent_identity"] for node in nodes
        ],
        "EPI": [float(snapshot.nodes[node]["EPI"]) for node in nodes],
        "nu_f": [float(snapshot.nodes[node]["nu_f"]) for node in nodes],
        "phase": [float(snapshot.nodes[node]["theta"]) for node in nodes],
        "DeltaNFR": [
            float(snapshot.nodes[node]["delta_nfr"]) for node in nodes
        ],
        "dEPI_dt": [
            float(snapshot.nodes[node]["dEPI_dt"]) for node in nodes
        ],
    }


def _trajectory_digest(
    snapshots: Sequence[nx.Graph], times: Sequence[float]
) -> str:
    """Content-address the complete in-memory trajectory telemetry."""
    records = [
        _snapshot_record(snapshot, time)
        for snapshot, time in zip(snapshots, times)
    ]
    return _content_digest(records)


def _content_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest of a JSON-compatible value."""
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _representative_snapshots(
    snapshots: Sequence[nx.Graph], times: Sequence[float]
) -> list[dict[str, Any]]:
    """Expose initial, midpoint and final telemetry without bloating stdout."""
    indices = sorted({0, len(snapshots) // 2, len(snapshots) - 1})
    return [_snapshot_record(snapshots[index], times[index]) for index in indices]


def _trajectory_summary(certificate: Any) -> dict[str, Any]:
    """Select the temporal decisions needed to audit this experiment."""
    intervals = tuple(certificate.intervals)
    switching = certificate.switching_stability
    return {
        "snapshot_count": len(certificate.times),
        "interval_count": len(intervals),
        "joint_temporal_conditions_pass": bool(
            certificate.joint_temporal_conditions_pass
        ),
        "first_failed_interval_index": certificate.first_failed_interval_index,
        "first_failed_condition": certificate.first_failed_condition,
        "fixed_transport_operator": bool(certificate.fixed_transport_operator),
        "maximum_transport_operator_entry_difference": float(
            certificate.maximum_transport_operator_entry_difference
        ),
        "conditions": {
            name: bool(passed)
            for name, passed in certificate.numerical_conditions
        },
        "failed_conditions": list(certificate.failed_conditions),
        "all_intervals_euler_stable": all(
            interval.euler_relaxation.is_euler_stable
            for interval in intervals
        ),
        "maximum_interval_dt_over_euler_limit": max(
            (
                float(
                    interval.dt
                    / interval.euler_relaxation.euler_stability_limit
                )
                for interval in intervals
            ),
            default=0.0,
        ),
        "maximum_scaled_epi_update_residual": max(
            (
                float(interval.scaled_epi_update_residual_linf)
                for interval in intervals
            ),
            default=0.0,
        ),
        "common_lyapunov_initial": float(certificate.common_lyapunov_values[0]),
        "common_lyapunov_final": float(certificate.common_lyapunov_values[-1]),
        "maximum_scaled_common_lyapunov_increase": float(
            certificate.maximum_scaled_common_lyapunov_increase
        ),
        "cumulative_scaled_common_lyapunov_positive_variation": float(
            certificate.cumulative_scaled_common_lyapunov_positive_variation
        ),
        "spectral_relative_tolerance": float(certificate.spectral_tolerance),
        "failed_intervals": [
            {
                "index": int(interval.index),
                "left_time": float(interval.left_time),
                "right_time": float(interval.right_time),
                "failed_conditions": list(interval.failed_conditions),
            }
            for interval in intervals
            if not interval.interval_conditions_pass
        ],
        "switching_stability": {
            "regime_count": int(switching.regime_count),
            "shares_exact_common_metric": bool(
                switching.shares_exact_common_metric
            ),
            "common_metric_residual": float(
                switching.common_metric_residual
            ),
            "uniform_exponential_rate": float(
                switching.uniform_exponential_rate
            ),
            "supports_exact_switching_theorem": bool(
                switching.supports_exact_switching_theorem
            ),
        },
        "scope": certificate.scope,
    }


def _json_ready(value: Any) -> Any:
    """Convert small comparison payloads to JSON-compatible Python values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return str(value)


def _refinement_summary(comparison: Any) -> dict[str, Any]:
    """Expose direct persistent-id errors and diagnostic quotient distances."""
    samples = []
    for sample in comparison.samples:
        quotient = sample.quotient_structural_distance
        if hasattr(quotient, "distance"):
            quotient = quotient.distance
        samples.append(
            {
                "coarse_time": float(sample.coarse_time),
                "fine_time": float(sample.fine_time),
                "direct_epi_errors_by_persistent_id": {
                    str(node): float(error)
                    for node, error in sample.direct_epi_differences
                },
                "direct_epi_error_linf": float(sample.direct_epi_error_linf),
                "scaled_direct_epi_error_linf": float(
                    sample.scaled_direct_epi_error_linf
                ),
                "quotient_structural_distance_diagnostic": float(quotient),
            }
        )
    representative_indices = sorted({0, len(samples) // 2, len(samples) - 1})
    return {
        "common_time_sample_count": len(samples),
        "common_time_interval": {
            "start": _json_ready(comparison.common_time_pairs[0]),
            "end": _json_ready(comparison.common_time_pairs[-1]),
        },
        "common_time_pairs_digest": _content_digest(
            _json_ready(comparison.common_time_pairs)
        ),
        "all_coarse_times_matched": bool(comparison.all_coarse_times_matched),
        "maximum_scaled_direct_epi_error": float(
            comparison.maximum_scaled_direct_epi_error
        ),
        "maximum_coarse_step": float(comparison.maximum_coarse_step),
        "maximum_fine_step": float(comparison.maximum_fine_step),
        "fine_grid_is_strict_refinement": bool(
            comparison.fine_grid_is_strict_refinement
        ),
        "common_time_agreement_within_tolerance": bool(
            comparison.common_time_agreement_within_tolerance
        ),
        "same_dynamics_declared": bool(comparison.same_dynamics_declared),
        "conditions": {
            name: bool(passed)
            for name, passed in comparison.numerical_conditions
        },
        "joint_refinement_conditions_pass": bool(
            comparison.joint_refinement_conditions_pass
        ),
        "failed_conditions": list(comparison.failed_conditions),
        "trajectory_tolerance": float(comparison.trajectory_tolerance),
        "spectral_relative_tolerance": float(comparison.spectral_tolerance),
        "agreement_tolerance": float(comparison.agreement_tolerance),
        "time_tolerance": float(comparison.time_tolerance),
        "full_direct_error_series_digest": _content_digest(samples),
        "representative_samples": [
            samples[index] for index in representative_indices
        ],
        "scope": comparison.scope,
    }


def run_protocol() -> dict[str, Any]:
    """Run both meshes, their temporal certificates and direct comparison."""
    initial = reference_state()
    parameters = protocol_parameters(initial)
    coarse_snapshots, coarse_times = integrate_forward_euler(
        initial,
        dt=float(parameters["coarse_dt"]),
        step_count=int(parameters["coarse_steps"]),
    )
    fine_snapshots, fine_times = integrate_forward_euler(
        initial,
        dt=float(parameters["fine_dt"]),
        step_count=int(parameters["fine_steps"]),
    )

    comparison = compare_core_research_trajectory_refinement(
        coarse_snapshots,
        coarse_times,
        fine_snapshots,
        fine_times,
        PARTITION,
        scales=SCALES,
        agreement_tolerance=float(parameters["refinement_tolerance"]),
        same_dynamics_declared=True,
        trajectory_tolerance=NUMERICAL_TOLERANCE,
        spectral_tolerance=SPECTRAL_RELATIVE_TOLERANCE,
        time_tolerance=0.0,
        node_label_attributes=NODE_LABEL_ATTRIBUTES,
    )
    coarse_certificate = comparison.coarse_trajectory
    fine_certificate = comparison.fine_trajectory
    exact_reference = exact_reference_diagnostics(
        initial, coarse_snapshots, fine_snapshots, comparison
    )

    coarse_diagnostic = diagnose_euler_relaxation_window(
        initial,
        dt=float(parameters["coarse_dt"]),
        tolerance=SPECTRAL_RELATIVE_TOLERANCE,
    )
    fine_diagnostic = diagnose_euler_relaxation_window(
        initial,
        dt=float(parameters["fine_dt"]),
        tolerance=SPECTRAL_RELATIVE_TOLERANCE,
    )
    all_snapshots = coarse_snapshots + fine_snapshots
    residuals = [snapshot_consistency(snapshot) for snapshot in all_snapshots]
    maximum_pressure_residual = max(
        item["pure_epi_pressure_residual_linf"] for item in residuals
    )
    maximum_nodal_residual = max(
        item["nodal_equation_residual_linf"] for item in residuals
    )
    verdicts_agree = (
        coarse_certificate.joint_temporal_conditions_pass
        == fine_certificate.joint_temporal_conditions_pass
    )
    finite_protocol_conditions = (
        (
            "coarse_trajectory_certificate",
            bool(coarse_certificate.joint_temporal_conditions_pass),
        ),
        (
            "fine_trajectory_certificate",
            bool(fine_certificate.joint_temporal_conditions_pass),
        ),
        ("temporal_verdicts_agree", bool(verdicts_agree)),
        (
            "joint_refinement_conditions",
            bool(comparison.joint_refinement_conditions_pass),
        ),
        ("coarse_euler_stable", bool(coarse_diagnostic.is_euler_stable)),
        ("fine_euler_stable", bool(fine_diagnostic.is_euler_stable)),
        (
            "snapshot_pure_epi_pressure_consistency",
            maximum_pressure_residual <= NUMERICAL_TOLERANCE,
        ),
        (
            "snapshot_nodal_equation_consistency",
            maximum_nodal_residual <= NUMERICAL_TOLERANCE,
        ),
        (
            "fine_closer_to_exact_at_noninitial_common_times",
            bool(
                exact_reference[
                    "fine_strictly_closer_at_every_noninitial_common_time"
                ]
            ),
        ),
    )
    finite_protocol_pass = all(passed for _, passed in finite_protocol_conditions)
    return {
        "initial": initial,
        "parameters": parameters,
        "coarse_snapshots": coarse_snapshots,
        "coarse_times": coarse_times,
        "fine_snapshots": fine_snapshots,
        "fine_times": fine_times,
        "coarse_certificate": coarse_certificate,
        "fine_certificate": fine_certificate,
        "comparison": comparison,
        "exact_reference": exact_reference,
        "coarse_diagnostic": coarse_diagnostic,
        "fine_diagnostic": fine_diagnostic,
        "maximum_pressure_residual": maximum_pressure_residual,
        "maximum_nodal_residual": maximum_nodal_residual,
        "verdicts_agree": verdicts_agree,
        "finite_protocol_conditions": finite_protocol_conditions,
        "finite_protocol_pass": finite_protocol_pass,
    }


def build_report(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Build the JSON report and validate its reproducibility manifest."""
    parameters = protocol["parameters"]
    coarse_snapshots = protocol["coarse_snapshots"]
    coarse_times = protocol["coarse_times"]
    fine_snapshots = protocol["fine_snapshots"]
    fine_times = protocol["fine_times"]
    repository = Path(__file__).resolve().parents[2]
    git_sha, source_dirty, dirty_source_hash = current_git_source_provenance(
        repository, SOURCE_SNAPSHOT_PATHS
    )
    manifest = CoreExperimentManifest(
        claim_id="S16-RESTRICTED-PURE-EPI-TRAJECTORY-REFINEMENT",
        git_sha=git_sha,
        versions={"python": platform.python_version(), "tnfr": tnfr.__version__},
        graph_construction=(
            "four-node reflection-symmetric weighted path with persistent ids; "
            "fixed support and conductance"
        ),
        capacity_specification=(
            "positive frozen heterogeneous nu_f=(0.5,1.5,1.5,0.5) Hz_str; "
            "reflection-invariant within quotient fibers"
        ),
        solver=(
            "forward Euler on fixed pure-EPI generator; coarse dt is one quarter "
            "of its strict modal stability limit and fine dt=coarse dt/2"
        ),
        result_status=ClaimStatus.MEASURED,
        seed=None,
        timestep=float(parameters["coarse_dt"]),
        operator_sequence=(),
        telemetry=(
            "EPI",
            "nu_f",
            "phase",
            "DeltaNFR",
            "dEPI",
            "direct persistent-id refinement error",
            "coarse and fine EPI error to the fixed linear semigroup",
        ),
        controls=(
            "per-snapshot pure-EPI pressure residual",
            "per-snapshot nodal-equation residual",
            "graph-specific Euler modal stability bound",
            "exactly nested common-time half-step grid",
            "explicit same fixed dynamics declaration for both meshes",
            "unique persistent node labels",
            "fixed-generator exact-semigroup reference",
        ),
        artifacts=("stdout JSON report", "full-trajectory SHA-256 digests"),
        source_dirty=source_dirty,
        dirty_source_hash=dirty_source_hash,
    )
    manifest.validate_for_admission()

    coarse_diagnostic = protocol["coarse_diagnostic"]
    fine_diagnostic = protocol["fine_diagnostic"]
    report = {
        "claim_id": manifest.claim_id,
        "finite_protocol_pass": bool(protocol["finite_protocol_pass"]),
        "finite_protocol_conditions": {
            name: bool(passed)
            for name, passed in protocol["finite_protocol_conditions"]
        },
        "temporal_verdicts_agree_under_half_step_refinement": bool(
            protocol["verdicts_agree"]
        ),
        "design": {
            "nodes": list(NODES),
            "persistent_identity_mapping": [
                {"coarse": node, "fine": node} for node in NODES
            ],
            "edge_weights": list(EDGE_WEIGHTS),
            "initial_epi": list(INITIAL_EPI),
            "nu_f_hz_str": list(FREQUENCY),
            "fixed_phase": list(PHASE),
            "reversible_reflection_partition": [
                list(block) for block in PARTITION
            ],
            "node_alignment_policy": (
                "direct persistent ids at every common time; no relabeling or "
                "time-dependent quotient minimizer"
            ),
            "same_dynamics_declared": bool(
                protocol["comparison"].same_dynamics_declared
            ),
            "mesh_parameters": {
                name: float(value) if isinstance(value, float) else int(value)
                for name, value in parameters.items()
            },
            "agreement_tolerance_policy": (
                "one coarse-step change at the declared EPI-rate scale; this is "
                "a predeclared resolution scale, not a truncation-error bound"
            ),
        },
        "euler_diagnostics": {
            "coarse": {
                "dt": float(coarse_diagnostic.dt),
                "stability_limit": float(
                    coarse_diagnostic.euler_stability_limit
                ),
                "dt_over_stability_limit": float(
                    coarse_diagnostic.dt
                    / coarse_diagnostic.euler_stability_limit
                ),
                "maximum_modal_factor": float(
                    coarse_diagnostic.maximum_modal_factor
                ),
                "spectral_relative_tolerance": float(
                    coarse_diagnostic.spectral_relative_tolerance
                ),
                "spectral_zero_threshold": float(
                    coarse_diagnostic.spectral_zero_threshold
                ),
                "all_modal_multipliers_nonnegative": bool(
                    np.all(coarse_diagnostic.modal_multipliers >= 0.0)
                ),
                "is_euler_stable": bool(coarse_diagnostic.is_euler_stable),
            },
            "fine": {
                "dt": float(fine_diagnostic.dt),
                "stability_limit": float(fine_diagnostic.euler_stability_limit),
                "dt_over_stability_limit": float(
                    fine_diagnostic.dt / fine_diagnostic.euler_stability_limit
                ),
                "maximum_modal_factor": float(
                    fine_diagnostic.maximum_modal_factor
                ),
                "spectral_relative_tolerance": float(
                    fine_diagnostic.spectral_relative_tolerance
                ),
                "spectral_zero_threshold": float(
                    fine_diagnostic.spectral_zero_threshold
                ),
                "all_modal_multipliers_nonnegative": bool(
                    np.all(fine_diagnostic.modal_multipliers >= 0.0)
                ),
                "is_euler_stable": bool(fine_diagnostic.is_euler_stable),
            },
        },
        "snapshot_telemetry": {
            "maximum_pure_epi_pressure_residual_linf": float(
                protocol["maximum_pressure_residual"]
            ),
            "maximum_nodal_equation_residual_linf": float(
                protocol["maximum_nodal_residual"]
            ),
            "coarse_full_trajectory_digest": _trajectory_digest(
                coarse_snapshots, coarse_times
            ),
            "fine_full_trajectory_digest": _trajectory_digest(
                fine_snapshots, fine_times
            ),
            "coarse_representative_snapshots": _representative_snapshots(
                coarse_snapshots, coarse_times
            ),
            "fine_representative_snapshots": _representative_snapshots(
                fine_snapshots, fine_times
            ),
        },
        "coarse_trajectory": _trajectory_summary(
            protocol["coarse_certificate"]
        ),
        "fine_trajectory": _trajectory_summary(protocol["fine_certificate"]),
        "refinement": _refinement_summary(protocol["comparison"]),
        "exact_linear_reference": protocol["exact_reference"],
        "manifest": manifest.to_dict(),
        "scope": {
            "established": (
                "finite sampled persistence of the restricted S16 intersection "
                "for this fixed pure-EPI path on two stable nested Euler meshes"
            ),
            "interpretation": (
                "the fine mesh is checked against the fixed-generator semigroup "
                "at common times; two meshes still do not establish convergence "
                "or its order"
            ),
            "open": (
                "phase evolution, nonlinear pressure, operator histories, changing "
                "support/topology, REMESH/nesting, adaptive timesteps, and a mesh "
                "limit remain open"
            ),
        },
        "falsifiers": [
            (
                "Any sampled interval fails an endpoint S16 condition or its "
                "forward-Euler nodal update."
            ),
            (
                "The coarse and fine temporal certificate verdicts differ at "
                "the same physical horizon."
            ),
            (
                "A coarse time lacks one exact fine-grid match or persistent "
                "node ids cease to agree."
            ),
            (
                "The maximum direct EPI discrepancy exceeds one coarse-step "
                "reference EPI increment."
            ),
            (
                "The fine mesh is not strictly closer to the fixed-generator "
                "semigroup at every noninitial common time."
            ),
            (
                "A further nested refinement stops reducing direct persistent-id "
                "error or reveals an intermediate hypothesis failure."
            ),
        ],
    }
    return report


def main() -> None:
    """Execute the deterministic protocol and emit its JSON evidence record."""
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
