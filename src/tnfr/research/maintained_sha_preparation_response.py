"""One fixed preparation-family bound associated with both nominal proofs.

The public boundary re-admits the SHA center, verifies receipt-bound saved
observations with their original arithmetic owner, and only then constructs
adjacent-coordinate hulls. The inherited trajectory and derivative premises
remain those of the frozen nominal producers. No nominal motion is replayed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from fractions import Fraction as Q
from pathlib import Path
from types import ModuleType

from .._exact_time import finite_represented_real
from ..mathematics._rational_interval import I, pi_interval
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    read_bytes_bounded,
    sha256_bytes,
    validate_artifact_path,
)
from .local_capacity_history import MODEL, TIMES
from .maintained_parent_thol import _signature
from .maintained_sha_cut_separator import CutSeparatorSource, _capture_cut_source
from .maintained_sha_motion_source import validate_live_sha_successor
from .sine_preparation_bounds import (
    AdjacentCoordinateHull,
    TwoBlockErrorBound,
    adjacent_coordinate_hull,
    bound_two_block_errors,
    connected_marked_error_squared,
    cut_marked_error_squared,
)

HORIZON = Q(1, 2)
COMPARISON_STEPS = 64
MAX_RECORD_BYTES = 16 * 1024**2
VERIFIER_SHA256 = "d0a1f2ac10fc284b5f8d3a19f09f58a58cf52c283f90acf8f9c2e3140ac5d40b"
_CONNECTED = "artifacts/parent-child-direct-response-20261022/"
_CUT = "artifacts/parent-child-cut-direct-response-20261023/"
_PATHS = {
    "connected_result": _CONNECTED + "result.json",
    "connected_final_state": _CONNECTED + "final-state.json",
    "cut_result": _CUT + "result.json",
    "cut_final_state": _CUT + "final-state.json",
    "observation_verifier": _CUT + "compare_connected.py",
}


@dataclass(frozen=True)
class FrozenNominalEvidence:
    """Prospectively supplied byte receipts, not cached scientific success.

    Paths are fixed repository-relative locations. The audit driver verifies
    the complete prior frozen inventories before and after this consumer; this
    boundary binds its five consumed records and the final-state result links.
    Receipt consistency does not independently reprove old derivative bounds.
    """

    repository_root: Path
    connected_result: dict
    connected_final_state: dict
    cut_result: dict
    cut_final_state: dict
    observation_verifier: dict


@dataclass(frozen=True)
class NominalPreparationBounds:
    """Rebuilt complete-window observations with inherited proof provenance."""

    connected_clockwise: I
    cut_clockwise: I
    connected_radius_squared: Q
    cut_radius_squared: Q
    connected_cells_checked: int = 256
    cut_cells_checked: int = 256
    evidence_receipts: tuple[dict, ...] = ()
    premises: tuple[str, ...] = (
        "original_nominal_whole_cell_enclosures_and_derivative_remainders_inherited",
        "fixed_original_half_unit_window_continuous_lifts_and_marked_scale",
        "saved_observation_arithmetic_rebuilt_without_trajectory_or_field_replay",
    )


@dataclass(frozen=True)
class PreparationCoefficients:
    dynamic_indices: tuple[int, ...]
    degrees: tuple[int, ...]
    weighted_row_sums: tuple[Q, ...]
    a_numerator: Q
    b_numerator: Q
    A_upper: Q
    B_upper: Q


@dataclass(frozen=True)
class PreparationGate:
    """A strict sufficient margin; failure is never a family counterexample."""

    margin: Q
    status: str


@dataclass(frozen=True)
class ShaPreparationResponse:
    source: CutSeparatorSource
    nominal: NominalPreparationBounds
    hulls: tuple[AdjacentCoordinateHull, ...]
    family_members: int
    pi: I
    connected_coefficients: PreparationCoefficients
    cut_coefficients: PreparationCoefficients
    connected_comparison: TwoBlockErrorBound
    cut_comparison: TwoBlockErrorBound
    child_form_error: Q
    child_phase_error: Q
    connected_error_squared: Q
    cut_error_squared: Q
    connected_radius_gate: PreparationGate
    cut_radius_gate: PreparationGate
    nominal_gap: I
    nominal_gap_gate: PreparationGate
    status: str
    source_receipts: tuple[dict, ...] = ()
    persistent_source_unchanged: bool = True
    scope: tuple[str, ...] = (
        "same_supplied_post_event_family_initializes_both_complete_held_laws",
        "full_predecessor_center_successor_family_and_closed_real_hull",
        "all_forms_and_registered_real_phase_lifts_no_mean_projection",
        "cut_child_constant_at_each_members_own_initial_coordinates",
        "fixed_64_nonnegative_comparison_steps_per_branch_no_retry_or_width_search",
        "strict_squared_radius_and_nominal_gap_above_pi_sufficient_gates",
        "common_initial_angle_error_cancels_without_inverse_trigonometric_estimator",
        "insufficient_bound_neither_observes_radius_crossing_nor_reverses_ordering",
        "nominal_derivative_premises_inherited_from_frozen_point_certificates",
        "no_new_field_jet_solver_event_trajectory_or_family_sampling",
        "no_uniform_event_admission_formation_autonomous_identity_or_physical_uncertainty",
    )


def _same_typed(actual, expected):
    """Reject scalar coercions before comparing an admitted primitive tree."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _same_typed(actual[key], value) for key, value in expected.items()
        )
    if type(expected) is tuple:
        return len(actual) == len(expected) and all(
            _same_typed(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _receipt(receipt, expected_path):
    if (
        type(receipt) is not dict
        or set(receipt) != {"path", "bytes", "sha256"}
        or type(receipt["path"]) is not str
        or receipt["path"] != expected_path
        or type(receipt["bytes"]) is not int
        or not 0 < receipt["bytes"] <= MAX_RECORD_BYTES
        or type(receipt["sha256"]) is not str
        or len(receipt["sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in receipt["sha256"])
    ):
        raise ValueError("fixed positive typed evidence receipt required")
    validate_artifact_path(receipt["path"])
    return receipt


def _evidence_bytes(evidence, key):
    if type(evidence) is not FrozenNominalEvidence:
        raise TypeError("frozen nominal byte evidence required")
    receipt = _receipt(getattr(evidence, key), _PATHS[key])
    root = Path(evidence.repository_root).resolve()
    path = (root / receipt["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("evidence path leaves its repository")
    raw = read_bytes_bounded(path, max_bytes=receipt["bytes"])
    if len(raw) != receipt["bytes"] or sha256_bytes(raw) != receipt["sha256"]:
        raise ValueError("frozen nominal evidence bytes differ")
    return path, raw


def _evidence_record(evidence, key):
    _, raw = _evidence_bytes(evidence, key)
    record = decode_exact_tree(json_loads(raw))
    if type(record) is not dict:
        raise ValueError("nominal evidence requires a record object")
    return record


def _load_observation_verifier(evidence):
    """Execute only the exact previously frozen arithmetic owner, never main."""
    path, raw = _evidence_bytes(evidence, "observation_verifier")
    if sha256_bytes(raw) != VERIFIER_SHA256:
        raise ValueError("the immutable nominal observation verifier differs")
    module = ModuleType("tnfr_frozen_preparation_observation_verifier")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def _final_state_link(state, receipt):
    rows = state.get("evidence")
    if type(rows) is not tuple:
        raise ValueError("final state lacks its frozen evidence inventory")
    label = Path(receipt["path"]).name
    matches = tuple(
        row for row in rows if type(row) is dict and row.get("path") == label
    )
    expected = dict(receipt, path=label)
    if len(matches) != 1 or not _same_typed(matches[0], expected):
        raise ValueError("final state does not bind the consumed nominal record")


def _verify_nominal_evidence(source, source_receipts, evidence):
    """Rebuild both saved observations before deriving any family coordinate."""
    connected = _evidence_record(evidence, "connected_result")
    connected_state = _evidence_record(evidence, "connected_final_state")
    cut = _evidence_record(evidence, "cut_result")
    cut_state = _evidence_record(evidence, "cut_final_state")
    _final_state_link(connected_state, evidence.connected_result)
    _final_state_link(cut_state, evidence.cut_result)
    _final_state_link(cut_state, evidence.observation_verifier)
    if cut_state.get("admitted_complete_cut_certificate") is not True:
        raise ValueError("cut evidence lacks its admitted complete producer provenance")
    prepared = cut["report"]["preparation"]
    if not _same_typed(prepared["source"], asdict(source)):
        raise ValueError("saved cut source differs from fresh complete primitives")
    if not _same_typed(prepared["source_receipts"], source_receipts) or not _same_typed(
        connected["report"]["source_receipts"], source_receipts
    ):
        raise ValueError("nominal records differ from fresh source evidence")
    verifier = _load_observation_verifier(evidence)
    connected_angle = verifier.verify_connected(
        connected, asdict(source), source_receipts
    )
    cut_angle = verifier.verify_cut(cut["report"])
    minima = tuple(
        min(verifier.interval(row["radius_squared"]).lo for row in report["cells"])
        for report in (connected["report"], cut["report"])
    )
    if min(minima) <= 0:
        raise ValueError("both nominal whole-window radii must be strictly positive")
    return NominalPreparationBounds(
        connected_angle,
        cut_angle,
        *minima,
        evidence_receipts=tuple(deepcopy(getattr(evidence, key)) for key in _PATHS),
    )


def _coefficient_bounds(source, *, cut, pi):
    indices = source.host_indices if cut else tuple(range(len(source.nodes)))
    edges = source.cut_edges if cut else source.edges
    weights = source.cut_conductances if cut else source.conductances
    degrees = source.cut_degrees if cut else source.connected_degrees
    sums = [Q(0)] * len(source.nodes)
    for (left, right), weight in zip(edges, weights, strict=True):
        sums[left] += weight
        sums[right] += weight
    a = 2 * max(source.capacity[index] for index in indices)
    b = 2 * max(
        source.capacity[index] * sums[index] / degrees[index] for index in indices
    )
    A, B = (I(a) / pi).hi, (I(b) / pi).hi
    if HORIZON * max(A, B) > COMPARISON_STEPS:
        raise ValueError("fixed 64-step comparison work domain unavailable")
    return PreparationCoefficients(
        indices,
        tuple(degrees[index] for index in indices),
        tuple(sums[index] for index in indices),
        a,
        b,
        A,
        B,
    )


def _gate(margin):
    if type(margin) is not Q:
        raise TypeError("an exact sufficient margin is required")
    return PreparationGate(margin, "pass" if margin > 0 else "insufficient_bound")


def _bound_preparation_response(source, nominal):
    """Conditional private arithmetic; only the public boundary binds evidence."""
    if (
        type(source) is not CutSeparatorSource
        or type(nominal) is not NominalPreparationBounds
    ):
        raise TypeError("fresh source and reconstructed nominal observations required")
    for value in (nominal.connected_radius_squared, nominal.cut_radius_squared):
        if type(value) is not Q or value <= 0:
            raise ValueError("strictly positive exact nominal squared radii required")
    if (
        type(nominal.connected_clockwise) is not I
        or type(nominal.cut_clockwise) is not I
    ):
        raise TypeError("reconstructed nominal angular intervals required")
    if (
        type(nominal.connected_cells_checked) is not int
        or type(nominal.cut_cells_checked) is not int
        or nominal.connected_cells_checked != 256
        or nominal.cut_cells_checked != 256
    ):
        raise ValueError("both complete nominal windows are required")
    pi = pi_interval()
    connected = _coefficient_bounds(source, cut=False, pi=pi)
    cut = _coefficient_bounds(source, cut=True, pi=pi)
    centers = source.epi + source.phase
    floats = []
    for index, center in enumerate(centers):
        if type(center) is not Q:
            raise TypeError("captured coordinate centers must be exact fractions")
        represented, exact = finite_represented_real(center, f"family center {index}")
        if exact != center:
            raise ValueError(
                "family center does not round-trip to its represented value"
            )
        floats.append(represented)
    # Every center is associated and exactly representable before any neighbor.
    hulls = tuple(adjacent_coordinate_hull(value) for value in floats)
    size = len(source.nodes)
    form, phase = hulls[:size], hulls[size:]

    def propagate(coefficients):
        indices = coefficients.dynamic_indices
        return bound_two_block_errors(
            coefficients.A_upper,
            coefficients.B_upper,
            max(form[index].radius for index in indices),
            max(phase[index].radius for index in indices),
            HORIZON,
            steps=COMPARISON_STEPS,
        )

    connected_flow, cut_flow = propagate(connected), propagate(cut)
    child_form = form[source.child_index].radius
    child_phase = phase[source.child_index].radius
    connected_error = connected_marked_error_squared(
        *connected_flow.final, source.reference_conductance
    )
    cut_error = cut_marked_error_squared(
        *cut_flow.final, child_form, child_phase, source.reference_conductance
    )
    connected_gate = _gate(nominal.connected_radius_squared - connected_error)
    cut_gate = _gate(nominal.cut_radius_squared - cut_error)
    gap = nominal.connected_clockwise - nominal.cut_clockwise
    gap_gate = _gate(gap.lo - pi.hi)
    passed = all(gate.status == "pass" for gate in (connected_gate, cut_gate, gap_gate))
    return ShaPreparationResponse(
        source,
        nominal,
        hulls,
        3 ** len(hulls),
        pi,
        connected,
        cut,
        connected_flow,
        cut_flow,
        child_form,
        child_phase,
        connected_error,
        cut_error,
        connected_gate,
        cut_gate,
        gap,
        gap_gate,
        "family_ordering_certified" if passed else "insufficient_bound",
    )


def audit_sha_preparation_response(admitted, *, evidence):
    """Re-admit once, associate both fixed proofs, then evaluate one method.

    The caller performs the initial file-backed load. This fresh re-admission
    is the second historical load, not a new trajectory. Invalid or unavailable
    nominal evidence raises before any family widths. A failed sufficient
    family inequality is instead an ordinary ``insufficient_bound`` result.
    """
    if type(evidence) is not FrozenNominalEvidence:
        raise TypeError("frozen nominal byte evidence required")
    fresh = validate_live_sha_successor(admitted)
    if (
        Path(evidence.repository_root).resolve()
        != Path(fresh.repository_root).resolve()
    ):
        raise ValueError("nominal evidence and admitted source repositories differ")
    graph = fresh.graph
    before = _signature(graph)
    if len(graph) != 21 or graph.number_of_edges() != 53:
        raise ValueError(
            "preparation audit requires the retained 21-node 53-edge source"
        )
    _, clock = finite_represented_real(graph.graph.get("_t"), "structural clock")
    if clock != TIMES[-1]:
        raise ValueError("preparation audit requires the original structural clock")
    source = _capture_cut_source(
        graph, parent=2, child="2_sub_0", reference_model=MODEL
    )
    receipts = deepcopy(fresh.receipts)
    nominal = _verify_nominal_evidence(source, receipts, evidence)
    result = _bound_preparation_response(source, nominal)
    if _signature(graph) != before:
        raise ValueError("read-only preparation audit changed its source")
    return replace(result, source_receipts=receipts)
