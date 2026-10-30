"""Read-only fixed-source constitutive discrimination from frozen sine evidence.

The known validated producer supplies the reference-flow enclosure premise.
This owner checks its immutable association and consumed primitive data, then
rebuilds the static energy obstruction. It neither repeats a trajectory nor
imports arbitrary caller-supplied numerical proof records.
"""

from __future__ import annotations

import argparse
import ast
import io
import zipfile
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval
from ..physics.relational_sine_comparison import (
    SineExchangeComparison,
    _comparison_from_state,
    _sine_phase_storage,
    _sine_state_from_rows,
)
from ..physics.relational_sine_forecast import _admit_support
from ..physics.relational_sine_partition import _private_leaf_support
from ..physics.relational_sine_regional import _branches
from ..utils.io import json_dumps, json_loads, safe_write
from .artifact_io import _require
from .artifact_io import exact_record as _exact
from .artifact_io import read_bytes_bounded
from .artifact_io import sha256_bytes as _sha
from .artifact_io import verify_archive_members as _verify_archive

__all__ = ("SineConstitutiveRobustness", "assess_sine_constitutive_robustness")

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIRECTORY = ROOT / "docs/assets/sine_metric_connection"
_BUNDLE_SHA256 = "16c0b77f8ee439fa6f688e4cffcc84e3a582b1d62abf64dbec77f77eeb473193"
_ENTRIES = {
    "response-v1.json": (
        112761509,
        "5c1e8fe8960e40b1d4d2747d1f859c51964aa35013574a49ecaa3bb37372f92b",
    ),
    "response-v1.protocol.json": (
        277649,
        "6858cdd1e29851c7242126afd8dcd2142323f14a2bf611e11f585fc58de05271",
    ),
    "response-v1.source.zip": (
        3239558,
        "e0456e5f0479f757e9fda89ff72835ea377cb7421e6ad38d2e31d1844dca8ede",
    ),
}
_FLOW_FUNCTIONS = (
    ("src/tnfr/physics/relational_sine_forecast.py", "_sine_flow"),
    ("src/tnfr/physics/relational_sine_metric_forecast.py", "_saddle_flow"),
)
_METHOD_FILES = (
    "src/tnfr/research/sine_constitutive_robustness.py",
    "src/tnfr/research/artifact_io.py",
    "src/tnfr/research/relational_acquisition.py",
    "src/tnfr/mathematics/_rational_interval.py",
    "src/tnfr/physics/_sine_admission.py",
    "src/tnfr/physics/relational_sine_forecast.py",
    "src/tnfr/physics/relational_sine_comparison.py",
    "src/tnfr/physics/relational_sine_regional.py",
    "src/tnfr/physics/relational_sine_partition.py",
)


def _method_path(name):
    return PACKAGE_ROOT / name.removeprefix("src/tnfr/")


def _interval(raw):
    _require(type(raw) is dict and set(raw) == {"lo", "hi"}, "invalid interval record")
    lower, upper = _exact(raw["lo"]), _exact(raw["hi"])
    _require(lower <= upper, "interval endpoints are reversed")
    return I(lower, upper)


def _box(raw):
    _require(
        type(raw) is list and len(raw) == 20, "twenty full-state coordinates required"
    )
    return tuple(_interval(value) for value in raw)


def _row(raw, size):
    _require(type(raw) is list and len(raw) == size, "invalid exact row dimension")
    return tuple(_exact(value) for value in raw)


def _function_ast(data, name):
    module = ast.parse(data.decode("utf-8-sig"))
    definitions = [
        item
        for item in module.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    ]
    _require(len(definitions) == 1, "missing or ambiguous declared flow definition")
    return ast.dump(definitions[0], include_attributes=False)


def _load_frozen_evidence(directory):
    """Verify exact known bytes without importing the archived package."""
    path = Path(directory) / "response-v1.evidence.zip"
    bundle = read_bytes_bounded(path, max_bytes=16 * 1024**2)
    _require(
        _sha(bundle) == _BUNDLE_SHA256, "unsupported or changed frozen evidence bundle"
    )
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        _require(
            len(archive.namelist()) == 3 and set(archive.namelist()) == set(_ENTRIES),
            "evidence inventory differs",
        )
        content = {}
        for name, (size, digest) in _ENTRIES.items():
            info = archive.getinfo(name)
            _require(
                info.file_size == size and not info.flag_bits & 1,
                "evidence entry size or encoding differs",
            )
            data = archive.read(name)
            _require(_sha(data) == digest, "evidence entry digest differs")
            content[name] = data
    record = json_loads(content["response-v1.json"])
    protocol = json_loads(content["response-v1.protocol.json"])
    _require(
        record["protocol_sha256"] == _ENTRIES["response-v1.protocol.json"][1],
        "protocol association differs",
    )
    _require(
        record["source_archive_sha256"] == _ENTRIES["response-v1.source.zip"][1],
        "source archive association differs",
    )
    _verify_archive(
        io.BytesIO(content["response-v1.source.zip"]), protocol["source_sha256"]
    )
    with zipfile.ZipFile(io.BytesIO(content["response-v1.source.zip"])) as archive:
        declaration = json_loads(archive.read(protocol["declaration_path"]))
        _require(declaration == protocol["declaration"], "archived declaration differs")
        for filename, function in _FLOW_FUNCTIONS:
            _require(
                _function_ast(archive.read(filename), function)
                == _function_ast(_method_path(filename).read_bytes(), function),
                "current and archived declared flow syntax differ",
            )
    return record, protocol


def _rebuild_source(report, declaration):
    """Admit primitive state, full support and actual normalized clock law."""
    expected = {
        "schema": "tnfr.sine-metric-connection-declaration.v1",
        "law": "normalized_sine_reciprocal_e0_w1_beta1",
        "support": "fixed_simple_connected_unit_undirected",
        "forcing": "none",
        "events": "none",
        "clock": "original_structural_t; tau=t/pi",
    }
    _require(
        all(declaration.get(key) == value for key, value in expected.items()),
        "unsupported complete reference declaration",
    )
    source = report["source"]
    nodes = source["nodes"]
    _require(
        type(nodes) is list
        and all(type(v) is int for v in nodes)
        and nodes == declaration["nodes"] == list(range(10)),
        "unsupported source node order",
    )
    _require(
        source["law"] == "normalized_sine_reciprocal_exchange", "source law differs"
    )
    model = source["reference_model"]
    _require(
        set(model) == {"epi_weight", "phase_weight", "storage_scale", "phase_domain"},
        "model fields differ",
    )
    coefficients = tuple(
        exact_or_represented_real(model[key], key)
        for key in ("epi_weight", "phase_weight", "storage_scale")
    )
    _require(
        coefficients == (0, 1, 1) and model["phase_domain"] == "regular",
        "conservative unit law required",
    )
    reference_model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    edges = source["edges"]
    _require(
        edges == declaration["edges"] and type(edges) is list, "source support differs"
    )
    neighbors = [[] for _ in nodes]
    for edge in edges:
        _require(
            type(edge) is list
            and len(edge) == 2
            and all(type(i) is int and 0 <= i < 10 for i in edge),
            "invalid source edge",
        )
        i, j = edge
        neighbors[i].append(j)
        neighbors[j].append(i)
    form, phase, capacity = (
        _row(source[key], 10) for key in ("epi", "phase", "capacity")
    )
    for key, actual in (("form", form), ("phase", phase), ("capacity", capacity)):
        _require(
            type(declaration[key]) is list
            and all(type(value) is str for value in declaration[key]),
            "literal exact preparation required",
        )
        _require(
            tuple(Q(value) for value in declaration[key]) == actual,
            "source primitive differs from frozen preparation",
        )
    _require(capacity == (Q(1),) * 10, "unit held capacities required")
    neighbors, _ = _admit_support(neighbors, capacity[:-1], reference_model)
    state = _comparison_from_state(
        _sine_state_from_rows(
            tuple(nodes), tuple(map(tuple, edges)), form, phase, capacity, neighbors
        ),
        reference_model,
    )
    cycle = declaration["cycle_indices"]
    _require(
        type(cycle) is list and all(type(value) is int for value in cycle),
        "invalid cycle indices",
    )
    state, edge_indices, indices, pairs, _ = _private_leaf_support(state, cycle)
    observed_cycle = report["cycle_indices"]
    _require(
        type(observed_cycle) is list
        and all(type(value) is int for value in observed_cycle)
        and tuple(observed_cycle) == indices,
        "observed cycle differs",
    )
    return state, edge_indices, indices, pairs


def _check_chain(forecast, source, direction, count):
    """Check consumed association/grid without replaying its Taylor proof."""
    _require(
        type(forecast["direction"]) is int and forecast["direction"] == direction,
        "forecast direction differs",
    )
    _require(
        forecast["source"]["nodes"] == list(source.nodes), "forecast node order differs"
    )
    _require(
        forecast["source"]["edges"] == [list(edge) for edge in source.edges]
        and forecast["source"]["law"] == source.law
        and forecast["clock"]
        == "original structural t; direction times elapsed duration",
        "forecast support, law or clock differs",
    )
    model = forecast["source"]["reference_model"]
    _require(
        set(model) == {"epi_weight", "phase_weight", "storage_scale", "phase_domain"}
        and model["phase_domain"] == "regular"
        and tuple(
            exact_or_represented_real(model[key], key)
            for key in ("epi_weight", "phase_weight", "storage_scale")
        )
        == (0, 1, 1),
        "forecast complete coefficients differ",
    )
    for name, values in (
        ("epi", source.epi),
        ("phase", source.phase),
        ("capacity", source.capacity),
    ):
        _require(
            _row(forecast["source"][name], 10) == values,
            "forecast source association differs",
        )
    center = _row(forecast["initial_center"], 20)
    _require(center == source.epi + source.phase, "forecast initial center differs")
    radius = _exact(forecast["initial_metric_radius"])
    steps = forecast["steps"]
    _require(
        type(steps) is list and len(steps) >= count,
        "required validated prefix unavailable",
    )
    for index, step in enumerate(steps[:count]):
        _require(
            _exact(step["time"]) == index and _exact(step["duration"]) == 1,
            "retained prefix grid differs",
        )
        _require(
            _row(step["initial_center"], 20) == center
            and _exact(step["initial_radius"]) == radius
            and radius >= 0,
            "retained metric chain differs",
        )
        center, radius = _row(step["endpoint_center"], 20), _exact(
            step["endpoint_radius"]
        )
        _require(radius >= 0, "negative endpoint radius")
    return steps


@dataclass(frozen=True)
class SineConstitutiveRobustness:
    """Static exclusion at one fixed reference-flow source, not a new run."""

    source: SineExchangeComparison
    eta: Q
    cycle: tuple[int, ...]
    initial_box: tuple[I, ...]
    source_image_box: tuple[I, ...]
    declared_initial_coordinate_radius: Q
    source_time: Q
    reference_target_window: tuple[Q, Q]
    source_turn_offsets: tuple[int, ...] | None
    source_principal_gap_bounds: tuple[I, ...] | None
    source_winding: int | None
    reference_target_acute_margin_lower_bound: Q
    initial_sine_storage_bounds: I
    added_phase_storage_bounds: I
    perturbed_source_storage_bounds: I
    sector_barrier: Q
    energy_margin_lower_bound: Q
    same_source_acute_formation_excluded: bool
    comparison_gate_outcome: str
    evidence_sha256: tuple[tuple[str, str], ...]
    method_source_sha256: tuple[tuple[str, str], ...]
    retained_protocol_passed: bool
    status: str
    reasons: tuple[str, ...]
    independent_numerical_source_ball_certified: bool = False
    law: str = "cubic_sine_reciprocal_storage_eta_1_over_100"
    clock: str = "original structural t; tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    conditional_premises: tuple[str, ...] = (
        "the_immutable_archived_validated_producer_supplies_the_reference_flow_enclosure_proof",
        "digests_associate_evidence_and_source_bytes_but_are_not_a_mathematical_proof",
        "the_exact_fixed_source_is_R_Phi_sine_237_z_not_a_source_retuned_under_the_new_law",
        "the_source_field_is_the_intermediate_preparation_its_flow_image_is_separately_retained",
        "all_twenty_coordinates_full_support_unit_capacities_no_forcing_or_events",
        "sine_storage_at_the_mapped_source_equals_initial_sine_storage_by_conservation_and_form_reversal",
        "the_added_potential_is_summed_over_all_receiver_and_contact_edges",
        "U_eta_is_nonnegative_even_periodic_and_its_derivative_drives_the_reciprocal_form_row",
        "the_C5_Omega_unit_winding_boundary_minimum_is_7_over_2_plus_47_eta_over_24_for_eta_nonnegative",
        "fixed_source_winding_zero_and_strict_energy_sublevel_exclude_acute_unit_winding_at_any_time",
        "the_outer_source_image_box_is_not_an_arbitrarily_preparable_independent_source_ball",
        "no_claim_of_perpetual_zero_winding_absence_of_all_patterns_or_physical_law_selection",
        "no_new_trajectory_coefficient_scan_numerical_budget_retry_or_frozen_evidence_change",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        self.source.to_dict()
        return {
            "schema": "tnfr.sine-constitutive-robustness.v1",
            "report": _project(self),
        }


def _assess_record(record, protocol, *, evidence_sha256, method_source_sha256):
    """Consume only the known loader's validated-enclosure premise."""
    _require(
        record["schema"] == "tnfr.sine-metric-connection-response.v1"
        and record["evaluation_error"] is None,
        "completed known response required",
    )
    _require(
        type(record["passed"]) is bool and not record["passed"],
        "original qualified verdict differs",
    )
    _require(
        protocol["schema"] == "tnfr.sine-metric-connection-protocol.v1"
        and record["response"]["schema"] == "tnfr.sine-metric-connection.v1",
        "evidence schema differs",
    )
    report = record["response"]["report"]
    source, edges, indices, pairs = _rebuild_source(report, protocol["declaration"])
    forward = _check_chain(report["forward"], source, 1, 237)
    backward = _check_chain(report["backward"], source, -1, 234)
    initial = _box(forward[0]["initial_box"])
    delta = _exact(report["forward"]["initial_coordinate_radius"])
    _require(
        delta
        == Q(protocol["declaration"]["initial_coordinate_radius"])
        == Q(1, 2**100),
        "frozen independent preparation radius differs",
    )
    _require(
        all(
            bound.lo <= value - delta and bound.hi >= value + delta
            for bound, value in zip(initial, source.epi + source.phase)
        ),
        "initial enclosure does not cover the declared preparation cube",
    )
    endpoint = _box(forward[236]["endpoint"])
    image = tuple(-value for value in endpoint[:10]) + endpoint[10:]
    offsets, gaps, winding, _ = _branches(
        tuple(image[10 + j] - image[10 + i] for i, j in pairs)
    )
    target_margins, target_offsets = [], None
    for step in backward[230:234]:
        tube = _box(step["tube"])
        branch, _, number, margin = _branches(
            tuple(tube[10 + j] - tube[10 + i] for i, j in pairs)
        )
        _require(
            number == 1 and margin is not None and margin > 0,
            "retained reference target not strictly acute winding one",
        )
        _require(
            target_offsets is None or target_offsets == branch,
            "reference target branches differ",
        )
        target_offsets = branch
        target_margins.append(margin)
    _require(Q(4) > pi_interval().hi, "retained target shorter than one scaled unit")
    forms, phases = initial[:10], initial[10:]
    h0 = sum(
        ((forms[j] - forms[i]) ** 2 / 2 for i, j in edges), I(0)
    ) + _sine_phase_storage(phases, edges)
    costs = []
    for i, j in edges:
        cosine = cos(image[10 + j] - image[10 + i])
        costs.append((1 - cosine) ** 2 * (2 + cosine) / 3)
    added = sum(costs, I(0))
    eta = Q(1, 100)
    perturbed = h0 + eta * added
    barrier = Q(7, 2) + Q(47, 24) * eta
    margin = barrier - perturbed.hi
    excluded = winding == 0 and margin > 0
    reasons = tuple(
        reason
        for valid, reason in (
            (winding == 0, "fixed_source_winding_zero_not_certified"),
            (margin > 0, "strict_perturbed_storage_barrier_not_certified"),
        )
        if not valid
    )
    return SineConstitutiveRobustness(
        source=source,
        eta=eta,
        cycle=indices,
        initial_box=initial,
        source_image_box=image,
        declared_initial_coordinate_radius=delta,
        source_time=Q(237),
        reference_target_window=(Q(467), Q(471)),
        source_turn_offsets=offsets,
        source_principal_gap_bounds=gaps,
        source_winding=winding,
        reference_target_acute_margin_lower_bound=min(target_margins),
        initial_sine_storage_bounds=h0,
        added_phase_storage_bounds=added,
        perturbed_source_storage_bounds=perturbed,
        sector_barrier=barrier,
        energy_margin_lower_bound=margin,
        same_source_acute_formation_excluded=excluded,
        comparison_gate_outcome=(
            "same_source_acute_formation_excluded_by_conserved_storage"
            if excluded
            else "unavailable"
        ),
        evidence_sha256=evidence_sha256,
        method_source_sha256=method_source_sha256,
        retained_protocol_passed=record["passed"],
        status="certified" if excluded else "unavailable",
        reasons=reasons,
    )


def assess_sine_constitutive_robustness(evidence_directory=DEFAULT_EVIDENCE_DIRECTORY):
    """Audit the fixed eta=1/100 test once from the known frozen evidence bundle.

    Missing, altered or unsupported evidence rejects. The recorded producer's
    enclosure validity remains explicit: source hashes are checked, consumed
    observations are rebuilt, and no general Taylor-proof replay is claimed.
    """
    record, protocol = _load_frozen_evidence(evidence_directory)
    return _assess_record(
        record,
        protocol,
        evidence_sha256=(("response-v1.evidence.zip", _BUNDLE_SHA256),)
        + tuple((name, digest) for name, (_, digest) in _ENTRIES.items()),
        method_source_sha256=tuple(
            (name, _sha(_method_path(name).read_bytes())) for name in _METHOD_FILES
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-directory", type=Path, default=DEFAULT_EVIDENCE_DIRECTORY
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError("retain existing constitutive postprocessing evidence")
    report = assess_sine_constitutive_robustness(args.evidence_directory)
    encoded = json_dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False)
    if args.output is None:
        print(encoded)
    else:
        safe_write(args.output, lambda stream: stream.write(encoded + "\n"))
    return 0 if report.status == "certified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
