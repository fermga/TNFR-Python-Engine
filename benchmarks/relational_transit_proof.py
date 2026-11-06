"""Freeze and run a validated transit audit or a single form intervention.

This is post-evaluation mathematical verification, not a reserved prediction
and not a replacement for the immutable finite-executor response. The separate
zero-form and reversed-form modes are prospective comparisons against it.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import platform
import zipfile

import networkx as nx

from benchmarks.relational_capture_audit import EXPECTED_SHA256, RECORDS
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.physics import relational_transit as owner
from tnfr.sdk.relational_reports import relational_report_to_dict

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PROOF_SHA256 = {
    "continuous-transit.audit.json": "ac3104c42cf9054968eac770fee5c1b1e174547a408875edac0b635fe04ca781",
    "continuous-transit.audit.protocol.json": "d8913d6af0d40fd1c903eaf24be29fb8ebeb6d3be28d6415d4e8dafb9974db99",
    "continuous-transit.audit.source.zip": "381212374968f32fe59751c752b4dc812844a3309e00f771704a72b31a154c93",
}


def _project(value):
    if isinstance(value, Q):
        return {"numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(type(value).__name__)


def _exact(record):
    if set(record) == {"numerator", "denominator"}:
        return Q(record["numerator"], record["denominator"])
    return record


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_hook=_exact)


def _write(path, record):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            record, stream, indent=2, sort_keys=True, allow_nan=False, default=_project
        )
        stream.write("\n")


def _retained():
    data = {}
    for name, digest in EXPECTED_SHA256.items():
        raw = (RECORDS / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"immutable response changed: {name}")
        data[name] = json.loads(raw, object_hook=_exact)
    if data["result.prediction.json"] != data["result.json"]["prediction"]:
        raise ValueError("retained response does not match its prediction")
    return data["result.prediction.json"], data["result.json"]


def _reference_proof():
    for name, digest in REFERENCE_PROOF_SHA256.items():
        if hashlib.sha256((RECORDS / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"immutable continuous reference changed: {name}")
    report = _read(RECORDS / "continuous-transit.audit.json")
    if report["protocol"] != _read(RECORDS / "continuous-transit.audit.protocol.json"):
        raise ValueError("continuous reference protocol mismatch")
    certificate = report["certificate"]["report"]
    if not report["continuous_capture_admitted"] or certificate["target_sector"] != 1:
        raise ValueError("positive continuous reference was not admitted")
    return report


def _static_zero_form(prediction):
    """Initial derivatives, not an evolved or fitted response."""
    from tnfr.mathematics._rational_interval import I

    box = tuple(
        I(value)
        for value in (
            0,
            0,
            Q(prediction["initial_phase"][0]),
            Q(prediction["initial_phase"][4]),
        )
    )
    model = RelationalExchangeModel(**prediction["model"])
    e, w, beta = map(Q, (model.epi_weight, model.phase_weight, model.storage_scale))
    jets = owner._flow_jets(box, 2, e, w, beta)
    rate = tuple(row[1] for row in jets)
    acceleration = tuple(2 * row[2] for row in jets)
    third_loss = -2 * e * (rate[0] ** 2 * Q(4, 3) + 2 * rate[1] ** 2)

    def bounds(value):
        return value.lo, value.hi

    return {
        "coordinates": "q,r,a,b",
        "initial_rate_bounds": tuple(map(bounds, rate)),
        "initial_acceleration_bounds": tuple(map(bounds, acceleration)),
        "initial_storage_bounds": bounds(owner._storage(box, beta)),
        "storage_third_derivative_bounds": bounds(third_loss),
        "resultant_real_lower_bounds": owner._regular_bounds(box),
        "zero_form_is_not_equilibrium": rate[0].lo > 0 and rate[1].hi < 0,
        "initial_phase_rate_zero": rate[2] == rate[3] == I(0),
        "phase_acceleration_directions": acceleration[2].lo > 0
        and acceleration[3].hi < 0,
        "scope": "exact_initial_derivatives_only_no_trajectory_or_selected_terminal_basin",
    }


def _graph(protocol):
    """Materialize exactly the declared scalar state and fixed unit support."""
    graph = nx.Graph()
    graph.add_nodes_from(protocol["nodes"])
    graph.add_edges_from((i, j, {"weight": 1.0}) for i, j in protocol["edges"])
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True)
    for i, node in enumerate(protocol["nodes"]):
        graph.nodes[node].update(
            EPI=protocol["initial_form"][i],
            theta=protocol["initial_phase"][i],
            nu_f=protocol["capacity"][i],
        )
    return graph


def _static_reversed_form(prediction):
    """Exact equal-storage pair and initial directional response; no evolution."""
    from tnfr.mathematics._rational_interval import I

    control = dict(
        prediction, initial_form=[-value for value in prediction["initial_form"]]
    )
    model = RelationalExchangeModel(**prediction["model"])
    e, w, beta = map(Q, (model.epi_weight, model.phase_weight, model.storage_scale))
    native = tuple(
        evaluate_relational_exchange(_graph(spec), model=model)
        for spec in (prediction, control)
    )
    rows = []
    phase_rates = []

    def bounds(value):
        return value.lo, value.hi

    for spec in (prediction, control):
        A, B = Q(spec["initial_form"][0]), Q(spec["initial_form"][4])
        q, r = 3 * A - B, 2 * B - A
        a, b = Q(spec["initial_phase"][0]), Q(spec["initial_phase"][4])
        box = tuple(I(value) for value in (q, r, a, b))
        rate = owner._flow(box, e, w, beta)
        source = owner._flow((I(0), I(0), box[2], box[3]), e, w, beta)
        # Invert q=3A-B,r=2B-A for the phase-generated form rates w*g.
        w_g0, w_g4 = (2 * source[0] + source[1]) / 5, (source[0] + 3 * source[1]) / 5
        phase_rates.append(rate[2:])
        rows.append(
            {
                "coordinates": (q, r, a, b),
                "form_storage": Q(4, 5) * (q + r / 2) ** 2 + r**2,
                "total_storage_bounds": bounds(owner._storage(box, beta)),
                "storage_rate": -e * (Q(4, 3) * q * q + 2 * r * r),
                "storage_acceleration_bounds": bounds(
                    -e * (Q(8, 3) * q * rate[0] + 4 * r * rate[1])
                ),
                "phase_rate_bounds": tuple(map(bounds, rate[2:])),
                "exchange_flux_bounds": bounds(4 * (q * w_g0 + r * w_g4)),
            }
        )
    ref, rev = rows
    difference = I(*rev["storage_acceleration_bounds"]) - I(
        *ref["storage_acceleration_bounds"]
    )
    gates = {
        "exact_form_negation": all(
            Q(x) == -Q(y)
            for x, y in zip(control["initial_form"], prediction["initial_form"])
        ),
        "same_phase_geometry": native[0].phase == native[1].phase,
        "form_storage_equal": ref["form_storage"] == rev["form_storage"],
        "total_storage_bounds_equal": ref["total_storage_bounds"]
        == rev["total_storage_bounds"],
        "storage_rate_equal": ref["storage_rate"] == rev["storage_rate"],
        "phase_rate_negated": all(
            x == -y for x, y in zip(phase_rates[0], phase_rates[1])
        ),
        "native_form_storage_equal": native[0].form_storage == native[1].form_storage,
        "native_loss_equal": native[0].continuous_loss == native[1].continuous_loss,
        "native_phase_rate_negated": all(
            x == -y for x, y in zip(native[0].phase_rate, native[1].phase_rate)
        ),
    }
    if not all(gates.values()) or difference.lo <= 0:
        raise ValueError(
            "reversed-form initial equal-storage admission did not resolve"
        )
    return {
        "reference": ref,
        "control": rev,
        "gates": gates,
        "acceleration_difference_bounds": bounds(difference),
        "scope": "exact_initial_equalities_and_certified_local_derivatives_not_a_selected_limit",
    }


def prepare_protocol(*, zero_form=False, reverse_form=False):
    if type(zero_form) is not bool or type(reverse_form) is not bool:
        raise TypeError("intervention switches require booleans")
    if zero_form and reverse_form:
        raise ValueError(
            "zero-form and reversed-form interventions are mutually exclusive"
        )
    if (
        Path(owner.__file__).resolve()
        != ROOT / "src/tnfr/physics/relational_transit.py"
    ):
        raise ValueError("set PYTHONPATH=src to use workspace source")
    prediction, result = _retained()
    paths = sorted((ROOT / "src/tnfr").rglob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "benchmarks/relational_capture_audit.py",
    ]
    protocol = {
        "analysis": "original_IVP_validated_transit_v1",
        "status": "post_evaluation_mathematical_verification_not_reserved_prediction",
        "original_records_sha256": dict(EXPECTED_SHA256),
        "original_finite_prediction_passed": result["finite_prediction_passed"],
        "nodes": prediction["nodes"],
        "edges": prediction["edges"],
        "cycles": prediction["cycles"],
        "initial_form": prediction["initial_form"],
        "initial_phase": prediction["initial_phase"],
        "capacity": prediction["capacity"],
        "model": prediction["model"],
        "numerical_policy": {
            "horizon": Q(32),
            "time_step": Q(1, 8),
            "order": 12,
            "interval_bits": 128,
            "picard_inflation": Q(5, 4),
            "picard_absolute_padding": Q(1, 1 << 90),
            "picard_attempts": 16,
            "stopping": "first_unresolved_proof_step_or_fixed_horizon",
            "retry": "none",
        },
        "acceptance": "initial_winding_zero_and_whole_endpoint_Rplus_and_storage_upper_below_7beta_after_complete_validated_transit",
        "runtime": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "arithmetic": "exact_Fraction_with_outward_dyadic128_and_rational_series",
            "randomness": "none",
        },
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }
    if zero_form or reverse_form:
        reference = _reference_proof()
        for key in (
            "initial_form",
            "initial_phase",
            "capacity",
            "model",
            "edges",
            "cycles",
            "nodes",
        ):
            if reference["protocol"][key] != prediction[key]:
                raise ValueError("continuous reference is not the same preparation")
        protocol.update(
            analysis=(
                "zero_initial_form_validated_ablation_v1"
                if zero_form
                else "reversed_initial_form_validated_control_v1"
            ),
            status="prospective_conditional_model_discriminator_not_physical_validation",
            intervention="zero_initial_form" if zero_form else "reversed_initial_form",
            initial_form=(
                [0.0] * len(prediction["nodes"])
                if zero_form
                else [-value for value in prediction["initial_form"]]
            ),
            reference_proof_sha256=dict(REFERENCE_PROOF_SHA256),
            reference_target_sector=1,
            requested_sector=None,
            initial_derivation=(
                _static_zero_form(prediction)
                if zero_form
                else _static_reversed_form(prediction)
            ),
            acceptance="complete_validated_horizon_and_whole_endpoint_in_any_protected_basin_with_storage_below_7beta",
            competing_outcomes={
                "1": (
                    "same_positive_limit_without_supplied_initial_form_contrast"
                    if zero_form
                    else "same_positive_limit_does_not_establish_storage_sufficiency"
                ),
                "0": "consensus_different_limit_from_positive_reference",
                "-1": "negative_twist_different_limit_from_positive_reference",
                "unavailable": "no_asymptotic_decision_from_unresolved_enclosure_or_unclassified_endpoint",
            },
            predicted_local_response=(
                "form_contrast_begins_immediately_phase_rates_start_zero_but_accelerations_do_not"
                if zero_form
                else "same_initial_storage_and_dissipation_opposite_phase_velocity_different_storage_acceleration"
            ),
            terminal_prediction="unselected_competing_basin_outcomes; local_signs_do_not_predict_the_limit",
        )
    return protocol


def evaluate_protocol(protocol):
    zero_form = protocol.get("intervention") == "zero_initial_form"
    reverse_form = protocol.get("intervention") == "reversed_initial_form"
    if zero_form:
        current = prepare_protocol(zero_form=True)
    elif reverse_form:
        current = prepare_protocol(reverse_form=True)
    else:
        current = prepare_protocol()
    # JSON round trip normalizes lists/tuples without losing rational inputs.
    current = json.loads(json.dumps(current, default=_project), object_hook=_exact)
    if current != protocol:
        raise ValueError("frozen proof protocol or source/runtime changed")
    graph = _graph(protocol)
    policy = protocol["numerical_policy"]
    options = {"requested_sector": None} if zero_form or reverse_form else {}
    certificate = owner.certify_relational_transit_capture(
        graph,
        model=RelationalExchangeModel(**protocol["model"]),
        cycles=protocol["cycles"],
        horizon=policy["horizon"],
        time_step=policy["time_step"],
        order=policy["order"],
        **options,
    )
    report = {
        "protocol": protocol,
        "certificate": relational_report_to_dict(certificate),
        "continuous_capture_admitted": certificate.admitted
        and certificate.initial_winding_zero,
        "original_finite_prediction_passed": protocol[
            "original_finite_prediction_passed"
        ],
    }
    if zero_form or reverse_form:
        report.update(
            terminal_basin_admitted=certificate.admitted,
            target_sector=certificate.target_sector,
            different_limit_certified=(
                certificate.target_sector != 1 if certificate.admitted else None
            ),
            outcome=(
                str(certificate.target_sector)
                if certificate.admitted
                else "unavailable"
            ),
        )
        # This flag retains its historical positive-pattern meaning.
        report["continuous_capture_admitted"] = (
            certificate.admitted
            and certificate.initial_winding_zero
            and certificate.target_sector == 1
        )
        if zero_form:
            report["phase_geometry_sufficient_for_positive_capture"] = (
                certificate.target_sector == 1 if certificate.admitted else None
            )
        else:
            report["same_positive_limit"] = (
                certificate.target_sector == 1 if certificate.admitted else None
            )
            report["initial_storage_selector_refuted"] = report[
                "different_limit_certified"
            ]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    intervention = parser.add_mutually_exclusive_group()
    intervention.add_argument(
        "--zero-form",
        action="store_true",
        help="prepare/evaluate the single zero-form ablation",
    )
    intervention.add_argument(
        "--reverse-form",
        action="store_true",
        help="prepare/evaluate the equal-storage sign-reversal control",
    )
    args = parser.parse_args(argv)
    output = args.output
    protocol_path = output.with_suffix(".protocol.json")
    source_path = output.with_suffix(".source.zip")
    if output.exists():
        raise FileExistsError("refusing to replace a retained proof audit")
    if args.prepare:
        if protocol_path.exists() or source_path.exists():
            raise FileExistsError("refusing to replace a frozen proof protocol/source")
        if args.zero_form:
            protocol = prepare_protocol(zero_form=True)
        elif args.reverse_form:
            protocol = prepare_protocol(reverse_form=True)
        else:
            protocol = prepare_protocol()
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            source_path, "x", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path, digest in protocol["source_sha256"].items():
                data = (ROOT / path).read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError("source changed during archive creation")
                archive.writestr(path, data)
        _write(protocol_path, protocol)
        print(
            "Frozen validated proof policy and exact source; no trajectory evaluated.",
            flush=True,
        )
        return 0
    protocol = _read(protocol_path)
    if args.zero_form != (protocol.get("intervention") == "zero_initial_form"):
        raise ValueError("--zero-form must match the frozen protocol")
    if args.reverse_form != (protocol.get("intervention") == "reversed_initial_form"):
        raise ValueError("--reverse-form must match the frozen protocol")
    try:
        report = evaluate_protocol(protocol)
    except Exception as exc:
        _write(output, {"protocol": protocol, "error": f"{type(exc).__name__}: {exc}"})
        raise
    _write(output, report)
    certificate = report["certificate"]["report"]
    print(
        json.dumps(
            {
                "continuous_capture_admitted": report["continuous_capture_admitted"],
                "validated_horizon": certificate["validated_horizon"],
                "steps": len(certificate["steps"]),
                "unavailable_reasons": certificate["unavailable_reasons"],
                "original_finite_prediction_passed": report[
                    "original_finite_prediction_passed"
                ],
                "terminal_basin_admitted": report.get("terminal_basin_admitted"),
                "outcome": report.get("outcome"),
            }
        ),
        flush=True,
    )
    resolved = report.get(
        "terminal_basin_admitted", report["continuous_capture_admitted"]
    )
    return 0 if resolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
