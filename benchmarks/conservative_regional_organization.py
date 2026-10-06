"""Freeze and evaluate one conservative regional organization control.

Preparation and evaluation are separate invocations. Both reuse the shared
evidence transport; evaluation retains a failed or partial response without
changing the declared source, target, horizon or numerical budget.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from dataclasses import fields
from fractions import Fraction as Q
from pathlib import Path

from benchmarks import relational_seeded_response as evidence
from tnfr._exact_time import exact_or_represented_real
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_taylor import ValidatedTaylorStep
from tnfr.physics.relational_sine_forecast import (
    SineForecast,
    _admit_support,
    bound_sine_flow,
)
from tnfr.physics.relational_sine_regional import assess_sine_regional_organization
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/assets/conservative_regional_organization/declaration.json"
DEFAULT_OUTPUT = (
    ROOT / "artifacts/research/conservative_regional_organization/response-v1.json"
)


def _files(declaration_path=None):
    files = evidence._source_files()
    for path in (Path(__file__).resolve(), declaration_path or DECLARATION):
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    return files


def _source_files(declaration_path):
    # Preserve the original instrument's default and its frozen source reader.
    return (
        _files()
        if declaration_path is None
        else _files(Path(declaration_path).resolve())
    )


def _declared_real(value):
    """Decode exact JSON rational strings, retaining shared real admission."""
    if isinstance(value, str):
        value = Q(value)
    return exact_or_represented_real(value, "declared coordinate or budget")


def _evaluation_kind(declaration):
    kind = declaration.get("evaluation_kind", "regional_organization")
    evidence._require(
        kind in ("regional_organization", "reversible_preparation"),
        "unsupported declaration evaluation_kind",
    )
    return kind


def _reversible_checkpoint(declaration, preparation):
    """Admit the exact checkpoint and its target without evaluating a flow."""
    from tnfr.physics.relational_sine_comparison import (
        _comparison_from_state,
        _sine_state_from_rows,
    )
    from tnfr.physics.relational_sine_regional import assess_sine_cycle_retention

    evidence._require(
        declaration["selection_rule"]
        == "first_certified_declared_endpoint_after_backward_retention_window",
        "unsupported reversible endpoint selection rule",
    )
    nodes = tuple(declaration["nodes"])
    size = len(nodes)

    def row(key):
        values = declaration[key]
        evidence._require(
            isinstance(values, list) and len(values) == size,
            f"{key} must retain every exact checkpoint coordinate",
        )
        return tuple(_declared_real(value) for value in values)

    form, phase = row("checkpoint_form"), row("checkpoint_phase")
    evidence._require(
        row("form") == tuple(-value for value in form) and row("phase") == phase,
        "reverse preparation must be exactly R(checkpoint)=(-form, phase)",
    )
    cycle = declaration["receiver_cycle_indices"]
    evidence._require(
        isinstance(cycle, list) and all(type(index) is int for index in cycle),
        "receiver cycle must retain ordered integer node indices",
    )
    options = {
        key: _declared_real(declaration[key])
        for key in (
            "target_error_bound",
            "source_error_bound",
            "scaled_retention_duration",
        )
    }
    evidence._require(
        all(value > 0 for value in options.values()),
        "reversible source and target radii and retention duration must be positive",
    )
    checkpoint = _comparison_from_state(
        _sine_state_from_rows(
            nodes,
            tuple(tuple(edge) for edge in declaration["edges"]),
            form,
            phase,
            (Q(1),) * size,
            preparation["neighbors"],
        ),
        preparation["model"],
    )
    target = assess_sine_cycle_retention(
        checkpoint,
        cycle=tuple(cycle),
        scaled_duration=options["scaled_retention_duration"],
        source_error_bound=options["target_error_bound"],
    )
    evidence._require(
        target.whole_window_retention_certified,
        "the declared checkpoint target does not certify finite retention",
    )
    return checkpoint, {"cycle": tuple(cycle), **options}


def _preparation(declaration):
    """Build the declared full state without observing a future response."""
    kind = _evaluation_kind(declaration)
    evidence._require(
        declaration["schema"]
        == "tnfr.conservative-regional-organization-declaration.v1"
        and declaration["law"] == "normalized_sine_reciprocal_e0_w1_beta1"
        and declaration["forcing"] == "none"
        and declaration["events"] == "none"
        and declaration["clock"] == "original_structural_t; tau=t/pi"
        and declaration["support"] == "fixed_simple_connected_unit_undirected",
        "unsupported complete-law declaration",
    )
    nodes = declaration["nodes"]
    size = len(nodes)
    evidence._require(
        all(type(i) is int for i in nodes) and nodes == list(range(size)),
        "the declared state layout requires consecutive node indices",
    )
    neighbors = [[] for _ in nodes]
    for edge in declaration["edges"]:
        evidence._require(
            len(edge) == 2 and all(type(i) is int and 0 <= i < size for i in edge),
            "invalid declared edge",
        )
        i, j = edge
        neighbors[i].append(j)
        neighbors[j].append(i)
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    form, phase, capacity = (
        tuple(I(_declared_real(value)) for value in declaration[key])
        for key in ("form", "phase", "capacity")
    )
    evidence._require(
        len(form) == len(phase) == len(capacity) == size
        and all(value == I(1) for value in capacity),
        "the preparation requires complete state and unit held capacities",
    )
    neighbors, visible = _admit_support(
        tuple(tuple(row) for row in neighbors), (Q(1),) * (size - 1), model
    )
    at, end, step = (
        _declared_real(declaration[key])
        for key in ("observation_time", "end_time", "time_step")
    )
    if kind == "reversible_preparation":
        evidence._require(
            at == 0, "reversible preparation requires observation_time zero"
        )
    evidence._require(
        type(declaration["maximum_steps"]) is int
        and 1 <= declaration["maximum_steps"] <= 256
        and 0 <= at < end
        and step > 0
        and (end - at) / step <= declaration["maximum_steps"],
        "declared horizon exceeds the admitted step budget",
    )
    evidence._require(
        type(declaration["order"]) is int and 1 <= declaration["order"] <= 16,
        "Taylor order must be an integer from one to sixteen",
    )
    if kind == "regional_organization":
        evidence._require(
            _declared_real(declaration["minimum_duration"]) > 0
            and _declared_real(declaration["acute_margin"]) >= 0,
            "invalid declared observation budgets",
        )
    preparation = dict(
        initial=form + phase + (capacity[-1],),
        neighbors=neighbors,
        visible_capacity=visible,
        model=model,
        observation_time=at,
        end_time=end,
        time_step=step,
        order=declaration["order"],
    )
    if kind == "reversible_preparation":
        _reversible_checkpoint(declaration, preparation)
    return preparation


def prepare_protocol(declaration_path=None):
    evidence._verify_runtime_source()
    path = DECLARATION if declaration_path is None else Path(declaration_path).resolve()
    declaration = json_loads(path.read_bytes())
    _preparation(declaration)
    return {
        "schema": "tnfr.conservative-regional-organization-protocol.v1",
        "declaration": declaration,
        "runtime": evidence._runtime(),
        "declaration_path": path.relative_to(ROOT).as_posix(),
        "source_sha256": evidence._manifest(_source_files(declaration_path)),
        "archive_scope": "all src/tnfr Python, evidence transport, producer, declaration and pyproject; not dependency binaries or provenance authentication",
    }


def evaluate(declaration):
    preparation = _preparation(declaration)
    if _evaluation_kind(declaration) == "reversible_preparation":
        from tnfr.physics.relational_sine_regional import (
            assess_sine_reversible_preparation,
        )

        checkpoint, options = _reversible_checkpoint(declaration, preparation)
        forecast = bound_sine_flow(**preparation)
        return assess_sine_reversible_preparation(
            forecast, checkpoint, **options
        ).to_dict()
    forecast = bound_sine_flow(**preparation)
    assessment = assess_sine_regional_organization(
        forecast,
        cycle_indices=tuple(declaration["receiver_cycle_indices"]),
        minimum_duration=_declared_real(declaration["minimum_duration"]),
        acute_margin=_declared_real(declaration["acute_margin"]),
    )
    return assessment.to_dict()


def _exact_projection(value):
    """Read this producer's exact rational projection, without numeric coercion."""
    evidence._require(
        isinstance(value, dict)
        and set(value) == {"numerator", "denominator"}
        and type(value["numerator"]) is int
        and type(value["denominator"]) is int
        and value["denominator"] > 0,
        "expected an exact rational projection",
    )
    result = Q(value["numerator"], value["denominator"])
    evidence._require(
        result.numerator == value["numerator"]
        and result.denominator == value["denominator"],
        "rational projection must be reduced",
    )
    return result


def _interval_projection(value):
    evidence._require(
        isinstance(value, dict) and set(value) == {"lo", "hi"},
        "expected a complete interval projection",
    )
    return I(_exact_projection(value["lo"]), _exact_projection(value["hi"]))


def _projected_fields(value, kind):
    evidence._require(
        isinstance(value, dict)
        and set(value) == {field.name for field in fields(kind)},
        f"expected all {kind.__name__} projection fields",
    )


def _projection_row(values, decode):
    evidence._require(isinstance(values, list), "expected a projected sequence")
    return tuple(decode(value) for value in values)


def _text(value):
    evidence._require(isinstance(value, str) and bool(value), "expected nonempty text")
    return value


def _integer(value):
    evidence._require(type(value) is int, "expected an integer, not a Boolean")
    return value


def _forecast_projection(value):
    """Decode only the retained no-prior forecast; not a general SDK checkpoint."""
    _projected_fields(value, SineForecast)
    model = value["model"]
    _projected_fields(model, RelationalExchangeModel)
    coefficients = {
        key: exact_or_represented_real(model[key], key)
        for key in ("storage_scale", "epi_weight", "phase_weight")
    }
    evidence._require(
        coefficients == {"storage_scale": 1, "epi_weight": 0, "phase_weight": 1},
        "the retained declaration requires normalized unit zero-loss coefficients",
    )
    evidence._require(
        value["prior_admission"] is None and type(value["freeze_hidden"]) is bool,
        "only a complete no-prior forecast projection is supported",
    )
    steps = []
    for step in _projection_row(value["steps"], lambda item: item):
        _projected_fields(step, ValidatedTaylorStep)
        steps.append(
            ValidatedTaylorStep(
                **{
                    key: (
                        _projection_row(raw, _interval_projection)
                        if key in ("tube", "endpoint", "local_remainder_bounds")
                        else (
                            _projection_row(raw, _exact_projection)
                            if key
                            in ("domain_lower_bounds", "propagated_initial_radii")
                            else _exact_projection(raw)
                        )
                    )
                    for key, raw in step.items()
                }
            )
        )
    return SineForecast(
        model=RelationalExchangeModel(
            **coefficients, phase_domain=_text(model["phase_domain"])
        ),
        neighbors=_projection_row(
            value["neighbors"], lambda row: _projection_row(row, _integer)
        ),
        visible_capacity=_projection_row(value["visible_capacity"], _exact_projection),
        initial_box=_projection_row(value["initial_box"], _interval_projection),
        observation_time=_exact_projection(value["observation_time"]),
        end_time=_exact_projection(value["end_time"]),
        time_step=_exact_projection(value["time_step"]),
        order=_integer(value["order"]),
        steps=tuple(steps),
        validated_end_time=_exact_projection(value["validated_end_time"]),
        endpoint=_projection_row(value["endpoint"], _interval_projection),
        failed_tube=(
            None
            if value["failed_tube"] is None
            else _projection_row(value["failed_tube"], _interval_projection)
        ),
        status=_text(value["status"]),
        reasons=_projection_row(value["reasons"], _text),
        forecast_start=(
            None
            if value["forecast_start"] is None
            else _exact_projection(value["forecast_start"])
        ),
        freeze_hidden=value["freeze_hidden"],
        method=_text(value["method"]),
        scope=_projection_row(value["scope"], _text),
    )


def _load_channel_source(source):
    """Verify historical binding, retaining the original full forecast evidence."""
    source = Path(source)
    protocol_path, archive = source.with_suffix(".protocol.json"), source.with_suffix(
        ".source.zip"
    )
    for path in (source, protocol_path):
        evidence._require(
            path.stat().st_size <= 32 * 1024**2, "source record too large"
        )
    raw, protocol_raw = source.read_bytes(), protocol_path.read_bytes()
    record, protocol = json_loads(raw), json_loads(protocol_raw)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    evidence._require(
        record["schema"] == "tnfr.conservative-regional-organization-response.v1"
        and protocol["schema"] == "tnfr.conservative-regional-organization-protocol.v1"
        and record["protocol"] == protocol
        and record["protocol_sha256"] == hashlib.sha256(protocol_raw).hexdigest()
        and record["source_archive_sha256"] == archive_sha,
        "retained response, protocol or archive association differs",
    )
    _verify_archive(archive, protocol["source_sha256"])
    with zipfile.ZipFile(archive) as bundle:
        declaration_key = protocol.get(
            "declaration_path", DECLARATION.relative_to(ROOT).as_posix()
        )
        archived = json_loads(bundle.read(declaration_key))
    declaration = protocol["declaration"]
    evidence._require(archived == declaration, "archived declaration differs")
    evidence._require(
        record["evaluation_error"] is None
        and record["response"]["schema"] == "tnfr.sine-regional-organization.v1",
        "source must retain a completed response evaluation",
    )
    forecast = _forecast_projection(record["response"]["report"]["forecast"])
    preparation = _preparation(declaration)
    evidence._require(
        forecast.initial_box == preparation.pop("initial")
        and all(
            getattr(forecast, key) == expected for key, expected in preparation.items()
        )
        and declaration["clock"] == "original_structural_t; tau=t/pi"
        and _integer(declaration["maximum_steps"]) >= len(forecast.steps)
        and forecast.forecast_start is None
        and forecast.freeze_hidden is False
        and type(forecast.order) is int,
        "forecast differs from its frozen preparation, support, clock or budget",
    )
    metadata = {
        "source_response_sha256": hashlib.sha256(raw).hexdigest(),
        "source_protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "source_archive_sha256": archive_sha,
    }
    return forecast, tuple(declaration["receiver_cycle_indices"]), metadata


def analyze_channels(source, output):
    """Retrospective reader of frozen enclosures; never run a new trajectory."""
    from tnfr.physics.relational_sine_regional import assess_sine_regional_channels

    output = Path(output)
    archive = output.with_suffix(".source.zip")
    if output.exists() or archive.exists():
        raise FileExistsError("retain existing channel analysis and source archive")
    forecast, cycle, source_metadata = _load_channel_source(source)
    evidence._verify_runtime_source()
    source_files = _files()
    manifest, runtime = evidence._manifest(source_files), evidence._runtime()
    evidence._archive(archive, source_files)
    _verify_archive(archive, manifest)
    analysis_archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    analysis, error = None, None
    try:
        analysis = assess_sine_regional_channels(
            forecast, cycle_indices=cycle
        ).to_dict()
        evidence._encoded(analysis)
        evidence._verify_runtime_source()
        evidence._require(
            manifest == evidence._manifest(_files())
            and runtime == evidence._runtime()
            and analysis_archive_sha == hashlib.sha256(archive.read_bytes()).hexdigest()
            and source_metadata == _load_channel_source(source)[2],
            "analysis source, runtime or retained input changed during analysis",
        )
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "error": str(exc)}
    evidence._write(
        output,
        {
            "schema": "tnfr.conservative-regional-channel-analysis.v1",
            **source_metadata,
            "analysis_source_sha256": manifest,
            "analysis_source_archive_sha256": analysis_archive_sha,
            "runtime": runtime,
            "scope": "retrospective_analysis_of_supplied_Taylor_enclosures_no_solver_replay_or_authentication",
            "analysis": analysis,
            "analysis_error": error,
        },
    )
    print(
        f"Retained retrospective channel analysis: {output}; error={error}", flush=True
    )
    return 0 if error is None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--analyze-channels", type=Path, metavar="SOURCE_RESPONSE")
    parser.add_argument("--declaration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.analyze_channels is not None:
        if args.declaration is not None:
            parser.error("retrospective analysis uses its source's frozen declaration")
        if args.output is None:
            parser.error("retrospective channel analysis requires a new --output path")
        return analyze_channels(args.analyze_channels, args.output)
    if args.declaration is not None:
        if args.output is None:
            parser.error("a selected declaration requires an explicit --output path")
        args.declaration = args.declaration.resolve()
    output = args.output or DEFAULT_OUTPUT
    protocol_path = output.with_suffix(".protocol.json")
    archive = output.with_suffix(".source.zip")
    if output.exists():
        raise FileExistsError(f"retain existing response: {output}")
    if args.prepare:
        if protocol_path.exists() or archive.exists():
            raise FileExistsError("retain existing protocol and source archive")
        protocol = prepare_protocol(args.declaration)
        evidence._archive(archive, _source_files(args.declaration))
        _verify_archive(archive, protocol["source_sha256"])
        evidence._write(protocol_path, protocol)
        print(f"Frozen protocol: {protocol_path}", flush=True)
        return 0

    evidence._require(
        protocol_path.stat().st_size <= 32 * 1024**2, "protocol too large"
    )
    protocol_bytes = protocol_path.read_bytes()
    protocol = json_loads(protocol_bytes)
    evidence._require(
        evidence._encoded(protocol)
        == evidence._encoded(prepare_protocol(args.declaration)),
        "frozen source, runtime or declaration changed",
    )
    _verify_archive(archive, protocol["source_sha256"])
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    print("Evaluating the fixed complete-law forecast; no adaptive retries", flush=True)
    response, error, passed = None, None, False
    try:
        response = evaluate(json_loads(evidence._encoded(protocol["declaration"])))
        evidence._encoded(response)
        if _evaluation_kind(protocol["declaration"]) == "reversible_preparation":
            evidence._require(
                response.get("schema") == "tnfr.sine-reversible-preparation.v1",
                "reversible evaluation requires its declared response schema",
            )
            passed = response["report"]["outcome"] == "certified"
        else:
            passed = response["report"]["outcome"] == "finite_acute_retention_certified"
        evidence._verify_runtime_source()
        evidence._require(
            protocol["source_sha256"]
            == evidence._manifest(_source_files(args.declaration))
            and protocol["runtime"] == evidence._runtime()
            and protocol_path.read_bytes() == protocol_bytes
            and hashlib.sha256(archive.read_bytes()).hexdigest() == archive_sha,
            "source, runtime or frozen evidence changed during evaluation",
        )
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "error": str(exc)}
        passed = False
    record = {
        "schema": "tnfr.conservative-regional-organization-response.v1",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_archive_sha256": archive_sha,
        "response": response,
        "evaluation_error": error,
        "passed": passed,
    }
    evidence._write(output, record)
    outcome = None if response is None else response["report"]["outcome"]
    print(
        f"Retained response: {output}; outcome={outcome}; passed={passed}", flush=True
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
