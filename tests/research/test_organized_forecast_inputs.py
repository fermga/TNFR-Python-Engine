"""Synthetic input controls; these tests never extract the archived target record."""

from dataclasses import asdict
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.research import _organized_forecast_inputs as owner
from tnfr.research.artifact_io import file_receipt


ROOT = Path(__file__).resolve().parents[2]


def _interval(value):
    return asdict(value)


def _step(radius=Q(0)):
    growth = I(owner._growth_reference_upper(Q(1, 8))).hi
    return {
        "time": Q(0),
        "duration": Q(1, 8),
        "tube": (_interval(I(-1, 1)),) * 10,
        "picard_margin": Q(1, 2),
        "polynomial": (_interval(I(0)),) * 10,
        "remainder": (_interval(I(0)),) * 10,
        "growth": growth,
        "local_error": Q(0),
        "endpoint_center": (Q(0),) * 10,
        "endpoint_radius": I(growth * radius).hi,
    }


def test_metric_step_preserves_an_exact_stationary_center_and_uncertain_ball():
    radius = Q(1, 10**20)
    center, end_radius = owner._audit_metric_step(
        _step(radius), (Q(0),) * 10, radius, time=Q(0), duration=Q(1, 8)
    )
    assert center == (Q(0),) * 10
    assert end_radius >= radius * owner._growth_reference_upper(Q(1, 8)) > radius


@pytest.mark.parametrize(
    "key,value,message",
    (
        ("time", True, "exact record"),
        ("time", Q(1), "clock"),
        ("duration", Q(1, 4), "clock"),
        ("picard_margin", Q(0), "Picard"),
        ("growth", Q(1), "growth"),
        ("local_error", Q(-1), "local metric"),
        ("endpoint_radius", Q(1, 10**10), "radius"),
        ("endpoint_center", (Q(1),) * 10, "center"),
        ("endpoint_center", (False,) * 10, "exact record"),
    ),
)
def test_metric_step_rejects_bad_clock_domain_or_evidence(key, value, message):
    step = _step()
    step[key] = value
    with pytest.raises(ValueError, match=message):
        owner._audit_metric_step(step, (Q(0),) * 10, Q(0), time=Q(0), duration=Q(1, 8))


def test_metric_step_counts_all_ten_local_error_coordinates():
    step = _step()
    unit = Q(1, 2**80)
    step["polynomial"] = (_interval(I(-unit, unit)),) * 10
    step["local_error"] = unit
    with pytest.raises(ValueError, match="all coordinates"):
        owner._audit_metric_step(step, (Q(0),) * 10, Q(0), time=Q(0), duration=Q(1, 8))


@pytest.mark.parametrize("raw", (True, 0.5, "0", {"lo": False, "hi": 1}))
def test_interval_admission_rejects_nonexact_or_boolean_coordinates(raw):
    with pytest.raises(ValueError):
        owner._interval(raw)


def test_interval_admission_does_not_silently_requantize_retained_endpoints():
    with pytest.raises(ValueError, match="already be outward dyadics"):
        owner._interval({"lo": Q(1, 3), "hi": Q(1, 3)})


def test_strict_json_preserves_duplicate_and_nonfinite_rejection():
    for raw in (b'{"time": 0, "time": 1}', b'{"time": NaN}'):
        with pytest.raises(ValueError):
            owner._decode(raw)


def test_schedule_contains_all_thirty_plateau_endpoints_and_no_future_endpoint():
    times = owner._sample_times()
    expected = []
    elapsed = Q(200)
    for _ in range(15):
        expected.extend((elapsed, elapsed + Q(1, 100000)))
        elapsed += 1 + Q(1, 100000)
    assert times == tuple(expected)
    assert len(times) == 30
    assert times[-1] == Q(21400015, 100000)
    assert Q(216) not in times


@pytest.fixture(scope="module")
def synthetic_compact_design():
    # Constant formal columns isolate exact sample/column association and
    # propagation arithmetic. They make no claim about the real forcing.
    columns = [tuple(Q(i == j) for i in range(5)) for j in range(15)]
    rows = tuple(tuple(_interval(I(int(i == j))) for j in range(15)) for i in range(5))
    steps = []
    elapsed = Q(200)
    durations = (Q(1, 100000), Q(1)) * 15 + (Q(99985, 100000),)
    for duration in durations:
        end = elapsed + duration
        while elapsed < end:
            step_duration = min(Q(1, 8), end - elapsed)
            for column in range(15):
                steps.append(
                    {
                        "column": column,
                        "time": elapsed,
                        "duration": step_duration,
                        "polynomial": tuple(_interval(I(v)) for v in columns[column]),
                        "remainder": (_interval(I(0)),) * 5,
                        "endpoint_center": columns[column],
                        "endpoint_radius": Q(0),
                    }
                )
            elapsed += step_duration
    assert elapsed == 216
    return {
        "observation_matrix": rows * 30,
        "future_matrix": rows,
        "steps": tuple(steps),
        "observation_times": tuple(t - 200 for t in owner._sample_times()),
        # These cached summaries must not substitute for primitive evidence.
        "maximum_coefficient_radius": object(),
        "admitted": False,
    }


def test_coefficient_replay_rebuilds_joint_samples_from_primitive_steps(
    synthetic_compact_design,
):
    design = owner._rebuild_design(synthetic_compact_design)
    assert design.maximum_coefficient_radius == design.maximum_local_error == 0
    assert design.admitted is True
    assert design.steps_per_column == 143
    assert len(design.observation_matrix) == 150
    for row_index, row in enumerate(design.observation_matrix):
        assert tuple(value.midpoint for value in row) == tuple(
            Q(column == row_index % 5) for column in range(15)
        )


def test_coefficient_replay_rejects_a_misassociated_sample(synthetic_compact_design):
    raw = dict(synthetic_compact_design)
    rows = list(raw["observation_matrix"])
    rows[0], rows[1] = rows[1], rows[0]
    raw["observation_matrix"] = tuple(rows)
    with pytest.raises(ValueError, match="same history|retained step history"):
        owner._rebuild_design(raw)


def test_synthetic_extraction_returns_only_form_prefix_and_associations(monkeypatch):
    preparation = tuple(Q(4092 * (c - 2)) for c in range(5)) + (Q(0),) * 5
    stationary = tuple(Q(c, 1000) for c in range(5)) + (Q(987654321),) * 5
    radius = Q(1, 10**21)
    report = {
        "clock": "tau=(1023/1024)*t",
        "coordinate_order": "Y[0:5],Psi[0:5]; exact command removed",
        "branch": "outer_one",
        "numerical_method": "prescribed_time_identity_metric_Taylor16_dyadic128",
        "growth_upper": Q(1, 3000),
        "maximum_step": Q(1, 8),
        "source_scope": "conditional_on_original_independent_preparation_and_zero_sums",
        "formation": {
            "initial_center": preparation,
            "initial_radius": Q(0),
            "endpoint_center": stationary,
            "endpoint_radius": radius,
            "steps": ({},) * 1600,
        },
        "observed": {
            "name": "finite_observation",
            "start": Q(200),
            "initial_center": stationary,
            "initial_radius": radius,
            "steps": ({},) * 143,
            # If the extractor attempts to consume the reserved endpoint,
            # these deliberately invalid values must cause the test to fail.
            "endpoint_center": object(),
            "endpoint_radius": object(),
        },
        "omitted": object(),
    }
    inspected = []

    def audited(_step, _center, _radius, *, time, duration):
        inspected.append(time + duration)
        assert time + duration <= owner._sample_times()[-1]
        return stationary, radius

    monkeypatch.setattr(owner, "_audit_metric_step", audited)
    record = owner._extract_record(report)
    assert record.times == owner._sample_times()
    assert record.values == (stationary[:5],) * 30
    assert len(record.extraction_errors) == 30
    assert all(radius < error <= Q(1, 10**14) for error in record.extraction_errors)
    assert record.samples[0].endpoint_step is None
    assert record.samples[0].time == 200
    assert max(inspected) == record.times[-1]
    assert record.record_kind == "archived_synthetic_software_demonstration"
    assert set(asdict(record)) == {
        "times",
        "values",
        "extraction_errors",
        "samples",
        "provenance",
        "record_kind",
    }


@pytest.mark.parametrize("spec", (owner._TRAJECTORY, owner._GAIN))
def test_fixed_archive_receipts_and_attempt_sources_without_extracting_values(spec):
    folder = ROOT / "docs/assets" / spec.name
    assert file_receipt(folder / "attempt.json")["sha256"] == spec.attempt_sha256
    assert file_receipt(folder / "response.zip") == {
        "path": (folder / "response.zip").as_posix(),
        "bytes": spec.archive_bytes,
        "sha256": spec.archive_sha256,
    }
    receipt = owner._decode((folder / "receipt.json").read_bytes())
    assert receipt["result"] == {
        "path": "result.json",
        "bytes": spec.result_bytes,
        "sha256": spec.result_sha256,
    }
    attempt = owner._decode((folder / "attempt.json").read_bytes())
    assert attempt["source_revision"] == spec.revision
    assert attempt["source_tree"] == spec.source_tree
    assert attempt["source_inventory_sha256"] == spec.inventory_sha256


def test_changed_attempt_is_rejected_before_any_archive_is_opened(tmp_path):
    folder = tmp_path / "docs/assets" / owner._GAIN.name
    folder.mkdir(parents=True)
    (folder / "attempt.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="attempt digest"):
        owner._read_archive(tmp_path, owner._GAIN)
