"""Offline time/gap/resource controls for bounded temporal ingestion."""

import hashlib
import importlib.util
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks/temporal_interface_benchmark.py"
)
_SPEC = importlib.util.spec_from_file_location("temporal_ingestion_benchmark", _PATH)
BENCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = BENCH
_SPEC.loader.exec_module(BENCH)


def archive(tmp_path, text, extra=()):
    path = tmp_path / "input.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("trace.csv", text)
        for name, content in extra:
            output.writestr(name, content)
    return path


def test_stride_preserves_every_timestamp_and_missing_row(tmp_path):
    raw = (
        "timestamp;frequency\n2020-01-01T00:00:00Z;50,01\n"
        "2020-01-01T00:00:01Z;\n2020-01-01T00:00:03Z;NaN\n"
        "2020-01-01T00:00:04Z;bad\n2020-01-01T00:00:08Z;49,99\n"
    )
    path = archive(tmp_path, raw)
    record = BENCH.load_grid_frequency_record(path, max_points=2)
    assert record.values_hz == (50.01, None, None, None, 49.99)
    assert record.missing == (False, True, True, True, False)
    assert record.elapsed_seconds == (0, 1, 3, 4, 8)
    assert record.source_line_numbers == (2, 3, 4, 5, 6)
    assert record.selected_indices == (0, 3)
    assert record.stride == 3
    assert record.time_status == "relative_seconds"
    assert record.archive_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record.member_sha256 == hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize(
    "times,status",
    [
        (("invalid", "2020-01-01T00:00:01Z"), "unavailable_timestamp"),
        (("2020-01-01T00:00:01Z", "2020-01-01T00:00:00Z"), "nonmonotone_timestamp"),
        (("2020-01-01T00:00:00", "2020-01-01T00:00:01Z"), "mixed_timezone_unavailable"),
    ],
)
def test_time_unavailability_is_explicit_without_dropping_samples(
    tmp_path, times, status
):
    path = archive(tmp_path, f"timestamp,frequency\n{times[0]},50\n{times[1]},51\n")
    record = BENCH.load_grid_frequency_record(path)
    assert record.timestamps == times
    assert record.values_hz == (50, 51)
    assert record.time_status == status


@pytest.mark.parametrize(
    "times,elapsed,status",
    [
        (
            ("2020-01-01T00:00:00Z", "2020-01-01T00:00:01.25Z"),
            (0.0, 1.25),
            "relative_seconds",
        ),
        (
            ("2020-01-01T00:00:00Z", "2020-01-01T01:00:01+01:00"),
            (0.0, 1.0),
            "relative_seconds",
        ),
        (
            ("2020-01-01T00:00:01Z", "2020-01-01T00:00:00Z"),
            (0.0, -1.0),
            "nonmonotone_timestamp",
        ),
        (
            ("2020-01-01T00:00:00", "2020-01-01T00:00:01Z"),
            (None, None),
            "mixed_timezone_unavailable",
        ),
        (
            ("2020-01-01Z", "2020-01-01T00:00:01Z"),
            (None, None),
            "unavailable_timestamp",
        ),
    ],
)
def test_utc_designator_with_legacy_iso_parser_preserves_time_admission(
    tmp_path, monkeypatch, times, elapsed, status
):
    received = []

    class LegacyDatetime:
        @staticmethod
        def fromisoformat(value):
            received.append(value)
            # Reproduce Python 3.10's missing terminal-Z support without
            # requiring that interpreter on every development machine.
            if value.endswith("Z"):
                raise ValueError("Invalid isoformat string")
            return datetime.fromisoformat(value)

    monkeypatch.setattr(BENCH, "datetime", LegacyDatetime)
    path = archive(tmp_path, f"timestamp,frequency\n{times[0]},50\n{times[1]},\n")
    record = BENCH.load_grid_frequency_record(path)
    assert record.timestamps == times
    assert record.values_hz == (50.0, None)
    assert record.missing == (False, True)
    assert record.source_line_numbers == (2, 3)
    assert record.elapsed_seconds == elapsed
    assert record.time_status == status
    assert len(received) == 2 and all(not value.endswith("Z") for value in received)


@pytest.mark.parametrize(
    "limits,match",
    [
        ({"max_bytes": 8}, "compressed"),
        ({"max_expanded_bytes": 50}, "expanded"),
        ({"max_member_bytes": 60}, "member exceeds"),
        ({"max_members": 1}, "member-count"),
        ({"max_rows": 1}, "row limit"),
    ],
)
def test_cached_zip_resources_are_bounded_before_analysis(tmp_path, limits, match):
    path = archive(
        tmp_path,
        "timestamp,frequency\n2020-01-01T00:00:00Z,50\n2020-01-01T00:00:01Z,50\n",
        extra=(("unused.txt", "a" * 100),),
    )
    with pytest.raises(ValueError, match=match):
        BENCH.load_grid_frequency_record(path, **limits)


def test_download_rejects_oversized_existing_cache_without_network(
    tmp_path, monkeypatch
):
    path = tmp_path / "cache.zip"
    path.write_bytes(b"x" * 32)
    monkeypatch.setattr(
        BENCH, "urlopen", lambda *a, **k: pytest.fail("network must not run")
    )
    assert (
        BENCH.download_grid_frequency_month(2020, 1, cache_path=path, max_bytes=31)
        is None
    )


@pytest.mark.parametrize("bad", [True, 0, -1, 2.5])
def test_invalid_resource_limits_are_not_coerced(tmp_path, bad):
    with pytest.raises(ValueError, match="positive integer"):
        BENCH.load_grid_frequency_record(tmp_path / "not-opened", max_points=bad)


def test_grid_benchmark_abstains_instead_of_compacting_gaps(tmp_path, monkeypatch):
    path = archive(
        tmp_path,
        "timestamp,frequency\n2020-01-01T00:00:00Z,50\n2020-01-01T00:00:01Z,\n",
    )
    monkeypatch.setattr(BENCH, "download_grid_frequency_month", lambda *a, **k: path)
    report = BENCH.run_temporal_benchmark(
        source="grid",
        year=2020,
        month=1,
        config=BENCH.TemporalInterfaceConfig(),
        max_points=100,
        max_bytes=10000,
    )
    assert report["status"] == "unavailable"
    assert report["ingestion"]["missing"] == (False, True)
    assert report["prospective_prediction"] is False


def test_expected_digest_binds_cache_before_zip_parse(tmp_path, monkeypatch):
    path = archive(tmp_path, "timestamp,frequency\n2020-01-01T00:00:00Z,50\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    admitted = BENCH.load_grid_frequency_record(
        path,
        expected_archive_sha256="sha256:" + digest.upper(),
    )
    assert admitted.archive_sha256 == digest
    path.write_bytes(b"altered cache, not even a ZIP")
    monkeypatch.setattr(
        BENCH.zipfile,
        "ZipFile",
        lambda *a, **k: pytest.fail("mismatch must precede ZIP parsing"),
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        BENCH.load_grid_frequency_record(path, expected_archive_sha256=digest)


@pytest.mark.parametrize("digest", [True, "", "0" * 63, "g" * 64, "sha256:" + "1" * 65])
def test_expected_digest_schema_is_strict_before_file_io(tmp_path, digest):
    with pytest.raises(ValueError, match="must be a SHA-256 digest"):
        BENCH.load_grid_frequency_record(
            tmp_path / "absent.zip", expected_archive_sha256=digest
        )


def test_multiple_csv_members_require_explicit_selection(tmp_path):
    first = "timestamp,frequency\n2020-01-01T00:00:00Z,50\n"
    second = "timestamp,frequency\n2020-01-01T00:00:00Z,51\n"
    path = archive(tmp_path, first, extra=(("other.csv", second),))
    with pytest.raises(ValueError, match="explicit member_name"):
        BENCH.load_grid_frequency_record(path)
    record = BENCH.load_grid_frequency_record(path, member_name="other.csv")
    assert record.values_hz == (51,)
    assert record.member_name == "other.csv"
