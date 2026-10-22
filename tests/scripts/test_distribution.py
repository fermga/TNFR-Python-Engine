"""Publication guard tests use tiny archives, without building or extracting."""

from __future__ import annotations

import io
import json
import stat
import tarfile
import warnings
import zipfile
from pathlib import Path

import pytest

from scripts.check_distribution import ReviewLimits, check_distributions, main

VERSION = "1.2.3"
WHEEL = f"tnfr-{VERSION}-py3-none-any.whl"
SDIST = f"tnfr-{VERSION}.tar.gz"
PREFIX = f"tnfr-{VERSION}/"
PACKAGE = {
    "__init__.py": b'"""Tiny package."""\n',
    "api.pyi": b"def answer() -> int: ...\n",
    "py.typed": b"",
    "schemas/grammar.json": b'{"type":"object"}\n',
}


def _wheel_files():
    prefix = f"tnfr-{VERSION}.dist-info/"
    return {
        **{f"tnfr/{name}": data for name, data in PACKAGE.items()},
        prefix + "METADATA": b"Metadata-Version: 2.4\nName: tnfr\nVersion: 1.2.3\n",
        prefix + "WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\n",
        prefix + "RECORD": b"",
        prefix + "licenses/LICENSE.md": b"MIT\n",
    }


def _sdist_files():
    return {
        **{PREFIX + f"src/tnfr/{name}": data for name, data in PACKAGE.items()},
        **{
            PREFIX + name: b"maintained source metadata\n"
            for name in (
                "MANIFEST.in",
                "pyproject.toml",
                "README.md",
                "LICENSE.md",
                "CITATION.cff",
                "setup.cfg",
                "PKG-INFO",
                "scripts/tnfr_is_prime.py",
                "src/tnfr.egg-info/SOURCES.txt",
            )
        },
    }


def _write_archive(path: Path, files, *, special=None):
    if path.name.endswith(".whl"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(
                path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for name, data in files:
                    info = zipfile.ZipInfo(name)
                    # Keep malformed names as supplied, even on Windows.
                    info.filename = name
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data)
                if special is not None:
                    info = zipfile.ZipInfo(special)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    archive.writestr(info, b"../../outside")
    else:
        with tarfile.open(path, "w:gz") as archive:
            root = tarfile.TarInfo(PREFIX)
            root.type = tarfile.DIRTYPE
            archive.addfile(root)
            for name, data in files:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            if special is not None:
                info = tarfile.TarInfo(special)
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                archive.addfile(info)


@pytest.fixture
def distribution(tmp_path):
    source = tmp_path / "source"
    for name, data in PACKAGE.items():
        path = source / "src" / "tnfr" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    output = tmp_path / "dist"
    output.mkdir()
    _write_archive(output / WHEEL, _wheel_files().items())
    _write_archive(output / SDIST, _sdist_files().items())
    return source, output


def test_complete_archives_report_payload_without_extraction(distribution):
    source, output = distribution
    before = {path: path.read_bytes() for path in output.iterdir()}
    report = check_distributions(output, source_root=source)
    assert report["source_files"] == 4
    assert {item["kind"] for item in report["archives"]} == {"wheel", "sdist"}
    for item in report["archives"]:
        assert item["categories"]["python"]["files"] == 1
        assert item["categories"]["stubs"]["files"] == 1
        assert item["categories"]["schemas"]["files"] == 1
        assert item["categories"]["typing_marker"]["files"] == 1
        assert item["expanded_bytes"] > 0
        assert item["largest_files"][0]["bytes"] >= item["largest_files"][-1]["bytes"]
    assert {path: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize("kind", ("wheel", "sdist"))
@pytest.mark.parametrize(
    "suffix",
    (
        "docs/theory.md",
        "tests/test_state.py",
        "__pycache__/state.pyc",
        "assets/raw.json",
        "archive/old.py",
        "logs/run.log",
        ".venv/runner.py",
        "source.zip",
    ),
)
def test_forbidden_package_baggage_rejects(distribution, kind, suffix):
    source, output = distribution
    files = _wheel_files() if kind == "wheel" else _sdist_files()
    prefix = "tnfr/" if kind == "wheel" else PREFIX + "src/tnfr/"
    files[prefix + suffix] = b"unwanted payload"
    archive = output / (WHEEL if kind == "wheel" else SDIST)
    _write_archive(archive, files.items())
    with pytest.raises(ValueError, match="Forbidden package"):
        check_distributions(output, source_root=source)


@pytest.mark.parametrize("kind", ("wheel", "sdist"))
@pytest.mark.parametrize(
    "name",
    (
        "../outside.py",
        "/absolute.py",
        "tnfr\\hidden.py",
        "tnfr/./hidden.py",
        "tnfr//hidden.py",
    ),
)
def test_noncanonical_paths_reject_without_extracting(distribution, kind, name):
    source, output = distribution
    files = _wheel_files() if kind == "wheel" else _sdist_files()
    files[name] = b"payload"
    archive = output / (WHEEL if kind == "wheel" else SDIST)
    _write_archive(archive, files.items())
    with pytest.raises(ValueError, match="Noncanonical archive path"):
        check_distributions(output, source_root=source)
    assert not (output.parent / "outside.py").exists()


@pytest.mark.parametrize("kind", ("wheel", "sdist"))
def test_duplicate_and_nonregular_members_reject(distribution, kind):
    source, output = distribution
    files = _wheel_files() if kind == "wheel" else _sdist_files()
    archive = output / (WHEEL if kind == "wheel" else SDIST)
    duplicate = next(iter(files.items()))
    _write_archive(archive, [*files.items(), duplicate])
    with pytest.raises(ValueError, match="duplicate archive path"):
        check_distributions(output, source_root=source)
    prefix = "tnfr/" if kind == "wheel" else PREFIX + "src/tnfr/"
    _write_archive(archive, files.items(), special=prefix + "link.py")
    with pytest.raises(ValueError, match="nonregular member"):
        check_distributions(output, source_root=source)


def test_directory_entries_cannot_hide_expanded_payload(distribution):
    source, output = distribution
    with zipfile.ZipFile(
        output / WHEEL, "a", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("tnfr/", b"hidden data")
    with pytest.raises(ValueError, match="directory contains payload"):
        check_distributions(output, source_root=source)


@pytest.mark.parametrize("kind", ("wheel", "sdist"))
@pytest.mark.parametrize("change", ("missing", "extra", "modified"))
def test_archive_source_must_match_current_modules_and_stubs(
    distribution, kind, change
):
    source, output = distribution
    files = _wheel_files() if kind == "wheel" else _sdist_files()
    prefix = "tnfr/" if kind == "wheel" else PREFIX + "src/tnfr/"
    if change == "missing":
        del files[prefix + "api.pyi"]
    elif change == "extra":
        files[prefix + "retired.py"] = b"pass\n"
    else:
        files[prefix + "__init__.py"] = b"outdated = True\n"
    archive = output / (WHEEL if kind == "wheel" else SDIST)
    _write_archive(archive, files.items())
    with pytest.raises(ValueError, match="package source|source content differs"):
        check_distributions(output, source_root=source)


@pytest.mark.parametrize("kind", ("wheel", "sdist"))
def test_required_metadata_cannot_be_omitted_or_binary(distribution, kind):
    source, output = distribution
    files = _wheel_files() if kind == "wheel" else _sdist_files()
    name = (
        f"tnfr-{VERSION}.dist-info/METADATA" if kind == "wheel" else PREFIX + "PKG-INFO"
    )
    del files[name]
    archive = output / (WHEEL if kind == "wheel" else SDIST)
    _write_archive(archive, files.items())
    with pytest.raises(ValueError, match="missing metadata"):
        check_distributions(output, source_root=source)
    files[name] = b"\x00binary"
    _write_archive(archive, files.items())
    with pytest.raises(ValueError, match="binary content"):
        check_distributions(output, source_root=source)


@pytest.mark.parametrize("budget", ("compressed", "expanded", "member"))
def test_explicit_review_budgets_are_enforced_before_success(distribution, budget):
    source, output = distribution
    with pytest.raises(ValueError, match="review budget"):
        check_distributions(
            output, source_root=source, limits=ReviewLimits(**{budget: 1})
        )


def test_both_package_kinds_and_one_version_are_required(distribution):
    source, output = distribution
    sdist = output / SDIST
    original = sdist.read_bytes()
    sdist.unlink()
    with pytest.raises(ValueError, match="both a wheel and an sdist"):
        check_distributions(output, source_root=source)
    (output / "tnfr-0.1.tar.gz").write_bytes(original)
    with pytest.raises(ValueError, match="multiple package versions"):
        check_distributions(output, source_root=source)


def test_cli_machine_report_and_deliberate_budget_override(distribution, capsys):
    source, output = distribution
    main(
        [
            str(output),
            "--source-root",
            str(source),
            "--max-member-mib",
            "0.5",
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["limits_bytes"]["member"] == 512 * 1024
    assert report["scope"] == "distribution_payload_and_review_budgets"


@pytest.mark.parametrize("budget", ("nan", "inf", "0", "-1", "1e-1000"))
def test_cli_invalid_budget_fails_before_inspection(distribution, capsys, budget):
    _, output = distribution
    with pytest.raises(SystemExit) as exc:
        main([str(output), "--max-expanded-mib", budget])
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""


def test_cli_invalid_archive_exits_nonzero_without_success_report(distribution, capsys):
    source, output = distribution
    (output / WHEEL).write_bytes(b"broken zip")
    with pytest.raises(SystemExit) as exc:
        main([str(output), "--source-root", str(source)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Distribution review failed" in captured.err
