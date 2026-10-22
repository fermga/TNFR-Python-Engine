"""Bounded bundle recovery tests use synthetic bytes, never remote producers."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import development_bundle_evidence as helper


def _entry(path, data, *, tracked=True):
    result = {
        "path": path,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if tracked:
        result.update(
            git_blob=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
            git_mode="100644",
        )
    return result


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "bundle-root"
    root.mkdir()
    monkeypatch.setattr(helper, "ROOT", root)
    monkeypatch.setattr(helper, "MANIFEST", root / "bundle/manifest.json")
    supplied = {"README.md": b"Read the owners.\n", "src/tnfr/model.py": b"x = 1\n"}
    generated = {"START_HERE.md": b"Read AGENTS.md first.\n"}
    omitted = {
        "docs/assets/one/result.json": b'{"observed":true}\n',
        "docs/assets/two/source.zip": b"opaque nested archive, never executed",
    }
    manifest = {
        "schema": 1,
        "version": "7.8.9",
        "commit": "1" * 40,
        "repository": helper.REPOSITORY,
        "included": [_entry(name, data) for name, data in supplied.items()],
        "omitted": [_entry(name, data) for name, data in omitted.items()],
        "generated": [
            _entry(name, data, tracked=False) for name, data in generated.items()
        ],
    }
    for name, data in {**supplied, **generated}.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    helper.MANIFEST.parent.mkdir()
    helper.MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest, omitted


def _archive(path, members):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in members:
                if isinstance(name, zipfile.ZipInfo):
                    archive.writestr(name, data)
                else:
                    info = zipfile.ZipInfo(name)
                    # Do not let ZipInfo sanitize malformed test names.
                    info.filename = name
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data)
    return path


def test_verify_distinguishes_missing_evidence_from_missing_supplied_file(bundle):
    root, manifest, _ = bundle
    assert helper.verify(helper.load_manifest()) == (3, 2)
    with pytest.raises(ValueError, match="some evidence is absent"):
        helper.verify(manifest, True)
    (root / "README.md").unlink()
    with pytest.raises(ValueError, match="Missing regular file: README.md"):
        helper.verify(manifest)


def test_verify_rejects_edited_included_and_restored_files(bundle):
    root, manifest, omitted = bundle
    name, data = next(iter(omitted.items()))
    target = root / name
    target.parent.mkdir(parents=True)
    target.write_bytes(data[:-1] + b"x")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        helper.verify(manifest)
    target.unlink()
    (root / "README.md").write_bytes(b"user edits")
    with pytest.raises(ValueError, match="Size mismatch"):
        helper.verify(manifest)


@pytest.mark.parametrize("version, width", [("0.1.0", 40), ("12.3.4rc1", 64)])
def test_metadata_is_not_tied_to_an_old_release(bundle, version, width):
    _, manifest, _ = bundle
    manifest["version"] = version
    manifest["commit"] = "a" * width
    for group in ("included", "omitted"):
        for entry in manifest[group]:
            entry["git_blob"] = "b" * width
    manifest["provenance"] = {"scope": "recorded bytes"}
    helper.validate_manifest(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", True),
        ("schema", 2),
        ("version", 1),
        ("version", "v1.0"),
        ("version", "1\n2"),
        ("commit", "main"),
        ("commit", "A" * 40),
        ("repository", "someone/else"),
        ("included", {}),
        ("omitted", None),
    ],
)
def test_invalid_manifest_metadata(bundle, field, value):
    _, manifest, _ = bundle
    manifest[field] = value
    with pytest.raises(ValueError):
        helper.validate_manifest(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("bytes", True),
        ("bytes", -1),
        ("bytes", 1.0),
        ("bytes", helper.MAX_FILE_BYTES + 1),
        ("sha256", ["a"] * 64),
        ("sha256", "A" * 64),
        ("git_blob", "b" * 64),
        ("git_mode", 100644),
        ("git_mode", "120000"),
        ("path", 1),
    ],
)
def test_invalid_manifest_entry(bundle, field, value):
    _, manifest, _ = bundle
    manifest["omitted"][0][field] = value
    with pytest.raises(ValueError):
        helper.validate_manifest(manifest)


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/absolute",
        "docs//file",
        "docs/./file",
        "docs\\file",
        "docs/NUL.txt",
        "docs/COM¹",
        "docs/trailing.",
        "docs/trailing ",
        "docs/a\nb",
        "docs/a\x00b",
        "docs/e\u0301.txt",
        "docs/a\ud800b",
    ],
)
def test_portable_names_reject_ambiguous_paths(name):
    with pytest.raises(ValueError, match="Unsafe path"):
        helper.canonical_name(name)


@pytest.mark.parametrize(
    "name", ["docs/assets/mathjax.js", "docs/assets-other/a", "src/data.json"]
)
def test_only_declared_evidence_domain_can_be_omitted(bundle, name):
    _, manifest, _ = bundle
    manifest["omitted"][0]["path"] = name
    with pytest.raises(ValueError, match="Unexpected omitted evidence"):
        helper.validate_manifest(manifest)


@pytest.mark.parametrize(
    "name",
    ["README.md", "readme.md", "src", "SRC/other.py", "src/tnfr/model.py/child"],
)
def test_cross_group_and_implicit_directory_collisions(bundle, name):
    _, manifest, _ = bundle
    manifest["generated"][0]["path"] = name
    with pytest.raises(ValueError):
        helper.validate_manifest(manifest)


@pytest.mark.parametrize("text", ['{"schema":1,"schema":1}', "[1,2]", "null"])
def test_strict_json_object_and_duplicates(bundle, text):
    helper.MANIFEST.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        helper.load_manifest()


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_values_in_informational_fields_are_rejected(bundle, value):
    _, manifest, _ = bundle
    raw = json.dumps(manifest)[:-1] + ',"extra":' + value + "}"
    helper.MANIFEST.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="Nonfinite"):
        helper.load_manifest()


def test_read_and_aggregate_review_budgets_are_enforced(bundle, monkeypatch):
    _, manifest, _ = bundle
    monkeypatch.setattr(helper, "MAX_MANIFEST_BYTES", 8)
    with pytest.raises(ValueError, match="review limit"):
        helper.load_manifest()
    monkeypatch.setattr(helper, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="review limit"):
        helper.validate_manifest(manifest)
    monkeypatch.setattr(helper, "MAX_TOTAL_BYTES", 10_000)
    monkeypatch.setattr(helper, "MAX_ENTRIES", 2)
    with pytest.raises(ValueError, match="review limit"):
        helper.validate_manifest(manifest)


def test_prefix_selection_has_directory_boundaries(bundle):
    _, manifest, _ = bundle
    assert helper.select_entries(manifest, "docs/assets/one/", False) == [
        manifest["omitted"][0]
    ]
    assert helper.select_entries(manifest, None, True) == manifest["omitted"]
    with pytest.raises(ValueError, match="No omitted evidence matches"):
        helper.select_entries(manifest, "docs/assets/on", False)


class _Response(io.BytesIO):
    def __init__(self, data, url):
        super().__init__(data)
        self.url = url
        self.requests = []

    def geturl(self):
        return self.url

    def read(self, size=-1):
        assert size >= 0, "Downloads must use a bounded read"
        self.requests.append(size)
        return super().read(size)


def test_fetch_uses_pinned_official_urls_and_preserves_idempotence(bundle, monkeypatch):
    root, manifest, omitted = bundle
    requests = []
    responses = []
    manifest["omitted"][0]["download_url"] = "https://untrusted.invalid/payload"
    manifest["version"] = "55.66.77"
    manifest["commit"] = "9" * 40

    def download(request, *, timeout):
        assert timeout == 60
        requests.append(request.full_url)
        prefix = (
            f"https://raw.githubusercontent.com/{helper.REPOSITORY}/"
            f"{manifest['commit']}/"
        )
        assert request.full_url.startswith(prefix)
        data = omitted[request.full_url.removeprefix(prefix)]
        response = _Response(data, request.full_url)
        responses.append(response)
        return response

    monkeypatch.setattr(helper, "urlopen", download)
    assert helper.restore(manifest, manifest["omitted"]) == (2, 0)
    assert helper.restore(manifest, manifest["omitted"]) == (0, 2)
    assert len(requests) == 2
    assert [response.requests for response in responses] == [
        [len(data) + 1] for data in omitted.values()
    ]
    assert helper.verify(manifest, True) == (5, 0)
    assert (root / "docs/assets/two/source.zip").read_bytes() == omitted[
        "docs/assets/two/source.zip"
    ]
    assert not list(root.rglob(".tnfr-*"))


@pytest.mark.parametrize("failure", ["size", "hash", "redirect", "network"])
def test_download_failure_never_installs_unverified_bytes(bundle, monkeypatch, failure):
    root, manifest, omitted = bundle
    entry = manifest["omitted"][0]

    def download(request, *, timeout):
        if failure == "network":
            raise OSError("offline")
        data = omitted[entry["path"]]
        if failure == "size":
            data += b"extra"
        if failure == "hash":
            data = b"x" * len(data)
        url = (
            "https://untrusted.invalid/file"
            if failure == "redirect"
            else request.full_url
        )
        return _Response(data, url)

    monkeypatch.setattr(helper, "urlopen", download)
    with pytest.raises((ValueError, OSError)):
        helper.restore(manifest, [entry])
    assert not (root / entry["path"]).exists()
    assert not list(root.rglob(".tnfr-*"))


def test_completed_downloads_survive_a_later_failure(bundle, monkeypatch):
    root, manifest, omitted = bundle
    first = manifest["omitted"][0]

    def download(request, *, timeout):
        if request.full_url.endswith(first["path"]):
            return _Response(omitted[first["path"]], request.full_url)
        raise OSError("connection lost")

    monkeypatch.setattr(helper, "urlopen", download)
    with pytest.raises(OSError, match="connection lost"):
        helper.restore(manifest, manifest["omitted"])
    assert (root / first["path"]).read_bytes() == omitted[first["path"]]
    assert helper.verify(manifest) == (4, 1)


def test_existing_user_edit_is_not_downloaded_or_overwritten(bundle, monkeypatch):
    root, manifest, _ = bundle
    entry = manifest["omitted"][0]
    target = root / entry["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user investigation")

    def forbidden(*args, **kwargs):
        pytest.fail("Existing edited files must be rejected before downloading")

    monkeypatch.setattr(helper, "urlopen", forbidden)
    with pytest.raises(ValueError, match="(?:Size|SHA256) mismatch"):
        helper.restore(manifest, [entry])
    assert target.read_bytes() == b"user investigation"


@pytest.mark.parametrize("matching", [False, True])
def test_atomic_install_preserves_a_competing_file(bundle, monkeypatch, matching):
    root, manifest, omitted = bundle
    entry = manifest["omitted"][0]
    data = omitted[entry["path"]]
    real_link = os.link

    def competing_write(source, destination):
        destination.write_bytes(data if matching else b"concurrent user edit")
        return real_link(source, destination)

    monkeypatch.setattr(helper.os, "link", competing_write)
    if matching:
        assert helper._install(data, entry) is False
    else:
        with pytest.raises(ValueError, match="Size mismatch"):
            helper._install(data, entry)
    assert (root / entry["path"]).read_bytes() == (
        data if matching else b"concurrent user edit"
    )
    assert not list(root.rglob(".tnfr-*"))


def test_archive_restore_is_complete_and_never_opens_nested_archives(bundle, tmp_path):
    root, manifest, omitted = bundle
    path = _archive(
        tmp_path / "release.zip",
        [("release/", b""), *[(f"release/{k}", v) for k, v in omitted.items()]],
    )
    assert helper.restore(manifest, manifest["omitted"], path) == (2, 0)
    assert helper.verify(manifest, True) == (5, 0)
    assert helper.restore(manifest, manifest["omitted"], path) == (0, 2)
    assert (root / "docs/assets/two/source.zip").read_bytes().startswith(b"opaque")


@pytest.mark.parametrize(
    "extra",
    [
        [("other/extra", b"x")],
        [("release/../escape", b"x")],
        [("release/docs\\escape", b"x")],
        [("release/docs/assets/one/result.json", b"duplicate")],
        [("release/DOCS/other", b"x")],
        [("release/docs/assets/one", b"x")],
        [("release/docs/assets/one/RESULT.json", b"x")],
    ],
)
def test_archive_paths_are_checked_even_for_unselected_files(bundle, tmp_path, extra):
    root, manifest, omitted = bundle
    path = _archive(
        tmp_path / "bad.zip",
        [*[(f"release/{k}", v) for k, v in omitted.items()], *extra],
    )
    with pytest.raises(ValueError):
        helper.restore(manifest, [manifest["omitted"][0]], path)
    assert not (root / manifest["omitted"][0]["path"]).exists()


def test_archive_symlink_and_missing_wrong_members_are_rejected(bundle, tmp_path):
    _, manifest, omitted = bundle
    link = zipfile.ZipInfo("release/docs/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    path = _archive(tmp_path / "link.zip", [(link, b"/outside")])
    with pytest.raises(ValueError, match="Unsupported archive member"):
        helper.restore(manifest, manifest["omitted"], path)
    path = _archive(tmp_path / "missing.zip", [("release/unrelated", b"x")])
    with pytest.raises(ValueError, match="Missing/wrong archive member"):
        helper.restore(manifest, manifest["omitted"], path)
    name = next(iter(omitted))
    path = _archive(tmp_path / "wrong.zip", [(f"release/{name}", b"x")])
    with pytest.raises(ValueError, match="Missing/wrong archive member"):
        helper.restore(manifest, manifest["omitted"], path)


def test_archive_directory_is_bounded_before_zipfile_allocation(
    bundle, tmp_path, monkeypatch
):
    _, manifest, _ = bundle
    path = _archive(tmp_path / "oversized.zip", [("release/a", b"a")])
    data = bytearray(path.read_bytes())
    position = data.rfind(b"PK\x05\x06")
    struct.pack_into("<L", data, position + 12, helper.MAX_CENTRAL_DIRECTORY_BYTES + 1)
    path.write_bytes(data)

    def forbidden(*args, **kwargs):
        pytest.fail("ZipFile must not parse an oversized directory")

    monkeypatch.setattr(helper.zipfile, "ZipFile", forbidden)
    with pytest.raises(ValueError, match="oversized ZIP directory"):
        helper.restore(manifest, manifest["omitted"], path)


def test_destination_symlink_is_refused_without_touching_outside(bundle, tmp_path):
    root, manifest, _ = bundle
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "docs").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this platform")
    with pytest.raises(ValueError, match="Link destination"):
        helper.restore(manifest, manifest["omitted"])
    assert list(outside.iterdir()) == []


def test_manifest_ancestor_symlink_is_refused(bundle, tmp_path):
    root, _, _ = bundle
    outside = tmp_path / "outside"
    outside.mkdir()
    helper.MANIFEST.unlink()
    helper.MANIFEST.parent.rmdir()
    try:
        (root / "bundle").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this platform")
    (outside / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Link destination"):
        helper.load_manifest()


@pytest.mark.parametrize(
    "mode,attributes",
    [(stat.S_IFLNK, 0), (stat.S_IFDIR, 0x400)],
)
def test_link_and_windows_reparse_admission_without_platform_privilege(
    bundle, monkeypatch, mode, attributes
):
    root, _, _ = bundle
    target = root / "docs"
    original = Path.lstat

    def synthetic_status(path):
        if path == target:
            return SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
        return original(path)

    monkeypatch.setattr(
        helper.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400, raising=False
    )
    monkeypatch.setattr(Path, "lstat", synthetic_status)
    with pytest.raises(ValueError, match="Link destination"):
        helper.destination("docs/assets/result.json")


def test_existing_disk_case_alias_is_refused(bundle):
    root, _, _ = bundle
    (root / "Docs").mkdir()
    with pytest.raises(ValueError, match="Existing case-colliding"):
        helper.destination("docs/assets/result.json")


def test_cli_reports_failures_without_tracebacks_and_can_fetch(
    bundle, tmp_path, capsys
):
    _, _, omitted = bundle
    assert helper.main(["list"]) == 0
    assert "docs/assets/one/result.json" in capsys.readouterr().out
    assert helper.main(["verify", "--require-all-evidence"]) == 1
    assert "ERROR: Full evidence requested" in capsys.readouterr().err
    archive = _archive(
        tmp_path / "release.zip", [(f"release/{k}", v) for k, v in omitted.items()]
    )
    assert helper.main(["fetch", "--all", "--archive", str(archive)]) == 0
    assert helper.main(["verify", "--require-all-evidence"]) == 0


def test_helper_runs_without_site_packages_or_an_installed_engine(bundle):
    root, _, _ = bundle
    target = root / "scripts/development_bundle_evidence.py"
    target.parent.mkdir()
    shutil.copyfile(Path(helper.__file__), target)
    process = subprocess.run(
        [sys.executable, "-I", "-S", str(target), "list"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert process.returncode == 0, process.stderr
    assert "docs/assets/two/source.zip" in process.stdout


def test_restore_rejects_non_manifest_entries_before_any_download(bundle, monkeypatch):
    _, manifest, _ = bundle
    entry = copy.deepcopy(manifest["omitted"][0])
    entry["path"] = "docs/assets/unregistered"

    def forbidden(*args, **kwargs):
        pytest.fail("Unregistered entries must not initiate downloads")

    monkeypatch.setattr(helper, "urlopen", forbidden)
    with pytest.raises(ValueError, match="manifest-admitted"):
        helper.restore(manifest, [entry])
