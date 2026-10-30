"""Exercise release bundles with tiny committed trees, not research producers."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_development_bundle as bundle
from scripts.research_book import (
    BOOK_PATH,
    CATALOG_PATH,
    Alias,
    Book,
    Chapter,
    write_book,
    write_modular_book,
)

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Release builder uses Python 3.11+ tomllib"
)


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()


def _commit(root):
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "Synthetic release source")
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Bundle test")
    _git(root, "config", "user.email", "bundle@example.invalid")
    _git(root, "config", "core.autocrlf", "false")
    for name in bundle.REQUIRED | {
        "theory/EMERGENT_ONTOLOGY.md",
        "theory/research/FIVE_STAGE_EXECUTION_PLAN.md",
    }:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"retained input\r\n")
    (root / "src/tnfr/_version.py").write_text('__version__ = "1.2.3"\n')
    (root / "pyproject.toml").write_text(
        '[project]\nname = "tnfr"\nversion = "1.2.3"\n'
    )
    (root / bundle.TEMPLATE).write_bytes(
        b"TNFR ${version}\n${commit}\n${repository}\n${omitted_bytes}\n"
    )
    evidence = root / "docs/assets/sample/response.json"
    evidence.parent.mkdir()
    evidence.write_bytes(b'{"answer":42}\r\n')
    (root / "src/tnfr/example.py").write_bytes(b"VALUE = 42\r\n")
    (root / ".gitignore").write_text("ignored.tmp\n")
    commit = _commit(root)
    _git(root, "tag", "v1.2.3")
    return root, commit


def test_selected_commit_and_raw_bytes_control_bundle_not_working_tree(
    source, tmp_path
):
    root, commit = source
    (root / "src/tnfr/example.py").write_bytes(b"VALUE = -1\n")
    (root / "untracked.txt").write_text("not in release")
    (root / "ignored.tmp").write_text("not in release")
    output = tmp_path / "output"
    report = bundle.build_bundle(output, root=root, expected_tag="v1.2.3")
    assert report["commit"] == commit
    assert report["version"] == "1.2.3"
    path = Path(report["archive"])
    prefix = "TNFR-1.2.3-light-development/"
    with zipfile.ZipFile(path) as archive:
        assert archive.read(prefix + "src/tnfr/example.py") == b"VALUE = 42\r\n"
        assert prefix + "docs/assets/mathjax.js" in archive.namelist()
        assert prefix + "docs/assets/sample/response.json" not in archive.namelist()
        assert not any(
            "untracked" in name or "ignored.tmp" in name for name in archive.namelist()
        )
        manifest = json.loads(archive.read(prefix + "bundle/manifest.json"))
        assert manifest["repository"] == "fermga/TNFR-Python-Engine"
        assert (
            manifest["omitted"][0]["sha256"]
            == hashlib.sha256(b'{"answer":42}\r\n').hexdigest()
        )
        assert manifest["omitted"][0]["download_url"] == (
            f"https://raw.githubusercontent.com/fermga/TNFR-Python-Engine/{commit}/docs/assets/sample/response.json"
        )
        assert archive.read(prefix + "START_HERE.md").startswith(
            f"TNFR 1.2.3\n{commit}\n".encode()
        )
    assert Path(report["checksum"]).read_bytes() == (
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n".encode()
    )
    assert (root / "src/tnfr/example.py").read_bytes() == b"VALUE = -1\n"


def test_same_source_rebuild_is_identical_and_idempotent(source, tmp_path):
    root, _ = source
    first = bundle.build_bundle(tmp_path / "first", root=root)
    before = Path(first["archive"]).stat().st_mtime_ns
    repeated = bundle.build_bundle(tmp_path / "first", root=root)
    second = bundle.build_bundle(tmp_path / "second", root=root)
    assert first["sha256"] == repeated["sha256"] == second["sha256"]
    assert Path(first["archive"]).stat().st_mtime_ns == before


def test_new_version_uses_its_own_commit_and_keeps_older_tag(source, tmp_path):
    root, old_commit = source
    (root / "src/tnfr/_version.py").write_text('__version__ = "1.2.4"\n')
    (root / "pyproject.toml").write_text('[project]\nversion = "1.2.4"\n')
    new_commit = _commit(root)
    _git(root, "tag", "v1.2.4")
    new = bundle.build_bundle(tmp_path / "output", root=root, expected_tag="v1.2.4")
    old = bundle.build_bundle(
        tmp_path / "output", root=root, revision="v1.2.3", expected_tag="v1.2.3"
    )
    assert (new["version"], new["commit"]) == ("1.2.4", new_commit)
    assert (old["version"], old["commit"]) == ("1.2.3", old_commit)


@pytest.mark.parametrize("tag", ["v9.9.9", "1.2.3"])
def test_wrong_release_tag_creates_no_artifacts(source, tmp_path, tag):
    root, _ = source
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Release tag"):
        bundle.build_bundle(output, root=root, expected_tag=tag)
    assert not output.exists()


def test_tag_with_matching_version_but_different_commit_rejects(source, tmp_path):
    root, _ = source
    (root / "README.md").write_text("next commit")
    _commit(root)
    with pytest.raises(ValueError, match="Release tag"):
        bundle.build_bundle(tmp_path / "out", root=root, expected_tag="v1.2.3")


@pytest.mark.parametrize(
    "name,content,error",
    [
        (".github/agents/my-agent.md", b"drift", "agent instructions differ"),
        (
            "pyproject.toml",
            b'[project]\nversion = "9.9"\n',
            "version declarations differ",
        ),
        ("START_HERE.md", b"collision", "conflicts"),
        ("bundle/user.txt", b"collision", "conflicts"),
        ("start_here.md", b"collision", "conflicts"),
        ("Bundle/manifest.json", b"collision", "conflicts"),
        ("bundle", b"collision", "conflicts"),
    ],
)
def test_inconsistent_source_rejects_before_output(
    source, tmp_path, name, content, error
):
    root, _ = source
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    _commit(root)
    with pytest.raises(ValueError, match=error):
        bundle.build_bundle(tmp_path / "out", root=root)
    assert not (tmp_path / "out").exists()


def test_missing_required_owner_rejects(source, tmp_path):
    root, _ = source
    (root / "theory/EMERGENT_ONTOLOGY.md").unlink()
    _commit(root)
    with pytest.raises(ValueError, match="required owners"):
        bundle.build_bundle(tmp_path / "out", root=root)


@pytest.mark.parametrize("suffix", [".zip", ".zip.sha256"])
def test_existing_different_artifact_is_preserved(source, tmp_path, suffix):
    root, _ = source
    output = tmp_path / "out"
    output.mkdir()
    target = output / f"TNFR-1.2.3-light-development{suffix}"
    target.write_bytes(b"user content")
    with pytest.raises(ValueError, match="Existing artifact differs"):
        bundle.build_bundle(output, root=root)
    assert target.read_bytes() == b"user content"
    assert sorted(p.name for p in output.iterdir()) == [target.name]


def test_git_symlink_entry_rejected_without_following(source, tmp_path):
    root, _ = source
    blob = (
        subprocess.check_output(
            ["git", "-C", str(root), "hash-object", "-w", "--stdin"],
            input=b"../outside",
        )
        .decode()
        .strip()
    )
    _git(root, "update-index", "--add", "--cacheinfo", f"120000,{blob},linked")
    _git(root, "commit", "-qm", "Synthetic symlink")
    with pytest.raises(ValueError, match="regular tracked files"):
        bundle.build_bundle(tmp_path / "out", root=root)


def test_case_collision_rejected_on_any_host(source, tmp_path):
    root, _ = source
    blob = _git(root, "rev-parse", "HEAD:README.md")
    _git(root, "update-index", "--add", "--cacheinfo", f"100644,{blob},readme.md")
    _git(root, "commit", "-qm", "Synthetic case collision")
    with pytest.raises(ValueError, match="case-colliding"):
        bundle.build_bundle(tmp_path / "out", root=root)


def test_directory_case_aliases_rejected_on_any_host(source, tmp_path):
    root, _ = source
    blob = _git(root, "rev-parse", "HEAD:README.md")
    for name in ("Foo/a", "foo/b"):
        _git(root, "update-index", "--add", "--cacheinfo", f"100644,{blob},{name}")
    _git(root, "commit", "-qm", "Synthetic directory aliases")
    with pytest.raises(ValueError, match="Case-colliding"):
        bundle.build_bundle(tmp_path / "out", root=root)
    assert not (tmp_path / "out").exists()


def test_git_replace_cannot_change_reported_source(source, tmp_path):
    root, original = source
    (root / "src/tnfr/example.py").write_bytes(b"VALUE = -99\n")
    replacement = _commit(root)
    _git(root, "replace", original, replacement)
    report = bundle.build_bundle(
        tmp_path / "out", root=root, revision=original, expected_tag="v1.2.3"
    )
    assert report["commit"] == original
    with zipfile.ZipFile(report["archive"]) as archive:
        assert (
            archive.read("TNFR-1.2.3-light-development/src/tnfr/example.py")
            == b"VALUE = 42\r\n"
        )


@pytest.mark.parametrize(
    "pyproject",
    [
        b"[project]\nversion = \"1.2.3\"\nversion = '9.9'\n",
        b"[project]\ndescription = '''\nversion = \"1.2.3\"\n'''\n",
        b"[project]\nversion = true\n",
    ],
)
def test_false_toml_version_declarations_reject(source, tmp_path, pyproject):
    root, _ = source
    (root / "pyproject.toml").write_bytes(pyproject)
    _commit(root)
    with pytest.raises(ValueError):
        bundle.build_bundle(tmp_path / "out", root=root)
    assert not (tmp_path / "out").exists()


def test_symlink_sidecar_rejected_before_zip_publication(source, tmp_path, monkeypatch):
    root, _ = source
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path.name.endswith(".zip.sha256") or original(path),
    )
    with pytest.raises(ValueError, match="symlink"):
        bundle.build_bundle(tmp_path / "out", root=root)
    assert not list((tmp_path / "out").iterdir())


@pytest.mark.parametrize("response", ["wrong-header", "truncated", "wrong-hash"])
def test_git_batch_reader_rejects_invalid_bytes(response):
    expected = b"abc"
    blob = hashlib.sha1(b"blob 3\0" + expected).hexdigest()
    header = f"{blob} blob 3\n".encode()
    data = {
        "wrong-header": f"{blob} missing\n".encode(),
        "truncated": header + b"a",
        "wrong-hash": header + b"xyz\n",
    }[response]
    process = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(data))
    with pytest.raises(ValueError):
        bundle._read_blob(process, blob, 3)


def test_sourcebook_commit_replaces_legacy_required_owners(source, tmp_path):
    root, _ = source
    for name in (
        "theory/EMERGENT_ONTOLOGY.md",
        "theory/research/FIVE_STAGE_EXECUTION_PLAN.md",
    ):
        (root / name).unlink()
    write_book(
        root,
        Book(
            "# Sourcebook\n",
            (
                Chapter(
                    "theory/PROOF.md", "Proof", "Results", b"Retained proof: x = y.\r\n"
                ),
            ),
            (),
        ),
    )
    payload = (root / BOOK_PATH).read_bytes()
    (root / "scripts/research_book.py").write_text("# retained sourcebook reader\n")
    _commit(root)
    report = bundle.build_bundle(tmp_path / "sourcebook-output", root=root)
    with zipfile.ZipFile(report["archive"]) as archive:
        assert (
            archive.read("TNFR-1.2.3-light-development/theory/RESEARCH_PROGRAM.md")
            == payload
        )
        assert (
            "TNFR-1.2.3-light-development/theory/EMERGENT_ONTOLOGY.md"
            not in archive.namelist()
        )
    (root / "scripts/research_book.py").unlink()
    _commit(root)
    with pytest.raises(ValueError, match="research_book.py"):
        bundle.build_bundle(tmp_path / "missing-reader", root=root)
    assert not (tmp_path / "missing-reader").exists()


def _modular_source(root):
    for name in (
        "theory/EMERGENT_ONTOLOGY.md",
        "theory/research/FIVE_STAGE_EXECUTION_PLAN.md",
    ):
        (root / name).unlink()
    book = Book(
        "# Research program\n\nThe sole active question.\n",
        (
            Chapter(
                "theory/nodal/FOUNDATION.md",
                "Foundation",
                "Foundations",
                b"# Foundation\r\n\r\nPreserved proof.\r\n",
            ),
        ),
        (),
    )
    write_modular_book(root, book)
    (root / "scripts/research_book.py").write_text("# retained modular reader\n")
    return book


def test_modular_bundle_uses_committed_catalog_and_physical_chapter_bytes(
    source, tmp_path
):
    root, _ = source
    book = _modular_source(root)
    catalog = (root / CATALOG_PATH).read_bytes()
    commit = _commit(root)
    (root / CATALOG_PATH).write_bytes(b'{"version":false}\n')
    (root / book.chapters[0].path).write_bytes(b"Uncommitted replacement.\n")

    report = bundle.build_bundle(tmp_path / "modular-output", root=root)

    assert report["commit"] == commit
    prefix = "TNFR-1.2.3-light-development/"
    with zipfile.ZipFile(report["archive"]) as archive:
        assert archive.read(prefix + CATALOG_PATH) == catalog
        assert archive.read(prefix + book.chapters[0].path) == book.chapters[0].body
        assert len(archive.read(prefix + BOOK_PATH)) < 96 * 1024


@pytest.mark.parametrize("missing", ("chapter", "compatibility"))
def test_missing_committed_modular_owner_cannot_be_repaired_by_working_tree(
    source, tmp_path, missing
):
    root, _ = source
    book = _modular_source(root)
    if missing == "chapter":
        name = book.chapters[0].path
        (root / name).unlink()
    else:
        name = "theory/research/PLAN.md"
        book = replace(book, aliases=(Alias(name, "Retired plan"),))
        write_modular_book(root, book)
    _commit(root)
    (root / name).write_bytes(b"An uncommitted owner cannot repair the release.\n")

    with pytest.raises(ValueError, match="required owners.*" + Path(name).name):
        bundle.build_bundle(tmp_path / "incomplete-modular-output", root=root)

    assert not (tmp_path / "incomplete-modular-output").exists()


def test_modular_program_without_committed_catalog_rejects(source, tmp_path):
    root, _ = source
    _modular_source(root)
    (root / CATALOG_PATH).unlink()
    _commit(root)

    with pytest.raises(ValueError, match="research_catalog.json"):
        bundle.build_bundle(tmp_path / "missing-catalog-output", root=root)

    assert not (tmp_path / "missing-catalog-output").exists()


@pytest.mark.parametrize("corruption", ("traversal", "duplicate", "version"))
def test_invalid_committed_research_catalog_rejects_before_bundle_output(
    source, tmp_path, corruption
):
    root, _ = source
    _modular_source(root)
    catalog = json.loads((root / CATALOG_PATH).read_bytes())
    if corruption == "traversal":
        catalog["chapters"][0]["path"] = "theory/../README.md"
    elif corruption == "duplicate":
        catalog["chapters"].append(catalog["chapters"][0])
    else:
        catalog["version"] = True
    (root / CATALOG_PATH).write_text(json.dumps(catalog), encoding="utf-8")
    _commit(root)

    with pytest.raises(ValueError):
        bundle.build_bundle(tmp_path / "invalid-modular-output", root=root)

    assert not (tmp_path / "invalid-modular-output").exists()
