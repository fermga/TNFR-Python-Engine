"""Maintenance removes declared outputs without following directory redirects."""

from pathlib import Path

import pytest

from scripts import clean_repository, prepare_docs


def test_cleanup_removes_outputs_and_src_metadata_but_preserves_sources(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(clean_repository, "ROOT", tmp_path)
    for relative in ("build/docs-source", "build/lib", "dist", "src/tnfr.egg-info"):
        directory = tmp_path / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "generated.txt").write_text("generated", encoding="utf-8")
    source = tmp_path / "src" / "retained.py"
    source.write_text("# retained source\n", encoding="utf-8")
    frozen = tmp_path / "build" / "organized-receiver-enclosure-freeze"
    frozen.mkdir()
    evaluator = frozen / "evaluate_receiver.py"
    evaluator.write_bytes(b"# retained frozen evaluator\n")
    other = tmp_path / "build" / "local-preparation.txt"
    other.write_bytes(b"independent preparation\n")

    assert clean_repository.main() == 0

    assert not (tmp_path / "build" / "docs-source").exists()
    assert not (tmp_path / "build" / "lib").exists()
    assert not (tmp_path / "dist").exists()
    assert not (tmp_path / "src" / "tnfr.egg-info").exists()
    assert source.read_text(encoding="utf-8") == "# retained source\n"
    assert evaluator.read_bytes() == b"# retained frozen evaluator\n"
    assert other.read_bytes() == b"independent preparation\n"


def test_cleanup_validates_all_targets_before_removing_any(tmp_path, monkeypatch):
    monkeypatch.setattr(clean_repository, "ROOT", tmp_path)
    retained = tmp_path / "build" / "docs-source"
    retained.mkdir(parents=True)
    (tmp_path / "dist").write_text("unexpected file", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a directory"):
        clean_repository.main()
    assert retained.is_dir()


@pytest.mark.parametrize("redirect", ["src", "../outside"])
@pytest.mark.parametrize("operation", ["cleanup", "documentation"])
def test_redirected_outputs_reject_before_recursive_removal(
    tmp_path, monkeypatch, redirect, operation
):
    root = tmp_path / "workspace"
    root.mkdir()
    original_resolve = Path.resolve
    relative = "build/docs-source"
    declared = root / relative
    redirected = original_resolve(root / redirect)

    def resolve(path, *args, **kwargs):
        if path == declared:
            return redirected
        return original_resolve(path, *args, **kwargs)

    def forbidden_removal(*args, **kwargs):
        pytest.fail("unvalidated path reached recursive deletion")

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(clean_repository.shutil, "rmtree", forbidden_removal)
    if operation == "cleanup":
        monkeypatch.setattr(clean_repository, "ROOT", root)
        execute = clean_repository.main
    else:
        monkeypatch.setattr(prepare_docs, "REPO_ROOT", root)
        monkeypatch.setattr(prepare_docs, "STAGE_DIR", declared)
        execute = prepare_docs.prepare
    with pytest.raises(RuntimeError, match="redirected"):
        execute()


@pytest.mark.parametrize("relative", [".", "../outside"])
def test_repository_root_and_parent_are_never_generated_targets(tmp_path, relative):
    with pytest.raises(RuntimeError):
        clean_repository.generated_directory(tmp_path, relative)


def _write(root: Path, relatives: list[str]) -> None:
    for relative in relatives:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_cache_cleanup_is_limited_to_tool_caches_and_source_bytecode(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(clean_repository, "ROOT", tmp_path)
    removed = [
        ".mypy_cache/3.13/cache.json",
        ".pytest_cache/v/cache/nodeids",
        "__pycache__/conftest.cpython-313.pyc",
        "src/tnfr/__pycache__/core.cpython-313.pyc",
        "tests/unit/__pycache__/test_core.cpython-313-pytest.pyc",
        "dist/tnfr.whl",
    ]
    kept = [
        "src/tnfr/core.py",
        "build/two-port-freeze/__pycache__/frozen.cpython-313.pyc",
        "artifacts/run/__pycache__/record.cpython-313.pyc",
        ".venv/Lib/site-packages/__pycache__/package.cpython-313.pyc",
        "src/environment/pyvenv.cfg",
        "src/environment/__pycache__/site.cpython-313.pyc",
        "bibliografia/book.pdf",
    ]
    _write(tmp_path, removed + kept)

    assert clean_repository.main(["--caches"]) == 0

    assert not any((tmp_path / relative).exists() for relative in removed)
    assert all((tmp_path / relative).exists() for relative in kept)


def test_default_cleanup_retains_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(clean_repository, "ROOT", tmp_path)
    cache = "src/tnfr/__pycache__/core.cpython-313.pyc"
    _write(tmp_path, [cache, ".mypy_cache/state.json", "site/index.html"])

    assert clean_repository.main() == 0

    assert (tmp_path / cache).exists()
    assert (tmp_path / ".mypy_cache/state.json").exists()
    assert not (tmp_path / "site").exists()


def test_dry_run_and_report_remove_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(clean_repository, "ROOT", tmp_path)
    created = [
        "dist/tnfr.whl",
        ".mypy_cache/state.json",
        "src/tnfr/__pycache__/core.cpython-313.pyc",
        "artifacts/release-env/pyvenv.cfg",
        "artifacts/release-env/Lib/site.py",
        "artifacts/run/record.json",
    ]
    _write(tmp_path, created)

    assert clean_repository.main(["--dry-run", "--caches"]) == 0
    listed = capsys.readouterr().out
    assert "would remove dist (" in listed
    assert "would remove .mypy_cache (" in listed
    assert "would remove 1 __pycache__ directories" in listed

    assert clean_repository.main(["--report"]) == 0
    report = capsys.readouterr().out
    assert "artifacts/release-env" in report
    assert "Removable by `python scripts/clean_repository.py`" in report

    assert all((tmp_path / relative).exists() for relative in created)


def test_cache_cleanup_rejects_redirected_bytecode_before_any_removal(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(clean_repository, "ROOT", tmp_path)
    _write(tmp_path, ["dist/tnfr.whl", "src/tnfr/__pycache__/core.cpython-313.pyc"])
    declared = tmp_path / "src/tnfr/__pycache__"
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == declared:
            return original_resolve(tmp_path / "outside")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(RuntimeError, match="redirected"):
        clean_repository.main(["--caches"])
    assert (tmp_path / "dist/tnfr.whl").exists()
