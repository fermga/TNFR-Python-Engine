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
