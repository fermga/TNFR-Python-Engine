"""Documentation corruption must fail even when Python assertions are disabled.

Only declared temporary documents are modified. The child processes import the
real checker and contract registry, then point its filesystem reads at a sandbox.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[2] / "scripts" / "check_documentation.py"
CHILD_CHECK = """
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("documentation_under_test", sys.argv[1])
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.REPO_ROOT = Path(sys.argv[2])
getattr(checker, sys.argv[3])()
"""


@pytest.fixture
def documented_workspace(tmp_path):
    spec = importlib.util.spec_from_file_location("documentation_fixture", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    contracts = tmp_path / "docs" / "API_CONTRACTS.md"
    contracts.parent.mkdir()
    contracts.write_text(
        "# Declared test contracts\n\n"
        + checker.CONTRACT_START
        + "\n\n"
        + checker.render_contract_table()
        + "\n\n"
        + checker.CONTRACT_END
        + "\n\nUnrelated explanatory prose.\n",
        encoding="utf-8",
    )
    canonical = b"# Declared canonical test instructions\n"
    (tmp_path / "AGENTS.md").write_bytes(canonical)
    mirror = tmp_path / ".github" / "agents" / "my-agent.md"
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(canonical)
    return tmp_path, checker


def _run_check(workspace, check, *, optimized=True):
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.extend(["-c", CHILD_CHECK, str(CHECKER), str(workspace), check])
    return subprocess.run(
        command,
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


@pytest.fixture
def publication_workspace(tmp_path):
    """Supply independent metadata, without reading the release under preparation."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "9.8.7"\n', encoding="utf-8"
    )
    (tmp_path / ".zenodo.json").write_text(
        '{"version": "9.8.7", "publication_date": "2026-01-02"}\n',
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        'version: "9.8.7"\ndate-released: "2026-01-02"\n'
        'url: "https://github.com/fermga/TNFR-Python-Engine/releases/tag/v9.8.7"\n',
        encoding="utf-8",
    )
    fallback = tmp_path / "src" / "tnfr" / "_version.py"
    fallback.parent.mkdir(parents=True)
    fallback.write_text('__version__ = "9.8.7"\n', encoding="utf-8")
    return tmp_path


def test_consistent_publication_metadata_passes_with_optimization(
    publication_workspace,
):
    result = _run_check(publication_workspace, "check_publication_metadata")

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("relative", "expected"),
    (
        (".zenodo.json", "Zenodo version differs from project"),
        ("CITATION.cff", "citation version differs from publication metadata"),
    ),
)
def test_publication_version_drift_fails_with_optimization(
    publication_workspace, relative, expected
):
    path = publication_workspace / relative
    corrupted = path.read_text(encoding="utf-8").replace("9.8.7", "9.8.6", 1)
    path.write_text(corrupted, encoding="utf-8")

    result = _run_check(publication_workspace, "check_publication_metadata")

    assert result.returncode != 0
    assert expected in result.stderr
    assert path.read_text(encoding="utf-8") == corrupted


@pytest.mark.parametrize("fallback", ('__version__ = "9.8.6"\n', "# Missing version\n"))
def test_missing_or_stale_source_fallback_fails_with_optimization(
    publication_workspace, fallback
):
    path = publication_workspace / "src" / "tnfr" / "_version.py"
    path.write_text(fallback, encoding="utf-8")

    result = _run_check(publication_workspace, "check_publication_metadata")

    assert result.returncode != 0
    assert "source fallback version differs from project" in result.stderr
    assert path.read_text(encoding="utf-8") == fallback


@pytest.mark.parametrize("check", ("check_agent_mirror", "check_contract_view"))
def test_valid_temporary_documents_pass_with_optimization(documented_workspace, check):
    workspace, _ = documented_workspace

    result = _run_check(workspace, check)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("optimized", (False, True))
def test_changed_postcondition_fails_with_operator_identity_preserved(
    documented_workspace, optimized
):
    workspace, _ = documented_workspace
    path = workspace / "docs" / "API_CONTRACTS.md"
    text = path.read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| Emission |"))
    identity, _ = row.rsplit(" | ", 1)
    corrupt_row = identity + " | Arbitrary deletion of every graph node is permitted. |"
    path.write_text(text.replace(row, corrupt_row, 1), encoding="utf-8")

    result = _run_check(workspace, "check_contract_view", optimized=optimized)

    assert corrupt_row.rsplit(" | ", 1)[0] == identity
    assert result.returncode != 0
    assert "operator contract view drifted" in result.stderr
    assert path.read_text(encoding="utf-8") == text.replace(row, corrupt_row, 1)


@pytest.mark.parametrize("check", ("check_contract_view", "update_contract_view"))
@pytest.mark.parametrize("corruption", ("duplicate_start", "duplicate_end", "reversed"))
def test_ambiguous_or_reversed_markers_fail_before_writing(
    documented_workspace, check, corruption
):
    workspace, checker = documented_workspace
    path = workspace / "docs" / "API_CONTRACTS.md"
    text = path.read_text(encoding="utf-8")
    start, end = checker.CONTRACT_START, checker.CONTRACT_END
    if corruption == "duplicate_start":
        text += "\n" + start
    elif corruption == "duplicate_end":
        text += "\n" + end
    else:
        text = text.replace(start, "PLACEHOLDER", 1).replace(end, start, 1)
        text = text.replace("PLACEHOLDER", end, 1)
    path.write_text(text, encoding="utf-8")

    result = _run_check(workspace, check)

    assert result.returncode != 0
    expected = "markers are reversed" if corruption == "reversed" else "exactly one"
    assert expected in result.stderr
    assert path.read_text(encoding="utf-8") == text


def test_agent_mirror_drift_fails_with_optimization(documented_workspace):
    workspace, _ = documented_workspace
    canonical = (workspace / "AGENTS.md").read_bytes()
    mirror = workspace / ".github" / "agents" / "my-agent.md"
    mirror.write_bytes(canonical + b"Contradictory mirror-only instruction.\n")

    result = _run_check(workspace, "check_agent_mirror")

    assert result.returncode != 0
    assert "canonical agent mirror has drifted" in result.stderr
    assert (workspace / "AGENTS.md").read_bytes() == canonical


@pytest.fixture
def reference_workspace(tmp_path, monkeypatch):
    path = CHECKER.with_name("verify_internal_references.py")
    spec = importlib.util.spec_from_file_location("reference_checker_fixture", path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    manual = tmp_path / "manual"
    manual.mkdir()
    return tmp_path, manual, checker


def test_manual_links_include_own_repository_urls_and_reference_style(
    reference_workspace,
):
    workspace, manual, checker = reference_workspace
    theory = workspace / "theory"
    theory.mkdir()
    (theory / "Pressure Form.md").write_text(
        "# Pressure and form\n\n## Capacity\n", encoding="utf-8"
    )
    (manual / "guide.md").write_text(
        "# User manual\n\n"
        "[Local chapter](../theory/Pressure%20Form.md#pressure-and-form)\n\n"
        "[Repository chapter](https://github.com/fermga/TNFR-Python-Engine/"
        "blob/main/theory/Pressure%20Form.md#capacity)\n\n"
        "[Capacity details][capacity]\n\n"
        '[capacity]: ../theory/Pressure%20Form.md#capacity "Capacity definition"\n',
        encoding="utf-8",
    )

    references, failures = checker.verify(["."])

    assert references == 3
    assert failures == []


def test_fenced_markdown_examples_do_not_create_live_references(reference_workspace):
    _, manual, checker = reference_workspace
    (manual / "target.md").write_text("# Real target\n", encoding="utf-8")
    (manual / "guide.md").write_text(
        "# Guide\n\n[Real](target.md#real-target)\n\n"
        "````markdown\n[Not a link](missing-backtick.md)\n"
        "```\n[not-a-reference]: missing-reference.md\n````\n\n"
        "~~~markdown\n[Not a link](missing-tilde.md#missing)\n~~~\n\n"
        "`[Inline example](missing-inline.md)`\n",
        encoding="utf-8",
    )

    references, failures = checker.verify(["."])

    assert references == 1
    assert failures == []


@pytest.mark.parametrize("fragment", ("real-topic", "absent"))
def test_directory_fragments_follow_the_index_used_by_site_staging(
    reference_workspace, fragment
):
    workspace, manual, checker = reference_workspace
    docs = workspace / "docs"
    docs.mkdir()
    (docs / "README.md").write_text("# Real topic\n", encoding="utf-8")
    (manual / "guide.md").write_text(
        f"[Directory index](../docs/#{fragment})\n", encoding="utf-8"
    )

    references, failures = checker.verify(["."])

    assert references == 1
    assert bool(failures) == (fragment == "absent")
    if failures:
        assert failures[0].startswith("missing fragment:")


def test_staging_preserves_inline_and_reference_link_destinations(
    tmp_path, monkeypatch
):
    path = CHECKER.with_name("prepare_docs.py")
    spec = importlib.util.spec_from_file_location("staging_under_test", path)
    staging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staging)
    monkeypatch.setattr(staging, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(staging, "STAGE_DIR", tmp_path / "build" / "docs-source")
    for relative in staging.ROOT_FILES:
        (tmp_path / relative).write_text("placeholder\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Home\n", encoding="utf-8")
    manual = tmp_path / "manual"
    manual.mkdir()
    original = (
        "[Inline](../README.md#home)\n[Reference][home]\n"
        '[home]: <../README.md#home> "Home title"\n'
        "[Subdirectory](../docs/)\n[Source folder](../src/tnfr/)\n"
    )
    (manual / "guide.md").write_text(original, encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text("# Documentation\n", encoding="utf-8")
    (tmp_path / "src" / "tnfr").mkdir(parents=True)

    output = staging.prepare()

    rendered = (output / "manual" / "guide.md").read_text(encoding="utf-8")
    assert "[Inline](../index.md#home)" in rendered
    assert '[home]: <../index.md#home> "Home title"' in rendered
    assert "[Subdirectory](../docs/README.md)" in rendered
    assert "https://github.com/fermga/TNFR-Python-Engine/tree/main/src/tnfr" in rendered
    assert (output / "index.md").is_file()
    assert not (output / "README.md").exists()
    assert (manual / "guide.md").read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "link, expected_failure",
    (
        ("[Missing file](absent.md)", "missing target"),
        (
            "[Missing heading](https://github.com/fermga/TNFR-Python-Engine/"
            "blob/main/manual/target.md#absent)",
            "missing fragment",
        ),
        (
            "[Code heading][code]\n\n[code]: target.md#only-in-code",
            "missing fragment",
        ),
    ),
)
def test_live_missing_targets_and_fragments_are_rejected(
    reference_workspace, link, expected_failure
):
    _, manual, checker = reference_workspace
    (manual / "target.md").write_text(
        "# Real heading\n\n```markdown\n## Only in code\n```\n",
        encoding="utf-8",
    )
    (manual / "guide.md").write_text("# Guide\n\n" + link + "\n", encoding="utf-8")

    references, failures = checker.verify(["."])

    assert references == 1
    assert len(failures) == 1
    assert failures[0].startswith(expected_failure + ":")
    assert "guide.md" in failures[0]
