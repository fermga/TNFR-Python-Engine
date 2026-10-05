"""Documentation corruption must fail even when Python assertions are disabled.

Only declared temporary documents are modified. The child processes import the
real checker and contract registry, then point its filesystem reads at a sandbox.
"""

from __future__ import annotations

import importlib.util
import json
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


def test_documentation_uses_checkout_when_source_path_already_follows_installed_package(
    tmp_path,
):
    stale = tmp_path / "installed" / "tnfr"
    stale.mkdir(parents=True)
    (stale / "__init__.py").write_text(
        'raise RuntimeError("obsolete installed TNFR imported")\n', encoding="utf-8"
    )
    child = """
import importlib.util
import sys
from pathlib import Path

checker_path = Path(sys.argv[1])
source = checker_path.parent.parent / "src"
sys.path[:0] = [sys.argv[2], str(source)]
spec = importlib.util.spec_from_file_location("checkout_checker", checker_path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.render_contract_table()
import tnfr
if Path(tnfr.__file__).resolve() != (source / "tnfr" / "__init__.py").resolve():
    raise RuntimeError("documentation did not use the requested checkout")
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(CHECKER), str(stale.parent)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
        json.dumps(
            {
                "version": "9.8.7",
                "publication_date": "2026-01-02",
                "description": (
                    '<p><a href="https://github.com/fermga/TNFR-Python-Engine/'
                    'releases/tag/v9.8.7">Release</a> '
                    '<a href="https://github.com/fermga/TNFR-Python-Engine/'
                    'tree/v9.8.7">Source</a> '
                    '<a href="https://pypi.org/project/tnfr/9.8.7/">Package</a></p>'
                ),
                "related_identifiers": [
                    {"identifier": "https://pypi.org/project/tnfr/9.8.7/"}
                ],
            }
        )
        + "\n",
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


def test_publication_links_accept_single_quoted_html_attributes(publication_workspace):
    path = publication_workspace / ".zenodo.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["description"] = metadata["description"].replace('"', "'")
    path.write_text(json.dumps(metadata), encoding="utf-8")

    result = _run_check(publication_workspace, "check_publication_metadata")

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("corruption", ("overescaped", "text", "comment", "duplicate"))
def test_zenodo_requires_real_unambiguous_html_links_with_optimization(
    publication_workspace, corruption
):
    path = publication_workspace / ".zenodo.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    description = metadata["description"]
    if corruption == "overescaped":
        description = description.replace('"', '\\"')
    elif corruption == "text":
        description = description.replace("<a href=", "<span data-href=")
        description = description.replace("</a>", "</span>")
    elif corruption == "comment":
        description = "<!-- " + description + " -->"
    else:
        description = description.replace("<a href=", '<a href="wrong" href=', 1)
    metadata["description"] = description
    corrupted = json.dumps(metadata)
    path.write_text(corrupted, encoding="utf-8")

    result = _run_check(publication_workspace, "check_publication_metadata")

    assert result.returncode != 0
    assert "Zenodo description" in result.stderr
    assert path.read_text(encoding="utf-8") == corrupted


@pytest.mark.parametrize(
    "stale_target",
    (
        "https://github.com/fermga/TNFR-Python-Engine/releases/tag/v9.8.6",
        "https://github.com/fermga/TNFR-Python-Engine/tree/v9.8.6",
        "https://github.com/fermga/TNFR-Python-Engine/blob/v9.8.6/theory/README.md",
        "https://pypi.org/project/tnfr/9.8.6/",
    ),
)
def test_zenodo_rejects_stale_links_even_beside_current_links(
    publication_workspace, stale_target
):
    path = publication_workspace / ".zenodo.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["description"] += f'<a href="{stale_target}">Stale publication</a>'
    corrupted = json.dumps(metadata)
    path.write_text(corrupted, encoding="utf-8")

    result = _run_check(publication_workspace, "check_publication_metadata")

    assert result.returncode != 0
    assert "description links to a different publication version" in result.stderr
    assert path.read_text(encoding="utf-8") == corrupted


@pytest.mark.parametrize(
    "identifiers", ([], [{"identifier": "https://pypi.org/project/tnfr/9.8.6/"}])
)
def test_zenodo_related_package_must_match_version(publication_workspace, identifiers):
    path = publication_workspace / ".zenodo.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["related_identifiers"] = identifiers
    corrupted = json.dumps(metadata)
    path.write_text(corrupted, encoding="utf-8")

    result = _run_check(publication_workspace, "check_publication_metadata")

    assert result.returncode != 0
    assert "related PyPI identifier differs from project version" in result.stderr
    assert path.read_text(encoding="utf-8") == corrupted


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
    examples = tmp_path / "examples"
    examples.mkdir()
    return tmp_path, examples, checker


def test_documentation_links_include_own_repository_urls_and_reference_style(
    reference_workspace,
):
    workspace, examples, checker = reference_workspace
    theory = workspace / "theory"
    theory.mkdir()
    (theory / "Pressure Form.md").write_text(
        "# Pressure and form\n\n## Capacity\n", encoding="utf-8"
    )
    (examples / "guide.md").write_text(
        "# User guide\n\n"
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
    _, examples, checker = reference_workspace
    (examples / "target.md").write_text("# Real target\n", encoding="utf-8")
    (examples / "guide.md").write_text(
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


@pytest.mark.parametrize(
    "original, expected",
    [
        (
            "Before `multi\n[example](docs/)` after\n",
            "Before `multi\n[example](docs/)` after\n",
        ),
        (r"Escaped \` [real](docs/) \`", r"Escaped \` [real](docs/README.md) \`"),
        (
            "``[example](docs/) `inner` `` [live](docs/)",
            "``[example](docs/) `inner` `` [live](docs/README.md)",
        ),
        (
            "`[example](docs/)\\` [live](docs/)",
            "`[example](docs/)\\` [live](docs/README.md)",
        ),
    ],
)
def test_prose_rewriting_preserves_code_span_boundaries(original, expected):
    from scripts.verify_internal_references import rewrite_markdown_prose

    assert (
        rewrite_markdown_prose(
            original, lambda text: text.replace("(docs/)", "(docs/README.md)")
        )
        == expected
    )


def test_unmatched_backticks_cannot_hide_links_in_another_paragraph(
    reference_workspace,
):
    _, examples, checker = reference_workspace
    original = "Unmatched `\n\n[real](absent.md)\n\nAnother `\n"
    (examples / "guide.md").write_text(original, encoding="utf-8")

    rewritten = checker.rewrite_markdown_prose(
        original, lambda text: text.replace("absent.md", "replacement.md")
    )
    references, failures = checker.verify(["."])

    assert "[real](replacement.md)" in rewritten
    assert references == 1
    assert len(failures) == 1
    assert "absent.md" in failures[0]


@pytest.mark.parametrize("fragment", ("real-topic", "absent"))
def test_directory_fragments_follow_the_index_used_by_site_staging(
    reference_workspace, fragment
):
    workspace, examples, checker = reference_workspace
    docs = workspace / "docs"
    docs.mkdir()
    (docs / "README.md").write_text("# Real topic\n", encoding="utf-8")
    (examples / "guide.md").write_text(
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
    examples = tmp_path / "examples"
    examples.mkdir()
    original = (
        "[Inline](../README.md#home)\n[Reference][home]\n"
        '[home]: <../README.md#home> "Home title"\n'
        "[Subdirectory](../docs/)\n[Source folder](../src/tnfr/)\n"
        "`[Inline example](../README.md#home)`\n"
        "````markdown\n[Code example](../README.md#home)\n"
        "```\n[example]: ../README.md#home\n````\n"
        "~~~markdown\n[Other example](../docs/)\n~~~\n"
    )
    (examples / "guide.md").write_text(original, encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text("# Documentation\n", encoding="utf-8")
    (tmp_path / "src" / "tnfr").mkdir(parents=True)

    output = staging.prepare()

    rendered = (output / "examples" / "guide.md").read_text(encoding="utf-8")
    assert "[Inline](../index.md#home)" in rendered
    assert '[home]: <../index.md#home> "Home title"' in rendered
    assert "[Subdirectory](../docs/README.md)" in rendered
    assert "https://github.com/fermga/TNFR-Python-Engine/tree/main/src/tnfr" in rendered
    assert "`[Inline example](../README.md#home)`" in rendered
    assert (
        "````markdown\n[Code example](../README.md#home)\n"
        "```\n[example]: ../README.md#home\n````\n"
    ) in rendered
    assert "~~~markdown\n[Other example](../docs/)\n~~~\n" in rendered
    assert (output / "index.md").is_file()
    assert not (output / "README.md").exists()
    assert (examples / "guide.md").read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "link, expected_failure",
    (
        ("[Missing file](absent.md)", "missing target"),
        (
            "[Missing heading](https://github.com/fermga/TNFR-Python-Engine/"
            "blob/main/examples/target.md#absent)",
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
    _, examples, checker = reference_workspace
    (examples / "target.md").write_text(
        "# Real heading\n\n```markdown\n## Only in code\n```\n",
        encoding="utf-8",
    )
    (examples / "guide.md").write_text("# Guide\n\n" + link + "\n", encoding="utf-8")

    references, failures = checker.verify(["."])

    assert references == 1
    assert len(failures) == 1
    assert failures[0].startswith(expected_failure + ":")
    assert "guide.md" in failures[0]


def _populate_catalog(workspace, checker, directory):
    folder = workspace / directory
    excluded = folder / ("research/archive" if directory == "theory" else "assets")
    (folder / "nodal").mkdir(parents=True)
    excluded.mkdir(parents=True)
    (folder / "FOUNDATION.md").write_text("# Foundation\n", encoding="utf-8")
    (folder / "nodal" / "CLOSURE.md").write_text("# Closure\n", encoding="utf-8")
    (excluded / "HISTORICAL.md").write_text("# Historical result\n", encoding="utf-8")
    start = getattr(checker, directory.upper() + "_CATALOG_START")
    end = getattr(checker, directory.upper() + "_CATALOG_END")
    contract_row = (
        "| [Contracts](API_CONTRACTS.md) | Registry |\n" if directory == "docs" else ""
    )
    (folder / "README.md").write_text(
        "# Owner index\n\n"
        "[Quick foundation link](FOUNDATION.md)\n" + start + "\n\n## Foundations\n\n"
        "| Document | Scope |\n| --- | --- |\n"
        "| [Foundation](FOUNDATION.md#foundation) | Definition |\n"
        "| [Closure](nodal/CLOSURE.md) | Conditional theorem |\n\n"
        + contract_row
        + end
        + f"\n\n[Excluded owner]({excluded.relative_to(folder).as_posix()}/HISTORICAL.md)\n",
        encoding="utf-8",
    )


@pytest.fixture
def theory_catalog_workspace(documented_workspace, monkeypatch):
    workspace, checker = documented_workspace
    monkeypatch.setattr(checker, "REPO_ROOT", workspace)
    _populate_catalog(workspace, checker, "theory")
    return workspace, checker


@pytest.fixture
def docs_catalog_workspace(documented_workspace, monkeypatch):
    workspace, checker = documented_workspace
    monkeypatch.setattr(checker, "REPO_ROOT", workspace)
    _populate_catalog(workspace, checker, "docs")
    return workspace, checker


def test_theory_catalog_covers_nested_owners_without_counting_quick_links_or_archive(
    theory_catalog_workspace,
):
    workspace, checker = theory_catalog_workspace
    before = (workspace / "theory" / "README.md").read_bytes()
    checker.check_theory_catalog()
    assert (workspace / "theory" / "README.md").read_bytes() == before


@pytest.mark.parametrize(
    "link, expected",
    (
        ("[Repeated foundation](./FOUNDATION.md#another-fragment)", "duplicate"),
        ("[Deleted owner](nodal/DELETED.md)", "unknown or archived"),
        ("[Historical](research/archive/HISTORICAL.md)", "unknown or archived"),
        ("[Technical contract](../docs/API_CONTRACTS.md)", "unknown or archived"),
    ),
)
def test_theory_catalog_rejects_duplicate_stale_archive_and_nonowner_entries(
    theory_catalog_workspace, link, expected
):
    workspace, checker = theory_catalog_workspace
    index = workspace / "theory" / "README.md"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            checker.THEORY_CATALOG_END,
            "| " + link + " | Extra primary entry |\n" + checker.THEORY_CATALOG_END,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=expected + " primary theory documents"):
        checker.check_theory_catalog()


def test_theory_catalog_requires_a_primary_entry_despite_a_quick_link(
    theory_catalog_workspace,
):
    workspace, checker = theory_catalog_workspace
    index = workspace / "theory" / "README.md"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            "| [Foundation](FOUNDATION.md#foundation) | Definition |\n", ""
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError, match="missing primary theory documents.*FOUNDATION"
    ):
        checker.check_theory_catalog()


def test_new_theory_owner_cannot_be_lost_even_with_python_optimization(
    theory_catalog_workspace,
):
    workspace, _ = theory_catalog_workspace
    (workspace / "theory" / "nodal" / "NEW_RESULT.md").write_text(
        "# New conditional result\n", encoding="utf-8"
    )
    result = _run_check(workspace, "check_theory_catalog", optimized=True)
    assert result.returncode != 0
    assert "missing primary theory documents" in result.stderr
    assert "NEW_RESULT.md" in result.stderr


def test_theory_catalog_ignores_fenced_illustrations_and_external_links(
    theory_catalog_workspace,
):
    workspace, checker = theory_catalog_workspace
    index = workspace / "theory" / "README.md"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            checker.THEORY_CATALOG_END,
            "```markdown\n[Not a primary entry](ABSENT.md)\n```\n"
            "[External example](https://example.org/OTHER.md)\n"
            + checker.THEORY_CATALOG_END,
        ),
        encoding="utf-8",
    )
    checker.check_theory_catalog()


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "reversed"))
def test_theory_catalog_requires_one_ordered_region(
    theory_catalog_workspace, corruption
):
    workspace, checker = theory_catalog_workspace
    index = workspace / "theory" / "README.md"
    document = index.read_text(encoding="utf-8")
    if corruption == "missing":
        document = document.replace(checker.THEORY_CATALOG_END, "")
    elif corruption == "duplicate":
        document += checker.THEORY_CATALOG_START
    else:
        document = document.replace(checker.THEORY_CATALOG_START, "temporary-start")
        document = document.replace(
            checker.THEORY_CATALOG_END, checker.THEORY_CATALOG_START
        )
        document = document.replace("temporary-start", checker.THEORY_CATALOG_END)
    index.write_text(document, encoding="utf-8")
    with pytest.raises(RuntimeError, match="catalog region|markers are reversed"):
        checker.check_theory_catalog()


def _populate_navigation(workspace, checker, directory):
    start = getattr(checker, directory.upper() + "_NAVIGATION_START")
    end = getattr(checker, directory.upper() + "_NAVIGATION_END")
    (workspace / "mkdocs.yml").write_text(
        "custom: !!python/name:example.callback\nnav:\n"
        "  - Home: index.md\n"
        + start
        + "\n  - Obsolete tree: obsolete.md\n"
        + end
        + "\n      - Historical archive: theory/research/archive/README.md\n"
        "  - Examples: examples/README.md\n",
        encoding="utf-8",
    )


@pytest.fixture
def theory_navigation_workspace(theory_catalog_workspace):
    workspace, checker = theory_catalog_workspace
    _populate_navigation(workspace, checker, "theory")
    return workspace, checker


@pytest.fixture
def docs_navigation_workspace(docs_catalog_workspace):
    workspace, checker = docs_catalog_workspace
    _populate_navigation(workspace, checker, "docs")
    return workspace, checker


@pytest.fixture(params=["theory", "docs"])
def catalog_navigation_workspace(request):
    workspace, checker = request.getfixturevalue(
        request.param + "_navigation_workspace"
    )
    return workspace, checker, request.param


def test_catalog_navigation_generation_is_idempotent_and_preserves_ancillary_yaml(
    catalog_navigation_workspace,
):
    workspace, checker, directory = catalog_navigation_workspace
    path = workspace / "mkdocs.yml"
    index = workspace / directory / "README.md"
    original_index = index.read_bytes()
    before = path.read_text(encoding="utf-8")
    start, end = getattr(checker, f"{directory}_navigation_region")(before)
    getattr(checker, f"update_{directory}_navigation")()
    generated = path.read_text(encoding="utf-8")
    assert generated.startswith(before[:start])
    assert generated.endswith(before[end:])
    assert '      - "Foundations":\n' in generated
    assert f'          - "Foundation": {directory}/FOUNDATION.md\n' in generated
    assert f'          - "Closure": {directory}/nodal/CLOSURE.md\n' in generated
    assert "#foundation" not in generated
    assert "Quick foundation" not in generated
    getattr(checker, f"check_{directory}_navigation")()
    getattr(checker, f"update_{directory}_navigation")()
    assert path.read_text(encoding="utf-8") == generated
    assert index.read_bytes() == original_index


@pytest.mark.parametrize(
    "old, new, expected_label",
    (
        ("## Foundations", '## State: "definitions"', 'State: "definitions"'),
        (
            "[Foundation](FOUNDATION.md#foundation)",
            '[Form: "EPI"](FOUNDATION.md#foundation)',
            'Form: "EPI"',
        ),
    ),
)
def test_catalog_title_or_group_change_rejects_stale_nav_and_generates_safe_yaml_labels(
    catalog_navigation_workspace, old, new, expected_label
):
    import json

    workspace, checker, directory = catalog_navigation_workspace
    getattr(checker, f"update_{directory}_navigation")()
    index = workspace / directory / "README.md"
    index.write_text(
        index.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )
    getattr(checker, f"check_{directory}_catalog")()
    with pytest.raises(RuntimeError, match=directory + " navigation drifted"):
        getattr(checker, f"check_{directory}_navigation")()
    getattr(checker, f"update_{directory}_navigation")()
    getattr(checker, f"check_{directory}_navigation")()
    label = json.dumps(expected_label, ensure_ascii=False)
    assert "- " + label + ":" in (workspace / "mkdocs.yml").read_text(encoding="utf-8")


def test_catalog_and_navigation_share_only_the_first_link_of_primary_rows(
    catalog_navigation_workspace,
):
    workspace, checker, directory = catalog_navigation_workspace
    index = workspace / directory / "README.md"
    index.write_text(
        index.read_text(encoding="utf-8")
        .replace("| Definition |", "| See [closure](nodal/CLOSURE.md) |")
        .replace(
            getattr(checker, directory.upper() + "_CATALOG_END"),
            "| No primary link | [Context only](ABSENT.md) |\n"
            + getattr(checker, directory.upper() + "_CATALOG_END"),
        )
        + "\n## Unrelated reading route\n| [Extra title](FOUNDATION.md) | Context |\n",
        encoding="utf-8",
    )
    getattr(checker, f"check_{directory}_catalog")()
    rendered = getattr(checker, f"render_{directory}_navigation")()
    assert rendered.count(f"{directory}/FOUNDATION.md") == 1
    assert rendered.count(f"{directory}/nodal/CLOSURE.md") == 1
    assert "Extra title" not in rendered
    assert "Unrelated reading route" not in rendered


@pytest.mark.parametrize("corruption", ("missing", "indented", "reversed"))
def test_generated_navigation_requires_ordered_column_zero_markers(
    catalog_navigation_workspace, corruption
):
    workspace, checker, directory = catalog_navigation_workspace
    path = workspace / "mkdocs.yml"
    document = path.read_text(encoding="utf-8")
    if corruption == "missing":
        document = document.replace(
            getattr(checker, directory.upper() + "_NAVIGATION_START"), ""
        )
    elif corruption == "indented":
        document = document.replace(
            getattr(checker, directory.upper() + "_NAVIGATION_START"),
            "  " + getattr(checker, directory.upper() + "_NAVIGATION_START"),
        )
    else:
        document = document.replace(
            getattr(checker, directory.upper() + "_NAVIGATION_START"), "temporary-start"
        )
        document = document.replace(
            getattr(checker, directory.upper() + "_NAVIGATION_END"),
            getattr(checker, directory.upper() + "_NAVIGATION_START"),
        )
        document = document.replace(
            "temporary-start", getattr(checker, directory.upper() + "_NAVIGATION_END")
        )
    path.write_text(document, encoding="utf-8")
    with pytest.raises(RuntimeError, match="column zero|markers are reversed"):
        getattr(checker, f"update_{directory}_navigation")()
    assert path.read_text(encoding="utf-8") == document


def test_write_generated_updates_both_registry_contracts_and_catalog_navigation(
    theory_navigation_workspace, monkeypatch, capsys
):
    workspace, checker = theory_navigation_workspace
    _populate_catalog(workspace, checker, "docs")
    navigation = workspace / "mkdocs.yml"
    navigation.write_text(
        navigation.read_text(encoding="utf-8")
        + checker.DOCS_NAVIGATION_START
        + "\n  - Old docs: obsolete.md\n"
        + checker.DOCS_NAVIGATION_END
        + "\n",
        encoding="utf-8",
    )
    contracts = workspace / "docs" / "API_CONTRACTS.md"
    document = contracts.read_text(encoding="utf-8")
    start, end = checker.contract_region(document)
    contracts.write_text(
        document[:start] + "\nobsolete\n" + document[end:], encoding="utf-8"
    )
    for check in (
        "check_versions_and_retired_claims",
        "check_publication_metadata",
        "check_documented_examples",
        "check_build_inputs",
        "check_glossary",
        "update_glossary_index",
    ):
        monkeypatch.setattr(checker, check, lambda: None)
    monkeypatch.setattr(sys, "argv", [str(CHECKER), "--write-generated"])
    assert checker.main() == 0
    checker.check_contract_view()
    checker.check_theory_navigation()
    checker.check_docs_catalog()
    checker.check_docs_navigation()
    assert "Documentation integrity checks passed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "link, expected",
    (
        ("[Repeated](./FOUNDATION.md#foundation)", "duplicate"),
        ("[Stale](contracts/REMOVED.md)", "unknown or excluded"),
        ("[Asset](assets/HISTORICAL.md)", "unknown or excluded"),
        ("[Outside](../AGENTS.md)", "unknown or excluded"),
    ),
)
def test_docs_catalog_rejects_nonowners_and_duplicate_rows(
    docs_catalog_workspace, link, expected
):
    workspace, checker = docs_catalog_workspace
    index = workspace / "docs" / "README.md"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            checker.DOCS_CATALOG_END,
            f"| {link} | Extra row |\n" + checker.DOCS_CATALOG_END,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=expected + " primary docs documents"):
        checker.check_docs_catalog()


def test_new_docs_owner_requires_a_primary_catalog_entry_under_optimization(
    docs_catalog_workspace,
):
    workspace, _ = docs_catalog_workspace
    chapter = workspace / "docs" / "contracts" / "NEW_CONTRACT.md"
    chapter.parent.mkdir()
    chapter.write_text("# New admission contract\n", encoding="utf-8")
    result = _run_check(workspace, "check_docs_catalog", optimized=True)
    assert result.returncode != 0
    assert "missing primary docs documents" in result.stderr
    assert "NEW_CONTRACT.md" in result.stderr


def test_retired_claim_check_covers_nested_docs_without_a_retired_path_manifest(
    docs_catalog_workspace,
):
    workspace, checker = docs_catalog_workspace
    (workspace / "pyproject.toml").write_text('version = "9.8.7"\n', encoding="utf-8")
    for relative in (
        "README.md",
        "ARCHITECTURE.md",
        "AGENTS.md",
        ".github/WORKFLOWS.md",
        "theory/README.md",
    ):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Reference 9.8.7\n", encoding="utf-8")
    checker.check_versions_and_retired_claims()
    nested = workspace / "docs" / "nodal" / "CLOSURE.md"
    nested.write_text("# complete system observability\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="retired claim.*CLOSURE.md"):
        checker.check_versions_and_retired_claims()
