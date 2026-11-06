"""Concept cards are the sole source for a checked glossary and its index."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from scripts import check_glossary as glossary


def _card(
    identifier,
    title,
    *,
    basis="primitive",
    emergence="not-claimed",
    dependencies="none",
):
    return (
        f'<a id="{identifier}"></a>\n\n### {title}\n\n'
        "- **Status:** maintained\n"
        f"- **Basis:** {basis}\n"
        f"- **Emergence:** {emergence}\n"
        f"- **Definition:** The declared {title.lower()} for this finite model.\n"
        "- **Domain:** A real coordinate on supplied finite support.\n"
        "- **Premises:** Fixed support and the stated constitutive law.\n"
        f"- **Dependencies:** {dependencies}\n"
        "- **Owner:** [Foundation](FOUNDATION.md#state)\n"
        "- **Evidence:** [Conditional result](FOUNDATION.md#result)\n"
        "- **Implementation:** [Kernel](../src/tnfr/kernel.py)\n"
        "- **Tests:** [Kernel contracts](../tests/test_kernel.py)\n"
        "- **Limits:** This card does not assert a physical identification.\n\n"
    )


@pytest.fixture
def glossary_workspace(tmp_path):
    theory = tmp_path / "theory"
    theory.mkdir()
    (theory / "FOUNDATION.md").write_text(
        "# Foundation\n\n## State\n\nA definition.\n\n## Result\n\nA conditional proof.\n",
        encoding="utf-8",
    )
    kernel = tmp_path / "src" / "tnfr" / "kernel.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("# Declared implementation owner.\n", encoding="utf-8")
    tests = tmp_path / "tests" / "test_kernel.py"
    tests.parent.mkdir()
    tests.write_text("# Declared contract-test owner.\n", encoding="utf-8")
    path = theory / "GLOSSARY.md"
    path.write_text(
        "# Small glossary\n\nPreserved introduction.\n\n"
        + glossary.INDEX_START
        + "\nobsolete index\n"
        + glossary.INDEX_END
        + "\n\n"
        + glossary.CARDS_START
        + "\n\n## State and laws\n\n"
        + _card("state", "Structural state")
        + _card(
            "response",
            "Conditional response",
            basis="derived",
            emergence="conditional",
            dependencies="state",
        )
        + glossary.CARDS_END
        + "\n\nPreserved ending.\n",
        encoding="utf-8",
    )
    glossary.update_concept_index(tmp_path)
    return tmp_path, path


def _replace(path, old, new):
    original = path.read_text(encoding="utf-8")
    assert old in original
    path.write_text(original.replace(old, new, 1), encoding="utf-8")


def _replace_response(path, old, new):
    original = path.read_text(encoding="utf-8")
    before, response = original.split('<a id="response"></a>', 1)
    assert old in response
    path.write_text(
        before + '<a id="response"></a>' + response.replace(old, new, 1),
        encoding="utf-8",
    )


def test_valid_cards_are_detached_ordered_and_read_only(glossary_workspace):
    root, path = glossary_workspace
    original = path.read_bytes()
    cards = glossary.load_cards(root)
    assert tuple(card.identifier for card in cards) == ("state", "response")
    assert cards[1].field("Dependencies") == "state"
    assert cards[1].field("Emergence") == "conditional"
    with pytest.raises(FrozenInstanceError):
        cards[0].title = "Mutated title"
    glossary.check_glossary(root)
    assert path.read_bytes() == original


def test_index_requires_explicit_refresh_and_preserves_authored_cards(
    glossary_workspace,
):
    root, path = glossary_workspace
    _replace(path, "### Conditional response", "### Audited response")
    authored = path.read_text(encoding="utf-8").split(glossary.CARDS_START, 1)[1]
    changed = path.read_bytes()
    with pytest.raises(RuntimeError):
        glossary.check_glossary(root)
    assert path.read_bytes() == changed
    glossary.update_concept_index(root)
    updated = path.read_text(encoding="utf-8")
    assert "[Audited response](#response)" in updated
    assert updated.split(glossary.CARDS_START, 1)[1] == authored
    assert "Preserved introduction." in updated
    assert updated.endswith("Preserved ending.\n")
    glossary.check_glossary(root)
    glossary.update_concept_index(root)
    assert path.read_text(encoding="utf-8") == updated


def test_added_concept_and_comma_dependencies_generate_the_index_from_cards(
    glossary_workspace,
):
    root, path = glossary_workspace
    _replace(
        path,
        glossary.CARDS_END,
        _card(
            "balance",
            "Derived balance",
            basis="derived",
            emergence="conditional",
            dependencies="state, response",
        )
        + glossary.CARDS_END,
    )
    with pytest.raises(RuntimeError):
        glossary.check_glossary(root)
    glossary.update_concept_index(root)
    glossary.check_glossary(root)
    assert "[Derived balance](#balance)" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "marker", ("CARDS_START", "CARDS_END", "INDEX_START", "INDEX_END")
)
def test_duplicate_regions_reject_without_rewriting_any_content(
    glossary_workspace, marker
):
    root, path = glossary_workspace
    document = path.read_text(encoding="utf-8") + getattr(glossary, marker) + "\n"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(RuntimeError):
        glossary.update_concept_index(root)
    assert path.read_text(encoding="utf-8") == document


def test_index_region_cannot_overlap_and_erase_the_authored_cards(glossary_workspace):
    root, path = glossary_workspace
    document = path.read_text(encoding="utf-8").replace(glossary.INDEX_END + "\n", "")
    document = document.replace(
        glossary.CARDS_END, glossary.INDEX_END + "\n" + glossary.CARDS_END
    )
    path.write_text(document, encoding="utf-8")
    with pytest.raises(RuntimeError):
        glossary.update_concept_index(root)
    assert path.read_text(encoding="utf-8") == document


def test_fenced_template_cannot_supply_the_active_card_region(glossary_workspace):
    root, path = glossary_workspace
    _replace(path, glossary.CARDS_START, "```markdown\n" + glossary.CARDS_START)
    _replace(path, glossary.CARDS_END, glossary.CARDS_END + "\n```")
    changed = path.read_bytes()
    with pytest.raises(RuntimeError):
        glossary.update_concept_index(root)
    assert path.read_bytes() == changed


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("- **Domain:** A real coordinate on supplied finite support.\n", ""),
        (
            "- **Status:** maintained",
            "- **Status:** maintained\n- **Status:** maintained",
        ),
        ("- **Premises:**", "- **Premise:**"),
        (
            "- **Domain:** A real coordinate on supplied finite support.",
            "- **Domain:**",
        ),
        ("A real coordinate on supplied finite support.", "TODO"),
        ("A real coordinate on supplied finite support.", "<description>"),
        ("A real coordinate on supplied finite support.", "[insert domain]"),
        (
            "- **Domain:** A real coordinate on supplied finite support.",
            "```markdown\n- **Domain:** A real coordinate on supplied finite support.\n```",
        ),
    ),
    ids=(
        "missing",
        "duplicate",
        "typo",
        "empty",
        "placeholder",
        "angle-placeholder",
        "bracket-placeholder",
        "fenced-only",
    ),
)
def test_incomplete_or_ambiguous_card_fields_reject_before_writing(
    glossary_workspace, old, new
):
    root, path = glossary_workspace
    _replace(path, old, new)
    changed = path.read_bytes()
    with pytest.raises(RuntimeError):
        glossary.update_concept_index(root)
    assert path.read_bytes() == changed


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("- **Status:** maintained", "- **Status:** canonical"),
        ("- **Basis:** primitive", "- **Basis:** proven"),
        ("- **Emergence:** not-claimed", "- **Emergence:** automatic"),
        ("- **Emergence:** not-claimed", "- **Emergence:** conditional"),
        ("- **Basis:** primitive", "- **Basis:** hypothesis"),
    ),
    ids=(
        "unknown-status",
        "unknown-basis",
        "unknown-emergence",
        "primitive-emergence",
        "maintained-hypothesis",
    ),
)
def test_classification_cannot_promote_a_premise_or_hypothesis(
    glossary_workspace, old, new
):
    root, path = glossary_workspace
    _replace(path, old, new)
    with pytest.raises(RuntimeError):
        glossary.load_cards(root)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("- **Dependencies:** state", "- **Dependencies:** absent"),
        ("- **Dependencies:** state", "- **Dependencies:** response"),
        ("- **Dependencies:** none", "- **Dependencies:** response"),
        ('<a id="response"></a>', '<a id="state"></a>'),
    ),
    ids=("unknown", "self-cycle", "two-card-cycle", "duplicate-id"),
)
def test_dependency_graph_requires_unambiguous_acyclic_concepts(
    glossary_workspace, old, new
):
    root, path = glossary_workspace
    _replace(path, old, new)
    with pytest.raises(RuntimeError):
        glossary.load_cards(root)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("FOUNDATION.md#state", "ABSENT.md#state"),
        ("FOUNDATION.md#state", "FOUNDATION.md#absent"),
        ("../src/tnfr/kernel.py", "../src/tnfr/missing.py"),
        ("../tests/test_kernel.py", "../tests/missing.py"),
    ),
    ids=("missing-owner", "missing-anchor", "missing-implementation", "missing-test"),
)
def test_card_references_resolve_against_the_declared_repository_root(
    glossary_workspace, old, new
):
    root, path = glossary_workspace
    _replace(path, old, new)
    with pytest.raises(RuntimeError):
        glossary.load_cards(root)


def test_derived_card_requires_a_declared_prerequisite(glossary_workspace):
    root, path = glossary_workspace
    _replace(path, "- **Dependencies:** state", "- **Dependencies:** none")
    with pytest.raises(RuntimeError):
        glossary.load_cards(root)


def test_distinct_concept_ids_cannot_hide_duplicate_titles(glossary_workspace):
    root, path = glossary_workspace
    _replace(path, "### Conditional response", "### Structural state")
    with pytest.raises(RuntimeError):
        glossary.load_cards(root)


@pytest.mark.parametrize(
    "replacement",
    (
        "[Passing tests](../tests/test_kernel.py)",
        "[Unlocated theorem](FOUNDATION.md)",
        "[Historical theorem](research/archive/OLD.md#result)",
    ),
    ids=("tests-only", "no-proof-fragment", "archive-only"),
)
def test_derived_evidence_requires_a_current_located_mathematical_owner(
    glossary_workspace, replacement
):
    root, path = glossary_workspace
    archive = root / "theory" / "research" / "archive"
    archive.mkdir(parents=True)
    (archive / "OLD.md").write_text("# Old result\n\n## Result\n", encoding="utf-8")
    _replace_response(path, "[Conditional result](FOUNDATION.md#result)", replacement)
    with pytest.raises(RuntimeError):
        glossary.load_cards(root)


@pytest.mark.parametrize("field", ("Owner", "Evidence"))
def test_glossary_cannot_supply_its_own_definition_owner_or_derivation(
    glossary_workspace, field
):
    root, path = glossary_workspace
    current = {
        "Owner": "[Foundation](FOUNDATION.md#state)",
        "Evidence": "[Conditional result](FOUNDATION.md#result)",
    }[field]
    _replace_response(path, current, "[Glossary summary](GLOSSARY.md#state)")
    with pytest.raises(RuntimeError):
        glossary.load_cards(root)


@pytest.mark.parametrize("filename", ("README.md", "conftest.py", "helper.py"))
def test_test_evidence_requires_an_actual_test_module(glossary_workspace, filename):
    root, path = glossary_workspace
    (root / "tests" / filename).write_text(
        "# Ancillary test support.\n", encoding="utf-8"
    )
    _replace(path, "../tests/test_kernel.py", "../tests/" + filename)
    with pytest.raises(RuntimeError):
        glossary.load_cards(root)


def test_candidate_can_report_absent_runtime_without_inventing_an_api(
    glossary_workspace,
):
    root, path = glossary_workspace
    _replace(path, "- **Status:** maintained", "- **Status:** candidate")
    _replace(path, "- **Basis:** primitive", "- **Basis:** hypothesis")
    _replace(
        path, "[Kernel](../src/tnfr/kernel.py)", "none: this is an open hypothesis"
    )
    _replace(
        path, "[Kernel contracts](../tests/test_kernel.py)", "none: no executable claim"
    )
    glossary.update_concept_index(root)
    glossary.check_glossary(root)


def test_implemented_maintained_card_cannot_omit_contract_tests(glossary_workspace):
    root, path = glossary_workspace
    _replace(
        path, "[Kernel contracts](../tests/test_kernel.py)", "none: not yet tested"
    )
    with pytest.raises(RuntimeError):
        glossary.load_cards(root)


@pytest.mark.parametrize("optimized", (False, True))
def test_invalid_glossary_fails_even_when_python_assertions_are_disabled(
    glossary_workspace, optimized
):
    root, path = glossary_workspace
    _replace(path, "- **Dependencies:** none", "- **Dependencies:** response")
    changed = path.read_bytes()
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.extend(
        [
            "-c",
            "from pathlib import Path; import sys; "
            "from scripts.check_glossary import check_glossary; "
            "check_glossary(Path(sys.argv[1]))",
            str(root),
        ]
    )
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "RuntimeError" in result.stderr
    assert path.read_bytes() == changed


def test_documentation_gate_and_writer_use_the_redirected_root(glossary_workspace):
    root, path = glossary_workspace
    checker_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "check_documentation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "glossary_documentation_gate", checker_path
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.REPO_ROOT = root
    checker.check_glossary()
    _replace(path, "### Conditional response", "### Scoped response")
    with pytest.raises(RuntimeError):
        checker.check_glossary()
    checker.update_glossary_index()
    checker.check_glossary()
    assert "[Scoped response](#response)" in path.read_text(encoding="utf-8")
