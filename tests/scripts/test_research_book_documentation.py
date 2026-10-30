"""The research container preserves logical ownership and published routes."""

from dataclasses import replace

import pytest

from scripts import check_documentation as checker
from scripts import prepare_docs as staging
from scripts.research_book import (
    BOOK_PATH,
    Alias,
    Book,
    Chapter,
    documents,
    front,
    read_book,
    read_document,
    write_book,
    write_modular_book,
)


@pytest.fixture
def book_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(staging, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(staging, "STAGE_DIR", tmp_path / "build/docs-source")
    for relative in staging.ROOT_FILES:
        if relative != "TNFR_lineas_de_investigacion.txt":
            (tmp_path / relative).write_text(
                "Retained root document.\n", encoding="utf-8"
            )
    (tmp_path / "README.md").write_text("# Home\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/README.md").write_text("# Documentation\n", encoding="utf-8")
    book = Book(
        "# Research program\n\n## Research objective\n\nA single current question.\n",
        (
            Chapter(
                "theory/FOUNDATION.md",
                "Foundation",
                "Foundations",
                b"# Foundation\r\n\r\nPreserved conditional result.\r\n",
            ),
            Chapter(
                "theory/nodal/CLOSURE.md",
                "Closure",
                "Foundations",
                b"# Closure\n\n[Foundation](../FOUNDATION.md#foundation)\n",
            ),
        ),
        (
            Alias(
                "theory/research/FIVE_STAGE_EXECUTION_PLAN.md",
                "Retired execution plan",
                ("current-checkpoint",),
            ),
            Alias("TNFR_lineas_de_investigacion.txt", "Retired research portfolio"),
        ),
    )
    write_book(tmp_path, book)
    (tmp_path / "theory/README.md").write_text(
        "# Theory\n\n" + checker.THEORY_CATALOG_START + "\n\n## Current program\n\n"
        "| Document | Scope |\n| --- | --- |\n"
        "| [Research program](RESEARCH_PROGRAM.md) | Current question |\n"
        "\n## Foundations\n\n| Document | Scope |\n| --- | --- |\n"
        "| [Foundation](FOUNDATION.md#foundation) | Definitions |\n"
        "| [Closure](nodal/CLOSURE.md) | Conditional result |\n"
        "\n## Compatibility\n\n| Document | Scope |\n| --- | --- |\n"
        "| [Retired plan](research/FIVE_STAGE_EXECUTION_PLAN.md) | Inactive route |\n"
        + checker.THEORY_CATALOG_END
        + "\n",
        encoding="utf-8",
    )
    checker.update_theory_catalog()
    return tmp_path, book


def test_logical_catalog_and_navigation_retain_original_routes(book_workspace):
    root, _ = book_workspace
    expected = {
        BOOK_PATH,
        "theory/FOUNDATION.md",
        "theory/nodal/CLOSURE.md",
        "theory/research/FIVE_STAGE_EXECUTION_PLAN.md",
    }
    before = (root / BOOK_PATH).read_bytes()

    checker.check_theory_catalog()
    checker.check_documentation_size()
    rendered = checker.render_theory_navigation()

    assert checker._catalog_paths("theory") == {root / path for path in expected}
    assert not (root / "theory/nodal").exists()
    assert not (root / "theory/research").exists()
    for path in expected:
        assert rendered.count(path) == 1
    assert "#foundation" not in rendered
    assert (root / BOOK_PATH).read_bytes() == before


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "unknown"))
def test_logical_catalog_keeps_completeness_and_uniqueness_checks(
    book_workspace, corruption
):
    root, _ = book_workspace
    path = root / "theory/README.md"
    original = path.read_text(encoding="utf-8")
    row = "| [Closure](RESEARCH_PROGRAM.md#chapter-theory-nodal-closure-md) | Sourcebook chapter |\n"
    if corruption == "missing":
        modified = original.replace(row, "")
    elif corruption == "duplicate":
        modified = original.replace(row, row * 2)
    else:
        modified = original.replace(
            "chapter-theory-nodal-closure-md", "chapter-theory-nodal-absent-md"
        )
    path.write_text(modified, encoding="utf-8")

    assert modified != original
    with pytest.raises(RuntimeError, match="theory catalog drifted"):
        checker.check_theory_catalog()


def test_new_physical_study_still_requires_catalog_entry(book_workspace):
    root, _ = book_workspace
    (root / "theory/UNREGISTERED.md").write_text("# New study\n", encoding="utf-8")

    with pytest.raises(
        RuntimeError, match="missing primary theory documents.*UNREGISTERED"
    ):
        checker.check_theory_catalog()


def test_size_limit_applies_to_each_chapter_not_whole_book(book_workspace):
    root, book = book_workspace
    chapters = tuple(
        replace(chapter, body=b"Retained proof line.\r\n" * 2500)
        for chapter in book.chapters
    )
    write_book(root, replace(book, chapters=chapters))
    assert len((root / BOOK_PATH).read_bytes().splitlines()) > 4000

    checker.check_documentation_size()

    oversized = replace(chapters[1], body=b"Proof line.\r\n" * 4000 + b"Final line.")
    write_book(root, replace(book, chapters=(chapters[0], oversized)))
    before = (root / BOOK_PATH).read_bytes()
    with pytest.raises(RuntimeError, match="CLOSURE.md has 4001 physical lines"):
        checker.check_documentation_size()
    assert (root / BOOK_PATH).read_bytes() == before


def test_current_program_front_has_its_own_size_limit(book_workspace):
    root, book = book_workspace
    write_book(root, replace(book, front="# Current program\n" * 4001))

    with pytest.raises(RuntimeError, match="RESEARCH_PROGRAM.md has .*physical lines"):
        checker.check_documentation_size()


def test_grammar_refresh_changes_only_its_logical_chapter(book_workspace, monkeypatch):
    root, book = book_workspace
    grammar = Chapter(
        "theory/UNIFIED_GRAMMAR_RULES.md",
        "Grammar",
        "Foundations",
        (
            "# Grammar\r\n\r\nPreserved introduction.\r\n"
            + checker.GRAMMAR_START
            + "\r\n\r\nobsolete\r\n\r\n"
            + checker.GRAMMAR_END
            + "\r\n\r\nPreserved conclusion.\r\n"
        ).encode("utf-8"),
    )
    write_book(root, replace(book, chapters=(*book.chapters, grammar)))
    monkeypatch.setattr(checker, "render_grammar_roles_table", lambda: "| Fresh role |")
    with pytest.raises(RuntimeError, match="grammar role view drifted"):
        checker.check_grammar_view()

    checker.update_grammar_view()
    checker.check_grammar_view()

    changed = read_book(root)
    assert changed.front == book.front and changed.aliases == book.aliases
    assert changed.chapters[:-1] == book.chapters
    assert changed.chapters[-1].body == grammar.body.replace(
        b"obsolete", b"| Fresh role |"
    )
    assert not (root / grammar.path).exists()
    first_update = (root / BOOK_PATH).read_bytes()
    checker.update_grammar_view()
    assert (root / BOOK_PATH).read_bytes() == first_update


def test_staging_emits_front_chapters_and_compact_compatibility_views(book_workspace):
    root, book = book_workspace
    closure = replace(
        book.chapters[1],
        body=(
            b"# Closure\n\n[Home](../../README.md#home)\n"
            b"[Local index](./#nodal-index)\n"
            b"[Foundation](../FOUNDATION.md#foundation)\n"
            b"`[Literal](../../README.md#home)`\n"
            b"```markdown\n[Example](../../README.md#home)\n```\n"
        ),
    )
    index = Chapter(
        "theory/nodal/README.md", "Nodal index", "Foundations", b"# Nodal index\n"
    )
    write_book(root, replace(book, chapters=(book.chapters[0], closure, index)))
    before = (root / BOOK_PATH).read_bytes()
    assert not (root / "theory/nodal").exists()
    assert not (root / "TNFR_lineas_de_investigacion.txt").exists()

    output = staging.prepare()

    rendered = (output / closure.path).read_text(encoding="utf-8")
    assert "[Home](../../index.md#home)" in rendered
    assert "[Local index](README.md#nodal-index)" in rendered
    assert "[Foundation](../FOUNDATION.md#foundation)" in rendered
    assert "`[Literal](../../README.md#home)`" in rendered
    assert "```markdown\n[Example](../../README.md#home)\n```" in rendered
    assert (output / book.chapters[0].path).read_bytes() == book.chapters[0].body
    assert (output / BOOK_PATH).read_text(encoding="utf-8") == front(root)
    assert "Preserved conditional result." not in (output / BOOK_PATH).read_text(
        encoding="utf-8"
    )
    for alias in book.aliases:
        assert (output / alias.path).read_bytes() == documents(root)[alias.path]
        assert "Historical task instructions are inactive" in read_document(
            root, alias.path
        )
    assert (root / BOOK_PATH).read_bytes() == before
    assert not (root / closure.path).exists()


def test_invalid_book_does_not_destroy_prior_staging(book_workspace):
    root, _ = book_workspace
    prior = staging.STAGE_DIR / "prior.json"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b'{"retained":"prior build"}\n')
    (root / BOOK_PATH).write_bytes(b"# Missing container boundaries\n")

    with pytest.raises(ValueError, match="TNFR ROUTES boundary"):
        staging.prepare()

    assert prior.read_bytes() == b'{"retained":"prior build"}\n'


def test_staging_routes_book_and_main_repository_links_to_local_chapters(
    book_workspace,
):
    root, _ = book_workspace
    base = "https://github.com/fermga/TNFR-Python-Engine/blob/main/"
    book_link = base + "theory/RESEARCH_PROGRAM.md#chapter-theory-foundation-md"
    old_commit = base.replace("/main/", "/55eb72af/") + "theory/FOUNDATION.md"
    (root / "README.md").write_text(
        "# Home\n\n[Definition](" + book_link + ")\n"
        "[Relative](theory/RESEARCH_PROGRAM.md#chapter-theory-nodal-closure-md)\n"
        "[Program](" + base + "theory/RESEARCH_PROGRAM.md#research-objective)\n"
        "[Pinned](" + old_commit + ")\n"
        "[Unavailable payload](" + base + "docs/assets/absent.json)\n"
        "[Reference][concept]\n\n[concept]: " + book_link + "\n"
        "`[Literal](" + book_link + ")`\n",
        encoding="utf-8",
    )
    output = staging.prepare()
    result = (output / "index.md").read_text(encoding="utf-8")
    assert "[Definition](theory/FOUNDATION.md)" in result
    assert "[Relative](theory/nodal/CLOSURE.md)" in result
    assert "[Program](theory/RESEARCH_PROGRAM.md#research-objective)" in result
    assert "[concept]: theory/FOUNDATION.md" in result
    assert "[Pinned](" + old_commit + ")" in result
    assert "[Unavailable payload](" + base + "docs/assets/absent.json)" in result
    assert "`[Literal](" + book_link + ")`" in result


def test_modular_staging_observes_direct_chapter_edits(book_workspace):
    root, book = book_workspace
    write_modular_book(root, book)
    chapter = root / book.chapters[0].path
    edited = b"# Foundation\r\n\r\nA directly edited premise.\r\n"
    chapter.write_bytes(edited)
    program = (root / BOOK_PATH).read_bytes()

    output = staging.prepare()

    assert (output / book.chapters[0].path).read_bytes() == edited
    assert (root / BOOK_PATH).read_bytes() == program
    assert "directly edited premise" not in (output / BOOK_PATH).read_text(
        encoding="utf-8"
    )


def test_missing_modular_chapter_preserves_prior_staging(book_workspace):
    root, book = book_workspace
    write_modular_book(root, book)
    prior = staging.STAGE_DIR / "prior.json"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b'{"retained":"prior build"}\n')
    (root / book.chapters[1].path).unlink()

    with pytest.raises((FileNotFoundError, ValueError), match="CLOSURE.md"):
        staging.prepare()

    assert prior.read_bytes() == b'{"retained":"prior build"}\n'


def test_modular_program_has_a_byte_budget_even_with_few_long_lines(book_workspace):
    root, book = book_workspace
    write_modular_book(root, book)
    # The editor must not be asked to load a second monolith hidden in long lines.
    program = root / BOOK_PATH
    program.write_bytes(
        program.read_bytes() + b"x" * checker.MAX_RESEARCH_PROGRAM_BYTES
    )

    with pytest.raises(RuntimeError, match="RESEARCH_PROGRAM.md has .*bytes"):
        checker.check_documentation_size()
