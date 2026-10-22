"""Book metadata owns both repository catalog and published chapter navigation."""

import json
from dataclasses import replace

import pytest

from scripts import check_documentation as checker
from scripts.research_book import (
    BOOK_PATH,
    CATALOG_PATH,
    Alias,
    Book,
    Chapter,
    documents,
    write_book,
    write_modular_book,
)


@pytest.fixture
def catalog_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    book = Book(
        "# Research program\n",
        (
            Chapter(
                "theory/FOUNDATION.md",
                "Foundation",
                "Foundations",
                b"# Foundation\n",
            ),
            Chapter(
                "theory/nodal/RESULT.md",
                "Conditional result",
                "Technical results",
                b"# Conditional result\n",
            ),
        ),
        (
            Alias("theory/research/PLAN.md", "Retired plan", ("old-decision",)),
            Alias("TNFR_lineas_de_investigacion.txt", "Retired portfolio"),
        ),
    )
    write_book(tmp_path, book)
    index = tmp_path / "theory/README.md"
    index.write_text(
        "# Retained index\n\nIntroductory navigation only.\n\n"
        + checker.THEORY_CATALOG_START
        + "\n\nObsolete manually maintained catalog.\n\n"
        + checker.THEORY_CATALOG_END
        + '\n\n<a id="published-old-anchor"></a>\nRetained compatibility prose.\n',
        encoding="utf-8",
    )
    return tmp_path, book, index


def test_generated_catalog_uses_physical_book_anchors_and_nav_uses_logical_routes(
    catalog_workspace,
):
    root, _, index = catalog_workspace
    original_book = (root / BOOK_PATH).read_bytes()

    checker.update_theory_catalog()
    checker.check_theory_catalog()
    catalog = index.read_text(encoding="utf-8")
    navigation = checker.render_theory_navigation()

    assert "[Foundation](RESEARCH_PROGRAM.md#chapter-theory-foundation-md)" in catalog
    assert (
        "[Conditional result](RESEARCH_PROGRAM.md#chapter-theory-nodal-result-md)"
        in catalog
    )
    assert "[Retired plan](RESEARCH_PROGRAM.md#research-current-question)" in catalog
    assert "TNFR_lineas_de_investigacion.txt" not in catalog
    assert "(FOUNDATION.md)" not in catalog
    assert '"Foundation": theory/FOUNDATION.md' in navigation
    assert '"Conditional result": theory/nodal/RESULT.md' in navigation
    assert '"Retired plan": theory/research/PLAN.md' in navigation
    assert "#chapter-" not in navigation
    assert (root / BOOK_PATH).read_bytes() == original_book


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_catalog_generation_preserves_surrounding_bytes_and_is_idempotent(
    catalog_workspace, newline
):
    _, _, index = catalog_workspace
    original = b"\xef\xbb\xbf" + index.read_bytes().replace(b"\r\n", b"\n").replace(
        b"\n", newline.encode("ascii")
    )
    index.write_bytes(original)
    prefix = original.split(checker.THEORY_CATALOG_START.encode())[0]
    suffix = original.split(checker.THEORY_CATALOG_END.encode())[1]

    checker.update_theory_catalog()
    generated = index.read_bytes()
    checker.update_theory_catalog()

    assert generated.startswith(prefix)
    assert generated.endswith(suffix)
    assert b'\n<a id="published-old-anchor"></a>' in generated
    assert index.read_bytes() == generated
    checker.check_theory_catalog()


@pytest.mark.parametrize("field", ("title", "group", "path", "alias"))
def test_changed_book_metadata_requires_catalog_regeneration(catalog_workspace, field):
    root, book, index = catalog_workspace
    checker.update_theory_catalog()
    if field == "alias":
        changed = replace(
            book, aliases=(replace(book.aliases[0], title="Changed route"),)
        )
        expected = "Changed route"
    else:
        expected = {
            "title": "Changed title",
            "group": "Changed group",
            "path": "theory/CHANGED.md",
        }[field]
        chapter = replace(book.chapters[0], **{field: expected})
        changed = replace(book, chapters=(chapter, book.chapters[1]))
    write_book(root, changed)
    stale = index.read_bytes()

    with pytest.raises(RuntimeError, match="theory catalog drifted"):
        checker.check_theory_catalog()
    assert index.read_bytes() == stale
    checker.update_theory_catalog()
    checker.check_theory_catalog()
    assert expected in checker.render_theory_navigation()


def test_catalog_labels_cannot_inject_rows_links_or_html(catalog_workspace):
    root, book, index = catalog_workspace
    chapter = replace(
        book.chapters[0],
        title='[A] | <b>B</b> & "C" `D`',
        group="A & B: [comparison]",
    )
    write_book(root, replace(book, chapters=(chapter, book.chapters[1])))

    checker.update_theory_catalog()
    checker.check_theory_catalog()
    catalog = index.read_text(encoding="utf-8")
    navigation = checker.render_theory_navigation()

    assert "&#91;A&#93; &#124; &lt;b&gt;B&lt;/b&gt; &amp;" in catalog
    assert "&#96;D&#96;" in catalog
    assert "## A &amp; B: &#91;comparison&#93;" in catalog
    assert '\\"C\\"' in navigation
    assert "&#91;A&#93;" not in navigation


@pytest.mark.parametrize("corruption", ("missing", "reversed", "duplicate"))
def test_bad_catalog_boundaries_cannot_trigger_partial_writes(
    catalog_workspace, corruption
):
    _, _, index = catalog_workspace
    original = index.read_text(encoding="utf-8")
    if corruption == "missing":
        bad = original.replace(checker.THEORY_CATALOG_END, "")
    elif corruption == "duplicate":
        bad = original + checker.THEORY_CATALOG_START
    else:
        bad = (
            original.replace(checker.THEORY_CATALOG_START, "PLACEHOLDER")
            .replace(checker.THEORY_CATALOG_END, checker.THEORY_CATALOG_START)
            .replace("PLACEHOLDER", checker.THEORY_CATALOG_END)
        )
    index.write_text(bad, encoding="utf-8")
    before = index.read_bytes()

    with pytest.raises(RuntimeError, match="exactly one|markers are reversed"):
        checker.update_theory_catalog()
    assert index.read_bytes() == before


def test_unintegrated_physical_theory_owner_is_still_reported(catalog_workspace):
    root, _, _ = catalog_workspace
    checker.update_theory_catalog()
    (root / "theory/UNINTEGRATED.md").write_text("# Unintegrated\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing primary theory documents"):
        checker.check_theory_catalog()


def test_no_book_catalog_writer_preserves_the_existing_catalog(
    catalog_workspace,
):
    root, _, index = catalog_workspace
    (root / BOOK_PATH).unlink()
    before = index.read_bytes()

    checker.update_theory_catalog()

    assert index.read_bytes() == before


def test_modular_catalog_links_to_editable_chapters(catalog_workspace):
    root, book, index = catalog_workspace
    write_modular_book(root, book)
    checker.update_theory_catalog()
    checker.check_theory_catalog()

    catalog = index.read_text(encoding="utf-8")
    assert "[Foundation](FOUNDATION.md)" in catalog
    assert "[Conditional result](nodal/RESULT.md)" in catalog
    assert "[Retired plan](RESEARCH_PROGRAM.md#research-current-question)" in catalog
    assert "RESEARCH_PROGRAM.md#chapter-" not in catalog
    assert '"Foundation": theory/FOUNDATION.md' in checker.render_theory_navigation()
    assert (root / book.chapters[0].path).read_bytes() == book.chapters[0].body


def test_modular_metadata_edit_refreshes_physical_program_routes(catalog_workspace):
    root, book, index = catalog_workspace
    write_modular_book(root, book)
    checker.update_theory_catalog()
    catalog_path = root / CATALOG_PATH
    catalog = json.loads(catalog_path.read_bytes())
    catalog["chapters"][0]["title"] = "Clarified foundation"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    chapter_bytes = (root / book.chapters[0].path).read_bytes()

    with pytest.raises(RuntimeError, match="program chapter routes drifted"):
        checker.check_theory_catalog()

    checker.update_theory_catalog()
    checker.check_theory_catalog()
    assert "[Clarified foundation](FOUNDATION.md)" in index.read_text(encoding="utf-8")
    assert "[Clarified foundation](FOUNDATION.md)" in (root / BOOK_PATH).read_text(
        encoding="utf-8"
    )
    assert (root / book.chapters[0].path).read_bytes() == chapter_bytes


@pytest.mark.parametrize("drift", ("missing", "modified"))
def test_modular_retired_routes_are_local_generated_views(catalog_workspace, drift):
    root, book, _ = catalog_workspace
    write_modular_book(root, book)
    checker.update_theory_catalog()
    checker.check_theory_catalog()
    expected = documents(root)
    for alias in book.aliases:
        assert (root / alias.path).read_bytes() == expected[alias.path]
    route = root / book.aliases[0].path
    if drift == "missing":
        route.unlink()
    else:
        route.write_bytes(b"Obsolete independent task instructions.\n")

    with pytest.raises(RuntimeError, match="generated compatibility views drifted"):
        checker.check_theory_catalog()

    checker.update_theory_catalog()
    checker.check_theory_catalog()
    assert route.read_bytes() == expected[book.aliases[0].path]
    assert b"Historical task instructions are inactive" in route.read_bytes()
    assert b"Obsolete independent task instructions" not in route.read_bytes()
    assert (root / book.chapters[0].path).read_bytes() == book.chapters[0].body
