"""Logical research chapters retain their original Markdown link contexts."""

import importlib.util
from pathlib import Path

import pytest


def _checker(root, monkeypatch):
    script = (
        Path(__file__).resolve().parents[2] / "scripts/verify_internal_references.py"
    )
    spec = importlib.util.spec_from_file_location(
        "research_book_reference_fixture", script
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    monkeypatch.setattr(checker, "REPO_ROOT", root)
    return checker


@pytest.fixture
def book_workspace(tmp_path, monkeypatch):
    checker = _checker(tmp_path, monkeypatch)
    (tmp_path / "theory").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src/engine.py").write_text("# Kept source\n", encoding="utf-8")
    (tmp_path / "theory/README.md").write_text("# Theory index\n", encoding="utf-8")
    # The actual parser owns extraction. This fixture tests its stated API:
    # book text is not a page whose every chapter shares the book directory.
    (tmp_path / checker.RESEARCH_BOOK).write_text(
        "# Raw book\n[Chapter-relative link](../../src/engine.py)\n",
        encoding="utf-8",
    )
    pages = {
        "theory/nodal/FIRST.md": b"# First\n## Repeat\n",
        "theory/SECOND.md": b"# Second\n## Repeat\n## Repeat\n",
        "theory/nodal/README.md": b"# Nodal index\n",
    }
    front = {
        "text": '# Research programme\n<a id="chapter-theory-nodal-first-md"></a>\n'
    }

    def relative(path):
        path = Path(path)
        if path.is_absolute():
            return path.relative_to(tmp_path).as_posix()
        return path.as_posix()

    def read_document(root, path):
        assert root == tmp_path
        key = relative(path)
        if key == checker.RESEARCH_BOOK.as_posix():
            return front["text"]
        if key in pages:
            return pages[key].decode("utf-8-sig")
        return (root / key).read_text(encoding="utf-8-sig")

    def document_exists(root, path):
        return relative(path) in pages or (root / path).is_file()

    monkeypatch.setattr(checker, "documents", lambda root: pages.copy())
    monkeypatch.setattr(checker, "read_document", read_document)
    monkeypatch.setattr(checker, "document_exists", document_exists)
    return tmp_path, checker, pages, front


def test_book_front_and_logical_chapters_use_distinct_relative_contexts(book_workspace):
    root, checker, pages, front = book_workspace
    front["text"] += (
        "[First](nodal/FIRST.md#repeat)\n" "[Wrapper](#chapter-theory-nodal-first-md)\n"
    )
    pages["theory/nodal/FIRST.md"] += (
        b"[Second](../SECOND.md#repeat-1)\n"
        b"[Code](../../src/engine.py)\n"
        b"[Index](../README.md#theory-index)\n"
        b"[Own heading](#repeat)\n"
    )
    assert not (root / "theory/nodal").exists()

    references, failures = checker.verify(["."])

    assert references == 6
    assert failures == []


def test_selecting_physical_book_checks_every_logical_chapter_once(book_workspace):
    _, checker, pages, front = book_workspace
    front["text"] += "[Front](#research-programme)\n"
    pages["theory/nodal/FIRST.md"] += b"[First](#first)\n"
    pages["theory/SECOND.md"] += b"[Second](#second)\n"

    references, failures = checker.verify([checker.RESEARCH_BOOK.as_posix(), "theory"])

    assert references == 3
    assert failures == []


@pytest.mark.parametrize("selection", ("theory/nodal", "theory/nodal/FIRST.md"))
def test_removed_logical_path_is_a_valid_selected_search_root(
    book_workspace, selection
):
    root, checker, pages, _ = book_workspace
    pages["theory/nodal/FIRST.md"] += b"[Other](../SECOND.md#repeat)\n"
    pages["theory/SECOND.md"] += b"[Not selected](absent.md)\n"
    assert not (root / selection).exists()

    references, failures = checker.verify([selection])

    assert references == 1
    assert failures == []


def test_logical_heading_counters_and_explicit_ids_do_not_leak_between_chapters(
    book_workspace,
):
    _, checker, pages, _ = book_workspace
    pages["theory/SECOND.md"] += b'<a id="second-only"></a>\n'
    pages["theory/nodal/FIRST.md"] += (
        b"[Other duplicate](../SECOND.md#repeat-1)\n"
        b"[Missing local duplicate](#repeat-1)\n"
        b"[Foreign ID](#second-only)\n"
    )

    references, failures = checker.verify(["theory/nodal/FIRST.md"])

    assert references == 3
    assert len(failures) == 2
    assert all(failure.startswith("missing fragment:") for failure in failures)
    assert "Missing local duplicate" not in "\n".join(failures)
    assert "#repeat-1" in failures[0] and "#second-only" in failures[1]
    assert "repeat-1" not in checker._anchors(
        checker.REPO_ROOT / "theory/nodal/FIRST.md"
    )


def test_current_repository_urls_resolve_logical_chapter_fragments(book_workspace):
    root, checker, _, _ = book_workspace
    (root / "guide.md").write_text(
        "[Current](https://github.com/fermga/TNFR-Python-Engine/blob/main/"
        "theory/nodal/FIRST.md#repeat)\n"
        "[Historical](https://github.com/fermga/TNFR-Python-Engine/blob/"
        "a1b2c3/theory/removed.md#historical)\n",
        encoding="utf-8",
    )

    references, failures = checker.verify(["guide.md"])

    assert references == 1
    assert failures == []


def test_directory_fragment_uses_a_logical_index(book_workspace):
    _, checker, pages, _ = book_workspace
    pages["theory/SECOND.md"] += b"[Directory](nodal/#nodal-index)\n"

    references, failures = checker.verify(["theory/SECOND.md"])

    assert references == 1
    assert failures == []


def test_logical_directory_without_index_keeps_failure_diagnostic(book_workspace):
    _, checker, pages, _ = book_workspace
    del pages["theory/nodal/README.md"]
    pages["theory/SECOND.md"] += b"[Directory](nodal/#nodal-index)\n"

    references, failures = checker.verify(["theory/SECOND.md"])

    assert references == 1
    assert len(failures) == 1
    assert failures[0].startswith("directory fragment has no local index:")


def test_logical_chapter_missing_targets_and_repository_escape_remain_failures(
    book_workspace,
):
    _, checker, pages, _ = book_workspace
    pages[
        "theory/nodal/FIRST.md"
    ] += b"[Removed](MISSING.md)\n[Outside](../../../outside.md)\n"

    references, failures = checker.verify(["theory/nodal/FIRST.md", "missing-owner.md"])

    assert references == 2
    assert failures[0] == "missing search root: missing-owner.md"
    assert failures[1].startswith("missing target:")
    assert failures[2].startswith("target outside repository:")


def test_logical_links_keep_code_and_math_masks(book_workspace):
    _, checker, pages, _ = book_workspace
    pages["theory/nodal/FIRST.md"] += (
        b"[Real](../SECOND.md#second)\n\n"
        b"```markdown\n[Example](absent.md)\n```\n\n"
        b"`[Inline](absent.md)`\n\n"
        b"\\[ [algebra](absent.md) \\]\n"
    )

    references, failures = checker.verify(["theory/nodal/FIRST.md"])

    assert references == 1
    assert failures == []


@pytest.fixture
def real_book_workspace(tmp_path, monkeypatch):
    checker = _checker(tmp_path, monkeypatch)
    from scripts.research_book import Alias, Book, Chapter, chapter_id, write_book

    owner_path = "theory/nodal/OWNER.md"
    book = Book(
        front=(
            '# Research programme\n<a id="research-objective"></a>\n'
            '<a id="research-current-question"></a>\n'
            f"[Wrapper](#{chapter_id(owner_path)})\n"
        ),
        chapters=(
            Chapter(
                path=owner_path,
                title="Owner",
                group="Technical proofs",
                body=(
                    b"# Owner\n## Result\n"
                    b"[Source](../../src/engine.py)\n"
                    b"[Peer](../PEER.md#repeat-1)\n"
                    b"[Prior plan](../research/PLAN.md#closed-gate)\n"
                    b"[Own](#result)\n"
                ),
            ),
            Chapter(
                path="theory/PEER.md",
                title="Peer",
                group="Technical proofs",
                body=b"# Peer\n## Repeat\n## Repeat\n",
            ),
        ),
        aliases=(Alias("theory/research/PLAN.md", "Prior plan", ("closed-gate",)),),
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src/engine.py").write_text("# Kept source\n", encoding="utf-8")
    write_book(tmp_path, book)
    return tmp_path, checker, book


def test_real_book_extraction_keeps_contexts_wrappers_and_retired_routes(
    real_book_workspace,
):
    root, checker, _ = real_book_workspace
    assert not (root / "theory/nodal").exists()
    assert not (root / "theory/research").exists()

    references, failures = checker.verify([checker.RESEARCH_BOOK.as_posix()])

    assert references == 9
    assert failures == []
    assert "result" in checker._anchors(root / "theory/nodal/OWNER.md")
    assert "result" not in checker._anchors(root / checker.RESEARCH_BOOK)
    assert "closed-gate" in checker._anchors(root / "theory/research/PLAN.md")


def test_real_book_changed_chapter_is_revalidated_at_its_logical_path(
    real_book_workspace,
):
    root, checker, book = real_book_workspace
    from dataclasses import replace

    from scripts.research_book import write_book

    assert checker.verify(["theory/nodal"])[1] == []
    changed_owner = replace(
        book.chapters[0], body=book.chapters[0].body + b"[Absent](MISSING.md)\n"
    )
    write_book(root, replace(book, chapters=(changed_owner, book.chapters[1])))

    references, failures = checker.verify(["theory/nodal"])

    assert references == 5
    assert len(failures) == 1
    assert failures[0].startswith("missing target:")


def test_modular_book_keeps_wrapper_links_and_detects_direct_edit_failures(
    real_book_workspace,
):
    root, checker, book = real_book_workspace
    from scripts.research_book import write_modular_book

    write_modular_book(root, book)
    references, failures = checker.verify([checker.RESEARCH_BOOK.as_posix()])
    assert references == 9
    assert failures == []
    owner = root / book.chapters[0].path
    owner.write_bytes(owner.read_bytes() + b"[Absent](MISSING.md)\n")

    references, failures = checker.verify(["theory/nodal"])

    assert references == 5
    assert len(failures) == 1
    assert failures[0].startswith("missing target:")
    assert "OWNER.md -> MISSING.md" in failures[0]
