"""The research container keeps chapter bytes and admits unambiguous routes."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import research_book as book_io


def _wire(
    *,
    routes=b'{"version":1,"aliases":[]}',
    chapter=b'{"path":"theory/A.md","title":"A","group":"Foundations"}',
    body=b"# Proof\r\nLast line without newline",
    front=b"# Program\r\n",
):
    """An independently written instance of the documented Markdown container."""
    return (
        front
        + b"\n<!-- TNFR ROUTES "
        + routes
        + b" -->\n"
        + b"\n<!-- TNFR CHAPTER "
        + chapter
        + b" -->\n"
        + body
        + b"\n<!-- END TNFR CHAPTER -->\n"
    )


@pytest.fixture
def sample_book():
    return book_io.Book(
        "# Program\r\n\r\nA bounded current question.\r\n",
        (
            book_io.Chapter(
                "theory/A.md",
                "First result",
                "Foundations",
                b"# First\r\n\r\nOld statement.\r\nLast line",
            ),
            book_io.Chapter(
                "theory/nodal/B.md",
                "Second result",
                "Dynamics",
                b"\xef\xbb\xbf# Second\n\r\nSECRET PROOF BODY\n"
                b'<a id="same-local-anchor"></a>',
            ),
            book_io.Chapter(
                "theory/C.md",
                "Third result",
                "Dynamics",
                b"# Third\n\nNo final newline",
            ),
        ),
        (
            book_io.Alias(
                "theory/research/OLD_PLAN.md",
                "Retired plan",
                ("old-gate", "old-result"),
            ),
            book_io.Alias("OLD_PORTFOLIO.txt", "Retired portfolio"),
        ),
    )


@pytest.mark.parametrize(
    "body",
    (
        b"",
        b"# Proof\r\n\r\nAll CRLF.\r\n",
        b"# Proof\nMixed\r\nNo ending newline",
        "# Théorème\r\nπ is a chart constant.".encode(),
        b"\xef\xbb\xbf# A retained BOM\r\n",
    ),
)
def test_independent_wire_roundtrip_preserves_body_bytes(body):
    parsed = book_io.parse_book(_wire(body=body))

    assert parsed.front == "# Program\r\n"
    assert parsed.chapters[0].body == body
    rendered = book_io.render_book(parsed)
    assert book_io.parse_book(rendered) == parsed
    assert book_io.parse_book(rendered).chapters[0].body == body
    assert book_io.render_book(book_io.parse_book(rendered)) == rendered


def test_disk_roundtrip_and_logical_read_do_not_materialize_chapters(
    tmp_path, sample_book
):
    book_io.write_book(tmp_path, sample_book)

    assert book_io.read_book(tmp_path) == sample_book
    assert book_io.documents(tmp_path)["theory/nodal/B.md"] == (
        sample_book.chapters[1].body
    )
    assert book_io.read_document_bytes(tmp_path, "theory/A.md") == (
        sample_book.chapters[0].body
    )
    assert book_io.read_document(tmp_path, "theory/nodal/B.md").startswith(
        "# Second\n\n"
    )
    assert book_io.document_exists(tmp_path, "theory/nodal/B.md")
    assert not (tmp_path / "theory/nodal").exists()
    assert not (tmp_path / "theory/A.md").exists()


def test_one_chapter_update_preserves_siblings_front_and_metadata(
    tmp_path, sample_book
):
    book_io.write_book(tmp_path, sample_book)
    book_io.write_document(
        tmp_path,
        tmp_path / "theory/A.md",
        "# First\n\nCorrected statement.\nLast line",
    )
    updated = book_io.read_book(tmp_path)

    assert updated.front == sample_book.front
    assert updated.aliases == sample_book.aliases
    assert updated.chapters[0] == replace(
        sample_book.chapters[0],
        body=b"# First\r\n\r\nCorrected statement.\r\nLast line",
    )
    assert updated.chapters[1:] == sample_book.chapters[1:]
    assert not (tmp_path / "theory/A.md").exists()


def test_normalized_noop_preserves_original_crlf_book_without_writing(
    tmp_path, sample_book, monkeypatch
):
    book_io.write_book(tmp_path, sample_book)
    path = tmp_path / book_io.BOOK_PATH
    before = path.read_bytes()

    def forbidden_write(*args, **kwargs):
        pytest.fail("A normalized no-op must not rewrite the sourcebook")

    monkeypatch.setattr(book_io, "write_book", forbidden_write)
    book_io.write_document(
        tmp_path, "theory/A.md", "# First\n\nOld statement.\nLast line"
    )
    assert path.read_bytes() == before


def test_changed_chapter_is_seen_after_an_already_cached_read(tmp_path, sample_book):
    book_io.write_book(tmp_path, sample_book)
    assert book_io.read_document(tmp_path, "theory/C.md").endswith("No final newline")

    book_io.write_document(tmp_path, "theory/C.md", "# Third\n\nRevised result.")

    assert (
        book_io.read_document(tmp_path, "theory/C.md") == "# Third\n\nRevised result."
    )
    assert book_io.read_book(tmp_path).chapters[1].body == sample_book.chapters[1].body


@pytest.mark.parametrize("route", ("theory/research/OLD_PLAN.md", "OLD_PORTFOLIO.txt"))
def test_generated_aliases_retain_anchors_and_reject_writes(
    tmp_path, sample_book, route
):
    book_io.write_book(tmp_path, sample_book)
    source = tmp_path / book_io.BOOK_PATH
    before = source.read_bytes()
    generated = book_io.read_document(tmp_path, route)

    assert "Historical task instructions are inactive." in generated
    assert book_io.document_exists(tmp_path, route)
    if route.endswith("OLD_PLAN.md"):
        assert '<a id="old-gate"></a>' in generated
        assert '<a id="old-result"></a>' in generated
        assert "../RESEARCH_PROGRAM.md#research-current-question" in generated
    else:
        assert "theory/RESEARCH_PROGRAM.md#research-current-question" in generated
    with pytest.raises(ValueError, match="generated"):
        book_io.write_document(tmp_path, route, "Overwritten history")
    assert source.read_bytes() == before
    assert not (tmp_path / route).exists()


def test_front_is_generated_navigation_with_stable_anchors_not_proof_bodies(
    tmp_path, sample_book
):
    book_io.write_book(tmp_path, sample_book)
    visible = book_io.read_document(tmp_path, book_io.BOOK_PATH)

    assert visible.startswith("# Program\n\nA bounded current question.")
    assert book_io.read_book(tmp_path).front == sample_book.front
    assert "SECRET PROOF BODY" not in visible
    assert "same-local-anchor" not in visible
    assert "TNFR CHAPTER" not in visible
    assert "[First result](A.md)" in visible
    assert "[Second result](nodal/B.md)" in visible
    assert '<a id="chapter-theory-nodal-b-md"></a>' in visible
    assert visible.count("### Dynamics") == 1

    renamed = replace(sample_book.chapters[1], title="Revised title", group="Other")
    book_io.write_book(
        tmp_path,
        replace(sample_book, chapters=(renamed, sample_book.chapters[0])),
    )
    revised = book_io.front(tmp_path)
    assert '<a id="chapter-theory-nodal-b-md"></a>' in revised
    assert "[Revised title](nodal/B.md)" in revised
    with pytest.raises(ValueError, match="Book.front"):
        book_io.write_document(tmp_path, book_io.BOOK_PATH, revised)


@pytest.mark.parametrize(
    "route", ("theory/A.md", "theory/research/OLD_PLAN.md", book_io.BOOK_PATH)
)
def test_logical_byte_budget_includes_generated_views(tmp_path, sample_book, route):
    book_io.write_book(tmp_path, sample_book)
    exact = book_io.read_document_bytes(tmp_path, route)

    assert book_io.read_document_bytes(tmp_path, route, max_bytes=len(exact)) == exact
    with pytest.raises(ValueError, match="byte budget"):
        book_io.read_document_bytes(tmp_path, route, max_bytes=len(exact) - 1)


@pytest.mark.parametrize("with_book", (False, True))
def test_real_file_fallback_reads_and_updates_existing_documents(
    tmp_path, sample_book, with_book
):
    if with_book:
        book_io.write_book(tmp_path, sample_book)
    target = tmp_path / "README.md"
    target.write_bytes(b"\xef\xbb\xbf# Root\r\nReal document.")

    assert book_io.read_document_bytes(tmp_path, target) == target.read_bytes()
    assert book_io.read_document(tmp_path, "README.md") == "# Root\nReal document."
    assert book_io.document_exists(tmp_path, target)
    if not with_book:
        assert book_io.documents(tmp_path) == {}
    book_io.write_document(tmp_path, "README.md", "# Root\nUpdated.\n")
    assert target.read_text(encoding="utf-8") == "# Root\nUpdated.\n"
    if with_book:
        assert book_io.read_book(tmp_path) == sample_book


def test_real_file_budget_rejects_before_reading_contents(tmp_path, monkeypatch):
    target = tmp_path / "README.md"
    target.write_bytes(b"A" * 100)
    real_read = Path.read_bytes

    def guarded_read(path):
        if path == target:
            pytest.fail("The size guard must run before reading a large real file")
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(ValueError, match="byte budget"):
        book_io.read_document_bytes(tmp_path, "README.md", max_bytes=99)


def test_logical_chapter_takes_precedence_over_stale_physical_copy(
    tmp_path, sample_book
):
    book_io.write_book(tmp_path, sample_book)
    stale = tmp_path / "theory/A.md"
    stale.write_bytes(b"obsolete parallel copy")

    assert book_io.read_document_bytes(tmp_path, stale) == sample_book.chapters[0].body
    book_io.write_document(tmp_path, stale, "# Corrected logical result")
    assert stale.read_bytes() == b"obsolete parallel copy"
    assert book_io.read_book(tmp_path).chapters[0].body == b"# Corrected logical result"


@pytest.mark.parametrize("absolute", (False, True))
def test_document_access_cannot_escape_repository(tmp_path, absolute):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("Preserved external document", encoding="utf-8")
    path = outside if absolute else "../outside.md"

    for operation in (book_io.read_document_bytes, book_io.document_exists):
        with pytest.raises(ValueError, match="outside repository"):
            operation(root, path)
    with pytest.raises(ValueError, match="outside repository"):
        book_io.write_document(root, path, "replacement")
    assert outside.read_text(encoding="utf-8") == "Preserved external document"


@pytest.mark.parametrize(
    "path",
    (
        "",
        "../OUTSIDE.md",
        "/absolute.md",
        "C:/absolute.md",
        "theory\\A.md",
        "./theory/A.md",
        "theory//A.md",
        "theory/../A.md",
        "theory/A.md/",
        book_io.BOOK_PATH,
        book_io.BOOK_PATH.lower(),
    ),
)
def test_unsafe_or_self_referential_logical_paths_reject(path):
    metadata = json.dumps({"path": path, "title": "A", "group": "Theory"}).encode()
    with pytest.raises(ValueError):
        book_io.parse_book(_wire(chapter=metadata))


@pytest.mark.parametrize(
    "routes",
    (
        b'{"version":2,"aliases":[]}',
        b'{"version":true,"aliases":[]}',
        b'{"version":1.0,"aliases":[]}',
        b'{"version":1,"aliases":{}}',
        b'{"version":1,"aliases":null}',
        b'{"version":1,"aliases":[null]}',
        b'{"version":1,"aliases":[],"extra":0}',
        b'{"version":1,"version":1,"aliases":[]}',
        b"[]",
        b'{"version":1,"aliases":[{"path":"OLD.md","title":3,"anchors":[]}]}',
        b'{"version":1,"aliases":[{"path":"OLD.md","title":"Old","anchors":[1]}]}',
        b'{"version":1,"aliases":[{"path":"OLD.md","title":"Old","anchors":["bad id"]}]}',
        b'{"version":1,"aliases":[{"path":"OLD.md","path":"OTHER.md","title":"Old","anchors":[]}]}',
    ),
)
def test_malformed_route_metadata_rejects_without_coercion(routes):
    with pytest.raises(ValueError):
        book_io.parse_book(_wire(routes=routes))


@pytest.mark.parametrize(
    "metadata",
    (
        b"[]",
        b"null",
        b'{"path":"theory/A.md","title":"A"}',
        b'{"path":"theory/A.md","title":"A","group":0}',
        b'{"path":"theory/A.md","title":"","group":"Theory"}',
        b'{"path":"theory/A.md","title":"A","group":"Theory","extra":1}',
        b'{"path":"theory/A.md","path":"theory/B.md","title":"A","group":"Theory"}',
    ),
)
def test_malformed_chapter_metadata_rejects(metadata):
    with pytest.raises(ValueError):
        book_io.parse_book(_wire(chapter=metadata))


@pytest.mark.parametrize(
    "data",
    (
        b"# No routes boundary",
        b'# Front\n<!-- TNFR ROUTES {"version":1,"aliases":[]}',
        _wire() + b"unowned trailing text",
        _wire().replace(b"\n<!-- END TNFR CHAPTER -->\n", b""),
        _wire().replace(b'"group":"Foundations"} -->\n', b'"group":"Foundations"}'),
        _wire(body=b"invalid UTF-8: \xff"),
        _wire(front=b"invalid UTF-8: \xff"),
        _wire(body=b'# Body\n<!-- TNFR CHAPTER {"path":"nested"} -->\nHidden'),
    ),
)
def test_malformed_boundaries_and_encoding_reject(data):
    with pytest.raises(ValueError):
        book_io.parse_book(data)


@pytest.mark.parametrize(
    "first,second,alias",
    (
        ("theory/A.md", "theory/A.md", False),
        ("theory/A.md", "theory/a.md", False),
        ("theory/A.md", "theory/A.md", True),
        ("theory/A.md", "theory/a.MD", True),
        ("theory/A_B.md", "theory/A-B.md", False),
        ("theory/A_B.md", "theory/A-B.md", True),
    ),
)
def test_duplicate_routes_and_colliding_landing_anchors_reject(first, second, alias):
    chapter = book_io.Chapter(first, "First", "Theory", b"# First")
    if alias:
        invalid = book_io.Book("# Front", (chapter,), (book_io.Alias(second, "Old"),))
    else:
        invalid = book_io.Book(
            "# Front",
            (chapter, book_io.Chapter(second, "Second", "Theory", b"# Second")),
        )
    with pytest.raises(ValueError, match="duplicate|collid"):
        book_io.render_book(invalid)


@pytest.mark.parametrize(
    "reserved",
    (b"\n<!-- TNFR CHAPTER ", b"\n<!-- END TNFR CHAPTER -->\n"),
)
def test_reserved_body_boundary_rejects_before_replacing_a_valid_book(
    tmp_path, sample_book, reserved
):
    book_io.write_book(tmp_path, sample_book)
    source = tmp_path / book_io.BOOK_PATH
    before = source.read_bytes()
    invalid = replace(
        sample_book,
        chapters=(replace(sample_book.chapters[0], body=b"Proof" + reserved),),
    )

    with pytest.raises(ValueError, match="reserved"):
        book_io.write_book(tmp_path, invalid)
    assert source.read_bytes() == before


def test_failed_atomic_replacement_keeps_the_previous_sourcebook(
    tmp_path, sample_book, monkeypatch
):
    book_io.write_book(tmp_path, sample_book)
    path = tmp_path / book_io.BOOK_PATH
    original = path.read_bytes()

    def fail_replace(source, destination):
        assert Path(source).parent == path.parent
        assert Path(destination) == path
        assert Path(source).read_bytes() != original
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(book_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement"):
        book_io.write_book(
            tmp_path, replace(sample_book, front="# Revised objective\n")
        )
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".research-book-*.tmp"))


def test_modular_migration_keeps_exact_chapters_and_small_program(
    tmp_path, sample_book
):
    book_io.write_book(tmp_path, sample_book)
    assert book_io.migrate_book(tmp_path) == sample_book

    for chapter in sample_book.chapters:
        assert (tmp_path / chapter.path).read_bytes() == chapter.body
    program = (tmp_path / book_io.BOOK_PATH).read_bytes()
    assert b"SECRET PROOF BODY" not in program
    assert b"chapter-theory-nodal-b-md" in program
    assert program == book_io.front(tmp_path).encode("utf-8")
    assert book_io.read_book(tmp_path).front == sample_book.front
    catalog = json.loads((tmp_path / book_io.CATALOG_PATH).read_bytes())
    assert catalog["version"] == 2
    assert all(set(item) == {"path", "title", "group"} for item in catalog["chapters"])
    assert b"SECRET PROOF BODY" not in (tmp_path / book_io.CATALOG_PATH).read_bytes()
    assert not (tmp_path / sample_book.aliases[0].path).exists()


def test_modular_migration_rejects_conflicting_work_before_any_write(
    tmp_path, sample_book
):
    book_io.write_book(tmp_path, sample_book)
    source = tmp_path / book_io.BOOK_PATH
    before = source.read_bytes()
    existing = tmp_path / sample_book.chapters[0].path
    existing.write_bytes(b"Independent uncommitted work")

    with pytest.raises(ValueError, match="conflicting physical"):
        book_io.migrate_book(tmp_path)

    assert source.read_bytes() == before
    assert existing.read_bytes() == b"Independent uncommitted work"
    assert not (tmp_path / book_io.CATALOG_PATH).exists()
    assert not (tmp_path / sample_book.chapters[1].path).exists()


def test_failed_catalog_switch_restores_readable_legacy_program(
    tmp_path, sample_book, monkeypatch
):
    book_io.write_book(tmp_path, sample_book)
    original = (tmp_path / book_io.BOOK_PATH).read_bytes()
    actual_replace = book_io.os.replace

    def fail_catalog(source, destination):
        if Path(destination).name == "research_catalog.json":
            raise OSError("catalog publish failed")
        actual_replace(source, destination)

    monkeypatch.setattr(book_io.os, "replace", fail_catalog)
    with pytest.raises(OSError, match="catalog publish failed"):
        book_io.migrate_book(tmp_path)

    assert (tmp_path / book_io.BOOK_PATH).read_bytes() == original
    assert not (tmp_path / book_io.CATALOG_PATH).exists()
    assert book_io.read_book(tmp_path) == sample_book
    # Exact extracted copies remain safe to retry; no uncommitted file is deleted.
    assert (
        tmp_path / sample_book.chapters[0].path
    ).read_bytes() == sample_book.chapters[0].body


def test_modular_reader_observes_direct_chapter_edits(tmp_path, sample_book):
    book_io.write_modular_book(tmp_path, sample_book)
    assert book_io.read_book(tmp_path) == sample_book
    chapter = tmp_path / sample_book.chapters[0].path
    revised = b"# First\n\nDirect editor correction.\n"
    chapter.write_bytes(revised)

    assert book_io.read_document_bytes(tmp_path, chapter) == revised
    assert book_io.read_book(tmp_path).chapters[0].body == revised
    assert book_io.documents(tmp_path)[sample_book.chapters[0].path] == revised


def test_modular_single_chapter_write_preserves_other_files(
    tmp_path, sample_book, monkeypatch
):
    book_io.write_modular_book(tmp_path, sample_book)
    program = (tmp_path / book_io.BOOK_PATH).read_bytes()
    catalog = (tmp_path / book_io.CATALOG_PATH).read_bytes()
    actual_replace = book_io.os.replace
    replacements = []

    def record_replace(source, destination):
        replacements.append(Path(destination).relative_to(tmp_path).as_posix())
        actual_replace(source, destination)

    monkeypatch.setattr(book_io.os, "replace", record_replace)
    book_io.write_document(tmp_path, "theory/A.md", "# First\n\nCorrected statement.\n")

    assert replacements == ["theory/A.md"]
    assert (tmp_path / book_io.BOOK_PATH).read_bytes() == program
    assert (tmp_path / book_io.CATALOG_PATH).read_bytes() == catalog
    assert (
        tmp_path / "theory/A.md"
    ).read_bytes() == b"# First\r\n\r\nCorrected statement.\r\n"
    for chapter in sample_book.chapters[1:]:
        assert (tmp_path / chapter.path).read_bytes() == chapter.body


def test_modular_book_update_avoids_unchanged_replacements(
    tmp_path, sample_book, monkeypatch
):
    book_io.write_modular_book(tmp_path, sample_book)
    actual_replace = book_io.os.replace
    replacements = []

    def record_replace(source, destination):
        replacements.append(Path(destination).relative_to(tmp_path).as_posix())
        actual_replace(source, destination)

    monkeypatch.setattr(book_io.os, "replace", record_replace)
    book_io.write_book(tmp_path, sample_book)
    assert replacements == []
    book_io.write_book(
        tmp_path, replace(sample_book, front="# Revised current question\n")
    )
    assert replacements == [book_io.BOOK_PATH]
    assert book_io.read_book(tmp_path).chapters == sample_book.chapters


def test_modular_failed_chapter_replacement_retains_original(
    tmp_path, sample_book, monkeypatch
):
    book_io.write_modular_book(tmp_path, sample_book)

    def fail_replace(source, destination):
        raise OSError("chapter replace failed")

    monkeypatch.setattr(book_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="chapter replace failed"):
        book_io.write_document(tmp_path, "theory/A.md", "# Proposed correction\n")

    assert book_io.read_book(tmp_path) == sample_book
    assert not list((tmp_path / "theory").glob(".research-book-*.tmp"))


def test_modular_aliases_are_generated_and_cannot_be_overwritten(tmp_path, sample_book):
    book_io.write_modular_book(tmp_path, sample_book)
    alias = sample_book.aliases[0]
    assert "old-gate" in book_io.read_document(tmp_path, alias.path)
    assert book_io.document_exists(tmp_path, alias.path)
    with pytest.raises(ValueError, match="generated"):
        book_io.write_document(tmp_path, alias.path, "Active competing plan")


def test_modular_invalid_proposal_is_rejected_before_any_write(tmp_path, sample_book):
    book_io.write_modular_book(tmp_path, sample_book)
    invalid = replace(
        sample_book,
        front="# Proposed revision\n",
        chapters=(
            *sample_book.chapters,
            replace(sample_book.chapters[0], path="../escape.md"),
        ),
    )
    with pytest.raises(ValueError):
        book_io.write_modular_book(tmp_path, invalid)
    assert book_io.read_book(tmp_path) == sample_book


@pytest.mark.parametrize("missing", (book_io.BOOK_PATH, "theory/A.md"))
def test_modular_missing_canonical_source_fails_closed(tmp_path, sample_book, missing):
    book_io.write_modular_book(tmp_path, sample_book)
    (tmp_path / missing).unlink()
    with pytest.raises(FileNotFoundError):
        book_io.documents(tmp_path)


def test_modular_byte_budget_rejects_chapter_before_read(
    tmp_path, sample_book, monkeypatch
):
    book_io.write_modular_book(tmp_path, sample_book)
    chapter = tmp_path / "theory/A.md"
    actual_read = Path.read_bytes

    def deny_chapter_read(path):
        if path == chapter:
            pytest.fail("Budget rejection must precede reading chapter content")
        return actual_read(path)

    monkeypatch.setattr(Path, "read_bytes", deny_chapter_read)
    with pytest.raises(ValueError, match="byte budget"):
        book_io.read_document_bytes(tmp_path, chapter, max_bytes=1)


@pytest.mark.parametrize("outside", (False, True))
def test_modular_canonical_chapter_rejects_symlink_route(
    tmp_path, sample_book, outside
):
    root = tmp_path / "repo"
    root.mkdir()
    book_io.write_modular_book(root, sample_book)
    chapter = root / "theory/A.md"
    target = (tmp_path if outside else root) / "external.md"
    target.write_bytes(chapter.read_bytes())
    chapter.unlink()
    try:
        chapter.symlink_to(target)
    except OSError:
        pytest.skip("Platform does not allow creating symbolic links")

    with pytest.raises(ValueError, match="symlinks|outside repository"):
        book_io.read_book(root)
    with pytest.raises(ValueError, match="symlinks|outside repository"):
        book_io.write_modular_book(root, sample_book)
    assert target.read_bytes() == sample_book.chapters[0].body


@pytest.mark.parametrize(
    "payload",
    (
        b'{"version":2,"version":2,"chapters":[],"aliases":[]}',
        b'{"version":true,"chapters":[],"aliases":[]}',
        b'{"version":2,"chapters":[],"aliases":[],"body":"duplicated"}',
        b'{"version":2,"chapters":[{"path":"../A.md","title":"A","group":"G"}],"aliases":[]}',
        b'{"version":2,"chapters":[{"path":"theory/A.md","title":"A","group":"G"},{"path":"theory/a.md","title":"B","group":"G"}],"aliases":[]}',
    ),
)
def test_catalog_admits_only_one_unambiguous_metadata_owner(payload):
    with pytest.raises(ValueError):
        book_io.parse_catalog(payload)


def test_generated_physical_aliases_are_compact_and_rebuildable(tmp_path, sample_book):
    book_io.write_modular_book(tmp_path, sample_book)
    assert book_io.compatibility_view_drift(tmp_path) == tuple(
        alias.path for alias in sample_book.aliases
    )
    before = book_io.read_book(tmp_path)

    book_io.write_compatibility_views(tmp_path)

    assert book_io.compatibility_view_drift(tmp_path) == ()
    assert book_io.read_book(tmp_path) == before
    for alias in sample_book.aliases:
        physical = (tmp_path / alias.path).read_bytes()
        assert physical == book_io.read_document_bytes(tmp_path, alias.path)
        assert b"Historical task instructions are inactive." in physical
        assert len(physical) < 1000
    stale = sample_book.aliases[0]
    (tmp_path / stale.path).write_bytes(b"Obsolete competing instruction")
    assert book_io.compatibility_view_drift(tmp_path) == (stale.path,)
    book_io.write_compatibility_views(tmp_path)
    assert book_io.compatibility_view_drift(tmp_path) == ()


def test_generated_alias_refresh_does_not_replace_unchanged_files(
    tmp_path, sample_book, monkeypatch
):
    book_io.write_modular_book(tmp_path, sample_book)
    book_io.write_compatibility_views(tmp_path)

    def forbidden_replace(source, destination):
        pytest.fail("An unchanged generated route must not be replaced")

    monkeypatch.setattr(book_io.os, "replace", forbidden_replace)
    book_io.write_compatibility_views(tmp_path)


def test_alias_directory_conflict_rejects_before_first_view_write(
    tmp_path, sample_book
):
    book_io.write_modular_book(tmp_path, sample_book)
    blocked = tmp_path / sample_book.aliases[-1].path
    blocked.mkdir()

    for operation in (
        book_io.write_compatibility_views,
        book_io.compatibility_view_drift,
    ):
        with pytest.raises(ValueError, match="not a file"):
            operation(tmp_path)

    assert not (tmp_path / sample_book.aliases[0].path).exists()
    assert blocked.is_dir()


def test_alias_symlink_admission_precedes_all_view_writes(
    tmp_path, sample_book, monkeypatch
):
    book_io.write_modular_book(tmp_path, sample_book)
    blocked = tmp_path / sample_book.aliases[-1].path
    actual_is_symlink = Path.is_symlink

    def simulate_link(path):
        return path == blocked or actual_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", simulate_link)
    for operation in (
        book_io.write_compatibility_views,
        book_io.compatibility_view_drift,
    ):
        with pytest.raises(ValueError, match="symlinks"):
            operation(tmp_path)

    assert not (tmp_path / sample_book.aliases[0].path).exists()


def test_legacy_aliases_remain_virtual_without_modular_catalog(tmp_path, sample_book):
    book_io.write_book(tmp_path, sample_book)
    assert book_io.compatibility_view_drift(tmp_path) == ()
    book_io.write_compatibility_views(tmp_path)
    assert all(not (tmp_path / alias.path).exists() for alias in sample_book.aliases)
