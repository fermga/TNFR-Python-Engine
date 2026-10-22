"""Read the research program and its uniquely owned Markdown chapters.

The current layout uses a small program, physical chapters and a metadata-only
catalog. Legacy sourcebook parsing remains available for historical captures.
Both layouts retain chapter bytes, paths, link contexts and evidence associations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path, PurePosixPath

BOOK_PATH = "theory/RESEARCH_PROGRAM.md"
CATALOG_PATH = "theory/research_catalog.json"
_START = b"\n<!-- TNFR CHAPTER "
_END = b"\n<!-- END TNFR CHAPTER -->\n"
_ALIASES = b"\n<!-- TNFR ROUTES "
_ROUTES_START = b"\n<!-- BEGIN RESEARCH CHAPTER ROUTES -->\n"
_ROUTES_END = b"<!-- END RESEARCH CHAPTER ROUTES -->\n"


@dataclass(frozen=True)
class Chapter:
    path: str
    title: str
    group: str
    body: bytes


@dataclass(frozen=True)
class Alias:
    path: str
    title: str
    anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class Book:
    front: str
    chapters: tuple[Chapter, ...]
    aliases: tuple[Alias, ...] = ()


def chapter_id(path: str) -> str:
    """Stable sourcebook landing anchor; chapter-local IDs stay independent."""
    return "chapter-" + re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")


def _name(path: str) -> str:
    if not isinstance(path, str) or not path or "\\" in path or ":" in path:
        raise ValueError("research path must be repository-relative POSIX text")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("research path must not escape or alias the repository")
    if path.casefold() in {BOOK_PATH.casefold(), CATALOG_PATH.casefold()}:
        raise ValueError("research documents cannot contain their program or catalog")
    return path


def _json(data: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate research metadata key")
            result[key] = value
        return result

    result = json.loads(data, object_pairs_hook=pairs)
    if not isinstance(result, dict):
        raise ValueError("expected research metadata object")
    return result


def parse_book(data: bytes) -> Book:
    """Admit boundaries, unique routes and UTF-8 without changing body bytes."""
    boundary = data.find(_ALIASES)
    if boundary < 0:
        raise ValueError("sourcebook is missing its TNFR ROUTES boundary")
    front_text = data[:boundary].decode("utf-8")
    tail = data[boundary + len(_ALIASES) :]
    close = tail.find(b" -->\n")
    if close < 0:
        raise ValueError("unterminated research routes")
    metadata = _json(tail[:close])
    if (
        set(metadata) != {"version", "aliases"}
        or type(metadata["version"]) is not int
        or metadata["version"] != 1
        or not isinstance(metadata["aliases"], list)
    ):
        raise ValueError("unsupported research book schema")
    aliases = []
    for item in metadata["aliases"]:
        if not isinstance(item, dict) or set(item) != {"path", "title", "anchors"}:
            raise ValueError("invalid research alias")
        if not isinstance(item["title"], str) or not isinstance(item["anchors"], list):
            raise ValueError("invalid research alias values")
        if any(
            not isinstance(a, str) or not re.fullmatch(r"[\w.-]+", a)
            for a in item["anchors"]
        ):
            raise ValueError("invalid compatibility anchor")
        aliases.append(
            Alias(_name(item["path"]), item["title"], tuple(item["anchors"]))
        )
    tail = tail[close + len(b" -->\n") :]
    chapters = []
    while tail:
        if not tail.startswith(_START):
            raise ValueError("unexpected text outside a research chapter")
        tail = tail[len(_START) :]
        close = tail.find(b" -->\n")
        if close < 0:
            raise ValueError("unterminated research chapter metadata")
        item = _json(tail[:close])
        if set(item) != {"path", "title", "group"} or any(
            not isinstance(v, str) or not v for v in item.values()
        ):
            raise ValueError("invalid research chapter metadata")
        tail = tail[close + len(b" -->\n") :]
        end = tail.find(_END)
        if end < 0:
            raise ValueError("unterminated research chapter")
        body = tail[:end]
        wrapper = f'<a id="{chapter_id(item["path"])}"></a>\n\n'.encode("utf-8")
        if body.startswith(wrapper):
            body = body[len(wrapper) :]
        if _START in body or _ALIASES in body:
            raise ValueError("nested research boundary in body")
        if not item["path"].startswith("theory/") or not item["path"].endswith(".md"):
            raise ValueError("technical chapters require a theory Markdown path")
        body.decode("utf-8")
        chapters.append(
            Chapter(_name(item["path"]), item["title"], item["group"], body)
        )
        tail = tail[end + len(_END) :]
    names = [item.path for item in (*chapters, *aliases)]
    if len(set(n.casefold() for n in names)) != len(names):
        raise ValueError("duplicate research document route")
    ids = [chapter_id(n) for n in names]
    if len(set(ids)) != len(ids):
        raise ValueError("colliding research chapter anchors")
    return Book(front_text, tuple(chapters), tuple(aliases))


def render_book(book: Book) -> bytes:
    """Serialize and re-admit a book, preserving every chapter body verbatim."""
    aliases = [
        {"path": a.path, "title": a.title, "anchors": list(a.anchors)}
        for a in book.aliases
    ]
    data = book.front.encode("utf-8") + _ALIASES
    data += (
        json.dumps({"version": 1, "aliases": aliases}, ensure_ascii=False).encode(
            "utf-8"
        )
        + b" -->\n"
    )
    for chapter in book.chapters:
        if _END in chapter.body or _START in chapter.body:
            raise ValueError("reserved research chapter boundary in body")
        metadata = {
            "path": chapter.path,
            "title": chapter.title,
            "group": chapter.group,
        }
        data += (
            _START
            + json.dumps(metadata, ensure_ascii=False).encode("utf-8")
            + b" -->\n"
        )
        data += f'<a id="{chapter_id(chapter.path)}"></a>\n\n'.encode("utf-8")
        data += chapter.body + _END
    if parse_book(data) != book:
        raise ValueError("sourcebook does not round-trip")
    return data


def parse_catalog(data: bytes) -> Book:
    """Admit metadata without consulting a working tree or supplying body bytes."""
    metadata = _json(data)
    if (
        set(metadata) != {"version", "chapters", "aliases"}
        or type(metadata["version"]) is not int
        or metadata["version"] != 2
        or not isinstance(metadata["chapters"], list)
        or not isinstance(metadata["aliases"], list)
    ):
        raise ValueError("unsupported research catalog schema")
    chapters = []
    for item in metadata["chapters"]:
        if not isinstance(item, dict) or set(item) != {"path", "title", "group"}:
            raise ValueError("invalid research chapter metadata")
        if any(not isinstance(value, str) or not value for value in item.values()):
            raise ValueError("invalid research chapter metadata values")
        chapters.append(Chapter(item["path"], item["title"], item["group"], b""))
    aliases = []
    for item in metadata["aliases"]:
        if not isinstance(item, dict) or set(item) != {"path", "title", "anchors"}:
            raise ValueError("invalid research alias")
        if not isinstance(item["anchors"], list):
            raise ValueError("invalid research alias anchors")
        aliases.append(Alias(item["path"], item["title"], tuple(item["anchors"])))
    result = Book("", tuple(chapters), tuple(aliases))
    # Reuse the strict path, UTF-8, title, alias and uniqueness admission shared
    # with legacy captures, instead of maintaining a competing metadata schema.
    render_book(result)
    return result


def render_catalog(book: Book) -> bytes:
    """Serialize ordering and route metadata; never duplicate chapter content."""
    data = (
        json.dumps(
            {
                "version": 2,
                "chapters": [
                    {"path": c.path, "title": c.title, "group": c.group}
                    for c in book.chapters
                ],
                "aliases": [
                    {"path": a.path, "title": a.title, "anchors": list(a.anchors)}
                    for a in book.aliases
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    parse_catalog(data)
    return data


def _navigation(book: Book) -> str:
    lines = ["", "## Technical chapters", ""]
    group = None
    for chapter in book.chapters:
        if chapter.group != group:
            group = chapter.group
            lines += [f"### {group}", ""]
        relative = Path(chapter.path).relative_to("theory").as_posix()
        lines += [
            f'<a id="{chapter_id(chapter.path)}"></a>',
            f"- [{chapter.title}]({relative})",
            "",
        ]
    return "\n".join(lines)


def render_program(book: Book) -> bytes:
    """Render the small program with generated bridges for published anchors."""
    data = book.front.encode("utf-8")
    if _ROUTES_START in data or _ROUTES_END in data or _ALIASES in data:
        raise ValueError("authored program contains reserved research boundaries")
    return data + _ROUTES_START + _navigation(book).encode("utf-8") + _ROUTES_END


def _authored_front(data: bytes) -> str:
    if data.count(_ROUTES_START) != 1 or data.count(_ROUTES_END) != 1:
        raise ValueError("program requires one generated research routes region")
    start = data.index(_ROUTES_START)
    end = data.index(_ROUTES_END)
    if end <= start or end + len(_ROUTES_END) != len(data):
        raise ValueError("invalid generated research routes boundaries")
    return data[:start].decode("utf-8")


def is_modular(root: Path) -> bool:
    """Identify the catalog layout, including invalid entries needing rejection."""
    path = Path(root) / CATALOG_PATH
    return path.exists() or path.is_symlink()


def _safe_source(root: Path, name: str) -> Path:
    base = Path(root).resolve()
    candidate = base / name
    if not candidate.resolve().is_relative_to(base):
        raise ValueError("research source is outside repository")
    current = base
    for part in PurePosixPath(name).parts:
        current = current / part
        if current.is_symlink() or (
            hasattr(current, "is_junction") and current.is_junction()
        ):
            raise ValueError("research source cannot use symlinks or junctions")
    return candidate


@lru_cache(maxsize=512)
def _cached_bytes(path: str, modified: int, changed: int, size: int) -> bytes:
    return Path(path).read_bytes()


def _source_bytes(root: Path, name: str) -> bytes:
    path = _safe_source(root, name)
    stat = path.stat()
    return _cached_bytes(str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _catalog(data: bytes) -> Book:
    return parse_catalog(data)


def _metadata(root: Path) -> Book:
    return _catalog(_source_bytes(root, CATALOG_PATH))


@lru_cache(maxsize=4)
def _cached_book(path: str, modified: int, size: int) -> Book:
    return parse_book(Path(path).read_bytes())


def read_book(root: Path) -> Book:
    if is_modular(root):
        metadata = _metadata(root)
        front_text = _authored_front(_source_bytes(root, BOOK_PATH))
        chapters = []
        for chapter in metadata.chapters:
            body = _source_bytes(root, chapter.path)
            body.decode("utf-8")
            chapters.append(replace(chapter, body=body))
        return replace(metadata, front=front_text, chapters=tuple(chapters))
    path = Path(root).resolve() / BOOK_PATH
    stat = path.stat()
    return _cached_book(str(path), stat.st_mtime_ns, stat.st_size)


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".research-book-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    _cached_bytes.cache_clear()
    _cached_book.cache_clear()


def write_modular_book(root: Path, book: Book) -> None:
    """Write admitted documents atomically, leaving unchanged files untouched.

    Validation covers the complete proposed book and every target before the
    first write. Each file replacement is atomic; this is not a multi-file
    filesystem transaction. Removing a catalog entry does not delete its file.
    """
    render_book(book)
    program = render_program(book)
    catalog = render_catalog(book)
    payloads = [
        *((chapter.path, chapter.body) for chapter in book.chapters),
        (BOOK_PATH, program),
        (CATALOG_PATH, catalog),
    ]
    targets = [(_safe_source(root, name), data) for name, data in payloads]
    for alias in book.aliases:
        _safe_source(root, alias.path)
    for target, _ in targets:
        if target.exists() and not target.is_file():
            raise ValueError(f"research document target is not a file: {target}")
    for target, data in targets:
        _atomic_write(target, data)


def migrate_book(root: Path, book: Book | None = None) -> Book:
    """Externalize legacy chapters after rejecting conflicting physical copies.

    The catalog is published last. If that switch fails, restore the legacy
    program; already extracted exact chapter copies remain available for retry.
    """
    if is_modular(root):
        raise ValueError("research catalog already exists")
    proposed = read_book(root) if book is None else book
    for chapter in proposed.chapters:
        target = _safe_source(root, chapter.path)
        if target.exists() and target.read_bytes() != chapter.body:
            raise ValueError(f"conflicting physical research chapter: {chapter.path}")
    program = _safe_source(root, BOOK_PATH)
    previous = program.read_bytes() if program.exists() else None
    try:
        write_modular_book(root, proposed)
    except Exception:
        if not is_modular(root):
            if previous is None:
                program.unlink(missing_ok=True)
            else:
                _atomic_write(program, previous)
        raise
    return read_book(root)


def write_book(root: Path, book: Book) -> None:
    if is_modular(root):
        write_modular_book(root, book)
    else:
        _atomic_write(Path(root).resolve() / BOOK_PATH, render_book(book))


def _relative(root: Path, path: Path | str) -> str:
    base = Path(root).resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError("document is outside repository")
    return resolved.relative_to(base).as_posix()


def _alias_body(alias: Alias) -> bytes:
    relative = Path(
        os.path.relpath(BOOK_PATH, str(PurePosixPath(alias.path).parent))
    ).as_posix()
    anchors = "\n".join(f'<a id="{a}"></a>' for a in alias.anchors)
    return (
        f"# {alias.title}\n\nThis retired route is a generated compatibility view.\n"
        f"The [TNFR research program]({relative}#research-objective) owns the objective,\n"
        "current decision and admission criteria. Historical task instructions are inactive.\n\n"
        f"{anchors}\n\nSee the [current research question]({relative}#research-current-question).\n"
    ).encode("utf-8")


def _compatibility_targets(root: Path) -> list[tuple[str, Path, bytes]]:
    if not is_modular(root):
        return []
    targets = [
        (alias.path, _safe_source(root, alias.path), _alias_body(alias))
        for alias in _metadata(root).aliases
    ]
    for name, target, _ in targets:
        if target.exists() and not target.is_file():
            raise ValueError(f"compatibility view target is not a file: {name}")
    return targets


def write_compatibility_views(root: Path) -> None:
    """Regenerate compact inactive routes after admitting every target path."""
    for _, target, body in _compatibility_targets(root):
        _atomic_write(target, body)


def compatibility_view_drift(root: Path) -> tuple[str, ...]:
    """List missing or stale generated routes, rejecting unsafe physical paths."""
    return tuple(
        name
        for name, target, body in _compatibility_targets(root)
        if not target.is_file() or target.read_bytes() != body
    )


def documents(root: Path) -> dict[str, bytes]:
    """Logical chapter/compatibility bodies; no-book workspaces return empty."""
    if not (Path(root) / BOOK_PATH).is_file() and not is_modular(root):
        return {}
    book = read_book(root)
    return {
        **{c.path: c.body for c in book.chapters},
        **{a.path: _alias_body(a) for a in book.aliases},
    }


def front(root: Path) -> str:
    """Navigable front used by the site instead of rendering all proofs at once."""
    if is_modular(root):
        book = replace(
            _metadata(root), front=_authored_front(_source_bytes(root, BOOK_PATH))
        )
        return render_program(book).decode("utf-8")
    book = read_book(root)
    return book.front.rstrip() + "\n" + _navigation(book)


def read_document_bytes(
    root: Path, path: Path | str, *, max_bytes: int | None = None
) -> bytes:
    name = _relative(root, path)
    if name == BOOK_PATH and (Path(root) / BOOK_PATH).is_file():
        data = front(root).encode("utf-8")
    elif is_modular(root):
        metadata = _metadata(root)
        alias = next((a for a in metadata.aliases if a.path == name), None)
        if alias is not None:
            data = _alias_body(alias)
        else:
            source = _safe_source(root, name)
            if max_bytes is not None and source.stat().st_size > max_bytes:
                raise ValueError("document exceeds byte budget")
            data = _source_bytes(root, name)
    else:
        logical = documents(root)
        if name in logical:
            data = logical[name]
        else:
            source = Path(root) / name
            if max_bytes is not None and source.stat().st_size > max_bytes:
                raise ValueError("document exceeds byte budget")
            data = source.read_bytes()
    if max_bytes is not None and len(data) > max_bytes:
        raise ValueError("document exceeds byte budget")
    return data


def read_document(root: Path, path: Path | str) -> str:
    return read_document_bytes(root, path).decode("utf-8-sig").replace("\r\n", "\n")


def document_exists(root: Path, path: Path | str) -> bool:
    name = _relative(root, path)
    if is_modular(root):
        metadata = _metadata(root)
        if any(a.path == name for a in metadata.aliases):
            return True
        return _safe_source(root, name).exists()
    return name in documents(root) or (Path(root) / name).exists()


def write_document(root: Path, path: Path | str, text: str) -> None:
    name = _relative(root, path)
    if is_modular(root):
        metadata = _metadata(root)
        if name == BOOK_PATH:
            raise ValueError(
                "edit the front through Book.front; rendered navigation is generated"
            )
        if any(a.path == name for a in metadata.aliases):
            raise ValueError(
                "compatibility views are generated; edit the research program"
            )
        chapter = next((c for c in metadata.chapters if c.path == name), None)
        if chapter is not None:
            original = _source_bytes(root, name)
            normalized = text.replace("\r\n", "\n")
            if original.decode("utf-8-sig").replace("\r\n", "\n") == normalized:
                return
            newline = "\r\n" if b"\r\n" in original else "\n"
            body = normalized.replace("\n", newline).encode("utf-8")
            render_book(Book("", (replace(chapter, body=body),)))
            _atomic_write(_safe_source(root, name), body)
            return
        _atomic_write(_safe_source(root, name), text.encode("utf-8"))
        return
    if (Path(root) / BOOK_PATH).is_file():
        book = read_book(root)
        if name == BOOK_PATH:
            raise ValueError(
                "edit the front through Book.front; rendered navigation is generated"
            )
        for chapter in book.chapters:
            if chapter.path == name:
                if read_document(root, name) == text.replace("\r\n", "\n"):
                    return
                newline = "\r\n" if b"\r\n" in chapter.body else "\n"
                body = text.replace("\r\n", "\n").replace("\n", newline).encode("utf-8")
                write_book(
                    root,
                    replace(
                        book,
                        chapters=tuple(
                            replace(c, body=body) if c.path == name else c
                            for c in book.chapters
                        ),
                    ),
                )
                return
        if any(a.path == name for a in book.aliases):
            raise ValueError(
                "compatibility views are generated; edit the research program"
            )
    (Path(root) / name).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "extract", "replace"))
    parser.add_argument("path", nargs="?")
    parser.add_argument(
        "--file", type=Path, help="working Markdown file to extract or replace"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    if args.command == "list":
        for chapter in read_book(args.root).chapters:
            print(f"{chapter.path}\t{chapter.group}\t{chapter.title}")
        return
    if not args.path or args.file is None:
        parser.error("extract/replace need a logical path and --file")
    target = args.file.resolve()
    if args.command == "extract":
        if target.exists():
            parser.error("extraction target already exists")
        target.write_bytes(read_document_bytes(args.root, args.path))
    else:
        write_document(args.root, args.path, target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
