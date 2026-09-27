#!/usr/bin/env python3
"""Validate declared concept cards and their generated index without engine imports.

The glossary is the source; this checker validates structure, declared dependency
edges and local evidence references. It neither proves a claim nor promotes its
scientific status. Normal checks are read-only; index replacement is explicit.
"""

import importlib.util
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlsplit

CARDS_START = "<!-- BEGIN CONCEPT CARDS -->"
CARDS_END = "<!-- END CONCEPT CARDS -->"
INDEX_START = "<!-- BEGIN GENERATED CONCEPT INDEX -->"
INDEX_END = "<!-- END GENERATED CONCEPT INDEX -->"
FIELD_NAMES = (
    "Status",
    "Basis",
    "Emergence",
    "Definition",
    "Domain",
    "Premises",
    "Dependencies",
    "Owner",
    "Evidence",
    "Implementation",
    "Tests",
    "Limits",
)
_SINGLE_LINE = frozenset(("Status", "Basis", "Emergence", "Dependencies"))
_ENUMS = {
    "Status": {"maintained", "candidate"},
    "Basis": {
        "primitive",
        "identity",
        "constitutive",
        "derived",
        "diagnostic",
        "policy",
        "auxiliary",
        "hypothesis",
    },
    "Emergence": {"not-claimed", "conditional"},
}
_ID = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*"
_ANCHOR = re.compile(r'<a id="(' + _ID + r')"></a>')
_FIELD = re.compile(r"^(?:- )?\*\*([^*]+):\*\*[ \t]*(.*)$")
_PLACEHOLDER = re.compile(r"\b(?:TODO|TBD|FIXME|TBC)\b", re.I)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@dataclass(frozen=True)
class ConceptCard:
    """One immutable parsed declaration, without a scientific validity seal."""

    identifier: str
    title: str
    group: str
    fields: tuple[tuple[str, str], ...]

    def field(self, name: str) -> str:
        """Return a named field from the single retained declaration."""
        return dict(self.fields)[name]

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Read declared foundational edges, not related-reading links."""
        value = self.field("Dependencies")
        return (
            () if value == "none" else tuple(part.strip() for part in value.split(","))
        )


@lru_cache(maxsize=1)
def _references():
    path = Path(__file__).with_name("verify_internal_references.py")
    spec = importlib.util.spec_from_file_location("glossary_reference_syntax", path)
    _require(
        spec is not None and spec.loader is not None, "reference checker is unavailable"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _region(document: str, start_marker: str, end_marker: str) -> tuple[int, int]:
    starts, ends = [], []
    offset = 0
    for line, prose in _references()._markdown_lines(document):
        content = line.rstrip("\r\n")
        if prose and content == start_marker:
            starts.append(offset + len(content))
        if prose and content == end_marker:
            ends.append(offset)
        offset += len(line)
    _require(
        len(starts) == len(ends) == 1,
        f"glossary requires exactly one {start_marker} region",
    )
    start, end = starts[0], ends[0]
    _require(start < end, "glossary region markers are reversed")
    return start, end


def _parse_cards(document: str) -> tuple[ConceptCard, ...]:
    start, end = _region(document, CARDS_START, CARDS_END)
    index_start, index_end = _region(document, INDEX_START, INDEX_END)
    _require(
        end <= index_start or index_end <= start,
        "concept card and generated index regions must be disjoint",
    )
    cards = []
    group = identifier = title = None
    blank_after_anchor = False
    pairs = []

    def finish():
        nonlocal identifier, title, pairs
        if identifier is None:
            return
        _require(
            title is not None,
            f"concept {identifier}: anchor requires a following H3 title",
        )
        names = tuple(name for name, _ in pairs)
        _require(
            names == FIELD_NAMES,
            f"concept {identifier}: fields must occur exactly once in order {FIELD_NAMES}; got {names}",
        )
        cards.append(
            ConceptCard(
                identifier,
                title,
                group,
                tuple((name, "\n".join(lines).strip()) for name, lines in pairs),
            )
        )
        identifier, title, pairs = None, None, []

    for raw, prose in _references()._markdown_lines(document[start:end]):
        line = raw.rstrip("\r\n")
        if not line.strip():
            if identifier is not None and title is None:
                blank_after_anchor = True
            if pairs:
                pairs[-1][1].append("")
            continue
        heading = (
            re.fullmatch(r"##[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*", line) if prose else None
        )
        anchor = _ANCHOR.fullmatch(line) if prose else None
        if heading:
            finish()
            group = heading.group(1)
            continue
        if anchor:
            finish()
            _require(group is not None, "concept cards require an H2 group")
            identifier = anchor.group(1)
            blank_after_anchor = False
            continue
        heading = (
            re.fullmatch(r"###[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*", line) if prose else None
        )
        if heading:
            _require(
                identifier is not None and title is None and blank_after_anchor,
                "each concept requires an explicit ID anchor, a blank line and one H3 title",
            )
            title = heading.group(1)
            continue
        _require(
            identifier is not None and title is not None,
            "unexpected text outside a concept card or missing ID/title",
        )
        field = _FIELD.fullmatch(line) if prose else None
        if field:
            name, value = field.groups()
            if name in _SINGLE_LINE:
                _require(
                    bool(value.strip()),
                    f"concept {identifier}: {name} must have a single-line value",
                )
            pairs.append((name, [value]))
        else:
            _require(
                bool(pairs),
                f"concept {identifier}: content must belong to a named field",
            )
            pairs[-1][1].append(line.strip())
    finish()
    _require(bool(cards), "glossary must contain at least one concept card")
    return tuple(cards)


def _meaningful(value: str, label: str) -> None:
    normalized = " ".join(value.split()).strip("` .").lower()
    _require(bool(normalized), f"{label}: value must be nonempty")
    _require(
        not _PLACEHOLDER.search(value)
        and re.fullmatch(
            r"(?:<[^>]*(?:reason|description|placeholder|value|text|id)[^>]*>|\[insert\b.*\])",
            normalized,
        )
        is None
        and normalized
        not in {
            "none",
            "n/a",
            "na",
            "null",
            "unknown",
            "placeholder",
            "to be determined",
            "coming soon",
            "…",
        },
        f"{label}: placeholder is not evidence or an explanation",
    )


def _local_links(value: str, source: Path, root: Path, label: str):
    references = _references()
    prose = "".join(
        text if rendered else ""
        for text, rendered in references._markdown_segments(value)
    )
    links = []
    for _, raw in references.LINK_PATTERN.findall(prose):
        target = references._target(raw)
        repository_target = references._repository_url(target)
        if repository_target is not None:
            target, base = repository_target, root
        else:
            _require(
                not references._is_external(target),
                f"{label}: requires local evidence links",
            )
            base = source.parent
        parsed = urlsplit(target)
        path = (
            (base / unquote(parsed.path).replace("\\", "/")).resolve()
            if parsed.path
            else source
        )
        _require(
            path.is_relative_to(root), f"{label}: link leaves the repository: {target}"
        )
        _require(path.is_file(), f"{label}: missing local file: {target}")
        fragment = unquote(parsed.fragment)
        if fragment and path.suffix.lower() == ".md":
            _require(
                fragment in references._anchors(path),
                f"{label}: missing fragment: {target}",
            )
        links.append((path, fragment))
    return tuple(links)


def _archived(path: Path, root: Path) -> bool:
    return bool(
        {"archive", "archives"}
        & {part.lower() for part in path.relative_to(root).parts}
    )


def _under(path: Path, root: Path, directory: str) -> bool:
    return path.is_relative_to(root / directory)


def _validate_card(card: ConceptCard, source: Path, root: Path) -> None:
    for name, value in card.fields:
        label = f"concept {card.identifier} {name}"
        _require(bool(value.strip()), f"{label}: value must be nonempty")
        if name in _SINGLE_LINE:
            _require("\n" not in value, f"{label}: requires a single-line value")
        if name != "Dependencies":
            _meaningful(value, label)
        if name in _ENUMS:
            _require(value in _ENUMS[name], f"{label}: unknown value {value!r}")
    basis, emergence, status = (
        card.field(name) for name in ("Basis", "Emergence", "Status")
    )
    _require(
        emergence != "conditional" or basis == "derived",
        f"concept {card.identifier}: conditional emergence requires derived basis",
    )
    _require(
        basis != "hypothesis" or status == "candidate",
        f"concept {card.identifier}: hypothesis requires candidate status",
    )
    dependencies = card.field("Dependencies")
    _require(
        dependencies == "none"
        or re.fullmatch(_ID + r"(?:[ \t]*,[ \t]*" + _ID + r")*", dependencies)
        is not None,
        f"concept {card.identifier}: Dependencies must be comma-separated IDs or none",
    )
    _require(
        len(card.dependencies) == len(set(card.dependencies)),
        f"concept {card.identifier}: duplicate dependencies",
    )
    _require(
        basis != "derived" or bool(card.dependencies),
        f"concept {card.identifier}: derived concepts require at least one dependency",
    )

    owners = _local_links(
        card.field("Owner"), source, root, f"concept {card.identifier} Owner"
    )
    _require(
        any(
            path != source.resolve()
            and path.suffix.lower() == ".md"
            and not _archived(path, root)
            and (_under(path, root, "theory") or _under(path, root, "docs"))
            for path, _ in owners
        ),
        f"concept {card.identifier}: Owner needs a maintained theory/docs Markdown link",
    )
    evidence = _local_links(
        card.field("Evidence"), source, root, f"concept {card.identifier} Evidence"
    )
    _require(
        any(not _archived(path, root) for path, _ in evidence),
        f"concept {card.identifier}: Evidence needs a non-archive local link",
    )
    if basis == "derived":
        _require(
            any(
                fragment
                and path != source.resolve()
                and path.suffix.lower() == ".md"
                and _under(path, root, "theory")
                and not _archived(path, root)
                for path, fragment in evidence
            ),
            f"concept {card.identifier}: derived evidence requires a maintained theory derivation anchor",
        )

    linked = {}
    for name, directory in (("Implementation", "src"), ("Tests", "tests")):
        value = card.field(name)
        label = f"concept {card.identifier} {name}"
        if value.startswith("none:"):
            _meaningful(value[len("none:") :], label + " reason")
            linked[name] = False
            continue
        links = _local_links(value, source, root, label)
        linked[name] = any(
            _under(path, root, directory)
            and (
                (
                    path.suffix == ".py"
                    and (
                        path.name.startswith("test_") or path.name.endswith("_test.py")
                    )
                )
                if name == "Tests"
                else path.suffix.lower() in {".py", ".pyi"}
            )
            for path, _ in links
        )
        _require(
            linked[name],
            f"{label}: needs a {directory}/ {'test Python' if name == 'Tests' else 'code'} file link or none: <reason>",
        )
    _require(
        status != "maintained" or not linked["Implementation"] or linked["Tests"],
        f"concept {card.identifier}: maintained implemented concepts require a test link",
    )


def load_cards(root: Path) -> tuple[ConceptCard, ...]:
    """Validate card structure, status boundaries, references and dependency DAG."""
    root = Path(root).resolve()
    source = root / "theory" / "GLOSSARY.md"
    cards = _parse_cards(source.read_text(encoding="utf-8"))
    identifiers = {card.identifier for card in cards}
    _require(len(identifiers) == len(cards), "glossary contains duplicate concept IDs")
    _require(
        len({" ".join(card.title.split()).casefold() for card in cards}) == len(cards),
        "glossary contains duplicate concept titles",
    )
    for card in cards:
        _validate_card(card, source, root)
        unknown = set(card.dependencies) - identifiers
        _require(
            not unknown,
            f"concept {card.identifier}: unknown dependencies {sorted(unknown)}",
        )
    pending = {card.identifier: set(card.dependencies) for card in cards}
    completed = set()
    while pending:
        ready = {
            identifier
            for identifier, dependencies in pending.items()
            if dependencies <= completed
        }
        _require(bool(ready), "concept dependency cycle: " + ", ".join(sorted(pending)))
        completed.update(ready)
        for identifier in ready:
            del pending[identifier]
    return cards


def render_concept_index(cards: tuple[ConceptCard, ...]) -> str:
    """Render the compact index in source order without changing card status."""
    rows = ["| Concept | Basis | Emergence | Status |", "| --- | --- | --- | --- |"]
    for card in cards:
        title = (
            card.title.replace("|", "&#124;")
            .replace("[", "&#91;")
            .replace("]", "&#93;")
        )
        rows.append(
            f"| [{title}](#{card.identifier}) | {card.field('Basis')} | {card.field('Emergence')} | {card.field('Status')} |"
        )
    return "\n".join(rows)


def check_glossary(root: Path) -> None:
    """Check declarations and exact generated-index text without writing files."""
    root = Path(root).resolve()
    cards = load_cards(root)
    document = (root / "theory" / "GLOSSARY.md").read_text(encoding="utf-8")
    start, end = _region(document, INDEX_START, INDEX_END)
    expected = "\n\n" + render_concept_index(cards) + "\n\n"
    _require(
        document[start:end] == expected,
        "concept index drifted; run scripts/check_documentation.py --write-generated",
    )


def update_concept_index(root: Path) -> None:
    """Replace only the generated index after validating the source cards."""
    root = Path(root).resolve()
    cards = load_cards(root)
    path = root / "theory" / "GLOSSARY.md"
    document = path.read_text(encoding="utf-8")
    start, end = _region(document, INDEX_START, INDEX_END)
    path.write_text(
        document[:start]
        + "\n\n"
        + render_concept_index(cards)
        + "\n\n"
        + document[end:],
        encoding="utf-8",
    )
