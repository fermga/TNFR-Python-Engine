#!/usr/bin/env python3
"""Check executable and single-source assumptions in TNFR documentation."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import re
import sys
from collections import Counter
from contextlib import redirect_stdout
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
# The path may already follow site-packages (for example through PYTHONPATH).
# Documentation must consume this checkout, never an older installed release.
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))


def require(condition: bool, message: str) -> None:
    """Keep documentation gates active even under Python optimization."""
    if not condition:
        raise RuntimeError(message)


def _project_version() -> str:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if match is None:
        raise AssertionError("project version missing from pyproject.toml")
    return match.group(1)


def check_agent_mirror() -> None:
    canonical = (REPO_ROOT / "AGENTS.md").read_bytes()
    mirror = (REPO_ROOT / ".github/agents/my-agent.md").read_bytes()
    require(canonical == mirror, "canonical agent mirror has drifted")


def check_versions_and_retired_claims() -> None:
    version = _project_version()
    for relative in ("README.md", "ARCHITECTURE.md", "AGENTS.md"):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        require(version in text, f"{relative} does not identify version {version}")

    current_docs = (
        "README.md",
        "ARCHITECTURE.md",
        "docs/README.md",
        ".github/WORKFLOWS.md",
        "theory/README.md",
        *(
            path.relative_to(REPO_ROOT.resolve()).as_posix()
            for path in sorted(_catalog_paths("docs"))
        ),
    )
    retired = (
        "1,633 tests",
        "2,448 tests",
        "all possible coherent transformations",
        "complete system observability",
        "complete integrability",
        "Grammar Inevitability",
    )
    for relative in current_docs:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for phrase in retired:
            require(phrase not in text, f"retired claim in {relative}: {phrase}")


class _PublicationLinks(HTMLParser):
    """Read real anchor targets; literal URLs and escaped HTML are not links."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        targets = [value for name, value in attrs if name == "href"]
        if not targets:
            return
        require(
            len(targets) == 1 and isinstance(targets[0], str) and bool(targets[0]),
            "Zenodo description has an invalid or ambiguous anchor href",
        )
        self.hrefs.append(targets[0])


def check_publication_metadata() -> None:
    """Keep distribution, citation and archive metadata on the reviewed version."""
    from tnfr.utils.io import json_loads

    version = _project_version()
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    archive = json_loads((REPO_ROOT / ".zenodo.json").read_bytes())
    require(archive["version"] == version, "Zenodo version differs from project")
    for key, expected in (
        ("version", version),
        ("date-released", archive["publication_date"]),
    ):
        match = re.search(rf'^{key}:\s*"([^\"]+)"\s*$', citation, re.MULTILINE)
        require(
            match is not None and match.group(1) == expected,
            f"citation {key} differs from publication metadata",
        )
    date.fromisoformat(archive["publication_date"])
    tree = ast.parse((REPO_ROOT / "src/tnfr/_version.py").read_text(encoding="utf-8"))
    fallback = next(
        (
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
        ),
        None,
    )
    require(fallback == version, "source fallback version differs from project")
    require(
        f"releases/tag/v{version}" in citation, "citation lacks the exact release URL"
    )
    repository = "https://github.com/fermga/TNFR-Python-Engine"
    pypi_project = "https://pypi.org/project/tnfr/"
    release_url = f"{repository}/releases/tag/v{version}"
    source_url = f"{repository}/tree/v{version}"
    pypi_url = f"{pypi_project}{version}/"
    description = archive.get("description")
    require(isinstance(description, str), "Zenodo description must contain HTML links")
    links = _PublicationLinks()
    links.feed(description)
    links.close()
    for label, expected in (
        ("release", release_url),
        ("versioned source", source_url),
        ("PyPI", pypi_url),
    ):
        require(
            expected in links.hrefs,
            f"Zenodo description lacks a valid {label} href for project version",
        )
    for target in links.hrefs:
        if target.startswith(f"{repository}/releases/tag/"):
            valid = target == release_url
        elif target.startswith(f"{repository}/tree/"):
            valid = target == source_url or target.startswith(source_url + "/")
        elif target.startswith(f"{repository}/blob/v"):
            valid = target.startswith(f"{repository}/blob/v{version}/")
        elif target.startswith(pypi_project):
            valid = target == pypi_url
        else:
            continue
        require(valid, "Zenodo description links to a different publication version")
    identifiers = archive.get("related_identifiers", [])
    require(
        isinstance(identifiers, list)
        and all(isinstance(item, dict) for item in identifiers),
        "Zenodo related identifiers must be a list of records",
    )
    pypi_identifiers = [
        item["identifier"]
        for item in identifiers
        if isinstance(item.get("identifier"), str)
        and item["identifier"].startswith(pypi_project)
    ]
    require(
        pypi_identifiers == [pypi_url],
        "Zenodo related PyPI identifier differs from project version",
    )


CONTRACT_START = "<!-- BEGIN GENERATED OPERATOR CONTRACTS -->"
CONTRACT_END = "<!-- END GENERATED OPERATOR CONTRACTS -->"
THEORY_CATALOG_START = "<!-- BEGIN THEORY CATALOG -->"
THEORY_CATALOG_END = "<!-- END THEORY CATALOG -->"
THEORY_NAVIGATION_START = "# BEGIN GENERATED THEORY NAVIGATION"
THEORY_NAVIGATION_END = "# END GENERATED THEORY NAVIGATION"
DOCS_CATALOG_START = "<!-- BEGIN DOCS CATALOG -->"
DOCS_CATALOG_END = "<!-- END DOCS CATALOG -->"
DOCS_NAVIGATION_START = "# BEGIN GENERATED DOCS NAVIGATION"
DOCS_NAVIGATION_END = "# END GENERATED DOCS NAVIGATION"
MAX_DOCUMENTATION_OWNER_LINES = 4_000


def _catalog_entries(
    directory: str, catalog_start: str, catalog_end: str
) -> list[tuple[str, str, Path]]:
    """Read H2 groups and the first inline link of each primary table row."""
    index = REPO_ROOT / directory / "README.md"
    document = index.read_text(encoding="utf-8")
    require(
        document.count(catalog_start) == 1 and document.count(catalog_end) == 1,
        f"{directory} index must have exactly one primary catalog region",
    )
    start = document.index(catalog_start) + len(catalog_start)
    end = document.index(catalog_end)
    require(start < end, f"{directory} catalog markers are reversed")

    # Load the existing syntax reader independently of cwd and import mode.
    # Tests may redirect REPO_ROOT without copying the actual script modules.
    reference_path = Path(__file__).with_name("verify_internal_references.py")
    spec = importlib.util.spec_from_file_location(
        "documentation_catalog_references", reference_path
    )
    require(
        spec is not None and spec.loader is not None, "reference checker is unavailable"
    )
    references = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(references)
    region = references._without_fenced_code(document[start:end])
    entries = []
    group = None
    for line in region.splitlines():
        heading = re.match(r"^##[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", line)
        if heading:
            group = heading.group(1)
            continue
        first_cell = re.match(r"^\s*\|((?:\\.|[^|])*)\|", line)
        links = references.LINK_PATTERN.findall(first_cell[1]) if first_cell else []
        if not links:
            continue
        label, raw = links[0]
        target = urlsplit(references._target(raw))
        if target.scheme or target.netloc:
            continue
        relative = unquote(target.path)
        if not relative.lower().endswith(".md"):
            continue
        require(
            group is not None,
            f"primary {directory} documents require an H2 catalog group",
        )
        entries.append((group, label, (index.parent / relative).resolve()))
    return entries


def _catalog_paths(directory: str) -> set[Path]:
    index = REPO_ROOT / directory / "README.md"
    folder = index.parent.resolve()
    excluded = folder / ("research/archive" if directory == "theory" else "assets")
    return {
        path.resolve()
        for path in folder.rglob("*.md")
        if path.resolve() != index.resolve()
        and not path.resolve().is_relative_to(excluded)
    }


def check_documentation_size() -> None:
    """Bound maintained owner size using the existing catalog scope."""
    for directory in ("theory", "docs"):
        owners = _catalog_paths(directory) | {
            (REPO_ROOT / directory / "README.md").resolve()
        }
        for path in sorted(owners):
            with path.open(encoding="utf-8") as document:
                line_count = sum(1 for _ in document)
            relative = path.relative_to(REPO_ROOT.resolve()).as_posix()
            require(
                line_count <= MAX_DOCUMENTATION_OWNER_LINES,
                f"{relative} has {line_count} physical lines "
                f"(limit {MAX_DOCUMENTATION_OWNER_LINES}); "
                "split by responsibility and preserve frozen evidence",
            )


def _check_catalog(directory: str, entries: list[tuple[str, str, Path]]) -> None:
    counts = Counter(path for _, _, path in entries)
    expected = _catalog_paths(directory)

    def names(paths) -> str:
        return ", ".join(
            sorted(
                (
                    str(path.relative_to(REPO_ROOT.resolve()))
                    if path.is_relative_to(REPO_ROOT.resolve())
                    else str(path)
                )
                for path in paths
            )
        )

    unknown = set(counts) - expected
    unknown_kind = (
        "unknown or archived" if directory == "theory" else "unknown or excluded"
    )
    require(
        not unknown, f"{unknown_kind} primary {directory} documents: " + names(unknown)
    )
    missing = expected - set(counts)
    require(not missing, f"missing primary {directory} documents: " + names(missing))
    duplicates = {path for path, count in counts.items() if count != 1}
    require(
        not duplicates, f"duplicate primary {directory} documents: " + names(duplicates)
    )


def _theory_catalog_entries() -> list[tuple[str, str, Path]]:
    return _catalog_entries("theory", THEORY_CATALOG_START, THEORY_CATALOG_END)


def _docs_catalog_entries() -> list[tuple[str, str, Path]]:
    return _catalog_entries("docs", DOCS_CATALOG_START, DOCS_CATALOG_END)


def check_theory_catalog() -> None:
    """Require one primary row per maintained theory owner, excluding archives."""
    _check_catalog("theory", _theory_catalog_entries())


def check_docs_catalog() -> None:
    """Require one primary row per maintained docs owner, excluding assets.

    Coverage derives from the recursive source tree. Link and fragment validation
    remains with verify_internal_references.py; no second owner manifest is added.
    """
    _check_catalog("docs", _docs_catalog_entries())


def _render_catalog_navigation(
    directory: str, title: str, index_title: str, entries: list[tuple[str, str, Path]]
) -> str:

    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    rows = [
        "  - " + quote(title) + ":",
        "      - " + quote(index_title) + f": {directory}/README.md",
    ]
    previous_group = None
    for group, label, path in entries:
        if group != previous_group:
            rows.append("      - " + quote(group) + ":")
            previous_group = group
        rows.append(
            "          - "
            + quote(label)
            + ": "
            + path.relative_to(REPO_ROOT.resolve()).as_posix()
        )
    return "\n".join(rows)


def render_theory_navigation() -> str:
    """Derive the MkDocs theory tree from its sole maintained owner catalog."""
    check_theory_catalog()
    return _render_catalog_navigation(
        "theory",
        "Theory and research",
        "Reading routes and implementation map",
        _theory_catalog_entries(),
    )


def render_docs_navigation() -> str:
    """Derive the technical documentation tree from docs/README.md."""
    check_docs_catalog()
    return _render_catalog_navigation(
        "docs", "Technical documentation", "Documentation map", _docs_catalog_entries()
    )


def _navigation_region(
    document: str, directory: str, navigation_start: str, navigation_end: str
) -> tuple[int, int]:
    starts = list(
        re.finditer("^" + re.escape(navigation_start) + "$", document, re.MULTILINE)
    )
    ends = list(
        re.finditer("^" + re.escape(navigation_end) + "$", document, re.MULTILINE)
    )
    require(
        len(starts) == 1 and len(ends) == 1,
        f"MkDocs must have exactly one generated {directory} navigation region at column zero",
    )
    start, end = starts[0].end(), ends[0].start()
    require(start < end, f"{directory} navigation markers are reversed")
    return start, end


def theory_navigation_region(document: str) -> tuple[int, int]:
    return _navigation_region(
        document, "theory", THEORY_NAVIGATION_START, THEORY_NAVIGATION_END
    )


def docs_navigation_region(document: str) -> tuple[int, int]:
    return _navigation_region(
        document, "docs", DOCS_NAVIGATION_START, DOCS_NAVIGATION_END
    )


def _update_navigation(region_reader, renderer) -> None:
    path = REPO_ROOT / "mkdocs.yml"
    document = path.read_text(encoding="utf-8")
    start, end = region_reader(document)
    rendered = renderer()
    path.write_text(
        document[:start] + "\n" + rendered + "\n" + document[end:], encoding="utf-8"
    )


def _check_navigation(directory: str, region_reader, renderer) -> None:
    document = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    start, end = region_reader(document)
    require(
        document[start:end].strip("\n") == renderer(),
        f"{directory} navigation drifted; run scripts/check_documentation.py --write-generated",
    )


def update_theory_navigation() -> None:
    _update_navigation(theory_navigation_region, render_theory_navigation)


def check_theory_navigation() -> None:
    _check_navigation("theory", theory_navigation_region, render_theory_navigation)


def update_docs_navigation() -> None:
    _update_navigation(docs_navigation_region, render_docs_navigation)


def check_docs_navigation() -> None:
    _check_navigation("docs", docs_navigation_region, render_docs_navigation)


def _glossary_checker():
    """Load the prose-only checker independently of cwd and a redirected root."""
    path = Path(__file__).with_name("check_glossary.py")
    spec = importlib.util.spec_from_file_location("documentation_glossary", path)
    require(
        spec is not None and spec.loader is not None, "glossary checker is unavailable"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_glossary() -> None:
    """Validate the sole concept register and its generated index."""
    _glossary_checker().check_glossary(REPO_ROOT)


def update_glossary_index() -> None:
    """Regenerate only the glossary index from validated concept declarations."""
    _glossary_checker().update_concept_index(REPO_ROOT)


def render_contract_table() -> str:
    """Render registry metadata, not an independently maintained contract list."""
    from tnfr.operators.operator_contracts import iter_contracts

    contracts = tuple(iter_contracts())
    require(len(contracts) == 13, "review the documented operator catalog size")

    def cell(value: str) -> str:
        return value.replace("\\", "\\\\").replace("|", "&#124;").replace("\n", " ")

    rows = [
        "| Operator | Token | Glyph | Primary channel | Scale | Context | Registered postcondition |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in contracts:
        name = (
            "Self-organization"
            if c.english_name == "SelfOrganization"
            else c.english_name
        )
        values = (
            name,
            c.name,
            c.glyph,
            c.primary_channel.value,
            c.scale.value,
            c.context.value,
            c.postcondition,
        )
        rows.append("| " + " | ".join(cell(v) for v in values) + " |")
    return "\n".join(rows)


def contract_region(document: str) -> tuple[int, int]:
    require(
        document.count(CONTRACT_START) == 1 and document.count(CONTRACT_END) == 1,
        "operator table must have exactly one generated region",
    )
    start = document.index(CONTRACT_START) + len(CONTRACT_START)
    end = document.index(CONTRACT_END)
    require(start < end, "operator table markers are reversed")
    return start, end


def update_contract_view() -> None:
    path = REPO_ROOT / "docs/API_CONTRACTS.md"
    document = path.read_text(encoding="utf-8")
    start, end = contract_region(document)
    path.write_text(
        document[:start] + "\n\n" + render_contract_table() + "\n\n" + document[end:],
        encoding="utf-8",
    )


def check_contract_view() -> None:
    document = (REPO_ROOT / "docs/API_CONTRACTS.md").read_text(encoding="utf-8")
    start, end = contract_region(document)
    require(
        document[start:end].strip() == render_contract_table(),
        "operator contract view drifted; run scripts/check_documentation.py --write-generated",
    )


def _check_readme_quickstart() -> None:
    """Execute the checkout's actual SDK quick start and compare its own output."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    sections = re.findall(
        r"^## Quick start[ \t]*\n(.*?)(?=^## |\Z)", readme, re.MULTILINE | re.DOTALL
    )
    require(len(sections) == 1, "README requires one Quick start section")
    blocks = {}
    for language in ("python", "text"):
        matches = re.findall(
            rf"^```{language}[ \t]*\n(.*?)^```[ \t]*$",
            sections[0],
            re.MULTILINE | re.DOTALL,
        )
        require(len(matches) == 1, f"README Quick start requires one {language} block")
        blocks[language] = matches[0]

    code = compile(blocks["python"], "README Quick start", "exec")
    output = io.StringIO()
    namespace = {"__name__": "__readme_quickstart__"}
    with redirect_stdout(output):
        exec(code, namespace)
    require(
        output.getvalue().strip() == blocks["text"].strip(),
        "README Quick start output has drifted",
    )


def check_documented_examples() -> None:
    from tnfr.operators.definitions import Coherence, Emission, Silence
    from tnfr.sdk.fluent import NetworkConfig, TNFRNetwork
    from tnfr.structural import create_nfr, run_sequence

    _check_readme_quickstart()

    graph, node = create_nfr("documentation-seed", epi=0.1, vf=1.0, theta=0.0)
    run_sequence(graph, node, [Emission(), Coherence(), Silence()])

    config = NetworkConfig(random_seed=7, default_epi_range=(0.1, 0.5))
    result = (
        TNFRNetwork("experiment", config)
        .add_nodes(20, phase_range=(0.0, 0.1))
        .connect_nodes(connection_pattern="ring")
        .apply_sequence(["emission", "coherence", "silence"])
        .measure()
    )
    require(
        0.0 <= result.coherence <= 1.0,
        "documented fluent example coherence out of range",
    )


def check_build_inputs() -> None:
    for relative in (
        "mkdocs.yml",
        "scripts/prepare_docs.py",
        "scripts/verify_internal_references.py",
        "scripts/check_glossary.py",
    ):
        require(
            (REPO_ROOT / relative).is_file(), f"missing documentation input: {relative}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-generated",
        action="store_true",
        help="Refresh operator contracts, theory/docs navigation and the concept-card index",
    )
    args = parser.parse_args()
    if args.write_generated:
        update_glossary_index()
        update_contract_view()
        update_theory_navigation()
        update_docs_navigation()
    checks = (
        check_agent_mirror,
        check_versions_and_retired_claims,
        check_publication_metadata,
        check_contract_view,
        check_documentation_size,
        check_theory_catalog,
        check_theory_navigation,
        check_docs_catalog,
        check_docs_navigation,
        check_glossary,
        check_documented_examples,
        check_build_inputs,
    )
    for check in checks:
        check()
        print(f"OK {check.__name__}")
    print("Documentation integrity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
