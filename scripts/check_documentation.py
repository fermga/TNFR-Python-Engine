#!/usr/bin/env python3
"""Check executable and single-source assumptions in TNFR documentation."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
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
        "docs/API_CONTRACTS.md",
        "docs/grammar/PHYSICS_VERIFICATION.md",
        ".github/WORKFLOWS.md",
        "theory/README.md",
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


def check_publication_metadata() -> None:
    """Keep distribution, citation and archive metadata on the reviewed version."""
    version = _project_version()
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    archive = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
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


CONTRACT_START = "<!-- BEGIN GENERATED OPERATOR CONTRACTS -->"
CONTRACT_END = "<!-- END GENERATED OPERATOR CONTRACTS -->"
THEORY_CATALOG_START = "<!-- BEGIN THEORY CATALOG -->"
THEORY_CATALOG_END = "<!-- END THEORY CATALOG -->"
THEORY_NAVIGATION_START = "# BEGIN GENERATED THEORY NAVIGATION"
THEORY_NAVIGATION_END = "# END GENERATED THEORY NAVIGATION"


def _theory_catalog_entries() -> list[tuple[str, str, Path]]:
    """Read H2 groups and the first inline link of each primary table row."""
    index = REPO_ROOT / "theory" / "README.md"
    document = index.read_text(encoding="utf-8")
    require(
        document.count(THEORY_CATALOG_START) == 1
        and document.count(THEORY_CATALOG_END) == 1,
        "theory index must have exactly one primary catalog region",
    )
    start = document.index(THEORY_CATALOG_START) + len(THEORY_CATALOG_START)
    end = document.index(THEORY_CATALOG_END)
    require(start < end, "theory catalog markers are reversed")

    # Load the existing syntax reader independently of cwd and import mode.
    # Tests may redirect REPO_ROOT without copying the actual script modules.
    reference_path = Path(__file__).with_name("verify_internal_references.py")
    spec = importlib.util.spec_from_file_location(
        "theory_catalog_references", reference_path
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
        links = (
            references.LINK_PATTERN.findall(line)
            if line.lstrip().startswith("|")
            else []
        )
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
            group is not None, "primary theory documents require an H2 catalog group"
        )
        entries.append((group, label, (index.parent / relative).resolve()))
    return entries


def check_theory_catalog() -> None:
    """Require one primary table entry per maintained mathematical owner.

    The index owns this inventory: no independently maintained filename list
    or fixed document count is needed. Full link and fragment validation stays
    with verify_internal_references.py; this gate checks catalog coverage.
    """
    index = REPO_ROOT / "theory" / "README.md"
    counts = Counter(path for _, _, path in _theory_catalog_entries())

    theory = index.parent.resolve()
    archive = theory / "research" / "archive"
    expected = {
        path.resolve()
        for path in theory.rglob("*.md")
        if path.resolve() != index.resolve()
        and not path.resolve().is_relative_to(archive)
    }

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
    require(
        not unknown, "unknown or archived primary theory documents: " + names(unknown)
    )
    missing = expected - set(counts)
    require(not missing, "missing primary theory documents: " + names(missing))
    duplicates = {path for path, count in counts.items() if count != 1}
    require(not duplicates, "duplicate primary theory documents: " + names(duplicates))


def render_theory_navigation() -> str:
    """Derive the MkDocs theory tree from the sole maintained owner catalog."""
    check_theory_catalog()

    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    rows = [
        "  - " + quote("Theory and research") + ":",
        "      - "
        + quote("Reading routes and implementation map")
        + ": theory/README.md",
    ]
    previous_group = None
    for group, label, path in _theory_catalog_entries():
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


def theory_navigation_region(document: str) -> tuple[int, int]:
    starts = list(
        re.finditer(
            "^" + re.escape(THEORY_NAVIGATION_START) + "$", document, re.MULTILINE
        )
    )
    ends = list(
        re.finditer(
            "^" + re.escape(THEORY_NAVIGATION_END) + "$", document, re.MULTILINE
        )
    )
    require(
        len(starts) == 1 and len(ends) == 1,
        "MkDocs must have exactly one generated theory navigation region at column zero",
    )
    start, end = starts[0].end(), ends[0].start()
    require(start < end, "theory navigation markers are reversed")
    return start, end


def update_theory_navigation() -> None:
    path = REPO_ROOT / "mkdocs.yml"
    document = path.read_text(encoding="utf-8")
    start, end = theory_navigation_region(document)
    rendered = render_theory_navigation()
    path.write_text(
        document[:start] + "\n" + rendered + "\n" + document[end:], encoding="utf-8"
    )


def check_theory_navigation() -> None:
    document = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    start, end = theory_navigation_region(document)
    require(
        document[start:end].strip("\n") == render_theory_navigation(),
        "theory navigation drifted; run scripts/check_documentation.py --write-generated",
    )


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


def check_documented_examples() -> None:
    from tnfr.operators.definitions import Coherence, Emission, Silence
    from tnfr.sdk import TNFR
    from tnfr.sdk.fluent import NetworkConfig, TNFRNetwork
    from tnfr.structural import create_nfr, run_sequence

    net = TNFR.create(20).ring().evolve(5)
    summary = net.results().summary()
    tetrad = net.tetrad().summary()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    require(summary in readme, "README result output has drifted")
    require(tetrad in readme, "README tetrad output has drifted")

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
    ):
        require(
            (REPO_ROOT / relative).is_file(), f"missing documentation input: {relative}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-generated",
        action="store_true",
        help="Refresh the registry-owned contract table and catalog-owned theory navigation",
    )
    args = parser.parse_args()
    if args.write_generated:
        update_contract_view()
        update_theory_navigation()
    checks = (
        check_agent_mirror,
        check_versions_and_retired_claims,
        check_publication_metadata,
        check_contract_view,
        check_theory_catalog,
        check_theory_navigation,
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
