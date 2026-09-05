#!/usr/bin/env python3
"""Check executable and single-source assumptions in TNFR documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _project_version() -> str:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if match is None:
        raise AssertionError("project version missing from pyproject.toml")
    return match.group(1)


def check_agent_mirror() -> None:
    canonical = (REPO_ROOT / "AGENTS.md").read_bytes()
    mirror = (REPO_ROOT / ".github/agents/my-agent.md").read_bytes()
    assert canonical == mirror, "canonical agent mirror has drifted"


def check_versions_and_retired_claims() -> None:
    version = _project_version()
    for relative in ("README.md", "ARCHITECTURE.md", "AGENTS.md"):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert version in text, f"{relative} does not identify version {version}"

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
            assert phrase not in text, f"retired claim in {relative}: {phrase}"


def check_contract_view() -> None:
    from tnfr.operators.operator_contracts import (
        iter_contracts,
        verify_contract_consistency,
    )

    verify_contract_consistency()
    document = (REPO_ROOT / "docs/API_CONTRACTS.md").read_text(encoding="utf-8")
    contracts = iter_contracts()
    assert len(contracts) == 13
    for contract in contracts:
        public_name = (
            "Self-organization"
            if contract.english_name == "SelfOrganization"
            else contract.english_name
        )
        assert public_name in document
        assert f"| {contract.glyph} |" in document


def check_documented_examples() -> None:
    from tnfr.operators.definitions import Coherence, Emission, Silence
    from tnfr.sdk import TNFR
    from tnfr.sdk.fluent import NetworkConfig, TNFRNetwork
    from tnfr.structural import create_nfr, run_sequence

    net = TNFR.create(20).ring().evolve(5)
    summary = net.results().summary()
    tetrad = net.tetrad().summary()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert summary in readme, "README result output has drifted"
    assert tetrad in readme, "README tetrad output has drifted"

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
    assert 0.0 <= result.coherence <= 1.0


def check_build_inputs() -> None:
    for relative in (
        "mkdocs.yml",
        "scripts/prepare_docs.py",
        "scripts/verify_internal_references.py",
    ):
        assert (REPO_ROOT / relative).is_file(), f"missing documentation input: {relative}"


def main() -> int:
    checks = (
        check_agent_mirror,
        check_versions_and_retired_claims,
        check_contract_view,
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
