"""Minimal installation verification for the TNFR Spectral Factorization lab.

Run this script after installing the engine dependencies to check the local
lab API and CLI. The lab is a source application, not a separately built
distribution; the child process explicitly uses its package directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _run_cli(args: Sequence[str]) -> list[dict[str, object]]:
    """Execute the CLI in JSON mode and return the parsed payload."""

    completed = subprocess.run(
        [sys.executable, "-m", "tnfr_factorization.cli", *args, "--json"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
    )
    return json.loads(completed.stdout)


def main() -> None:
    from tnfr_factorization import SpectralPaleyFactorizer

    target = 221
    factorizer = SpectralPaleyFactorizer()
    result = factorizer.analyze(target)
    print(
        "API check: n=%d modulus=%d laplacian_gap=%.6f DeltaNFR=%.3e"
        % (
            result.n,
            result.modulus,
            result.laplacian_gap,
            result.arithmetic_delta_nfr,
        )
    )

    payload = _run_cli([str(target)])
    first = payload[0]
    print(
        "CLI check: n=%d node_count=%s coherence_score=%.4f"
        % (
            first["n"],
            first["node_count"],
            first["coherence_score"],
        )
    )
    print("Installation verification passed.")


if __name__ == "__main__":  # pragma: no cover
    main()
