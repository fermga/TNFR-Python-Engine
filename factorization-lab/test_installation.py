"""Minimal installation verification for the TNFR Spectral Factorization lab.

Run this script after installing the package (editable mode or sdist/wheel) to
ensure the Python API and CLI entry point both work and emit TNFR telemetry.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Sequence


def _run_cli(args: Sequence[str]) -> list[dict[str, object]]:
    """Execute the CLI in JSON mode and return the parsed payload."""

    completed = subprocess.run(
        [sys.executable, "-m", "tnfr_factorization.cli", *args, "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    from tnfr_factorization import SpectralPaleyFactorizer

    target = 221
    factorizer = SpectralPaleyFactorizer()
    result = factorizer.analyze(target)
    print(
        "API check: n=%d modulus=%d laplacian_gap=%.6f ΔNFR=%.3e"
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
