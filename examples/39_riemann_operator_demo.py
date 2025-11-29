"""Small numerical experiment with a toy TNFR–Riemann operator.

This example builds a simple prime-labeled path graph, constructs the
experimental ``H_TNFR`` operator from :mod:`tnfr.riemann`, and prints
its lowest eigenvalues for a few values of ``sigma`` (the analogue of
Re(s)).

The purpose is pedagogical and exploratory only.  It is *not* intended
as evidence about the actual Riemann Hypothesis, but as a concrete
numerical sandbox connected to ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md``.
"""

from __future__ import annotations

import numpy as np

from tnfr.riemann import build_h_tnfr, build_prime_path_graph


def main() -> None:
    # Small prime path graph for quick experimentation
    G = build_prime_path_graph(10, weight_by_log_gap=True)

    sigmas = [0.25, 0.5, 0.75]

    for sigma in sigmas:
        H, _ = build_h_tnfr(G, sigma=sigma)
        eigvals = np.linalg.eigvalsh(H)
        eigvals_sorted = np.sort(eigvals)
        print("sigma =", sigma)
        print("  lowest eigenvalues:")
        for val in eigvals_sorted[:5]:
            print(f"    {val: .6f}")
        print()


if __name__ == "__main__":  # pragma: no cover - manual example
    main()
