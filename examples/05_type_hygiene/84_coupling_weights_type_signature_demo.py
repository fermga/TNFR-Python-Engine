"""Example 84 — Coupling-Weights-Type Signature demo (B6a, §13quadraginta-nona).

Computes the Coupling-Weights-Type Signature
:math:`\\mathcal{S}_{W}` at two canonical probe resolutions and
prints the certificate summaries.  The diagnostic probes canonical
TNFR mixing-weight reads (``DNFR_WEIGHTS``, ``SI_WEIGHTS``,
``SELECTOR_WEIGHTS``) on two orthogonal axes:

- Scalar-storage axis: fraction of canonical-slot component
  values that are structurally scalar-coercible.
- Node-permutation-invariance axis: divergence of the canonical
  weighted-sum verdict across deterministic node relabelings.

Both axes are expected to certify
``"SCALAR_WEIGHTS_ADEQUATE"`` under canonical defaults — i.e.
``scalar_storage_fraction == 1.0`` and ``signature == 0.0`` —
structurally, because every canonical consumer site reads a single
global ``float`` per component and applies it uniformly to every
node (e.g. ``src/tnfr/dynamics/dnfr.py:2762-2764``).

Scope: diagnostic only; does NOT advance G4 = RH.
"""

from __future__ import annotations

from tnfr.riemann import compute_coupling_weights_type_signature


def main() -> None:
    print("=" * 72)
    print("Coupling-Weights-Type Signature demo (B6a, Sec 13quadraginta-nona)")
    print("=" * 72)

    print("\n--- Canonical small probe (n_nodes=24, n_permutations=12) ---")
    cert_small = compute_coupling_weights_type_signature(
        n_nodes=24,
        n_permutations=12,
        seed=23,
    )
    print(cert_small.summary())

    print("\n--- Canonical medium probe (n_nodes=48, n_permutations=24) ---")
    cert_med = compute_coupling_weights_type_signature(
        n_nodes=48,
        n_permutations=24,
        seed=23,
    )
    print(cert_med.summary())

    print("\n--- Frozen empirical signature (B6a) ---")
    print(
        f"  small:  S_W = {cert_small.signature:.6f}, "
        f"scalar_storage = {cert_small.scalar_storage_fraction:.4f}, "
        f"verdict = {cert_small.verdict}"
    )
    print(
        f"  medium: S_W = {cert_med.signature:.6f}, "
        f"scalar_storage = {cert_med.scalar_storage_fraction:.4f}, "
        f"verdict = {cert_med.verdict}"
    )
    print(
        "\nExpected (structural): S_W = 0.000000, "
        "scalar_storage = 1.0000, verdict = SCALAR_WEIGHTS_ADEQUATE"
    )
    print("Scope: necessary-condition diagnostic; does NOT advance G4 = RH.")


if __name__ == "__main__":
    main()
