"""Example 86 — Currents-Closure Signature demo (B8a, Sec 13quinquaginta-quarta).

Computes the Currents-Closure Signature :math:`\\mathcal{S}_{CC}`
at two canonical probe resolutions and prints the certificate
summaries. The diagnostic probes the two canonical current-field
functions (``compute_phase_current``, ``compute_dnfr_flux``) and
the conservation aggregator (``compute_current_divergence``) on
two orthogonal axes:

- Input-domain-closure axis: fraction of per-node input values
  touched by the current pipeline (canonical Tier-1+Tier-2
  scalar slots ``theta`` and the resolved ``DeltaNFR`` payload)
  that are structurally scalar-coercible.
- Output-scalar-closure axis: fraction of current/divergence
  output values that are structurally scalar-coercible (every
  per-node entry of the three ``dict[node, float]`` returns).

Both axes are expected to certify
``"SCALAR_CLOSURE_ADEQUATE"`` under canonical defaults — i.e.
``input_scalar_fraction == 1.0``,
``output_scalar_fraction == 1.0``, and ``signature == 0.0`` —
structurally, because every canonical current implementation at
``src/tnfr/physics/extended.py:60,182`` and the divergence
aggregator at ``src/tnfr/physics/conservation.py:209`` reads
only scalar per-node attributes plus the graph metric, and
returns scalar values explicitly coerced via ``float(...)``.

Scope: diagnostic only; does NOT advance G4 = RH. B8 is a
closure question (Phase b is n/a); Phase c
(Sec 13quinquaginta-quinta) emits the final verdict by direct
source-code trace of the canonical current functions and the
divergence aggregator.
"""

from __future__ import annotations

from tnfr.riemann import compute_currents_closure_signature


def main() -> None:
    print("=" * 72)
    print("Currents-Closure Signature demo (B8a, Sec 13quinquaginta-quarta)")
    print("=" * 72)

    print("\n--- Canonical small probe (n_nodes=24) ---")
    cert_small = compute_currents_closure_signature(
        n_nodes=24,
        seed=31,
    )
    print(cert_small.summary())

    print("\n--- Canonical medium probe (n_nodes=48) ---")
    cert_med = compute_currents_closure_signature(
        n_nodes=48,
        seed=31,
    )
    print(cert_med.summary())

    print("\n--- Frozen empirical signature (B8a) ---")
    print(
        f"  small:  S_CC = {cert_small.signature:.6f}, "
        f"input_scalar = {cert_small.input_scalar_fraction:.4f}, "
        f"output_scalar = {cert_small.output_scalar_fraction:.4f}, "
        f"verdict = {cert_small.verdict}"
    )
    print(
        f"  medium: S_CC = {cert_med.signature:.6f}, "
        f"input_scalar = {cert_med.input_scalar_fraction:.4f}, "
        f"output_scalar = {cert_med.output_scalar_fraction:.4f}, "
        f"verdict = {cert_med.verdict}"
    )
    print(
        "\nExpected (structural): S_CC = 0.000000, "
        "input_scalar = 1.0000, output_scalar = 1.0000, "
        "verdict = SCALAR_CLOSURE_ADEQUATE"
    )
    print(
        "Scope: necessary-condition diagnostic; does NOT advance G4 = RH."
    )


if __name__ == "__main__":
    main()
