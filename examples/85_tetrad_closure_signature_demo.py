"""Example 85 — Tetrad-Closure Signature demo (B7a, Sec 13quinquaginta-secunda).

Computes the Tetrad-Closure Signature :math:`\\mathcal{S}_{TC}`
at two canonical probe resolutions and prints the certificate
summaries. The diagnostic probes the four canonical tetrad-field
functions
(``compute_structural_potential``, ``compute_phase_gradient``,
``compute_phase_curvature``, ``estimate_coherence_length``) on
two orthogonal axes:

- Input-domain-closure axis: fraction of per-node input values
  touched by the tetrad pipeline (canonical Tier-1+Tier-2 scalar
  slots ``EPI``, ``theta``, ``nu_f``, and the resolved
  ``DeltaNFR`` payload) that are structurally scalar-coercible.
- Output-scalar-closure axis: fraction of tetrad-field output
  values that are structurally scalar-coercible (every per-node
  entry of the three ``dict[node, float]`` returns plus the
  global ``float`` returned by ``estimate_coherence_length``).

Both axes are expected to certify
``"SCALAR_CLOSURE_ADEQUATE"`` under canonical defaults — i.e.
``input_scalar_fraction == 1.0``,
``output_scalar_fraction == 1.0``, and ``signature == 0.0`` —
structurally, because every canonical tetrad-field
implementation at ``src/tnfr/physics/canonical.py:199,609,640,756``
reads only scalar per-node attributes plus the graph metric, and
returns scalar values explicitly coerced via ``float(...)``.

Scope: diagnostic only; does NOT advance G4 = RH. B7 is a
closure question (Phase b is n/a); Phase c
(Sec 13quinquaginta-tertia) emits the final verdict by direct
source-code trace of the four canonical tetrad-field functions.
"""

from __future__ import annotations

from tnfr.riemann import compute_tetrad_closure_signature


def main() -> None:
    print("=" * 72)
    print("Tetrad-Closure Signature demo (B7a, Sec 13quinquaginta-secunda)")
    print("=" * 72)

    print("\n--- Canonical small probe (n_nodes=24) ---")
    cert_small = compute_tetrad_closure_signature(
        n_nodes=24,
        seed=31,
    )
    print(cert_small.summary())

    print("\n--- Canonical medium probe (n_nodes=48) ---")
    cert_med = compute_tetrad_closure_signature(
        n_nodes=48,
        seed=31,
    )
    print(cert_med.summary())

    print("\n--- Frozen empirical signature (B7a) ---")
    print(
        f"  small:  S_TC = {cert_small.signature:.6f}, "
        f"input_scalar = {cert_small.input_scalar_fraction:.4f}, "
        f"output_scalar = {cert_small.output_scalar_fraction:.4f}, "
        f"verdict = {cert_small.verdict}"
    )
    print(
        f"  medium: S_TC = {cert_med.signature:.6f}, "
        f"input_scalar = {cert_med.input_scalar_fraction:.4f}, "
        f"output_scalar = {cert_med.output_scalar_fraction:.4f}, "
        f"verdict = {cert_med.verdict}"
    )
    print(
        "\nExpected (structural): S_TC = 0.000000, "
        "input_scalar = 1.0000, output_scalar = 1.0000, "
        "verdict = SCALAR_CLOSURE_ADEQUATE"
    )
    print(
        "Scope: necessary-condition diagnostic; does NOT advance G4 = RH."
    )


if __name__ == "__main__":
    main()
