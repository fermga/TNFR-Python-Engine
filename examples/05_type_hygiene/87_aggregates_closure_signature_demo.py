"""Example 87 — Aggregates-Closure Signature demo (B9a, Sec 13quinquaginta-sexta).

Computes the Aggregates-Closure Signature :math:`\\mathcal{S}_{AC}`
at two canonical probe resolutions and prints the certificate
summaries. The diagnostic probes the four canonical scalar
aggregates — global coherence :math:`C(t)`, per-node Sense Index
:math:`S_i`, per-node energy density :math:`\\mathcal{E}(i)`, and
per-node topological charge :math:`\\mathcal{Q}(i)` — on two
orthogonal axes:

- Input-domain-closure axis: fraction of per-node input values
  touched by the aggregate pipeline (canonical Tier-1+Tier-2
  scalar slots ``nu_f``, ``EPI``, ``theta`` and the resolved
  ``DeltaNFR`` payload) that are structurally scalar-coercible.
- Output-scalar-closure axis: fraction of aggregate output
  values that are structurally scalar-coercible (the single
  global ``float`` from ``compute_coherence`` plus every per-node
  entry of the three ``dict[node, float]`` returns).

Both axes are expected to certify
``"SCALAR_CLOSURE_ADEQUATE"`` under canonical defaults — i.e.
``input_scalar_fraction == 1.0``,
``output_scalar_fraction == 1.0``, and ``signature == 0.0`` —
structurally, because every canonical aggregate implementation
at ``src/tnfr/metrics/common.py:29``,
``src/tnfr/metrics/sense_index.py:665``,
``src/tnfr/physics/unified.py:136``, and
``src/tnfr/physics/unified.py:209`` reads only scalar per-node
attributes plus the graph metric (with the tetrad and currents
already discharged as scalar-closed under B7 and B8 respectively),
and returns scalar values explicitly coerced via ``float(...)``.

Scope: diagnostic only; does NOT advance G4 = RH. B9 is a
closure question (Phase b is n/a); Phase c
(Sec 13quinquaginta-septima) emits the final verdict by direct
source-code trace of the four canonical aggregate functions.
"""

from __future__ import annotations

from tnfr.riemann import compute_aggregates_closure_signature


def main() -> None:
    print("=" * 72)
    print("Aggregates-Closure Signature demo " "(B9a, Sec 13quinquaginta-sexta)")
    print("=" * 72)

    print("\n--- Canonical small probe (n_nodes=24) ---")
    cert_small = compute_aggregates_closure_signature(
        n_nodes=24,
        seed=31,
    )
    print(cert_small.summary())

    print("\n--- Canonical medium probe (n_nodes=48) ---")
    cert_med = compute_aggregates_closure_signature(
        n_nodes=48,
        seed=31,
    )
    print(cert_med.summary())

    print("\n--- Frozen empirical signature (B9a) ---")
    print(
        f"  small:  S_AC = {cert_small.signature:.6f}, "
        f"input_scalar = {cert_small.input_scalar_fraction:.4f}, "
        f"output_scalar = {cert_small.output_scalar_fraction:.4f}, "
        f"verdict = {cert_small.verdict}"
    )
    print(
        f"  medium: S_AC = {cert_med.signature:.6f}, "
        f"input_scalar = {cert_med.input_scalar_fraction:.4f}, "
        f"output_scalar = {cert_med.output_scalar_fraction:.4f}, "
        f"verdict = {cert_med.verdict}"
    )
    print(
        "\nExpected (structural): S_AC = 0.000000, "
        "input_scalar = 1.0000, output_scalar = 1.0000, "
        "verdict = SCALAR_CLOSURE_ADEQUATE"
    )
    print("Scope: necessary-condition diagnostic; does NOT advance G4 = RH.")


if __name__ == "__main__":
    main()
