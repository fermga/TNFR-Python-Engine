"""B11 (OCD) Phase a demo — Operator-Catalog Discipline signature.

Runs ``compute_operator_catalog_discipline_signature`` and prints all
per-probe results. The diagnostic is deterministic (read-only inspection
of module-level mappings) so a single invocation suffices; we run it
twice to confirm idempotency.

Both invocations are expected to return ``CATALOG_DISCIPLINE_ADEQUATE``
with S_OC = 0.000000, witnessing that:

- The canonical TNFR operator registry contains exactly 13 entries.
- Every entry is a subclass of ``Operator`` with a non-empty lowercase
  string name.
- ``OPERATOR_METADATA`` contains exactly 13 ``OperatorMeta`` instances
  whose ``grammar_roles`` and ``contracts`` fields are tuples of strings.
- ``definitions.__all__`` exposes the 13 canonical operator class names.
- The registry is idempotent: re-invoking ``_ensure_loaded()`` never
  expands the catalog past 13 (no hidden 14th operator).

Scope guard: methodological diagnostic only. Does NOT advance G4 = RH
(Conjecture T-HP, §13septies of TNFR_RIEMANN_RESEARCH_NOTES.md).
"""

from __future__ import annotations

from tnfr.riemann import compute_operator_catalog_discipline_signature


def _print_probe(label: str) -> None:
    cert = compute_operator_catalog_discipline_signature()
    print(f"\n=== {label} ===")
    print(cert.summary())
    print(f"anomalies={cert.anomalies} / total_probes={cert.total_probes}")
    print("Per-probe results:")
    for name in cert.probes:
        print(f"  {name:48s} {cert.probe_results[name]}")
    assert cert.verdict == "CATALOG_DISCIPLINE_ADEQUATE", (
        f"Unexpected verdict: {cert.verdict}"
    )
    assert cert.S_OC == 0.0, f"Unexpected S_OC: {cert.S_OC}"
    assert cert.registry_size == 13
    assert cert.metadata_size == 13


def main() -> None:
    print("B11 (OCD) Phase a — Operator-Catalog Discipline diagnostic")
    print("=" * 70)
    _print_probe("Invocation A (cold)")
    _print_probe("Invocation B (idempotent)")
    print(
        "\nBoth invocations: S_OC=0.000000, CATALOG_DISCIPLINE_ADEQUATE."
    )
    print(
        "Witness: registry == 13 entries, metadata == 13 entries, "
        "definitions exports == 13 canonical class names. No callable "
        "kernel, no measure, no operator-valued intermediate, no matrix "
        "lift attached to the catalog surface."
    )


if __name__ == "__main__":
    main()
