"""B10 (URC) Phase a demo — U-Rules Consistency Signature diagnostic.

Runs ``compute_urules_consistency_signature`` with two distinct seeds and
prints per-rule input/output type classifications. Both probes are expected
to return ``TYPE_HYGIENE_ADEQUATE`` with S_UR = 0.000000, witnessing the
type-hygiene closure of the U1-U6 rule checkers in
``src/tnfr/operators/grammar_core.py`` and ``grammar_u6.py``.

Scope guard: methodological diagnostic only. Does NOT advance G4 = RH
(Conjecture T-HP, §13septies of TNFR_RIEMANN_RESEARCH_NOTES.md).
"""

from __future__ import annotations

from tnfr.riemann import compute_urules_consistency_signature


def _print_probe(label: str, n_nodes: int, seed: int) -> None:
    cert = compute_urules_consistency_signature(n_nodes=n_nodes, seed=seed)
    print(f"\n=== {label} (n_nodes={n_nodes}, seed={seed}) ===")
    print(cert.summary())
    print(f"leaking_inputs={cert.leaking_inputs}, "
          f"leaking_outputs={cert.leaking_outputs}")
    print("Per-rule classifications:")
    for name in cert.rules_probed:
        ins = cert.input_classifications[name]
        out = cert.output_classifications[name]
        print(f"  {name:40s} inputs={ins} | output={out}")
    assert cert.verdict == "TYPE_HYGIENE_ADEQUATE", (
        f"Unexpected verdict: {cert.verdict}"
    )
    assert cert.S_UR == 0.0, f"Unexpected S_UR: {cert.S_UR}"


def main() -> None:
    print("B10 (URC) Phase a — U-Rules Consistency Signature diagnostic")
    print("=" * 70)
    _print_probe("Probe A", n_nodes=24, seed=31)
    _print_probe("Probe B", n_nodes=48, seed=131)
    print("\nBoth probes: S_UR=0.000000, TYPE_HYGIENE_ADEQUATE.")
    print(
        "Witness: U1-U6 rule checkers read only operator-name sequences + "
        "scalar telemetry (Φ_s dicts + thresholds) and return only "
        "scalar/tuple-of-scalar verdicts. No callable kernel, no measure, "
        "no operator-valued intermediate."
    )


if __name__ == "__main__":
    main()
