#!/usr/bin/env python3
"""R8 arithmetic transformation-to-operator certification benchmark.

Issues a reproducible certificate for each candidate transformation, mapping it to
a canonical TNFR operator only when its measured effect matches the operator
contract and its grammar word validates. Two candidates certify (emission at zero
-> Emission; residue-edge propagation -> Resonance); four are rejected with an
explicit reason (CRT projection, p-adic lift, affine map, power map).

Honest scope: no fourteenth operator is invented; the four negatives are the
boundary result (an arithmetic operation that does not fit the 13 operators
without an external axiom sharpens the TNFR boundary). Claim NT-P08 OPEN.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.mathematics.operator_certificates import (  # noqa: E402
    all_certificates,
    certified_mappings,
    rejected_mappings,
    verify_certificate,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)


def main() -> int:
    print("R8 arithmetic operator certification (2 certified, 4 rejected)")
    print()
    all_ok = True
    for cert in all_certificates():
        ok = verify_certificate(cert)
        all_ok &= ok
        target = cert.canonical_operator if cert.certified else "REJECTED"
        print(f"  [{target:>9}] {cert.transformation}")
        print(f"      channel={cert.state_channel}  scale={cert.scale}  "
              f"grammar_valid={cert.grammar_valid}  verify={ok}")
        res = {k: round(v, 4) for k, v in cert.residuals.items()}
        print(f"      residuals={res}")
        if cert.rejected:
            print(f"      reason: {cert.rejection_reason[:72]}...")

    certified = certified_mappings()
    rejected = rejected_mappings()
    audit = CircularityAudit()  # contract-matching only; no factoring
    manifest = ExperimentManifest(
        claim_id="NT-P08",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,
        operator_sequence=tuple(
            op for c in certified for op in c.grammar_word
        ),
        uses_known_factors=False,
        input_bits=input_bit_length(15),
        controls=("grammar_word_validation", "contract_residuals",
                  "negative_mapping_cases"),
        artifacts=(),
    )
    print()
    print(f"  certified: {[c.canonical_operator for c in certified]}")
    print(f"  rejected : {len(rejected)} candidates (boundary result)")
    print(f"  all certificates verify: {all_ok}")
    print(f"  claim status : {ClaimStatus.MEASURED.value} (certificates) / "
          f"NT-P08 open")
    print(f"  circularity  : {audit.verdict.value}")
    print(f"  manifest ops : {len(manifest.operator_sequence)} glyphs")
    return 0 if (all_ok and len(certified) == 2 and len(rejected) == 4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
