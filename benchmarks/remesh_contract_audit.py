#!/usr/bin/env python3
"""N09 REMESH contract audit benchmark (R4b).

The p-adic tower transport is an exact morphism (R4/N08): a static lift/scale map
that transports structure instantaneously. REMESH (Recursivity) additionally
requires a temporal echo EPI_new = (1-a)^2 EPI(t) + a(1-a) EPI(t-tau_l) +
a EPI(t-tau_g). This benchmark runs one predefined campaign auditing the four
REMESH-contract conditions (temporal echo, NETWORK scale, preserved identity, U5
multiscale coherence) on the static p-adic projection Lift.R_e versus the genuine
temporal recurrence.

Honest result: the static tower map satisfies three conditions but FAILS the
temporal echo -- it is a projection morphism, not REMESH (NT-P04b negative for the
tower). The temporal recurrence passes all four, so the gate is a real
discriminator, not vacuous. The only ingredient the lift lacks is the
EPI(t) <- EPI(t-tau) recursion. No complexity / crypto / Millennium claim.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.mathematics.remesh_audit import remesh_campaign  # noqa: E402
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)


def _row(label, a):
    return (f"  {label:<28} {str(a.epi_recursion_verified):>5} "
            f"{str(a.network_scale_verified):>5} "
            f"{str(a.identity_preserved_verified):>5} "
            f"{str(a.u5_multiscale_verified):>5} {str(a.realizes_remesh):>8}")


def main() -> int:
    print("N09 REMESH contract audit: is the p-adic transport REMESH?")
    print(f"  {'candidate':<28} {'echo':>5} {'net':>5} {'id':>5} {'u5':>5} "
          f"{'REMESH':>8}")
    camp = remesh_campaign(p=3, e=1, alpha=0.5)
    print(_row("static p-adic Lift.R_e", camp.static_lift))
    print(_row("temporal REMESH recurrence", camp.temporal_recurrence))

    audit = CircularityAudit()  # pure structural dynamics
    _ = ExperimentManifest(
        claim_id="NT-P04b",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=0,
        operator_sequence=("REMESH",),
        uses_known_factors=False,
        input_bits=input_bit_length(9),
        controls=("temporal_echo_discriminates", "static_lift_no_echo",
                  "recurrence_all_four", "and_gate_requires_all"),
        artifacts=(),
    )
    print()
    print(f"  tower realizes REMESH   : {camp.tower_realizes_remesh} "
          "(honest close: lift is a morphism, not REMESH)")
    print(f"  audit discriminates     : {camp.audit_discriminates} "
          "(rejects static, accepts genuine REMESH)")
    print(f"  REMESH for the tower    : {ClaimStatus.NEGATIVE.value} "
          "(NT-P04b; missing temporal echo)")
    print(f"  four-condition gate     : {ClaimStatus.MEASURED.value} "
          "(discriminator, not vacuous)")
    print(f"  circularity             : {audit.verdict.value}")
    ok = (not camp.tower_realizes_remesh) and camp.audit_discriminates
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
