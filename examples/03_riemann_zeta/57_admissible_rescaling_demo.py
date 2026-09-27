"""Finite congruence rescaling to supplied classical smooth spectral targets.

For a positive Hermitian P14 matrix H=U diag(lambda) U*, choose
F=U diag(sqrt(target/lambda)) U*. Then F H F* has the target spectrum in exact
arithmetic; the implementation checks finite numerical agreement. The targets
come from a classical smooth zero-count approximation, not autonomous nodal
emergence. The comparison with reference zeros and the configured enrichment
sweep test only these supplied constructions. Their residual is neither an
RH-equivalent observable nor proof that a new canonical operator is necessary.
Current scope: theory/TNFR_RIEMANN_RESEARCH_NOTES.md.
"""

from __future__ import annotations

from tnfr.riemann.admissible_rescaling import compute_admissible_rescaling_certificate


def _run(label: str, n_targets: int) -> None:
    print("=" * 72)
    print(f"  {label}  (n_targets = {n_targets})")
    print("=" * 72)
    cert = compute_admissible_rescaling_certificate(
        n_targets=n_targets,
        p14_n_primes=max(40, n_targets * 2),
        p14_max_power=6,
        oscillatory_mode="phi_log",
    )
    print(cert.summary())
    print()


def _sweep_oscillatory_modes(n_targets: int = 20) -> None:
    print("=" * 72)
    print(f"  Oscillatory enrichment sweep  (n_targets = {n_targets})")
    print("=" * 72)
    for mode in ("phi_log", "gamma_e", "pi_density"):
        cert = compute_admissible_rescaling_certificate(
            n_targets=n_targets,
            p14_n_primes=max(40, n_targets * 2),
            p14_max_power=6,
            oscillatory_mode=mode,
        )
        improv = cert.oscillatory_improvement_over_smooth
        amp = cert.oscillatory_amplitude
        w1 = cert.w1_oscillatory_vs_true
        print(
            f"  mode={mode:<11}  best amp={amp:.4e}  "
            f"W1={w1:.4e}  improv={improv:+.2f}%"
        )
    print()


if __name__ == "__main__":
    _run("Resolution A (fast)", n_targets=20)
    _run("Resolution B (medium)", n_targets=40)
    _sweep_oscillatory_modes(n_targets=20)
    print(
        "Honest scope: smooth half operationally closed; "
        "oscillatory half + canonicity + positivity remain OPEN. "
        "Finite target matching does not derive zeros or prove RH."
    )
