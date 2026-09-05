# Resolution: Grammar and Tetrad Theorem Scope

**Scope:** Documentation contradictions T07, T08, T10, and T11 in
[THEORY_CONTRADICTIONS_2026-09-05.md](THEORY_CONTRADICTIONS_2026-09-05.md).
The numerical grammar policies, operator sets, field calculations, and operator
evolution are unchanged. This resolution corrects mathematical assertions;
it does not claim to prove the stronger statements rejected by the witnesses.

## Canonical synthesis

[DIAGNOSTIC_AND_GRAMMAR_SCOPE.md](../../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md)
now centralizes exact identities, necessary hypotheses, engine policies, and
open reconstruction questions:

| Item | Current statement |
|------|-------------------|
| T07: potential confinement | Φ_s = B_G ΔNFR is linear. The bound is ‖Φ_s‖∞ ≤ ‖B_G‖∞ ‖ΔNFR‖∞. Potential π/4 and drift π/2 are selected policies, not phase-wrap consequences. |
| T08: recency and debt | Window 3 and capacity 2 use the scalar q = 1−ν_f dt surrogate. Individual Euler modes use 1−ν_f dt λ_k. The functions' fallback branches are compatibility policy, not Euler-stability certificates. |
| T10: initiation and convergence | EPI = 0 need not have an undefined derivative. U1 is an initialization contract. Finite-horizon existence, bounded partial integrals, improper-integral convergence, and absolute integrability are distinct. U2 acceptance alone is not an infinite-horizon proof. |
| T11: tetrad completeness | Four canonical diagnostic channels do not prove complete state reconstruction. Laplacian algebra generation is not sufficiency of lossy field summaries; a target state quotient and observable class must be specified. |

The potential note also distinguishes fixed-topology drift from graph-kernel
change, gives shell-growth hypotheses for summability, and shows why chain
series do not uniquely select exponent 2. The exact phase maxima remain π;
0.9π is a selected curvature warning margin.

## Files reconciled

- [UNIFIED_GRAMMAR_RULES.md](../../theory/UNIFIED_GRAMMAR_RULES.md): one coherent
  U1–U6 synthesis replaces duplicated derivations, obsolete U6=2.0 assertions,
  and historical proposed-rule fragments. It retains canonical role sets,
  three-operation recency, two-unit prefix debt, pre-existing-EPI context,
  the narrow explicit diagnostic allowance, declared depth checks, and the
  distinction between word validation, operator preconditions, and telemetry.
- [MINIMAL_STRUCTURAL_DEGREES.md](../../theory/MINIMAL_STRUCTURAL_DEGREES.md):
  diagnostic selection, algebraic generation, reconstruction, and removal
  studies are distinguished. The existing canonical-versus-dispersion
  coherence correction is preserved with the 1/2 versus 1/3 example.
- [STRUCTURAL_FIELDS_TETRAD.md](../STRUCTURAL_FIELDS_TETRAD.md): field definitions,
  current policies, exact/approximate potential behavior, and the correlation
  fit versus spectral fallback are consistent with the implementation. The
  runnable example uses the actual estimate_coherence_length(G) signature.
- [FUNDAMENTAL_THEORY.md](../../theory/FUNDAMENTAL_THEORY.md): immediate canonical
  summaries use the shared scope. Its existing sections and examples remain;
  diffusion, Dirichlet gradient flow, damped graph-wave approximation, and
  isotropic harmonic substrate are explicitly separated.
- [physics_derivation.py](../../src/tnfr/config/physics_derivation.py) and
  [constants/canonical.py](../../src/tnfr/constants/canonical.py): comments and
  docstrings describe the selected policies and their mathematical scope.
  Executable statements are unchanged.

No literal incoming Markdown fragment links to the three rewritten guides
were found in the repository scan. All relative Markdown links in the five
theory/field guides resolve. Broader research histories are not rewritten;
their mathematical claims must be evaluated under their explicit hypotheses.

## Verification

The parent task completed the pre-edit baseline: **3,082 passed, 11 skipped**.
The focused command

    .venv312/Scripts/python.exe -m pytest tests/operators/test_grammar_canonical_consistency.py -q

passed **25 tests** after the source documentation edits. Comparing parsed
Python syntax trees against HEAD after removing docstrings found no executable
differences in either edited Python module. The canonical window, capacity,
potential thresholds, gradient warning, and curvature warning were checked
against their original values. The exact witness checks returned:

| Deterministic fixture | Result |
|-----------------------|--------|
| K₄, every phase 0, every pressure 1 | Φ_s = 3 at each node; phase gradient and curvature zero |
| K₄, every phase 0, every pressure 2 | Φ_s = 6 at each node |
| P₂₁, unweighted normalized Laplacian | λ₂ = 0.012311659404862377; trace/N = 1 |
| P₂₁ Fiedler mode, ν_f=1, dt=0.5 | Amplitude after 3 steps = 0.9816459603401984; 231 steps needed below 1/(π+1) |
| P₄, matrices I, L_rw, L_rw², L_rw³ flattened as columns | Rank 4 |
| Field-guide Watts–Strogatz fixture, N=60, k=4, p=0.2, seed=42 | All three node-field maps have 60 entries; ξ_C = 3.679286048526831 |

The analytic counterexamples x(t)=sin(t), p(t)=cos(t), ν_f=1 and
p(t)=1/(1+t), ν_f=1 distinguish boundedness from convergence and vanishing
pressure from integrability. No stochastic operator sequence is used in the
finite-graph witnesses. There is no before/after C(t) improvement claim:
this change updates theorem scope and leaves evolution unchanged.

## Remaining mathematical work

Universal confinement would require explicit pressure and graph-geometry
bounds. A graph-aware relaxation theorem needs the relevant generator,
stationary-mode treatment, stability region, norm, and operator gains.
Infinite-horizon grammar sufficiency requires an actual controlled-flow
estimate. Tetrad minimality/completeness requires a specified state quotient,
observable class, and injectivity/non-redundancy proofs. These remain open;
their absence does not authorize bypassing the existing grammar contracts.
