# TNFR structural research program R1-R9

This document is the stable index for nine arithmetic and spectral research
lines. Each result is scoped as derived, measured, negative, conjectural or
open. None closes an external open problem.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| Derived | Exact under the hypotheses stated in the referenced note |
| Measured | Reproduced on a recorded finite domain |
| Negative | The tested stronger claim failed |
| Conjectural | Supported by a model or evidence but unproved |
| Open | No current derivation or adequate evidence |

## Program map

| Line | Result and boundary | Status | Source |
| --- | --- | --- | --- |
| R1 | Diffusion sectors and isolated operator actions are equivariant; word composition and pointed selectors state their additional hypotheses. | Derived and measured | [Structural observability](TNFR_STRUCTURAL_OBSERVABILITY.md) |
| R2 | For the specified prime residue network, the arithmetic pulse rank follows `gcd(k, p-1) + 1`; modal amplitudes are multiplicity fractions. | Derived and measured | [Arithmetic dynamics](TNFR_ARITHMETIC_DYNAMICS.md) |
| R3 | CRT product transport has the stated Kronecker form and supplies a U5-compatible synthesis. | Derived | [CRT fractality](TNFR_CRT_FRACTALITY.md) |
| R4 | Projective p-adic transport is exact for the defined maps; the static lift is a morphism and does not by itself satisfy REMESH's temporal-echo contract. | Derived with an open REMESH bridge | [p-adic dynamics](TNFR_PADIC_DYNAMICS.md) |
| R5 | Finite-field regression and trace-collision formulas hold for their stated fields; general type detection remains parameter-sensitive. | Derived with conjectural classification | [Algebraic number fields](TNFR_ALGEBRAIC_NUMBER_FIELDS.md) |
| R6 | The controlled additive construction reduces to a Fourier reading and provides no demonstrated TNFR-specific excess. | Derived negative | [Additive dynamics](TNFR_ADDITIVE_DYNAMICS.md) |
| R7 | The arithmetic-pressure terms are sufficient and functionally independent on the tested domain, but redundant for primality; completeness is open. | Mixed; completeness open | [Arithmetic pressure](TNFR_ARITHMETIC_PRESSURE.md) |
| R8 | Structural transformations are classified against existing operator contracts; rejected candidates define boundaries and do not create a fourteenth operator. | Measured classification | [Arithmetic operators](TNFR_ARITHMETIC_OPERATORS.md) |
| R9 | Scalar-capacity directed diffusion admits a clock change and can show transient amplification; heterogeneous capacity requires separate bounds. | Derived with open U2 metric | [Directed non-normal dynamics](TNFR_DIRECTED_NONNORMAL_DYNAMICS.md) |

## Shared acceptance rules

Every experiment in this program must:

1. record its domain, graph construction, seed and tolerances;
2. distinguish exact rational identities from floating-point measurements;
3. report whether known factors or labels enter the construction;
4. test relabeling or basis dependence where a spectral observable is used;
5. preserve the six TNFR invariants and the operator contracts;
6. state negative and indeterminate outcomes without promotion to a theorem.

## Current boundaries

- The tetrad is the canonical diagnostic interface, not a proved minimal state
  reconstruction basis.
- Arithmetic structural pressure characterizes primality but does not supply a
  faster factorization theorem.
- REMESH requires temporal echo; a scale-compatible static map is insufficient.
- Directed non-normal evolution needs a metric-aware transient analysis; spectral
  stability alone does not prove U2 boundedness.
- Operator certification establishes conformance to an existing contract, not
  universal catalog completeness.

The implementation map is maintained in [ARCHITECTURE.md](../ARCHITECTURE.md),
and the synthesized status is maintained in [AGENTS.md](../AGENTS.md).
