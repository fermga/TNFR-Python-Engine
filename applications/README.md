# Arithmetic utility

The primality utility reuses selected TNFR utilities for supplied arithmetic
problems. It is separate from the core engine under `src/tnfr` and from the
[research program](../theory/RESEARCH_PROGRAM.md#research-current-question).
Its formulas, heuristic scores and compatibility names do not establish
physical emergence or a new primality complexity result.

| Project | Purpose | Boundary |
| --- | --- | --- |
| [Primality utility](primality-test/README.md) | Maintained standalone arithmetic-pressure descriptors, numerical predicate and basic CLI | Consumes divisor/factor statistics; does not generate primes through nodal dynamics |

The optional `tnfr.primality` adapter retains access to the basic utility.
`tnfr.factorization` and the SDK factorization methods are deprecated bridges
for a separately supplied legacy package. The former in-repository
factorization lab and advanced primality adapters were retired on 2026-11-06;
see the [recovery index](../theory/research/archive/README.md#recovery-records).
The main engine wheel and source distribution exclude this tree.

Only the basic primality utility remains in the maintained application test
selection in [TESTING.md](../TESTING.md). The
[number-theory document](../theory/TNFR_NUMBER_THEORY.md) owns mathematical
scope; exact arithmetic identities, Paley/CRT results and their core tests
remain maintained.
