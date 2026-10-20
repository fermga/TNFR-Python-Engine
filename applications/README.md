# Arithmetic utility and application archive

These source projects reuse selected TNFR utilities for supplied arithmetic
problems. They are separate from the core engine under `src/tnfr` and from the
active generative research queue in the
[execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md).
Their formulas, heuristic scores and compatibility names do not establish
physical emergence or a new primality/factorization complexity result.

| Project | Retained purpose | Boundary |
| --- | --- | --- |
| [Primality utility](primality-test/README.md) | Maintained standalone arithmetic-pressure descriptors, numerical predicate and basic CLI | Consumes divisor/factor statistics; does not generate primes through nodal dynamics |
| [Factorization lab archive](factorization-lab/README.md) | Non-maintained source and recorded outputs retained at their original paths | Known acceptance and completeness defects prevent using its certificates as factorization evidence |

The optional `tnfr.primality` adapter retains access to the basic utility.
`tnfr.factorization` and the SDK factorization methods are deprecated bridges
for a separately supplied legacy package; they do not activate the archived
checkout. Generic manifest, observation and dry-run orchestration belongs to
the engine and no longer needs the lab. The main engine wheel and source
distribution exclude both application trees.

Only the basic primality utility remains in the maintained application test
selection in [TESTING.md](../TESTING.md). The lab and advanced primality
adapters are historical material, with source identities and migration guidance
in their respective guides. They create no active research campaign. The
[number-theory document](../theory/TNFR_NUMBER_THEORY.md) owns mathematical scope.
Exact arithmetic identities, Paley/CRT results and their independent core tests
remain maintained; retiring these applications changes none of those proofs.

Historical JSON captures and the lab's `notebooks/archive/` retain their original
bytes, including recorded old paths. They are evidence of the recorded run,
not current invocation instructions or validation of the current source.
