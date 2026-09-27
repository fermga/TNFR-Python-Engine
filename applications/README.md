# Optional arithmetic applications

These source projects reuse selected TNFR utilities for supplied arithmetic
problems. They are separate from the core engine under `src/tnfr` and from the
active generative research queue in the
[execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md).
Their formulas, heuristic scores and compatibility names do not establish
physical emergence or a new primality/factorization complexity result.

| Project | Retained purpose | Boundary |
| --- | --- | --- |
| [Primality utility](primality-test/README.md) | Standalone arithmetic-pressure predicate, optional engine adapters and CLI | Consumes divisor/factor statistics; does not generate primes through nodal dynamics |
| [Factorization lab](factorization-lab/README.md) | Spectral candidate heuristics, arithmetic checks, partition/replay and diagnostic workflows | Candidates and configured acceptance need independent divisibility checks; supplied graphs are not an emergent substrate |

The projects moved here from the root `primality-test/` and `factorization-lab/`
directories. Existing Python package names and the optional `tnfr.primality`,
`tnfr.factorization` and SDK entry points remain available. Checkout adapters
discover this directory lazily; an installed optional package takes precedence.
The main engine wheel and source distribution exclude these projects.

These are maintained compatibility applications, not additional active research
campaigns. Select their tests explicitly using [TESTING.md](../TESTING.md).
Their package guides own installation and usage; the
[number-theory document](../theory/TNFR_NUMBER_THEORY.md) owns mathematical scope.

Historical JSON captures and the lab's `notebooks/archive/` retain their original
bytes, including recorded old paths. They are evidence of the recorded run,
not current invocation instructions. Relocation does not regenerate them or
turn their historical verdicts into current-source validation.
