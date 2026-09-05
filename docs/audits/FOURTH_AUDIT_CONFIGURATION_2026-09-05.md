# Fourth audit: configuration, initialization, and reproducibility

Scope: graph configuration, initialization, mathematical feature flags, preset
ownership, legacy default imports, and offline secure-configuration parsing.
The nodal contract requires `νf=0` to remain inactive; a fixed seed must preserve
the specified initialization stream. No operator sequence or physical default
value was changed. SDK, persistence, cache implementation, and theory changes
belong to separate audit tracks.

## Reproduced failures and corrections

1. **P1 — Initialization replaced existing legacy values, including zero νf.**
   With seed `37` and node attributes `phase=0.7, nu_f=0, psi=0.4,
   sense_index=0`, `init_node_attrs(..., override=False)` added higher-priority
   canonical attributes, including `νf=5.531009690742192`. This activated a node
   that the caller had explicitly left inactive. Initialization now uses the
   existing alias definitions and shared write helper. False overrides preserve
   the original mapping; true overrides update the first existing accepted
   alias. Existing seeded draw order is unchanged for canonical input mappings.
   See [initialization.py](../../src/tnfr/initialization.py).

2. **P1 — Mutable fallback and preset state leaked across experiments.**
   Changing `get_param(empty_graph, "COHERENCE")["weights"]["phase"]` changed
   subsequent default reads in unrelated graphs. A caller could also append
   operators or change a nested THOL block's repeat count in `get_preset` and
   thereby alter every future call. Canonical default lookup now delegates to
   the owned-copy fallback; instances snapshot supplied defaults. Presets return
   deep copies, retaining the original sequence and nested block identity within
   each independent result. Graph-owned mutable configuration remains mutable.
   See [configuration facade](../../src/tnfr/config/__init__.py),
   [TNFRConfig](../../src/tnfr/config/tnfr_config.py), and
   [presets.py](../../src/tnfr/config/presets.py).

3. **P1 — Empty configurations reset values, and invalid updates were partial.**
   Applying `{}` to a graph with `DT=0.125, RANDOM_SEED=37` replaced them with
   `0.5` and `0`. Truth-value fallback also ignored an instance's explicit empty
   defaults mapping. Separately, setting `VF_MIN=2` on a graph whose `VF_MAX=1`
   succeeded because only the incoming fragment was checked. Unknown override
   keys were discovered after earlier values had already been written. One
   staged update path now validates the effective configuration before writes;
   absent defaults and empty defaults have different semantics. The backend
   service configurator also validates all field names before assigning any,
   and rejects method replacement such as `get_backend_config=False`.

4. **P2 — Scalar conversion accepted invalid states and rejected valid zero steps.**
   NaN/Infinity passed bounds checks; invalid strings produced inconsistent
   exceptions, and a negative `VF_MAX` without `VF_MIN` was accepted. `DT=0`
   failed configuration validation despite the integrators' no-op contract.
   Shared finite-real validation now covers the implemented bounds and numeric
   initialization fields; zero steps and nonnegative frequency limits are valid.
   Invalid initialization values and non-integral supplied seeds fail before
   node writes. `RANDOM_SEED=None` selects a local unseeded initialization RNG.
   The existing initializer's annotated optional seed previously reached
   `int(None)` and failed. No deterministic-trajectory claim is made for None.

5. **P2 — Flag overrides leaked across threads/tasks; boolean parsers disagreed.**
   A worker thread observed another thread's temporary math-dynamics override.
   The global stack also restored incorrect settings when asynchronous contexts
   interleaved. Context-local token restoration replaces that stack. Meanwhile
   `bool("false")` enabled graph flags, and backend/TLS parsers treated
   `" true "` as false. A shared
   [boolean parser](../../src/tnfr/config/parsing.py) preserves explicit false
   spellings and whitespace handling. Existing environment fallback policy is
   retained for math flags; invalid backend environment flags now take the
   existing warning/default branch. Invalid Redis TLS flags are rejected.

6. **P2 — Redis configuration and its security report contradicted each other.**
   Synthetic percent-encoded passwords were returned encoded, while reserved
   characters in individual passwords broke constructed-URL validation. IPv6
   hosts were not bracketed for that validation. Both database input paths
   accepted negative indices and raised different errors for invalid text.
   Parsing now decodes URL passwords, escapes constructed user information,
   brackets IPv6 authorities, and shares nonnegative database validation.
   The security auditor reads the same effective configuration: a credentialed
   `rediss://` URL no longer receives false missing-password/TLS reports, and an
   overriding plaintext URL is no longer hidden by unrelated individual TLS
   settings. All tests are offline with synthetic values; no Redis connection
   or real credential was accessed. See
   [security.py](../../src/tnfr/config/security.py).

7. **P3 — Legacy modules duplicated canonical default definitions.**
   `constants/init.py` and `constants/metric.py` were full duplicate definitions.
   Baseline public values matched; class and mapping identities differed.
   They now re-export the canonical definitions while retaining every legacy
   public name, including historical helper imports. Identity and public-symbol
   parity are covered. This removed duplicated definitions without selecting
   different physical values.

## Validation

- Initial focused baseline: **98 passed**, covering structural triad, nodal
  equation, and integrator numerics.
- The first reproduction batch produced **38 failures and 4 passing controls**
  before source corrections. Further batches reproduced preset, environment,
  URL-reporting, and initialization input failures before their fixes.
- Final configuration/security regressions: **77 passed**. With alias and
  integrator regressions: **163 passed**.
- An intermediate broad live-tree run: **3,071 passed, 11 skipped, 96 warnings,
  1 failed** in 106.16 seconds. The one failure was the concurrently developed
  SDK invalid-node-generation test's expected exception type; it did not arise
  in configuration. The root audit owns final integrated validation.
- No change to `manual/`, dependencies, canonical physical constants, or RNG
  hashing/cache algorithms.

## Scope and remaining boundaries

- Initializer reproducibility assumes the same graph node iteration order and
  options, as before. None is an initialization-only unseeded option; runtime
  jitter and some operators still require an integer graph seed. Persisting one
  sampled seed across all runtime consumers is a separate ownership decision.
- Exported nested defaults remain low-level shared definitions. Supported
  accessor/injection paths isolate their mutable values; this is not a deep
  freeze of every exported dictionary in the package.
- Graph configuration and backend service settings remain separate APIs with
  different defaults (`DT=0.5/euler` versus the service recommendation
  `dt=0.1/rk4`). Service settings do not rewrite a graph's integration settings.
  Backend field-name validation is not a complete per-field value schema.
- Base environment flags are cached on first read; runtime precision modes
  remain deliberately process-wide. Context isolation applies to temporary
  mathematical flags, not arbitrary concurrent mutation of global settings.
- Finite scalar/bounds validation does not prove ΔNFR semantics, grammar
  compliance, unit correctness, convergence, or physical stability. Docstrings
  now state the actual numerical scope of these checks.
- Redis configuration remains its existing URL/path and individual-variable
  interface, not a full Redis client URL implementation (for example, no new
  query-option or ACL-user contract was introduced).
