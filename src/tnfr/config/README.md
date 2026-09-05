# TNFR Config — Canonical Module Hub (Single Source of Truth)

English-only hub for configuration schema, defaults, and loaders.

- Canonical physics/invariants: `AGENTS.md`
- Grammar: `UNIFIED_GRAMMAR_RULES.md`
- Computational hub: `src/tnfr/mathematics/README.md`

## Scope
- Default settings, environment overrides, secure config hooks

## Guarantees
- Traceable, minimal config surface; no theory duplication

## Ownership and parsing

`get_param` returns a graph-owned value when present, and an isolated copy of a
mutable default otherwise. `TNFRConfig(defaults=...)` snapshots its supplied
defaults. `inject_defaults` and `merge_overrides` stage updates and validate the
effective configuration before changing graph parameters; an explicit empty
mapping injects no parameter values. Low-level exported default dictionaries
are shared definitions and should be treated as read-only.

Boolean strings use explicit true/false spellings, ignoring case and surrounding
whitespace. Unknown strings raise in direct configuration, keep the documented
fallback for math environment flags, and produce a warning with the existing
default retained for backend environment settings. `context_flags` overrides
are local to their execution context, including nested asynchronous tasks.

Graph defaults and `backend_config.TNFRConfig` have separate ownership. The latter
configures optional services; its recommended integration settings do not
replace a graph's `DT`/`INTEGRATOR_METHOD`. Runtime precision settings remain
process-wide. Neither configuration validation nor a finite ΔNFR scalar proves
grammar compliance or a model's physical semantics.
