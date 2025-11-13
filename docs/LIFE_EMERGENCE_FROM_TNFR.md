# Life Emergence from TNFR

!!! abstract "Cell Note — Life Emergence"
	- Status: Canonical hub online; code integrated; unit tests for life module passing; experiments running (Exp2, Exp3 passing; Exp1 tuning pending)
	- Date: 2025-11-13
	- Scope: Detect and characterize life-like autopoietic behavior from the TNFR nodal equation under unified grammar (U1–U6)
	- Inputs: time series of EPI, ΔNFR (internal/external), νf; graph topology; phase telemetry
	- Outputs: LifeTelemetry (Vi, A, S, M), threshold time, experiment JSON exports (outputs/life/*.json)
	- How to run:
		```pwsh
		python -m examples.life_experiments
		python -m examples.run_life_experiments_export
		```
	- Acceptance (initial):
		- Exp1 (emergence): t* ≥ 0 and A_max > 1.0
		- Exp2 (self-maintenance): C_final > 0.6 and C_std_last < 0.02
		- Exp3 (replication): Fr > 0.8
	- Safety telemetry: monitor Φ_s, |∇φ|, K_φ, ξ_C from `tnfr.physics.fields`; U6 drift stays below threshold; functions are read-only telemetry respecting operator grammar

This page is the canonical hub for the Life track: theory → math → code → experiments, all derived from the TNFR nodal equation and unified grammar.

- Physics basis: ∂EPI/∂t = νf · ΔNFR(t) with operators as the only valid structural transformations
- Target: Detect and characterize when a network manifests life-like autopoietic behavior under TNFR constraints

## Documents

- Theoretical Framework: docs/LIFE_EMERGENCE_THEORETICAL_FRAMEWORK.md
- Mathematical Derivation: docs/LIFE_MATHEMATICAL_DERIVATION.md

Both documents map directly to the operator set, preserve canonicity (U1–U6), and define metric contracts.

## API (tnfr.physics.life)

Public API is exported via `tnfr.physics` (see `src/tnfr/physics/life.py`). Prefer these centralized exports over importing private modules directly; avoid redundant implementations.

- LifeTelemetry
- compute_self_generation(epi_series, gamma, epi_max)
- compute_autopoietic_coefficient(G_epi, dEPI_dt, dnfr_external)
- compute_self_org_index(epi_series, epsilon, gamma, epi_max, d_dnfr_external_dt)
- compute_stability_margin(epi_series, epi_max)
- detect_life_emergence(times, epi_series, dEPI_dt, dnfr_external, d_dnfr_external_dt, epsilon, gamma, epi_max)

Contracts and invariants:
- No direct EPI mutation; inputs are time series/telemetry (Invariant #1)
- Units: νf in Hz_str; ΔNFR semantics preserved (Invariant #2–#3)
- Stabilizers required when destabilizers are in play (U2), phase checks for coupling (U3)

## Examples and Experiments

- Minimal demo: `examples/life_demo.py`
- Experiments: `examples/life_experiments.py`
- Export runner: `examples/run_life_experiments_export.py` → writes JSON to `outputs/life/`

Run (optional):
- python -m examples.life_experiments
- python -m examples.run_life_experiments_export

Acceptance criteria (initial):
- Life emergence (Exp1): detects threshold t* ≥ 0 with A_max > 1.0
- Self-maintenance (Exp2): final coherence C_final > 0.6, last-window std C_std < 0.02
- Replication fidelity (Exp3): Fr > 0.8

Note: Parameters are research-tunable; criteria are conservative and encoded in the exporter for reproducibility.

## Telemetry and Safety

All Life track code respects the Canonical Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) as read-only safety telemetry:
- Import from `tnfr.physics.fields` and monitor alongside life metrics
- U6 safety check (Φ_s drift) remains a soft guard; life functions don’t alter operator sequences

## Provenance

- Code: `src/tnfr/physics/life.py`, exported via `src/tnfr/physics/__init__.py`
- Tests: `tests/test_life_module.py`
- Outputs: `outputs/life/*.json` (timestamped)

## Next steps

- Add parameters table and a short tuning guide
- Integrate Life results into docs/STRUCTURAL_FIELDS_TETRAD.md cross-links
- Extend Exp1 to scan parameter grid for robust threshold detection
