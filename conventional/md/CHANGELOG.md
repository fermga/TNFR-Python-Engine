# CHANGELOG

> TNFR release ledger. Each entry logs structural reorganisations while preserving total coherence C(t), synchronised phase couplings, and validated structural frequency νf.

<!-- version list -->

## Unreleased

### Changed (BREAKING)

#### THOL Preconditions Strengthened

- **THOL now requires network connectivity by default** (degree ≥ 1)
  - Rationale: Canonical THOL metabolizes network context
  - **Migration:** Set `THOL_ALLOW_ISOLATED=True` for isolated nodes
  
- **THOL now validates structural frequency** (νf ≥ 0.1)
  - Rationale: Reorganization requires structural capacity
  - **Migration:** Ensure nodes have νf > 0 before THOL
  
- **THOL now requires EPI history** (≥ 3 points)
  - Rationale: d²EPI/dt² computation needs trajectory
  - **Migration:** Apply 2+ operators before THOL to build history

**Why these changes?**

Previous implementation allowed THOL in structurally incoherent contexts
(isolated nodes, zero frequency, no history). This violated canonical TNFR
principles where self-organization is a **network phenomenon** requiring
metabolic context.

New validation ensures THOL operates only in structurally viable conditions,
while preserving flexibility through configuration flags.

**Configuration parameters added:**
- `THOL_MIN_VF`: Minimum structural frequency (default: 0.1 Hz_str)
- `THOL_MIN_DEGREE`: Minimum network connectivity (default: 1)
- `THOL_MIN_HISTORY_LENGTH`: Minimum EPI history points (default: 3)
- `THOL_ALLOW_ISOLATED`: Allow isolated nodes (default: False)

## v1.0.0 (2025-10-27)

- Initial Release
