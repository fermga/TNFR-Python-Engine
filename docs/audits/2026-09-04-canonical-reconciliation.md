# Canonical reconciliation — 2026-09-04 handoff packet vs current branch

**Workstream:** C0 (reconcile the current branch with the historical audit).
**Method:** ran the packet's reproducible audit
(`TNFR_revised_handoff_packet_2026-09-04/evidence/baseline_0.0.3.5/tnfr_canonicity_audit.py`)
against the current `HEAD` with `--repo .`, plus the targeted regression suites.
The audited baseline is the `0.0.3.5` ZIP; its historical measurements are **not**
assumed to describe the current branch.

## Environment

| Item | Value |
|---|---|
| Baseline | tnfr 0.0.3.5 (ZIP), seed 20260904 |
| Current branch | `main` @ `ed4b289f` (pre-C4/C5) |
| Python | 3.13 |

## Reconciliation matrix

| Issue | Baseline | Current | Evidence |
|---|---|---|---|
| **C1** directed outgoing transport | OPEN (used `L_in`) | **FIXED** | directed outgoing residual `4.44e-16`; incoming residual `1.605` rules out the old orientation |
| **C1** edge weights in fused ΔNFR | OPEN (ignored) | **FIXED** | undirected weighted residual vs weighted `L_rw` = `1.11e-16`; vs unweighted kernel = `0.518` (weights applied) |
| **C1** undirected unweighted identity | PASS | **FIXED** | residual `0.0` |
| **C2** U3 mandatory on public paths | OPEN (flag-gated) | **FIXED** | `tests/operators/test_u3_hard_invariant.py` green (6) |
| **C3** single ξ_C semantics | OPEN (NaN-prone) | **FIXED** | `tests/physics/test_xi_c_parity.py` green (3) |
| **C4** basis-invariant spectral observables | OPEN (basis-dependent `η²>0.9`) | **PARTIAL → FIXED (this session)** | near-exact subspace certificate 53/53 correct vs `0.9` threshold 46/53; Example 122 pattern; reusable module added (PR-04) |
| **C5** claim/manifest/circularity infra | OPEN | **FIXED (this session)** | `src/tnfr/research/` added (PR-05) |
| dependency envelope | WARNING (`cachetools 7.x` vs `<7`) | resolved | `cachetools>=5.0,<8.0` in `pyproject.toml` |
| prime signature (QR rank-3 ⟺ prime) | PASS | PASS | 0 mismatches over odd `5..119` |

## Status classification

- **FIXED**: C1 (orientation + weights + undirected), C2, C3 — done in the
  prior canonicity-audit sessions (commit history through `ed4b289f`) and
  re-verified here on the current branch.
- **PARTIAL → addressed this session**: C4 — the Example 122 pattern was
  present; a reusable, tested module
  (`src/tnfr/physics/spectral_projectors.py`, `spectral_certificates.py`) now
  generalizes it (PR-04).
- **OPEN → addressed this session**: C5 — `src/tnfr/research/` (claims,
  manifests, certificates, circularity) added (PR-05).

## Exit criterion

The physics-to-code transport identity (`ΔNFR_epi = −L_rw · EPI`) is closed for
every declared graph class (directed/undirected × weighted/unweighted), so the
new research lines R1–R9 are unblocked.
