# ⚠️ DEPRECATED: U6 Temporal Ordering Experimental Scripts

**STATUS**: **DEPRECATED** - This experimental U6 "Temporal Ordering" research has been **superseded** by the canonical **U6: STRUCTURAL POTENTIAL CONFINEMENT** (promoted 2025-11-11).

**Migration**: See [docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](../docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md) for the canonical U6 specification.

---

## Historical Context

This directory contained experimental scripts to validate the U6 (Temporal Ordering) **research proposal** that explored τ_relax-based spacing between destabilizers.

**Why Deprecated**: After 2,400+ experiments, the structural potential field (Φ_s) approach demonstrated superior predictive power (corr = -0.822, R² ≈ 0.68) across 5 topology families, leading to its promotion as the canonical U6 constraint.

**Temporal Ordering Status**: Remains a valid **research direction** but is **NOT canonical**. The τ_relax-based approach may be revisited as a future U7 or complementary heuristic.

---

## Original Purpose (Historical)

This directory contains experimental scripts to validate the U6 (Temporal Ordering) proposal described in `docs/grammar/U6_TEMPORAL_ORDERING.md` **(now removed - see UNIFIED_GRAMMAR_RULES.md § Proposed U7 for historical context)**.

## 🔬 Status: EXPERIMENTAL (DEPRECATED)

U6 is a proposed constraint under investigation. **Not canonical** (yet). These experiments aim to:
1. Measure observed vs estimated τ_relax
2. Quantify non-linear accumulation α(Δt)
3. Correlate bifurcation index B with C(t) fragmentation

## 📝 Available Scripts

### `experiment_u6.py`

Main U6 experiment runner.

**Usage**:
```bash
# Experiment A: Measure τ_relax
python scripts/experiment_u6.py --experiment A --output results/tau_relax.json

# Experiment B: α(Δt) curves
python scripts/experiment_u6.py --experiment B --vf 1.0 --topology ring --output results/alpha_curves.json

# Experiment C: B vs C(t)
python scripts/experiment_u6.py --experiment C --output results/bifurcation_index.json

# All experiments
python scripts/experiment_u6.py --experiment all
```

**Parameters**:
- `--experiment {A,B,C,all}`: Which protocol to execute
- `--vf FLOAT`: Structural frequency (Hz_str), default=1.0
- `--topology {star,ring,grid,random}`: Network topology
- `--output PATH`: Path to save JSON results

## 📊 Experimental Protocols

### Experiment A: τ_relax Measurement

**Objective**: Validate τ_relax = (k_top/νf)·k_op·ln(1/ε)

**Method**:
1. Apply OZ (dissonance) at t=0
2. Monitor |ΔNFR(t)| and C(t)
3. Record time until recovery (|ΔNFR| < 5% initial, C > 95% initial)
4. Compare with theoretical estimate

**Output**: JSON with observations vs predictions for multiple (νf, topology).

### Experiment B: Non-Linear Accumulation α(Δt)

**Objective**: Characterize α(Δt) = (ΔNFR_actual - ΔNFR_linear) / (ΔNFR_0 · ΔNFR_before)

**Method**:
1. Apply first OZ at t=0
2. Wait Δt (vary from 0.1 to 5·τ_relax)
3. Apply second OZ
4. Measure actual ΔNFR vs linear expectation

**Expectation**: α(Δt) > 1 for Δt < τ_relax (amplification), α → 1 for Δt ≥ τ_relax.

**Output**: α vs Δt/τ_relax normalized curves.

### Experiment C: Bifurcation Index B

**Objective**: Establish B_crit for fragmentation prediction

**Method**:
1. Execute sequences with different temporal spacing
2. Calculate B = (1/νf²)|∂²EPI/∂t²| at each step
3. Measure C(t) drop correlated with B peaks
4. Determine threshold B_crit (provisional: B > 3.0 → critical)

**Output**: B(t) and C(t) trajectories for different sequences.

## 🎯 Validation Objectives

For U6 to achieve STRONG canonicity (60-80% confidence):

✅ **Formal derivation** from nodal equation (show divergence without spacing)  
✅ **Parameter endogenization** (k_top from spectral analysis, k_op from energetics)  
✅ **Statistical validation** (>80% violations → fragmentation)  
✅ **Independence** (cases that pass U2+U4 but fail only due to U6)

## 📖 References

- **Complete specification**: `docs/grammar/U6_TEMPORAL_ORDERING.md`
- **Prior research**: `docs/research/U6_INVESTIGATION_REPORT.md`
- **Unified grammar**: `UNIFIED_GRAMMAR_RULES.md` § Proposed Constraints
- **TNFR invariants**: `AGENTS.md`

## ⚠️ Important Note

U6 is in research phase. Current implementation:
- ✅ Adds optional experimental validation (`GrammarValidator(experimental_u6=True)`)
- ✅ Generates warnings, does not fail validation (does not block execution)
- ✅ Records telemetry for analysis
- ❌ No es canónica (no se aplica por defecto)

**No usar U6 en producción hasta completar validación y elevar a STRONG.**

---

**Última actualización**: 2025-11-11  
**Estado**: 🔬 Experimental  
**Confianza**: ~55% (físicamente motivada, pendiente formalización completa)
