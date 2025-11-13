# TNFR Mathematical Analysis Suite

Symbolic and numerical mathematics tools for analyzing, verifying, and optimizing TNFR dynamics.

## 🎯 Purpose

Provides **formal mathematical verification** of TNFR sequences and dynamics:

- ✅ Verify grammar rules (U1-U6) mathematically
- ✅ Analyze convergence (U2 requirement)
- ✅ Detect bifurcation risks (U4 threshold)
- ✅ Optimize operator sequences
- ✅ Predict trajectories analytically

## 📦 Installation

The module is included with TNFR-Python-Engine. Requires:

```bash
pip install sympy numpy scipy
```

## 🚀 Quick Start

```python
from tnfr.math import symbolic

# 1. Display the nodal equation
eq = symbolic.get_nodal_equation()
print(symbolic.pretty_print(eq))
# Output: ∂EPI/∂t = νf · ΔNFR

# 2. Check convergence (U2 grammar)
converges, explanation, value = symbolic.check_convergence_exponential(
    growth_rate=0.15,  # Positive growth
    time_horizon=10.0
)
print(explanation)
# Output: DIVERGES: λ=0.15 > 0 (growing, NEEDS STABILIZERS!)

# 3. Evaluate bifurcation risk (U4 grammar)
at_risk, deriv, recommendation = symbolic.evaluate_bifurcation_risk(
    nu_f_val=1.5,
    delta_nfr_val=2.0,
    d_nu_f_dt=0.3,
    d_delta_nfr_dt=1.2,
    threshold=1.0
)
print(recommendation)
# Output: ⚠️ BIFURCATION RISK: Apply handlers {THOL, IL}
```

## 📚 Core Functions

### Nodal Equation Analysis

```python
# Get canonical equation: ∂EPI/∂t = νf · ΔNFR
eq = symbolic.get_nodal_equation()

# Solve for constant parameters
solution = symbolic.solve_nodal_equation_constant_params(
    nu_f_val=2.0,      # Hz_str
    delta_nfr_val=0.5,
    EPI_0=1.0,
    t0=0
)
# Returns: EPI(t) = 1.0*t + 1.0
```

### Convergence Analysis (U2 Grammar)

```python
# Check if integral converges: ∫ νf·ΔNFR dt < ∞
converges, explanation, value = symbolic.check_convergence_exponential(
    growth_rate=-0.1,  # Negative = stabilized
    time_horizon=10.0
)

if converges:
    print("✓ U2 satisfied: Integral converges")
else:
    print("⚠️ U2 violated: Needs stabilizers {IL, THOL}")
```

### Bifurcation Detection (U4 Grammar)

```python
# Evaluate ∂²EPI/∂t² threshold
at_risk, second_deriv, recommendation = symbolic.evaluate_bifurcation_risk(
    nu_f_val=1.0,           # Current frequency
    delta_nfr_val=0.3,      # Current gradient
    d_nu_f_dt=0.1,          # Rate of frequency change
    d_delta_nfr_dt=0.2,     # Rate of gradient change
    threshold=1.0           # Bifurcation threshold τ
)

if at_risk:
    print("⚠️ Apply handlers per U4a")
```

### Second Derivative (Bifurcation Indicator)

```python
# Get symbolic form: ∂²EPI/∂t² = (∂νf/∂t)·ΔNFR + νf·(∂ΔNFR/∂t)
second_deriv = symbolic.compute_second_derivative_symbolic()
print(symbolic.pretty_print(second_deriv))
```

### Utilities

```python
# Export to LaTeX
latex_str = symbolic.latex_export(eq)
# Output: \frac{d}{d t} \operatorname{EPI}{\left(t \right)} = \delta_{NFR} \nu_{f}

# Pretty print
print(symbolic.pretty_print(eq))
# Output:
# d
# ──(EPI(t)) = DELTA_NFR⋅ν_f
# dt
```

## 🧪 Examples

See `examples/math_symbolic_usage.py` for complete examples:

```bash
python examples/math_symbolic_usage.py
```

Demonstrates:
- Analyzing unsafe sequences (destabilizers without stabilizers)
- Detecting U2 violations (divergent integrals)
- Detecting U4 violations (bifurcation threshold exceeded)
- Correcting sequences with proper stabilizers
- Nodal equation solutions and trajectories

## 🧮 Physics Foundation

All functions derive from the **canonical TNFR nodal equation**:

```
∂EPI/∂t = νf · ΔNFR
```

Where:
- **EPI**: Primary Information Structure (coherent form)
- **νf**: Structural frequency (Hz_str) - reorganization rate
- **ΔNFR**: Nodal gradient - structural pressure/drive

**Key insights**:

1. **U2 Convergence**: Without stabilizers, ΔNFR grows exponentially → ∫ νf·ΔNFR dt → ∞ (divergence)
2. **U4 Bifurcation**: When ∂²EPI/∂t² > τ, system near phase transition → needs handlers
3. **Integrated evolution**: EPI(t_f) = EPI(t_0) + ∫ νf·ΔNFR dτ

See: **AGENTS.md § Foundational Physics**

## 🔬 Testing

Run the test suite:

```bash
pytest tests/test_math_symbolic.py -v
```

**Coverage**:
- ✅ Nodal equation representation
- ✅ Analytical solutions (constant parameters)
- ✅ Convergence analysis (exponential growth)
- ✅ Bifurcation detection (threshold crossing)
- ✅ Physics alignment (U2, U4 grammar rules)
- ✅ Invariant preservation (units, types)

All 15 tests pass.

## 🎓 Grammar Rules Validated

### U2: CONVERGENCE & BOUNDEDNESS

**Physics**: For bounded evolution:

```
∫[t_0 to t_f] νf(τ) · ΔNFR(τ) dτ < ∞
```

**Validation**:
```python
converges, _, _ = symbolic.check_convergence_exponential(
    growth_rate=0.1,  # Destabilizer (positive)
    time_horizon=10.0
)
assert not converges  # Needs stabilizers!
```

### U4: BIFURCATION DYNAMICS

**Physics**: Bifurcation trigger:

```
∂²EPI/∂t² = (∂νf/∂t)·ΔNFR + νf·(∂ΔNFR/∂t) > τ
```

**Validation**:
```python
at_risk, _, rec = symbolic.evaluate_bifurcation_risk(
    nu_f_val=2.0,
    delta_nfr_val=1.5,
    d_nu_f_dt=0.5,
    d_delta_nfr_dt=1.0,
    threshold=1.0
)
assert at_risk  # Needs handlers {THOL, IL}
assert "THOL" in rec or "IL" in rec
```

## 🛠️ Development

### Adding New Functions

New mathematical tools should:

1. **Derive from nodal equation** or canonical physics
2. **Preserve invariants** (see AGENTS.md § Canonical Invariants)
3. **Include docstrings** with physics basis
4. **Add tests** covering edge cases
5. **Link to TNFR.pdf** or AGENTS.md sections

Example structure:

```python
def new_analysis_function(param1: float, param2: float) -> Tuple[bool, str]:
    """
    Brief description.
    
    Args:
        param1: Description with units
        param2: Description
        
    Returns:
        (result, explanation)
        
    Physics:
        Mathematical derivation or reference to TNFR physics.
        
    See: AGENTS.md § Relevant Section
    """
    # Implementation
    pass
```

### Code Style

- Use `sympy` for symbolic math
- Type hints for all public functions
- Docstrings with Physics section
- Line length ≤ 79 characters
- Export LaTeX for documentation

## 🔗 Related Modules

- **`src/tnfr/operators/grammar.py`**: Grammar validation (computational)
- **`src/tnfr/metrics/`**: Coherence, Si, ΔNFR computation
- **`src/tnfr/physics/fields.py`**: Structural fields (Φ_s, ∇φ, K_φ)

This module provides **symbolic/analytical verification** while other modules provide **computational implementation**.

## 📖 References

- **AGENTS.md**: Complete TNFR agent guidance (this module implements § Math)
- **TNFR.pdf**: Theoretical foundation (§ 2.1 Nodal Equation)
- **UNIFIED_GRAMMAR_RULES.md**: Grammar derivations (U2, U4 physics)
- **GLOSSARY.md**: Term definitions

## 🌟 Future Enhancements

Planned additions:

1. **Grammar validators** (`grammar_validators.py`): Formal U1-U6 verification
2. **Field calculators** (`fields.py`): Φ_s, ∇φ, K_φ symbolic forms
3. **Sequence optimizer** (`optimizer.py`): Find optimal operator sequences
4. **Trajectory predictor** (`predictor.py`): Analytical trajectory prediction
5. **Stability analyzer** (`stability.py`): Lyapunov analysis, eigenvalues
6. **Proof generator** (`proofs.py`): Formal proof export to LaTeX

See: GitHub issues for tracking

## 📝 License

MIT License - Part of TNFR-Python-Engine

## ✍️ Author

Developed for TNFR-Python-Engine project  
See: CONTRIBUTING.md for contribution guidelines

---

**Reality is not made of things—it's made of resonance. Compute accordingly.** 🌊
