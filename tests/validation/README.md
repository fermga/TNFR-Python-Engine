# N-Body Quantitative Validation Suite

## Overview

This validation suite implements the experiments described in [09_classical_mechanics_numerical_validation.md](../../docs/source/theory/09_classical_mechanics_numerical_validation.md) to rigorously validate TNFR n-body dynamics against classical mechanics predictions.

## Location

- **Tests**: `tests/validation/test_nbody_validation.py`
- **Example Script**: `examples/nbody_quantitative_validation.py`
- **Outputs**: `validation_outputs/` (generated on script execution)

## Experiments Implemented

### Experiment 1: Harmonic Oscillator Mass Scaling
**Validates**: m = 1/νf relationship  
**Acceptance**: Period error < 0.1%  
**Outputs**: Phase portraits for different νf values

### Experiment 2: Conservation Laws
**Validates**: Energy, momentum, and angular momentum conservation  
**Acceptance**: Conservation error < 10⁻⁶  
**Outputs**: Conservation drift plots over time

### Experiment 3: Kepler Two-Body Orbits
**Validates**: Agreement with Kepler's laws  
**Acceptance**: Conservation < 10⁻⁶  
**Outputs**: 3D trajectories and orbital plots

### Experiment 4: Three-Body Lagrange Configuration
**Validates**: Stability of three-body configurations  
**Acceptance**: Energy drift < 10% (chaotic system)  
**Outputs**: 3-body trajectories and energy evolution

### Experiment 5: Chaos Detection (Lyapunov Exponents)
**Validates**: Chaotic dynamics detection  
**Acceptance**: Positive Lyapunov exponent  
**Outputs**: Trajectory divergence and Lyapunov fit

### Experiment 6: Coherence Metrics (C(t) and Si)
**Validates**: Coherence tracking for structural multitudes  
**Acceptance**: C(t) stable for conservative systems  
**Outputs**: Coherence evolution and energy components

## Running the Validation Suite

### Run Tests
```bash
# Run all validation tests
pytest tests/validation/test_nbody_validation.py -v -s

# Run specific experiment
pytest tests/validation/test_nbody_validation.py::TestExperiment1HarmonicMassScaling -v -s
```

### Run Example Script (with visualizations)
```bash
cd /path/to/TNFR-Python-Engine
python examples/nbody_quantitative_validation.py
```

This will:
1. Execute all 6 validation experiments
2. Generate quantitative error tables
3. Create visualization outputs in `validation_outputs/`
4. Print summary report

## Reproducibility

All experiments use a fixed random seed:
```python
SEED = 42
np.random.seed(SEED)
```

This ensures:
- Identical results across runs
- Reproducible figures and tables
- Deterministic chaos measurements

## Parameters and Acceptance Criteria

### Experiment 1: Harmonic Oscillator
- **νf values**: [0.5, 1.0, 1.5, 2.0] Hz_str
- **Stiffness**: k = 1.0
- **Time step**: dt = 0.01
- **Duration**: t_sim = 100.0
- **Acceptance**: |T_num - T_theo| / T_theo < 0.001

### Experiment 2: Conservation Laws
- **System**: 2-body circular orbit
- **Masses**: [1.0, 0.1]
- **Time step**: dt = 0.01
- **Duration**: t_sim = 100.0
- **Acceptance**:
  - Energy: |ΔE/E| < 10⁻⁶
  - Momentum: |ΔP| < 10⁻⁶
  - Angular momentum: |ΔL/L| < 10⁻⁶

### Experiment 3: Kepler Orbits
- **System**: 2-body circular orbit
- **Masses**: [1.0, 0.1]
- **Time step**: dt = 0.005
- **Duration**: 3 orbital periods
- **Acceptance**:
  - Energy: |ΔE/E| < 10⁻⁵
  - Angular momentum: |ΔL/L| < 10⁻⁵

### Experiment 4: Three-Body
- **System**: Equilateral triangle, equal masses
- **Masses**: [1.0, 1.0, 1.0]
- **Time step**: dt = 0.005
- **Duration**: t_sim = 5.0
- **Acceptance**: |ΔE/E| < 0.10 (10%)

### Experiment 5: Chaos (Lyapunov)
- **System**: Perturbed 3-body
- **Perturbation**: δ = 10⁻⁸
- **Time step**: dt = 0.01
- **Duration**: t_measure = 10.0
- **Acceptance**: λ > 0 (positive exponent)

### Experiment 6: Coherence
- **System**: 2-body circular orbit
- **Time step**: dt = 0.01
- **Duration**: t_sim = 50.0
- **Acceptance**:
  - Mean coherence: C_mean > 0.99
  - Coherence std: C_std < 0.01

## Generated Outputs

All outputs are saved to `validation_outputs/` (this directory is in `.gitignore` and will be created when you run the script):

1. `exp1_harmonic_phase_portraits.png` - Phase portraits for different νf
2. `exp2_conservation_laws.png` - Energy, momentum, angular momentum conservation
3. `exp3_kepler_trajectories.png` - 3D and 2D Keplerian orbits
4. `exp4_three_body.png` - Three-body trajectories and energy evolution
5. `exp5_lyapunov.png` - Trajectory divergence and Lyapunov exponent fit
6. `exp6_coherence.png` - Coherence metrics and energy components

**Note**: The `validation_outputs/` directory is generated when you run the example script and is not tracked in version control.

## TNFR Canonical Invariants Validated

All experiments verify:

1. **EPI encoding**: Position and velocity properly stored as EPI components
2. **νf = 1/m**: Structural frequency inversely related to mass
3. **ΔNFR semantics**: Reorganization gradient drives evolution via nodal equation
4. **Operator closure**: Evolution preserves TNFR structural properties
5. **Phase verification**: Synchrony checked for couplings
6. **Controlled determinism**: Reproducible with seeds
7. **Structural metrics**: C(t) and Si properly tracked

## Integration with CI/CD

The validation suite can be integrated into continuous integration:

```yaml
# .github/workflows/validation.yml
- name: Run N-Body Validation Suite
  run: |
    pytest tests/validation/test_nbody_validation.py -v
```

## References

1. **Validation Document**: [09_classical_mechanics_numerical_validation.md](../../docs/source/theory/09_classical_mechanics_numerical_validation.md)
2. **N-Body Module**: `src/tnfr/dynamics/nbody.py`
3. **TNFR Theory**: `TNFR.pdf` (repository root)
4. **AGENTS.md**: Canonical invariants §3

## Future Extensions

Planned additions:
- Poincaré section generation for forced systems
- Bifurcation diagram generator
- Sense index heatmaps for parameter sweeps
- Multi-scale validation (many-body systems)
- GPU-accelerated parameter space exploration

## Troubleshooting

### Tests fail with energy drift
- Check time step is small enough (try halving dt)
- Verify symplectic integrator is being used
- Ensure no numerical overflow in forces

### Lyapunov exponent is negative
- Three-body systems are chaotic but may not show it immediately
- Try longer measurement time
- Increase perturbation magnitude slightly

### Plots not generated
- Ensure matplotlib is installed: `pip install 'tnfr[viz-basic]'`
- Check write permissions for `validation_outputs/`
- Run in non-headless mode for interactive display

## Contact

For questions about the validation suite:
- Open an issue on GitHub
- Reference validation document in issue
- Include output logs and parameter values
