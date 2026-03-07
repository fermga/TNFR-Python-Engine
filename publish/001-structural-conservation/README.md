# Structural Conservation Law — Zenodo Package

**A Structural Continuity Law for Grammar-Constrained Dynamics in TNFR**

> Under grammatical constraints U1–U6, the TNFR nodal dynamics induce a
> structural continuity law with an associated conserved charge and current.
> In reproducible simulations across several graph topologies, conservation
> holds under valid operator sequences and degrades measurably when
> grammatical constraints are violated.

## Quick Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the central experiment (generates results/ folder)
python src/run_conservation_experiment.py

# 3. Verify claims programmatically
python -m pytest tests/ -v
```

All figures and tables in `results/` are regenerated from scratch by step 2.

## Package Contents

```
publish/001-structural-conservation/
├── README.md                         # This file
├── CITATION.cff                      # Machine-readable citation
├── requirements.txt                  # Pinned dependencies
├── preprint.md                       # Preprint source (Markdown + LaTeX)
├── src/
│   └── run_conservation_experiment.py  # Single reproducible experiment
├── tests/
│   └── test_conservation_claims.py   # Assertion-level claim verification
└── results/                          # Generated outputs (git-ignored)
    ├── metrics.csv                   # Per-topology summary table
    └── figures/                      # PNG figures for the preprint
        ├── charge_vs_time.png
        ├── energy_vs_time.png
        ├── valid_vs_invalid.png
        └── topology_summary.png
```

## System Requirements

- Python ≥ 3.10
- `tnfr` package (this repository or `pip install tnfr`)
- `networkx`, `numpy`, `matplotlib` (installed with tnfr)

## Experiment Design

**Topologies** (5 graphs, fixed seeds):
- Path graph (N=20)
- Cycle graph (N=20)
- Small grid (5×5)
- Binary tree (depth=4)
- Erdős–Rényi (N=25, p=0.3, seed=42)

**Arms** (3 regimes per topology):
1. **Valid**: Simplified nodal evolution step (phase + diffusive ΔNFR relaxation)
2. **Perturbed**: Valid evolution + i.i.d. Gaussian phase noise (σ = 0.01)
3. **Invalid**: Grammar-violating control (U2 + U3 deliberately broken)

**Metrics**:
- Relative charge drift: |Q(t) − Q(0)| / |Q(0)|
- Lyapunov stability %: fraction of steps with dE/dt ≤ 0
- Mean conservation quality (RMS balance residual)
- Mean Lyapunov energy derivative: dE/dt

## Claim Scope

This work establishes a **mathematically explicit, computationally
reproducible conservation principle internal to the TNFR formalism**.

It does **not** claim:
- A universal physical law
- Equivalence to Noether's theorem in physics
- Any connection to the Riemann Hypothesis, cosmology, or consciousness

## Citation

See [CITATION.cff](CITATION.cff) for machine-readable metadata.

## License

MIT — same as the parent TNFR-Python-Engine repository.
