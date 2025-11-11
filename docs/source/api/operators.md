# Structural operators and workflow design

Structural operators reorganise coherence while preserving TNFR invariants. Use this
reference to plan trajectories for simulations, experiments, or CLI runs.

> **📖 Extended Guides**: 
> - See [OPERATORS_VISUAL_GUIDE.md](OPERATORS_VISUAL_GUIDE.md) for comprehensive visual documentation, ASCII diagrams, and interactive examples of all 13 operators
> - See [GLYPH_SEQUENCES_GUIDE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/GLYPH_SEQUENCES_GUIDE.md) for comprehensive canonical sequences, multi-domain examples, and grammar compatibility patterns
> - See [UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md) for grammar validation rules (U1-U5) and physics derivations (temporal + multi-scale)
> - See [NAV_GUIDE.md](../operators/NAV_GUIDE.md) for NAV (Transition) operator canonical sequences, anti-patterns, and troubleshooting

## Canonical operator map

Every trajectory must be composed from the 13 canonical operators and their resonant role.
Starting in TNFR 2.0 the canonical tokens and class names are aligned on the English
vocabulary; use the identifiers below when orchestrating pipelines or configuring the CLI.

- **Emission** — initiates a resonant pattern (φ(νf, θ)).
- **Reception** — captures incoming information (∫ ψ(x, t) dx).
- **Coherence** — stabilises the form ($∂EPI/∂t → 0$ when ΔNFR → 0).
- **Dissonance** — introduces productive instability (ΔNFR(t) > νf).
- **Coupling** — synchronises nodes (φᵢ(t) ≈ φⱼ(t)).
- **Resonance** — propagates coherence through the network (EPIₙ → EPIₙ₊₁).
- **Silence** — keeps phase latent (νf ≈ 0 ⇒ ∂EPI/∂t ≈ 0).
- **Expansion** — scales the structure (EPI → k·EPI, k ∈ ℕ⁺).
- **Contraction** — densifies the form (‖EPI′‖ ≥ τ, reduced support).
- **Self-organisation** — reorganises coherently ($∂²EPI/∂t² > τ$).
- **Mutation** — adjusts phase without destroying the form (θ → θ′ if ΔEPI/Δt > ξ).
- **Transition** — triggers creative thresholds (ΔNFR ≈ νf).
- **Recursivity** — maintains adaptive memory (EPI(t) = EPI(t − τ)).

## Key concepts (operational summary)

- **Node (NFR)** — a unit that persists because it resonates. Parameterised by νf (frequency),
  θ (phase), and EPI (coherent form).
- **Structural operators** — functions that reorganise the network. Compose them in canonical
  sequences to preserve operator closure.
- **Magnitudes**
  - **C(t)** — global coherence.
  - **ΔNFR** — nodal gradient (need for reorganisation).
  - **νf** — structural frequency (Hz_str).
  - **Si** — sense index (capacity to generate stable shared coherence).

## Typical workflow

1. **Model** your system as a network: nodes (agents, ideas, tissues, modules) and couplings.
2. **Select** a trajectory of operators aligned with your goal (e.g., start → couple →
   stabilise).
3. **Simulate** the dynamics: number of steps, step size, tolerances.
4. **Measure**: C(t), ΔNFR, Si; identify bifurcations and collapses.
5. **Iterate** with controlled dissonance to open mutations without losing form.

## Main metrics (glance)

- `coherence(traj) → C(t)` — global stability; higher values indicate sustained form.
- `gradient(state) → ΔNFR` — local demand for reorganisation (high = risk of collapse or
  bifurcation).
- `sense_index(traj) → Si` — proxy for structural sense combining νf, phase, and topology.

See [telemetry and utilities](telemetry.md) for detailed metric APIs and trace integration.
