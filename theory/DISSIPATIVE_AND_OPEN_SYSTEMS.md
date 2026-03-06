# Dissipative and Open Systems

The structural conservation law $\partial\rho/\partial t + \nabla\cdot\mathbf{J} = S_{\text{grammar}}$ (see [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)) holds exactly under grammar-compliant evolution. This document extends that result to **dissipative regimes** where Lindblad-type collapse operators introduce controlled decoherence.

**Status**: CANONICAL — Extends the conservation theorem to open systems.

---

## 1. Dissipative Continuity Theorem

### 1.1 Extended Continuity Equation

For a TNFR structural density $\rho$ evolving under both grammar-compliant dynamics and environmental coupling:

$$
\frac{\partial\rho_s}{\partial t} + \operatorname{div}\mathbf{J} = S_{\text{grammar}} + D[\rho]
$$

where:
- $S_{\text{grammar}} \to 0$ under U1–U6 (conservative part, from conservation.py)
- $D[\rho]$ is the Lindblad dissipator contribution (new term)

### 1.2 Lindblad Dissipator

The dissipator takes the standard Lindblad form:

$$
D[\rho] = \sum_k \left(L_k\,\rho\,L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k,\;\rho\}\right)
$$

where $L_k$ are collapse operators representing environmental coupling channels.

### 1.3 Dissipation Bound

The dissipation rate is bounded by purity:

$$
|D[\rho]| \le \sum_k \|L_k\|^2 \cdot \left(1 - \operatorname{Tr}(\rho^2)\right)
$$

**Physical consequence**: More mixed states dissipate faster. Dissipation vanishes for pure states at steady state ($\rho^2 = \rho$).

---

## 2. Purity Decay

### 2.1 Purity Evolution

The purity $P = \operatorname{Tr}(\rho^2)$ evolves as:

$$
\frac{dP}{dt} = 2\operatorname{Tr}(\rho \cdot D[\rho])
$$

with the bound:

$$
\left|\frac{dP}{dt}\right| \le 2\sum_k \|L_k\|^2 \cdot P \cdot \left(1 - \frac{P}{d}\right)
$$

where $d$ is the system dimension. Purity decreases monotonically toward $1/d$ (maximally mixed state), measuring information loss to the environment.

### 2.2 Charge Leak Rate

The Noether charge $Q = \operatorname{Tr}(\rho \cdot O_Q)$ changes at a rate bounded by the dissipator strength and initial coherence.

---

## 3. Entropy Production

### 3.1 Von Neumann Entropy

$$
S = -\operatorname{Tr}(\rho\ln\rho)
$$

Under Lindblad evolution, $S$ increases monotonically, quantifying irreversibility:

$$
\frac{dS}{dt} \ge 0
$$

Equality holds only at the steady state $\mathcal{L}[\rho_{ss}] = 0$.

### 3.2 Steady-State Conservation

At the fixed point $\mathcal{L}[\rho_{ss}] = 0$, a *modified* conservation law holds with $D[\rho_{ss}]$ absorbed into the definition of conserved quantities.

---

## 4. Dissipation Tiers

Empirical classification of dissipation strength based on purity loss and entropy gain:

| Tier | Purity loss threshold | Physical regime |
|------|----------------------|-----------------|
| **Weak** | $< 0.001$ | Near-conservative; grammar almost exactly satisfied |
| **Moderate** | $< 0.05$ | Controlled decoherence; structural identity preserved |
| **Strong** | $< 0.2$ | Significant information loss; approaching fragmentation |
| **Critical** | $\ge 0.2$ | Structural breakdown; conservation violations observable |

---

## 5. TNFR Connection

### 5.1 Grammar Violations as Collapse Operators

The Lindblad collapse operators $L_k$ map to specific TNFR grammar violations:

| Collapse operator $L_k$ | TNFR interpretation |
|--------------------------|---------------------|
| Destabilisers without stabilisers | OZ, ZHIR, VAL applied without IL, THOL (U2 violation) |
| Phase-incompatible coupling | UM, RA with $|\phi_i - \phi_j| > \Delta\phi_{\max}$ (U3 violation) |
| Uncontrolled bifurcation | OZ, ZHIR without handlers (U4 violation) |

The dissipation bound $|D[\rho]| \sim \|L_k\|^2$ corresponds to grammar U2 violation magnitude. Steady-state convergence corresponds to grammar closure (U1b + U2).

### 5.2 Conservative Limit

When all grammar rules are satisfied:
- $S_{\text{grammar}} \to 0$ (by conservation theorem)
- $D[\rho] \to 0$ (no collapse operators active)
- The standard conservation law $\partial\rho/\partial t + \operatorname{div}\mathbf{J} = 0$ is recovered

---

## 6. Monitoring and Diagnostics

The dissipative conservation tracker provides real-time monitoring via:

| Metric | Definition | Use |
|--------|-----------|-----|
| Purity $P(t)$ | $\operatorname{Tr}(\rho^2)$ | Information loss quantification |
| Entropy $S(t)$ | $-\operatorname{Tr}(\rho\ln\rho)$ | Irreversibility measure |
| Dissipation rate $|D[\rho]|$ | From Lindblad formula | Environmental coupling strength |
| Charge drift $\Delta Q$ | $Q(t) - Q(0)$ | Conservation violation magnitude |
| Dissipation tier | Threshold classification | Operational safety level |

---

## Implementation Reference

| Module | Content |
|--------|---------|
| `src/tnfr/physics/dissipative_conservation.py` | Lindblad dissipator, purity decay, entropy, tier classification |
| `src/tnfr/physics/conservation.py` | Conservative limit (Noether charge, Lyapunov energy) |

**Dataclasses**: `DissipativeSnapshot`, `DissipativeBalance`, `DissipativeTimeSeries`

**Tracker**: `DissipativeConservationTracker` — accumulates time-series data and classifies dissipation tier.

**Tests**: `tests/test_dissipative_conservation.py`

---

## Implementation & Examples

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [28_dissipative_systems_demo.py](../examples/28_dissipative_systems_demo.py) | Lindblad decoherence, purity decay, entropy production, dissipative regime classification, grammar violations as collapse operators |

### Key Source Modules

- `src/tnfr/physics/dissipative_conservation.py` — Dissipative continuity equation
- `src/tnfr/physics/conservation.py` — Conservative baseline

---

## Cross-References

- Conservative conservation: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)
- Grammar rules: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- Lyapunov stability: [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md)
- Extended fields: [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md)
