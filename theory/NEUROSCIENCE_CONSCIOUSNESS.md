# Neuroscience Field Memo

**Status**: Analytical reference  
**Version**: 0.3.0 (November 30, 2025)  
**Owner**: `theory/NEUROSCIENCE_CONSCIOUSNESS.md`

---

## 1. Structural Hypothesis

- Neural assemblies are modeled as coupled TNFR nodes.  
- Conscious access correlates with closed-loop operator sequences that maintain \(C(t) > C_{th}\) for a minimum dwell time.  
- The hypothesis is testable via electrophysiological data (EEG/MEG) and hemodynamic imaging; all runs must export telemetry to `results/neuroscience_consciousness/`.

---

## 2. Modeling Approach

1. Define recurrent networks with operators `UM`, `RA`, `IL`, and `SHA`.  
2. Simulate stimulus-evoked activity and measure whether patterns persist after input removal.  
3. Compare persistence metrics with empirical working-memory/awareness datasets; record seeds, operator schedules, and telemetry.

---

## 3. Plasticity Equation

Retain the coupling update rule
\[
\frac{dK_{ij}}{dt} = \alpha (1 - |\phi_i - \phi_j|) - \beta K_{ij},
\]
and fit \(\alpha, \beta\) to experimental datasets instead of treating the rule as speculative. Store fitted parameters alongside dataset identifiers.

---

## 4. Experimental Design

- Provide reproducible scripts for minimal networks (e.g., `examples/19_neuroscience_demo.py`).  
- Report coherence, phase gradients, and structural potential over time; include statistical comparisons with empirical signals (phase-locking values, spectral power).

---

## 5. Outstanding Work

1. Publish validated datasets linking \(C(t)\) dwell times to EEG/MEG correlates of conscious access.  
2. Extend plasticity models to include neuromodulator-dependent gains and fit against long-term potentiation studies.  
3. Add regression tests ensuring simulations reproduce benchmark working-memory persistence curves.
