# Biomedical Applications of TNFR

This directory contains executable examples demonstrating TNFR structural operators in biomedical and clinical contexts, with emphasis on the **SHA (Silence)** operator.

## Overview

Each script implements a complete clinical or physiological protocol using TNFR operators, showing how structural operators model real-world biomedical processes with quantitative telemetry and validation.

## Examples

### 1. Cardiac Coherence Training (`cardiac_coherence_sha.py`)

**Protocol**: `AL → IL → RA → SHA`  
**Context**: Heart Rate Variability (HRV) biofeedback training  
**Key Insight**: SHA consolidates coherent cardiac rhythm patterns before session end, creating "physiological memory"

**Run**:
```bash
python examples/biomedical/cardiac_coherence_sha.py
```

**Expected Output**:
- Telemetry showing EPI preservation (pattern maintained)
- νf suppression (structural pause achieved)
- Low ΔNFR (stable coherent state)
- Preservation integrity < 5%

**Clinical Applications**:
- Anxiety and stress management
- Autonomic regulation training
- Performance psychology
- Post-cardiac event rehabilitation

---

### 2. Trauma Containment (`trauma_containment_sha.py`)

**Protocol**: `AL → EN → OZ → SHA`  
**Context**: PTSD therapy with protective pause  
**Key Insight**: SHA contains dissonance (high ΔNFR) without resolving it, stabilizing patient during intense work

**Run**:
```bash
python examples/biomedical/trauma_containment_sha.py
```

**Expected Output**:
- ΔNFR remains HIGH (dissonance contained, not eliminated)
- νf dramatically reduced (protective pause)
- EPI preserved (no dissociation)
- Patient stabilized but aware

**Critical Distinction**:
- SHA is NOT suppression or avoidance
- SHA creates safe pause for future processing (THOL/ZHIR)
- Enables safe session termination during crisis

**Clinical Applications**:
- PTSD and complex trauma therapy
- Crisis intervention
- Affect regulation training
- Exposure therapy safety protocols

---

### 3. Sleep & Memory Consolidation (`sleep_consolidation_sha.py`)

**Protocol**: `[Day] AL → EN → IL → RA → IL` → `[Night] SHA` → `[Next Day] NAV → AL`  
**Context**: Sleep-dependent memory consolidation  
**Key Insight**: SHA models deep sleep where νf → 0 preserves learned patterns (EPI) intact

**Run**:
```bash
python examples/biomedical/sleep_consolidation_sha.py
```

**Expected Output**:
- Learning phase: EPI increases (pattern acquisition)
- Sleep phase: νf < 0.05, EPI variance < 2% (preservation)
- Recall phase: High fidelity recall (>90% retention)

**Neuroscientific Correlates**:
- SHA ↔ Slow-wave sleep (δ waves 0.5-4 Hz)
- νf → 0 ↔ Reduced neuronal firing rates
- EPI preservation ↔ Synaptic consolidation

**Research Applications**:
- Sleep disorder impact on learning
- Optimal study-sleep schedules
- Aging and memory research
- Computational neuroscience models

---

### 4. Post-Exercise Recovery (`recovery_protocols_sha.py`)

**Protocol**: `[Training] VAL → OZ → IL` → `[Recovery] SHA` → `[Next Session] NAV → AL`  
**Context**: Athletic training and adaptation  
**Key Insight**: SHA enables adaptation consolidation through structural pause (reduced activity)

**Run**:
```bash
python examples/biomedical/recovery_protocols_sha.py
```

**Expected Output**:
- Training: High ΔNFR (metabolic stress), elevated νf
- Recovery Day 1: νf = 0.15, ΔNFR decreasing
- Recovery Day 2: νf < 0.1, EPI increases (adaptation), ΔNFR normalized
- Next training: Improved baseline EPI (+8% structural gain)

**Physiological Markers**:
- SHA activation ↔ HRV recovery, resting HR
- νf reduction ↔ Metabolic downregulation
- EPI growth ↔ Muscle hypertrophy, performance gains
- ΔNFR normalization ↔ Stress marker clearance

**Training Applications**:
- Periodization design
- Overtraining prevention
- Performance optimization
- Recovery monitoring

---

## Comprehensive Documentation

For detailed clinical protocols, expected telemetry, physiological correlates, and scientific references, see:

**📚 [SHA Clinical Applications Documentation](../../docs/source/examples/SHA_CLINICAL_APPLICATIONS.md)**

This comprehensive guide includes:
- 6 detailed clinical protocols
- Expected telemetry specifications
- Physiological and neural correlates
- Research applications and experimental predictions
- Mathematical models
- Case studies
- Cross-domain synthesis

## Common SHA Patterns

Across all biomedical applications, SHA exhibits:

1. **Pre-SHA stabilization**: Often preceded by IL (Coherence)
2. **Pressure containment**: High ΔNFR contained, not resolved
3. **Duration variability**: Seconds (cardiac) to months (organizational)
4. **Preservation fidelity**: EPI variance during SHA predicts outcome
5. **Reactivation protocol**: Exits through NAV or AL with stabilization

## SHA Quality Metrics

**Good SHA** (across all domains):
- νf < 0.1 (effective suppression)
- EPI variance < 5% baseline (tight preservation)
- ΔNFR stable or decreasing
- Clear purpose and exit strategy

**Poor SHA**:
- νf > 0.2 (inadequate suppression)
- EPI variance > 10% (pattern drifting)
- ΔNFR increasing (pressure building)
- Excessive or insufficient duration

## Running All Examples

```bash
# Run all biomedical examples
cd examples/biomedical

python cardiac_coherence_sha.py
python trauma_containment_sha.py
python sleep_consolidation_sha.py
python recovery_protocols_sha.py
```

## Dependencies

All examples use only core TNFR functionality:
- `tnfr.structural` (create_nfr, run_sequence)
- `tnfr.operators.definitions` (structural operators)
- `tnfr.constants` (node attributes)
- `tnfr.dynamics` (ΔNFR hooks)

No additional dependencies required.

## Scientific References

### Cardiac Coherence
- McCraty, R., & Shaffer, F. (2015). Heart rate variability: new perspectives. *Glob Adv Health Med*, 4(1), 46-61.
- Lehrer, P. M., & Gevirtz, R. (2014). Heart rate variability biofeedback. *Biofeedback*, 42(1), 26-31.

### Trauma Therapy
- van der Kolk, B. A. (2015). *The Body Keeps the Score*. Penguin Books.
- Ogden, P., & Fisher, J. (2015). *Sensorimotor Psychotherapy*. Norton.
- Porges, S. W. (2011). *The Polyvagal Theory*. Norton.

### Sleep & Memory
- Tononi, G., & Cirelli, C. (2014). Sleep and the price of plasticity. *Neuron*, 81(1), 12-34.
- Rasch, B., & Born, J. (2013). About sleep's role in memory. *Physiol Rev*, 93(2), 681-766.
- Diekelmann, S., & Born, J. (2010). The memory function of sleep. *Nat Rev Neurosci*, 11(2), 114-126.

### Exercise Recovery
- Kellmann, M., et al. (2018). *Recovery and Performance in Sport*. Routledge.
- Halson, S. L. (2014). Monitoring training load to understand fatigue. *Sports Med*, 44(S2), 139-147.
- Bompa, T. O., & Haff, G. (2009). *Periodization: Theory and Methodology*. Human Kinetics.

## Contributing

When adding new biomedical examples:

1. Follow the established pattern (protocol → telemetry → validation)
2. Include expected outputs with thresholds
3. Map TNFR metrics to physiological correlates
4. Provide scientific references
5. Document in main SHA_CLINICAL_APPLICATIONS.md

## License

MIT License - See repository root for details.
