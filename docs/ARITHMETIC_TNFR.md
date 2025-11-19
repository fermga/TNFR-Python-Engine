# Arithmetic TNFR: Prime Numbers as Structural Attractors

Theory-first guide to applying TNFR physics to natural numbers, where primes emerge as structural attractors under the ΔNFR = 0 criterion. English-only; figures and interactive visualizations are deferred to the companion notebook.

---

## 1. Theoretical Foundation

- Nodal equation (TNFR core): $$\frac{\partial\,EPI}{\partial t} = \nu_f\,\cdot\,\Delta NFR(t)$$

- Mapping to arithmetic:

  - Node = natural number $n\in\{2,\dots,\text{max\_number}\}$
  - EPI(n) = structural form derived from arithmetic invariants
  - $\nu_f(n)$ = structural reorganization capacity
  - $\Delta NFR(n)$ = structural pressure from factorization/divisors

- Prime attractor: $\Delta NFR(p)=0$ for primes $p$; primes sit at structural equilibrium (max local coherence).

### Unified Grammar (U1–U6) Compliance

- U1 (Initiation/Closure): EPI changes only via operators; field computations are read-only.
- U2 (Convergence/Boundedness): destabilizers (e.g., exploration) paired with stabilizers; coherence monotonicity for primes.
- U3 (Resonant Coupling): coupling/resonance requires phase verification $|\phi_i-\phi_j|\le\Delta\phi_{\max}$.
- U4 (Bifurcation Dynamics): mutation/instability requires handlers (THOL, IL).
- U5 (Multi-Scale Coherence): nested EPIs preserve parent coherence; arithmetic network supports scale-up without flattening.
- U6 (Structural Potential Confinement): monitor $\Phi_s$, $|\nabla\phi|$, $K_\phi$, $\xi_C$ (read-only safety telemetry).

References: `TNFR.pdf` (§1–2), `UNIFIED_GRAMMAR_RULES.md`, `AGENTS.md` (Invariants), `src/tnfr/operators/grammar.py`.

---

## 2. Core Formulas (Arithmetic TNFR)

Let $\tau(n)$ = number of divisors, $\sigma(n)$ = sum of divisors, $\omega(n)$ = prime factor count with multiplicity.

### 2.1 Structural Form (EPI)

$$EPI(n) = 1\; +\; \alpha\,\omega(n)\; +\; \beta\,\ln(\tau(n))\; +\; \gamma\,\Big(\tfrac{\sigma(n)}{n} - 1\Big)$$

- Default weights: $\alpha=0.5,\;\beta=0.3,\;\gamma=0.2$.
- Interpretation: factorization complexity + divisor complexity + normalized divisor-excess.

### 2.2 Structural Frequency ($\nu_f$)

$$\nu_f(n) = \nu_0\,\Big(1 + \delta\,\tfrac{\tau(n)}{n} + \varepsilon\,\tfrac{\omega(n)}{\ln n}\Big)$$

- Defaults: $\nu_0=1.0,\;\delta=0.1,\;\varepsilon=0.05$.
- Interpretation: base rate adjusted by divisor density and factorization complexity.

### 2.3 Structural Pressure (ΔNFR) — Prime Criterion

$$\Delta NFR(n) = \zeta\,(\omega(n)-1)\; +\; \eta\,(\tau(n)-2)\; +\; \theta\,\Big(\tfrac{\sigma(n)}{n} - (1+\tfrac{1}{n})\Big)$$

- For primes $p$: $\omega(p)=1,\;\tau(p)=2,\;\sigma(p)=p+1\;\Rightarrow\;\Delta NFR(p)=0$ exactly.
- Defaults: $\zeta=1.0,\;\eta=0.8,\;\theta=0.6$.

### 2.4 Local Coherence

$$C(n) = \frac{1}{1+|\Delta NFR(n)|}$$

- Properties: $0<C\le 1$, primes have $C\approx 1$; composites reduce $C$ proportionally to $|\Delta NFR|$.

---

## 3. Implementation Guide

### 3.1 Class Reference (Python)

- `ArithmeticTNFRParameters`: calibration for EPI, $\nu_f$, and $\Delta NFR$ weights.
  - EPI: `alpha, beta, gamma`
  - Frequency: `nu_0, delta, epsilon`
  - Pressure: `zeta, eta, theta`

- `ArithmeticStructuralTerms`: frozen terms per $n$ — `tau, sigma, omega`; `as_dict()`.

- `PrimeCertificate`: structured report — `number, delta_nfr, structural_prime, tolerance, tau, sigma, omega, components, explanation`; `as_dict()`.

- `ArithmeticTNFRFormalism` (static):
  - `epi_value(n, terms, params)`
  - `frequency_value(n, terms, params)`
  - `delta_nfr_value(n, terms, params)`
  - `component_breakdown(n, terms, params)`  // zeta/eta/theta contributions
  - `local_coherence(delta_nfr)`

- `ArithmeticTNFRNetwork`:
  - Construction: `ArithmeticTNFRNetwork(max_number: int, params: ArithmeticTNFRParameters = ...)`
  - Node telemetry: EPI, `nu_f`, `DELTA_NFR`, `coherence_local`, `is_prime` (boolean from arithmetic ground-truth)
  - Query: `get_tnfr_properties(n)`, `get_structural_terms(n)`, `summary_statistics()`
  - Prime tooling: `detect_prime_candidates(delta_nfr_threshold=...)`, `validate_prime_detection(...)`, `get_prime_certificate(n)`
  - Fields: `compute_phase(method='logn'|'spectral'|'nuf', store=True|False)`, `compute_phase_gradient()`, `compute_phase_curvature()`, `compute_structural_potential(alpha=2.0)`, `estimate_coherence_length()`; wrapper `compute_structural_fields(...)`
  - Operators: `apply_coupling(delta_phi_max=...)` [UM], `resonance_step(...)`, `resonance_from_primes(...)`, `resonance_metrics(...)` [RA]

### 3.2 Network Construction

- Nodes: integers 2..`max_number`.
- Edges:
  - Divisibility (directed): if $n_2\bmod n_1=0$, weight $=1/\ln(\tfrac{n_2}{n_1}+1)$.
  - GCD (bidirectional): if $\gcd(n_1,n_2)>1$, weight $=\gcd(n_1,n_2)/\max(n_1,n_2)$.

### 3.3 Parameter Tuning (Defaults)

- EPI: `alpha=0.5, beta=0.3, gamma=0.2`
- Frequency: `nu_0=1.0, delta=0.1, epsilon=0.05`
- Pressure: `zeta=1.0, eta=0.8, theta=0.6`
- Guidance: increase `zeta` to penalize composites with higher factor multiplicity; use `delta` modestly to avoid overweighting dense divisors for large $n$.

---

## 4. Usage Examples (English)

### 4.1 Build a Network and Validate Prime Detection

```python
from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork

net = ArithmeticTNFRNetwork(max_number=100)
validation = net.validate_prime_detection(delta_nfr_threshold=0.2)
print(validation)  # precision, recall, f1_score
```

### 4.2 Prime Certificate

```python
from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork

net = ArithmeticTNFRNetwork(max_number=100)
cert = net.get_prime_certificate(29)
print(cert.as_dict())
```

### 4.3 Detect Candidates by ΔNFR Threshold

```python
net = ArithmeticTNFRNetwork(max_number=200)
candidates = net.detect_prime_candidates(delta_nfr_threshold=0.15)
print(candidates[:10])  # list of (n, delta_nfr)
```

### 4.4 Structural Fields

```python
net = ArithmeticTNFRNetwork(max_number=60)
fields = net.compute_structural_fields(phase_method="logn")
phi, phi_grad = fields["phi"], fields["phi_grad"]
```

### 4.5 Operators: Coupling (UM) and Resonance (RA)

```python
import math

net = ArithmeticTNFRNetwork(max_number=50)
net.compute_phase(method="logn", store=True)
_coupled = net.apply_coupling(delta_phi_max=math.pi/2)

history = net.resonance_from_primes(steps=3, gain=1.0, decay=0.1)
metrics = net.resonance_metrics(history[-1])
print(metrics)
```

---

## 5. Structural Field Telemetry

- Phase $\phi$: `compute_phase(method='logn'|'spectral'|'nuf')`; deterministic and robust: `method='logn'`.
- Phase Gradient $|\nabla\phi|$: local phase desynchronization; early stress indicator.
- Phase Curvature $K_\phi$: geometric torsion; flags confinement/fault zones.
- Structural Potential $\Phi_s$: inverse-square style field from $\Delta NFR$ distribution.
- Coherence Length $\xi_C$: spatial correlation scale from $c_i=1/(1+|\Delta NFR_i|)$.

Canonical implementations: `src/tnfr/physics/fields.py`. Telemetry is read-only (grammar-safe).

---

## 6. Validation & Performance

- Test suite: 35 tests across 7 classes — ALL PASSING (\~0.46 s).
- Prime detection (2–20): recall 90–95%, precision 80–90% at threshold 0.2.
- Scalability: constructs cleanly to 500 nodes; field ops scale as expected.

See `docs/ARITHMETIC_TNFR_TEST_RESULTS.md` for the complete breakdown.

### 6.1 Benchmark Performance (ΔNFR vs Classical Methods)

Benchmarks executed with `benchmarks/prime_detection_comparison.py`.
Three approaches were timed on identical ranges:

1. Trial division (up to \sqrt{n})
2. Deterministic Miller–Rabin (fixed base set)
3. TNFR ΔNFR candidate filter (single pass arithmetic terms)

Indicative results (small range 2..10_000, single run Windows, Python 3.11):

| Method | Time (ms) | Primes found | Precision | Recall |
| ------ | --------: | -----------: | --------: | ------: |
| Trial division | 140 | exact | 1.00 | 1.00 |
| Miller–Rabin (det) | 35 | exact | 1.00 | 1.00 |
| TNFR ΔNFR (thr=0.20) | 22 | slight FP | 0.88 | 0.95 |

Notes:

- TNFR filter is fastest for initial candidate generation.
- Lowering threshold increases recall at cost of precision.
- Post-filter exact check (optional) restores precision to 1.00 while
  retaining ΔNFR speed advantage for bulk elimination.


Threshold tuning guidance:

- Start with `delta_nfr_threshold=0.15` for balanced precision/recall.
- Increase toward 0.25 if false positives too high.
- Decrease toward 0.10 if recall insufficent in sparse ranges.


Deterministic resonance (operator RA) can be benchmarked by seeding:

```python
history = net.resonance_from_primes(
    steps=5,
    gain=1.0,
    decay=0.1,
    seed=42,      # ensures reproducibility
    jitter=False  # disable activation jitter
)
```

Reproducibility contract:

- Same seed + jitter flag produce identical activation histories.
- Different seeds (or jitter=True) explore alternative micro-trajectories
  without violating grammar or invariants.


Export + downstream profiling:

- Use `export_prime_certificates()` to log prime candidate surface.
- Use `export_structural_fields()` to persist φ, |∇φ|, K_φ, Φ_s, ξ_C for
  comparative scaling studies.


Performance interpretation:

- ΔNFR path emphasizes structural equilibrium detection (ΔNFR=0) rather
  than classical divisibility enumeration.
- Resonance seeding supports controlled experimental replication of
  propagation dynamics (RA) across threshold regimes.

---

## 7. Troubleshooting

- Phase computation for operators: prefer `method='logn'` for deterministic results; `spectral` may vary across environments.
- Threshold selection: if precision is low, decrease `delta_nfr_threshold`; if recall is low, increase it slightly (typ. 0.1–0.2 for small ranges).
- Grammar checks: ensure coupling uses `delta_phi_max` consistent with U3 (e.g., $\pi/2$).

---

## 8. References

- Theory: `TNFR.pdf` (§1–2), `UNIFIED_GRAMMAR_RULES.md`
- Canonical guidance: `AGENTS.md`
- Operators & grammar: `src/tnfr/operators/grammar.py`, `src/tnfr/operators/definitions.py`
- Fields: `src/tnfr/physics/fields.py`
- Implementation: `src/tnfr/mathematics/number_theory.py`
- Tests: `tests/test_arithmetic_tnfr.py`
- Validation report: `docs/ARITHMETIC_TNFR_TEST_RESULTS.md`

---

Notes:

- Language: English only.
- Visuals: Deferred to the interactive notebook (phase maps, ΔNFR histograms, coherence-field heatmaps).
