# TNFR mathematical foundation

## Theoretical Background

TNFR (Resonant Fractal Nature Theory) provides a novel framework for understanding mathematical structures through coherent patterns and resonance dynamics. In the context of primality testing, TNFR reveals that prime numbers exhibit **perfect structural coherence**, while composite numbers show **structural pressure**.

## The ΔNFR Eeuation

The core of TNFR primality testing is the arithmetic pressure equation:

```
ΔNFR(n) = ζ·(Ω(n)−1) + η·(τ(n)−2) + θ·(σ(n)/n − (1+1/n))
```

### Mathematical components

**Arithmetic functions:**
- `Ω(n)` = prime factor count with multiplicity (big Omega)
- `τ(n)` = total number of divisors of n  
- `σ(n)` = sum of all divisors of n

**TNFR pressure coefficients** (written as (φ, γ, π, e) combinations; audit
2026: notational, NOT derived — chosen to approximate the original empirical
values ζ=1.0, η=0.8, θ=0.6):
- `ζ = φ×γ ≈ 0.9340` = factorization pressure coefficient
- `η = (γ/φ)×π ≈ 1.1207` = divisor pressure coefficient  
- `θ = 1/φ ≈ 0.6180` = abundance pressure coefficient

These are a notational convention (only π is a genuine structural
scale). The narrative below is mnemonic, not a derivation:
- ζ = φ×γ (a combination near the empirical 1.0)
- η = (γ/φ)×π (a combination near the empirical 0.8)
- θ = 1/φ = φ−1 (a combination near the empirical 0.6)

### Pressure components

**1. Factorization pressure: `ζ·(Ω(n)−1)`**
- For primes: Ω(p) = 1, so this term = 0
- For composites: Ω(n) > 1, creating positive pressure
- Uses multiplicity: Ω(8)=3, Ω(12)=3, Ω(30)=3
- Measures total factorization complexity (structural pressure lever)

**2. Divisor pressure: `η·(τ(n)−2)`**
- For primes: τ(p) = 2 (divisors: 1, p), so this term = 0
- For composites: τ(n) > 2, creating positive pressure  
- Measures deviation from minimal divisor count

**3. Abundance pressure: `θ·(σ(n)/n − (1+1/n))`**
- For primes: σ(p) = p+1, making σ(p)/p = (p+1)/p = 1+1/p
- This makes the abundance term exactly 0 for primes
- For composites: σ(n) deviates from this relationship
- Measures deviation from prime abundance pattern

## Primality criterion  

**Fundamental theorem:** `n is prime ⟺ ΔNFR(n) = 0`

This equivalence holds because:
1. All three pressure components are zero if and only if n is prime
2. Any composite structure introduces positive pressure in at least one component
3. The coefficients are written as (φ, γ, π, e) combinations (notational, not derived)
4. ΔNFR(p) = 0 is independent of coefficient values (all three terms vanish individually for primes)
5. Coefficient values affect composite separation (TNFR pressure landscape), not prime detection correctness

## Examples

**Prime example (n = 17):**
- Ω(17) = 1 → Factorization pressure = 0.9340 × (1−1) = 0
- τ(17) = 2 → Divisor pressure = 1.1207 × (2−2) = 0  
- σ(17) = 18 → Abundance pressure = 0.6180 × (18/17 − (1+1/17)) = 0
- **Result: ΔNFR(17) = 0 → Prime**

**Composite example (n = 15):**
- Ω(15) = 2 → Factorization pressure = 0.9340 × (2−1) = 0.9340
- τ(15) = 4 → Divisor pressure = 1.1207 × (4−2) = 2.2414
- σ(15) = 24 → Abundance pressure = 0.6180 × (24/15 − (1+1/15)) ≈ 0.3296
- **Result: ΔNFR(15) ≈ 3.505 > 0 → Composite**

**Prime power example (n = 8 = 2³):**
- Ω(8) = 3 → Factorization pressure = 0.9340 × (3−1) = 1.8680
- τ(8) = 4 → Divisor pressure = 1.1207 × (4−2) = 2.2414
- σ(8) = 15 → Abundance pressure = 0.6180 × (15/8 − (1+1/8)) ≈ 0.4635
- **Result: ΔNFR(8) ≈ 4.573 > 0 → Composite**
- Note: with Ω (multiplicity), prime powers get strong pressure signals

## Computational complexity

- **Time Complexity:** O(√n) for computing ω(n), τ(n), σ(n)
- **Space Complexity:** O(1) for basic implementation, O(cache_size) with optimization
- **Accuracy:** 100% deterministic (no probabilistic components)

## Performance characteristics

**Benchmarked performance:**
- Small numbers (2-4 digits): ~10-15 μs
- Medium numbers (5-6 digits): ~15-30 μs  
- Large numbers (9+ digits): ~5-10 ms
- Very large numbers (10+ digits): ~10-50 ms

**Optimization techniques:**
- LRU caching for arithmetic functions
- Batch processing for multiple numbers
- Sieve integration for range generation
- Performance monitoring and statistics

## Theoretical significance

TNFR primality testing takes a different approach from traditional methods:

**Traditional methods:**
- Focus on divisibility testing or probabilistic verification
- Answer the question: "Is this number prime?"

**TNFR method:**  
- Analyzes structural coherence in arithmetic systems
- Answers the question: "Why is this number prime/composite?"
- Provides insight into the mathematical nature of primality itself
- Exposes the full structural triad (EPI, νf, ΔNFR) for each number
- Implements the dual-lever interpretation: νf (capacity) × ΔNFR (pressure)
- Connects arithmetic to the nodal equation `∂EPI/∂t = νf · ΔNFR(t)`

### Dual-lever interpretation (March 2026 discovery)

The nodal equation `∂EPI/∂t = νf · ΔNFR(t)` decomposes structural evolution
into two independent levers:

1. **Capacity lever (νf)**: How fast a number *can* reorganize.
   Depends on divisor structure and factorization complexity.
2. **Pressure lever (ΔNFR)**: How much reorganization is *demanded*.
   Zero for primes (perfect equilibrium), positive for composites.

For primes: ΔNFR = 0, so `∂EPI/∂t = 0` regardless of νf. Primes are
**zero-pressure fixed points** in the arithmetic structural manifold.

Experimental result (example 39): Φ_s responds linearly to ΔNFR
perturbations with |r| = 1.000, confirming the pressure lever's
direct connection to the structural potential field.

This deeper understanding makes TNFR valuable not just as a computational tool but as a theoretical framework for understanding the fundamental structure of numbers.