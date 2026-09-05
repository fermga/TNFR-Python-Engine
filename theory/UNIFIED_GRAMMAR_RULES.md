# Unified TNFR Grammar: Canonical Requirements and Mathematical Scope

## Purpose and authority

This document specifies the six canonical grammar rules U1–U6. The operator
classifications come from the contract predicates in
[physics_derivation.py](../src/tnfr/config/physics_derivation.py), are re-exported
by [grammar_types.py](../src/tnfr/operators/grammar_types.py), and are materialized
in [grammar_canon.py](../src/tnfr/operators/grammar_canon.py). Validation is exposed
through [grammar.py](../src/tnfr/operators/grammar.py).

The nodal equation

    ∂EPI/∂t = ν_f · ΔNFR

supplies the dynamical interpretation of form, capacity, and pressure. The
grammar adds initialization, operator-composition, coupling, and monitoring
contracts. These are mandatory engine requirements; they are not all theorems
of the bare differential identity. Precise hypotheses and finite-graph
counterexamples are centralized in
[Mathematical Scope of Structural Diagnostics and Grammar](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).

U1–U5 govern operator sequences and their context. U3 also requires actual
phase verification by operator preconditions. U6 is a read-only potential
monitor. Passing a word validator does not certify every operator
precondition, an infinite-horizon convergence theorem, or a future U6 reading.

## 1. Canonical operator roles

Public identifiers are the English operator names; glyphs are internal symbols.

| Role | Operators | Rule |
|------|-----------|------|
| Generators | Emission (AL), Transition (NAV), Recursivity (REMESH) | U1a |
| Closures | Silence (SHA), Transition (NAV), Recursivity (REMESH), Dissonance (OZ) | U1b |
| Stabilizers | Coherence (IL), Self-organization (THOL) | U2 |
| Destabilizers | Dissonance (OZ), Mutation (ZHIR), Expansion (VAL) | U2 |
| Coupling/resonance | Coupling (UM), Resonance (RA) | U3 |
| Bifurcation triggers | Dissonance (OZ), Mutation (ZHIR) | U4a |
| Bifurcation handlers | Coherence (IL), Self-organization (THOL) | U4a |
| Transformers | Mutation (ZHIR), Self-organization (THOL) | U4b |

Transition is not a U2 destabilizer: its contract describes a controlled regime
change. Contraction is not a runtime closure. The historical TNFR.pdf syntax
includes Contraction as a return-to-potential closure; the engine's supported
closure set is the one above, as documented in grammar_canon.py. Consumers must
import the shared sets rather than reproduce them.

These roles classify sequence obligations. The operator's primary dynamical
channel is a separate classification in
[operator_contracts.py](../src/tnfr/operators/operator_contracts.py): capacity
ν_f, pressure ΔNFR, phase, or form. Sharing a grammatical role does not imply
identical gains or measured field responses.

## 2. U1 — Structural initiation and closure

### U1a: Initialization

A sequence starting new structure requires a generator from
{Emission, Transition, Recursivity}. Emission supplies new form, Transition
activates latent form, and Recursivity uses an available structural echo.
Each still has its own execution preconditions.

This requirement is an initialization contract. EPI = 0 does not by itself make
the nodal derivative undefined: finite ν_f = 1 and ΔNFR = 1 give ∂EPI/∂t = 1,
including at zero. A derivation of an initialization restriction therefore
needs the operator contract, not only the numerical value of EPI.

The context-aware sequence API supports an already initialized structure through
the initial_epi_nonzero context flag. The structural execution entry point
derives this context from the target graph/node. This contextual start
allowance does not waive other sequence or operator requirements.

### U1b: Closure

A normal sequence ends with {Silence, Transition, Recursivity, Dissonance}.
These are supported closure modes:

- Silence reduces reorganization capacity and supports latency.
- Transition hands off to a controlled regime.
- Recursivity closes through a structural echo.
- Dissonance permits intentional activation/tension at the endpoint.

A closure token identifies an allowed endpoint. It does not assert that every
endpoint is an equilibrium or that a finite sequence proves asymptotic
stability. The narrow Dissonance–Mutation probe allowance in the context-aware
validator requires explicit diagnostic context; it is not a generally valid
production word.

## 3. U2 — Stabilization and boundedness policy

Integrating the nodal equation gives the exact finite-interval identity

    EPI(t) − EPI(t₀) = ∫[t₀,t] ν_f(s) ΔNFR(s) ds.

Local integrability supplies finite-horizon existence. On an infinite horizon,
bounded partial integrals, convergence of the improper integral, and absolute
integrability are different conditions. For example, EPI(t) = sin(t) is bounded
but does not converge, and ΔNFR(t) = 1/(1+t) with ν_f = 1 tends to zero while
its integral diverges.

U2 controls declared destabilizing actions through the following policy:

1. A sequence containing {Dissonance, Mutation, Expansion} must contain
   {Coherence, Self-organization}.
2. A destabilizer adds one unit of uncompensated debt. A stabilizer removes one
   outstanding unit, with a floor of zero; neutral operations leave debt fixed.
3. No executed prefix may exceed the canonical debt capacity **2**. A later
   stabilizer does not repair an earlier over-capacity prefix, and earlier
   stabilization cannot prepay future debt.
4. Recursivity combined with a destabilizer also requires a stabilizer because
   recursive amplification retains the same stabilization obligation.

The accounting kernel is
[grammar_debt.py](../src/tnfr/operators/grammar_debt.py). It measures operator
obligations, not instantaneous physical pressure. Persistent execution context
preserves debt that is older than a bounded retained trace.

Coherence and Self-organization provide their contracted negative feedback.
Proving convergence for repeated actions additionally requires a pressure law,
gain and frequency bounds, elapsed times, and a suitable norm. The absence of
named stabilizers does not imply positive feedback: pure EPI diffusion itself
has a restoring pressure −L_rw EPI. Conversely, the mere presence of a
stabilizer is not an analytic convergence proof.

## 4. U3 — Resonant coupling

Coupling and Resonance require phase compatibility before an actual coupling:

    |wrap(φ_i − φ_j)| ≤ Δφ_max,    Δφ_max = π/2 by default.

Wrapping respects the circular phase geometry. The choice of π/2 is the
canonical coupling gate. A sequence-level role check records this obligation;
the graph-specific phase comparison belongs to operator preconditions. Operator
execution must preserve this separation and must not replace a real phase
check with acceptance of a word.

## 5. U4 — Bifurcation dynamics

### U4a: Triggers need handlers

Sequences containing Dissonance or Mutation require a handler from
{Coherence, Self-organization}. The policy pairs operations that can induce
reorganization with a contracted stabilization mechanism. A trigger token does
not prove a numerical threshold was crossed; its execution and telemetry
determine that outcome.

### U4b: Transformers need context

Mutation and Self-organization require a preceding destabilizer within the
canonical **three-operation** recency window. The same window applies to every
destabilizer. Mutation additionally requires prior Coherence as a stable base.
That prior-Coherence requirement does not expire with the recency window;
the recent destabilizer and the prior stable base are distinct context facts.

The numerical window and U2 capacity use one scalar calibration:

    ρ = 1,    q = 1 − ν_f dt ρ,
    window = first n with qⁿ < 1/(π+1),
    capacity = floor(1/(ν_f dt ρ)).

For ν_f = 1 and dt = 0.5 these give **3** and **2**, respectively. The public
functions derive_bifurcation_window_from_physics and
derive_u2_debt_capacity_from_physics preserve those formulas and values.

The calibration uses the mean-rate surrogate ρ = 1. On loopless graphs without
isolates, mean eigenvalue trace(L_rw)/N = 1, but an individual diffusion mode
decays by 1 − ν_f dt λ_k. On a 21-node path the Fiedler mode retains about
0.981646 of its amplitude after three such steps and requires 231 steps to
fall below 1/(π+1). Thus the grammar window is a recency policy calibrated to a
surrogate, not a uniform relaxation time for all graphs and perturbations.
Function fallback branches likewise carry no Euler-stability guarantee.
See [the scope note, §2](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#2-modal-relaxation-and-the-grammar-calibration).

## 6. U5 — Multi-scale coherence

Nested EPIs retain their identity and need stabilization at each participating
scale. The sequence validator checks declared Recursivity depth: depth greater
than one requires Coherence or Self-organization within the same three-operation
window before or after the Recursivity operation. Runtime depth validation and
the operator's nesting contract remain necessary.

A hierarchy with a differentiable representation EPI_parent = f(EPI_children)
satisfies the chain-rule identity

    ν_parent ΔNFR_parent = Σ_i (∂f/∂EPI_child_i) ν_i ΔNFR_i.

This identity assumes the representation f exists and that the parent dynamics
is compatible with it. It does not by itself imply a coherence inequality.
The canonical coherence kernel is 1/(1+|ΔNFR|+|dEPI|), rather than 1/|ΔNFR|.
A quantitative target such as C_parent ≥ α Σ_i C_child_i therefore requires
a specified hierarchy, coupling weights, normalization, and admissible α.

The grammar's scale-stabilizer requirement operationalizes preservation of
nested structure. It does not prove that target for every arbitrary hierarchy.
A word can satisfy U2 and U4b while missing U5: Emission, deep Recursivity,
Silence contains no destabilizer or transformer, yet lacks a nearby scale
stabilizer.

## 7. U6 — Structural potential confinement telemetry

Structural potential aggregates pressure through the canonical inverse-square
distance kernel:

    Φ_s(i) = Σ_(reachable j ≠ i) ΔNFR_j / d(i,j)².

The field API defaults to exact evaluation. U6 compares before/after potential
telemetry using U6_STRUCTURAL_POTENTIAL_LIMIT = **π/2**. The related per-node
warning value PHI_S_VON_KOCH_THRESHOLD = **π/4** is distinct from the drift
check. Monitor reports must specify their node aggregation and baseline.

These values are selected π-scaled safety policies, not universal upper bounds
derived from angular wrapping. On K₄, zero phase and unit pressure give Φ_s = 3
at every node. For a fixed graph kernel B, the actual general bound is

    ||Φ_s||∞ ≤ ||B||∞ ||ΔNFR||∞.

A proof of confinement must bound pressure and graph geometry; graph changes
also change B. Writing thresholds as fractions of π does not supply those
hypotheses. The earlier experimental threshold 2.0 is not the current U6
policy, and empirical correlations on particular protocols do not establish
topology-independent confinement.

U6 is read-only. Crossing its threshold flags a potential-confinement policy
violation for that measurement; it is not, by itself, a theorem of fragmentation.
Passing U1–U5 does not guarantee the field will remain below the threshold.

## 8. Composition and implementation

Named macros are structural fragments, not necessarily complete words:

| Fragment | Operators |
|----------|-----------|
| Bootstrap | Emission, Coupling, Coherence |
| Stabilize | Coherence, Silence |
| Explore | Dissonance, Mutation, Coherence |
| Propagate | Resonance, Coupling |

A typical complete word is Emission, Coupling, Coherence, Silence; Coupling
still needs compatible phases. The Explore fragment also needs a supported
start and closure plus prior Coherence for Mutation.

Nesting and branching are governed by their existing operators and contracts.
They do not introduce extra canonical operators or a new U7/U8 rule. There are
six currently supported rules; this catalog statement does not prove that
every possible dynamical risk is expressible by six word constraints.

| Responsibility | Source |
|----------------|--------|
| Operator roles and calibration | [physics_derivation.py](../src/tnfr/config/physics_derivation.py) |
| Re-exported sets and window | [grammar_types.py](../src/tnfr/operators/grammar_types.py) |
| Declarative grammar registry | [grammar_canon.py](../src/tnfr/operators/grammar_canon.py) |
| Operator-list validation | [grammar_core.py](../src/tnfr/operators/grammar_core.py) |
| Context-aware name/glyph validation | [grammar_patterns.py](../src/tnfr/operators/grammar_patterns.py) |
| Causal debt and prior Coherence | [grammar_debt.py](../src/tnfr/operators/grammar_debt.py) |
| Dynamic selection and application | [grammar_dynamics.py](../src/tnfr/operators/grammar_dynamics.py), [grammar_application.py](../src/tnfr/operators/grammar_application.py) |
| Validated execution boundary | [grammar_execution.py](../src/tnfr/operators/grammar_execution.py) |
| Field measurements | [fields.py](../src/tnfr/physics/fields.py) |

The legacy C1/RC1 initiation rules map to U1, C2/RC2 to U2, RC3 to U3, and
C3/RC4 to U4. U5 covers declared hierarchy and U6 covers potential telemetry.
Historical proposed spacing rules do not override this registry.

## 9. Verification and reporting

Verification has separate targets:

- **Grammar contracts:** generators, closures, causal debt, transformer context,
  hierarchy depth, explicit phase checks, and context-specific allowances.
- **Operator contracts:** Coherence monotonicity outside dissonance tests,
  controlled bifurcation, Resonance propagation, Silence latency, Mutation's
  threshold, nested identity, and reproducibility.
- **Analytic claims:** explicit graph, pressure law, frequency assumptions,
  integrator, norm, and gain bounds. Validate the claimed quantity rather than
  substituting a surrogate.
- **Telemetry:** record C(t), Si, ν_f in Hz_str, phase, ΔNFR, and the tetrad with
  seeds, operator sequence, and before/after field baselines.

The numerical policies and operator sets remain fixed by their existing source
definitions. Tests of those values establish compatibility. They do not close
the open mathematical sufficiency questions identified in the scope note.

## References

- [AGENTS.md](../AGENTS.md): synthesized canonical guidance.
- [Structural operators](STRUCTURAL_OPERATORS.md): operator contracts and composition.
- [Structural field tetrad](../docs/STRUCTURAL_FIELDS_TETRAD.md): field definitions.
- [Diagnostic and grammar scope](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md): hypotheses and witnesses.
- [Testing](../TESTING.md): repository validation workflow.