# TNFR theory and research index

This directory contains mathematical scope, derivations and research programs
for the TNFR Python Engine. [AGENTS.md](../AGENTS.md) is the synthesized canonical
reference. Source code and tests decide implemented behavior; a theory document
must state whether a result is exact, conditional, empirical or open.

## Core framework

| Document | Scope |
| --- | --- |
| [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) | Nodal equation, structural triad and framework overview |
| [DIAGNOSTIC_AND_GRAMMAR_SCOPE.md](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) | Exact hypotheses, finite-graph witnesses and limits of current claims |
| [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) | Mathematical representation of graph dynamics |
| [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) | Four diagnostic channels and the open minimal-state question |
| [STRUCTURAL_OPERATORS.md](STRUCTURAL_OPERATORS.md) | Operator semantics and channel effects |
| [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) | U1-U6 grammar and its operational derivations |
| [GLOSSARY.md](GLOSSARY.md) | Shared terminology and status of constants and thresholds |

Field APIs and bounds are centralized in
[Structural Fields](../docs/STRUCTURAL_FIELDS_TETRAD.md); executable operator
contracts are centralized in [API Contracts](../docs/API_CONTRACTS.md).

## Dynamics and geometry

| Document | Scope |
| --- | --- |
| [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) | Conservation diagnostics, residuals and Lyapunov candidates |
| [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) | Variational models and their stated bridge conditions |
| [DISSIPATIVE_AND_OPEN_SYSTEMS.md](DISSIPATIVE_AND_OPEN_SYSTEMS.md) | Open-system and dissipative extensions |
| [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md) | Stability diagnostics and invariant monitoring |
| [TNFR_DIFFUSION_STABILITY_THEOREM.md](TNFR_DIFFUSION_STABILITY_THEOREM.md) | Fixed/time-varying and exact-common-metric EPI diffusion, exact reversible single-eigenmode Euler solution/error/convergence theorem, directed transient criterion, and conditional affine hybrid bounds |
| [TNFR_SCALE_GEOMETRY_AND_BRIDGE.md](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) | Pure-EPI quotient, coherence geometry, decoupled metriplectic bridge, and restricted S16 endpoint/path certificates |
| [PHYSICAL_REGIME_CORRESPONDENCES.md](PHYSICAL_REGIME_CORRESPONDENCES.md) | Scoped comparisons with diffusive, inertial and modal regimes |
| [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md) | Gauge and polarization models |
| [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md) | Derived field quantities |
| [EMERGENT_ONTOLOGY.md](EMERGENT_ONTOLOGY.md) | Structural interpretations; analogies remain explicitly non-physical |

## Arithmetic structure

| Document | Scope |
| --- | --- |
| [TNFR_NUMBER_THEORY.md](TNFR_NUMBER_THEORY.md) | Arithmetic pressure and primality criterion |
| [TNFR_ARITHMETIC_DYNAMICS.md](TNFR_ARITHMETIC_DYNAMICS.md) | Arithmetic-network dynamics |
| [TNFR_ARITHMETIC_OPERATORS.md](TNFR_ARITHMETIC_OPERATORS.md) | Operator realization in arithmetic domains |
| [TNFR_ARITHMETIC_PRESSURE.md](TNFR_ARITHMETIC_PRESSURE.md) | Pressure-channel analysis and limitations |
| [TNFR_ADDITIVE_DYNAMICS.md](TNFR_ADDITIVE_DYNAMICS.md) | Additive/Fourier constructions |
| [TNFR_CRT_FRACTALITY.md](TNFR_CRT_FRACTALITY.md) | Chinese-remainder synthesis |
| [TNFR_PADIC_DYNAMICS.md](TNFR_PADIC_DYNAMICS.md) | Projective p-adic transport |
| [TNFR_ALGEBRAIC_NUMBER_FIELDS.md](TNFR_ALGEBRAIC_NUMBER_FIELDS.md) | Finite and algebraic-field extensions |
| [TNFR_STRUCTURAL_OBSERVABILITY.md](TNFR_STRUCTURAL_OBSERVABILITY.md) | Observability diagnostics and selector scope |

## Active research programs

| Program | Document |
| --- | --- |
| Core dynamics S1-S16 | [CORE_RESEARCH_PROGRAM.md](CORE_RESEARCH_PROGRAM.md) |
| Riemann and spectral ladders | [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) |
| Navier-Stokes | [TNFR_NAVIER_STOKES_RESEARCH_NOTES.md](TNFR_NAVIER_STOKES_RESEARCH_NOTES.md) |
| Yang-Mills | [TNFR_YANG_MILLS_RESEARCH_NOTES.md](TNFR_YANG_MILLS_RESEARCH_NOTES.md) |
| P versus NP | [TNFR_P_VS_NP_RESEARCH_NOTES.md](TNFR_P_VS_NP_RESEARCH_NOTES.md) |
| Birch-Swinnerton-Dyer | [TNFR_BSD_RESEARCH_NOTES.md](TNFR_BSD_RESEARCH_NOTES.md) |
| Hodge | [TNFR_HODGE_RESEARCH_NOTES.md](TNFR_HODGE_RESEARCH_NOTES.md) |

These are research programs, not solutions to the corresponding classical
problems. Supporting maps include
[STRUCTURAL_RESEARCH_PROGRAM.md](STRUCTURAL_RESEARCH_PROGRAM.md),
[NUCLEUS_A_PRIME_LADDER_ATLAS.md](NUCLEUS_A_PRIME_LADDER_ATLAS.md), and
[NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md](NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md).

## Catalog studies

- [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) separates the
  clipped runtime, finite cyclic filter and finite companion recurrence; it
  derives the cyclic projector and the restricted augmented-history stability
  theorem. It also records the one-transition binary64 residual bridge and the
  exact finite telescope that binds each applied REMESH result to the next
  represented schedule and recorded history head. Its effective-P2 reference
  family specializes the general reversible eigenmode theorem from
  [TNFR_DIFFUSION_STABILITY_THEOREM.md](TNFR_DIFFUSION_STABILITY_THEOREM.md#exact-reversible-single-eigenmode-euler-reference-theorem),
  then adds exact ideal REMESH error scaling and an explicit runtime residual
  bound. The general pure kernel proves conditional exact-real partition
  convergence, but runtime binding beyond P2, binary64 asymptotic convergence,
  shared causal multi-cycle provenance, repeated runtime stability and the
  runtime infinity limit remain open.
- [CATALOG_TYPE_HYGIENE_PROGRAMME.md](CATALOG_TYPE_HYGIENE_PROGRAMME.md) records
  tested catalog extensions and their classification.
- [APPLIED_STRUCTURAL_ANALYSIS.md](APPLIED_STRUCTURAL_ANALYSIS.md) links selected
  theory to executable analyses.

## Reproducibility rule

Every empirical claim must identify inputs, seed, operator sequence, telemetry
and executable code. A green test suite verifies implemented contracts; it does
not prove an open mathematical statement.
