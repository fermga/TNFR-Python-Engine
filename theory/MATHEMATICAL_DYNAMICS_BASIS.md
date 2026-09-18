# Mathematical dynamics: state, evolution and diagnostic basis

This page routes foundational questions to their shared mathematical owners.
The nodal equation `dEPI/dt=nu_f*DeltaNFR` is the starting premise; neither a
diagnostic basis nor a storage representation completes the missing laws.

## 1. State and directed evolution

| Question | Definition and derivation owner |
| --- | --- |
| What is EPI, and what type of object is its pressure? | [Fundamental theory, sections 2.4-2.6](FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation) |
| Which parameter is a coordinate, observation or configured coefficient? | [All-parameter foundations](NODAL_PARAMETER_FOUNDATIONS.md) |
| What changes with units, form chart and graph symmetry? | [Covariance and conditional diffusion](NODAL_PARAMETER_FOUNDATIONS.md#3-joint-changes-of-form-and-time-units) |
| How do phase and capacity changes affect pressure and EPI acceleration? | [Joint response and source compatibility](NODAL_PARAMETER_FOUNDATIONS.md#10-joint-pressure-response-and-the-capacity-product-rule) |
| Which phase, capacity and topology laws are actually supplied? | [Constitutive audit](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#14-constitutive-closure-audit-from-the-nodal-law) |
| When does a reduced observation have its own closed evolution? | [Derived EPI memory](DERIVED_EPI_MEMORY.md), with continuous closure and event-intertwining conditions in the foundation owners |
| When can a joint potential generate the stated channels? | [Variational scope](TNFR_VARIATIONAL_PRINCIPLE.md) |

Representation, constitutive closure, numerical execution and physical
correspondence are separate obligations. An invertible change of coordinates
can preserve a supplied law without deriving it. A lossy observation requires
projectability before it can replace the full state.

## 2. Circular geometry and diagnostic scope

The complex exponential has period **2*pi**, and principal wrapped angular
separations have magnitude at most **pi** in the radian chart. This bounds the
mean absolute phase mismatch and defined circular curvature. Curvature is a
wrapped displacement from a phasor direction, not an arithmetic mean of angles.
Its direction requires a nonzero resultant; numerical representation and
undefined-resultant reporting must not be confused with a new physical law.

The potential uses a chosen source-distance kernel. Coherence length uses an
uncentered static product fit, with a distinct dimensionless spectral fallback.
Their units, estimation conditions and policies differ from the exact angular
bound. Neither phase wrapping nor the nodal product alone fixes these kernels,
warning cuts or normalization coefficients.

The four channels remain the canonical diagnostic interface. Grouping them by
aggregation, local difference and correlation does not prove independence,
minimality or complete state reconstruction. See
[Minimal Structural Degrees](MINIMAL_STRUCTURAL_DEGREES.md).

## 3. One implementation and one research queue

Use the [structural-field guide](../docs/STRUCTURAL_FIELDS_TETRAD.md) for current
API definitions, numerical domains and provenance, and the
[field implementation](../src/tnfr/physics/fields.py) for shared read-outs.
This index does not repeat their thresholds or maintain another status table.

The [nodal execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) alone owns
the active research sequence. Mathematical controls constrain admissible
completions; empirical correspondence still requires independent measurements
and reserved predictions under the supporting protocol.
