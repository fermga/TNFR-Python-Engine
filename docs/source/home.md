# TNFR Documentation Index (Phase 3 scaffold)

Welcome to the canonical reference for the TNFR Python Engine. This page orients you to the
major documentation areas so you can quickly find the right level of detail—whether you are
bootstrapping an environment, validating operator semantics, or diving into the underlying
theory.

## Quick References for New Contributors

- **[README.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/README.md)** – Start here! Accessible introduction to TNFR with quick installation and first steps.
- **[GLOSSARY.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/GLOSSARY.md)** – Unified glossary of TNFR variables, operators, and
  canonical concepts. Essential reference for understanding EPI, νf, ΔNFR, and structural operators.
- **[GLYPH_SEQUENCES_GUIDE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/GLYPH_SEQUENCES_GUIDE.md)** – Examples of common structural
  operator sequences with expected behaviors and metrics.

## Documentation map

- **Getting started** – begin with the practical [Quickstart](getting-started/quickstart.md) to
  spin up a TNFR node, then review the [migrating guide](getting-started/migrating-remesh-window.md)
  if you are coming from Remesh Window.
- **API reference** – consult the [overview](api/overview.md) plus the focused guides on
  [structural operators](api/operators.md) and [telemetry utilities](api/telemetry.md) when you need
  concrete call signatures or examples.
- **Mathematical Foundations** – the notebooks under `theory/` connect the canonical equations with
  implementation choices. Use them when you must align derivations with code paths.
- **Examples** – cloneable scenarios in [examples/README.md](examples/README.md) that demonstrate
  cross-scale coherence checks.
- **Security** – operational guidance for monitoring and supply-chain hygiene in
  [security/](security/monitoring.md).
- **Releases** – version-by-version summaries in the [release notes](releases.md).

!!! important "Mathematical Foundations"
    The [Mathematical Foundations overview](theory/00_overview.ipynb) anchors the canonical
    nodal equation and structural operators. Each primer (structural frequency, phase synchrony,
    ΔNFR gradient fields, coherence metrics, sense index, and recursivity cascades) expands the
    derivations used by the engine. Refer back here whenever you need to validate analytical
    assumptions or reproduce the derivations behind telemetry outputs.

!!! tip "Quick-start pathways"
    * For implementers: follow the [Quickstart](getting-started/quickstart.md) to configure
      dependencies, initialize a seed, and run your first coherence sweep.
    * For theorists: the [Mathematical Quick Start](foundations.md) bridges the primer notebooks with
      the code-level abstractions.

## Release cadence

Stable builds, bug fixes, and structural operator updates are catalogued in the
[Release notes](releases.md). Use that page to confirm which operators, telemetry fields, and
notebook revisions shipped in a given version before you align experiments or migrations.

## Need a different entry point?

Use the navigation sidebar (Material theme) to jump directly into operators, notebooks, or example
bundles. Each section cross-links back to this index so you can maintain orientation while
exploring deeper content.
