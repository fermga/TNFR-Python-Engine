# TNFR Python Engine Documentation

The TNFR Python Engine models multiscale structural coherence following the canonical
Resonant Fractal Nature Theory (TNFR). It provides node factories, structural operator
pipelines, dynamics integrators, and telemetry so you can activate, observe, and adapt
coherent forms across domains while preserving the nodal equation
$\partial EPI/\partial t = \nu_f \cdot \Delta NFR(t)$.

## Start here

- [Quickstart guide](getting-started/quickstart.md): installation, optional dependencies,
  and the first Python and CLI runs.
- [API overview](api/overview.md): architecture, package map, and data flow for structural
  operators and telemetry.
- [Structural operators](api/operators.md): canonical grammar, invariants, and workflow
  planning aids.
- [Telemetry and utilities](api/telemetry.md): coherence metrics, trace capture, locks, and
  helper facades for reproducible experiments.
- [Examples library](examples/README.md): runnable scenarios (controlled dissonance loop and
  optical cavity feedback) with CLI counterparts.
- [Release notes](releases.md): API transitions, compatibility windows, and deprecation timelines.

## Canonical references

- [Architecture guide](../ARCHITECTURE.md) for orchestration layers and invariants.
- [Contribution guidelines](../CONTRIBUTING.md) with the structural test suite contract.
- [TNFR paradigm PDF](../TNFR.pdf) detailing the theoretical background and operator grammar.

## Navigating the docs

Each section emphasizes the TNFR invariants and structural operators. Follow the quickstart
first, then drill into the API chapters that match your focus:

1. **Operate** — use the Python/CLI quickstart to validate your environment.
2. **Observe** — review telemetry, metrics, and trace capture to interpret C(t), ΔNFR, Si,
   phase, and νf.
3. **Extend** — consult the API pages before introducing new operators, telemetry streams, or
   utilities so your additions stay canonical.
4. **Experiment** — clone the runnable examples, adapt the node parameters, and reproduce the
   CLI workflows for automation or batch testing.

All examples are deterministic when you set seeds through the provided utilities. Keep
structural frequency (νf) expressed in Hz_str and respect operator closure when composing
new scenarios.
