# Release notes

## Upcoming (1.x compatibility window)

- English operator identifiers are now canonical. The registry publishes
  `Emission`, `Reception`, `Coherence`, `Resonance`, etc. under their
  English tokens and the CLI expects the same literals. Spanish class
  names remain available in :mod:`tnfr.operators.compat` as wrappers that
  raise :class:`DeprecationWarning` upon instantiation. They will be
  removed after the 1.x line exits maintenance (target: first 2.0.0
  pre-release). Update pipelines and stored sequences to rely on the
  English identifiers during this window.

- Renamed the network preparation helper to `prepare_network` for
  consistency with the English-facing API. The previous Spanish name
  `preparar_red` now emits a :class:`DeprecationWarning`, is no longer
  exported via ``__all__`` and will be removed on **2025-06-01**. Use
  the English helper directly to stay within the supported contract.

All other helpers continue to honour the existing dependency manifest
and import semantics.
