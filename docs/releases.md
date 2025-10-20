# Release notes

## Upcoming (1.x compatibility window)

- Renamed the network preparation helper to `prepare_network` for
  consistency with the English-facing API. The previous Spanish name
  `preparar_red` is still exported as a legacy alias and will keep
  forwarding to `prepare_network` for every 1.x release.
- Deprecation timeline: the alias will be removed in the first 2.0.0
  pre-release. Projects using `preparar_red` should migrate now and
  monitor the release notes for the final removal date.

All other helpers continue to honour the existing dependency manifest
and import semantics.
